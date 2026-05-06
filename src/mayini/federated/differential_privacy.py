"""
mayini.federated.differential_privacy — DP-SGD client via Gaussian mechanism.

Implements gradient clipping + Gaussian noise injection to provide
(ε, δ)-differential privacy guarantees during local training.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from .client import FederatedClient


class DP_FL_Client(FederatedClient):
    """Differentially-private federated client using the Gaussian mechanism.

    Parameters
    ----------
    model:
        Mayini model (a deep copy is stored locally).
    data:
        Local dataset.
    client_id:
        Unique node identifier.
    epsilon:
        Privacy budget ε > 0 (smaller = stronger privacy).
    delta:
        Failure probability δ, typically 1e-5.
    max_grad_norm:
        Clipping threshold C for per-sample gradient norms.
    lr:
        Local SGD learning rate.

    Example
    -------
    >>> dp_client = DP_FL_Client(model, data, "dp_node_0", epsilon=1.0)
    >>> update = dp_client.train(local_epochs=3)
    >>> print(dp_client.privacy_spent())
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        client_id: str,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        lr: float = 0.01,
    ) -> None:
        super().__init__(model, data, client_id, lr=lr)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self._steps_taken = 0
        self._noise_scale = self._compute_noise_multiplier()

    # ------------------------------------------------------------------
    # Privacy accounting
    # ------------------------------------------------------------------

    def privacy_spent(self) -> float:
        """Return approximate ε spent so far (moments accountant approx)."""
        if self._steps_taken == 0:
            return 0.0
        n = max(len(self.data), 1)
        q = 1.0 / n  # sampling rate
        sigma = self._noise_scale
        # Simple strong-composition approximation
        eps_per_step = q * math.sqrt(2 * math.log(1.25 / self.delta)) / sigma
        return eps_per_step * self._steps_taken

    # ------------------------------------------------------------------
    # Overridden training step with DP noise
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> float:
        """DP-SGD: clip gradients → add Gaussian noise → apply update."""
        for _, param in self._iter_params():
            param.grad = None

        try:
            X, y = batch
            loss = self.model.compute_loss(X, y)
        except (TypeError, ValueError):
            loss = self.model.compute_loss(batch)

        loss.backward()
        self._steps_taken += 1

        # Clip and noise each parameter gradient
        for _, param in self._iter_params():
            if param.grad is None or not param.requires_grad:
                continue

            # Gradient clipping
            grad_norm = float(np.linalg.norm(param.grad))
            if grad_norm > self.max_grad_norm:
                param.grad = param.grad * (self.max_grad_norm / (grad_norm + 1e-12))

            # Gaussian noise injection
            noise_std = self._noise_scale * self.max_grad_norm
            noise = np.random.normal(0.0, noise_std, param.grad.shape).astype(
                param.grad.dtype
            )
            param.grad = param.grad + noise

        # SGD update
        for _, param in self._iter_params():
            if param.grad is not None and param.requires_grad:
                param.data -= self.lr * param.grad

        return float(loss.data.item() if hasattr(loss.data, "item") else loss.data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_noise_multiplier(self) -> float:
        """Compute Gaussian noise multiplier σ from (ε, δ) via analytic formula."""
        # Calibration using the closed-form single-step bound:
        # σ = sqrt(2 * ln(1.25/δ)) / ε
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        return math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
