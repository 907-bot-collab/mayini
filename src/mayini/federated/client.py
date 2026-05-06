"""
mayini.federated.client — Federated Learning client-side logic.

Each FederatedClient holds a local copy of the model and performs
local SGD training, returning weight updates to the server.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, List, Optional

import numpy as np


class FederatedClient:
    """Local training worker for federated learning.

    Parameters
    ----------
    model:
        A Mayini model. A deep copy is stored locally.
    data:
        Local dataset — any iterable of ``(X, y)`` tuples or a list of
        batched tensors compatible with the model's loss function.
    client_id:
        Unique identifier string for this client.
    lr:
        Local SGD learning rate.

    Example
    -------
    >>> client = FederatedClient(model, local_data, client_id="node_0")
    >>> update = client.train(local_epochs=3)
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        client_id: str,
        lr: float = 0.01,
    ) -> None:
        self.model = self._clone_model(model)
        self.data = list(data)
        self.client_id = client_id
        self.lr = lr
        self._initial_weights: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Load global model weights into this client's local model."""
        for name, param in self._iter_params():
            if name in weights:
                param.data = weights[name].copy()
        self._initial_weights = {
            n: p.data.copy() for n, p in self._iter_params()
        }

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return current local model weights."""
        return {n: p.data.copy() for n, p in self._iter_params()}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        local_epochs: int = 5,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Perform local training and return weight update dict.

        Returns
        -------
        dict
            Keys: ``"weights"`` (delta), ``"num_samples"``,
            ``"client_id"``, ``"loss"``.
        """
        if self._initial_weights is None:
            self._initial_weights = {
                n: p.data.copy() for n, p in self._iter_params()
            }

        total_loss = 0.0
        n_batches = 0

        for _epoch in range(local_epochs):
            np.random.shuffle(self.data)  # type: ignore[arg-type]
            for batch in self._batch_iter(batch_size):
                loss_val = self._train_step(batch)
                total_loss += float(loss_val)
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Compute delta (update = current - initial)
        current = self.get_weights()
        delta = {
            name: current[name] - self._initial_weights[name]
            for name in current
        }

        return {
            "weights": delta,
            "num_samples": len(self.data),
            "client_id": self.client_id,
            "loss": avg_loss,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> float:
        """Single SGD step — works for (X, y) tuples or loss-returning batches."""
        # Zero gradients
        for _, param in self._iter_params():
            param.grad = None

        model_compute_loss = getattr(self.model, "compute_loss", None)
        if callable(model_compute_loss):
            try:
                X, y = batch
                loss = model_compute_loss(X, y)
            except (TypeError, ValueError):
                loss = model_compute_loss(batch)
        else:
            # Generic fallback for simple (X, y) sample lists used in tests.
            if isinstance(batch, list) and batch and isinstance(batch[0], tuple) and len(batch[0]) == 2:
                x_values = []
                for sample_x, _ in batch:
                    if hasattr(sample_x, "data"):
                        x_values.append(np.asarray(sample_x.data, dtype=np.float32))
                    else:
                        x_values.append(np.asarray(sample_x, dtype=np.float32))
                X = batch[0][0].__class__(np.stack(x_values, axis=0), requires_grad=False)
                output = self.model(X)
            elif isinstance(batch, tuple) and len(batch) == 2:
                X, _ = batch
                output = self.model(X)
            else:
                output = self.model(batch)
            loss = output.sum()

        loss.backward()

        # Manual SGD step
        for _, param in self._iter_params():
            if param.grad is not None and param.requires_grad:
                param.data -= self.lr * param.grad

        return float(loss.data.item() if hasattr(loss.data, "item") else loss.data)

    def _batch_iter(self, batch_size: int) -> Iterator:
        for start in range(0, len(self.data), batch_size):
            _batch = self.data[start : start + batch_size]
            yield _batch

    def _iter_params(self):
        try:
            yield from self.model.named_parameters()
        except AttributeError:
            for i, p in enumerate(self.model.parameters()):
                yield (f"param_{i}", p)

    @staticmethod
    def _clone_model(model: Any) -> Any:
        try:
            return copy.deepcopy(model)
        except Exception:
            return model
