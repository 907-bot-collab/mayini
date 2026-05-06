"""
mayini.robust.uncertainty — Uncertainty estimation tools.

Provides:
  - MCDropoutEstimator  : Monte Carlo Dropout (Gal & Ghahramani 2016)
  - EnsembleEstimator   : Deep Ensemble uncertainty via model agreement
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]


class MCDropoutEstimator:
    """Uncertainty estimation via Monte Carlo Dropout.

    Keeps dropout active at test time and runs N stochastic forward
    passes to estimate predictive mean and variance.

    Parameters
    ----------
    model:
        A Mayini model containing Dropout layers. Must be in training
        mode (dropout active) during inference — call ``model.train()``
        if your model supports it, or ensure Dropout ``p > 0``.
    n_samples:
        Number of stochastic forward passes (default 30).

    Example
    -------
    >>> estimator = MCDropoutEstimator(model, n_samples=50)
    >>> mean, var = estimator.predict(x)
    >>> print("Uncertainty:", var.mean())
    """

    def __init__(self, model: Any, n_samples: int = 30) -> None:
        self.model = model
        self.n_samples = n_samples

    def predict(
        self, x: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run N stochastic forward passes and return mean + variance.

        Parameters
        ----------
        x:
            Input Mayini Tensor or numpy array.

        Returns
        -------
        mean : np.ndarray
            Mean prediction across passes — shape ``(B, C)`` or ``(B,)``.
        variance : np.ndarray
            Predictive variance (epistemic uncertainty) — same shape.
        """
        if isinstance(x, np.ndarray):
            x_data = x.astype(np.float32, copy=False)
        elif hasattr(x, "data"):
            x_data = np.asarray(x.data, dtype=np.float32)
        else:
            x_data = np.asarray(x, np.float32)
        outputs: List[np.ndarray] = []

        # Enable training mode for dropout if possible
        _set_train(self.model, True)

        for _ in range(self.n_samples):
            x_t = Tensor(x_data.copy(), requires_grad=False)
            out = self.model(x_t)
            arr = out.data if hasattr(out, "data") else np.asarray(out)
            outputs.append(arr.copy())

        stack = np.stack(outputs, axis=0)   # (N, B, C) or (N, B)
        mean = stack.mean(axis=0)
        variance = stack.var(axis=0)
        return mean, variance

    def entropy(self, x: Any) -> np.ndarray:
        """Predictive entropy H[p(y|x)] — higher = more uncertain."""
        mean, _ = self.predict(x)
        # Clip for numerical stability
        p = np.clip(mean, 1e-12, 1.0)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        # Normalise rows if not already probabilities
        p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
        return -np.sum(p * np.log(p), axis=-1)   # (B,)


class EnsembleEstimator:
    """Uncertainty estimation via a deep ensemble of models.

    Parameters
    ----------
    models:
        List of independently-trained Mayini models.

    Example
    -------
    >>> ens = EnsembleEstimator([model_1, model_2, model_3])
    >>> mean, var = ens.predict(x)
    """

    def __init__(self, models: List[Any]) -> None:
        if not models:
            raise ValueError("Provide at least one model.")
        self.models = models

    def predict(self, x: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean prediction and variance.

        Returns
        -------
        mean : np.ndarray  — averaged prediction ``(B, C)``
        variance : np.ndarray — disagreement across models ``(B, C)``
        """
        x_data = x.data if hasattr(x, "data") else np.asarray(x, np.float32)
        outputs: List[np.ndarray] = []

        for model in self.models:
            _set_train(model, False)
            x_t = Tensor(x_data.copy(), requires_grad=False)
            out = model(x_t)
            arr = out.data if hasattr(out, "data") else np.asarray(out)
            outputs.append(arr.copy())

        stack = np.stack(outputs, axis=0)
        return stack.mean(axis=0), stack.var(axis=0)

    def add_model(self, model: Any) -> None:
        """Append another model to the ensemble."""
        self.models.append(model)

    def __len__(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_train(model: Any, mode: bool) -> None:
    """Call model.train(mode) if the method exists."""
    try:
        model.train(mode)
    except (AttributeError, TypeError):
        pass
