"""
mayini.tinyml.pruning — Structured & unstructured model pruning.

Methods: magnitude (L1), random, gradient-based (requires one backward pass).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class Pruner:
    """Remove weights from a model to reduce its size and computation.

    Parameters
    ----------
    model:
        Mayini model.
    sparsity:
        Fraction of weights to zero out (0.0–1.0). Default 0.5.
    method:
        ``"magnitude"`` (default), ``"random"``, or ``"gradient"``.

    Example
    -------
    >>> pruner = Pruner(model, sparsity=0.5)
    >>> pruner.prune()
    >>> pruner.print_sparsity()
    """

    def __init__(
        self,
        model: Any,
        sparsity: float = 0.5,
        method: str = "magnitude",
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("sparsity must be in [0, 1)")
        self.model = model
        self.sparsity = sparsity
        self.method = method
        self.masks: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(self) -> Any:
        """Apply pruning masks to model parameters and return model."""
        if self.method == "magnitude":
            self._magnitude_prune()
        elif self.method == "random":
            self._random_prune()
        elif self.method == "gradient":
            self._gradient_prune()
        else:
            raise ValueError(f"Unknown pruning method '{self.method}'")

        # Apply masks
        for name, param in self._iter_params():
            if name in self.masks:
                param.data = param.data * self.masks[name]
                param.requires_grad = False

        return self.model

    def apply_masks(self) -> None:
        """Re-apply stored masks (call after each optimiser step)."""
        for name, param in self._iter_params():
            if name in self.masks:
                param.data = param.data * self.masks[name]

    def print_sparsity(self) -> None:
        """Print actual sparsity per layer."""
        total, zero = 0, 0
        for name, mask in self.masks.items():
            n = mask.size
            z = int((mask == 0).sum())
            total += n
            zero += z
            pct = 100.0 * z / n if n else 0
            print(f"  {name:<40}  sparsity={pct:5.1f}%")
        if total:
            print(f"  {'TOTAL':<40}  sparsity={100.0*zero/total:5.1f}%")

    # ------------------------------------------------------------------
    # Pruning strategies
    # ------------------------------------------------------------------

    def _magnitude_prune(self) -> None:
        for name, param in self._iter_params():
            if param.data.ndim < 2:
                continue
            flat = param.data.flatten().astype(np.float64)
            k = max(1, int(flat.size * (1.0 - self.sparsity)))
            threshold = np.sort(np.abs(flat))[::-1][k - 1]
            self.masks[name] = (np.abs(param.data) >= threshold).astype(np.float32)

    def _random_prune(self) -> None:
        rng = np.random.default_rng(42)
        for name, param in self._iter_params():
            if param.data.ndim < 2:
                continue
            mask = (rng.random(param.data.shape) >= self.sparsity).astype(np.float32)
            self.masks[name] = mask

    def _gradient_prune(self) -> None:
        """Prune weights with smallest gradient magnitude (requires grads)."""
        for name, param in self._iter_params():
            if param.data.ndim < 2:
                continue
            if param.grad is not None:
                score = np.abs(param.grad)
            else:
                # Fallback to magnitude
                score = np.abs(param.data)
            flat_score = score.flatten()
            k = max(1, int(flat_score.size * (1.0 - self.sparsity)))
            threshold = np.sort(flat_score)[::-1][k - 1]
            self.masks[name] = (score >= threshold).astype(np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _iter_params(self):
        try:
            yield from self.model.named_parameters()
        except AttributeError:
            for i, p in enumerate(self.model.parameters()):
                yield (f"param_{i}", p)
