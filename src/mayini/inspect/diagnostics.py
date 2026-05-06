"""
mayini.inspect.diagnostics — Gradient health diagnostics.

Detects vanishing and exploding gradients across training steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class GradientDiagnostics:
    """Analyse gradient health over training to detect anomalies.

    Parameters
    ----------
    model:
        Mayini model to monitor.
    vanish_threshold:
        Mean absolute gradient below this value triggers a vanishing warning.
    explode_threshold:
        Gradient L2 norm above this value triggers an exploding warning.

    Example
    -------
    >>> diag = GradientDiagnostics(model)
    >>> for epoch in range(epochs):
    ...     loss.backward()
    ...     diag.step(epoch)
    >>> diag.summary()
    """

    def __init__(
        self,
        model: Any,
        vanish_threshold: float = 1e-6,
        explode_threshold: float = 1e3,
    ) -> None:
        self.model = model
        self.vanish_threshold = vanish_threshold
        self.explode_threshold = explode_threshold
        self.history: List[Dict] = []
        self.warnings: List[str] = []

    def step(self, step: int = 0) -> Dict:
        """Record gradient stats after one backward pass."""
        record: Dict = {"step": step, "layers": {}, "issues": []}

        for name, param in self._iter_params():
            if param.grad is None:
                continue
            g = param.grad
            norm = float(np.linalg.norm(g))
            mean_abs = float(np.abs(g).mean())

            record["layers"][name] = {"norm": norm, "mean_abs": mean_abs}

            if mean_abs < self.vanish_threshold:
                msg = f"[step {step}] VANISHING gradient in '{name}' (mean_abs={mean_abs:.2e})"
                record["issues"].append(msg)
                self.warnings.append(msg)

            if norm > self.explode_threshold:
                msg = f"[step {step}] EXPLODING gradient in '{name}' (norm={norm:.2e})"
                record["issues"].append(msg)
                self.warnings.append(msg)

        self.history.append(record)
        for issue in record["issues"]:
            print(f"⚠  {issue}")
        return record

    def summary(self) -> None:
        """Print a concise health summary over all recorded steps."""
        print(f"\nGradient Health Summary — {len(self.history)} steps recorded")
        print(f"  Total warnings : {len(self.warnings)}")
        if not self.history:
            return

        # Aggregate per-layer norms
        layer_norms: Dict[str, List[float]] = {}
        for rec in self.history:
            for name, stats in rec["layers"].items():
                layer_norms.setdefault(name, []).append(stats["norm"])

        print(f"\n  {'Layer':<40} {'Mean Norm':>12} {'Max Norm':>12}")
        print("  " + "-" * 66)
        for name, norms in layer_norms.items():
            print(
                f"  {name:<40} {np.mean(norms):>12.4f} {np.max(norms):>12.4f}"
            )

        if self.warnings:
            print(f"\n  Last 5 warnings:")
            for w in self.warnings[-5:]:
                print(f"    {w}")

    def _iter_params(self):
        try:
            yield from self.model.named_parameters()
        except AttributeError:
            for i, p in enumerate(self.model.parameters()):
                yield (f"param_{i}", p)
