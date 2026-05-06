"""
mayini.inspect.hooks — Forward/backward hook infrastructure.

Provides a HookManager plus two concrete recorders:
  - ActivationRecorder : records output shapes & stats per layer
  - GradientRecorder   : records gradient norms & stats per param
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Base hook manager
# ---------------------------------------------------------------------------

class HookManager:
    """Context-manager that installs and removes forward hooks on a model.

    Example
    -------
    >>> hm = HookManager()
    >>> def my_hook(name, output):
    ...     print(name, output.shape)
    >>> with hm.register_forward(model, my_hook):
    ...     model(x)
    """

    def __init__(self) -> None:
        self._hooks: List[Callable] = []

    def register_forward(self, model: Any, fn: Callable) -> "HookManager":
        """Monkey-patch each leaf module's ``forward`` to call *fn(name, out)*."""
        self._model = model
        self._fn = fn
        self._originals: Dict[str, Callable] = {}

        for name, module in self._named_leaves(model):
            orig_fwd = module.forward

            def _make_hook(n: str, orig: Callable):
                def hooked_forward(*args, **kwargs):
                    out = orig(*args, **kwargs)
                    fn(n, out)
                    return out
                return hooked_forward

            self._originals[name] = orig_fwd
            module.forward = _make_hook(name, orig_fwd)

        return self

    def remove_all(self) -> None:
        """Restore all patched forward methods."""
        for name, module in self._named_leaves(self._model):
            if name in self._originals:
                module.forward = self._originals[name]
        self._originals = {}

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *args) -> None:
        self.remove_all()

    @staticmethod
    def _named_leaves(model: Any):
        try:
            for name, module in model.named_modules():
                children = list(module.children()) if hasattr(module, "children") else []
                if not children:
                    yield name, module
        except AttributeError:
            yield ("model", model)


# ---------------------------------------------------------------------------
# Activation recorder
# ---------------------------------------------------------------------------

class ActivationRecorder:
    """Record activation statistics (shape, mean, std, min, max) per layer.

    Example
    -------
    >>> rec = ActivationRecorder(model)
    >>> rec.start()
    >>> model(x)
    >>> rec.stop()
    >>> rec.report()
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.log: Dict[str, List[Dict]] = defaultdict(list)
        self._manager = HookManager()

    def start(self) -> "ActivationRecorder":
        """Install hooks."""
        self._manager.register_forward(self.model, self._record)
        return self

    def stop(self) -> None:
        """Remove hooks."""
        self._manager.remove_all()

    def __enter__(self) -> "ActivationRecorder":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()

    def clear(self) -> None:
        self.log.clear()

    def report(self) -> None:
        """Print per-layer activation summary."""
        print(f"{'Layer':<40} {'Shape':>20} {'Mean':>10} {'Std':>10}")
        print("-" * 82)
        for name, records in self.log.items():
            last = records[-1]
            print(
                f"{name:<40} {str(last['shape']):>20} "
                f"{last['mean']:>10.4f} {last['std']:>10.4f}"
            )

    def _record(self, name: str, output: Any) -> None:
        data = output.data if hasattr(output, "data") else np.asarray(output)
        self.log[name].append({
            "shape": list(data.shape),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "min": float(data.min()),
            "max": float(data.max()),
        })


# ---------------------------------------------------------------------------
# Gradient recorder
# ---------------------------------------------------------------------------

class GradientRecorder:
    """Record gradient statistics after each backward pass.

    Usage: call :meth:`record` after ``loss.backward()`` to snapshot
    the current gradient state of all model parameters.

    Example
    -------
    >>> rec = GradientRecorder(model)
    >>> loss.backward()
    >>> rec.record(step=epoch)
    >>> rec.report()
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.log: List[Dict[str, Any]] = []

    def record(self, step: int = 0) -> Dict[str, Any]:
        """Snapshot gradient stats for all parameters."""
        snapshot: Dict[str, Any] = {"step": step, "params": {}}
        for name, param in self._iter_params():
            if param.grad is not None:
                g = param.grad
                snapshot["params"][name] = {
                    "norm": float(np.linalg.norm(g)),
                    "mean": float(g.mean()),
                    "std": float(g.std()),
                    "max_abs": float(np.abs(g).max()),
                }
        self.log.append(snapshot)
        return snapshot

    def report(self) -> None:
        """Print gradient statistics for the most recent backward pass."""
        if not self.log:
            print("No gradient snapshots recorded.")
            return
        last = self.log[-1]
        print(f"Gradient snapshot @ step {last['step']}")
        print(f"{'Parameter':<40} {'Norm':>10} {'Mean':>10} {'Std':>10}")
        print("-" * 72)
        for name, stats in last["params"].items():
            print(
                f"{name:<40} {stats['norm']:>10.4f} "
                f"{stats['mean']:>10.4f} {stats['std']:>10.4f}"
            )

    def _iter_params(self):
        try:
            yield from self.model.named_parameters()
        except AttributeError:
            for i, p in enumerate(self.model.parameters()):
                yield (f"param_{i}", p)
