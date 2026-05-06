"""
mayini.benchmark.zoo — Model registry (Model Zoo).

A lightweight, offline-first registry mapping model names to
constructor factories. Remote download support is optional.
"""

from __future__ import annotations

import json
import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ModelZoo:
    """Registry of named Mayini model factories.

    Built-in models are constructed locally (no download required).
    You can register custom factories with :meth:`register`.

    Example
    -------
    >>> zoo = ModelZoo()
    >>> zoo.list_models()
    >>> model = zoo.load("mlp_mnist")
    """

    _registry: Dict[str, Dict] = {}

    def __init__(self) -> None:
        self._register_builtins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        description: str = "",
        tags: Optional[list] = None,
    ) -> None:
        """Register a model factory function.

        Parameters
        ----------
        name:
            Unique model name (e.g., ``"my_cnn"``).
        factory:
            Zero-argument callable that returns a Mayini model.
        description:
            Human-readable description.
        tags:
            Optional list of tag strings (e.g., ``["classification", "mnist"]``).
        """
        ModelZoo._registry[name] = {
            "factory": factory,
            "description": description,
            "tags": tags or [],
            "source": "local",
        }

    def load(self, name: str) -> Any:
        """Instantiate and return a model by name.

        Parameters
        ----------
        name:
            Registered model name.

        Returns
        -------
        Mayini model instance.

        Raises
        ------
        KeyError
            If *name* is not in the registry.
        """
        if name not in ModelZoo._registry:
            available = list(ModelZoo._registry.keys())
            raise KeyError(
                f"Model '{name}' not in zoo. Available: {available}"
            )
        entry = ModelZoo._registry[name]
        return entry["factory"]()

    def list_models(self, tag: Optional[str] = None) -> None:
        """Print available models, optionally filtered by tag."""
        print(f"{'Name':<30} {'Source':<8} Tags")
        print("-" * 60)
        for name, info in ModelZoo._registry.items():
            if tag and tag not in info.get("tags", []):
                continue
            tags_str = ", ".join(info.get("tags", []))
            print(f"{name:<30} {info.get('source','?'):<8} {tags_str}")

    def info(self, name: str) -> Dict:
        """Return metadata dict for a model (without the factory callable)."""
        if name not in ModelZoo._registry:
            raise KeyError(f"Model '{name}' not in zoo.")
        entry = ModelZoo._registry[name].copy()
        entry.pop("factory", None)
        return entry

    # ------------------------------------------------------------------
    # Built-in model factories
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        self.register(
            "mlp_mnist",
            _build_mlp_mnist,
            description="3-layer MLP for MNIST (784→256→128→10)",
            tags=["classification", "mnist", "mlp"],
        )
        self.register(
            "mlp_tiny",
            _build_mlp_tiny,
            description="Tiny MLP for testing (2→8→1)",
            tags=["regression", "debug"],
        )
        self.register(
            "mlp_deep",
            _build_mlp_deep,
            description="Deep MLP (512→256→128→64→10)",
            tags=["classification", "deep"],
        )


# ---------------------------------------------------------------------------
# Built-in model constructors
# ---------------------------------------------------------------------------

def _build_mlp_mnist() -> Any:
    from mayini import nn  # type: ignore[import]
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def _build_mlp_tiny() -> Any:
    from mayini import nn  # type: ignore[import]
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def _build_mlp_deep() -> Any:
    from mayini import nn  # type: ignore[import]
    return nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10),
    )
