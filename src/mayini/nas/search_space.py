"""
mayini.nas.search_space — Layer-level search space definition.

Defines atomic operations (linear, conv-like, activation choices) and
how random/evolved architectures are represented as dicts.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Architecture node
# ---------------------------------------------------------------------------

@dataclass
class ArchNode:
    """Single layer specification in a NAS architecture.

    Attributes
    ----------
    op : str
        Operation type (e.g. ``"linear"``, ``"relu"``, ``"dropout"``).
    out_features : int
        Output size for linear/conv layers.
    activation : str
        Activation following this layer (``"relu"``, ``"tanh"``, ``"none"``).
    dropout : float
        Dropout rate (0 = disabled).
    """

    op: str = "linear"
    out_features: int = 128
    activation: str = "relu"
    dropout: float = 0.0

    def clone(self) -> "ArchNode":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

class SearchSpace:
    """Defines valid choices for each architectural dimension.

    Parameters
    ----------
    min_layers:
        Minimum number of hidden layers (default 1).
    max_layers:
        Maximum number of hidden layers (default 5).
    hidden_sizes:
        Candidate hidden layer widths.
    activations:
        Candidate activation functions.
    dropout_rates:
        Candidate dropout rates.

    Example
    -------
    >>> ss = SearchSpace(min_layers=1, max_layers=3)
    >>> arch = ss.random_architecture(in_features=784, out_features=10)
    >>> print(arch)
    """

    DEFAULT_HIDDEN_SIZES = [32, 64, 128, 256, 512]
    DEFAULT_ACTIVATIONS = ["relu", "tanh", "leaky_relu", "none"]
    DEFAULT_DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]

    def __init__(
        self,
        min_layers: int = 1,
        max_layers: int = 5,
        hidden_sizes: Optional[List[int]] = None,
        activations: Optional[List[str]] = None,
        dropout_rates: Optional[List[float]] = None,
    ) -> None:
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.hidden_sizes = hidden_sizes or self.DEFAULT_HIDDEN_SIZES
        self.activations = activations or self.DEFAULT_ACTIVATIONS
        self.dropout_rates = dropout_rates or self.DEFAULT_DROPOUTS

    def random_architecture(
        self,
        in_features: int = 784,
        out_features: int = 10,
    ) -> Dict[str, Any]:
        """Sample a random architecture from the search space."""
        n_layers = random.randint(self.min_layers, self.max_layers)
        nodes: List[ArchNode] = []
        for _ in range(n_layers):
            node = ArchNode(
                op="linear",
                out_features=random.choice(self.hidden_sizes),
                activation=random.choice(self.activations),
                dropout=random.choice(self.dropout_rates),
            )
            nodes.append(node)
        return {
            "in_features": in_features,
            "out_features": out_features,
            "layers": nodes,
        }

    def build_model(self, arch: Dict[str, Any]) -> Any:
        """Build a Mayini Sequential model from an architecture dict.

        Parameters
        ----------
        arch:
            Dict returned by :meth:`random_architecture`.

        Returns
        -------
        mayini.nn.Sequential
            Constructed model.
        """
        from mayini import nn  # type: ignore[import]

        layers: List[Any] = []
        in_f = arch["in_features"]

        for node in arch["layers"]:
            layers.append(nn.Linear(in_f, node.out_features))
            if node.activation == "relu":
                layers.append(nn.ReLU())
            elif node.activation == "tanh":
                layers.append(nn.Tanh())
            elif node.activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            if node.dropout > 0.0:
                layers.append(nn.Dropout(p=node.dropout))
            in_f = node.out_features

        layers.append(nn.Linear(in_f, arch["out_features"]))
        return nn.Sequential(*layers)

    def crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
    ) -> tuple:
        """Uniform crossover of two architecture dicts."""
        layers1 = parent1["layers"]
        layers2 = parent2["layers"]
        min_len = min(len(layers1), len(layers2))
        max_len = max(len(layers1), len(layers2))

        child1_layers, child2_layers = [], []
        for i in range(min_len):
            if random.random() > 0.5:
                child1_layers.append(layers1[i].clone())
                child2_layers.append(layers2[i].clone())
            else:
                child1_layers.append(layers2[i].clone())
                child2_layers.append(layers1[i].clone())

        # Inherit trailing layers from the longer parent
        if len(layers1) > min_len:
            child1_layers.extend([l.clone() for l in layers1[min_len:]])
        if len(layers2) > min_len:
            child2_layers.extend([l.clone() for l in layers2[min_len:]])

        c1 = {**parent1, "layers": child1_layers}
        c2 = {**parent2, "layers": child2_layers}
        return c1, c2

    def mutate(
        self,
        arch: Dict[str, Any],
        mutation_rate: float = 0.2,
    ) -> Dict[str, Any]:
        """Randomly mutate an architecture in-place (returns copy)."""
        mutated = copy.deepcopy(arch)
        for node in mutated["layers"]:
            if random.random() < mutation_rate:
                node.out_features = random.choice(self.hidden_sizes)
            if random.random() < mutation_rate:
                node.activation = random.choice(self.activations)
            if random.random() < mutation_rate:
                node.dropout = random.choice(self.dropout_rates)
        # Randomly add/remove a layer
        if random.random() < mutation_rate / 2 and len(mutated["layers"]) < self.max_layers:
            mutated["layers"].append(
                ArchNode(
                    out_features=random.choice(self.hidden_sizes),
                    activation=random.choice(self.activations),
                    dropout=random.choice(self.dropout_rates),
                )
            )
        elif random.random() < mutation_rate / 2 and len(mutated["layers"]) > self.min_layers:
            mutated["layers"].pop(random.randrange(len(mutated["layers"])))
        return mutated
