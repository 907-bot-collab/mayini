"""
mayini.nas.controller — NAS search controller.

Supports evolutionary (genetic) search and random search baselines.
Optionally wraps Optuna for Bayesian optimisation if installed.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .search_space import SearchSpace


class NASController:
    """High-level controller that orchestrates architecture search.

    Parameters
    ----------
    search_space:
        A :class:`~mayini.nas.SearchSpace` instance.
    budget:
        Total number of model evaluations (default 50).
    method:
        ``"evolutionary"`` (default), ``"random"``, or ``"bayesian"``.
    population_size:
        Population size for evolutionary search (default 20).

    Example
    -------
    >>> from mayini.nas import SearchSpace, NASController
    >>> ss = SearchSpace(min_layers=1, max_layers=3)
    >>> nas = NASController(ss, budget=30, method="evolutionary")
    >>> best_model = nas.search(
    ...     eval_fn=lambda model: evaluate(model, val_data),
    ...     in_features=784, out_features=10
    ... )
    """

    def __init__(
        self,
        search_space: SearchSpace,
        budget: int = 50,
        method: str = "evolutionary",
        population_size: int = 20,
    ) -> None:
        self.search_space = search_space
        self.budget = budget
        self.method = method
        self.population_size = population_size
        self.history: List[Tuple[Dict, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        eval_fn: Callable[[Any], float],
        in_features: int = 784,
        out_features: int = 10,
        verbose: bool = True,
    ) -> Any:
        """Run architecture search and return the best model found.

        Parameters
        ----------
        eval_fn:
            A function ``eval_fn(model) -> float`` returning a score
            to maximise (e.g., validation accuracy).
        in_features:
            Input dimension.
        out_features:
            Output dimension (number of classes).
        verbose:
            Print progress.

        Returns
        -------
        Mayini Sequential model
            Best architecture found within the budget.
        """
        self.history = []

        if self.method == "evolutionary":
            best_arch = self._evolutionary_search(
                eval_fn, in_features, out_features, verbose
            )
        elif self.method == "random":
            best_arch = self._random_search(
                eval_fn, in_features, out_features, verbose
            )
        elif self.method == "bayesian":
            best_arch = self._bayesian_search(
                eval_fn, in_features, out_features, verbose
            )
        else:
            raise ValueError(f"Unknown NAS method '{self.method}'")

        return self.search_space.build_model(best_arch)

    def best_score(self) -> float:
        """Return the best score observed during search."""
        if not self.history:
            return float("-inf")
        return max(score for _, score in self.history)

    def summary(self) -> None:
        """Print a compact search summary."""
        print(f"NAS Summary — method={self.method}, budget={self.budget}")
        print(f"  Evaluated architectures : {len(self.history)}")
        if self.history:
            scores = [s for _, s in self.history]
            print(f"  Best score              : {max(scores):.4f}")
            print(f"  Mean score              : {np.mean(scores):.4f}")

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        arch: Dict,
        eval_fn: Callable,
        in_features: int,
        out_features: int,
        verbose: bool,
        iteration: int,
    ) -> float:
        arch["in_features"] = in_features
        arch["out_features"] = out_features
        model = self.search_space.build_model(arch)
        score = eval_fn(model)
        self.history.append((arch, score))
        if verbose:
            n_layers = len(arch.get("layers", []))
            print(
                f"  [{iteration:4d}] layers={n_layers}  score={score:.4f}"
            )
        return score

    def _random_search(
        self, eval_fn, in_features, out_features, verbose
    ) -> Dict:
        if verbose:
            print(f"[NAS] Random search  budget={self.budget}")
        best_arch, best_score = None, float("-inf")
        for i in range(self.budget):
            arch = self.search_space.random_architecture(in_features, out_features)
            score = self._evaluate(arch, eval_fn, in_features, out_features, verbose, i)
            if score > best_score:
                best_score, best_arch = score, arch
        return best_arch  # type: ignore[return-value]

    def _evolutionary_search(
        self, eval_fn, in_features, out_features, verbose
    ) -> Dict:
        if verbose:
            print(
                f"[NAS] Evolutionary search  budget={self.budget} "
                f"population={self.population_size}"
            )
        # Initialise population
        population = [
            self.search_space.random_architecture(in_features, out_features)
            for _ in range(self.population_size)
        ]
        scores = [
            self._evaluate(arch, eval_fn, in_features, out_features, verbose, i)
            for i, arch in enumerate(population)
        ]

        remaining = self.budget - self.population_size
        iteration = self.population_size

        while remaining > 0:
            # Tournament selection (k=3)
            selected = self._tournament_select(population, scores, k=3, n=2)

            # Crossover
            child1, child2 = self.search_space.crossover(selected[0], selected[1])

            # Mutation
            child1 = self.search_space.mutate(child1)
            child2 = self.search_space.mutate(child2)

            for child in [child1, child2]:
                if remaining <= 0:
                    break
                s = self._evaluate(
                    child, eval_fn, in_features, out_features, verbose, iteration
                )
                # Replace worst in population
                worst_idx = int(np.argmin(scores))
                if s > scores[worst_idx]:
                    population[worst_idx] = child
                    scores[worst_idx] = s
                iteration += 1
                remaining -= 1

        best_idx = int(np.argmax(scores))
        return population[best_idx]

    def _bayesian_search(
        self, eval_fn, in_features, out_features, verbose
    ) -> Dict:
        try:
            import optuna  # type: ignore[import]
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            if verbose:
                print("[NAS] optuna not installed → falling back to random search")
            return self._random_search(eval_fn, in_features, out_features, verbose)

        ss = self.search_space

        def objective(trial: "optuna.Trial") -> float:
            n_layers = trial.suggest_int("n_layers", ss.min_layers, ss.max_layers)
            from .search_space import ArchNode
            nodes = []
            for i in range(n_layers):
                nodes.append(
                    ArchNode(
                        out_features=trial.suggest_categorical(
                            f"hidden_{i}", ss.hidden_sizes
                        ),
                        activation=trial.suggest_categorical(
                            f"act_{i}", ss.activations
                        ),
                        dropout=trial.suggest_categorical(
                            f"drop_{i}", ss.dropout_rates
                        ),
                    )
                )
            arch = {
                "in_features": in_features,
                "out_features": out_features,
                "layers": nodes,
            }
            model = ss.build_model(arch)
            score = eval_fn(model)
            self.history.append((arch, score))
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.budget, show_progress_bar=verbose)

        # Reconstruct best arch
        best_trial = study.best_trial
        n_layers = best_trial.params["n_layers"]
        from .search_space import ArchNode
        best_arch = {
            "in_features": in_features,
            "out_features": out_features,
            "layers": [
                ArchNode(
                    out_features=best_trial.params[f"hidden_{i}"],
                    activation=best_trial.params[f"act_{i}"],
                    dropout=best_trial.params[f"drop_{i}"],
                )
                for i in range(n_layers)
            ],
        }
        return best_arch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament_select(
        population: List[Dict],
        scores: List[float],
        k: int = 3,
        n: int = 2,
    ) -> List[Dict]:
        selected = []
        for _ in range(n):
            contenders_idx = random.sample(range(len(population)), min(k, len(population)))
            winner_idx = max(contenders_idx, key=lambda i: scores[i])
            selected.append(population[winner_idx])
        return selected
