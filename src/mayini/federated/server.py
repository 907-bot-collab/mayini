"""
mayini.federated.server — FedAvg / FedProx aggregation server.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np


class FederatedServer:
    """Federated Learning Server coordinating distributed training.

    Parameters
    ----------
    model:
        The global Mayini model.
    clients:
        List of :class:`~mayini.federated.FederatedClient` instances.
    aggregation:
        Aggregation strategy: ``"fedavg"`` (default) or ``"fedprox"``.
    mu:
        Proximal term coefficient for FedProx (default 0.01).

    Example
    -------
    >>> server = FederatedServer(model, clients, aggregation="fedavg")
    >>> server.train(rounds=10)
    >>> history = server.history
    """

    def __init__(
        self,
        model: Any,
        clients: List[Any],
        aggregation: str = "fedavg",
        mu: float = 0.01,
    ) -> None:
        self.global_model = model
        self.clients = clients
        self.aggregation = aggregation
        self.mu = mu
        self.history: List[Dict[str, Any]] = []
        self._round = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        rounds: int = 10,
        fraction_fit: float = 1.0,
        local_epochs: int = 5,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the federated training loop.

        Parameters
        ----------
        rounds:
            Number of communication rounds.
        fraction_fit:
            Fraction of clients to sample per round (1.0 = all).
        local_epochs:
            Local epochs each client trains per round.
        verbose:
            Print progress each round.

        Returns
        -------
        list
            Training history (one dict per round).
        """
        for r in range(rounds):
            self._round += 1
            selected = self._select_clients(fraction_fit)

            # Distribute global weights
            global_weights = self._get_global_weights()
            client_updates: List[Dict] = []

            for client in selected:
                client.set_weights(global_weights)
                update = client.train(local_epochs=local_epochs)
                client_updates.append(update)

            # Aggregate
            self._aggregate(client_updates)

            # Log
            avg_loss = np.mean([u["loss"] for u in client_updates])
            record = {
                "round": self._round,
                "clients": len(selected),
                "avg_loss": float(avg_loss),
                "aggregation": self.aggregation,
            }
            self.history.append(record)

            if verbose:
                print(
                    f"[FL Round {self._round:3d}/{rounds}] "
                    f"clients={len(selected)}  avg_loss={avg_loss:.4f}"
                )

        return self.history

    def evaluate(self, test_data: Any, loss_fn: Any = None) -> float:
        """Quick evaluation of the global model on test_data."""
        total_loss = 0.0
        n = 0
        for batch in test_data:
            try:
                X, y = batch
                out = self.global_model(X)
                if loss_fn:
                    loss = loss_fn(out, y)
                    total_loss += float(loss.data)
                    n += 1
            except Exception:
                pass
        return total_loss / max(n, 1)

    def get_privacy_spent(self) -> float:
        """Return cumulative privacy budget spent (DP clients only)."""
        total = 0.0
        for client in self.clients:
            if hasattr(client, "privacy_spent"):
                total += client.privacy_spent()
        return total

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_clients(self, fraction: float) -> List[Any]:
        k = max(1, int(len(self.clients) * fraction))
        return random.sample(self.clients, k)

    def _get_global_weights(self) -> Dict[str, np.ndarray]:
        weights: Dict[str, np.ndarray] = {}
        try:
            for name, param in self.global_model.named_parameters():
                weights[name] = param.data.copy()
        except AttributeError:
            for i, p in enumerate(self.global_model.parameters()):
                weights[f"param_{i}"] = p.data.copy()
        return weights

    def _set_global_weights(self, weights: Dict[str, np.ndarray]) -> None:
        try:
            for name, param in self.global_model.named_parameters():
                if name in weights:
                    param.data = weights[name].copy()
        except AttributeError:
            for i, p in enumerate(self.global_model.parameters()):
                key = f"param_{i}"
                if key in weights:
                    p.data = weights[key].copy()

    def _aggregate(self, updates: List[Dict]) -> None:
        if self.aggregation == "fedavg":
            self._fedavg(updates)
        elif self.aggregation == "fedprox":
            self._fedprox(updates)
        else:
            raise ValueError(f"Unknown aggregation '{self.aggregation}'")

    def _fedavg(self, updates: List[Dict]) -> None:
        """Weighted average of weight deltas (FedAvg)."""
        total_samples = sum(u["num_samples"] for u in updates)
        if total_samples == 0:
            return

        global_w = self._get_global_weights()
        for name in global_w:
            weighted_delta = sum(
                u["weights"].get(name, np.zeros_like(global_w[name]))
                * u["num_samples"]
                for u in updates
            )
            global_w[name] = global_w[name] + weighted_delta / total_samples

        self._set_global_weights(global_w)

    def _fedprox(self, updates: List[Dict]) -> None:
        """FedProx: same as FedAvg + proximal regularisation signal."""
        # FedProx proximal term is enforced client-side during local training;
        # server aggregation is identical to FedAvg.
        self._fedavg(updates)
