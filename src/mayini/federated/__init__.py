"""
mayini.federated — Federated Learning & Differential Privacy

Provides:
  - FederatedServer  : FedAvg / FedProx aggregation
  - FederatedClient  : local training & update computation
  - DP_FL_Client     : differentially-private client (Gaussian mechanism)
"""

from .server import FederatedServer
from .client import FederatedClient
from .differential_privacy import DP_FL_Client

__all__ = ["FederatedServer", "FederatedClient", "DP_FL_Client"]
