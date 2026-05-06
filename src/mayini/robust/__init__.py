"""
mayini.robust — Robustness & Uncertainty Tools

Provides FGSM adversarial examples, PGD attack, Monte-Carlo Dropout
uncertainty estimation, and a simple ensemble wrapper.
"""

from .adversarial import fgsm_attack, pgd_attack
from .uncertainty import MCDropoutEstimator, EnsembleEstimator

__all__ = [
    "fgsm_attack",
    "pgd_attack",
    "MCDropoutEstimator",
    "EnsembleEstimator",
]
