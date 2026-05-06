"""
mayini.experiments — Lightweight Experiment Tracking

Records hyperparameters, metrics, and checkpoints to JSON/CSV files.
Supports comparison utilities and Markdown summary generation.
"""

from .tracker import Experiment, ExperimentTracker

__all__ = ["Experiment", "ExperimentTracker"]
