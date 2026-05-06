"""
mayini.data — Data Diagnostics & Label-Noise Handling

Utilities for class-imbalance detection, duplicate/leakage checks,
outlier flagging, and noisy-label filtering.
"""

from .diagnostics import DataDiagnostics
from .noise import LabelNoiseHandler

__all__ = ["DataDiagnostics", "LabelNoiseHandler"]
