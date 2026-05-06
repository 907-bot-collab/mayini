"""
mayini.inspect — Training Introspection & Autograd Debugging

Provides hooks for recording activations and gradient statistics,
plus gradient health diagnostics (vanishing/exploding detection).
"""

from .hooks import HookManager, ActivationRecorder, GradientRecorder
from .diagnostics import GradientDiagnostics

__all__ = [
    "HookManager",
    "ActivationRecorder",
    "GradientRecorder",
    "GradientDiagnostics",
]
