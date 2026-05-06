"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework built from scratch in Python.
Includes: DeepLearning, Machine Learning, NEAT, Automated Preprocessing,
          Explainability, TinyML, Federated Learning, NAS, Multimodal,
          Benchmarking, Introspection, Data Diagnostics, Distillation,
          Robustness, and Experiment Tracking.
"""

__version__ = "0.9.0"
__author__ = "Abhishek Adari"
__email__ = "abhishekadari85@gmail.com"

# ============================================================================
# SAFE IMPORTS - NO CIRCULAR DEPENDENCIES
# ============================================================================

from .tensor import Tensor
from .ops import concatenate, stack

# ============================================================================
# CORE SUBMODULE (always loaded)
# ============================================================================

from . import nn

# ============================================================================
# LAZY LOADING — original + all new advanced submodules
# ============================================================================

_LAZY_MODULES = {
    # Original modules
    "ml", "neat", "preprocessing", "optim", "training",
    # Feature 1 – Explainability & Graph Visualization
    "explain",
    # Feature 2 – TinyML & Edge Deployment
    "tinyml",
    # Feature 3 – Federated Learning
    "federated",
    # Feature 4 – Neural Architecture Search
    "nas",
    # Feature 5 – Multimodal Foundation Models
    "multimodal",
    # Feature 6 – WebAssembly Deployment
    "web",
    # Feature 7 – Benchmarking & Model Zoo
    "benchmark",
    # Extended Plan A – Training Introspection
    "inspect",
    # Extended Plan B – Data Diagnostics
    "data",
    # Extended Plan C – Distillation & Graph Pipelines
    "distill",
    # Extended Plan D – Robustness & Uncertainty
    "robust",
    # Extended Plan E – Experiment Tracking
    "experiments",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        try:
            import importlib
            mod = importlib.import_module(f".{name}", package=__name__)
            # Cache on the package so subsequent access is O(1)
            import sys
            sys.modules[f"{__name__}.{name}"] = mod
            globals()[name] = mod
            return mod
        except ImportError as exc:
            raise AttributeError(
                f"Optional submodule '{name}' could not be loaded: {exc}"
            ) from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ============================================================================
# PUBLIC API — classes & functions directly accessible at top level
# ============================================================================

try:
    from .nn import (
        Module,
        Linear,
        Conv2D,
        LSTMCell,
        GRUCell,
        RNNCell,
        ReLU,
        Sigmoid,
        Tanh,
        Softmax,
        LeakyReLU,
        BatchNorm1d,
        MaxPool2D,
        AvgPool2D,
        Dropout,
        MSELoss,
        CrossEntropyLoss,
        BCELoss,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Some nn classes could not be imported: {e}", ImportWarning)

# ============================================================================
# __all__
# ============================================================================

__all__ = [
    # Metadata
    "__version__", "__author__", "__email__",
    # Core
    "Tensor", "concatenate", "stack",
    # nn classes
    "Module", "Linear", "Conv2D",
    "LSTMCell", "GRUCell", "RNNCell",
    "ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU",
    "BatchNorm1d", "MaxPool2D", "AvgPool2D", "Dropout",
    "MSELoss", "CrossEntropyLoss", "BCELoss",
    # Submodules (all lazy)
    "nn", "ml", "neat", "preprocessing", "optim", "training",
    "explain", "tinyml", "federated", "nas", "multimodal",
    "benchmark", "inspect", "data", "distill", "robust", "experiments",
]
