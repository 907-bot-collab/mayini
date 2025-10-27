"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework built from scratch in Python.
"""

__version__ = "0.2.0"
__author__ = "Abhishek Adari"
__email__ = "abhishekadari85@gmail.com"

# Expose only at top-level. DO NOT import sibling submodules here! 
# They are available as `mayini.nn`, `mayini.ml`, etc.
__all__ = [
    "Tensor",
    "nn",
    "ml",
    "neat",
    "preprocessing",
    "optim",
    "training",
]

# It's safe to expose the Tensor class at the root package
from .tensor import Tensor
