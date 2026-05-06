"""
mayini.nas — Neural Architecture Search

Provides evolutionary and random search over a configurable
layer-level search space, compatible with Mayini's nn API.
"""

from .search_space import SearchSpace, ArchNode
from .controller import NASController

__all__ = ["SearchSpace", "ArchNode", "NASController"]
