"""
mayini.benchmark — Benchmarking & Model Zoo

Provides standardised benchmark runners, latency/memory profiling,
and a simple model registry (zoo).
"""

from .runner import BenchmarkRunner
from .zoo import ModelZoo

__all__ = ["BenchmarkRunner", "ModelZoo"]
