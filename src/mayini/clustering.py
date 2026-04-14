"""
Clustering module alias for backward compatibility.
"""
from .ml.unsupervised.clustering import KMeans, DBSCAN, AgglomerativeClustering, HierarchicalClustering

__all__ = ["KMeans", "DBSCAN", "AgglomerativeClustering", "HierarchicalClustering"]
