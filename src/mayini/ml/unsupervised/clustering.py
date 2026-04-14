import numpy as np
from ..base import BaseEstimator, ClusterMixin


class KMeans(BaseEstimator, ClusterMixin):
    """
    K-Means Clustering
    """

    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Fit K-Means"""
        X, _ = self._validate_input(X)

        if self.random_state:
            np.random.seed(self.random_state)

        # Initialize centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].astype(np.float64)

        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centers = np.array(
                [X[labels == k].mean(axis=0) if np.any(labels == k) else self.cluster_centers_[k]
                 for k in range(self.n_clusters)]
            )

            # Check convergence
            if np.allclose(self.cluster_centers_, new_centers, atol=self.tol):
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict cluster labels"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)


class DBSCAN(BaseEstimator, ClusterMixin):
    """
    Density-Based Spatial Clustering
    """

    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X, y=None):
        """Fit DBSCAN"""
        X, _ = self._validate_input(X)

        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1 for noise
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                continue

            # Start new cluster
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
            cluster_id += 1

        self.is_fitted_ = True
        return self

    def _get_neighbors(self, X, idx):
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        self.labels_[idx] = cluster_id
        
        i = 0
        neighbors = list(neighbors)
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend([n for n in new_neighbors if n not in neighbors])
            
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
            i += 1

    def predict(self, X):
        """DBSCAN does not predict on new data easily, return labels if X is same as fit data"""
        return self.labels_


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    Agglomerative (Hierarchical) Clustering
    """

    def __init__(self, n_clusters=2, linkage="average"):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X, y=None):
        """Fit Agglomerative Clustering"""
        X, _ = self._validate_input(X)
        n_samples = X.shape[0]
        clusters = [[i] for i in range(n_samples)]

        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = cluster_id

        self.is_fitted_ = True
        return self

    def _cluster_distance(self, X, cluster1, cluster2):
        distances = np.linalg.norm(X[cluster1][:, np.newaxis] - X[cluster2], axis=2)
        if self.linkage == "single":
            return np.min(distances)
        elif self.linkage == "complete":
            return np.max(distances)
        elif self.linkage == "average":
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def predict(self, X):
        return self.labels_


class HierarchicalClustering(AgglomerativeClustering):
    """Alias for AgglomerativeClustering"""
    pass
