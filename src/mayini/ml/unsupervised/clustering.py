import numpy as np
from ..base import BaseEstimator, ClusterMixin


class KMeans(BaseEstimator, ClusterMixin):
    """
    K-Means Clustering

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters
    max_iter : int, default=300
        Maximum iterations
    tol : float, default=1e-4
        Tolerance for convergence
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.ml import KMeans
    >>> kmeans = KMeans(n_clusters=3)
    >>> labels = kmeans.fit_predict(X)
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
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
        self.cluster_centers_ = X[indices]

        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = np.array(
                [[np.linalg.norm(x - c) for c in self.cluster_centers_] for x in X]
            )
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centers = np.array(
                [X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
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

        distances = np.array(
            [[np.linalg.norm(x - c) for c in self.cluster_centers_] for x in X]
        )
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        """Fit and predict"""
        self.fit(X)
        return self.labels_


class DBSCAN(BaseEstimator, ClusterMixin):
    """
    Density-Based Spatial Clustering of Applications with Noise

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between samples
    min_samples : int, default=5
        Minimum samples in neighborhood

    Example
    -------
    >>> from mayini.ml import DBSCAN
    >>> dbscan = DBSCAN(eps=0.5, min_samples=5)
    >>> labels = dbscan.fit_predict(X)
    """

    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.core_sample_indices_ = None

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
                # Mark as noise (will be updated if added to cluster later)
                continue

            # Start new cluster
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
            cluster_id += 1

        self.is_fitted_ = True
        return self

    def _get_neighbors(self, X, idx):
        """Get neighbors within eps distance"""
        distances = np.linalg.norm(X - X[idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, idx, neighbors, cluster_id, visited):
        """Expand cluster from core point"""
        self.labels_[idx] = cluster_id

        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = self._get_neighbors(X, neighbor_idx)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])

            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id

            i += 1

    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_


class AgglomerativeClustering(BaseEstimator, ClusterMixin):
    """
    Agglomerative (Hierarchical) Clustering

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters
    linkage : str, default='average'
        Linkage criterion

    Example
    -------
    >>> from mayini.ml import AgglomerativeClustering
    >>> agg = AgglomerativeClustering(n_clusters=3, linkage='average')
    >>> labels = agg.fit_predict(X)
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

        # Initialize each sample as its own cluster
        clusters = [[i] for i in range(n_samples)]

        # Merge until we have n_clusters
        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = 0, 1

            # Find closest pair of clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)

        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = cluster_id

        self.is_fitted_ = True
        return self

    def _cluster_distance(self, X, cluster1, cluster2):
        """Compute distance between two clusters"""
        from scipy.spatial.distance import cdist

        distances = cdist(X[cluster1], X[cluster2])

        if self.linkage == "single":
            return np.min(distances)
        elif self.linkage == "complete":
            return np.max(distances)
        elif self.linkage == "average":
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def fit_predict(self, X):
        """Fit and return cluster labels"""
        self.fit(X)
        return self.labels_

class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Hierarchical (Agglomerative) Clustering
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters
    linkage : str, default='ward'
        Linkage method: 'ward', 'complete', 'average', 'single'
    
    Example
    -------
    >>> from mayini.ml import HierarchicalClustering
    >>> hc = HierarchicalClustering(n_clusters=3)
    >>> hc.fit(X)
    """
    
    def __init__(self, n_clusters=3, linkage='ward'):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def _euclidean_distance(self, x1, x2):
        """Calculate euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _compute_cluster_distance(self, cluster1, cluster2):
        """Compute distance between clusters"""
        if self.linkage == 'ward':
            # Ward linkage
            return np.linalg.norm(np.mean(cluster1, axis=0) - np.mean(cluster2, axis=0))
        elif self.linkage == 'complete':
            # Complete linkage - maximum distance
            max_dist = 0
            for x1 in cluster1:
                for x2 in cluster2:
                    dist = self._euclidean_distance(x1, x2)
                    max_dist = max(max_dist, dist)
            return max_dist
        elif self.linkage == 'average':
            # Average linkage
            total_dist = 0
            count = 0
            for x1 in cluster1:
                for x2 in cluster2:
                    total_dist += self._euclidean_distance(x1, x2)
                    count += 1
            return total_dist / count
        elif self.linkage == 'single':
            # Single linkage - minimum distance
            min_dist = float('inf')
            for x1 in cluster1:
                for x2 in cluster2:
                    dist = self._euclidean_distance(x1, x2)
                    min_dist = min(min_dist, dist)
            return min_dist
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")

    def fit(self, X):
        """Fit hierarchical clustering"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [[X[i]] for i in range(n_samples)]
        labels = list(range(n_samples))
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find two closest clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_cluster_distance(
                        np.array(clusters[i]), 
                        np.array(clusters[j])
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point in cluster:
                for i in range(n_samples):
                    if np.array_equal(point, X[i]):
                        self.labels_[i] = cluster_id
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict cluster labels (uses fit labels for new data)"""
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        # For hierarchical clustering, typically use training data
        # For new data, assign to nearest cluster center
        return self.labels_

