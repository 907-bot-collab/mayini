"""Clustering algorithms"""
import numpy as np
from ..base import BaseCluster

class KMeans(BaseCluster):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        if self.random_state:
            np.random.seed(self.random_state)

        # Initialize centroids
        indices = np.random.choice(X.shape, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_] for x in X])
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centers = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Check convergence
            if np.allclose(self.cluster_centers_, new_centers, atol=self.tol):
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        self.is_fitted_ = True
        return self

    def predict(self, X):
        distances = np.array([[np.linalg.norm(x - c) for c in self.cluster_centers_] for x in X])
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

