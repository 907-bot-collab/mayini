"""Dimensionality reduction"""
import numpy as np
from ..base import BaseEstimator

class PCA(BaseEstimator):
    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X, y=None):
        X, _ = self._validate_input(X)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store components
        if self.n_components is None:
            self.n_components = X.shape

        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        return X_transformed @ self.components_ + self.mean_

