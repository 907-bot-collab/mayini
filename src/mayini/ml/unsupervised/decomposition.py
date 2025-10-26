import numpy as np
from ..base import BaseEstimator


class PCA(BaseEstimator):
    """
    Principal Component Analysis

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep

    Example
    -------
    >>> from mayini.ml import PCA
    >>> pca = PCA(n_components=2)
    >>> X_reduced = pca.fit_transform(X)
    """

    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None

    def fit(self, X, y=None):
        """Fit PCA"""
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
            self.n_components = X.shape[1]

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)

    def predict(self, X):
        """Alias for transform"""
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Inverse transform"""
        self._check_is_fitted()
        return X_transformed @ self.components_ + self.mean_


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis

    Dimensionality reduction with class separation

    Parameters
    ----------
    n_components : int, default=None
        Number of components (max: n_classes - 1)

    Example
    -------
    >>> from mayini.ml import LDA
    >>> lda = LDA(n_components=2)
    >>> X_lda = lda.fit_transform(X, y)
    """

    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X, y):
        """Fit LDA"""
        if y is None:
            raise ValueError("LDA requires target labels")

        X, y = self._validate_input(X, y)

        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = len(classes)

        # Default n_components
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)

        # Overall mean
        mean_overall = np.mean(X, axis=0)
        self.mean_ = mean_overall

        # Within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        # Between-class scatter matrix
        S_B = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            n_c = X_c.shape[0]

            # Within-class scatter
            S_W += (X_c - mean_c).T @ (X_c - mean_c)

            # Between-class scatter
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ratio_ = (
            eigenvalues[: self.n_components] / np.sum(eigenvalues)
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X, y):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Alias for transform"""
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Inverse transform"""
        self._check_is_fitted()
        return X_transformed @ self.components_ + self.mean_
        
