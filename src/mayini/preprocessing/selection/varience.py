import numpy as np
from ..base import BaseTransformer


class VarianceThreshold(BaseTransformer):
    """
    Remove low-variance features

    Parameters:
    -----------
    threshold : float, default=0.0
        Features with variance below threshold are removed

    Example:
    --------
    >>> from mayini.preprocessing import VarianceThreshold
    >>> selector = VarianceThreshold(threshold=0.1)
    >>> X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]])
    >>> selector.fit_transform(X)
    """

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.variances_ = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Compute variances"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.variances_ = np.var(X, axis=0)
        self.selected_features_ = self.variances_ > self.threshold

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Remove low-variance features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X[:, self.selected_features_]

    def get_support(self):
        """Get mask of selected features"""
        self._check_is_fitted()
        return self.selected_features_
