import numpy as np
from ..base import BaseTransformer


class CorrelationThreshold(BaseTransformer):
    """
    Remove highly correlated features

    Parameters:
    -----------
    threshold : float, default=0.9
        Correlation threshold above which features are removed

    Example:
    --------
    >>> from mayini.preprocessing import CorrelationThreshold
    >>> selector = CorrelationThreshold(threshold=0.95)
    >>> X = np.array([[1, 1, 1], [2, 2, 3], [3, 3, 2], [4, 4, 4]])
    >>> selector.fit_transform(X)
    """

    def __init__(self, threshold=0.9):
        super().__init__()
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Identify highly correlated features"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        corr_matrix = np.corrcoef(X.T)
        n_features = X.shape[1]
        to_keep = np.ones(n_features, dtype=bool)

        for i in range(n_features):
            if not to_keep[i]:
                continue

            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > self.threshold:
                    to_keep[j] = False

        self.selected_features_ = to_keep
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Remove correlated features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X[:, self.selected_features_]

    def get_support(self):
        """Get mask of selected features"""
        self._check_is_fitted()
        return self.selected_features_

