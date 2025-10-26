import numpy as np
from ..base import BaseTransformer


class VarianceThreshold(BaseTransformer):
    """
    Remove features with low variance

    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance below this threshold will be removed

    Example
    -------
    >>> from mayini.preprocessing import VarianceThreshold
    >>> selector = VarianceThreshold(threshold=0.1)
    >>> X = [[0, 2, 0], [0, 3, 0], [0, 4, 0]]
    >>> X_selected = selector.fit_transform(X)
    >>> # Removes first and third columns (zero variance)
    """

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.variances_ = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Compute variances"""
        X, _ = self._validate_input(X)

        self.variances_ = np.var(X, axis=0)
        self.selected_features_ = self.variances_ > self.threshold

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Remove low-variance features"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """
        Get mask or indices of selected features

        Parameters
        ----------
        indices : bool, default=False
            If True, return indices; if False, return boolean mask

        Returns
        -------
        array-like
            Mask or indices of selected features
        """
        self._check_is_fitted()
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_
