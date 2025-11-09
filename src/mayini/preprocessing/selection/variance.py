"""Variance-based feature selection"""
import numpy as np


class VarianceThreshold:
    """
    Remove features with low variance
    
    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance below this threshold will be removed
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Compute variances"""
        X = np.array(X)
        self.variances_ = np.var(X, axis=0)
        self.selected_features_ = self.variances_ > self.threshold
        return self

    def transform(self, X):
        """Remove low-variance features"""
        X = np.array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
