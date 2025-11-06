import numpy as np

class VarianceThreshold:
    """
    Feature selector that removes all low-variance features.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None

    def fit(self, X, y=None):
        if isinstance(X, list):
            X = np.array(X)
        self.variances_ = np.var(X, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if self.variances_ is None:
            raise ValueError("VarianceThreshold has not been fitted yet.")
        mask = self.variances_ > self.threshold
        return X[:, mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
