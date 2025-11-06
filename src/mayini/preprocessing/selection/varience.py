import numpy as np

class VarianceThreshold:
    """
    Feature selector that removes all low-variance features.
    """
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None

    def fit(self, X, y=None):
        # Compute variance for each feature
        self.variances_ = np.var(X, axis=0)
        return self

    def transform(self, X):
        if self.variances_ is None:
            raise ValueError("Fit the VarianceThreshold before calling transform.")
        return X[:, self.variances_ > self.threshold]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
