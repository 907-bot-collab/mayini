"""Correlation-based feature selection"""
import numpy as np


class CorrelationSelector:
    """
    Select features based on correlation with target
    
    Parameters
    ----------
    threshold : float, default=0.5
        Minimum absolute correlation with target
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.correlations_ = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Compute correlations"""
        X = np.array(X)

        if y is not None:
            # Feature-Target correlation
            y = np.array(y)
            self.correlations_ = []
            for col in range(X.shape[1]):
                corr = np.corrcoef(X[:, col], y)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                self.correlations_.append(abs(corr))

            self.correlations_ = np.array(self.correlations_)
            self.selected_features_ = self.correlations_ >= self.threshold
        else:
            # Feature-Feature correlation
            corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
            to_drop = set()
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    if corr_matrix[i, j] > self.threshold:
                        to_drop.add(j)

            self.selected_features_ = np.ones(X.shape[1], dtype=bool)
            for idx in to_drop:
                self.selected_features_[idx] = False

        return self

    def transform(self, X):
        """Select features based on correlation"""
        X = np.array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class CorrelationThreshold(CorrelationSelector):
    """Alias for CorrelationSelector"""

    pass
