import numpy as np
from ..base import BaseTransformer


class CorrelationSelector(BaseTransformer):
    """
    Select features based on correlation with target

    Parameters
    ----------
    threshold : float, default=0.5
        Minimum absolute correlation with target
    method : str, default='pearson'
        Correlation method ('pearson' or 'spearman')

    Example
    -------
    >>> from mayini.preprocessing import CorrelationSelector
    >>> selector = CorrelationSelector(threshold=0.3)
    >>> X = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    >>> y = [1, 2, 3]
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(self, threshold=0.5, method="pearson"):
        super().__init__()
        self.threshold = threshold
        self.method = method
        self.correlations_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        """Compute correlations with target"""
        if y is None:
            raise ValueError("CorrelationSelector requires target y")

        X, y = self._validate_input(X, y)

        self.correlations_ = []
        for col in range(X.shape[1]):
            if self.method == "pearson":
                corr = np.corrcoef(X[:, col], y)[0, 1]
            elif self.method == "spearman":
                # Spearman correlation (rank-based)
                from scipy.stats import spearmanr

                corr, _ = spearmanr(X[:, col], y)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Handle NaN correlations
            if np.isnan(corr):
                corr = 0.0

            self.correlations_.append(abs(corr))

        self.correlations_ = np.array(self.correlations_)
        self.selected_features_ = self.correlations_ >= self.threshold

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Select features based on correlation"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return X[:, self.selected_features_]

    def get_support(self, indices=False):
        """Get mask or indices of selected features"""
        self._check_is_fitted()
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_
