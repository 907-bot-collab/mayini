import numpy as np
from .base import BaseTransformer


class OutlierDetector(BaseTransformer):
    """
    Detect and handle outliers using IQR or Z-score methods

    Parameters:
    -----------
    method : str, default='iqr'
        Detection method ('iqr' or 'zscore')
    threshold : float, default=1.5
        Threshold for outlier detection
        - For IQR: multiplier for IQR (typically 1.5 or 3.0)
        - For Z-score: number of standard deviations (typically 3.0)
    action : str, default='remove'
        Action to take ('remove' or 'cap')
        - 'remove': Remove rows containing outliers
        - 'cap': Cap outliers to threshold values

    Example:
    --------
    >>> from mayini.preprocessing import OutlierDetector
    >>> detector = OutlierDetector(method='iqr', threshold=1.5, action='cap')
    >>> X = np.array([[1], [2], [3], [4], [100]])
    >>> detector.fit_transform(X)
    """

    def __init__(self, method='iqr', threshold=1.5, action='remove'):
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.action = action
        self.bounds_ = {}

    def fit(self, X, y=None):
        """Compute outlier bounds"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]

        for i in range(n_features):
            if self.method == 'iqr':
                Q1 = np.percentile(X[:, i], 25)
                Q3 = np.percentile(X[:, i], 75)
                IQR = Q3 - Q1
                lower = Q1 - self.threshold * IQR
                upper = Q3 + self.threshold * IQR
                self.bounds_[i] = {'lower': lower, 'upper': upper}

            elif self.method == 'zscore':
                mean = np.mean(X[:, i])
                std = np.std(X[:, i])
                lower = mean - self.threshold * std
                upper = mean + self.threshold * std
                self.bounds_[i] = {'lower': lower, 'upper': upper}

            else:
                raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Handle outliers"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64).copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.action == 'remove':
            mask = np.ones(X.shape[0], dtype=bool)

            for i, bounds in self.bounds_.items():
                mask &= (X[:, i] >= bounds['lower']) & (X[:, i] <= bounds['upper'])

            return X[mask]

        elif self.action == 'cap':
            for i, bounds in self.bounds_.items():
                X[:, i] = np.clip(X[:, i], bounds['lower'], bounds['upper'])

            return X

        else:
            raise ValueError(f"Unknown action: {self.action}")
