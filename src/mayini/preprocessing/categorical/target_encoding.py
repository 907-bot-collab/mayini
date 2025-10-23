import numpy as np
from ..base import BaseTransformer


class TargetEncoder(BaseTransformer):
    """
    Encode categorical features using target statistics

    Parameters:
    -----------
    smoothing : float, default=1.0
        Smoothing parameter for regularization
    min_samples_leaf : int, default=1
        Minimum samples to calculate category statistic

    Example:
    --------
    >>> from mayini.preprocessing import TargetEncoder
    >>> encoder = TargetEncoder(smoothing=1.0)
    >>> X = np.array(['A', 'B', 'A', 'C', 'B', 'A']).reshape(-1, 1)
    >>> y = np.array([1, 0, 1, 1, 0, 1])
    >>> encoder.fit_transform(X, y)
    """

    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        super().__init__()
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings_ = None
        self.global_mean_ = None

    def fit(self, X, y):
        """Fit target encoder"""
        if y is None:
            raise ValueError("Target encoder requires y")

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.global_mean_ = np.mean(y)
        self.encodings_ = []

        for col in range(X.shape[1]):
            encoding = {}
            categories = np.unique(X[:, col])

            for category in categories:
                mask = X[:, col] == category
                n_samples = np.sum(mask)

                if n_samples >= self.min_samples_leaf:
                    category_mean = np.mean(y[mask])
                    smoothed_mean = (
                        n_samples * category_mean + self.smoothing * self.global_mean_
                    ) / (n_samples + self.smoothing)
                    encoding[category] = smoothed_mean
                else:
                    encoding[category] = self.global_mean_

            self.encodings_.append(encoding)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform using target encoding"""
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_encoded = np.zeros(X.shape, dtype=np.float64)

        for col in range(X.shape[1]):
            encoding = self.encodings_[col]
            for i, value in enumerate(X[:, col]):
                X_encoded[i, col] = encoding.get(value, self.global_mean_)

        return X_encoded


class FrequencyEncoder(BaseTransformer):
    """
    Encode categorical features by their frequency

    Example:
    --------
    >>> from mayini.preprocessing import FrequencyEncoder
    >>> encoder = FrequencyEncoder()
    >>> X = np.array(['A', 'B', 'A', 'C', 'B', 'A']).reshape(-1, 1)
    >>> encoder.fit_transform(X)
    """

    def __init__(self):
        super().__init__()
        self.frequencies_ = None

    def fit(self, X, y=None):
        """Fit frequency encoder"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.frequencies_ = []

        for col in range(X.shape[1]):
            values, counts = np.unique(X[:, col], return_counts=True)
            frequencies = counts / len(X)
            freq_dict = {val: freq for val, freq in zip(values, frequencies)}
            self.frequencies_.append(freq_dict)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform using frequency encoding"""
        self._check_is_fitted()
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_encoded = np.zeros(X.shape, dtype=np.float64)

        for col in range(X.shape[1]):
            freq_dict = self.frequencies_[col]
            for i, value in enumerate(X[:, col]):
                X_encoded[i, col] = freq_dict.get(value, 0.0)

        return X_encoded

