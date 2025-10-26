import numpy as np
from ..base import BaseTransformer


class TargetEncoder(BaseTransformer):
    """
    Target encoding (mean encoding)

    Replace categories with mean of target variable

    Parameters
    ----------
    smoothing : float, default=1.0
        Smoothing parameter
    min_samples_leaf : int, default=1
        Minimum samples to calculate statistics

    Example
    -------
    >>> from mayini.preprocessing import TargetEncoder
    >>> te = TargetEncoder()
    >>> X = [['cat'], ['dog'], ['cat']]
    >>> y = [1, 0, 1]
    >>> X_encoded = te.fit_transform(X, y)
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

        X, y = self._validate_input(X, y)

        self.global_mean_ = np.mean(y)
        self.encodings_ = []

        for col in range(X.shape[1]):
            col_encoding = {}
            categories = np.unique(X[:, col])

            for category in categories:
                mask = X[:, col] == category
                category_target = y[mask]
                n_samples = len(category_target)

                if n_samples >= self.min_samples_leaf:
                    # Smoothed mean
                    category_mean = np.mean(category_target)
                    smoothed_mean = (
                        n_samples * category_mean + self.smoothing * self.global_mean_
                    ) / (n_samples + self.smoothing)
                    col_encoding[category] = smoothed_mean
                else:
                    col_encoding[category] = self.global_mean_

            self.encodings_.append(col_encoding)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform using target encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        X_encoded = np.zeros_like(X, dtype=float)

        for col in range(X.shape[1]):
            col_encoding = self.encodings_[col]

            for i, val in enumerate(X[:, col]):
                if val in col_encoding:
                    X_encoded[i, col] = col_encoding[val]
                else:
                    X_encoded[i, col] = self.global_mean_

        return X_encoded


class FrequencyEncoder(BaseTransformer):
    """
    Frequency encoding

    Replace categories with their frequency in training data

    Example
    -------
    >>> from mayini.preprocessing import FrequencyEncoder
    >>> fe = FrequencyEncoder()
    >>> X = [['cat'], ['dog'], ['cat'], ['cat']]
    >>> X_encoded = fe.fit_transform(X)
    >>> # cat appears 3 times, dog 1 time
    """

    def __init__(self):
        super().__init__()
        self.frequencies_ = None

    def fit(self, X, y=None):
        """Fit frequency encoder"""
        X, _ = self._validate_input(X)

        self.frequencies_ = []

        for col in range(X.shape[1]):
            categories, counts = np.unique(X[:, col], return_counts=True)
            total = len(X)
            col_frequencies = {
                cat: count / total for cat, count in zip(categories, counts)
            }
            self.frequencies_.append(col_frequencies)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform using frequency encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        X_encoded = np.zeros_like(X, dtype=float)

        for col in range(X.shape[1]):
            col_frequencies = self.frequencies_[col]

            for i, val in enumerate(X[:, col]):
                if val in col_frequencies:
                    X_encoded[i, col] = col_frequencies[val]
                else:
                    X_encoded[i, col] = 0.0

        return X_encoded
