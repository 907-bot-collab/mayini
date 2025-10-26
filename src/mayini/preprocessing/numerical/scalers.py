import numpy as np
from ..base import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing mean and scaling to unit variance

    Formula: z = (x - mean) / std

    Example
    -------
    >>> from mayini.preprocessing import StandardScaler
    >>> scaler = StandardScaler()
    >>> X = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        """Compute mean and std for scaling"""
        X, _ = self._validate_input(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Scale features"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        """Inverse scale features"""
        self._check_is_fitted()
        X = np.asarray(X)
        return X * self.std_ + self.mean_


class MinMaxScaler(BaseTransformer):
    """
    Scale features to a given range (default [0, 1])

    Formula: X_scaled = (X - X_min) / (X_max - X_min)

    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data

    Example
    -------
    >>> from mayini.preprocessing import MinMaxScaler
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> X = [[1, 2], [2, 4], [3, 6]]
    >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute min and max for scaling"""
        X, _ = self._validate_input(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        # Compute scale
        data_range = self.max_ - self.min_
        data_range[data_range == 0] = 1.0
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Scale features to range"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        feature_min, _ = self.feature_range
        X_scaled = (X - self.min_) * self.scale_ + feature_min
        return X_scaled

    def inverse_transform(self, X):
        """Inverse scaling"""
        self._check_is_fitted()
        X = np.asarray(X)
        feature_min, _ = self.feature_range
        X_original = (X - feature_min) / self.scale_ + self.min_
        return X_original


class RobustScaler(BaseTransformer):
    """
    Scale features using statistics robust to outliers

    Uses median and interquartile range (IQR)

    Parameters
    ----------
    quantile_range : tuple, default=(25.0, 75.0)
        Quantile range used to calculate scale

    Example
    -------
    >>> from mayini.preprocessing import RobustScaler
    >>> scaler = RobustScaler()
    >>> X = [[1, 2], [2, 4], [3, 6], [100, 200]]
    >>> X_scaled = scaler.fit_transform(X)
    """

    def __init__(self, quantile_range=(25.0, 75.0)):
        super().__init__()
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """Compute center and scale"""
        X, _ = self._validate_input(X)

        # Use median as center
        self.center_ = np.median(X, axis=0)

        # Use IQR as scale
        q_min, q_max = self.quantile_range
        quantile_min = np.percentile(X, q_min, axis=0)
        quantile_max = np.percentile(X, q_max, axis=0)
        self.scale_ = quantile_max - quantile_min
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Scale features"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X):
        """Inverse scale features"""
        self._check_is_fitted()
        X = np.asarray(X)
        return X * self.scale_ + self.center_
