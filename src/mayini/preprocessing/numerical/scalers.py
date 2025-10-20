"""Scaling transformers"""
import numpy as np
from ..base import BaseTransformer

class StandardScaler(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # Avoid division by zero
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted first")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_


class MinMaxScaler(BaseTransformer):
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = (self.feature_range - self.feature_range) / (self.max_ - self.min_ + 1e-8)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted first")
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.min_) * self.scale_ + self.feature_range
        return X_scaled


class RobustScaler(BaseTransformer):
    def __init__(self, quantile_range=(25.0, 75.0)):
        super().__init__()
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.center_ = np.median(X, axis=0)
        q1 = np.percentile(X, self.quantile_range, axis=0)
        q3 = np.percentile(X, self.quantile_range, axis=0)
        self.scale_ = q3 - q1
        self.scale_[self.scale_ == 0] = 1.0
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted first")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.center_) / self.scale_

