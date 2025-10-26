import numpy as np
from scipy import stats
from ..base import BaseTransformer


class Normalizer(BaseTransformer):
    """
    Normalize samples individually to unit norm

    Parameters
    ----------
    norm : str, default='l2'
        Norm to use ('l1', 'l2', or 'max')

    Example
    -------
    >>> from mayini.preprocessing import Normalizer
    >>> normalizer = Normalizer(norm='l2')
    >>> X = [[4, 1, 2], [1, 3, 9]]
    >>> X_normalized = normalizer.fit_transform(X)
    """

    def __init__(self, norm="l2"):
        super().__init__()
        self.norm = norm

    def fit(self, X, y=None):
        """Normalizer doesn't require fitting"""
        X, _ = self._validate_input(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Normalize each sample"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if self.norm == "l1":
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == "l2":
            norms = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
        elif self.norm == "max":
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        # Avoid division by zero
        norms[norms == 0] = 1.0
        return X / norms


class PowerTransformer(BaseTransformer):
    """
    Apply power transform to make data more Gaussian-like

    Uses Box-Cox or Yeo-Johnson transformation

    Parameters
    ----------
    method : str, default='yeo-johnson'
        Transform method ('box-cox' or 'yeo-johnson')

    Example
    -------
    >>> from mayini.preprocessing import PowerTransformer
    >>> pt = PowerTransformer(method='yeo-johnson')
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> X_transformed = pt.fit_transform(X)
    """

    def __init__(self, method="yeo-johnson"):
        super().__init__()
        self.method = method
        self.lambdas_ = None

    def fit(self, X, y=None):
        """Fit power transformer"""
        X, _ = self._validate_input(X)

        self.lambdas_ = []
        for col in range(X.shape[1]):
            col_data = X[:, col]

            if self.method == "yeo-johnson":
                _, lambda_param = stats.yeojohnson(col_data)
            elif self.method == "box-cox":
                # Box-Cox requires positive data
                if np.any(col_data <= 0):
                    raise ValueError("Box-Cox requires strictly positive data")
                _, lambda_param = stats.boxcox(col_data)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.lambdas_.append(lambda_param)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply power transform"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        X_transformed = np.zeros_like(X)

        for col in range(X.shape[1]):
            col_data = X[:, col]
            lambda_param = self.lambdas_[col]

            if self.method == "yeo-johnson":
                X_transformed[:, col] = stats.yeojohnson(col_data, lmbda=lambda_param)
            elif self.method == "box-cox":
                X_transformed[:, col] = stats.boxcox(col_data, lmbda=lambda_param)

        return X_transformed
