import numpy as np
from scipy import stats
from ..base import BaseTransformer


class Normalizer(BaseTransformer):
    """
    Normalize samples individually to unit norm

    Parameters:
    -----------
    norm : str, default='l2'
        The norm to use ('l1', 'l2', or 'max')

    Example:
    --------
    >>> from mayini.preprocessing import Normalizer
    >>> normalizer = Normalizer(norm='l2')
    >>> X = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
    >>> normalizer.fit_transform(X)
    """

    def __init__(self, norm='l2'):
        super().__init__()
        self.norm = norm

    def fit(self, X, y=None):
        """Do nothing, normalization is done row-wise"""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Normalize each sample"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.norm == 'l1':
            norms = np.abs(X).sum(axis=1)
        elif self.norm == 'l2':
            norms = np.sqrt((X ** 2).sum(axis=1))
        elif self.norm == 'max':
            norms = np.abs(X).max(axis=1)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        norms[norms == 0] = 1  # Avoid division by zero
        return X / norms[:, np.newaxis]


class PowerTransformer(BaseTransformer):
    """
    Apply power transform to make data more Gaussian-like

    Parameters:
    -----------
    method : str, default='yeo-johnson'
        Power transform method ('yeo-johnson' or 'box-cox')
        Box-Cox requires strictly positive data
    standardize : bool, default=True
        Apply zero-mean, unit-variance normalization

    Example:
    --------
    >>> from mayini.preprocessing import PowerTransformer
    >>> pt = PowerTransformer(method='yeo-johnson')
    >>> X = np.array([[1, 2], [3, 2], [4, 5]])
    >>> pt.fit_transform(X)
    """

    def __init__(self, method='yeo-johnson', standardize=True):
        super().__init__()
        self.method = method
        self.standardize = standardize
        self.lambdas_ = None
        self.mean_ = None
        self.std_ = None

    def _yeo_johnson_transform(self, X, lmbda):
        """Apply Yeo-Johnson transformation"""
        X_trans = np.zeros_like(X)

        for i in range(X.shape[1]):
            x = X[:, i]
            l = lmbda[i]

            pos_mask = x >= 0
            neg_mask = x < 0

            if abs(l) < 1e-10:
                X_trans[pos_mask, i] = np.log1p(x[pos_mask])
            else:
                X_trans[pos_mask, i] = (np.power(x[pos_mask] + 1, l) - 1) / l

            if abs(l - 2) < 1e-10:
                X_trans[neg_mask, i] = -np.log1p(-x[neg_mask])
            else:
                X_trans[neg_mask, i] = -(np.power(-x[neg_mask] + 1, 2 - l) - 1) / (2 - l)

        return X_trans

    def _box_cox_transform(self, X, lmbda):
        """Apply Box-Cox transformation"""
        if (X <= 0).any():
            raise ValueError("Box-Cox requires strictly positive data")

        X_trans = np.zeros_like(X)
        for i in range(X.shape[1]):
            if abs(lmbda[i]) < 1e-10:
                X_trans[:, i] = np.log(X[:, i])
            else:
                X_trans[:, i] = (np.power(X[:, i], lmbda[i]) - 1) / lmbda[i]

        return X_trans

    def fit(self, X, y=None):
        """Fit the power transformer"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_features = X.shape[1]
        self.lambdas_ = np.zeros(n_features)

        for i in range(n_features):
            if self.method == 'yeo-johnson':
                best_lambda = 1.0
                best_score = -np.inf

                for lmbda in np.linspace(-2, 2, 41):
                    try:
                        transformed = self._yeo_johnson_transform(X[:, [i]], [lmbda])
                        _, p_value = stats.normaltest(transformed[:, 0])
                        if p_value > best_score:
                            best_score = p_value
                            best_lambda = lmbda
                    except:
                        continue

                self.lambdas_[i] = best_lambda

            elif self.method == 'box-cox':
                if (X[:, i] > 0).all():
                    self.lambdas_[i], _ = stats.boxcox_normmax(X[:, i])
                else:
                    raise ValueError("Box-Cox requires positive data")

        if self.standardize:
            if self.method == 'yeo-johnson':
                X_trans = self._yeo_johnson_transform(X, self.lambdas_)
            else:
                X_trans = self._box_cox_transform(X, self.lambdas_)

            self.mean_ = np.mean(X_trans, axis=0)
            self.std_ = np.std(X_trans, axis=0)
            self.std_[self.std_ == 0] = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply power transformation"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == 'yeo-johnson':
            X_trans = self._yeo_johnson_transform(X, self.lambdas_)
        else:
            X_trans = self._box_cox_transform(X, self.lambdas_)

        if self.standardize:
            X_trans = (X_trans - self.mean_) / self.std_

        return X_trans
