import numpy as np
from itertools import combinations_with_replacement
from ..base import BaseTransformer


class PolynomialFeatures(BaseTransformer):
    """
    Generate polynomial and interaction features

    Parameters:
    -----------
    degree : int, default=2
        Degree of polynomial features
    interaction_only : bool, default=False
        If True, only interaction features (no powers)
    include_bias : bool, default=True
        If True, include bias column (all ones)

    Example:
    --------
    >>> from mayini.preprocessing import PolynomialFeatures
    >>> poly = PolynomialFeatures(degree=2)
    >>> X = np.array([[0, 1], [2, 3], [4, 5]])
    >>> poly.fit_transform(X)
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def _combinations(self, n_features, degree):
        """Generate combinations of feature indices"""
        if self.interaction_only:
            combs = [
                c for c in combinations_with_replacement(range(n_features), degree)
                if len(set(c)) == degree
            ]
        else:
            combs = combinations_with_replacement(range(n_features), degree)

        return list(combs)

    def fit(self, X, y=None):
        """Compute number of output features"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_input_features_ = X.shape[1]

        n_output_features = 0
        if self.include_bias:
            n_output_features += 1

        for deg in range(1, self.degree + 1):
            n_output_features += len(self._combinations(self.n_input_features_, deg))

        self.n_output_features_ = n_output_features
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data to polynomial features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        features = []

        if self.include_bias:
            features.append(np.ones((n_samples, 1)))

        for degree in range(1, self.degree + 1):
            combs = self._combinations(self.n_input_features_, degree)

            for comb in combs:
                feature = np.ones(n_samples)
                for idx in comb:
                    feature *= X[:, idx]
                features.append(feature.reshape(-1, 1))

        return np.hstack(features)
