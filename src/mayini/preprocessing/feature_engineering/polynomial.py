import numpy as np
from itertools import combinations_with_replacement
from ..base import BaseTransformer


class PolynomialFeatures(BaseTransformer):
    """
    Generate polynomial and interaction features

    Parameters
    ----------
    degree : int, default=2
        Polynomial degree
    interaction_only : bool, default=False
        If True, only interaction features are produced
    include_bias : bool, default=True
        If True, include a bias column (all ones)

    Example
    -------
    >>> from mayini.preprocessing import PolynomialFeatures
    >>> poly = PolynomialFeatures(degree=2)
    >>> X = [[1, 2], [3, 4]]
    >>> X_poly = poly.fit_transform(X)
    >>> # Returns: [1, x1, x2, x1^2, x1*x2, x2^2]
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X, y=None):
        """Compute number of output features"""
        X, _ = self._validate_input(X)
        self.n_input_features_ = X.shape[1]

        # Calculate number of output features
        if self.interaction_only:
            # Only interaction terms
            n_output = 0
            for d in range(2, self.degree + 1):
                n_output += self._n_combinations(self.n_input_features_, d)
            if self.include_bias:
                n_output += 1
            n_output += self.n_input_features_  # Original features
        else:
            # All polynomial terms
            n_output = self._n_combinations_with_replacement(
                self.n_input_features_ + self.degree, self.degree
            )
            if not self.include_bias:
                n_output -= 1

        self.n_output_features_ = n_output
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data to polynomial features"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        n_samples = X.shape[0]
        features = []

        if self.include_bias:
            features.append(np.ones((n_samples, 1)))

        if self.interaction_only:
            # Original features
            features.append(X)

            # Interaction terms
            for d in range(2, self.degree + 1):
                for combo in combinations_with_replacement(
                    range(self.n_input_features_), d
                ):
                    if len(set(combo)) > 1:  # Only interactions
                        feature_product = np.prod(X[:, combo], axis=1, keepdims=True)
                        features.append(feature_product)
        else:
            # All polynomial combinations
            for d in range(1, self.degree + 1):
                for combo in combinations_with_replacement(
                    range(self.n_input_features_), d
                ):
                    feature_product = np.prod(X[:, combo], axis=1, keepdims=True)
                    features.append(feature_product)

        return np.hstack(features)

    def get_feature_names(self, input_features=None):
        """Get feature names for output"""
        self._check_is_fitted()

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]

        names = []

        if self.include_bias:
            names.append("1")

        if self.interaction_only:
            # Original features
            names.extend(input_features)

            # Interaction terms
            for d in range(2, self.degree + 1):
                for combo in combinations_with_replacement(range(len(input_features)), d):
                    if len(set(combo)) > 1:
                        feature_name = " ".join([input_features[i] for i in combo])
                        names.append(feature_name)
        else:
            # All polynomial combinations
            for d in range(1, self.degree + 1):
                for combo in combinations_with_replacement(range(len(input_features)), d):
                    feature_name = " ".join([input_features[i] for i in combo])
                    names.append(feature_name)

        return names

    @staticmethod
    def _n_combinations(n, k):
        """Number of combinations without replacement"""
        from math import factorial

        return factorial(n) // (factorial(k) * factorial(n - k))

    @staticmethod
    def _n_combinations_with_replacement(n, k):
        """Number of combinations with replacement"""
        from math import factorial

        return factorial(n + k - 1) // (factorial(k) * factorial(n - 1))

