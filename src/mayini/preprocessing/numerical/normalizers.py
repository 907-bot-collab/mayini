

# ============================================================================
# BUNDLE: REMAINING PREPROCESSING FILES
# ============================================================================

# ============================================================================
# FILE: preprocessing/numerical/normalizers.py
# ============================================================================
"""
Normalization and power transformations
"""

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
        The power transform method ('yeo-johnson' or 'box-cox')
    standardize : bool, default=True
        Whether to zero-mean unit-variance normalize after transform
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
        """Apply Box-Cox transformation (requires positive data)"""
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

        # Estimate optimal lambda for each feature
        for i in range(n_features):
            if self.method == 'yeo-johnson':
                # Use simple grid search for lambda
                best_lambda = 1.0
                best_score = -np.inf

                for lmbda in np.linspace(-2, 2, 41):
                    try:
                        transformed = self._yeo_johnson_transform(X[:, [i]], [lmbda])
                        # Use normality test score
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

        # Compute mean and std for standardization
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


# ============================================================================
# FILE: preprocessing/categorical/target_encoding.py
# ============================================================================
"""
Target encoding for categorical features
"""

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

                    # Apply smoothing
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


# ============================================================================
# FILE: preprocessing/feature_engineering/polynomial.py
# ============================================================================
"""
Polynomial feature generation
"""

import numpy as np
from itertools import combinations_with_replacement
from ..base import BaseTransformer


class PolynomialFeatures(BaseTransformer):
    """
    Generate polynomial and interaction features

    Parameters:
    -----------
    degree : int, default=2
        The degree of polynomial features
    interaction_only : bool, default=False
        If True, only interaction features are produced
    include_bias : bool, default=True
        If True, include a bias column (all ones)
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
            # Only interaction terms (no powers)
            combs = [
                c for c in combinations_with_replacement(range(n_features), degree)
                if len(set(c)) == degree
            ]
        else:
            # All polynomial terms
            combs = combinations_with_replacement(range(n_features), degree)

        return list(combs)

    def fit(self, X, y=None):
        """Compute number of output features"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_input_features_ = X.shape[1]

        # Count output features
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

        # Add features for each degree
        for degree in range(1, self.degree + 1):
            combs = self._combinations(self.n_input_features_, degree)

            for comb in combs:
                feature = np.ones(n_samples)
                for idx in comb:
                    feature *= X[:, idx]
                features.append(feature.reshape(-1, 1))

        return np.hstack(features)


# ============================================================================
# FILE: preprocessing/feature_engineering/interactions.py
# ============================================================================
"""
Feature interaction generation
"""

import numpy as np
from ..base import BaseTransformer


class FeatureInteractions(BaseTransformer):
    """
    Generate pairwise feature interactions

    Parameters:
    -----------
    interaction_type : str, default='multiply'
        Type of interaction ('multiply', 'add', 'subtract', 'divide')
    """

    def __init__(self, interaction_type='multiply'):
        super().__init__()
        self.interaction_type = interaction_type
        self.n_features_ = None

    def fit(self, X, y=None):
        """Fit transformer"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Generate interaction features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        interactions = [X]

        for i in range(self.n_features_):
            for j in range(i + 1, self.n_features_):
                if self.interaction_type == 'multiply':
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'add':
                    interaction = (X[:, i] + X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'subtract':
                    interaction = (X[:, i] - X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'divide':
                    interaction = (X[:, i] / (X[:, j] + 1e-10)).reshape(-1, 1)
                else:
                    raise ValueError(f"Unknown interaction type: {self.interaction_type}")

                interactions.append(interaction)

        return np.hstack(interactions)


# ============================================================================
# FILE: preprocessing/selection/variance.py
# ============================================================================
"""
Feature selection based on variance
"""

import numpy as np
from ..base import BaseTransformer


class VarianceThreshold(BaseTransformer):
    """
    Remove features with low variance

    Parameters:
    -----------
    threshold : float, default=0.0
        Features with variance below this threshold will be removed
    """

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.variances_ = None
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Compute variances"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.variances_ = np.var(X, axis=0)
        self.selected_features_ = self.variances_ > self.threshold

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Remove low-variance features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X[:, self.selected_features_]


# ============================================================================
# FILE: preprocessing/selection/correlation.py
# ============================================================================
"""
Feature selection based on correlation
"""

import numpy as np
from ..base import BaseTransformer


class CorrelationThreshold(BaseTransformer):
    """
    Remove highly correlated features

    Parameters:
    -----------
    threshold : float, default=0.9
        Correlation threshold above which features are removed
    """

    def __init__(self, threshold=0.9):
        super().__init__()
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Identify highly correlated features"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)

        # Find features to keep
        n_features = X.shape[1]
        to_keep = np.ones(n_features, dtype=bool)

        for i in range(n_features):
            if not to_keep[i]:
                continue

            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > self.threshold:
                    to_keep[j] = False

        self.selected_features_ = to_keep
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Remove correlated features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X[:, self.selected_features_]
