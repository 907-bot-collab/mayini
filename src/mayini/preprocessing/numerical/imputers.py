"""
src/mayini/preprocessing/numerical/imputers.py
Missing value imputation strategies
"""

import numpy as np
from ..base import BaseTransformer


class SimpleImputer(BaseTransformer):
    """
    Imputation transformer for completing missing values

    Parameters:
    -----------
    strategy : str, default='mean'
        The imputation strategy ('mean', 'median', 'most_frequent', 'constant')
    fill_value : scalar, default=None
        When strategy='constant', fill_value is used to replace missing values
    missing_values : number, default=np.nan
        The placeholder for the missing values
    """

    def __init__(self, strategy='mean', fill_value=None, missing_values=np.nan):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_ = None

    def _get_mask(self, X):
        """Get mask of missing values"""
        if np.isnan(self.missing_values):
            return np.isnan(X)
        else:
            return X == self.missing_values

    def _most_frequent(self, array):
        """Get most frequent value"""
        values, counts = np.unique(array, return_counts=True)
        return values[counts.argmax()]

    def fit(self, X, y=None):
        """Compute the statistics for imputation"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask = self._get_mask(X)
        n_features = X.shape[1]

        self.statistics_ = np.zeros(n_features)

        for i in range(n_features):
            column = X[:, i]
            column_mask = mask[:, i]

            if not column_mask.all():  # If not all values are missing
                valid_values = column[~column_mask]

                if self.strategy == 'mean':
                    self.statistics_[i] = np.mean(valid_values)
                elif self.strategy == 'median':
                    self.statistics_[i] = np.median(valid_values)
                elif self.strategy == 'most_frequent':
                    self.statistics_[i] = self._most_frequent(valid_values)
                elif self.strategy == 'constant':
                    if self.fill_value is None:
                        raise ValueError("fill_value must be provided for strategy='constant'")
                    self.statistics_[i] = self.fill_value
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
            else:
                # If all values are missing, use 0 or fill_value
                self.statistics_[i] = self.fill_value if self.fill_value is not None else 0

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Impute all missing values in X"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64).copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask = self._get_mask(X)

        for i in range(X.shape[1]):
            X[mask[:, i], i] = self.statistics_[i]

        return X


class KNNImputer(BaseTransformer):
    """
    Imputation using k-Nearest Neighbors

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighboring samples to use
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')
    missing_values : number, default=np.nan
        The placeholder for missing values
    """

    def __init__(self, n_neighbors=5, weights='uniform', missing_values=np.nan):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.missing_values = missing_values
        self.X_fit_ = None

    def _get_mask(self, X):
        """Get mask of missing values"""
        if np.isnan(self.missing_values):
            return np.isnan(X)
        else:
            return X == self.missing_values

    def _euclidean_distance(self, x1, x2, valid_mask):
        """Compute distance only on valid features"""
        diff = (x1 - x2) ** 2
        diff = diff * valid_mask  # Only consider valid features
        return np.sqrt(np.sum(diff))

    def fit(self, X, y=None):
        """Fit the imputer on X"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_fit_ = X.copy()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Impute all missing values in X"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64).copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mask = self._get_mask(X)

        # Impute each sample
        for i in range(X.shape[0]):
            missing_features = mask[i]

            if not missing_features.any():
                continue

            # Find k nearest neighbors
            valid_features = ~missing_features

            distances = []
            for j in range(self.X_fit_.shape[0]):
                if i == j:
                    continue

                # Only compute distance on valid features
                dist = self._euclidean_distance(X[i], self.X_fit_[j], valid_features)
                distances.append((dist, j))

            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.n_neighbors]

            # Impute missing features
            for feature_idx in np.where(missing_features)[0]:
                neighbor_values = []
                neighbor_weights = []

                for dist, neighbor_idx in k_nearest:
                    if not self._get_mask(self.X_fit_[[neighbor_idx]])[0, feature_idx]:
                        neighbor_values.append(self.X_fit_[neighbor_idx, feature_idx])

                        if self.weights == 'distance':
                            weight = 1 / (dist + 1e-10)
                        else:
                            weight = 1
                        neighbor_weights.append(weight)

                if neighbor_values:
                    X[i, feature_idx] = np.average(neighbor_values, weights=neighbor_weights)
                else:
                    # If no neighbors have valid value, use column mean
                    valid_values = self.X_fit_[:, feature_idx][
                        ~self._get_mask(self.X_fit_[:, [feature_idx]]).flatten()
                    ]
                    if len(valid_values) > 0:
                        X[i, feature_idx] = np.mean(valid_values)
                    else:
                        X[i, feature_idx] = 0

        return X

