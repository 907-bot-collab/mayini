import numpy as np
from ..base import BaseTransformer


class SimpleImputer(BaseTransformer):
    """
    Impute missing values using simple strategies

    Parameters
    ----------
    strategy : str, default='mean'
        Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
    fill_value : float, default=None
        Value to use when strategy='constant'

    Example
    -------
    >>> from mayini.preprocessing import SimpleImputer
    >>> imputer = SimpleImputer(strategy='mean')
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(self, strategy="mean", fill_value=None):
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X, y=None):
        """Compute imputation statistics"""
        X, _ = self._validate_input(X)

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            # For each column, find most frequent non-NaN value
            self.statistics_ = []
            for col in range(X.shape[1]):
                col_data = X[:, col]
                col_data_clean = col_data[~np.isnan(col_data)]
                if len(col_data_clean) > 0:
                    values, counts = np.unique(col_data_clean, return_counts=True)
                    most_frequent = values[counts.argmax()]
                    self.statistics_.append(most_frequent)
                else:
                    self.statistics_.append(0)
            self.statistics_ = np.array(self.statistics_)
        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be provided for constant strategy")
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Impute missing values"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        X_imputed = X.copy()

        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            X_imputed[mask, col] = self.statistics_[col]

        return X_imputed


class KNNImputer(BaseTransformer):
    """
    Impute missing values using k-Nearest Neighbors

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')

    Example
    -------
    >>> from mayini.preprocessing import KNNImputer
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> X_imputed = imputer.fit_transform(X)
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_fit_ = None

    def fit(self, X, y=None):
        """Store complete samples for imputation"""
        X, _ = self._validate_input(X)
        self.X_fit_ = X.copy()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Impute missing values using KNN"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        X_imputed = X.copy()

        # For each sample with missing values
        for i in range(X.shape[0]):
            if np.any(np.isnan(X[i])):
                # Find k nearest neighbors from complete cases
                distances = []
                for j in range(self.X_fit_.shape[0]):
                    # Compute distance only on non-missing features
                    mask = ~np.isnan(X[i])
                    if np.sum(mask) > 0:
                        dist = np.sqrt(
                            np.sum((X[i, mask] - self.X_fit_[j, mask]) ** 2)
                        )
                        distances.append((dist, j))

                if distances:
                    # Sort by distance and take k nearest
                    distances.sort()
                    k_nearest_idx = [idx for _, idx in distances[: self.n_neighbors]]

                    # Impute each missing feature
                    for col in range(X.shape[1]):
                        if np.isnan(X[i, col]):
                            neighbor_values = self.X_fit_[k_nearest_idx, col]
                            neighbor_values = neighbor_values[
                                ~np.isnan(neighbor_values)
                            ]

                            if len(neighbor_values) > 0:
                                if self.weights == "uniform":
                                    X_imputed[i, col] = np.mean(neighbor_values)
                                else:
                                    # Distance-weighted mean
                                    neighbor_distances = [
                                        distances[k][0] for k in range(len(k_nearest_idx))
                                    ]
                                    weights = 1 / (np.array(neighbor_distances) + 1e-10)
                                    X_imputed[i, col] = np.average(
                                        neighbor_values, weights=weights[: len(neighbor_values)]
                                    )

        return X_imputed

