import numpy as np
from .base import BaseTransformer


class IsolationForest(BaseTransformer):
    """
    Isolation Forest for outlier detection

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees
    contamination : float, default=0.1
        Expected proportion of outliers
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.preprocessing import IsolationForest
    >>> iso = IsolationForest(contamination=0.1)
    >>> X = [[1, 2], [2, 3], [3, 4], [100, 200]]
    >>> outliers = iso.fit_predict(X)
    >>> # Returns: [1, 1, 1, -1] (last sample is outlier)
    """

    def __init__(self, n_estimators=100, contamination=0.1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.threshold_ = None

    def fit(self, X, y=None):
        """Fit isolation forest"""
        X, _ = self._validate_input(X)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Compute anomaly scores
        scores = self._compute_anomaly_scores(X)

        # Determine threshold based on contamination
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))

        self.is_fitted_ = True
        return self

    def _compute_anomaly_scores(self, X):
        """Compute anomaly scores (simplified)"""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Distance to k-nearest neighbors (simplified score)
            distances = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            k = min(10, n_samples - 1)
            knn_distances = np.partition(distances, k)[:k]
            scores[i] = np.mean(knn_distances)

        return scores

    def predict(self, X):
        """
        Predict outliers

        Returns
        -------
        array-like
            1 for inliers, -1 for outliers
        """
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        scores = self._compute_anomaly_scores(X)
        predictions = np.where(scores <= self.threshold_, 1, -1)
        return predictions

    def fit_predict(self, X):
        """Fit and predict in one step"""
        return self.fit(X).predict(X)


class LocalOutlierFactor(BaseTransformer):
    """
    Local Outlier Factor for outlier detection

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors
    contamination : float, default=0.1
        Expected proportion of outliers

    Example
    -------
    >>> from mayini.preprocessing import LocalOutlierFactor
    >>> lof = LocalOutlierFactor(n_neighbors=5)
    >>> X = [[1, 2], [2, 3], [3, 4], [100, 200]]
    >>> outliers = lof.fit_predict(X)
    """

    def __init__(self, n_neighbors=20, contamination=0.1):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold_ = None

    def fit(self, X, y=None):
        """Fit LOF"""
        X, _ = self._validate_input(X)

        # Compute LOF scores
        lof_scores = self._compute_lof_scores(X)

        # Determine threshold
        self.threshold_ = np.percentile(lof_scores, 100 * (1 - self.contamination))

        self.is_fitted_ = True
        return self

    def _compute_lof_scores(self, X):
        """Compute Local Outlier Factor scores (simplified)"""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            # Find k-nearest neighbors
            distances = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            k = min(self.n_neighbors, n_samples - 1)
            knn_indices = np.argpartition(distances, k)[:k]

            # Local density (inverse of average distance)
            local_density = 1 / (np.mean(distances[knn_indices]) + 1e-10)

            # Compare with neighbors' densities (simplified LOF)
            neighbor_densities = []
            for j in knn_indices:
                neighbor_distances = np.sqrt(np.sum((X - X[j]) ** 2, axis=1))
                neighbor_k = min(self.n_neighbors, n_samples - 1)
                neighbor_knn = np.argpartition(neighbor_distances, neighbor_k)[
                    :neighbor_k
                ]
                neighbor_density = 1 / (
                    np.mean(neighbor_distances[neighbor_knn]) + 1e-10
                )
                neighbor_densities.append(neighbor_density)

            avg_neighbor_density = np.mean(neighbor_densities)
            scores[i] = avg_neighbor_density / (local_density + 1e-10)

        return scores

    def predict(self, X):
        """Predict outliers"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        scores = self._compute_lof_scores(X)
        predictions = np.where(scores <= self.threshold_, 1, -1)
        return predictions

    def fit_predict(self, X):
        """Fit and predict in one step"""
        return self.fit(X).predict(X)


class OutlierDetector(BaseTransformer):
    """
    Unified interface for outlier detection

    Parameters
    ----------
    method : str, default='iqr'
        Method for outlier detection ('iqr', 'zscore', 'isolation_forest', 'lof')
    action : str, default='remove'
        Action to take ('remove', 'cap', 'flag')
    **kwargs : dict
        Additional parameters for the method
    """

    def __init__(self, method="iqr", action="remove", **kwargs):
        super().__init__()
        self.method = method
        self.action = action
        self.kwargs = kwargs
        self.detector = None

    def fit(self, X, y=None):
        """Fit the outlier detector"""
        X, _ = self._validate_input(X)

        if self.method == "isolation_forest":
            self.detector = IsolationForest(**self.kwargs)
            self.detector.fit(X)
        elif self.method == "lof":
            self.detector = LocalOutlierFactor(**self.kwargs)
            self.detector.fit(X)
        elif self.method == "iqr":
            # Compute IQR thresholds
            self.q1_ = np.percentile(X, 25, axis=0)
            self.q3_ = np.percentile(X, 75, axis=0)
            self.iqr_ = self.q3_ - self.q1_
            self.lower_ = self.q1_ - 1.5 * self.iqr_
            self.upper_ = self.q3_ + 1.5 * self.iqr_
        elif self.method == "zscore":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Handle outliers based on action"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if self.method in ["isolation_forest", "lof"]:
            # These return 1 for inliers, -1 for outliers
            mask = self.detector.predict(X) == 1
        elif self.method == "iqr":
            mask = np.all((X >= self.lower_) & (X <= self.upper_), axis=1)
        elif self.method == "zscore":
            z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
            threshold = self.kwargs.get("threshold", 3.0)
            mask = np.all(z_scores <= threshold, axis=1)

        if self.action == "remove":
            return X[mask]
        elif self.action == "cap":
            if self.method == "iqr":
                return np.clip(X, self.lower_, self.upper_)
            else:
                # Cap at 3 std for zscore
                upper = self.mean_ + 3 * self.std_
                lower = self.mean_ - 3 * self.std_
                return np.clip(X, lower, upper)
        elif self.action == "flag":
            return mask.astype(int)

        return X
