import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import zscore, chi2
import warnings


# =============================================================================
# 1. ISOLATION FOREST
# =============================================================================

class IsolationForest:
    """
    Isolation Forest for Anomaly Detection

    Isolates anomalies by randomly selecting features and split values.
    Anomalies are isolated quickly, requiring fewer splits.

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of isolation trees

    max_samples : int or float, default='auto'
        Number of samples to draw for each tree

    max_depth : int, default=None
        Maximum depth of trees

    contamination : float, default=0.1
        Expected proportion of anomalies (0.0 to 1.0)

    random_state : int or None, default=None
        Random seed

    Example:
    --------
    >>> iso_forest = IsolationForest(n_estimators=100, contamination=0.1)
    >>> iso_forest.fit(X)
    >>> anomaly_scores = iso_forest.decision_function(X)
    >>> predictions = iso_forest.predict(X)  # -1 for anomalies, 1 for normal
    """

    def __init__(self, n_estimators=100, max_samples='auto', max_depth=None, 
                 contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.contamination = contamination
        self.random_state = random_state

        self.trees = []
        self.offset = 0

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X):
        """
        Fit the Isolation Forest

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)

        self.trees = []

        for _ in range(self.n_estimators):
            # Random sample
            idx = np.random.choice(n_samples, size=self.max_samples, replace=False)
            X_sample = X[idx]

            # Build tree
            tree = self._build_tree(X_sample, depth=0)
            self.trees.append(tree)

        # Calculate offset for normalization
        self.offset = 2.0 * (np.log(self.max_samples - 1) + 0.5772156649) - \
                     2.0 * (self.max_samples - 1) / self.max_samples

        return self

    def _build_tree(self, X, depth):
        """Build a single isolation tree"""
        n_samples, n_features = X.shape

        if depth >= (self.max_depth or np.log2(n_samples)) or n_samples <= 1:
            return {'leaf': True, 'size': n_samples}

        # Random feature and split value
        feature = np.random.randint(0, n_features)
        split_value = np.random.uniform(X[:, feature].min(), X[:, feature].max())

        # Split data
        left_mask = X[:, feature] < split_value
        left_X = X[left_mask]
        right_X = X[~left_mask]

        if len(left_X) == 0 or len(right_X) == 0:
            return {'leaf': True, 'size': n_samples}

        return {
            'leaf': False,
            'feature': feature,
            'split_value': split_value,
            'left': self._build_tree(left_X, depth + 1),
            'right': self._build_tree(right_X, depth + 1),
        }

    def _path_length(self, x, tree, depth=0):
        """Calculate path length in a tree"""
        if tree.get('leaf', False):
            return depth + self._c(tree['size'])

        feature = tree['feature']
        split_value = tree['split_value']

        if x[feature] < split_value:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)

    def _c(self, n):
        """Average path length for unsuccessful search"""
        if n <= 1:
            return 0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def decision_function(self, X):
        """
        Calculate anomaly scores

        Returns:
        --------
        scores : array of shape (n_samples,)
            Anomaly score for each sample (lower = more anomalous)
        """
        X = np.asarray(X, dtype=np.float64)
        scores = np.zeros(len(X))

        for i, x in enumerate(X):
            path_lengths = np.array([self._path_length(x, tree) for tree in self.trees])
            scores[i] = -2.0 ** (-path_lengths.mean() / self.offset)

        return scores

    def predict(self, X):
        """
        Predict anomalies

        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, self.contamination * 100)
        return np.where(scores < threshold, -1, 1)


# =============================================================================
# 2. LOCAL OUTLIER FACTOR (LOF)
# =============================================================================

class LocalOutlierFactor:
    """
    Local Outlier Factor for Anomaly Detection

    Detects anomalies based on local density around each point.
    Points in sparse regions are considered anomalies.

    Parameters:
    -----------
    n_neighbors : int, default=20
        Number of neighbors to use for density estimation

    contamination : float, default=0.1
        Expected proportion of anomalies

    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'cosine')

    Example:
    --------
    >>> lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    >>> lof.fit(X)
    >>> anomaly_scores = lof.decision_function(X)
    >>> predictions = lof.predict(X)  # -1 for anomalies, 1 for normal
    """

    def __init__(self, n_neighbors=20, contamination=0.1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.X_train = None
        self.lofs = None

    def fit(self, X):
        """
        Fit the LOF model

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_train = X

        # Calculate distances
        distances = cdist(X, X, metric=self.metric)

        # Find k-nearest neighbors
        self.k_distances = np.sort(distances, axis=1)[:, self.n_neighbors]

        # Calculate reachability distances
        self.reach_dists = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                self.reach_dists[i, j] = max(distances[i, j], self.k_distances[j])

        # Calculate local reachability density (LRD)
        self.lrds = np.zeros(len(X))
        for i in range(len(X)):
            neighbors = np.argsort(distances[i])[:self.n_neighbors]
            mean_reach = np.mean(self.reach_dists[i, neighbors])
            self.lrds[i] = 1.0 / mean_reach if mean_reach > 0 else 0

        return self

    def decision_function(self, X):
        """
        Calculate LOF scores

        Returns:
        --------
        scores : array of shape (n_samples,)
            LOF score for each sample
        """
        X = np.asarray(X, dtype=np.float64)

        if self.X_train is None:
            raise ValueError("Model not fitted yet")

        distances = cdist(X, self.X_train, metric=self.metric)
        lofs = np.zeros(len(X))

        for i in range(len(X)):
            neighbors = np.argsort(distances[i])[:self.n_neighbors]

            # Calculate LRD for test point
            reach_dists = np.maximum(distances[i, neighbors], self.k_distances[neighbors])
            lrd_i = 1.0 / np.mean(reach_dists) if np.mean(reach_dists) > 0 else 0

            # Calculate LOF
            lof = np.mean(self.lrds[neighbors]) / (lrd_i + 1e-10)
            lofs[i] = lof

        return lofs

    def predict(self, X):
        """
        Predict anomalies

        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        if self.X_train is None:
            raise ValueError("Model not fitted yet")

        # Calculate LOF for training data
        scores_train = self.decision_function(self.X_train)
        threshold = np.percentile(scores_train, (1 - self.contamination) * 100)

        # Predict on test set
        scores = self.decision_function(X)
        return np.where(scores > threshold, -1, 1)


# =============================================================================
# 3. ELLIPTIC ENVELOPE (Mahalanobis Distance)
# =============================================================================

class EllipticEnvelope:
    """
    Elliptic Envelope for Anomaly Detection

    Assumes data comes from a Gaussian distribution.
    Anomalies are points far from the mean using Mahalanobis distance.

    Parameters:
    -----------
    contamination : float, default=0.1
        Expected proportion of anomalies

    robust : bool, default=False
        Use robust covariance estimation

    Example:
    --------
    >>> ee = EllipticEnvelope(contamination=0.1, robust=False)
    >>> ee.fit(X)
    >>> anomaly_scores = ee.decision_function(X)
    >>> predictions = ee.predict(X)  # -1 for anomalies, 1 for normal
    """

    def __init__(self, contamination=0.1, robust=False):
        self.contamination = contamination
        self.robust = robust
        self.mean = None
        self.covariance = None
        self.threshold = None

    def fit(self, X):
        """
        Fit the Elliptic Envelope

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        # Estimate mean and covariance
        self.mean = np.mean(X, axis=0)

        if self.robust:
            # Robust covariance using Minimum Covariance Determinant
            self.covariance = self._mcd_covariance(X)
        else:
            # Standard covariance
            self.covariance = np.cov(X.T)

        # Regularize if singular
        if np.linalg.matrix_rank(self.covariance) < X.shape[1]:
            self.covariance += np.eye(X.shape[1]) * 1e-6

        return self

    def _mcd_covariance(self, X):
        """
        Minimum Covariance Determinant (robust covariance estimation)
        Simplified version using subset of data
        """
        n_samples = len(X)
        n_features = X.shape[1]

        # Use fraction of data for robust estimation
        h = int(n_samples * 0.5)

        # Random subset
        idx = np.random.choice(n_samples, size=h, replace=False)
        X_subset = X[idx]

        return np.cov(X_subset.T)

    def decision_function(self, X):
        """
        Calculate Mahalanobis distance

        Returns:
        --------
        scores : array of shape (n_samples,)
            Mahalanobis distance for each sample
        """
        X = np.asarray(X, dtype=np.float64)

        if self.mean is None:
            raise ValueError("Model not fitted yet")

        diff = X - self.mean
        inv_cov = np.linalg.pinv(self.covariance)

        mahal_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return mahal_dist

    def predict(self, X):
        """
        Predict anomalies

        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return np.where(scores > threshold, -1, 1)


# =============================================================================
# 4. STATISTICAL ANOMALY DETECTION (Z-Score)
# =============================================================================

class StatisticalAnomaly:
    """
    Statistical Anomaly Detection using Z-Score

    Identifies anomalies as points with extreme Z-scores.
    Assumes Gaussian distribution.

    Parameters:
    -----------
    threshold : float, default=3.0
        Z-score threshold for anomaly detection

    Example:
    --------
    >>> stat_anom = StatisticalAnomaly(threshold=3.0)
    >>> stat_anom.fit(X)
    >>> predictions = stat_anom.predict(X)  # -1 for anomalies, 1 for normal
    """

    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        Fit the Statistical Anomaly Detector

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def decision_function(self, X):
        """
        Calculate Z-scores

        Returns:
        --------
        scores : array of shape (n_samples,)
            Maximum Z-score across features for each sample
        """
        X = np.asarray(X, dtype=np.float64)

        if self.mean is None:
            raise ValueError("Model not fitted yet")

        z_scores = np.abs((X - self.mean) / (self.std + 1e-10))
        return np.max(z_scores, axis=1)

    def predict(self, X):
        """
        Predict anomalies

        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        scores = self.decision_function(X)
        return np.where(scores > self.threshold, -1, 1)


# =============================================================================
# 5. KMEANS ANOMALY DETECTION
# =============================================================================

class KMeansAnomaly:
    """
    Anomaly Detection using K-Means Clustering

    Points far from nearest cluster center are considered anomalies.

    Parameters:
    -----------
    n_clusters : int, default=5
        Number of clusters

    contamination : float, default=0.1
        Expected proportion of anomalies

    random_state : int or None, default=None
        Random seed

    Example:
    --------
    >>> kmeans_anom = KMeansAnomaly(n_clusters=5, contamination=0.1)
    >>> kmeans_anom.fit(X)
    >>> predictions = kmeans_anom.predict(X)  # -1 for anomalies, 1 for normal
    """

    def __init__(self, n_clusters=5, contamination=0.1, random_state=None):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.centers = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X):
        """
        Fit K-Means for anomaly detection

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)

        # Simple K-Means implementation
        n_samples, n_features = X.shape

        # Initialize centers randomly
        idx = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        self.centers = X[idx].copy()

        # Iterate
        for _ in range(100):
            # Assign to nearest center
            distances = cdist(X, self.centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            # Update centers
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centers[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(self.centers, new_centers):
                break

            self.centers = new_centers

        return self

    def decision_function(self, X):
        """
        Calculate distance to nearest cluster center

        Returns:
        --------
        scores : array of shape (n_samples,)
            Distance to nearest center for each sample
        """
        X = np.asarray(X, dtype=np.float64)

        if self.centers is None:
            raise ValueError("Model not fitted yet")

        distances = cdist(X, self.centers, metric='euclidean')
        return np.min(distances, axis=1)

    def predict(self, X):
        """
        Predict anomalies

        Returns:
        --------
        predictions : array of shape (n_samples,)
            -1 for anomalies, 1 for normal points
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return np.where(scores > threshold, -1, 1)


# =============================================================================
# UTILITY FUNCTION
# =============================================================================

def detect_anomalies(X, method='isolation_forest', **kwargs):
    """
    Quick anomaly detection using specified method

    Parameters:
    -----------
    X : array-like
        Input data

    method : str, default='isolation_forest'
        Method to use: 'isolation_forest', 'lof', 'elliptic_envelope', 
                      'statistical', 'kmeans'

    **kwargs : dict
        Additional parameters for the method

    Returns:
    --------
    predictions : array
        -1 for anomalies, 1 for normal points
    """
    methods = {
        'isolation_forest': IsolationForest,
        'lof': LocalOutlierFactor,
        'elliptic_envelope': EllipticEnvelope,
        'statistical': StatisticalAnomaly,
        'kmeans': KMeansAnomaly,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}")

    detector = methods[method](**kwargs)
    detector.fit(X)
    return detector.predict(X)
