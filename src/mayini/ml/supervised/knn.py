"""K-Nearest Neighbors algorithm"""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin


class KNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')
    metric : str, default='euclidean'
        Distance metric to use
    
    Example
    -------
    >>> from mayini.ml import KNN
    >>> knn = KNN(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit the KNN model"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        predictions = []

        for x in X:
            # Calculate distances
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

            # Get k nearest neighbors
            k_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]

            # Vote
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)


class KNNClassifier(KNN):
    """Alias for KNN Classifier"""
    pass


class KNNRegressor(BaseEstimator, RegressorMixin):
    """
    K-Nearest Neighbors Regressor
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : str, default='uniform'
        Weight function
    
    Example
    -------
    >>> from mayini.ml import KNNRegressor
    >>> knn = KNNRegressor(n_neighbors=3)
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, n_neighbors=5, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit the KNN model"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        """Predict continuous values"""
        X = np.array(X)
        predictions = []

        for x in X:
            # Calculate distances
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

            # Get k nearest neighbors
            k_indices = np.argsort(distances)[: self.n_neighbors]
            k_nearest_values = self.y_train[k_indices]

            # Average
            predictions.append(np.mean(k_nearest_values))

        return np.array(predictions)
