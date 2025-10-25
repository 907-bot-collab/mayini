import numpy as np
from ..base import BaseClassifier, BaseRegressor


class KNeighborsClassifier(BaseClassifier):
    """
    K-Nearest Neighbors Classifier

    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to use
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan')
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')

    Example:
    --------
    >>> from mayini.ml import KNeighborsClassifier
    >>> knn = KNeighborsClassifier(k=5)
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        super().__init__()
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier"""
        X, y = self._validate_input(X, y)
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def _compute_distances(self, X):
        """Compute distances between X and training data"""
        if self.metric == 'euclidean':
            # Euclidean distance
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.X_train) ** 2, axis=2))
        elif self.metric == 'manhattan':
            # Manhattan distance
            distances = np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return distances

    def predict(self, X):
        """Predict class labels"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        # Compute distances
        distances = self._compute_distances(X)

        # Find k nearest neighbors
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_indices]

        # Predict based on majority vote
        predictions = []
        for i, neighbors in enumerate(k_nearest_labels):
            if self.weights == 'uniform':
                # Simple majority vote
                unique, counts = np.unique(neighbors, return_counts=True)
                predictions.append(unique[counts.argmax()])
            elif self.weights == 'distance':
                # Distance-weighted vote
                neighbor_distances = distances[i, k_indices[i]]
                # Avoid division by zero
                weights = 1 / (neighbor_distances + 1e-10)

                weighted_votes = {}
                for label, weight in zip(neighbors, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight

                predictions.append(max(weighted_votes, key=weighted_votes.get))

        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        distances = self._compute_distances(X)
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_indices]

        probas = []
        for i, neighbors in enumerate(k_nearest_labels):
            class_probas = np.zeros(len(self.classes_))

            if self.weights == 'uniform':
                for label in neighbors:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probas[class_idx] += 1
                class_probas /= self.k
            elif self.weights == 'distance':
                neighbor_distances = distances[i, k_indices[i]]
                weights = 1 / (neighbor_distances + 1e-10)
                total_weight = np.sum(weights)

                for label, weight in zip(neighbors, weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probas[class_idx] += weight
                class_probas /= total_weight

            probas.append(class_probas)

        return np.array(probas)


class KNeighborsRegressor(BaseRegressor):
    """
    K-Nearest Neighbors Regressor

    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to use
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan')
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')

    Example:
    --------
    >>> from mayini.ml import KNeighborsRegressor
    >>> knn = KNeighborsRegressor(k=5)
    >>> knn.fit(X_train, y_train)
    >>> predictions = knn.predict(X_test)
    """

    def __init__(self, k=5, metric='euclidean', weights='uniform'):
        super().__init__()
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Fit the k-nearest neighbors regressor"""
        X, y = self._validate_input(X, y)
        self.X_train = X
        self.y_train = y
        self.is_fitted_ = True
        return self

    def _compute_distances(self, X):
        """Compute distances between X and training data"""
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.X_train) ** 2, axis=2))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        return distances

    def predict(self, X):
        """Predict continuous values"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        # Compute distances
        distances = self._compute_distances(X)

        # Find k nearest neighbors
        k_indices = np.argsort(distances, axis=1)[:, :self.k]

        # Get values of k nearest neighbors
        k_nearest_values = self.y_train[k_indices]

        # Predict based on average
        predictions = []
        for i, neighbors in enumerate(k_nearest_values):
            if self.weights == 'uniform':
                # Simple average
                predictions.append(np.mean(neighbors))
            elif self.weights == 'distance':
                # Distance-weighted average
                neighbor_distances = distances[i, k_indices[i]]
                weights = 1 / (neighbor_distances + 1e-10)
                weighted_sum = np.sum(neighbors * weights)
                predictions.append(weighted_sum / np.sum(weights))

        return np.array(predictions)
