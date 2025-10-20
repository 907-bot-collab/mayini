"""K-Nearest Neighbors for classification and regression"""
import numpy as np
from ..base import BaseClassifier, BaseRegressor

class KNeighborsClassifier(BaseClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X, y):
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._euclidean_distance(self.X_train_, x)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train_[k_indices]
            predictions.append(np.bincount(k_labels.astype(int)).argmax())
        return np.array(predictions)

