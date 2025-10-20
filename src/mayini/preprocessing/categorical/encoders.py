
"""Categorical encoders"""
import numpy as np
from ..base import BaseTransformer

class LabelEncoder(BaseTransformer):
    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        self.is_fitted_ = True
        return self

    def transform(self, y):
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted first")
        return np.array([self.class_to_index_[val] for val in y])

    def inverse_transform(self, y):
        return np.array([self.classes_[idx] for idx in y])


class OneHotEncoder(BaseTransformer):
    def __init__(self, sparse=False):
        super().__init__()
        self.sparse = sparse
        self.categories_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.categories_ = []
        for col in range(X.shape):
            self.categories_.append(np.unique(X[:, col]))

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Encoder must be fitted first")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_features = []

        for col in range(X.shape):
            categories = self.categories_[col]
            feature = X[:, col]

            # Create one-hot encoded columns
            for category in categories:
                encoded_features.append((feature == category).astype(int))

        return np.column_stack(encoded_features)
