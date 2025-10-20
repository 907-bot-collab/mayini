"""Base transformer classes"""
import numpy as np

class BaseTransformer:
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

