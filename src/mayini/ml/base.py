"""
Base classes for ML algorithms in Mayini framework.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Base class for all estimators in mayini
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the model to training data"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass

    def get_params(self):
        """Get parameters of this estimator"""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not key.endswith("_")
        }

    def set_params(self, **params):
        """Set parameters of this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _validate_input(self, X, y=None):
        """Validate input X and y"""
        X = np.array(X)
        if y is not None:
            y = np.array(y)
        return X, y

    def _check_is_fitted(self):
        """Check if the estimator has been fitted."""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise RuntimeError(
                "This {} instance is not fitted yet. Call 'fit' first."
                .format(self.__class__.__name__)
            )


class ClassifierMixin:
    """Mixin class for all classifiers"""

    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class RegressorMixin:
    """Mixin class for all regressors"""

    def score(self, X, y):
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # Avoid division by zero
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)


class ClusterMixin:
    """Mixin class for all clustering algorithms"""

    def fit_predict(self, X):
        """Fit and predict in one step"""
        self.fit(X)
        return getattr(self, "labels_", self.predict(X))


class TransformerMixin:
    """Mixin class for all transformers"""

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """Base class for all classifiers"""
    pass


class BaseRegressor(BaseEstimator, RegressorMixin):
    """Base class for all regressors"""
    pass
