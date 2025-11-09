"""
Base classes for ML algorithms in Mayini framework.
"""

import numpy as np

class BaseClassifier:
    """
    Base class for all classifiers in mayini.ml
    
    All classifiers should inherit from this class and implement
    the fit(), predict(), and predict_proba() methods.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X, y):
        """
        Fit the classifier to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict()")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        raise NotImplementedError("predict_proba not implemented")
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
            
        Returns:
        --------
        score : float
            Mean accuracy
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def _check_is_fitted(self):
        """Check if the estimator has been fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                "This {} instance is not fitted yet. Call 'fit' first."
                .format(self.__class__.__name__)
            )


class BaseRegressor:
    """
    Base class for all regressors in mayini.ml
    """
    
    def __init__(self):
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the regressor to training data."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """Predict values for samples in X."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        raise NotImplementedError("Subclasses must implement predict()")
    
    def score(self, X, y):
        """Return R^2 score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
"""Base classes for all machine learning models"""
import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Base class for all estimators in mayini
    
    All estimators should inherit from this class.
    """

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
            if not key.startswith("_")
        }

    def set_params(self, **params):
        """Set parameters of this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


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
        return 1 - (ss_res / ss_tot)


class ClusterMixin:
    """Mixin class for all clustering algorithms"""

    def fit_predict(self, X):
        """Fit and predict in one step"""
        return self.fit(X).predict(X)


class TransformerMixin:
    """Mixin class for all transformers"""

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
