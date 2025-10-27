"""Base classes for machine learning models"""
import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Base class for all estimators in mayini
    
    All estimators should inherit from this class and implement
    fit and predict methods.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model to training data
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Target values
            
        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        array-like
            Predictions
        """
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
    """
    Mixin class for all classifiers in mayini
    
    Provides common methods for classification tasks.
    """

    def score(self, X, y):
        """
        Calculate accuracy score
        
        Parameters
        ----------
        X : array-like
            Test data
        y : array-like
            True labels
            
        Returns
        -------
        float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def predict_proba(self, X):
        """
        Predict class probabilities (if applicable)
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        array-like
            Class probabilities
        """
        raise NotImplementedError("predict_proba not implemented for this classifier")


class RegressorMixin:
    """
    Mixin class for all regressors in mayini
    
    Provides common methods for regression tasks.
    """

    def score(self, X, y):
        """
        Calculate R² score
        
        Parameters
        ----------
        X : array-like
            Test data
        y : array-like
            True values
            
        Returns
        -------
        float
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class ClusterMixin:
    """
    Mixin class for all clustering algorithms in mayini
    """

    def fit_predict(self, X):
        """
        Fit and predict in one step
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        array-like
            Cluster labels
        """
        return self.fit(X).predict(X)


class TransformerMixin:
    """
    Mixin class for all transformers in mayini
    """

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        
        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target values
            
        Returns
        -------
        array-like
            Transformed data
        """
        return self.fit(X, y).transform(X)
