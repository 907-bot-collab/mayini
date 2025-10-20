import numpy as np
from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """Base class for all estimators in mayini.ml"""
    
    def __init__(self):
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model to training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values
            
        Returns:
        --------
        self : object
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predictions
        """
        pass
    
    def _check_is_fitted(self):
        """Check if the model has been fitted"""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions. "
                "Call fit() first."
            )
    
    def _validate_input(self, X, y=None):
        """
        Validate input data
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like, optional
            Target values
            
        Returns:
        --------
        X : np.ndarray
            Validated features
        y : np.ndarray or None
            Validated targets
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y is not None:
            y = np.asarray(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {y.shape[0]}"
                )
        
        return X, y
    
    def get_params(self):
        """Get parameters for this estimator"""
        return {key: val for key, val in self.__dict__.items() 
                if not key.endswith('_')}
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BaseClassifier(BaseEstimator):
    """Base class for all classifiers"""
    
    def score(self, X, y):
        """
        Return accuracy score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True labels
            
        Returns:
        --------
        score : float
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def predict_proba(self, X):
        """
        Predict class probabilities (optional, override in subclasses)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba"
        )


class BaseRegressor(BaseEstimator):
    """Base class for all regressors"""
    
    def score(self, X, y):
        """
        Return R² score
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        y : array-like, shape (n_samples,)
            True values
            
        Returns:
        --------
        score : float
            R² score
        """
        predictions = self.predict(X)
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / (ss_tot + 1e-10))


class BaseCluster(BaseEstimator):
    """Base class for clustering algorithms"""
    
    @abstractmethod
    def fit_predict(self, X):
        """
        Fit the model and return cluster labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Cluster labels
        """
        pass
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        labels : array-like, shape (n_samples,)
            Cluster labels
        """
        self._check_is_fitted()
        # Default implementation - override in subclasses if needed
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict"
        )


class BaseDecomposition(BaseEstimator):
    """Base class for dimensionality reduction"""
    
    @abstractmethod
    def transform(self, X):
        """
        Transform data to reduced dimensions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Ignored
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X):
        """Alias for transform (for compatibility)"""
        return self.transform(X)
