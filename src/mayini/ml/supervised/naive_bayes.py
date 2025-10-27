"""
Naive Bayes classifiers implementation.
"""

import numpy as np
from ..base import BaseClassifier

class GaussianNB(BaseClassifier):
    """
    Gaussian Naive Bayes classifier.
    
    Implements the Gaussian Naive Bayes algorithm for classification.
    Assumes that features follow a normal (Gaussian) distribution.
    """
    
    def __init__(self, var_smoothing=1e-9):
        """
        Initialize Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        var_smoothing : float, default=1e-9
            Portion of the largest variance added to variances for stability
        """
        super().__init__()
        self.var_smoothing = var_smoothing
        self.class_prior_ = None
        self.theta_ = None  # Mean of each feature per class
        self.var_ = None    # Variance of each feature per class
    
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Initialize parameters
        n_features = X.shape[1]
        self.theta_ = np.zeros((self.n_classes_, n_features))
        self.var_ = np.zeros((self.n_classes_, n_features))
        self.class_prior_ = np.zeros(self.n_classes_)
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + self.var_smoothing
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test vectors
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test vectors
            
        Returns:
        --------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        self._check_is_fitted()
        X = np.asarray(X)
        
        # Calculate log probability for each class
        log_proba = []
        for idx in range(self.n_classes_):
            # Log prior
            log_prior = np.log(self.class_prior_[idx])
            
            # Log likelihood (Gaussian PDF)
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[idx, :])
                + ((X - self.theta_[idx, :]) ** 2) / self.var_[idx, :],
                axis=1
            )
            
            log_proba.append(log_prior + log_likelihood)
        
        log_proba = np.array(log_proba).T
        
        # Convert log probabilities to probabilities
        # Using log-sum-exp trick for numerical stability
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_proba_max)
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
