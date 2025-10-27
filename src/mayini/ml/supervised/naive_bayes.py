"""Naive Bayes algorithms"""
import numpy as np
from ..base import BaseEstimator, ClassifierMixin


class NaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Base Naive Bayes classifier
    
    Base class for Naive Bayes algorithms.
    """

    def __init__(self):
        self.classes_ = None
        self.class_prior_ = None

    def fit(self, X, y):
        """Fit Naive Bayes classifier"""
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Calculate class priors
        self.class_prior_ = np.zeros(n_classes)
        for idx, c in enumerate(self.classes_):
            self.class_prior_[idx] = np.mean(y == c)
        
        return self

    def predict(self, X):
        """Predict class labels"""
        raise NotImplementedError("Subclasses must implement predict method")


class GaussianNB(NaiveBayes):
    """
    Gaussian Naive Bayes classifier
    
    Assumes features follow a Gaussian distribution.
    
    Example
    -------
    >>> from mayini.ml import GaussianNB
    >>> nb = GaussianNB()
    >>> nb.fit(X_train, y_train)
    >>> predictions = nb.predict(X_test)
    """

    def __init__(self):
        super().__init__()
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes"""
        super().fit(X, y)
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        # Calculate mean and variance for each class and feature
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + 1e-9  # Add small value to avoid division by zero
        
        return self

    def _calculate_likelihood(self, X):
        """Calculate likelihood for each class"""
        likelihoods = []
        
        for idx, c in enumerate(self.classes_):
            # Calculate Gaussian probability density
            mean = self.theta_[idx]
            var = self.var_[idx]
            
            # Log probability to avoid numerical underflow
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            likelihoods.append(log_prob)
        
        return np.array(likelihoods).T

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        
        # Calculate log posterior
        log_likelihood = self._calculate_likelihood(X)
        log_prior = np.log(self.class_prior_)
        log_posterior = log_likelihood + log_prior
        
        # Return class with highest posterior
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        
        # Calculate log posterior
        log_likelihood = self._calculate_likelihood(X)
        log_prior = np.log(self.class_prior_)
        log_posterior = log_likelihood + log_prior
        
        # Convert to probabilities using softmax
        # Subtract max for numerical stability
        log_posterior = log_posterior - np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior)
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        
        return posterior


class MultinomialNB(NaiveBayes):
    """
    Multinomial Naive Bayes classifier
    
    Suitable for discrete features (e.g., word counts).
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace) smoothing parameter
    
    Example
    -------
    >>> from mayini.ml import MultinomialNB
    >>> nb = MultinomialNB(alpha=1.0)
    >>> nb.fit(X_train, y_train)
    >>> predictions = nb.predict(X_test)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Fit Multinomial Naive Bayes"""
        super().fit(X, y)
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        # Calculate feature probabilities for each class
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Count occurrences and apply smoothing
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
        
        return self

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        
        # Calculate log posterior
        log_prob = X @ self.feature_log_prob_.T
        log_prob += np.log(self.class_prior_)
        
        return self.classes_[np.argmax(log_prob, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        
        # Calculate log posterior
        log_prob = X @ self.feature_log_prob_.T
        log_prob += np.log(self.class_prior_)
        
        # Convert to probabilities using softmax
        log_prob = log_prob - np.max(log_prob, axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob = prob / np.sum(prob, axis=1, keepdims=True)
        
        return prob
