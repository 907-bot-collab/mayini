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
    """Gaussian Naive Bayes classifier"""

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
        
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + 1e-9
        
        return self

    def _calculate_likelihood(self, X):
        """Calculate likelihood for each class"""
        likelihoods = []
        
        for idx, c in enumerate(self.classes_):
            mean = self.theta_[idx]
            var = self.var_[idx]
            
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            likelihoods.append(log_prob)
        
        return np.array(likelihoods).T

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        log_likelihood = self._calculate_likelihood(X)
        log_prior = np.log(self.class_prior_)
        log_posterior = log_likelihood + log_prior
        return self.classes_[np.argmax(log_posterior, axis=1)]


class MultinomialNB(NaiveBayes):
    """Multinomial Naive Bayes classifier"""

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
        
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
        
        return self

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        log_prob = X @ self.feature_log_prob_.T
        log_prob += np.log(self.class_prior_)
        return self.classes_[np.argmax(log_prob, axis=1)]


class BernoulliNB(NaiveBayes):
    """
    Bernoulli Naive Bayes classifier
    
    Suitable for binary features (0 or 1).
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Fit Bernoulli Naive Bayes"""
        super().fit(X, y)
        X = np.array(X)
        y = np.array(y)
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = len(X_c) + 2 * self.alpha
            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
        
        return self

    def predict(self, X):
        """Predict class labels"""
        X = np.array(X)
        log_prob = X @ self.feature_log_prob_.T
        log_prob += np.log(self.class_prior_)
        return self.classes_[np.argmax(log_prob, axis=1)]
