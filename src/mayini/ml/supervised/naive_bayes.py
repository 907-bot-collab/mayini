import numpy as np
from collections import defaultdict

class GaussianNB:
    """Gaussian Naive Bayes classifier."""
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
        return self

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                numerator = np.exp(- (x - self.mean[c]) ** 2 / (2 * self.var[c]))
                denominator = np.sqrt(2 * np.pi * self.var[c])
                posterior = np.sum(np.log(numerator / denominator)) + prior
                posteriors[c] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class MultinomialNB:
    """Multinomial Naive Bayes classifier."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.feature_log_prob = {}
        self.class_log_prior = {}
        self.n_features = None

    def fit(self, X, y):
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        self.n_features = n_features
        for c in self.classes:
            X_c = X[y == c]
            feature_count = np.sum(X_c, axis=0)
            self.feature_log_prob[c] = np.log(
                (feature_count + self.alpha) / (np.sum(feature_count) + self.alpha * n_features)
            )
            self.class_log_prior[c] = np.log(X_c.shape[0] / n_samples)
        return self

    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                posterior = self.class_log_prior[c] + np.sum(x * self.feature_log_prob[c])
                posteriors[c] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

class BernoulliNB(NaiveBayes):
    """
    Bernoulli Naive Bayes classifier
    
    Suitable for binary features (0 or 1).
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace) smoothing parameter
    
    Example
    -------
    >>> from mayini.ml import BernoulliNB
    >>> nb = BernoulliNB(alpha=1.0)
    >>> nb.fit(X_train, y_train)
    >>> predictions = nb.predict(X_test)
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
        
        # Calculate feature probabilities for each class
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Count occurrences (for binary features)
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = len(X_c) + 2 * self.alpha
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

