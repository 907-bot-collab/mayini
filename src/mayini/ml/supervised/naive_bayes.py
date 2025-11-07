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
