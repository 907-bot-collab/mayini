
import numpy as np
from ..base import BaseClassifier


class GaussianNB(BaseClassifier):
    """
    Gaussian Naive Bayes classifier

    Assumes features follow normal distribution

    Example:
    --------
    >>> from mayini.ml import GaussianNB
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([0, 0, 1, 1])
    >>> nb = GaussianNB()
    >>> nb.fit(X, y)
    >>> nb.predict([[2.5, 3.5]])
    """

    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # mean
        self.sigma_ = None  # variance

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes"""
        X, y = self._validate_input(X, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.sigma_[idx, :] = X_c.var(axis=0) + 1e-9  # Add epsilon
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]

        self.is_fitted_ = True
        return self

    def _gaussian_pdf(self, x, mean, var):
        """Gaussian probability density function"""
        return np.exp(-0.5 * ((x - mean) ** 2 / var)) / np.sqrt(2 * np.pi * var)

    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        posteriors = []
        for i, c in enumerate(self.classes_):
            prior = np.log(self.class_prior_[i])
            likelihood = np.sum(np.log(self._gaussian_pdf(
                X, self.theta_[i, :], self.sigma_[i, :]
            )), axis=1)
            posteriors.append(prior + likelihood)

        posteriors = np.array(posteriors).T
        # Convert log probabilities to probabilities
        posteriors = np.exp(posteriors - np.max(posteriors, axis=1, keepdims=True))
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        return posteriors

    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class MultinomialNB(BaseClassifier):
    """
    Multinomial Naive Bayes classifier

    Suitable for discrete features (e.g., word counts)

    Parameters:
    -----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter

    Example:
    --------
    >>> from mayini.ml import MultinomialNB
    >>> X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
    >>> y = np.array([0, 1, 0])
    >>> nb = MultinomialNB(alpha=1.0)
    >>> nb.fit(X, y)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Fit Multinomial Naive Bayes"""
        X, y = self._validate_input(X, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Count features
            feature_count = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(feature_count)

            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])

        self.is_fitted_ = True
        return self

    def predict_log_proba(self, X):
        """Predict log probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        log_proba = X @ self.feature_log_prob_.T + self.class_log_prior_
        return log_proba

    def predict_proba(self, X):
        """Predict class probabilities"""
        log_proba = self.predict_log_proba(X)
        # Normalize
        log_proba = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict class labels"""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class BernoulliNB(BaseClassifier):
    """
    Bernoulli Naive Bayes classifier

    Suitable for binary/boolean features

    Parameters:
    -----------
    alpha : float, default=1.0
        Additive smoothing parameter
    binarize : float, default=0.0
        Threshold for binarizing features

    Example:
    --------
    >>> from mayini.ml import BernoulliNB
    >>> X = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]])
    >>> y = np.array([0, 1, 0])
    >>> nb = BernoulliNB(alpha=1.0)
    >>> nb.fit(X, y)
    """

    def __init__(self, alpha=1.0, binarize=0.0):
        super().__init__()
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """Fit Bernoulli Naive Bayes"""
        X, y = self._validate_input(X, y)

        # Binarize features
        if self.binarize is not None:
            X = (X > self.binarize).astype(np.float64)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Calculate P(x_i = 1 | y = c)
            feature_count = np.sum(X_c, axis=0) + self.alpha
            total_count = X_c.shape[0] + 2 * self.alpha

            self.feature_log_prob_[idx, :] = np.log(feature_count / total_count)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])

        self.is_fitted_ = True
        return self

    def predict_log_proba(self, X):
        """Predict log probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if self.binarize is not None:
            X = (X > self.binarize).astype(np.float64)

        log_proba = []
        for i in range(len(self.classes_)):
            # P(x|y) = Product of P(x_i|y) for x_i=1 and P(1-x_i|y) for x_i=0
            log_prob = np.sum(X * self.feature_log_prob_[i, :], axis=1)
            log_prob += np.sum((1 - X) * np.log(1 - np.exp(self.feature_log_prob_[i, :])), axis=1)
            log_prob += self.class_log_prior_[i]
            log_proba.append(log_prob)

        return np.array(log_proba).T

    def predict_proba(self, X):
        """Predict class probabilities"""
        log_proba = self.predict_log_proba(X)
        log_proba = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict(self, X):
        """Predict class labels"""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
