import numpy as np
from scipy import linalg
from ..base import BaseRegressor, BaseClassifier


class LinearRegression(BaseRegressor):
    """
    Ordinary Least Squares Linear Regression

    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    normalize : bool, default=False
        Whether to normalize features before regression
    """

    def __init__(self, fit_intercept=True, normalize=False):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """Fit linear model"""
        X, y = self._validate_input(X, y)

        if self.normalize:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            X = (X - self.mean_) / (self.std_ + 1e-8)

        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        # Solve normal equations: (X^T X) beta = X^T y
        self.coef_ = linalg.lstsq(X, y)[0]

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using the linear model"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if self.normalize:
            X = (X - self.mean_) / (self.std_ + 1e-8)

        return X @ self.coef_ + self.intercept_


class Ridge(BaseRegressor):
    """
    Ridge Regression (L2 regularization)

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """Fit Ridge regression model"""
        X, y = self._validate_input(X, y)

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
            self.X_mean_ = X_mean
            self.y_mean_ = y_mean

        # Solve: (X^T X + alpha * I) beta = X^T y
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        self.coef_ = linalg.solve(A, b, assume_a='pos')

        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coef_
        else:
            self.intercept_ = 0.0

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using Ridge model"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return X @ self.coef_ + self.intercept_


class Lasso(BaseRegressor):
    """
    Lasso Regression (L1 regularization)
    Uses coordinate descent algorithm

    Parameters:
    -----------
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for optimization
    """

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, lambda_):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def fit(self, X, y):
        """Fit Lasso model using coordinate descent"""
        X, y = self._validate_input(X, y)

        n_samples, n_features = X.shape

        # Center data
        X_mean = np.mean(X, axis=0)
        y_mean = np.mean(y)
        X = X - X_mean
        y = y - y_mean

        # Initialize coefficients
        self.coef_ = np.zeros(n_features)

        # Coordinate descent
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y - X @ self.coef_ + X[:, j] * self.coef_[j]

                # Update j-th coefficient
                rho_j = X[:, j] @ residual
                self.coef_[j] = self._soft_threshold(rho_j / n_samples, self.alpha) / (np.sum(X[:, j]**2) / n_samples)

            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        self.intercept_ = y_mean - X_mean @ self.coef_
        self.X_mean_ = X_mean
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using Lasso model"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return X @ self.coef_ + self.intercept_


class LogisticRegression(BaseClassifier):
    """
    Logistic Regression for binary and multi-class classification

    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for stopping criterion
    penalty : str, default='l2'
        Regularization penalty ('l1', 'l2', or None)
    C : float, default=1.0
        Inverse of regularization strength
    """

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, penalty='l2', C=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _softmax(self, z):
        """Softmax activation for multi-class"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """Fit logistic regression model"""
        X, y = self._validate_input(X, y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # Binary classification
        if n_classes == 2:
            y_binary = (y == self.classes_[1]).astype(int)
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0

            for iteration in range(self.max_iter):
                # Forward pass
                z = X @ self.coef_ + self.intercept_
                predictions = self._sigmoid(z)

                # Compute gradients
                error = predictions - y_binary
                grad_w = (X.T @ error) / n_samples
                grad_b = np.mean(error)

                # Add regularization
                if self.penalty == 'l2':
                    grad_w += (1 / self.C) * self.coef_
                elif self.penalty == 'l1':
                    grad_w += (1 / self.C) * np.sign(self.coef_)

                # Update parameters
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b

                # Check convergence
                if np.linalg.norm(grad_w) < self.tol:
                    break

        # Multi-class classification (one-vs-rest)
        else:
            self.coef_ = np.zeros((n_classes, n_features))
            self.intercept_ = np.zeros(n_classes)

            for idx, class_label in enumerate(self.classes_):
                y_binary = (y == class_label).astype(int)

                for iteration in range(self.max_iter):
                    z = X @ self.coef_[idx] + self.intercept_[idx]
                    predictions = self._sigmoid(z)

                    error = predictions - y_binary
                    grad_w = (X.T @ error) / n_samples
                    grad_b = np.mean(error)

                    if self.penalty == 'l2':
                        grad_w += (1 / self.C) * self.coef_[idx]

                    self.coef_[idx] -= self.learning_rate * grad_w
                    self.intercept_[idx] -= self.learning_rate * grad_b

                    if np.linalg.norm(grad_w) < self.tol:
                        break

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if len(self.classes_) == 2:
            proba_1 = self._sigmoid(X @ self.coef_ + self.intercept_)
            return np.column_stack([1 - proba_1, proba_1])
        else:
            scores = X @ self.coef_.T + self.intercept_
            return self._softmax(scores)

    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
