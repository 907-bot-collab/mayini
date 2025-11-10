import numpy as np
from scipy import linalg
from ..base import BaseRegressor, BaseClassifier, BaseEstimator, ClassifierMixin, RegressorMixin


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Classifier
    
    Binary classification using logistic (sigmoid) function with
    gradient descent optimization.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    random_state : int, default=None
        Random seed
    
    Example
    -------
    >>> from mayini.ml import LogisticRegression
    >>> lr = LogisticRegression(max_iter=100)
    >>> lr.fit(X, y)
    >>> lr.predict(X)
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes_ = None
    
    def _validate_input(self, X, y=None):
        """Validate and convert input data to numpy arrays"""
        X = np.array(X)
        
        if y is not None:
            y = np.array(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {y.shape[0]}"
                )
            return X, y
        
        return X, None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Fit logistic regression model"""
        X, y = self._validate_input(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression only supports binary classification")
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to binary {0, 1}
        y_binary = np.where(y == self.classes_[0], 0, 1)
        
        # Gradient descent
        for iteration in range(self.max_iter):
            # Linear model
            z = np.dot(X, self.weights) + self.bias
            
            # Predictions (sigmoid)
            y_pred = self._sigmoid(z)
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_binary))
            db = (1 / n_samples) * np.sum(y_pred - y_binary)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = self._validate_input(X)
        
        z = np.dot(X, self.weights) + self.bias
        y_pred_prob = self._sigmoid(z)
        
        return np.where(y_pred_prob >= 0.5, self.classes_[1], self.classes_[0])
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = self._validate_input(X)
        
        z = np.dot(X, self.weights) + self.bias
        y_pred_prob = self._sigmoid(z)
        
        return np.column_stack([1 - y_pred_prob, y_pred_prob])



class Ridge(BaseRegressor):
    """
    Ridge Regression (L2 regularization)

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    fit_intercept : bool, default=True
        Whether to calculate the intercept

    Example
    -------
    >>> from mayini.ml import Ridge
    >>> ridge = Ridge(alpha=1.0)
    >>> ridge.fit(X_train, y_train)
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
        self.coef_ = linalg.solve(A, b, assume_a="pos")

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

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for optimization

    Example
    -------
    >>> from mayini.ml import Lasso
    >>> lasso = Lasso(alpha=0.1)
    >>> lasso.fit(X_train, y_train)
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
                self.coef_[j] = self._soft_threshold(
                    rho_j / n_samples, self.alpha
                ) / (np.sum(X[:, j] ** 2) / n_samples)

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

class ElasticNet(BaseEstimator, RegressorMixin):
    """
    Elastic Net Regression (L1 + L2 regularization)
    
    Combines Ridge (L2) and Lasso (L1) penalties.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, learning_rate=0.001):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            errors = y_pred - y
            
            # Gradient (L1 + L2 combined)
            l2_penalty = 2 * self.alpha * (1 - self.l1_ratio) * self.weights
            l1_penalty = self.alpha * self.l1_ratio * np.sign(self.weights)
            
            # Update weights
            self.weights -= self.learning_rate * (
                2 * np.dot(X.T, errors) / n_samples + l2_penalty + l1_penalty
            )
            self.bias -= self.learning_rate * 2 * np.mean(errors)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias



class LogisticRegression(BaseClassifier):
    """
    Logistic Regression for binary and multi-class classification

    Parameters
    ----------
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

    Example
    -------
    >>> from mayini.ml import LogisticRegression
    >>> lr = LogisticRegression()
    >>> lr.fit(X_train, y_train)
    >>> predictions = lr.predict(X_test)
    """

    def __init__(
        self, learning_rate=0.01, max_iter=1000, tol=1e-4, penalty="l2", C=1.0
    ):
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
                if self.penalty == "l2":
                    grad_w += (1 / self.C) * self.coef_
                elif self.penalty == "l1":
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

                    if self.penalty == "l2":
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
