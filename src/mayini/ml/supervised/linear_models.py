import numpy as np
from scipy import linalg
from ..base import BaseRegressor, BaseClassifier, BaseEstimator, ClassifierMixin, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    """
    Ordinary Least Squares Linear Regression
    
    Simple linear regression using the normal equation or gradient descent.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept
    
    Example
    -------
    >>> from mayini.ml import LinearRegression
    >>> lr = LinearRegression()
    >>> lr.fit(X, y)
    >>> lr.predict(X)
    """
    
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit linear regression model"""
        X = np.array(X)
        y = np.array(y)
        
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: (X^T X)^-1 X^T y
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict using the linear model"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        return np.dot(X, self.coef_) + self.intercept_

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



class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression Classifier
    
    Binary classification using logistic (sigmoid) function with gradient descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes
    ----------
    coef_ : array-like
        Learned coefficients
    intercept_ : float
        Learned intercept
    classes_ : array-like
        Unique class labels
    
    Example
    -------
    >>> from mayini.ml import LogisticRegression
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
    >>> y = np.array([0, 0, 1, 1])
    >>> lr = LogisticRegression(learning_rate=0.01, max_iter=100)
    >>> lr.fit(X, y)
    >>> lr.predict([[5, 5]])
    array([1])
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (binary)
        
        Returns
        -------
        self : LogisticRegression
            Fitted classifier
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        # Validate dimensions
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}"
            )
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Get unique classes
        self.classes_ = np.unique(y)
        
        # Only support binary classification
        if len(self.classes_) != 2:
            raise ValueError(
                f"LogisticRegression only supports binary classification. "
                f"Found {len(self.classes_)} classes: {self.classes_}"
            )
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Convert labels to binary {0, 1}
        y_binary = np.where(y == self.classes_[0], 0, 1).astype(np.float64)
        
        # Gradient descent training loop
        for iteration in range(self.max_iter):
            # Compute linear combination
            z = np.dot(X, self.coef_) + self.intercept_
            
            # Sigmoid activation
            y_pred = self._sigmoid(z)
            
            # Compute gradients
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y_binary))
            db = (1.0 / n_samples) * np.sum(y_pred - y_binary)
            
            # Update parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        # Check if model is fitted
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Compute predictions
        z = np.dot(X, self.coef_) + self.intercept_
        y_pred_prob = self._sigmoid(z)
        
        # Convert probabilities to class labels
        y_pred = np.where(y_pred_prob >= 0.5, self.classes_[1], self.classes_[0])
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            Class probabilities for both classes
        """
        # Check if model is fitted
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Compute predictions
        z = np.dot(X, self.coef_) + self.intercept_
        y_pred_prob = self._sigmoid(z)
        
        # Return probabilities for both classes
        proba = np.column_stack([1 - y_pred_prob, y_pred_prob])
        
        return proba
    
    def decision_function(self, X):
        """
        Compute the decision function of X
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values
        """
        # Check if model is fitted
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Compute decision function (linear combination)
        return np.dot(X, self.coef_) + self.intercept_
    
    @staticmethod
    def _sigmoid(z):
        """
        Sigmoid activation function
        
        Parameters
        ----------
        z : array-like
            Input values
        
        Returns
        -------
        sigmoid : array-like
            Sigmoid of input (clipped for numerical stability)
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
        
        Returns
        -------
        score : float
            Mean accuracy (0 to 1)
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
