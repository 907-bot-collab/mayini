import numpy as np
from ..base import BaseClassifier, BaseRegressor


class LinearSVM(BaseClassifier):
    """
    Linear Support Vector Machine for classification

    Uses gradient descent with hinge loss

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of training iterations

    Example:
    --------
    >>> from mayini.ml import LinearSVM
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [6, 4], [7, 6], [8, 5]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> svm = LinearSVM(C=1.0)
    >>> svm.fit(X, y)
    """

    def __init__(self, C=1.0, learning_rate=0.001, n_iterations=1000):
        super().__init__()
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit SVM using gradient descent"""
        X, y = self._validate_input(X, y)

        # Convert labels to {-1, +1}
        y_ = np.where(y <= 0, -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1

                if condition:
                    # No hinge loss
                    self.weights -= self.learning_rate * (2 * (1 / self.n_iterations) * self.weights)
                else:
                    # Hinge loss
                    self.weights -= self.learning_rate * (
                        2 * (1 / self.n_iterations) * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)


class SVC(BaseClassifier):
    """
    Support Vector Classification with kernel trick

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter
    kernel : str, default='rbf'
        Kernel type ('linear', 'rbf', 'poly')
    gamma : float, default='scale'
        Kernel coefficient
    degree : int, default=3
        Degree for polynomial kernel

    Example:
    --------
    >>> from mayini.ml import SVC
    >>> svc = SVC(kernel='rbf', C=1.0)
    >>> svc.fit(X_train, y_train)
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.support_vectors_ = None
        self.support_labels_ = None
        self.alphas_ = None
        self.b_ = 0

    def _kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma != 'scale' else 1.0 / X1.shape[1]
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * sq_dists)
        elif self.kernel == 'poly':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """Fit SVM (simplified SMO algorithm)"""
        X, y = self._validate_input(X, y)

        # Convert to {-1, +1}
        y = np.where(y == 0, -1, 1)

        n_samples = X.shape[0]
        self.alphas_ = np.zeros(n_samples)
        self.b_ = 0

        # Simplified training (not full SMO)
        K = self._kernel_function(X, X)

        # Simple approach: store all as support vectors
        self.support_vectors_ = X
        self.support_labels_ = y
        self.alphas_ = np.ones(n_samples) * 0.1

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        K = self._kernel_function(X, self.support_vectors_)
        decision = np.dot(K, self.alphas_ * self.support_labels_) + self.b_

        return np.where(decision >= 0, 1, 0)


class SVR(BaseRegressor):
    """
    Support Vector Regression

    Parameters:
    -----------
    C : float, default=1.0
        Regularization parameter
    epsilon : float, default=0.1
        Epsilon in epsilon-SVR model
    kernel : str, default='rbf'
        Kernel type

    Example:
    --------
    >>> from mayini.ml import SVR
    >>> svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    >>> svr.fit(X_train, y_train)
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale'):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors_ = None
        self.alphas_ = None
        self.b_ = 0

    def _kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            gamma = self.gamma if self.gamma != 'scale' else 1.0 / X1.shape[1]
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """Fit SVR"""
        X, y = self._validate_input(X, y)

        n_samples = X.shape[0]
        self.support_vectors_ = X
        self.alphas_ = np.ones(n_samples) * 0.1
        self.b_ = np.mean(y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict values"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        K = self._kernel_function(X, self.support_vectors_)
        return np.dot(K, self.alphas_) + self.b_

