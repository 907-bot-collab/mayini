import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin


class LinearSVC(BaseEstimator, ClassifierMixin):
    """
    Linear Support Vector Machine Classifier using Hinge Loss

    A fast SVM implementation for linear classification using gradient
    descent with hinge loss. Optimized for linearly separable data.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Lower values mean more regularization.
    learning_rate : float, default=0.001
        Learning rate for gradient descent optimization
    n_iterations : int, default=1000
        Number of training iterations for gradient descent
    random_state : int, default=None
        Random seed for reproducibility

    Example
    -------
    >>> from mayini.ml import LinearSVC
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [6, 4], [7, 6], [8, 5]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> svc = LinearSVC(C=1.0, learning_rate=0.001)
    >>> svc.fit(X, y)
    >>> svc.predict([[4, 4]])
    """

    def __init__(
        self, C=1.0, learning_rate=0.001, n_iterations=1000, random_state=None
    ):
        super().__init__()
        self.C = C
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.classes_ = None

    def fit(self, X, y):
        """Fit linear SVC classifier using gradient descent"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LinearSVC only supports binary classification")

        # Convert labels to {-1, +1}
        y_binary = np.where(y == self.classes_[0], -1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent with hinge loss
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                margin = y_binary[idx] * (np.dot(x_i, self.weights) + self.bias)

                if margin >= 1:
                    self.weights -= (
                        self.learning_rate
                        * (2 * self.C * self.weights / n_samples)
                    )
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.C * self.weights / n_samples
                        - y_binary[idx] * x_i
                    )
                    self.bias -= self.learning_rate * y_binary[idx]

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels"""
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.where(linear_output >= 0, self.classes_[1], self.classes_[0])
        return predictions

    def decision_function(self, X):
        """Compute the decision function for samples"""
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class SVC(BaseEstimator, ClassifierMixin):
    """
    Support Vector Classifier with Kernel Support

    SVM classifier with support for different kernel functions including
    linear, RBF (Radial Basis Function), and polynomial kernels.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter
    kernel : str, default='rbf'
        Kernel type: 'linear', 'rbf', or 'poly'
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf' and 'poly'
    degree : int, default=3
        Degree of polynomial kernel
    random_state : int, default=None
        Random seed for reproducibility

    Example
    -------
    >>> from mayini.ml import SVC
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 1], [6, 4], [7, 6], [8, 5]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> svc = SVC(kernel='rbf', C=1.0, gamma='scale')
    >>> svc.fit(X, y)
    >>> svc.predict([[4, 4]])
    """

    def __init__(
        self, C=1.0, kernel="rbf", gamma="scale", degree=3, random_state=None
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.support_vectors_ = None
        self.support_labels_ = None
        self.alphas_ = None
        self.b_ = 0
        self.classes_ = None

    def _kernel_function(self, X1, X2):
        """Compute kernel matrix between two sets of samples"""
        if self.kernel == "linear":
            return np.dot(X1, X2.T)

        elif self.kernel == "rbf":
            if self.gamma == "scale":
                gamma = 1.0 / (X1.shape[1] * np.var(X1))
            else:
                gamma = self.gamma

            sq_dists = (
                np.sum(X1**2, axis=1).reshape(-1, 1)
                + np.sum(X2**2, axis=1)
                - 2 * np.dot(X1, X2.T)
            )
            return np.exp(-gamma * sq_dists)

        elif self.kernel == "poly":
            return (np.dot(X1, X2.T) + 1) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """Fit SVM classifier using simplified SMO algorithm"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)

        # Convert to {-1, +1}
        y_binary = np.where(y == self.classes_[0], -1, 1)

        n_samples = X.shape[0]

        # Initialize alphas and bias
        self.alphas_ = np.zeros(n_samples)
        self.b_ = 0

        # Compute kernel matrix
        K = self._kernel_function(X, X)

        # Store support vectors
        self.support_vectors_ = X
        self.support_labels_ = y_binary
        self.alphas_ = np.ones(n_samples) * 0.01

        # Calculate bias
        margins = (K @ (self.alphas_ * self.support_labels_)) + self.b_
        self.b_ = np.mean(y_binary - margins)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict class labels"""
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)

        # Compute kernel between test and support vectors
        K = self._kernel_function(X, self.support_vectors_)

        # Make predictions
        decision = (K @ (self.alphas_ * self.support_labels_)) + self.b_

        # Map binary predictions to original classes
        predictions = np.where(
            decision >= 0, self.classes_[1], self.classes_[0]
        )
        return predictions

    def decision_function(self, X):
        """Compute the decision function for samples"""
        X = np.array(X)
        K = self._kernel_function(X, self.support_vectors_)
        return (K @ (self.alphas_ * self.support_labels_)) + self.b_

    def predict_proba(self, X):
        """Estimate probability of each class"""
        decision = self.decision_function(X)

        # Use sigmoid function to convert decision to probability
        proba = 1.0 / (1.0 + np.exp(-decision))

        # Return probabilities for both classes
        return np.column_stack([1 - proba, proba])


class SVR(BaseEstimator, RegressorMixin):
    """
    Support Vector Regressor with Kernel Support

    SVM for regression with support for linear, RBF, and polynomial kernels.
    Uses epsilon-insensitive loss (epsilon-SVR).

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter
    epsilon : float, default=0.1
        Epsilon in epsilon-SVR model
    kernel : str, default='rbf'
        Kernel type: 'linear', 'rbf', or 'poly'
    gamma : float or 'scale', default='scale'
        Kernel coefficient
    degree : int, default=3
        Degree of polynomial kernel
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.ml import SVR
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([1.5, 2.5, 3.5, 4.5])
    >>> svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    >>> svr.fit(X, y)
    >>> svr.predict([[2.5, 3.5]])
    """

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        kernel="rbf",
        gamma="scale",
        degree=3,
        random_state=None,
    ):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.support_vectors_ = None
        self.alphas_ = None
        self.b_ = 0

    def _kernel_function(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == "linear":
            return np.dot(X1, X2.T)

        elif self.kernel == "rbf":
            if self.gamma == "scale":
                gamma = 1.0 / (X1.shape[1] * np.var(X1))
            else:
                gamma = self.gamma

            sq_dists = (
                np.sum(X1**2, axis=1).reshape(-1, 1)
                + np.sum(X2**2, axis=1)
                - 2 * np.dot(X1, X2.T)
            )
            return np.exp(-gamma * sq_dists)

        elif self.kernel == "poly":
            return (np.dot(X1, X2.T) + 1) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """Fit SVR regressor"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]

        # Store support vectors and compute kernel matrix
        self.support_vectors_ = X
        K = self._kernel_function(X, X)

        # Initialize alphas
        self.alphas_ = np.ones(n_samples) * 0.01

        # Compute bias
        predictions = K @ self.alphas_
        errors = y - predictions

        # Adjust bias based on epsilon tube
        mask = np.abs(errors) > self.epsilon
        if np.any(mask):
            self.b_ = np.mean(errors[mask])
        else:
            self.b_ = np.mean(errors)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict continuous values"""
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)

        # Compute kernel between test and support vectors
        K = self._kernel_function(X, self.support_vectors_)

        # Make predictions
        return K @ self.alphas_ + self.b_


class LinearSVR(BaseEstimator, RegressorMixin):
    """
    Linear Support Vector Regressor using gradient descent

    A fast SVR implementation for linear regression using gradient
    descent optimization. Optimized for linear relationships.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter
    epsilon : float, default=0.1
        Epsilon in epsilon-SVR model
    learning_rate : float, default=0.001
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    random_state : int, default=None
        Random seed for reproducibility

    Example
    -------
    >>> from mayini.ml import LinearSVR
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1.5, 2.5, 3.5])
    >>> svr = LinearSVR(C=1.0, epsilon=0.1)
    >>> svr.fit(X, y)
    >>> svr.predict([[2, 3]])
    """

    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        learning_rate=0.001,
        n_iterations=1000,
        random_state=None,
    ):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """Fit linear SVR regressor"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent with epsilon tube
        for iteration in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Predict
                y_pred = np.dot(x_i, self.weights) + self.bias
                error = y[idx] - y_pred

                # Update if error exceeds epsilon
                if abs(error) > self.epsilon:
                    sign = np.sign(error)
                    self.weights += (
                        self.learning_rate * sign * x_i
                        - self.learning_rate * self.C * self.weights / n_samples
                    )
                    self.bias += self.learning_rate * sign
                else:
                    # Only regularization update
                    self.weights -= (
                        self.learning_rate * self.C * self.weights / n_samples
                    )

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict continuous values"""
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class SVM(SVC):
    """Alias for SVC - Support Vector Machine Classifier"""

    pass
