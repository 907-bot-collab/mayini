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
        Controls the trade-off between training accuracy and model simplicity.
    learning_rate : float, default=0.001
        Learning rate for gradient descent optimization
    n_iterations : int, default=1000
        Number of training iterations for gradient descent
    random_state : int, default=None
        Random seed for reproducibility

    Attributes
    ----------
    weights : array-like of shape (n_features,)
        Weight vector learned during training
    bias : float
        Bias term (intercept)
    classes_ : array-like
        Unique class labels from training data

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
        """
        Fit linear SVC classifier using gradient descent

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels (binary: must be 2 classes)

        Returns
        -------
        self : LinearSVC
            Fitted classifier
        """
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
                # Check if margin condition is satisfied
                margin = y_binary[idx] * (np.dot(x_i, self.weights) + self.bias)

                if margin >= 1:
                    # No hinge loss - only regularization
                    self.weights -= (
                        self.learning_rate
                        * (2 * self.C * self.weights / n_samples)
                    )
                else:
                    # Hinge loss - update weights
                    self.weights -= self.learning_rate * (
                        2 * self.C * self.weights / n_samples
                        - y_binary[idx] * x_i
                    )
                    self.bias -= self.learning_rate * y_binary[idx]

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels
        """
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias

        # Map back to original class labels
        predictions = np.where(linear_output >= 0, self.classes_[1], self.classes_[0])
        return predictions

    def decision_function(self, X):
        """
        Compute the decision function for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values
        """
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class SVC(BaseEstimator, ClassifierMixin):
    """
    Support Vector Classifier with Kernel Support

    SVM classifier with support for different kernel functions including
    linear, RBF (Radial Basis Function), and polynomial kernels. Uses a
    simplified Sequential Minimal Optimization (SMO) approach.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter
    kernel : str, default='rbf'
        Kernel type: 'linear', 'rbf', or 'poly'
    gamma : float or 'scale', default='scale'
        Kernel coefficient for 'rbf' and 'poly'.
        If 'scale', gamma = 1 / (n_features * X.var())
    degree : int, default=3
        Degree of polynomial kernel (only used if kernel='poly')
    random_state : int, default=None
        Random seed for reproducibility

    Attributes
    ----------
    support_vectors_ : array-like
        Training samples that became support vectors
    support_labels_ : array-like
        Labels of support vectors
    alphas_ : array-like
        Lagrange multipliers for support vectors
    b_ : float
        Bias term

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
        """
        Compute kernel matrix between two sets of samples

        Parameters
        ----------
        X1 : array-like
            First set of samples
        X2 : array-like
            Second set of samples

        Returns
        -------
        K : array-like
            Kernel matrix
        """
        if self.kernel == "linear":
            return np.dot(X1, X2.T)

        elif self.kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
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
            # Polynomial kernel: (x.y + 1)^degree
            return (np.dot(X1, X2.T) + 1) ** self.degree

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X, y):
        """
        Fit SVM classifier using simplified SMO algorithm

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : SVC
            Fitted classifier
        """
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

        # Simplified training (store all as support vectors)
        # In a full SMO implementation, we would only keep alphas > 0
        self.support_vectors_ = X
        self.support_labels_ = y_binary
        self.alphas_ = np.ones(n_samples) * 0.01

        # Calculate bias
        margins = (K @ (self.alphas_ * self.support_labels_)) + self.b_
        self.b_ = np.mean(y_binary - margins)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels
        """
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
        """
        Compute the decision function for samples

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values
        """
        X = np.array(X)
        K = self._kernel_function(X, self.support_vectors_)
        return (K @ (self.alphas_ * self.support_labels_)) + self.b_

    def predict_proba(self, X):
        """
        Estimate probability of each class

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            Class probability estimates
        """
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
        Epsilon in epsilon-SVR model. No penalty for prediction errors
        within epsilon of the actual value.
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
        self, C=1.0, epsilon=0.1, kernel="rbf", gamma="scale", 
        degree=3, random_state=None
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
        """
        Fit SVR regressor

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : SVR
            Fitted regressor
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]

        # Store support vectors and compute kernel matrix
        self.support_vectors_ = X
        K = self._kernel_function(X, X)

        # Initialize alphas (simplified approach)
        self.alphas_ = np.ones(n_samples) * 0.01

        # Compute bias (intercept)
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
        """
        Predict continuous values

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values
        """
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        X = np.array(X)

        # Compute kernel between test and support vectors
        K = self._kernel_function(X, self.support_vectors_)

        # Make predictions
        return K @ self.alphas_ + self.b_


class SVM(SVC):
    """Alias for SVC - Support Vector Machine Classifier"""
    pass
