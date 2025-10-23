import numpy as np
from ..base import BaseClassifier, BaseRegressor
from ..supervised.tree_models import DecisionTreeClassifier, DecisionTreeRegressor


class AdaBoostClassifier(BaseClassifier):
    """
    AdaBoost (Adaptive Boosting) Classifier
    
    Parameters:
    -----------
    n_estimators : int, default=50
        Number of weak learners
    learning_rate : float, default=1.0
        Weight applied to each classifier
    
    Example:
    --------
    >>> from mayini.ml import AdaBoostClassifier
    >>> ada = AdaBoostClassifier(n_estimators=50)
    >>> ada.fit(X_train, y_train)
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit AdaBoost classifier"""
        X, y = self._validate_input(X, y)
        
        self.classes_ = np.unique(y)
        
        # Convert to {-1, +1}
        y_ = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        
        for _ in range(self.n_estimators):
            # Train weak learner (decision stump)
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y)
            
            predictions = estimator.predict(X)
            predictions = np.where(predictions == self.classes_[0], -1, 1)
            
            # Calculate error
            incorrect = predictions != y_
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            sample_weights *= np.exp(-estimator_weight * y_ * predictions)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict using weighted majority vote"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        
        predictions = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            pred = estimator.predict(X)
            pred = np.where(pred == self.classes_[0], -1, 1)
            predictions += weight * pred
        
        return np.where(predictions >= 0, self.classes_[1], self.classes_[0])


class GradientBoostingClassifier(BaseClassifier):
    """
    Gradient Boosting Classifier
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Learning rate shrinks contribution of each tree
    max_depth : int, default=3
        Maximum depth of individual trees
    
    Example:
    --------
    >>> from mayini.ml import GradientBoostingClassifier
    >>> gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    >>> gb.fit(X_train, y_train)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.init_prediction_ = None
        self.classes_ = None
    
    def _sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def fit(self, X, y):
        """Fit gradient boosting classifier"""
        X, y = self._validate_input(X, y)
        
        self.classes_ = np.unique(y)
        
        # Convert to {0, 1}
        y_binary = (y == self.classes_[1]).astype(int)
        
        # Initialize with log odds
        positive_class_prior = np.mean(y_binary)
        self.init_prediction_ = np.log(positive_class_prior / (1 - positive_class_prior + 1e-10))
        
        # Current predictions (log odds)
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Compute pseudo-residuals (gradient)
            probs = self._sigmoid(predictions)
            residuals = y_binary - probs
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            self.estimators_.append(tree)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        probs = self._sigmoid(predictions)
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Predict class labels"""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class GradientBoostingRegressor(BaseRegressor):
    """
    Gradient Boosting Regressor
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Learning rate
    max_depth : int, default=3
        Maximum depth of trees
    
    Example:
    --------
    >>> from mayini.ml import GradientBoostingRegressor
    >>> gb = GradientBoostingRegressor(n_estimators=100)
    >>> gb.fit(X_train, y_train)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.init_prediction_ = None
    
    def fit(self, X, y):
        """Fit gradient boosting regressor"""
        X, y = self._validate_input(X, y)
        
        # Initialize with mean
        self.init_prediction_ = np.mean(y)
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        self.estimators_ = []
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update predictions
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            self.estimators_.append(tree)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict values"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions

