import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin


class AdaBoost(BaseEstimator, ClassifierMixin):
    """
    AdaBoost (Adaptive Boosting) Classifier
    
    A meta-algorithm that combines weak learners iteratively, giving more
    weight to misclassified samples in each iteration.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of weak learners to train
    learning_rate : float, default=1.0
        Learning rate (also called alpha). Controls the contribution of each learner
    random_state : int, default=None
        Random seed for reproducibility
    
    Attributes
    ----------
    estimators_ : list
        Trained weak learners
    estimator_weights_ : array-like
        Weights for each estimator
    
    Example
    -------
    >>> from mayini.ml import AdaBoost
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]])
    >>> y = np.array([0, 0, 1, 1])
    >>> ada = AdaBoost(n_estimators=10)
    >>> ada.fit(X, y)
    >>> ada.predict([[5, 5]])
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit AdaBoost classifier"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost only supports binary classification")
        
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for m in range(self.n_estimators):
            best_error = float('inf')
            best_feature = 0
            best_threshold = 0
            
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    predictions = np.where(X[:, feature] <= threshold,
                                         self.classes_[0],
                                         self.classes_[1])
                    
                    errors = (predictions != y).astype(int)
                    weighted_error = np.sum(sample_weights * errors)
                    
                    if weighted_error < best_error:
                        best_error = weighted_error
                        best_feature = feature
                        best_threshold = threshold
            
            if best_error == 0:
                best_error = 1e-10
            if best_error >= 0.5:
                best_error = 0.4999
            
            alpha = self.learning_rate * np.log((1 - best_error) / best_error)
            self.estimator_weights_.append(alpha)
            
            self.estimators_.append({
                'feature': best_feature,
                'threshold': best_threshold,
                'alpha': alpha
            })
            
            predictions = np.where(X[:, best_feature] <= best_threshold,
                                 self.classes_[0],
                                 self.classes_[1])
            errors = (predictions != y).astype(int)
            
            sample_weights *= np.exp(-alpha * (2 * errors - 1))
            sample_weights /= np.sum(sample_weights)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)
        
        for estimator in self.estimators_:
            feature = estimator['feature']
            threshold = estimator['threshold']
            alpha = estimator['alpha']
            
            predictions = np.where(X[:, feature] <= threshold, 0, 1)
            final_predictions += alpha * (2 * predictions - 1)
        
        return np.where(final_predictions >= 0, self.classes_[1], self.classes_[0])

class AdaBoostRegressor:
    """
    Adaptive Boosting (AdaBoost) Regressor

    Combines multiple weak regressors sequentially to create a strong regressor.
    Each weak regressor is trained on a weighted version of the dataset.
    The weights are adjusted based on the errors from previous regressors.

    Parameters:
    -----------
    n_estimators : int, default=50
        Number of weak learners (base regressors) to use

    learning_rate : float, default=0.1
        Shrinkage parameter - controls contribution of each regressor
        Typical range: 0.01 to 1.0

    random_state : int or None, default=None
        Random seed for reproducibility

    loss : str, default='linear'
        Loss function to use
        Options: 'linear', 'square', 'exponential'

    Attributes:
    -----------
    models : list
        List of trained weak regressors

    alphas : array
        Weights for each regressor

    errors : array
        Training errors at each iteration

    Example:
    --------
    >>> from mayini.ml import AdaBoostRegressor
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randn(100)
    >>> 
    >>> ada = AdaBoostRegressor(n_estimators=50, learning_rate=0.1)
    >>> ada.fit(X, y)
    >>> y_pred = ada.predict(X)
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, random_state=None, loss='linear'):
        """Initialize AdaBoostRegressor"""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.loss = loss

        self.models = []
        self.alphas = np.array([])
        self.errors = np.array([])
        self.train_errors = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
        Train the AdaBoost Regressor

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features

        y : array-like of shape (n_samples,)
            Training targets

        Returns:
        --------
        self : AdaBoostRegressor
            Fitted regressor
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples = X.shape[0]

        # Initialize sample weights uniformly
        sample_weights = np.ones(n_samples) / n_samples

        self.models = []
        self.alphas = []
        self.train_errors = []

        for iteration in range(self.n_estimators):
            # Fit weak learner on weighted samples
            model = self._fit_weak_learner(X, y, sample_weights)
            self.models.append(model)

            # Make predictions
            y_pred = self._predict_weak_learner(model, X)

            # Calculate error
            error = np.mean(sample_weights * np.abs(y - y_pred))
            self.train_errors.append(error)

            # Check for perfect prediction
            if error < 1e-10:
                alpha = 1.0
            elif error >= 0.5:
                # If error too high, stop boosting
                break
            else:
                # Calculate alpha based on error
                alpha = self.learning_rate * (np.log(1 - error) - np.log(error)) / 2

            self.alphas.append(alpha)

            # Update sample weights
            residuals = np.abs(y - y_pred)
            sample_weights *= np.exp(-alpha * residuals)
            sample_weights /= np.sum(sample_weights)

        self.alphas = np.array(self.alphas)
        self.errors = np.array(self.train_errors)

        return self

    def _fit_weak_learner(self, X, y, sample_weights):
        """
        Fit a weak learner (linear regression) with sample weights

        Uses weighted least squares regression
        """
        # Weighted least squares
        sqrt_weights = np.sqrt(sample_weights)
        X_weighted = X * sqrt_weights[:, np.newaxis]
        y_weighted = y * sqrt_weights

        # Simple linear regression using numpy
        # X_weighted.T @ X_weighted @ coef = X_weighted.T @ y_weighted
        try:
            coef = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
            intercept = np.mean(y_weighted - X_weighted @ coef)
        except:
            # Fallback to simple mean
            coef = np.zeros(X.shape[1])
            intercept = np.mean(y)

        return {'coef': coef, 'intercept': intercept}

    def _predict_weak_learner(self, model, X):
        """Make predictions with a weak learner"""
        return X @ model['coef'] + model['intercept']

    def predict(self, X):
        """
        Predict target values for X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X, dtype=np.float64)

        if len(self.models) == 0:
            raise ValueError("Model not fitted yet")

        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        for alpha, model in zip(self.alphas, self.models):
            y_pred += alpha * self._predict_weak_learner(model, X)

        return y_pred

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test features

        y : array-like of shape (n_samples,)
            Test targets

        Returns:
        --------
        score : float
            R² score between 0 and 1 (higher is better)
        """
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    def get_feature_importance(self):
        """
        Get feature importance based on coefficients across all models

        Returns:
        --------
        importance : array of shape (n_features,)
            Importance score for each feature
        """
        if len(self.models) == 0:
            raise ValueError("Model not fitted yet")

        n_features = self.models[0]['coef'].shape[0]
        importance = np.zeros(n_features)

        for alpha, model in zip(self.alphas, self.models):
            importance += alpha * np.abs(model['coef'])

        importance /= np.sum(importance)
        return importance

    def get_params(self):
        """Get model parameters"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'loss': self.loss,
            'n_models_': len(self.models),
        }

    def set_params(self, **params):
        """Set model parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    AdaBoost Classifier (Alias for AdaBoost)
    
    Uses the same implementation as AdaBoost for consistency with
    scikit-learn naming conventions.
    
    Example
    -------
    >>> from mayini.ml import AdaBoostClassifier
    >>> ada = AdaBoostClassifier(n_estimators=10)
    >>> ada.fit(X, y)
    >>> ada.predict(X)
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Fit AdaBoost classifier"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostClassifier only supports binary classification")
        
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples
        
        for m in range(self.n_estimators):
            best_error = float('inf')
            best_feature = 0
            best_threshold = 0
            
            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    predictions = np.where(X[:, feature] <= threshold,
                                         self.classes_[0],
                                         self.classes_[1])
                    
                    errors = (predictions != y).astype(int)
                    weighted_error = np.sum(sample_weights * errors)
                    
                    if weighted_error < best_error:
                        best_error = weighted_error
                        best_feature = feature
                        best_threshold = threshold
            
            if best_error == 0:
                best_error = 1e-10
            if best_error >= 0.5:
                best_error = 0.4999
            
            alpha = self.learning_rate * np.log((1 - best_error) / best_error)
            self.estimator_weights_.append(alpha)
            
            self.estimators_.append({
                'feature': best_feature,
                'threshold': best_threshold,
                'alpha': alpha
            })
            
            predictions = np.where(X[:, best_feature] <= best_threshold,
                                 self.classes_[0],
                                 self.classes_[1])
            errors = (predictions != y).astype(int)
            
            sample_weights *= np.exp(-alpha * (2 * errors - 1))
            sample_weights /= np.sum(sample_weights)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples)
        
        for estimator in self.estimators_:
            feature = estimator['feature']
            threshold = estimator['threshold']
            alpha = estimator['alpha']
            
            predictions = np.where(X[:, feature] <= threshold, 0, 1)
            final_predictions += alpha * (2 * predictions - 1)
        
        return np.where(final_predictions >= 0, self.classes_[1], self.classes_[0])


class GradientBoosting(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Regressor
    
    Builds an ensemble of decision trees sequentially, where each tree
    corrects the errors made by previous trees using gradient descent.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages
    learning_rate : float, default=0.1
        Shrinks the contribution of each tree
    max_depth : int, default=3
        Maximum depth of individual trees
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at leaf node
    random_state : int, default=None
        Random seed
    
    Example
    -------
    >>> from mayini.ml import GradientBoosting
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> y = np.array([1.5, 2.5, 3.5, 4.5])
    >>> gb = GradientBoosting(n_estimators=10)
    >>> gb.fit(X, y)
    >>> gb.predict([[2.5, 3.5]])
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.estimators_ = []
        self.initial_prediction_ = None
    
    def fit(self, X, y):
        """Fit Gradient Boosting regressor"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        self.initial_prediction_ = np.mean(y)
        current_predictions = np.full_like(y, self.initial_prediction_, dtype=float)
        
        for m in range(self.n_estimators):
            residuals = y - current_predictions
            tree_params = self._fit_tree_to_residuals(X, residuals)
            self.estimators_.append(tree_params)
            tree_predictions = self._predict_with_tree(X, tree_params)
            current_predictions += self.learning_rate * tree_predictions
        
        self.is_fitted_ = True
        return self
    
    def _fit_tree_to_residuals(self, X, residuals):
        """Fit a simple decision tree to residuals"""
        best_split = None
        best_mse = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_pred = np.mean(residuals[left_mask])
                right_pred = np.mean(residuals[right_mask])
                
                mse = np.sum((residuals[left_mask] - left_pred) ** 2) + \
                      np.sum((residuals[right_mask] - right_pred) ** 2)
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = {
                        'feature': feature,
                        'threshold': threshold,
                        'left_value': left_pred,
                        'right_value': right_pred
                    }
        
        return best_split if best_split else {'feature': 0, 'threshold': 0,
                                              'left_value': 0, 'right_value': 0}
    
    def _predict_with_tree(self, X, tree_params):
        """Predict using a single tree"""
        predictions = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            if X[i, tree_params['feature']] <= tree_params['threshold']:
                predictions[i] = tree_params['left_value']
            else:
                predictions[i] = tree_params['right_value']
        
        return predictions
    
    def predict(self, X):
        """Predict target values"""
        if not hasattr(self, 'is_fitted_') or not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X)
        predictions = np.full(X.shape[0], self.initial_prediction_)
        
        for tree_params in self.estimators_:
            tree_preds = self._predict_with_tree(X, tree_params)
            predictions += self.learning_rate * tree_preds
        
        return predictions
