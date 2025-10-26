import numpy as np
from ..base import BaseClassifier, BaseRegressor
from ..supervised.tree_models import DecisionTreeClassifier, DecisionTreeRegressor


class BaggingClassifier(BaseClassifier):
    """
    Bagging (Bootstrap Aggregating) Classifier

    Parameters
    ----------
    base_estimator : object, default=None
        Base estimator (default: DecisionTreeClassifier)
    n_estimators : int, default=10
        Number of base estimators
    max_samples : float, default=1.0
        Fraction of samples to draw for each base estimator
    max_features : float, default=1.0
        Fraction of features to draw for each base estimator
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.ml import BaggingClassifier
    >>> bagging = BaggingClassifier(n_estimators=10)
    >>> bagging.fit(X_train, y_train)
    >>> y_pred = bagging.predict(X_test)
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        random_state=None,
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        """Fit bagging classifier"""
        X, y = self._validate_input(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        # Determine sample and feature sizes
        n_samples_per_estimator = int(n_samples * self.max_samples)
        n_features_per_estimator = int(n_features * self.max_features)

        self.estimators_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample
            sample_indices = np.random.choice(
                n_samples, n_samples_per_estimator, replace=True
            )
            feature_indices = np.random.choice(
                n_features, n_features_per_estimator, replace=False
            )

            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]

            # Train base estimator
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier()
            else:
                estimator = self.base_estimator

            estimator.fit(X_sample, y_sample)
            self.estimators_.append((estimator, feature_indices))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using majority voting"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        predictions = []
        for estimator, feature_indices in self.estimators_:
            pred = estimator.predict(X[:, feature_indices])
            predictions.append(pred)

        predictions = np.array(predictions)

        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[counts.argmax()])

        return np.array(final_predictions)


class BaggingRegressor(BaseRegressor):
    """
    Bagging Regressor

    Parameters
    ----------
    base_estimator : object, default=None
        Base estimator (default: DecisionTreeRegressor)
    n_estimators : int, default=10
        Number of base estimators
    max_samples : float, default=1.0
        Fraction of samples to draw
    max_features : float, default=1.0
        Fraction of features to draw
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.ml import BaggingRegressor
    >>> bagging = BaggingRegressor(n_estimators=10)
    >>> bagging.fit(X_train, y_train)
    """

    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        random_state=None,
    ):
        super().__init__()
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        """Fit bagging regressor"""
        X, y = self._validate_input(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        n_samples_per_estimator = int(n_samples * self.max_samples)
        n_features_per_estimator = int(n_features * self.max_features)

        self.estimators_ = []

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(
                n_samples, n_samples_per_estimator, replace=True
            )
            feature_indices = np.random.choice(
                n_features, n_features_per_estimator, replace=False
            )

            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]

            if self.base_estimator is None:
                estimator = DecisionTreeRegressor()
            else:
                estimator = self.base_estimator

            estimator.fit(X_sample, y_sample)
            self.estimators_.append((estimator, feature_indices))

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using average"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        predictions = []
        for estimator, feature_indices in self.estimators_:
            pred = estimator.predict(X[:, feature_indices])
            predictions.append(pred)

        return np.mean(predictions, axis=0)
