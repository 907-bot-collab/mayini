import numpy as np
from ..base import BaseClassifier, BaseRegressor


class VotingClassifier(BaseClassifier):
    """
    Voting Classifier

    Combines multiple classifiers using voting

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples
    voting : str, default='hard'
        'hard' for majority vote, 'soft' for weighted probabilities
    weights : array-like, default=None
        Weights for each estimator

    Example
    -------
    >>> from mayini.ml import VotingClassifier, LogisticRegression
    >>> from mayini.ml import DecisionTreeClassifier, KNeighborsClassifier
    >>>
    >>> estimators = [
    ...     ('lr', LogisticRegression()),
    ...     ('dt', DecisionTreeClassifier()),
    ...     ('knn', KNeighborsClassifier())
    ... ]
    >>> voting = VotingClassifier(estimators=estimators, voting='hard')
    >>> voting.fit(X_train, y_train)
    """

    def __init__(self, estimators, voting="hard", weights=None):
        super().__init__()
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.classes_ = None

    def fit(self, X, y):
        """Fit all estimators"""
        X, y = self._validate_input(X, y)

        self.classes_ = np.unique(y)

        # Fit each estimator
        for name, estimator in self.estimators:
            estimator.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using voting"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if self.voting == "hard":
            # Majority voting
            predictions = np.array(
                [estimator.predict(X) for _, estimator in self.estimators]
            )

            if self.weights is not None:
                # Weighted voting
                weighted_votes = np.zeros((X.shape[0], len(self.classes_)))
                for i, (_, estimator) in enumerate(self.estimators):
                    pred = estimator.predict(X)
                    for j, cls in enumerate(self.classes_):
                        weighted_votes[:, j] += self.weights[i] * (pred == cls)
                return self.classes_[np.argmax(weighted_votes, axis=1)]
            else:
                # Simple majority
                final_predictions = []
                for i in range(X.shape[0]):
                    votes = predictions[:, i]
                    unique, counts = np.unique(votes, return_counts=True)
                    final_predictions.append(unique[counts.argmax()])
                return np.array(final_predictions)

        elif self.voting == "soft":
            # Soft voting (average probabilities)
            all_probas = []
            for name, estimator in self.estimators:
                if hasattr(estimator, "predict_proba"):
                    probas = estimator.predict_proba(X)
                else:
                    raise ValueError(
                        f"Estimator {name} doesn't support predict_proba"
                    )
                all_probas.append(probas)

            if self.weights is not None:
                avg_proba = np.average(all_probas, axis=0, weights=self.weights)
            else:
                avg_proba = np.mean(all_probas, axis=0)

            return self.classes_[np.argmax(avg_proba, axis=1)]


class VotingRegressor(BaseRegressor):
    """
    Voting Regressor

    Combines multiple regressors by averaging predictions

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples
    weights : array-like, default=None
        Weights for each estimator

    Example
    -------
    >>> from mayini.ml import VotingRegressor, LinearRegression
    >>> from mayini.ml import DecisionTreeRegressor
    >>> estimators = [
    ...     ('lr', LinearRegression()),
    ...     ('dt', DecisionTreeRegressor())
    ... ]
    >>> voting = VotingRegressor(estimators=estimators)
    >>> voting.fit(X_train, y_train)
    """

    def __init__(self, estimators, weights=None):
        super().__init__()
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        """Fit all estimators"""
        X, y = self._validate_input(X, y)

        for name, estimator in self.estimators:
            estimator.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using average"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        predictions = np.array(
            [estimator.predict(X) for _, estimator in self.estimators]
        )

        if self.weights is not None:
            return np.average(predictions, axis=0, weights=self.weights)
        else:
            return np.mean(predictions, axis=0)
