import numpy as np
from ..base import BaseEstimator, ClassifierMixin, RegressorMixin


class Node:
    """Decision tree node"""

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, value=None
    ):
        """
        Initialize a decision tree node

        Parameters
        ----------
        feature : int, optional
            Feature index to split on
        threshold : float, optional
            Threshold value for split
        left : Node, optional
            Left child node
        right : Node, optional
            Right child node
        value : scalar, optional
            Class label or regression value (for leaf nodes)
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        """Check if node is a leaf"""
        return self.value is not None


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier using CART algorithm

    Builds a decision tree using the Classification and Regression Trees (CART)
    algorithm, which uses Gini impurity or entropy as the splitting criterion.

    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree. If None, tree grows until all leaves
        are pure or contain < min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    criterion : str, default='gini'
        Function to measure split quality. Options: 'gini' or 'entropy'.

    Attributes
    ----------
    tree_ : Node
        The trained decision tree structure
    classes_ : array-like
        Unique class labels from training data
    n_classes_ : int
        Number of unique classes

    Example
    -------
    >>> from mayini.ml import DecisionTreeClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([0, 1, 1])
    >>> dt = DecisionTreeClassifier(max_depth=5)
    >>> dt.fit(X, y)
    >>> dt.predict([[2, 3]])
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.class_map_ = None

    def _gini(self, y):
        """
        Calculate Gini impurity

        Gini = 1 - sum(p_i^2) where p_i is the probability of class i
        """
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1 - np.sum(probabilities**2)

    def _entropy(self, y):
        """
        Calculate entropy

        Entropy = -sum(p_i * log2(p_i)) where p_i is the probability of class i
        """
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _split(self, X, y, feature, threshold):
        """Split dataset based on feature and threshold"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return (
            X[left_mask],
            X[right_mask],
            y[left_mask],
            y[right_mask],
        )

    def _best_split(self, X, y):
        """Find the best split to minimize impurity"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        parent_impurity = self._impurity(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(
                    X, y, feature, threshold
                )

                if (
                    len(y_left) < self.min_samples_leaf
                    or len(y_right) < self.min_samples_leaf
                ):
                    continue

                # Calculate information gain
                n_left, n_right = len(y_left), len(y_right)
                child_impurity = (n_left / n_samples) * self._impurity(
                    y_left
                ) + (n_right / n_samples) * self._impurity(y_right)
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_classes == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)

        # Split and recurse
        (X_left, X_right, y_left, y_right) = self._split(
            X, y, best_feature, best_threshold
        )
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def fit(self, X, y):
        """
        Fit decision tree classifier

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted classifier
        """
        X = np.array(X)
        y = np.array(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Map classes to integers
        self.class_map_ = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([self.class_map_[c] for c in y])

        self.tree_ = self._build_tree(X, y_mapped)
        self.is_fitted_ = True
        return self

    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for a single sample"""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

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
        predictions = np.array(
            [self._traverse_tree(x, self.tree_) for x in X]
        )
        return np.array([self.classes_[int(p)] for p in predictions])


class DecisionTree(DecisionTreeClassifier):
    """Alias for DecisionTreeClassifier for backward compatibility"""

    pass


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision Tree Regressor using CART algorithm

    Builds a regression tree using the CART algorithm with mean squared
    error (MSE) as the splitting criterion.

    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node

    Example
    -------
    >>> from mayini.ml import DecisionTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1.5, 2.5, 3.5])
    >>> dt = DecisionTreeRegressor(max_depth=5)
    >>> dt.fit(X, y)
    >>> dt.predict([[2, 3]])
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _mse(self, y):
        """Calculate mean squared error (variance)"""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _split(self, X, y, feature, threshold):
        """Split dataset based on feature and threshold"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return (
            X[left_mask],
            X[right_mask],
            y[left_mask],
            y[right_mask],
        )

    def _best_split(self, X, y):
        """Find best split to minimize MSE"""
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(
                    X, y, feature, threshold
                )

                if (
                    len(y_left) < self.min_samples_leaf
                    or len(y_right) < self.min_samples_leaf
                ):
                    continue

                mse = len(y_left) * self._mse(y_left) + len(y_right) * self._mse(
                    y_right
                )

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Recursively build regression tree"""
        n_samples = X.shape[0]

        if (
            self.max_depth is not None and depth >= self.max_depth
        ) or n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return Node(value=np.mean(y))

        X_left, X_right, y_left, y_right = self._split(
            X, y, best_feature, best_threshold
        )
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def fit(self, X, y):
        """
        Fit regression tree

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted regressor
        """
        X = np.array(X)
        y = np.array(y)
        self.tree_ = self._build_tree(X, y)
        self.is_fitted_ = True
        return self

    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction"""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

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
        return np.array([self._traverse_tree(x, self.tree_) for x in X])


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier

    An ensemble of decision trees trained on random subsets of the data
    (bootstrap samples). Predictions are made by majority voting.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=None
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf
    max_features : str or int, default='sqrt'
        Number of features to consider for splits ('sqrt' or 'log2')
    random_state : int, default=None
        Random seed for reproducibility

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        Trained trees
    classes_ : array-like
        Unique class labels

    Example
    -------
    >>> from mayini.ml import RandomForestClassifier
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([0, 1, 1])
    >>> rf = RandomForestClassifier(n_estimators=10)
    >>> rf.fit(X, y)
    >>> rf.predict([[2, 3]])
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Fit random forest classifier

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : RandomForestClassifier
            Fitted classifier
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.trees_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels using majority voting

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
        predictions = np.array([tree.predict(X) for tree in self.trees_])

        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[counts.argmax()])

        return np.array(final_predictions)


class RandomForest(RandomForestClassifier):
    """Alias for RandomForestClassifier for backward compatibility"""

    pass


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Random Forest Regressor

    An ensemble of regression trees trained on random subsets of the data.
    Predictions are made by averaging the predictions of all trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=None
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples required to split
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf
    random_state : int, default=None
        Random seed

    Example
    -------
    >>> from mayini.ml import RandomForestRegressor
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1.5, 2.5, 3.5])
    >>> rf = RandomForestRegressor(n_estimators=10)
    >>> rf.fit(X, y)
    >>> rf.predict([[2, 3]])
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_ = []

    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Fit random forest regressor

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : RandomForestRegressor
            Fitted regressor
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.array(X)
        y = np.array(y)
        self.trees_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict values using average of all trees

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
        predictions = np.array([tree.predict(X) for tree in self.trees_])

        # Average predictions
        return np.mean(predictions, axis=0)
