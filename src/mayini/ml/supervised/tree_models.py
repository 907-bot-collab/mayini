import numpy as np
from ..base import BaseClassifier, BaseRegressor


class Node:
    """Decision tree node"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier(BaseClassifier):
    """
    Decision Tree Classifier using CART algorithm

    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split
    min_samples_leaf : int, default=1
        Minimum samples required at leaf
    criterion : str, default='gini'
        Split criterion ('gini' or 'entropy')
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree_ = None
        self.n_classes_ = None
        self.classes_ = None

    def _gini(self, y):
        """Calculate Gini impurity"""
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """Calculate entropy"""
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _impurity(self, y):
        """Calculate impurity based on criterion"""
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)

    def _split(self, X, y, feature, threshold):
        """Split dataset based on feature and threshold"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        """Find the best split"""
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
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                # Calculate information gain
                n_left, n_right = len(y_left), len(y_right)
                child_impurity = (n_left / n_samples) * self._impurity(y_left) +                                 (n_right / n_samples) * self._impurity(y_right)
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
        if (self.max_depth is not None and depth >= self.max_depth) or            n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)

        # Find best split
        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = np.bincount(y.astype(int)).argmax()
            return Node(value=leaf_value)

        # Split and recurse
        X_left, X_right, y_left, y_right = self._split(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        """Fit decision tree"""
        X, y = self._validate_input(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Map classes to integers
        self.class_map_ = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([self.class_map_[c] for c in y])

        self.tree_ = self._build_tree(X, y_mapped)
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
        """Predict class labels"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        predictions = np.array([self._traverse_tree(x, self.tree_) for x in X])
        return np.array([self.classes_[int(p)] for p in predictions])


class DecisionTreeRegressor(BaseRegressor):
    """
    Decision Tree Regressor

    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum samples required to split
    min_samples_leaf : int, default=1
        Minimum samples required at leaf
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _mse(self, y):
        """Calculate mean squared error"""
        if len(y) == 0:
            return 0
        return np.var(y)

    def _split(self, X, y, feature, threshold):
        """Split dataset"""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        """Find best split"""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                mse = len(y_left) * self._mse(y_left) + len(y_right) * self._mse(y_right)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        """Build regression tree"""
        n_samples = X.shape[0]

        if (self.max_depth is not None and depth >= self.max_depth) or            n_samples < self.min_samples_split:
            return Node(value=np.mean(y))

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return Node(value=np.mean(y))

        X_left, X_right, y_left, y_right = self._split(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        """Fit regression tree"""
        X, y = self._validate_input(X, y)
        self.tree_ = self._build_tree(X, y)
        self.is_fitted_ = True
        return self

    def _traverse_tree(self, x, node):
        """Traverse tree"""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Predict values"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)
        return np.array([self._traverse_tree(x, self.tree_) for x in X])


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest Classifier

    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees
    max_depth : int, default=None
        Maximum depth of trees
    min_samples_split : int, default=2
        Minimum samples to split
    max_features : str or int, default='sqrt'
        Number of features to consider for splits
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt'):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_ = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """Fit random forest"""
        X, y = self._validate_input(X, y)
        self.classes_ = np.unique(y)
        self.trees_ = []

        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using majority voting"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees_])

        # Majority vote
        final_predictions = []
        for i in range(X.shape[0]):
            values, counts = np.unique(predictions[:, i], return_counts=True)
            final_predictions.append(values[counts.argmax()])

        return np.array(final_predictions)
