import numpy as np

class CorrelationSelector:
    """
    Feature selector that removes highly correlated features.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        if isinstance(X, list):
            X = np.array(X)
        corr_matrix = np.abs(np.corrcoef(X.T))
        self.selected_features_ = []
        n_features = X.shape[1]
        for i in range(n_features):
            keep = True
            for j in self.selected_features_:
                if corr_matrix[i, j] > self.threshold:
                    keep = False
                    break
            if keep:
                self.selected_features_.append(i)
        self.selected_features_ = np.array(self.selected_features_)
        return self

    def transform(self, X):
        if isinstance(X, list):
            X = np.array(X)
        if self.selected_features_ is None:
            raise ValueError("CorrelationSelector has not been fitted yet.")
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
