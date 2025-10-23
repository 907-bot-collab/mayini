import numpy as np
from ..base import BaseTransformer


class FeatureInteractions(BaseTransformer):
    """
    Generate pairwise feature interactions

    Parameters:
    -----------
    interaction_type : str, default='multiply'
        Type of interaction ('multiply', 'add', 'subtract', 'divide')

    Example:
    --------
    >>> from mayini.preprocessing import FeatureInteractions
    >>> interactions = FeatureInteractions(interaction_type='multiply')
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> interactions.fit_transform(X)
    """

    def __init__(self, interaction_type='multiply'):
        super().__init__()
        self.interaction_type = interaction_type
        self.n_features_ = None

    def fit(self, X, y=None):
        """Fit transformer"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Generate interaction features"""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        interactions = [X]

        for i in range(self.n_features_):
            for j in range(i + 1, self.n_features_):
                if self.interaction_type == 'multiply':
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'add':
                    interaction = (X[:, i] + X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'subtract':
                    interaction = (X[:, i] - X[:, j]).reshape(-1, 1)
                elif self.interaction_type == 'divide':
                    interaction = (X[:, i] / (X[:, j] + 1e-10)).reshape(-1, 1)
                else:
                    raise ValueError(f"Unknown interaction type: {self.interaction_type}")

                interactions.append(interaction)

        return np.hstack(interactions)

