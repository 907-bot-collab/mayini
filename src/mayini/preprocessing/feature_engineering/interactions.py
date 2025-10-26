import numpy as np
from ..base import BaseTransformer


class FeatureInteractions(BaseTransformer):
    """
    Generate pairwise feature interactions

    Parameters
    ----------
    interaction_type : str, default='multiply'
        Type of interaction ('multiply', 'add', 'subtract', 'divide')
    degree : int, default=2
        Degree of interactions (currently only 2 supported)

    Example
    -------
    >>> from mayini.preprocessing import FeatureInteractions
    >>> fi = FeatureInteractions(interaction_type='multiply')
    >>> X = [[1, 2, 3], [4, 5, 6]]
    >>> X_interact = fi.fit_transform(X)
    >>> # Returns original features + all pairwise products
    """

    def __init__(self, interaction_type="multiply", degree=2):
        super().__init__()
        self.interaction_type = interaction_type
        self.degree = degree
        self.n_features_ = None

    def fit(self, X, y=None):
        """Fit feature interactions"""
        X, _ = self._validate_input(X)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Generate interaction features"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        interactions = [X]  # Include original features

        # Generate pairwise interactions
        for i in range(self.n_features_):
            for j in range(i + 1, self.n_features_):
                if self.interaction_type == "multiply":
                    interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
                elif self.interaction_type == "add":
                    interaction = (X[:, i] + X[:, j]).reshape(-1, 1)
                elif self.interaction_type == "subtract":
                    interaction = (X[:, i] - X[:, j]).reshape(-1, 1)
                elif self.interaction_type == "divide":
                    # Avoid division by zero
                    interaction = (
                        X[:, i] / (X[:, j] + 1e-10)
                    ).reshape(-1, 1)
                else:
                    raise ValueError(f"Unknown interaction type: {self.interaction_type}")

                interactions.append(interaction)

        return np.hstack(interactions)

    def get_feature_names(self, input_features=None):
        """Get feature names for output"""
        self._check_is_fitted()

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_)]

        names = list(input_features)

        # Add interaction names
        op_symbol = {
            "multiply": "*",
            "add": "+",
            "subtract": "-",
            "divide": "/",
        }
        symbol = op_symbol.get(self.interaction_type, "*")

        for i in range(self.n_features_):
            for j in range(i + 1, self.n_features_):
                name = f"{input_features[i]}{symbol}{input_features[j]}"
                names.append(name)

        return names
