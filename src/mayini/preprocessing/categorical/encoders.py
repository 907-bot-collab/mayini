import numpy as np
from ..base import BaseTransformer


class LabelEncoder(BaseTransformer):
    """
    Encode target labels with value between 0 and n_classes-1

    Example
    -------
    >>> from mayini.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> y_encoded = le.fit_transform(['cat', 'dog', 'cat'])
    >>> # Returns: [0, 1, 0]
    """

    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.class_to_index_ = None

    def fit(self, y):
        """
        Fit label encoder

        Parameters
        ----------
        y : array-like
            Target values

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}
        self.is_fitted_ = True
        return self

    def transform(self, y):
        """Transform labels to normalized encoding"""
        self._check_is_fitted()
        y = np.asarray(y)
        return np.array([self.class_to_index_[val] for val in y])

    def inverse_transform(self, y):
        """Transform labels back to original encoding"""
        self._check_is_fitted()
        y = np.asarray(y)
        return np.array([self.classes_[val] for val in y])


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as one-hot numeric array

    Parameters
    ----------
    sparse : bool, default=False
        If True, return sparse matrix (not implemented yet)
    handle_unknown : str, default='error'
        How to handle unknown categories ('error' or 'ignore')

    Example
    -------
    >>> from mayini.preprocessing import OneHotEncoder
    >>> ohe = OneHotEncoder()
    >>> X = [['cat'], ['dog'], ['cat']]
    >>> X_encoded = ohe.fit_transform(X)
    """

    def __init__(self, sparse=False, handle_unknown="error"):
        super().__init__()
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def fit(self, X, y=None):
        """Fit OneHot encoder"""
        X, _ = self._validate_input(X)

        self.categories_ = []
        for col in range(X.shape[1]):
            categories = np.unique(X[:, col])
            self.categories_.append(categories)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform X using one-hot encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        n_samples = X.shape[0]
        encoded_columns = []

        for col in range(X.shape[1]):
            categories = self.categories_[col]
            n_categories = len(categories)

            # Create one-hot matrix for this column
            col_data = X[:, col]
            one_hot = np.zeros((n_samples, n_categories))

            for i, val in enumerate(col_data):
                if val in categories:
                    idx = np.where(categories == val)[0][0]
                    one_hot[i, idx] = 1
                elif self.handle_unknown == "ignore":
                    # Leave as all zeros
                    pass
                else:
                    raise ValueError(f"Unknown category: {val}")

            encoded_columns.append(one_hot)

        return np.hstack(encoded_columns)

    def get_feature_names(self, input_features=None):
        """Get feature names for output"""
        self._check_is_fitted()

        names = []
        for col_idx, categories in enumerate(self.categories_):
            if input_features is not None:
                col_name = input_features[col_idx]
            else:
                col_name = f"x{col_idx}"

            for category in categories:
                names.append(f"{col_name}_{category}")

        return names


class OrdinalEncoder(BaseTransformer):
    """
    Encode categorical features as integer array

    Parameters
    ----------
    handle_unknown : str, default='error'
        How to handle unknown categories

    Example
    -------
    >>> from mayini.preprocessing import OrdinalEncoder
    >>> oe = OrdinalEncoder()
    >>> X = [['low'], ['high'], ['medium']]
    >>> X_encoded = oe.fit_transform(X)
    """

    def __init__(self, handle_unknown="error"):
        super().__init__()
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def fit(self, X, y=None):
        """Fit ordinal encoder"""
        X, _ = self._validate_input(X)

        self.categories_ = []
        for col in range(X.shape[1]):
            categories = np.unique(X[:, col])
            self.categories_.append(categories)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform X using ordinal encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        X_encoded = np.zeros_like(X, dtype=float)

        for col in range(X.shape[1]):
            categories = self.categories_[col]
            category_to_index = {cat: idx for idx, cat in enumerate(categories)}

            for i, val in enumerate(X[:, col]):
                if val in category_to_index:
                    X_encoded[i, col] = category_to_index[val]
                elif self.handle_unknown == "error":
                    raise ValueError(f"Unknown category: {val}")
                else:
                    X_encoded[i, col] = -1

        return X_encoded.astype(float)
