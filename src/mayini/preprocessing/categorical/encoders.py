import numpy as np
from ..base import BaseTransformer


class LabelEncoder(BaseTransformer):
    """
    Encode target labels with value between 0 and n_classes-1

    Example:
    --------
    >>> from mayini.preprocessing import LabelEncoder
    >>> le = LabelEncoder()
    >>> X = np.array(['cat', 'dog', 'cat', 'bird']).reshape(-1, 1)
    >>> X_encoded = le.fit_transform(X)
    """

    def __init__(self):
        super().__init__()
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, X, y=None):
        """Fit label encoder"""
        X, _ = self._validate_input(X)

        # Handle both 1D and 2D arrays
        if X.ndim == 2:
            X = X.flatten()

        self.classes_ = np.unique(X)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform labels to normalized encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        original_shape = X.shape
        if X.ndim == 2:
            X = X.flatten()

        encoded = np.array([self.class_to_idx_.get(x, -1) for x in X])

        if len(original_shape) == 2:
            encoded = encoded.reshape(original_shape)

        return encoded.astype(float)

    def inverse_transform(self, X):
        """Transform labels back to original encoding"""
        self._check_is_fitted()

        if isinstance(X, (list, np.ndarray)):
            X = np.array(X)

        original_shape = X.shape
        if X.ndim == 2:
            X = X.flatten()

        decoded = np.array([self.classes_[int(x)] if 0 <= int(x) < len(self.classes_) else None 
                           for x in X])

        if len(original_shape) == 2:
            decoded = decoded.reshape(original_shape)

        return decoded


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array

    Parameters:
    -----------
    sparse : bool, default=False
        Return sparse matrix if True
    handle_unknown : str, default='error'
        How to handle unknown categories ('error' or 'ignore')

    Example:
    --------
    >>> from mayini.preprocessing import OneHotEncoder
    >>> ohe = OneHotEncoder()
    >>> X = np.array(['cat', 'dog', 'cat']).reshape(-1, 1)
    >>> X_encoded = ohe.fit_transform(X)
    """

    def __init__(self, sparse=False, handle_unknown='error'):
        super().__init__()
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Fit OneHot encoder"""
        X, _ = self._validate_input(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.categories_ = []

        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            self.categories_.append(unique_vals)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data to one-hot encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        encoded_arrays = []

        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            col_data = X[:, col_idx]

            # Create one-hot matrix for this column
            n_samples = len(col_data)
            n_categories = len(categories)
            one_hot = np.zeros((n_samples, n_categories))

            for i, val in enumerate(col_data):
                if val in categories:
                    cat_idx = np.where(categories == val)[0][0]
                    one_hot[i, cat_idx] = 1
                elif self.handle_unknown == 'ignore':
                    continue
                else:
                    raise ValueError(f"Unknown category: {val}")

            encoded_arrays.append(one_hot)

        result = np.hstack(encoded_arrays)

        if self.sparse:
            # Simple sparse representation (not using scipy.sparse)
            return result

        return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation"""
        self._check_is_fitted()

        feature_names = []
        for col_idx, categories in enumerate(self.categories_):
            prefix = f"x{col_idx}" if input_features is None else input_features[col_idx]
            for cat in categories:
                feature_names.append(f"{prefix}_{cat}")

        return np.array(feature_names)


class OrdinalEncoder(BaseTransformer):
    """
    Encode categorical features as integer ordinal values

    Parameters:
    -----------
    categories : list of arrays, default='auto'
        Categories per feature
    handle_unknown : str, default='error'
        How to handle unknown categories ('error' or 'use_encoded_value')
    unknown_value : int, default=-1
        Value to use for unknown categories if handle_unknown='use_encoded_value'

    Example:
    --------
    >>> from mayini.preprocessing import OrdinalEncoder
    >>> oe = OrdinalEncoder()
    >>> X = np.array([['cold'], ['warm'], ['hot'], ['cold']]).reshape(-1, 1)
    >>> X_encoded = oe.fit_transform(X)
    """

    def __init__(self, categories='auto', handle_unknown='error', unknown_value=-1):
        super().__init__()
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """Fit ordinal encoder"""
        X, _ = self._validate_input(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        if self.categories == 'auto':
            self.categories_ = []
            for col_idx in range(X.shape[1]):
                unique_vals = np.unique(X[:, col_idx])
                self.categories_.append(unique_vals)
        else:
            self.categories_ = self.categories

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform data to ordinal encoding"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_encoded = np.zeros_like(X, dtype=float)

        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            col_data = X[:, col_idx]

            for i, val in enumerate(col_data):
                if val in categories:
                    # Encode as position in category list
                    encoded_val = np.where(categories == val)[0][0]
                    X_encoded[i, col_idx] = encoded_val
                else:
                    if self.handle_unknown == 'use_encoded_value':
                        X_encoded[i, col_idx] = self.unknown_value
                    elif self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category '{val}' in column {col_idx}")
                    else:
                        raise ValueError(f"Invalid handle_unknown: {self.handle_unknown}")

        return X_encoded

    def inverse_transform(self, X):
        """Transform ordinal encoding back to original"""
        self._check_is_fitted()

        if isinstance(X, list):
            X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_decoded = np.empty_like(X, dtype=object)

        for col_idx in range(X.shape[1]):
            categories = self.categories_[col_idx]
            col_data = X[:, col_idx]

            for i, encoded_val in enumerate(col_data):
                encoded_val = int(encoded_val)
                if 0 <= encoded_val < len(categories):
                    X_decoded[i, col_idx] = categories[encoded_val]
                else:
                    X_decoded[i, col_idx] = None

        return X_decoded
