import numpy as np
import pandas as pd
from .base import BaseTransformer


class DataTypeConverter(BaseTransformer):
    """
    Automatically infer and convert data types

    Converts:
    - Numeric strings to float/int
    - Detects categorical vs numerical features

    Example:
    --------
    >>> from mayini.preprocessing import DataTypeConverter
    >>> converter = DataTypeConverter()
    >>> # Works with pandas DataFrames
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['cat', 'dog', 'cat']})
    >>> converter.fit_transform(df)
    """

    def __init__(self):
        super().__init__()
        self.type_mapping_ = {}

    def fit(self, X, y=None):
        """Infer data types"""
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col].astype(float)
                        self.type_mapping_[col] = 'numeric'
                    except:
                        self.type_mapping_[col] = 'categorical'
                else:
                    self.type_mapping_[col] = 'numeric'

        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            for i in range(X.shape[1]):
                try:
                    X[:, i].astype(float)
                    self.type_mapping_[i] = 'numeric'
                except:
                    self.type_mapping_[i] = 'categorical'

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Convert data types"""
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()

            for col, dtype in self.type_mapping_.items():
                if col in X_copy.columns and dtype == 'numeric':
                    X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')

            return X_copy

        elif isinstance(X, np.ndarray):
            X_copy = X.copy()

            if X_copy.ndim == 1:
                X_copy = X_copy.reshape(-1, 1)

            for i, dtype in self.type_mapping_.items():
                if dtype == 'numeric':
                    try:
                        X_copy[:, i] = X_copy[:, i].astype(float)
                    except:
                        pass

            return X_copy

        else:
            return X
