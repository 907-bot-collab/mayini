import numpy as np
from .base import BaseTransformer


class TypeConverter(BaseTransformer):
    """
    Convert data types

    Parameters
    ----------
    dtype : str or dtype, default='float32'
        Target data type

    Example
    -------
    >>> from mayini.preprocessing import TypeConverter
    >>> converter = TypeConverter(dtype='float32')
    >>> X = [[1, 2], [3, 4]]
    >>> X_converted = converter.fit_transform(X)
    """

    def __init__(self, dtype="float32"):
        super().__init__()
        self.dtype = dtype

    def fit(self, X, y=None):
        """Type converter doesn't need fitting"""
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Convert data type"""
        self._check_is_fitted()
        X = np.asarray(X)
        return X.astype(self.dtype)
