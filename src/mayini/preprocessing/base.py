import numpy as np
from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
    Base class for all transformers

    All transformers should inherit from this class
    """

    def __init__(self):
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit transformer

        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target values

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Transform data

        Parameters
        ----------
        X : array-like
            Input data

        Returns
        -------
        array-like
            Transformed data
        """
        pass

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step

        Parameters
        ----------
        X : array-like
            Input data
        y : array-like, optional
            Target values

        Returns
        -------
        array-like
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def _check_is_fitted(self):
        """Check if transformer has been fitted"""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before transforming. "
                "Call fit() first."
            )

    def _validate_input(self, X, y=None):
        """
        Validate input data

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like, optional
            Target values

        Returns
        -------
        X : np.ndarray
            Validated features
        y : np.ndarray or None
            Validated targets
        """
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            y = np.asarray(y)
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {y.shape[0]}"
                )

        return X, y
