import numpy as np
from .base import BaseTransformer
from .numerical.scalers import StandardScaler
from .numerical.imputers import SimpleImputer


class AutoPreprocessor(BaseTransformer):
    """
    Automatically preprocess data

    Applies common preprocessing steps automatically

    Parameters
    ----------
    scaling : bool, default=True
        Whether to apply scaling
    imputation : bool, default=True
        Whether to impute missing values
    imputation_strategy : str, default='mean'
        Imputation strategy

    Example
    -------
    >>> from mayini.preprocessing import AutoPreprocessor
    >>> auto = AutoPreprocessor()
    >>> X_preprocessed = auto.fit_transform(X)
    """

    def __init__(self, scaling=True, imputation=True, imputation_strategy="mean"):
        super().__init__()
        self.scaling = scaling
        self.imputation = imputation
        self.imputation_strategy = imputation_strategy
        self.imputer_ = None
        self.scaler_ = None
        self.steps_ = []

    def fit(self, X, y=None):
        """Fit preprocessing steps"""
        X, _ = self._validate_input(X)

        self.steps_ = []

        # Check for missing values
        if self.imputation and np.any(np.isnan(X)):
            self.imputer_ = SimpleImputer(strategy=self.imputation_strategy)
            X = self.imputer_.fit_transform(X)
            self.steps_.append(("imputation", self.imputer_))

        # Apply scaling
        if self.scaling:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
            self.steps_.append(("scaling", self.scaler_))

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply preprocessing steps"""
        self._check_is_fitted()
        X, _ = self._validate_input(X)

        for name, transformer in self.steps_:
            X = transformer.transform(X)

        return X

    def get_steps(self):
        """Get applied preprocessing steps"""
        self._check_is_fitted()
        return [(name, type(transformer).__name__) for name, transformer in self.steps_]
