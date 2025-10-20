"""Automated preprocessing detection and application"""
import numpy as np
from .numerical.scalers import StandardScaler
from .categorical.encoders import LabelEncoder

class AutoPreprocessor:
    def __init__(self):
        self.column_types_ = None
        self.transformers_ = {}
        self.is_fitted_ = False

    def detect_column_types(self, X):
        column_types = []

        for col in range(X.shape):
            column = X[:, col]

            try:
                # Try to convert to numeric
                numeric_col = column.astype(float)
                column_types.append('numerical')
            except:
                # Categorical if conversion fails
                column_types.append('categorical')

        return column_types

    def fit(self, X, y=None):
        X = np.asarray(X)

        # Detect column types
        self.column_types_ = self.detect_column_types(X)

        # Fit appropriate transformers
        for col, col_type in enumerate(self.column_types_):
            if col_type == 'numerical':
                scaler = StandardScaler()
                scaler.fit(X[:, col].reshape(-1, 1))
                self.transformers_[col] = scaler
            elif col_type == 'categorical':
                encoder = LabelEncoder()
                encoder.fit(X[:, col])
                self.transformers_[col] = encoder

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.is_fitted_:
            raise RuntimeError("AutoPreprocessor must be fitted first")

        X = np.asarray(X)
        transformed_cols = []

        for col in range(X.shape):
            if col in self.transformers_:
                transformed = self.transformers_[col].transform(X[:, col].reshape(-1, 1))
                transformed_cols.append(transformed.flatten())
            else:
                transformed_cols.append(X[:, col])

        return np.column_stack(transformed_cols)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

