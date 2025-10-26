import numpy as np


class Pipeline:
    """
    Chain multiple transformers together

    Parameters
    ----------
    steps : list of tuples
        List of (name, transformer) tuples

    Example
    -------
    >>> from mayini.preprocessing import Pipeline, StandardScaler, PCA
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('pca', PCA(n_components=2))
    ... ])
    >>> X_transformed = pipeline.fit_transform(X)
    """

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        """Validate pipeline steps"""
        if not self.steps:
            raise ValueError("Pipeline steps cannot be empty")

        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique")

    def fit(self, X, y=None):
        """Fit all transformers"""
        X_transformed = X

        for name, transformer in self.steps:
            X_transformed = transformer.fit_transform(X_transformed, y)

        return self

    def transform(self, X):
        """Apply all transformers"""
        X_transformed = X

        for name, transformer in self.steps:
            X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

    def get_params(self):
        """Get parameters"""
        params = {}
        for name, transformer in self.steps:
            if hasattr(transformer, "get_params"):
                step_params = transformer.get_params()
                for key, value in step_params.items():
                    params[f"{name}__{key}"] = value
        return params

    def __repr__(self):
        """String representation"""
        steps_str = ", ".join([f"('{name}', {transformer.__class__.__name__})"
                               for name, transformer in self.steps])
        return f"Pipeline([{steps_str}])"
