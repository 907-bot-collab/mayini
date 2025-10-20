"""Pipeline for chaining transformers"""

class Pipeline:
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

    def fit(self, X, y=None):
        X_transformed = X
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)

        # Fit final step
        self.steps[-1].fit(X_transformed, y)
        return self

    def transform(self, X):
        X_transformed = X
        for name, step in self.steps:
            if hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
        return X_transformed

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.steps[-1].predict(X_transformed)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

