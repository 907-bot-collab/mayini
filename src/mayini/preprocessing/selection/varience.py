import numpy as np


class VarianceThreshold:
    """
    Feature selector that removes all low-variance features.

    Features with a variance lower than the threshold will be removed.
    The default is to keep all features with non-zero variance.

    Parameters
    ----------
    threshold : float, default=0.0
        Features with a variance lower than this threshold will be removed.

    Attributes
    ----------
    variances_ : array, shape (n_features,)
        Variances of individual features.
    """

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None
        self._selected_features = None

    def fit(self, X, y=None):
        """
        Learn empirical variances from X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any, optional
            Ignored. This parameter exists only for compatibility.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        # Calculate variance for each feature
        self.variances_ = np.var(X, axis=0)

        # Determine which features to keep
        self._selected_features = self.variances_ > self.threshold

        return self

    def transform(self, X):
        """
        Reduce X to the selected features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        if self.variances_ is None:
            raise ValueError("VarianceThreshold has not been fitted yet. Call fit() first.")

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[1] != len(self.variances_):
            raise ValueError(
                f"X has {X.shape[1]} features, but VarianceThreshold "
                f"was fitted with {len(self.variances_)} features"
            )

        return X[:, self._selected_features]

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Sample vectors from which to compute variances.
        y : any, optional
            Ignored. This parameter exists only for compatibility.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, return an integer array of selected feature indices.
            Otherwise, return a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If indices is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention.
            If indices is True, this is an integer array of shape
            [# output features] whose values are indices into the input
            feature vector.
        """
        if self._selected_features is None:
            raise ValueError("VarianceThreshold has not been fitted yet. Call fit() first.")

        if indices:
            return np.where(self._selected_features)[0]
        else:
            return self._selected_features
