import numpy as np


class ActivationFunctions:
    """
    Collection of activation functions for NEAT

    Provides various activation functions used in neural networks

    Example
    -------
    >>> from mayini.neat.activation import ActivationFunctions
    >>> af = ActivationFunctions()
    >>> output = af.sigmoid(0.5)
    """

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function

        Parameters
        ----------
        x : float or array-like
            Input value(s)

        Returns
        -------
        float or array-like
            Sigmoid of input
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation function

        Parameters
        ----------
        x : float or array-like
            Input value(s)

        Returns
        -------
        float or array-like
            Tanh of input
        """
        return np.tanh(x)

    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit activation function

        Parameters
        ----------
        x : float or array-like
            Input value(s)

        Returns
        -------
        float or array-like
            ReLU of input
        """
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU activation function

        Parameters
        ----------
        x : float or array-like
            Input value(s)
        alpha : float, default=0.01
            Slope for negative values

        Returns
        -------
        float or array-like
            Leaky ReLU of input
        """
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def linear(x):
        """
        Linear activation function (identity)

        Parameters
        ----------
        x : float or array-like
            Input value(s)

        Returns
        -------
        float or array-like
            Input unchanged
        """
        return x

    @staticmethod
    def softplus(x):
        """
        Softplus activation function

        Parameters
        ----------
        x : float or array-like
            Input value(s)

        Returns
        -------
        float or array-like
            Softplus of input
        """
        return np.log(1 + np.exp(np.clip(x, -500, 500)))

    @staticmethod
    def get_activation(name):
        """
        Get activation function by name

        Parameters
        ----------
        name : str
            Name of activation function

        Returns
        -------
        callable
            Activation function

        Raises
        ------
        ValueError
            If activation function name is unknown
        """
        activations = {
            "sigmoid": ActivationFunctions.sigmoid,
            "tanh": ActivationFunctions.tanh,
            "relu": ActivationFunctions.relu,
            "leaky_relu": ActivationFunctions.leaky_relu,
            "linear": ActivationFunctions.linear,
            "softplus": ActivationFunctions.softplus,
        }

        if name not in activations:
            raise ValueError(
                f"Unknown activation function: {name}. "
                f"Available: {list(activations.keys())}"
            )

        return activations[name]
