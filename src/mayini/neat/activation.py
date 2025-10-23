import numpy as np


class ActivationFunctions:
    """Collection of activation functions"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent activation"""
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        """Rectified Linear Unit"""
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def identity(x):
        """Linear/identity activation"""
        return x
    
    @staticmethod
    def step(x):
        """Step function"""
        return np.where(x > 0, 1.0, 0.0)
    
    @staticmethod
    def gaussian(x):
        """Gaussian activation"""
        return np.exp(-x**2)
    
    @staticmethod
    def sin(x):
        """Sinusoidal activation"""
        return np.sin(x)
    
    @staticmethod
    def abs(x):
        """Absolute value"""
        return np.abs(x)
    
    @staticmethod
    def square(x):
        """Square activation"""
        return x**2
    
    @staticmethod
    def get_function(name):
        """Get activation function by name"""
        functions = {
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': ActivationFunctions.leaky_relu,
            'identity': ActivationFunctions.identity,
            'linear': ActivationFunctions.identity,
            'step': ActivationFunctions.step,
            'gaussian': ActivationFunctions.gaussian,
            'sin': ActivationFunctions.sin,
            'abs': ActivationFunctions.abs,
            'square': ActivationFunctions.square,
        }
        return functions.get(name, ActivationFunctions.sigmoid)
