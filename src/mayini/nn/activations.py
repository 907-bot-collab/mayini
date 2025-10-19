"""
Activation functions for MAYINI Deep Learning Framework.
"""

import numpy as np
from ..tensor import Tensor
from .modules import Module


# Functional activation functions
def relu(x: Tensor) -> Tensor:
    """ReLU activation function."""
    result_data = np.maximum(0, x.data)
    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = "ReLUBackward"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            grad = out.grad * (x.data > 0).astype(np.float32)
            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation function."""
    # Numerical stability: clip extreme values
    clipped_data = np.clip(x.data, -500, 500)
    result_data = 1.0 / (1.0 + np.exp(-clipped_data))

    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = "SigmoidBackward"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            # Derivative: sigmoid(x) * (1 - sigmoid(x))
            grad = out.grad * result_data * (1.0 - result_data)
            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


def tanh(x: Tensor) -> Tensor:
    """Tanh activation function."""
    result_data = np.tanh(x.data)
    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = "TanhBackward"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            # Derivative: 1 - tanh^2(x)
            grad = out.grad * (1.0 - result_data**2)
            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation function."""
    # Numerical stability: subtract max
    x_max = np.max(x.data, axis=dim, keepdims=True)
    x_shifted = x.data - x_max

    exp_data = np.exp(x_shifted)
    sum_exp = np.sum(exp_data, axis=dim, keepdims=True)
    result_data = exp_data / sum_exp

    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = f"SoftmaxBackward(dim={dim})"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            # Softmax gradient: softmax * (grad - (softmax * grad).sum())
            grad_sum = np.sum(out.grad * result_data, axis=dim, keepdims=True)
            grad = result_data * (out.grad - grad_sum)

            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


def gelu(x: Tensor) -> Tensor:
    """GELU activation function."""
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

    x_cubed = x.data**3
    inner = sqrt_2_over_pi * (x.data + 0.044715 * x_cubed)
    tanh_inner = np.tanh(inner)

    result_data = 0.5 * x.data * (1.0 + tanh_inner)

    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = "GELUBackward"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            # Approximate GELU derivative
            sech2_inner = 1.0 - tanh_inner**2
            derivative = (
                0.5 * (1.0 + tanh_inner)
                + 0.5
                * x.data
                * sech2_inner
                * sqrt_2_over_pi
                * (1.0 + 3 * 0.044715 * x.data**2)
            )

            grad = out.grad * derivative
            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Leaky ReLU activation function."""
    result_data = np.where(x.data > 0, x.data, negative_slope * x.data)

    out = Tensor(result_data, requires_grad=x.requires_grad)
    out.op = f"LeakyReLUBackward(slope={negative_slope})"
    out.is_leaf = False

    def _backward():
        if x.requires_grad and out.grad is not None:
            grad = np.where(x.data > 0, out.grad, negative_slope * out.grad)
            if x.grad is None:
                x.grad = grad
            else:
                x.grad = x.grad + grad

    out._backward = _backward
    out.prev = {x}
    return out


# Module-based activation functions
class ReLU(Module):
    """ReLU activation module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """Softmax activation module."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, dim=self.dim)

    def __repr__(self):
        return f"Softmax(dim={self.dim})"


class GELU(Module):
    """GELU activation module."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return gelu(x)

    def __repr__(self):
        return "GELU()"


class LeakyReLU(Module):
    """Leaky ReLU activation module."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return leaky_relu(x, self.negative_slope)

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"
