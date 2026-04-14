"""
Loss functions for MAYINI Deep Learning Framework.
"""

import numpy as np
from ..tensor import Tensor
from .modules import Module


class MSELoss(Module):
    """Mean Squared Error Loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MSE loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Compute squared differences
        diff = predictions - targets
        squared_diff = diff * diff

        # Apply reduction
        if self.reduction == "mean":
            return squared_diff.mean()
        elif self.reduction == "sum":
            return squared_diff.sum()
        else:  # 'none'
            return squared_diff

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"


class MAELoss(Module):
    """Mean Absolute Error Loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute MAE loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Compute absolute differences
        diff = predictions - targets
        abs_diff_data = np.abs(diff.data)
        abs_diff = Tensor(abs_diff_data, requires_grad=diff.requires_grad)
        abs_diff.op = "AbsBackward"
        abs_diff.is_leaf = False

        def _backward():
            if diff.requires_grad and abs_diff.grad is not None:
                # Gradient of abs(x) is sign(x)
                sign_data = np.sign(diff.data)
                grad = abs_diff.grad * sign_data

                if diff.grad is None:
                    diff.grad = grad
                else:
                    diff.grad = diff.grad + grad

        abs_diff._backward = _backward
        abs_diff.prev = {diff}

        # Apply reduction
        if self.reduction == "mean":
            return abs_diff.mean()
        elif self.reduction == "sum":
            return abs_diff.sum()
        else:  # 'none'
            return abs_diff

    def __repr__(self):
        return f"MAELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """Cross Entropy Loss for classification."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute cross entropy loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Apply log softmax for numerical stability
        log_probs = self._log_softmax(predictions)

        # Handle different target formats
        if targets.data.ndim == 1:  # Class indices
            # Convert to one-hot
            batch_size = targets.data.shape[0]
            num_classes = predictions.data.shape[1]
            targets_one_hot = np.zeros((batch_size, num_classes))
            targets_one_hot[np.arange(batch_size), targets.data.astype(int)] = 1
            targets = Tensor(targets_one_hot)

        # Compute negative log likelihood
        nll = -(log_probs * targets).sum(axis=1)

        # Apply reduction
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:  # 'none'
            return nll

    def _log_softmax(self, x: Tensor) -> Tensor:
        """Compute log softmax with numerical stability."""
        x_max = Tensor(np.max(x.data, axis=1, keepdims=True))
        x_shifted = x - x_max
        return x_shifted - x_shifted.exp().sum(axis=1, keepdims=True).log()

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"


class BCELoss(Module):
    """Binary Cross Entropy Loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute BCE loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Compute BCE: -[y*log(p) + (1-y)*log(1-p)]
        bce = -(targets * predictions.log() + (Tensor(1.0) - targets) * (Tensor(1.0) - predictions).log())

        # Apply reduction
        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        else:  # 'none'
            return bce

    def __repr__(self):
        return f"BCELoss(reduction='{self.reduction}')"


class HuberLoss(Module):
    """Huber Loss (smooth L1 loss)."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.delta = delta
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute Huber loss."""
        if not isinstance(targets, Tensor):
            targets = Tensor(targets)

        # Compute absolute difference
        diff = predictions - targets
        abs_diff_data = np.abs(diff.data)

        # Huber loss formula
        mask = abs_diff_data <= self.delta

        # Quadratic part: 0.5 * diff^2 for |diff| <= delta
        quadratic = 0.5 * diff * diff

        # Linear part: delta * (|diff| - 0.5 * delta) for |diff| > delta
        abs_diff = Tensor(abs_diff_data, requires_grad=diff.requires_grad)
        linear = self.delta * (abs_diff - 0.5 * self.delta)

        # Combine using mask
        loss_data = np.where(mask, quadratic.data, linear.data)
        loss = Tensor(
            loss_data,
            requires_grad=(predictions.requires_grad or targets.requires_grad),
        )

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

    def __repr__(self):
        return f"HuberLoss(delta={self.delta}, reduction='{self.reduction}')"
