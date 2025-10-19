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
            batch_size = targets.data.shape
            num_classes = predictions.data.shape
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
        # Subtract max for numerical stability
        x_max = Tensor(np.max(x.data, axis=1, keepdims=True))
        x_shifted = x - x_max

        # Compute exp
        exp_data = np.exp(x_shifted.data)
        exp_tensor = Tensor(exp_data, requires_grad=x.requires_grad)
        exp_tensor.op = "ExpBackward"
        exp_tensor.is_leaf = False

        def exp_backward():
            if x_shifted.requires_grad and exp_tensor.grad is not None:
                grad = exp_tensor.grad * exp_data
                if x_shifted.grad is None:
                    x_shifted.grad = grad
                else:
                    x_shifted.grad = x_shifted.grad + grad

        exp_tensor._backward = exp_backward
        exp_tensor.prev = {x_shifted}

        # Compute log sum exp
        sum_exp = exp_tensor.sum(axis=1, keepdims=True)
        log_sum_exp_data = np.log(sum_exp.data)
        log_sum_exp = Tensor(log_sum_exp_data, requires_grad=sum_exp.requires_grad)
        log_sum_exp.op = "LogBackward"
        log_sum_exp.is_leaf = False

        def log_backward():
            if sum_exp.requires_grad and log_sum_exp.grad is not None:
                grad = log_sum_exp.grad / sum_exp.data
                if sum_exp.grad is None:
                    sum_exp.grad = grad
                else:
                    sum_exp.grad = sum_exp.grad + grad

        log_sum_exp._backward = log_backward
        log_sum_exp.prev = {sum_exp}

        return x_shifted - log_sum_exp

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

        # Clamp predictions for numerical stability
        eps = 1e-7
        pred_clamped_data = np.clip(predictions.data, eps, 1 - eps)
        pred_clamped = Tensor(
            pred_clamped_data, requires_grad=predictions.requires_grad
        )

        # Compute BCE: -[y*log(p) + (1-y)*log(1-p)]
        log_pred_data = np.log(pred_clamped_data)
        log_pred = Tensor(log_pred_data, requires_grad=pred_clamped.requires_grad)

        log_one_minus_pred_data = np.log(1 - pred_clamped_data)
        log_one_minus_pred = Tensor(
            log_one_minus_pred_data, requires_grad=pred_clamped.requires_grad
        )

        # BCE formula
        bce = -(targets * log_pred + (Tensor(1.0) - targets) * log_one_minus_pred)

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
