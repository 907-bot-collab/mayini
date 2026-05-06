"""
mayini.explain.gradients — Gradient-based attribution methods.

Implements:
  - Integrated Gradients
  - FGSM sensitivity (gradient sign w.r.t. input)
  - GradientExplainer (higher-level API)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def integrated_gradients(
    model: Any,
    input_tensor: Any,
    target_class: Optional[int] = None,
    baseline: Optional[Any] = None,
    steps: int = 50,
) -> np.ndarray:
    """Compute Integrated Gradients attribution for *input_tensor*.

    IG approximates:
        IG_i(x) = (x_i - x'_i) * ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i dα

    Parameters
    ----------
    model:
        Callable with ``forward(x)`` or ``__call__(x)``.
    input_tensor:
        Mayini Tensor of shape ``(1, *features)`` or ``(*features,)``.
    target_class:
        Index of the output neuron to attribute. If ``None``, uses the
        scalar output directly.
    baseline:
        Reference Tensor (same shape as *input_tensor*). Defaults to zeros.
    steps:
        Number of interpolation steps (higher → more accurate).

    Returns
    -------
    np.ndarray
        Attribution array with the same shape as *input_tensor.data*.
    """
    # Import lazily to avoid circular imports
    from mayini.tensor import Tensor  # type: ignore[import]

    x = input_tensor.data.astype(np.float64)
    if baseline is None:
        x_base = np.zeros_like(x)
    else:
        x_base = baseline.data.astype(np.float64)

    alphas = np.linspace(0.0, 1.0, steps)
    gradients: list = []

    for alpha in alphas:
        interp_data = x_base + alpha * (x - x_base)
        interp = Tensor(interp_data.astype(np.float32), requires_grad=True)
        output = model(interp)

        # Select scalar to differentiate
        if target_class is not None and output.data.ndim > 0:
            scalar = output[target_class]
        else:
            scalar = output.sum()

        scalar.backward()
        grad = interp.grad
        if grad is not None:
            gradients.append(grad.copy())
        else:
            gradients.append(np.zeros_like(x))

    avg_grad = np.mean(np.stack(gradients, axis=0), axis=0)
    attributions = avg_grad * (x - x_base)
    return attributions


def fgsm_sensitivity(
    model: Any,
    input_tensor: Any,
    loss_fn: Callable,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Compute FGSM perturbation sign map — a fast sensitivity proxy.

    Parameters
    ----------
    model:
        Callable model.
    input_tensor:
        Input Mayini Tensor with ``requires_grad=True``.
    loss_fn:
        Loss function ``loss_fn(output, target)`` → scalar Tensor.
    epsilon:
        Perturbation magnitude (only used to scale the output).

    Returns
    -------
    np.ndarray
        Sign gradient array (same shape as input) scaled by *epsilon*.
    """
    from mayini.tensor import Tensor  # type: ignore[import]

    x = Tensor(input_tensor.data.copy(), requires_grad=True)
    output = model(x)
    # Use output sum as surrogate if no label available
    loss = output.sum()
    loss.backward()

    if x.grad is None:
        return np.zeros_like(input_tensor.data)

    return epsilon * np.sign(x.grad)


# ---------------------------------------------------------------------------
# High-level class API
# ---------------------------------------------------------------------------

class GradientExplainer:
    """High-level explainer that wraps multiple attribution methods.

    Parameters
    ----------
    model:
        A trained Mayini model (callable).

    Example
    -------
    >>> explainer = GradientExplainer(model)
    >>> attrs = explainer.explain(x, method="integrated_gradients", steps=50)
    """

    METHODS = ("integrated_gradients", "gradient", "fgsm")

    def __init__(self, model: Any) -> None:
        self.model = model
        self._hook_log: Dict[str, list] = {}

    def explain(
        self,
        input_tensor: Any,
        method: str = "integrated_gradients",
        target_class: Optional[int] = None,
        steps: int = 50,
        epsilon: float = 0.01,
        baseline: Optional[Any] = None,
    ) -> np.ndarray:
        """Generate attribution for *input_tensor*.

        Parameters
        ----------
        method:
            One of ``"integrated_gradients"``, ``"gradient"``,
            ``"fgsm"``.
        target_class:
            Output index to attribute (None → scalar sum).
        steps:
            Integration steps (for integrated_gradients).
        epsilon:
            Perturbation scale (for fgsm).
        baseline:
            Reference input (for integrated_gradients).

        Returns
        -------
        np.ndarray
            Attribution map (same shape as input data).
        """
        if method == "integrated_gradients":
            return integrated_gradients(
                self.model,
                input_tensor,
                target_class=target_class,
                baseline=baseline,
                steps=steps,
            )
        elif method == "gradient":
            return self._plain_gradient(input_tensor, target_class)
        elif method == "fgsm":
            return fgsm_sensitivity(
                self.model, input_tensor, loss_fn=lambda o, _: o.sum(), epsilon=epsilon
            )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from {self.METHODS}."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _plain_gradient(
        self, input_tensor: Any, target_class: Optional[int]
    ) -> np.ndarray:
        from mayini.tensor import Tensor  # type: ignore[import]

        x = Tensor(input_tensor.data.copy(), requires_grad=True)
        output = self.model(x)

        if target_class is not None and output.data.ndim > 0:
            scalar = output[target_class]
        else:
            scalar = output.sum()

        scalar.backward()
        return x.grad if x.grad is not None else np.zeros_like(input_tensor.data)
