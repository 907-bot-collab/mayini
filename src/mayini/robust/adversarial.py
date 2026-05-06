"""
mayini.robust.adversarial — Adversarial attack utilities.

Implements:
  - FGSM  : Fast Gradient Sign Method (single step)
  - PGD   : Projected Gradient Descent (multi-step, stronger)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]


def fgsm_attack(
    model: Any,
    x: Any,
    y: Any,
    loss_fn: Callable,
    epsilon: float = 0.03,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> np.ndarray:
    """Fast Gradient Sign Method adversarial perturbation.

    x_adv = clip(x + ε · sign(∇_x L(f(x), y)), clip_min, clip_max)

    Parameters
    ----------
    model:
        Callable Mayini model.
    x:
        Input tensor (Mayini Tensor or numpy array).
    y:
        Target labels (Mayini Tensor or numpy array).
    loss_fn:
        ``loss_fn(output, y) -> scalar Tensor``.
    epsilon:
        Perturbation magnitude (default 0.03, roughly 8/255 for images).
    clip_min, clip_max:
        Valid input range for clipping.

    Returns
    -------
    np.ndarray
        Adversarial example ``x_adv`` (same shape as x).
    """
    if isinstance(x, np.ndarray):
        x_data = x.astype(np.float32, copy=False)
    elif hasattr(x, "data"):
        x_data = np.asarray(x.data, dtype=np.float32)
    else:
        x_data = np.asarray(x, np.float32)
    x_t = Tensor(x_data.copy(), requires_grad=True)

    output = model(x_t)
    loss = loss_fn(output, y)
    loss.backward()

    if x_t.grad is None:
        return x_data

    x_adv = x_data + epsilon * np.sign(x_t.grad)
    x_adv = np.clip(x_adv, clip_min, clip_max)
    return x_adv.astype(np.float32)


def pgd_attack(
    model: Any,
    x: Any,
    y: Any,
    loss_fn: Callable,
    epsilon: float = 0.03,
    step_size: float = 0.007,
    num_steps: int = 10,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
) -> np.ndarray:
    """Projected Gradient Descent (Madry et al.) adversarial attack.

    Iterates FGSM steps with projection back into the ε-ball around x.

    Parameters
    ----------
    epsilon:
        L∞ perturbation budget.
    step_size:
        Per-step perturbation size (default ε/4).
    num_steps:
        Number of PGD iterations.
    random_start:
        Start from a random point inside the ε-ball (recommended).

    Returns
    -------
    np.ndarray
        Adversarial example ``x_adv`` (same shape as x).
    """
    x_data = x.data if hasattr(x, "data") else np.asarray(x, np.float32)
    x_orig = x_data.copy()

    if random_start:
        delta = np.random.uniform(-epsilon, epsilon, x_data.shape).astype(np.float32)
        x_adv = np.clip(x_data + delta, clip_min, clip_max)
    else:
        x_adv = x_data.copy()

    for _ in range(num_steps):
        x_t = Tensor(x_adv.copy(), requires_grad=True)
        output = model(x_t)
        loss = loss_fn(output, y)
        loss.backward()

        if x_t.grad is None:
            break

        # Gradient ascent step
        x_adv = x_adv + step_size * np.sign(x_t.grad)

        # Project back into ε-ball and valid range
        x_adv = x_orig + np.clip(x_adv - x_orig, -epsilon, epsilon)
        x_adv = np.clip(x_adv, clip_min, clip_max).astype(np.float32)

    return x_adv
