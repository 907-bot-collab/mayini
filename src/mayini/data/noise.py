"""
mayini.data.noise — Label-noise detection and filtering.

Implements small-loss selection (co-teaching-style) and per-sample
loss clipping to reduce the influence of mislabelled examples.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class LabelNoiseHandler:
    """Detect and mitigate label noise during training.

    Parameters
    ----------
    forget_rate:
        Fraction of samples to drop per epoch (treated as noisy).
        Default 0.2 (20 %).
    loss_clip:
        If set, clamp per-sample loss to this maximum value to limit
        the gradient influence of severely mislabelled samples.
    warm_up_epochs:
        Epochs before noise filtering starts (model needs time to fit
        clean samples first).

    Example
    -------
    >>> handler = LabelNoiseHandler(forget_rate=0.2)
    >>> # During training:
    >>> clean_indices = handler.select_clean(per_sample_losses, epoch=ep)
    >>> X_clean, y_clean = X[clean_indices], y[clean_indices]
    """

    def __init__(
        self,
        forget_rate: float = 0.2,
        loss_clip: Optional[float] = None,
        warm_up_epochs: int = 5,
    ) -> None:
        if not 0.0 <= forget_rate < 1.0:
            raise ValueError("forget_rate must be in [0, 1)")
        self.forget_rate = forget_rate
        self.loss_clip = loss_clip
        self.warm_up_epochs = warm_up_epochs
        self._loss_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_clean(
        self,
        per_sample_losses: np.ndarray,
        epoch: int = 0,
    ) -> np.ndarray:
        """Return indices of samples considered cleanly labelled.

        Uses the small-loss trick: low-loss samples are more likely to
        have correct labels.

        Parameters
        ----------
        per_sample_losses:
            1-D array of per-sample scalar losses (length = N).
        epoch:
            Current training epoch (noise filtering begins after
            *warm_up_epochs*).

        Returns
        -------
        np.ndarray
            Integer indices of the retained (clean) samples.
        """
        losses = np.asarray(per_sample_losses, dtype=np.float64)
        self._loss_history.append(losses.copy())

        if epoch < self.warm_up_epochs:
            return np.arange(len(losses))  # use all during warm-up

        n_keep = max(1, int(len(losses) * (1.0 - self.forget_rate)))
        sorted_idx = np.argsort(losses)         # ascending loss
        return sorted_idx[:n_keep]

    def clip_losses(self, per_sample_losses: np.ndarray) -> np.ndarray:
        """Clip per-sample losses to *loss_clip* (in-place safe copy).

        Parameters
        ----------
        per_sample_losses:
            1-D loss array.

        Returns
        -------
        np.ndarray
            Clipped loss array.
        """
        losses = np.asarray(per_sample_losses, dtype=np.float64)
        if self.loss_clip is not None:
            losses = np.clip(losses, a_min=None, a_max=self.loss_clip)
        return losses

    def noise_estimate(self) -> float:
        """Estimate empirical noise rate from loss history.

        Uses the assumption that samples with persistently high loss
        across epochs are likely mislabelled.

        Returns
        -------
        float
            Estimated noise fraction in [0, 1].
        """
        if len(self._loss_history) < 2:
            return 0.0
        stacked = np.stack(self._loss_history, axis=0)   # (epochs, N)
        mean_losses = stacked.mean(axis=0)
        threshold = np.percentile(mean_losses, 100 * (1 - self.forget_rate))
        noisy_mask = mean_losses > threshold
        return float(noisy_mask.mean())

    def summary(self) -> None:
        """Print handler configuration."""
        print(f"LabelNoiseHandler — forget_rate={self.forget_rate}"
              f"  loss_clip={self.loss_clip}"
              f"  warm_up={self.warm_up_epochs} epochs")
        if self._loss_history:
            print(f"  Epochs observed : {len(self._loss_history)}")
            print(f"  Estimated noise : {self.noise_estimate()*100:.1f}%")
