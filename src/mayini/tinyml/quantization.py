"""
mayini.tinyml.quantization — Post-training quantisation.

Supports 4-bit and 8-bit affine (asymmetric) quantisation, both
per-tensor and per-channel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class Quantizer:
    """Quantise model weights using affine (asymmetric) quantisation.

    Parameters
    ----------
    model:
        A Mayini model (has ``named_parameters()`` / ``parameters()``).
    bits:
        Bit-width for quantisation (default 8). Common: 4, 8.
    per_channel:
        If ``True``, compute scale/zero_point per output channel of
        weight matrices; otherwise per-tensor.
    symmetric:
        If ``True``, use symmetric quantisation (zero_point = 0).

    Example
    -------
    >>> quantizer = Quantizer(model, bits=8)
    >>> q_model = quantizer.quantize()
    >>> quantizer.print_stats()
    """

    def __init__(
        self,
        model: Any,
        bits: int = 8,
        per_channel: bool = False,
        symmetric: bool = False,
    ) -> None:
        self.model = model
        self.bits = bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.scales: Dict[str, np.ndarray] = {}
        self.zero_points: Dict[str, np.ndarray] = {}
        self._original_shapes: Dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, calibration_data=None) -> Any:
        """Quantise model parameters in-place and return the model.

        Parameters
        ----------
        calibration_data:
            Optional list of input tensors for calibration-based
            quantisation (not yet fully implemented; reserved for
            activation quantisation).
        """
        params = list(self._iter_params())
        for name, param in params:
            self._original_shapes[name] = param.data.shape
            q_data, scale, zp = self._quantize_array(param.data)
            self.scales[name] = scale
            self.zero_points[name] = zp
            param.data = q_data
            param.requires_grad = False

        return self.model

    def dequantize(self) -> None:
        """Dequantise all parameters back to float32 (for inspection)."""
        for name, param in self._iter_params():
            if name in self.scales:
                scale = self.scales[name]
                zp = self.zero_points[name]
                if self.symmetric:
                    param.data = (param.data * scale).astype(np.float32)
                else:
                    param.data = ((param.data.astype(np.float32) - zp) * scale).astype(
                        np.float32
                    )

    def print_stats(self) -> None:
        """Print quantisation statistics (scale, zero_point per layer)."""
        print(f"Quantisation: {self.bits}-bit, per_channel={self.per_channel}")
        print(f"{'Layer':<40} {'Scale':>12} {'ZeroPoint':>12}")
        print("-" * 66)
        for name in self.scales:
            s = float(np.mean(self.scales[name]))
            zp = float(np.mean(self.zero_points[name]))
            print(f"{name:<40} {s:>12.6f} {zp:>12.2f}")

    def memory_savings(self) -> float:
        """Return estimated memory savings factor vs float32 baseline."""
        return 32.0 / self.bits

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_params(self):
        """Yield (name, param) for all model parameters."""
        try:
            yield from self.model.named_parameters()
        except AttributeError:
            for i, p in enumerate(self.model.parameters()):
                yield (f"param_{i}", p)

    def _quantize_array(
        self, arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantise a numpy array. Returns (quantized, scale, zero_point)."""
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1

        arr_f = arr.astype(np.float64)

        if self.per_channel and arr_f.ndim >= 2:
            # Per output-channel (axis 0)
            ch_min = arr_f.min(axis=tuple(range(1, arr_f.ndim)), keepdims=True)
            ch_max = arr_f.max(axis=tuple(range(1, arr_f.ndim)), keepdims=True)
        else:
            ch_min = np.array([[arr_f.min()]])
            ch_max = np.array([[arr_f.max()]])

        if self.symmetric:
            abs_max = np.maximum(np.abs(ch_min), np.abs(ch_max))
            abs_max = np.where(abs_max == 0, 1.0, abs_max)
            scale = abs_max / qmax
            zero_point = np.zeros_like(scale)
        else:
            scale = (ch_max - ch_min) / (qmax - qmin)
            scale = np.where(scale == 0, 1.0, scale)
            zero_point = qmin - ch_min / scale

        q = np.round(arr_f / scale + zero_point).clip(qmin, qmax).astype(np.int8)
        return q, scale.astype(np.float32), zero_point.astype(np.float32)
