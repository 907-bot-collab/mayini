"""
mayini.multimodal.fusion — Cross-attention multimodal fusion layer.

Implements a bidirectional cross-attention mechanism that fuses
two modality feature streams into a single joint representation,
all using Mayini's NumPy autograd engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]
from mayini.nn.modules import Module, Linear  # type: ignore[import]


class CrossAttentionFusion(Module):
    """Bidirectional cross-attention for two modality streams.

    Parameters
    ----------
    dim_a:
        Feature dimension of modality A.
    dim_b:
        Feature dimension of modality B.
    hidden_dim:
        Projected dimension for attention computation.
    num_heads:
        Number of attention heads (must divide *hidden_dim*).

    Example
    -------
    >>> fusion = CrossAttentionFusion(dim_a=512, dim_b=256, hidden_dim=256)
    >>> fused = fusion(text_features, image_features)  # shape (B, 256)
    """

    def __init__(
        self,
        dim_a: int = 512,
        dim_b: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V from each modality
        self.proj_a = Linear(dim_a, hidden_dim)
        self.proj_b = Linear(dim_b, hidden_dim)

        # Output projections
        self.out_a = Linear(hidden_dim, hidden_dim)
        self.out_b = Linear(hidden_dim, hidden_dim)
        self.final = Linear(hidden_dim * 2, hidden_dim)

    def parameters(self):
        return (
            list(self.proj_a.parameters())
            + list(self.proj_b.parameters())
            + list(self.out_a.parameters())
            + list(self.out_b.parameters())
            + list(self.final.parameters())
        )

    def forward(self, feat_a: Any, feat_b: Any) -> Any:
        """Fuse two feature tensors via bidirectional cross-attention.

        Parameters
        ----------
        feat_a:
            Tensor of shape ``(B, dim_a)``.
        feat_b:
            Tensor of shape ``(B, dim_b)``.

        Returns
        -------
        Tensor
            Fused representation of shape ``(B, hidden_dim)``.
        """
        # Project to hidden space
        ha = self.proj_a(feat_a)   # (B, H)
        hb = self.proj_b(feat_b)   # (B, H)

        # Scaled dot-product attention A→B and B→A (single-head approx)
        scale = np.sqrt(self.hidden_dim)

        # A attends to B
        scores_ab = ha.matmul(hb.transpose())   # (B, B)
        scores_ab = scores_ab * (1.0 / scale)
        attn_ab = self._softmax_rows(scores_ab)  # (B, B)
        attended_a = attn_ab.matmul(hb)          # (B, H)

        # B attends to A
        scores_ba = hb.matmul(ha.transpose())
        scores_ba = scores_ba * (1.0 / scale)
        attn_ba = self._softmax_rows(scores_ba)
        attended_b = attn_ba.matmul(ha)

        out_a = self.out_a(attended_a)  # (B, H)
        out_b = self.out_b(attended_b)  # (B, H)

        # Concatenate and project
        concat_data = np.concatenate([out_a.data, out_b.data], axis=-1)
        concat_t = Tensor(concat_data, requires_grad=(out_a.requires_grad or out_b.requires_grad))
        fused = self.final(concat_t)
        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax_rows(x: Any) -> Any:
        """Row-wise softmax on a 2-D Tensor (numerically stable)."""
        data = x.data.astype(np.float64)
        shifted = data - data.max(axis=1, keepdims=True)
        exp_d = np.exp(shifted)
        softmax_d = exp_d / (exp_d.sum(axis=1, keepdims=True) + 1e-12)
        return Tensor(softmax_d.astype(np.float32), requires_grad=x.requires_grad)
