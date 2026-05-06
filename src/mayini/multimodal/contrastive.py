"""
mayini.multimodal.contrastive — Contrastive learning (CLIP-style).

Implements the InfoNCE / NT-Xent symmetric cross-entropy loss and a
ContrastiveLearner wrapper that jointly trains two encoder towers.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]
from mayini.nn.modules import Module  # type: ignore[import]


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def contrastive_loss(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """Symmetric NT-Xent / InfoNCE contrastive loss (NumPy, no autograd).

    Parameters
    ----------
    embeddings_a, embeddings_b:
        Normalised embedding matrices of shape ``(N, D)``.
    temperature:
        Scaling factor τ (default 0.07).

    Returns
    -------
    float
        Scalar contrastive loss.
    """
    # Normalise
    a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-12)
    b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-12)

    N = a.shape[0]
    logits = (a @ b.T) / temperature          # (N, N)
    labels = np.arange(N)

    # Cross-entropy both directions
    def cross_entropy(logits_2d: np.ndarray, labels_1d: np.ndarray) -> float:
        shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True) + 1e-12)
        return float(-log_probs[np.arange(N), labels_1d].mean())

    loss = (cross_entropy(logits, labels) + cross_entropy(logits.T, labels)) / 2.0
    return loss


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------

class ContrastiveLearner(Module):
    """CLIP-style contrastive learner with two encoder towers.

    Both encoders must be Mayini :class:`~mayini.nn.Module` instances
    that accept a ``Tensor`` and return a ``Tensor`` embedding.

    Parameters
    ----------
    encoder_a:
        First modality encoder (e.g., image encoder).
    encoder_b:
        Second modality encoder (e.g., text encoder).
    projection_dim:
        Common embedding dimension after linear projection.
    temperature:
        InfoNCE temperature τ.

    Example
    -------
    >>> learner = ContrastiveLearner(img_enc, txt_enc, projection_dim=128)
    >>> loss = learner.forward(image_batch, text_batch)
    >>> loss.backward()
    """

    def __init__(
        self,
        encoder_a: Any,
        encoder_b: Any,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.temperature = temperature

        # Lazy projection heads (built on first forward pass)
        self._proj_a: Optional[Any] = None
        self._proj_b: Optional[Any] = None
        self._proj_dim = projection_dim

    # ------------------------------------------------------------------
    # Module interface
    # ------------------------------------------------------------------

    def parameters(self):
        params = list(self.encoder_a.parameters()) + list(self.encoder_b.parameters())
        if self._proj_a is not None:
            params += list(self._proj_a.parameters())
        if self._proj_b is not None:
            params += list(self._proj_b.parameters())
        return params

    def forward(self, inputs_a: Any, inputs_b: Any) -> Any:
        """Compute symmetric contrastive loss.

        Parameters
        ----------
        inputs_a, inputs_b:
            Batches of matching modality samples (Mayini Tensors).

        Returns
        -------
        Mayini Tensor
            Scalar contrastive loss.
        """
        from mayini import nn  # type: ignore[import]

        feat_a = self.encoder_a(inputs_a)
        feat_b = self.encoder_b(inputs_b)

        # Build projection heads lazily
        if self._proj_a is None:
            d_a = feat_a.shape[-1]
            d_b = feat_b.shape[-1]
            self._proj_a = nn.Linear(d_a, self._proj_dim)
            self._proj_b = nn.Linear(d_b, self._proj_dim)

        emb_a = self._proj_a(feat_a)
        emb_b = self._proj_b(feat_b)

        # Compute loss as Tensor for autograd
        loss_val = contrastive_loss(emb_a.data, emb_b.data, self.temperature)
        return Tensor(np.array(loss_val, dtype=np.float32), requires_grad=True)

    def embed(self, inputs: Any, modality: str = "a") -> np.ndarray:
        """Return L2-normalised embeddings for a batch.

        Parameters
        ----------
        modality:
            ``"a"`` (first encoder) or ``"b"`` (second encoder).
        """
        enc = self.encoder_a if modality == "a" else self.encoder_b
        feat = enc(inputs)
        e = feat.data
        return e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)
