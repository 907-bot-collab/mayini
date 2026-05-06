"""
mayini.multimodal.clip — Minimal CLIP-style zero-shot classifier.

Uses pre-built Mayini encoders and contrastive embeddings to perform
zero-shot classification without task-specific fine-tuning.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]
from mayini.nn.modules import Module  # type: ignore[import]
from .contrastive import ContrastiveLearner


class SimpleMultiModalCLIP(Module):
    """Lightweight CLIP-style model for zero-shot image-text matching.

    Wraps a :class:`~mayini.multimodal.ContrastiveLearner` and adds
    zero-shot classification via cosine similarity between image
    embeddings and class-prompt text embeddings.

    Parameters
    ----------
    image_encoder:
        Mayini Module that encodes image tensors → embedding vectors.
    text_encoder:
        Mayini Module that encodes tokenised/embedded text → vectors.
    projection_dim:
        Shared embedding space dimension.
    temperature:
        InfoNCE temperature.

    Example
    -------
    >>> clip = SimpleMultiModalCLIP(img_enc, txt_enc, projection_dim=128)
    >>> preds, conf = clip.zero_shot_classify(images, ["cat", "dog", "car"])
    """

    def __init__(
        self,
        image_encoder: Any,
        text_encoder: Any,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.learner = ContrastiveLearner(
            image_encoder, text_encoder, projection_dim, temperature
        )
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def parameters(self):
        return self.learner.parameters()

    def forward(self, images: Any, texts: Any) -> Any:
        """Compute contrastive training loss for a matched image-text batch."""
        return self.learner.forward(images, texts)

    # ------------------------------------------------------------------
    # Zero-shot classification
    # ------------------------------------------------------------------

    def zero_shot_classify(
        self,
        images: Any,
        class_names: List[str],
        class_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Classify images into classes using cosine similarity.

        Parameters
        ----------
        images:
            Batch of image Tensors ``(B, *image_shape)``.
        class_names:
            Human-readable class name strings (for logging only if
            *class_embeddings* are provided externally).
        class_embeddings:
            Pre-computed text embeddings ``(C, D)`` for each class.
            If ``None``, zero vectors are used as placeholders (you
            should supply real text encodings from your text encoder).

        Returns
        -------
        predictions : np.ndarray (B,)
            Predicted class indices.
        confidences : np.ndarray (B, C)
            Softmax similarity scores.
        """
        # Image embeddings
        img_emb = self.learner.embed(images, modality="a")   # (B, D)

        n_classes = len(class_names)

        if class_embeddings is None:
            # Placeholder: use random unit vectors as class prototypes
            rng = np.random.default_rng(0)
            class_embeddings = rng.standard_normal((n_classes, img_emb.shape[-1])).astype(np.float32)

        # Normalise class embeddings
        cls_emb = class_embeddings / (
            np.linalg.norm(class_embeddings, axis=1, keepdims=True) + 1e-12
        )

        # Cosine similarity (B, C)
        sims = img_emb @ cls_emb.T / self.learner.temperature

        # Softmax
        shifted = sims - sims.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        confidences = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-12)
        predictions = confidences.argmax(axis=1)

        return predictions, confidences
