"""
mayini.multimodal — Multimodal Foundation Model Support

Contrastive learning (CLIP-style), cross-attention fusion,
and zero-shot classification utilities — all on top of the
Mayini NumPy/autograd stack.
"""

from .contrastive import ContrastiveLearner, contrastive_loss
from .fusion import CrossAttentionFusion
from .clip import SimpleMultiModalCLIP

__all__ = [
    "ContrastiveLearner",
    "contrastive_loss",
    "CrossAttentionFusion",
    "SimpleMultiModalCLIP",
]
