"""
mayini.distill — Knowledge Distillation & Graph-Style Model Composition

Includes:
  - DistillationTrainer  : KD loss (KL + CE) with temperature scaling
  - GraphModule          : DAG-wired model composition
"""

from .distillation import DistillationTrainer, distillation_loss
from .graph_module import GraphModule

__all__ = ["DistillationTrainer", "distillation_loss", "GraphModule"]
