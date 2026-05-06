"""
mayini.distill.distillation — Knowledge Distillation training utilities.

Implements temperature-scaled soft targets + hard-label KD loss,
following Hinton et al. (2015).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from mayini.tensor import Tensor  # type: ignore[import]


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z -= z.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / (exp_z.sum(axis=-1, keepdims=True) + 1e-12)


def _cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """Cross-entropy: targets can be soft (distributions) or hard (int labels)."""
    if targets.ndim == 1:
        n = len(targets)
        log_p = np.log(probs[np.arange(n), targets.astype(int)] + 1e-12)
        return float(-log_p.mean())
    else:
        log_p = np.log(probs + 1e-12)
        return float(-(targets * log_p).sum(axis=-1).mean())


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) element-wise."""
    return float((p * np.log(p / (q + 1e-12) + 1e-12)).sum(axis=-1).mean())


def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    hard_labels: np.ndarray,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> float:
    """Compute the knowledge distillation loss.

    L = α · τ² · KL(p_teacher ‖ p_student) + (1-α) · CE(z_student, y_hard)

    Parameters
    ----------
    student_logits:
        Raw logits from the student network ``(B, C)``.
    teacher_logits:
        Raw logits from the teacher network ``(B, C)`` (no grad needed).
    hard_labels:
        Ground-truth integer class labels ``(B,)``.
    temperature:
        Distillation temperature τ (higher → softer targets).
    alpha:
        Weight for the soft KD term.

    Returns
    -------
    float
        Scalar total distillation loss.
    """
    p_t = _softmax(teacher_logits, temperature)
    p_s = _softmax(student_logits, temperature)

    L_kd = _kl_divergence(p_t, p_s) * (temperature ** 2)
    L_ce = _cross_entropy(_softmax(student_logits, 1.0), hard_labels)

    return alpha * L_kd + (1.0 - alpha) * L_ce


# ---------------------------------------------------------------------------
# High-level trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Train a student model using knowledge distillation from a teacher.

    Parameters
    ----------
    teacher:
        Frozen Mayini model (teacher). Its parameters are NOT updated.
    student:
        Mayini model to train (student).
    optimizer:
        A Mayini optimizer compatible with ``student.parameters()``.
    temperature:
        Distillation temperature τ.
    alpha:
        Weight for the soft KD loss term (0 = pure hard-label CE,
        1 = pure distillation).

    Example
    -------
    >>> trainer = DistillationTrainer(teacher, student, optim.Adam(...))
    >>> history = trainer.fit(train_data, epochs=20)
    """

    def __init__(
        self,
        teacher: Any,
        student: Any,
        optimizer: Any,
        temperature: float = 4.0,
        alpha: float = 0.5,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.temperature = temperature
        self.alpha = alpha
        self.history: List[Dict] = []

        # Freeze teacher
        try:
            for _, p in teacher.named_parameters():
                p.requires_grad = False
        except AttributeError:
            for p in teacher.parameters():
                p.requires_grad = False

    def fit(
        self,
        data: Any,
        epochs: int = 10,
        verbose: bool = True,
    ) -> List[Dict]:
        """Run the distillation training loop.

        Parameters
        ----------
        data:
            Iterable of ``(X, y)`` batches (Mayini Tensors or numpy arrays).
        epochs:
            Number of training epochs.

        Returns
        -------
        list
            Training history (one dict per epoch with ``"epoch"`` and ``"loss"``).
        """
        data_list = list(data)

        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch in data_list:
                X, y = batch
                X_t = X if hasattr(X, "data") else Tensor(np.asarray(X, np.float32))
                y_np = y.data if hasattr(y, "data") else np.asarray(y)

                # Forward passes
                with _no_grad():
                    t_out = self.teacher(X_t)
                s_out = self.student(X_t)

                t_logits = t_out.data if hasattr(t_out, "data") else np.asarray(t_out)
                s_logits = s_out.data if hasattr(s_out, "data") else np.asarray(s_out)

                loss_val = distillation_loss(
                    s_logits, t_logits, y_np,
                    temperature=self.temperature,
                    alpha=self.alpha,
                )

                # Backprop through student
                loss_t = Tensor(np.array(loss_val, np.float32), requires_grad=True)
                loss_t.backward()
                self.optimizer.step()

                # Zero grads
                try:
                    self.optimizer.zero_grad()
                except AttributeError:
                    for _, p in self.student.named_parameters():
                        p.grad = None

                epoch_loss += loss_val
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            record = {"epoch": ep, "loss": avg}
            self.history.append(record)

            if verbose:
                print(f"[Distill] Epoch {ep:4d}/{epochs}  loss={avg:.6f}")

        return self.history


class _no_grad:
    """Minimal context manager to disable grad tracking temporarily."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
