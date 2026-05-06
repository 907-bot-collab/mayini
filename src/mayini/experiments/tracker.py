"""
mayini.experiments.tracker — Lightweight experiment tracking.

Records hyperparameters, per-step/epoch metrics, and final checkpoints
to JSON + optional CSV, with comparison and Markdown export utilities.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Experiment data class
# ---------------------------------------------------------------------------

class Experiment:
    """Single experiment record.

    Parameters
    ----------
    name:
        Human-readable experiment name.
    hyperparams:
        Dict of hyperparameter name → value.
    log_dir:
        Directory where logs and checkpoints are stored.

    Example
    -------
    >>> exp = Experiment("mlp_baseline", {"lr": 0.01, "epochs": 50})
    >>> exp.log({"loss": 0.42, "acc": 0.91}, step=1)
    >>> exp.finish()
    """

    def __init__(
        self,
        name: str,
        hyperparams: Optional[Dict[str, Any]] = None,
        log_dir: str = "experiments",
    ) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.hyperparams = hyperparams or {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics: Dict[str, List] = {}
        self.steps: List[int] = []
        self.artifacts: List[str] = []
        self.status: str = "running"
        self.start_time: str = datetime.now().isoformat()
        self.end_time: Optional[str] = None

        self._env = self._capture_env()
        self._save_metadata()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Record a dict of scalar metrics at a given step.

        Parameters
        ----------
        metrics:
            Dict mapping metric names to scalar values.
        step:
            Training step or epoch number.
        """
        self.steps.append(step)
        for key, value in metrics.items():
            self.metrics.setdefault(key, []).append(float(value))
        self._flush_csv(metrics, step)

    def log_artifact(self, path: str) -> None:
        """Record a file path as an artifact (e.g., model checkpoint)."""
        self.artifacts.append(str(path))
        self._save_metadata()

    def finish(self, best_metric: Optional[str] = None) -> None:
        """Mark experiment as complete and write final metadata."""
        self.status = "finished"
        self.end_time = datetime.now().isoformat()
        self._save_metadata()
        duration = self._duration_str()
        print(f"[exp:{self.id}] '{self.name}' finished in {duration}")
        if best_metric and best_metric in self.metrics:
            vals = self.metrics[best_metric]
            print(f"  Best {best_metric}: {max(vals):.4f}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        meta = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "hyperparams": self.hyperparams,
            "artifacts": self.artifacts,
            "environment": self._env,
            "metrics_summary": self._metrics_summary(),
        }
        path = self.log_dir / f"exp_{self.id}.json"
        path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def _flush_csv(self, metrics: Dict[str, float], step: int) -> None:
        csv_path = self.log_dir / f"exp_{self.id}_metrics.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step"] + list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({"step": step, **metrics})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _metrics_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for key, vals in self.metrics.items():
            if vals:
                import numpy as np
                summary[key] = {
                    "last": float(vals[-1]),
                    "best": float(max(vals)),
                    "mean": float(sum(vals) / len(vals)),
                    "steps": len(vals),
                }
        return summary

    def _duration_str(self) -> str:
        if self.end_time and self.start_time:
            try:
                start = datetime.fromisoformat(self.start_time)
                end = datetime.fromisoformat(self.end_time)
                secs = (end - start).total_seconds()
                m, s = divmod(int(secs), 60)
                return f"{m}m {s}s"
            except Exception:
                pass
        return "?"

    @staticmethod
    def _capture_env() -> Dict[str, str]:
        import numpy as np
        env = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": sys.platform,
        }
        try:
            import mayini  # type: ignore[import]
            env["mayini"] = mayini.__version__
        except Exception:
            pass
        return env


# ---------------------------------------------------------------------------
# Multi-experiment tracker / comparison utility
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Scan and compare multiple experiments stored in a directory.

    Example
    -------
    >>> tracker = ExperimentTracker("experiments")
    >>> tracker.list()
    >>> tracker.top(metric="acc", n=5)
    >>> tracker.to_markdown("results.md")
    """

    def __init__(self, log_dir: str = "experiments") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def new(
        self,
        name: str,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create and return a new :class:`Experiment`."""
        return Experiment(name, hyperparams, log_dir=str(self.log_dir))

    def load_all(self) -> List[Dict]:
        """Load all experiment metadata JSONs from the log directory."""
        records: List[Dict] = []
        for path in sorted(self.log_dir.glob("exp_*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return records

    def list(self) -> None:
        """Print a table of all experiments."""
        records = self.load_all()
        if not records:
            print("No experiments found.")
            return
        print(f"\n{'ID':<10} {'Name':<30} {'Status':<10} {'Start':>24}")
        print("-" * 78)
        for r in records:
            print(
                f"{r['id']:<10} {r['name']:<30} "
                f"{r['status']:<10} {r['start_time']:>24}"
            )

    def top(self, metric: str, n: int = 5, higher_is_better: bool = True) -> List[Dict]:
        """Return the top-n experiments ranked by a metric.

        Parameters
        ----------
        metric:
            Metric name (e.g., ``"acc"``, ``"loss"``).
        n:
            Number of experiments to return.
        higher_is_better:
            If True, rank by highest value; if False, by lowest.

        Returns
        -------
        list
            Sorted list of experiment metadata dicts.
        """
        records = self.load_all()
        scored = [
            r for r in records
            if metric in r.get("metrics_summary", {})
        ]
        key = "best" if higher_is_better else "last"
        scored.sort(
            key=lambda r: r["metrics_summary"][metric].get(key, 0),
            reverse=higher_is_better,
        )
        top = scored[:n]
        print(f"\nTop-{n} by '{metric}' ({'↑' if higher_is_better else '↓'})")
        print(f"{'ID':<10} {'Name':<30} {metric:>10}")
        print("-" * 52)
        for r in top:
            val = r["metrics_summary"][metric].get(key, "?")
            print(f"{r['id']:<10} {r['name']:<30} {val:>10.4f}")
        return top

    def to_markdown(self, output_file: str = "experiment_summary.md") -> Path:
        """Export all experiments to a Markdown table."""
        records = self.load_all()
        lines = [
            "# Mayini Experiment Summary",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "| ID | Name | Status | Started |",
            "|----|------|--------|---------|",
        ]
        for r in records:
            lines.append(
                f"| {r['id']} | {r['name']} | {r['status']} | {r['start_time'][:19]} |"
            )
        path = Path(output_file)
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[tracker] Markdown report → {path.resolve()}")
        return path
