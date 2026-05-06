"""
mayini.benchmark.runner — Standardised benchmark runner.

Measures accuracy, inference latency, memory footprint, and parameter
count across multiple models and datasets. Outputs Markdown and optional
HTML reports.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class BenchmarkRunner:
    """Run standardised benchmarks across Mayini models.

    Parameters
    ----------
    results_dir:
        Directory where reports are saved.

    Example
    -------
    >>> runner = BenchmarkRunner()
    >>> results = runner.run(
    ...     models=[model_a, model_b],
    ...     test_data=test_batches,
    ...     metrics=["accuracy", "latency", "params"],
    ... )
    >>> runner.save_report()
    """

    def __init__(self, results_dir: str = "benchmark_results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        models: List[Any],
        test_data: Any,
        metrics: Optional[List[str]] = None,
        loss_fn: Optional[Callable] = None,
        num_latency_runs: int = 50,
    ) -> List[Dict[str, Any]]:
        """Benchmark a list of models on *test_data*.

        Parameters
        ----------
        models:
            List of Mayini models (callable, with ``.parameters()``).
        test_data:
            Iterable of ``(X, y)`` batches or plain tensors.
        metrics:
            Any subset of ``["accuracy", "latency", "memory", "params", "flops"]``.
        loss_fn:
            Optional loss function for accuracy measurement.
        num_latency_runs:
            Number of forward passes to average for latency.

        Returns
        -------
        list
            One result dict per model.
        """
        if metrics is None:
            metrics = ["latency", "params"]

        data_list = list(test_data)

        for model in models:
            name = type(model).__name__
            record: Dict[str, Any] = {
                "model": name,
                "timestamp": datetime.now().isoformat(),
            }

            if "params" in metrics:
                record["params"] = self._count_params(model)

            if "latency" in metrics:
                record["latency_ms"] = self._measure_latency(
                    model, data_list, num_latency_runs
                )

            if "memory" in metrics:
                record["memory_bytes"] = self._estimate_memory(model)

            if "accuracy" in metrics and loss_fn is not None:
                record["accuracy"] = self._measure_accuracy(
                    model, data_list, loss_fn
                )

            self.results.append(record)
            self._print_record(record)

        return self.results

    def save_report(self, fmt: str = "markdown") -> Path:
        """Write benchmark results to disk.

        Parameters
        ----------
        fmt:
            ``"markdown"`` (default) or ``"json"``.

        Returns
        -------
        pathlib.Path
            Path to the saved report.
        """
        if fmt == "json":
            path = self.results_dir / "benchmark_results.json"
            path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
        else:
            path = self.results_dir / "benchmark_report.md"
            path.write_text(self._to_markdown(), encoding="utf-8")

        print(f"[benchmark] Report saved → {path.resolve()}")
        return path

    # ------------------------------------------------------------------
    # Individual measurements
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params(model: Any) -> int:
        total = 0
        try:
            for _, p in model.named_parameters():
                total += p.data.size
        except AttributeError:
            for p in model.parameters():
                total += p.data.size
        return total

    @staticmethod
    def _measure_latency(model: Any, data: list, n: int) -> Dict[str, float]:
        latencies: List[float] = []
        for i in range(n):
            sample = data[i % max(len(data), 1)]
            x = sample[0] if isinstance(sample, (list, tuple)) else sample
            t0 = time.perf_counter()
            model(x)
            latencies.append((time.perf_counter() - t0) * 1000)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }

    @staticmethod
    def _estimate_memory(model: Any) -> int:
        """Estimate parameter memory in bytes (float32 = 4 bytes)."""
        total = 0
        try:
            for _, p in model.named_parameters():
                total += p.data.nbytes
        except AttributeError:
            for p in model.parameters():
                total += p.data.nbytes
        return total

    @staticmethod
    def _measure_accuracy(model: Any, data: list, loss_fn: Callable) -> float:
        correct, total = 0, 0
        for batch in data:
            try:
                X, y = batch
                out = model(X)
                preds = out.data.argmax(axis=-1)
                labels = y.data if hasattr(y, "data") else np.asarray(y)
                correct += int((preds == labels).sum())
                total += labels.size
            except Exception:
                pass
        return correct / max(total, 1)

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def _to_markdown(self) -> str:
        lines = [
            "# Mayini Benchmark Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "| Model | Params | Latency (mean ms) | Memory (KB) | Accuracy |",
            "|-------|--------|-------------------|-------------|----------|",
        ]
        for r in self.results:
            lat = r.get("latency_ms", {})
            mean_lat = f"{lat.get('mean_ms', 0):.2f}" if isinstance(lat, dict) else "—"
            mem_kb = f"{r.get('memory_bytes', 0) / 1024:.1f}" if "memory_bytes" in r else "—"
            acc = f"{r.get('accuracy', 0):.4f}" if "accuracy" in r else "—"
            lines.append(
                f"| {r['model']} | {r.get('params','—')} | {mean_lat} | {mem_kb} | {acc} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _print_record(record: Dict) -> None:
        lat = record.get("latency_ms", {})
        mean_lat = f"{lat.get('mean_ms', 0):.2f} ms" if isinstance(lat, dict) else "—"
        print(
            f"[benchmark] {record['model']:30s}  "
            f"params={record.get('params','?')}  "
            f"latency={mean_lat}"
        )
