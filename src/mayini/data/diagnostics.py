"""
mayini.data.diagnostics — Dataset quality analysis tools.

Checks class balance, duplicates, data-leakage between splits,
feature distribution, and outliers — all with NumPy only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DataDiagnostics:
    """Run a suite of data-quality checks on a dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, *features)``.
    y : np.ndarray
        Label array of shape ``(N,)``.

    Example
    -------
    >>> diag = DataDiagnostics(X_train, y_train)
    >>> diag.run_all()
    >>> diag.report()
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Full suite
    # ------------------------------------------------------------------

    def run_all(
        self,
        imbalance_threshold: float = 10.0,
        outlier_zscore: float = 3.0,
    ) -> Dict[str, Any]:
        """Run all diagnostic checks and return the results dict."""
        self.results["class_distribution"] = self.class_distribution()
        self.results["imbalance"] = self.imbalance_ratio(imbalance_threshold)
        self.results["duplicates"] = self.duplicate_count()
        self.results["feature_stats"] = self.feature_statistics()
        self.results["outliers"] = self.outlier_count(outlier_zscore)
        return self.results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def class_distribution(self) -> Dict[Any, int]:
        """Return count of each class label."""
        classes, counts = np.unique(self.y, return_counts=True)
        return {int(c): int(n) for c, n in zip(classes, counts)}

    def imbalance_ratio(self, threshold: float = 10.0) -> Dict[str, Any]:
        """Compute max/min class count ratio."""
        dist = self.class_distribution()
        if not dist:
            return {"ratio": 0.0, "severe": False}
        counts = list(dist.values())
        ratio = max(counts) / max(min(counts), 1)
        return {"ratio": float(ratio), "severe": ratio > threshold}

    def duplicate_count(self) -> Dict[str, int]:
        """Count exact duplicate rows in X."""
        flat = self.X.reshape(len(self.X), -1)
        hashes = [hash(row.tobytes()) for row in flat]
        unique_hashes = set(hashes)
        duplicates = len(hashes) - len(unique_hashes)
        return {"total": len(hashes), "duplicates": duplicates, "unique": len(unique_hashes)}

    def leakage_check(
        self,
        X_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """Detect sample overlap between train and val/test splits."""
        train_hashes = {hash(r.tobytes()) for r in self.X.reshape(len(self.X), -1)}
        val_hashes = {hash(r.tobytes()) for r in np.asarray(X_val).reshape(len(X_val), -1)}
        overlap_val = len(train_hashes & val_hashes)
        result = {"train_val_overlap": overlap_val}
        if X_test is not None:
            test_hashes = {hash(r.tobytes()) for r in np.asarray(X_test).reshape(len(X_test), -1)}
            result["train_test_overlap"] = len(train_hashes & test_hashes)
        return result

    def feature_statistics(self) -> Dict[str, np.ndarray]:
        """Compute mean, std, min, max, median per feature dimension."""
        flat = self.X.reshape(len(self.X), -1).astype(np.float64)
        return {
            "mean": flat.mean(axis=0),
            "std": flat.std(axis=0),
            "min": flat.min(axis=0),
            "max": flat.max(axis=0),
            "median": np.median(flat, axis=0),
        }

    def outlier_count(self, z_threshold: float = 3.0) -> Dict[str, Any]:
        """Count samples with any feature z-score beyond *z_threshold*."""
        flat = self.X.reshape(len(self.X), -1).astype(np.float64)
        mu = flat.mean(axis=0)
        sigma = flat.std(axis=0) + 1e-12
        z_scores = np.abs((flat - mu) / sigma)
        outlier_mask = (z_scores > z_threshold).any(axis=1)
        return {
            "outlier_count": int(outlier_mask.sum()),
            "outlier_fraction": float(outlier_mask.mean()),
            "threshold": z_threshold,
        }

    def report(self) -> None:
        """Print a human-readable summary of all diagnostics."""
        if not self.results:
            self.run_all()

        print("=" * 60)
        print("  Mayini DataDiagnostics Report")
        print("=" * 60)

        # Class distribution
        dist = self.results.get("class_distribution", {})
        print(f"\n[Class Distribution]  {len(dist)} classes")
        for cls, cnt in sorted(dist.items()):
            bar = "█" * min(int(cnt / max(dist.values()) * 30), 30)
            print(f"  Class {cls:>4}: {cnt:>6}  {bar}")

        # Imbalance
        imb = self.results.get("imbalance", {})
        severe = "⚠ SEVERE" if imb.get("severe") else "OK"
        print(f"\n[Imbalance]  ratio={imb.get('ratio', 0):.2f}  {severe}")

        # Duplicates
        dup = self.results.get("duplicates", {})
        print(f"\n[Duplicates]  {dup.get('duplicates', 0)} duplicates"
              f" / {dup.get('total', 0)} total")

        # Outliers
        out = self.results.get("outliers", {})
        print(f"\n[Outliers]  {out.get('outlier_count', 0)} samples"
              f" ({out.get('outlier_fraction', 0)*100:.1f}%)"
              f" beyond z={out.get('threshold', 3.0)}")

        print("=" * 60)
