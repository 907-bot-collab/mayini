"""
mayini.explain.graph — Computation Graph Builder & Exporter

Traverses the dynamic autograd graph from any output Tensor and
produces a serialisable DAG (nodes + edges) for inspection or
visualisation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


class ComputationGraphBuilder:
    """Build and export the computation graph from a Mayini Tensor.

    Example
    -------
    >>> from mayini import Tensor
    >>> from mayini.explain import ComputationGraphBuilder
    >>> x = Tensor([1.0, 2.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> builder = ComputationGraphBuilder()
    >>> graph = builder.build(y)
    >>> print(graph["nodes"])
    """

    def __init__(self) -> None:
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Tuple[int, int]] = []
        self._visited: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, tensor: Any) -> Dict[str, Any]:
        """Traverse the computational graph rooted at *tensor*.

        Parameters
        ----------
        tensor:
            A Mayini :class:`~mayini.Tensor` that is the output of a
            computation (has ``.prev`` and ``.op`` attributes).

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [...]}`` ready for JSON export
            or visualisation.
        """
        self.nodes = []
        self.edges = []
        self._visited = set()
        self._traverse(tensor)
        return {"nodes": self.nodes, "edges": self.edges}

    def to_json(self, tensor: Any, path: Optional[str] = None) -> str:
        """Return the graph as a JSON string, optionally writing to *path*."""
        graph = self.build(tensor)
        json_str = json.dumps(graph, indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
        return json_str

    # ------------------------------------------------------------------
    # Internal traversal (DFS, iterative to avoid stack overflow)
    # ------------------------------------------------------------------

    def _traverse(self, root: Any) -> None:
        stack = [root]
        while stack:
            tensor = stack.pop()
            tid = id(tensor)
            if tid in self._visited:
                continue
            self._visited.add(tid)

            node = self._describe_tensor(tensor)
            self.nodes.append(node)

            for parent in getattr(tensor, "prev", set()):
                pid = id(parent)
                self.edges.append((pid, tid))
                if pid not in self._visited:
                    stack.append(parent)

    @staticmethod
    def _describe_tensor(t: Any) -> Dict[str, Any]:
        return {
            "id": id(t),
            "tensor_id": getattr(t, "id", None),
            "op": getattr(t, "op", "") or "leaf",
            "shape": list(getattr(t, "shape", [])),
            "requires_grad": getattr(t, "requires_grad", False),
            "is_leaf": getattr(t, "is_leaf", True),
            "dtype": str(getattr(t, "dtype", "")),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def export_graph(tensor: Any, path: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper — build and optionally save the computation graph.

    Parameters
    ----------
    tensor:
        Output tensor whose graph should be exported.
    path:
        If given, the graph JSON is written to this file.

    Returns
    -------
    dict
        Graph dictionary with ``"nodes"`` and ``"edges"`` keys.
    """
    builder = ComputationGraphBuilder()
    graph = builder.build(tensor)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(graph, fh, indent=2, default=str)
    return graph
