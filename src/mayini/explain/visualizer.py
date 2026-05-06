"""
mayini.explain.visualizer — Text & (optional) graphical graph visualisation.

Uses only stdlib by default; matplotlib/networkx are optional.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class GraphVisualizer:
    """Visualise a Mayini computation graph.

    Parameters
    ----------
    model:
        Optional model reference (not required; graph is built from a
        tensor output).

    Example
    -------
    >>> from mayini.explain import GraphVisualizer, export_graph
    >>> graph = export_graph(loss_tensor)
    >>> viz = GraphVisualizer()
    >>> viz.print_summary(graph)
    >>> viz.to_dot(graph, path="graph.dot")
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def print_summary(self, graph: Dict) -> None:
        """Print a human-readable summary of the graph."""
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        print(f"Computation Graph: {len(nodes)} nodes, {len(edges)} edges")
        print("-" * 50)
        for node in nodes:
            leaf_marker = " [LEAF]" if node.get("is_leaf") else ""
            grad_marker = " [grad]" if node.get("requires_grad") else ""
            shape = node.get("shape", [])
            print(
                f"  [{node.get('op', '?'):30s}]  shape={shape}"
                f"{grad_marker}{leaf_marker}"
            )

    # ------------------------------------------------------------------
    # DOT / Graphviz export
    # ------------------------------------------------------------------

    def to_dot(self, graph: Dict, path: Optional[str] = None) -> str:
        """Export graph as Graphviz DOT language string.

        Parameters
        ----------
        graph:
            Dict from :func:`~mayini.explain.export_graph`.
        path:
            If provided, write DOT to this file.

        Returns
        -------
        str
            DOT language representation.
        """
        lines = ["digraph MayiniGraph {", '  rankdir=TB;', '  node [shape=box];']

        id_map: Dict[int, int] = {}
        for i, node in enumerate(graph.get("nodes", [])):
            nid = node["id"]
            id_map[nid] = i
            label = node.get("op", "leaf")
            shape_str = str(node.get("shape", []))
            color = "lightblue" if node.get("is_leaf") else "lightyellow"
            lines.append(
                f'  n{i} [label="{label}\\n{shape_str}" style=filled fillcolor={color}];'
            )

        for src, dst in graph.get("edges", []):
            if src in id_map and dst in id_map:
                lines.append(f"  n{id_map[src]} -> n{id_map[dst]};")

        lines.append("}")
        dot_str = "\n".join(lines)

        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(dot_str)

        return dot_str

    # ------------------------------------------------------------------
    # Optional matplotlib visualisation
    # ------------------------------------------------------------------

    def plot(self, graph: Dict, figsize=(10, 6)) -> None:  # type: ignore[return]
        """Plot graph using networkx + matplotlib (optional deps).

        Raises
        ------
        ImportError
            If ``networkx`` or ``matplotlib`` are not installed.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "Install matplotlib and networkx for graph plotting: "
                "pip install matplotlib networkx"
            ) from exc

        G = nx.DiGraph()
        id_map: Dict[int, int] = {}
        labels: Dict[int, str] = {}

        for i, node in enumerate(graph.get("nodes", [])):
            nid = node["id"]
            id_map[nid] = i
            G.add_node(i)
            labels[i] = f"{node.get('op', 'leaf')}\n{node.get('shape', [])}"

        for src, dst in graph.get("edges", []):
            if src in id_map and dst in id_map:
                G.add_edge(id_map[src], id_map[dst])

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=figsize)
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_color="skyblue",
            node_size=2000,
            font_size=8,
            arrows=True,
        )
        plt.title("Mayini Computation Graph")
        plt.tight_layout()
        plt.show()
