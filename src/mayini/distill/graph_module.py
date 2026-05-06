"""
mayini.distill.graph_module — DAG-style model composition.

Allows wiring named Mayini sub-modules as a directed acyclic graph,
similar in spirit to PyTorch's ModuleDict + topological execution.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class GraphModule:
    """Compose Mayini modules as a DAG with named data-flow edges.

    Each node is a named sub-module. Edges define which node's output
    is fed as which argument of the next node.

    Parameters
    ----------
    None — add nodes and edges after construction.

    Example
    -------
    >>> gm = GraphModule()
    >>> gm.add_node("embed",  embed_module,  inputs=["x"])
    >>> gm.add_node("encode", encode_module, inputs=["embed"])
    >>> gm.add_node("head",   head_module,   inputs=["encode"])
    >>> gm.set_output("head")
    >>> out = gm.forward({"x": input_tensor})
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict] = {}          # name → {module, inputs}
        self._output_name: Optional[str] = None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        module: Any,
        inputs: List[str],
    ) -> "GraphModule":
        """Register a module as a graph node.

        Parameters
        ----------
        name:
            Unique node identifier.
        module:
            Callable Mayini module.
        inputs:
            List of node names (or ``"x"`` for the raw input dict key)
            whose outputs this node should receive as positional args.
        """
        self._nodes[name] = {"module": module, "inputs": inputs}
        return self

    def set_output(self, name: str) -> "GraphModule":
        """Declare which node's output is the final graph output."""
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not in graph.")
        self._output_name = name
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def forward(self, input_dict: Dict[str, Any]) -> Any:
        """Execute the graph in topological order.

        Parameters
        ----------
        input_dict:
            Dictionary mapping input names (e.g. ``"x"``) to tensors.

        Returns
        -------
        The output of the node declared via :meth:`set_output`.
        """
        if self._output_name is None:
            raise RuntimeError("Call set_output() before forward().")

        order = self._topological_order()
        storage: Dict[str, Any] = dict(input_dict)

        for name in order:
            node = self._nodes[name]
            module = node["module"]
            in_keys = node["inputs"]

            args = [storage[k] for k in in_keys]
            storage[name] = module(*args) if len(args) > 1 else module(args[0])

        return storage[self._output_name]

    def __call__(self, input_dict: Dict[str, Any]) -> Any:
        return self.forward(input_dict)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def parameters(self) -> List[Any]:
        params: List[Any] = []
        for node in self._nodes.values():
            m = node["module"]
            try:
                params.extend(m.parameters())
            except AttributeError:
                pass
        return params

    def named_parameters(self):
        for name, node in self._nodes.items():
            m = node["module"]
            try:
                for pname, p in m.named_parameters():
                    yield f"{name}.{pname}", p
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_order(self) -> List[str]:
        """Return node names in valid execution order."""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        dependents: Dict[str, List[str]] = {n: [] for n in self._nodes}

        for name, node in self._nodes.items():
            for dep in node["inputs"]:
                if dep in self._nodes:
                    in_degree[name] += 1
                    dependents[dep].append(name)

        queue: deque = deque(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []

        while queue:
            n = queue.popleft()
            order.append(n)
            for child in dependents[n]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._nodes):
            raise RuntimeError(
                "Cycle detected in GraphModule — check your add_node() calls."
            )
        return order

    def summary(self) -> None:
        """Print the graph structure."""
        print("GraphModule DAG")
        print(f"  Output node: {self._output_name}")
        for name, node in self._nodes.items():
            ins = " → ".join(node["inputs"])
            mod_cls = type(node["module"]).__name__
            print(f"  [{name}]  inputs=[{ins}]  module={mod_cls}")
