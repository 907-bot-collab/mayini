import numpy as np


class NetworkVisualizer:
    """
    Visualize NEAT neural networks

    Parameters
    ----------
    genome : Genome
        Genome to visualize

    Example
    -------
    >>> from mayini.neat import NetworkVisualizer
    >>> visualizer = NetworkVisualizer(genome)
    >>> info = visualizer.get_network_info()
    """

    def __init__(self, genome):
        self.genome = genome

    def get_network_info(self):
        """
        Get information about network structure

        Returns
        -------
        dict
            Network information including node counts and connections
        """
        n_inputs = self.genome.n_inputs
        n_outputs = self.genome.n_outputs
        n_hidden = len(self.genome.nodes) - n_inputs - n_outputs
        n_connections = len(
            [c for c in self.genome.connections.values() if c.enabled]
        )

        return {
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "n_hidden": n_hidden,
            "n_connections": n_connections,
            "total_nodes": len(self.genome.nodes),
            "total_genes": len(self.genome.connections),
        }

    def get_layers(self):
        """
        Compute network layers for visualization

        Returns
        -------
        dict
            Dictionary mapping node IDs to layer numbers
        """
        layers = {}

        # Input layer
        for i in range(self.genome.n_inputs):
            layers[i] = 0

        # Output layer (set to high number initially)
        for i in range(self.genome.n_outputs):
            node_id = self.genome.n_inputs + i
            layers[node_id] = 999

        # Compute layers for hidden nodes
        changed = True
        while changed:
            changed = False
            for conn in self.genome.connections.values():
                if not conn.enabled:
                    continue

                if conn.in_node in layers and conn.out_node in layers:
                    new_layer = layers[conn.in_node] + 1
                    if new_layer < layers[conn.out_node]:
                        layers[conn.out_node] = new_layer
                        changed = True
                elif conn.in_node in layers:
                    layers[conn.out_node] = layers[conn.in_node] + 1
                    changed = True

        # Normalize output layer
        max_layer = max(layers.values())
        for i in range(self.genome.n_outputs):
            node_id = self.genome.n_inputs + i
            layers[node_id] = max_layer + 1

        return layers

    def to_dict(self):
        """
        Convert network to dictionary format for export

        Returns
        -------
        dict
            Network structure as dictionary
        """
        nodes = []
        for node_id, node in self.genome.nodes.items():
            nodes.append(
                {"id": node_id, "type": node.type, "activation": node.activation}
            )

        connections = []
        for innov, conn in self.genome.connections.items():
            connections.append(
                {
                    "innovation": innov,
                    "in_node": conn.in_node,
                    "out_node": conn.out_node,
                    "weight": float(conn.weight),
                    "enabled": conn.enabled,
                }
            )

        return {
            "nodes": nodes,
            "connections": connections,
            "n_inputs": self.genome.n_inputs,
            "n_outputs": self.genome.n_outputs,
            "fitness": self.genome.fitness,
        }

    def __repr__(self):
        """String representation"""
        info = self.get_network_info()
        return (
            f"NetworkVisualizer(inputs={info['n_inputs']}, "
            f"outputs={info['n_outputs']}, hidden={info['n_hidden']}, "
            f"connections={info['n_connections']})"
        )
