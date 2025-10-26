import numpy as np
from .activation import ActivationFunctions


class NeuralNetwork:
    """
    Neural network for executing NEAT genomes

    Parameters
    ----------
    genome : Genome
        Genome defining network topology
    activation : str, default='sigmoid'
        Default activation function

    Example
    -------
    >>> from mayini.neat import NeuralNetwork, Genome
    >>> genome = Genome(n_inputs=2, n_outputs=1)
    >>> network = NeuralNetwork(genome)
    >>> output = network.activate([1.0, 0.5])
    """

    def __init__(self, genome, activation="sigmoid"):
        self.genome = genome
        self.activation_func = ActivationFunctions.get_activation(activation)
        self.values = {}

    def activate(self, inputs):
        """
        Activate network with given inputs

        Parameters
        ----------
        inputs : list or array-like
            Input values

        Returns
        -------
        list
            Output values

        Raises
        ------
        ValueError
            If number of inputs doesn't match network
        """
        if len(inputs) != self.genome.n_inputs:
            raise ValueError(
                f"Expected {self.genome.n_inputs} inputs, got {len(inputs)}"
            )

        # Reset node values
        self.values = {}

        # Set input values
        for i, value in enumerate(inputs):
            self.values[i] = value

        # Get computation order (topological sort)
        order = self._get_computation_order()

        # Compute values for each node in order
        for node_id in order:
            if node_id < self.genome.n_inputs:
                continue  # Skip input nodes

            # Sum incoming connections
            total = 0.0
            for conn in self.genome.connections.values():
                if conn.out_node == node_id and conn.enabled:
                    if conn.in_node in self.values:
                        total += self.values[conn.in_node] * conn.weight

            # Apply activation function
            node = self.genome.nodes[node_id]
            activation = ActivationFunctions.get_activation(node.activation)
            self.values[node_id] = activation(total)

        # Extract output values
        outputs = []
        for i in range(self.genome.n_outputs):
            node_id = self.genome.n_inputs + i
            outputs.append(self.values.get(node_id, 0.0))

        return outputs

    def _get_computation_order(self):
        """
        Get topological order for node computation

        Returns
        -------
        list
            Node IDs in computation order
        """
        # Build adjacency list
        adjacency = {node_id: [] for node_id in self.genome.nodes}
        in_degree = {node_id: 0 for node_id in self.genome.nodes}

        for conn in self.genome.connections.values():
            if conn.enabled:
                adjacency[conn.in_node].append(conn.out_node)
                in_degree[conn.out_node] += 1

        # Kahn's algorithm for topological sort
        queue = [
            node_id for node_id in self.genome.nodes if in_degree[node_id] == 0
        ]
        order = []

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)

            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def reset(self):
        """Reset network values"""
        self.values = {}
