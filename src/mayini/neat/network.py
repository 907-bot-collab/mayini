"""Feed-forward network from genome"""
import numpy as np

class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome
        self.build_network()

    def build_network(self):
        # Build activation order using topological sort
        self.layers = self._topological_sort()

    def _topological_sort(self):
        # Simplified layer-based approach
        nodes = self.genome.nodes
        connections = {k: v for k, v in self.genome.connections.items() if v.enabled}

        in_nodes = [n for n in nodes.values() if n.type == 'input']
        out_nodes = [n for n in nodes.values() if n.type == 'output']
        hidden_nodes = [n for n in nodes.values() if n.type == 'hidden']

        return [in_nodes, hidden_nodes, out_nodes]

    def activate(self, inputs):
        if len(inputs) != self.genome.n_inputs:
            raise ValueError(f"Expected {self.genome.n_inputs} inputs, got {len(inputs)}")

        # Store activations
        activations = {}

        # Set input activations
        for i, inp in enumerate(inputs):
            activations[i] = inp

        # Forward pass through layers
        for layer in self.layers[1:]:  # Skip input layer
            for node in layer:
                # Calculate weighted sum of inputs
                node_input = node.bias

                for conn in self.genome.connections.values():
                    if conn.enabled and conn.out_node == node.id:
                        if conn.in_node in activations:
                            node_input += activations[conn.in_node] * conn.weight

                # Apply activation function
                activations[node.id] = self._sigmoid(node_input)

        # Return output activations
        outputs = []
        for i in range(self.genome.n_outputs):
            out_id = self.genome.n_inputs + i
            outputs.append(activations.get(out_id, 0.0))

        return np.array(outputs)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

