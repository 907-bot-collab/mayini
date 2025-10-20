"""Genome representation"""
import numpy as np
from .gene import NodeGene, ConnectionGene

class Genome:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = {}
        self.connections = {}
        self.fitness = 0.0

        # Create input and output nodes
        for i in range(n_inputs):
            self.nodes[i] = NodeGene(i, 'input')

        for i in range(n_outputs):
            node_id = n_inputs + i
            self.nodes[node_id] = NodeGene(node_id, 'output')

    def add_connection(self, in_node, out_node, weight, innovation):
        if innovation not in self.connections:
            self.connections[innovation] = ConnectionGene(in_node, out_node, weight, innovation)

    def add_node(self, node_id, node_type='hidden'):
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeGene(node_id, node_type)

    def mutate_weights(self, mutation_rate=0.8, mutation_power=0.5):
        for conn in self.connections.values():
            if np.random.rand() < mutation_rate:
                if np.random.rand() < 0.9:
                    # Perturb weight
                    conn.weight += np.random.randn() * mutation_power
                else:
                    # New random weight
                    conn.weight = np.random.randn() * 2

    def mutate_add_connection(self, innovation_tracker):
        # Get possible connections
        possible_inputs = list(self.nodes.keys())
        possible_outputs = [n for n in self.nodes.keys() if self.nodes[n].type != 'input']

        if not possible_outputs:
            return

        # Try to add new connection
        for _ in range(20):  # Max attempts
            in_node = np.random.choice(possible_inputs)
            out_node = np.random.choice(possible_outputs)

            # Check if connection already exists
            exists = any(c.in_node == in_node and c.out_node == out_node 
                        for c in self.connections.values())

            if not exists and in_node != out_node:
                innovation = innovation_tracker.get_innovation(in_node, out_node)
                self.add_connection(in_node, out_node, np.random.randn(), innovation)
                break

    def mutate_add_node(self, innovation_tracker):
        if not self.connections:
            return

        # Choose random connection to split
        conn = np.random.choice(list(self.connections.values()))
        conn.enabled = False

        # Create new node
        new_node_id = max(self.nodes.keys()) + 1
        self.add_node(new_node_id, 'hidden')

        # Add two new connections
        innov1 = innovation_tracker.get_innovation(conn.in_node, new_node_id)
        innov2 = innovation_tracker.get_innovation(new_node_id, conn.out_node)

        self.add_connection(conn.in_node, new_node_id, 1.0, innov1)
        self.add_connection(new_node_id, conn.out_node, conn.weight, innov2)

    def copy(self):
        genome = Genome(self.n_inputs, self.n_outputs)
        genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        genome.connections = {k: v.copy() for k, v in self.connections.items()}
        genome.fitness = self.fitness
        return genome

