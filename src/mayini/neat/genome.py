import numpy as np
from .gene import NodeGene, ConnectionGene


class Genome:
    """
    NEAT Genome representation

    Represents a neural network topology with nodes and connections

    Parameters
    ----------
    n_inputs : int
        Number of input nodes
    n_outputs : int
        Number of output nodes

    Attributes
    ----------
    nodes : dict
        Dictionary of NodeGene objects
    connections : dict
        Dictionary of ConnectionGene objects
    fitness : float
        Fitness score of genome

    Example
    -------
    >>> from mayini.neat import Genome
    >>> genome = Genome(n_inputs=3, n_outputs=1)
    """

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = {}
        self.connections = {}
        self.fitness = 0.0

        # Create input and output nodes
        for i in range(n_inputs):
            self.nodes[i] = NodeGene(i, "input")

        for i in range(n_outputs):
            node_id = n_inputs + i
            self.nodes[node_id] = NodeGene(node_id, "output")

    def add_connection(self, in_node, out_node, weight, innovation):
        """Add a connection gene"""
        if innovation not in self.connections:
            self.connections[innovation] = ConnectionGene(
                in_node, out_node, weight, innovation
            )

    def add_node(self, node_id, node_type="hidden"):
        """Add a node gene"""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeGene(node_id, node_type)

    def mutate_weights(self, mutation_rate=0.8, mutation_power=0.5):
        """Mutate connection weights"""
        for conn in self.connections.values():
            if np.random.rand() < mutation_rate:
                if np.random.rand() < 0.9:
                    # Perturb weight
                    conn.weight += np.random.randn() * mutation_power
                else:
                    # New random weight
                    conn.weight = np.random.randn() * 2

    def mutate_add_connection(self, innovation_tracker):
        """Add a new connection through mutation"""
        # Get possible connections
        possible_inputs = list(self.nodes.keys())
        possible_outputs = [
            n for n in self.nodes.keys() if self.nodes[n].type != "input"
        ]

        if not possible_outputs:
            return

        # Try to add new connection
        for _ in range(20):  # Max attempts
            in_node = np.random.choice(possible_inputs)
            out_node = np.random.choice(possible_outputs)

            # Check if connection already exists
            exists = any(
                c.in_node == in_node and c.out_node == out_node
                for c in self.connections.values()
            )

            if not exists and in_node != out_node:
                innovation = innovation_tracker.get_innovation(in_node, out_node)
                self.add_connection(in_node, out_node, np.random.randn(), innovation)
                break

    def mutate_add_node(self, innovation_tracker):
        """Add a new node through mutation"""
        if not self.connections:
            return

        # Choose random connection to split
        conn = np.random.choice(list(self.connections.values()))
        conn.enabled = False

        # Create new node
        new_node_id = max(self.nodes.keys()) + 1
        self.add_node(new_node_id, "hidden")

        # Add two new connections
        innov1 = innovation_tracker.get_innovation(conn.in_node, new_node_id)
        innov2 = innovation_tracker.get_innovation(new_node_id, conn.out_node)

        self.add_connection(conn.in_node, new_node_id, 1.0, innov1)
        self.add_connection(new_node_id, conn.out_node, conn.weight, innov2)

    def copy(self):
        """Create a deep copy of the genome"""
        genome = Genome(self.n_inputs, self.n_outputs)
        genome.nodes = {k: v.copy() for k, v in self.nodes.items()}
        genome.connections = {k: v.copy() for k, v in self.connections.items()}
        genome.fitness = self.fitness
        return genome

    def distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        """
        Calculate genetic distance between two genomes

        Parameters
        ----------
        other : Genome
            Other genome to compare
        c1 : float
            Excess coefficient
        c2 : float
            Disjoint coefficient
        c3 : float
            Weight difference coefficient

        Returns
        -------
        float
            Genetic distance
        """
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        matching = innovations1 & innovations2
        disjoint_excess = innovations1 ^ innovations2

        if not matching:
            return float("inf")

        # Calculate weight difference for matching genes
        weight_diff = sum(
            abs(
                self.connections[innov].weight - other.connections[innov].weight
            )
            for innov in matching
        )
        weight_diff /= len(matching)

        # Count excess and disjoint
        max_innov1 = max(innovations1) if innovations1 else 0
        max_innov2 = max(innovations2) if innovations2 else 0

        excess = sum(1 for i in disjoint_excess if i > min(max_innov1, max_innov2))
        disjoint = len(disjoint_excess) - excess

        # Normalize by genome size
        N = max(len(self.connections), len(other.connections), 1)

        return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * weight_diff)
