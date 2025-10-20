"""Gene classes for NEAT"""
import numpy as np

class NodeGene:
    def __init__(self, node_id, node_type, activation='sigmoid'):
        self.id = node_id
        self.type = node_type  # 'input', 'hidden', 'output'
        self.activation = activation
        self.bias = np.random.randn() * 0.1

    def copy(self):
        gene = NodeGene(self.id, self.type, self.activation)
        gene.bias = self.bias
        return gene


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, innovation, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, 
                            self.innovation, self.enabled)

