class NodeGene:
    """
    Node Gene representation

    Parameters
    ----------
    node_id : int
        Unique node identifier
    node_type : str
        Type of node ('input', 'hidden', 'output')
    activation : str, default='sigmoid'
        Activation function name

    Example
    -------
    >>> from mayini.neat.gene import NodeGene
    >>> node = NodeGene(0, 'input')
    """

    def __init__(self, node_id, node_type, activation="sigmoid"):
        self.id = node_id
        self.type = node_type
        self.activation = activation

    def copy(self):
        """Create a copy of the node gene"""
        return NodeGene(self.id, self.type, self.activation)

    def __repr__(self):
        """String representation"""
        return f"NodeGene(id={self.id}, type={self.type})"


class ConnectionGene:
    """
    Connection Gene representation

    Parameters
    ----------
    in_node : int
        Input node ID
    out_node : int
        Output node ID
    weight : float
        Connection weight
    innovation : int
        Innovation number
    enabled : bool, default=True
        Whether connection is enabled

    Example
    -------
    >>> from mayini.neat.gene import ConnectionGene
    >>> conn = ConnectionGene(0, 1, 0.5, innovation=1)
    """

    def __init__(self, in_node, out_node, weight, innovation, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.innovation = innovation
        self.enabled = enabled

    def copy(self):
        """Create a copy of the connection gene"""
        return ConnectionGene(
            self.in_node, self.out_node, self.weight, self.innovation, self.enabled
        )

    def __repr__(self):
        """String representation"""
        status = "enabled" if self.enabled else "disabled"
        return (
            f"ConnectionGene({self.in_node}->{self.out_node}, "
            f"w={self.weight:.3f}, {status})"
        )
