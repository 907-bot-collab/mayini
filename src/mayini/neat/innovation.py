class InnovationTracker:
    """
    Track innovation numbers for connection genes

    Innovation numbers ensure that matching genes can be identified
    across different genomes for crossover operations.

    Attributes
    ----------
    innovations : dict
        Maps (in_node, out_node) pairs to innovation numbers
    current_innovation : int
        Current innovation counter

    Example
    -------
    >>> from mayini.neat import InnovationTracker
    >>> tracker = InnovationTracker()
    >>> innov = tracker.get_innovation(0, 3)
    """

    def __init__(self):
        self.innovations = {}
        self.current_innovation = 0

    def get_innovation(self, in_node, out_node):
        """
        Get or create innovation number for a connection

        Parameters
        ----------
        in_node : int
            Input node ID
        out_node : int
            Output node ID

        Returns
        -------
        int
            Innovation number
        """
        key = (in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.innovations[key]

    def reset(self):
        """Reset innovation counter"""
        self.innovations.clear()
        self.current_innovation = 0
