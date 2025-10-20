"""Innovation number tracking"""

class InnovationTracker:
    def __init__(self):
        self.innovations = {}
        self.current_innovation = 0

    def get_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.innovations[key]

    def reset(self):
        self.innovations = {}
        self.current_innovation = 0

