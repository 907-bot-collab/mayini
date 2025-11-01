import numpy as np
from typing import Callable, List
from mayini.nn.modules import Linear  # Integrate with your nn

class Genome:
    def __init__(self, input_size: int, output_size: int, innovation=0):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = list(range(input_size + output_size))  # 0..in-1 inputs, in..in+out-1 outputs
        self.connections = []  # List of (in_node, out_node, weight, enabled)
        self.fitness = 0
        self.innovation = innovation

    def add_connection(self, in_node: int, out_node: int, weight: float = np.random.normal()):
        self.connections.append({'in': in_node, 'out': out_node, 'weight': weight, 'enabled': True})

    def to_net(self) -> List[Linear]:  # Convert to your nn stack (simple MLP for now)
        layers = []
        hidden_size = len([c for c in self.connections if c['out'] not in range(self.input_size, self.input_size + self.output_size)])
        if hidden_size > 0:
            layers.append(Linear(self.input_size, hidden_size))
            layers.append(Linear(hidden_size, self.output_size))
        else:
            layers.append(Linear(self.input_size, self.output_size))
        # Set weights from connections (simplified)
        return layers

class NEAT:
    def __init__(self, input_size: int, output_size: int, population: int = 150, mutation_rate: float = 0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.population: List[Genome] = [self._random_genome() for _ in range(population)]
        self.mutation_rate = mutation_rate
        self.global_innovation = 0

    def _random_genome(self) -> Genome:
        g = Genome(self.input_size, self.output_size, self.global_innovation)
        for i in range(self.input_size):
            for o in range(self.input_size, self.input_size + self.output_size):
                if np.random.rand() > 0.5:
                    g.add_connection(i, o)
        self.global_innovation += len(g.connections)
        return g

    def evolve(self, fitness_fn: Callable[[List[Linear]], float], generations: int = 50) -> List[Linear]:
        for gen in range(generations):
            # Evaluate
            for genome in self.population:
                net = genome.to_net()
                genome.fitness = fitness_fn(net)
            # Select, crossover, mutate
            self.population = self._evolve_generation()
        best = max(self.population, key=lambda g: g.fitness)
        return best.to_net()

    def _evolve_generation(self) -> List[Genome]:
        # Simplified GA: Tournament select, uniform crossover, mutate
        new_pop = []
        for _ in range(len(self.population)):
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_pop.append(child)
        return new_pop

    def _tournament_select(self, size: int = 5) -> Genome:
        return max(np.random.choice(self.population, size, replace=False), key=lambda g: g.fitness)

    def _crossover(self, g1: Genome, g2: Genome) -> Genome:
        # Align by innovation, take from fitter parent
        fitter = g1 if g1.fitness > g2.fitness else g2
        child = Genome(self.input_size, self.output_size, max(g1.innovation, g2.innovation))
        child.connections = fitter.connections.copy()  # Simplified
        return child

    def _mutate(self, genome: Genome):
        if np.random.rand() < self.mutation_rate:
            # Add connection, etc.
            in_node = np.random.randint(0, self.input_size)
            out_node = np.random.randint(self.input_size, self.input_size + self.output_size)
            genome.add_connection(in_node, out_node)
        # Weight mutation...
        for conn in genome.connections:
            if np.random.rand() < 0.1:
                conn['weight'] += np.random.normal(0, 0.1)
