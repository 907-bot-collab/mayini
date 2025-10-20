"""Population management"""
import numpy as np
from .genome import Genome
from .species import Species
from .innovation import InnovationTracker

class Population:
    def __init__(self, n_inputs, n_outputs, pop_size=150):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.genomes = []
        self.species = []
        self.generation = 0
        self.innovation_tracker = InnovationTracker()
        self.best_genome = None

        # Initialize population
        for _ in range(pop_size):
            genome = Genome(n_inputs, n_outputs)
            # Add initial random connections
            for out in range(n_inputs, n_inputs + n_outputs):
                for inp in range(n_inputs):
                    innov = self.innovation_tracker.get_innovation(inp, out)
                    genome.add_connection(inp, out, np.random.randn(), innov)
            self.genomes.append(genome)

    def speciate(self, compatibility_threshold=3.0):
        # Clear existing species memberships
        for species in self.species:
            species.members = []

        # Assign genomes to species
        for genome in self.genomes:
            placed = False

            for species in self.species:
                if self._compatibility_distance(genome, species.representative) < compatibility_threshold:
                    species.add_member(genome)
                    placed = True
                    break

            if not placed:
                # Create new species
                new_species = Species(genome.copy())
                new_species.add_member(genome)
                self.species.append(new_species)

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Calculate fitness and choose representatives
        for species in self.species:
            species.calculate_average_fitness()
            species.choose_representative()

    def _compatibility_distance(self, genome1, genome2):
        # Calculate compatibility distance between two genomes
        innov1 = set(genome1.connections.keys())
        innov2 = set(genome2.connections.keys())

        disjoint = len(innov1.symmetric_difference(innov2))
        matching = innov1.intersection(innov2)

        weight_diff = 0.0
        if matching:
            weight_diff = sum(abs(genome1.connections[i].weight - genome2.connections[i].weight) 
                            for i in matching) / len(matching)

        N = max(len(innov1), len(innov2))
        N = max(N, 1)

        c1, c2, c3 = 1.0, 1.0, 0.4
        distance = (c1 * disjoint / N) + (c3 * weight_diff)

        return distance

    def evolve(self):
        # Remove weakest from each species
        for species in self.species:
            species.remove_weak()

        # Calculate offspring allocation
        total_avg_fitness = sum(s.average_fitness for s in self.species)
        offspring_counts = []

        for species in self.species:
            if total_avg_fitness > 0:
                offspring = int((species.average_fitness / total_avg_fitness) * self.pop_size)
            else:
                offspring = self.pop_size // len(self.species)
            offspring_counts.append(max(1, offspring))

        # Generate new population
        new_genomes = []

        for species, offspring_count in zip(self.species, offspring_counts):
            for _ in range(offspring_count):
                if len(species.members) == 1:
                    parent = species.members
                    child = parent.copy()
                else:
                    parent1 = np.random.choice(species.members)
                    parent2 = np.random.choice(species.members)
                    child = self._crossover(parent1, parent2)

                # Mutate
                self._mutate(child)
                new_genomes.append(child)

        # Ensure population size
        while len(new_genomes) < self.pop_size:
            genome = Genome(self.n_inputs, self.n_outputs)
            new_genomes.append(genome)

        self.genomes = new_genomes[:self.pop_size]
        self.generation += 1

    def _crossover(self, parent1, parent2):
        # More fit parent contributes more genes
        if parent1.fitness > parent2.fitness:
            better, worse = parent1, parent2
        else:
            better, worse = parent2, parent1

        child = Genome(self.n_inputs, self.n_outputs)

        # Inherit nodes from better parent
        child.nodes = {k: v.copy() for k, v in better.nodes.items()}

        # Inherit connections
        for innov, conn in better.connections.items():
            if innov in worse.connections:
                # Matching gene - randomly choose
                child.connections[innov] = (conn if np.random.rand() < 0.5 
                                          else worse.connections[innov]).copy()
            else:
                # Disjoint/excess from better parent
                child.connections[innov] = conn.copy()

        return child

    def _mutate(self, genome):
        if np.random.rand() < 0.8:
            genome.mutate_weights()
        if np.random.rand() < 0.05:
            genome.mutate_add_connection(self.innovation_tracker)
        if np.random.rand() < 0.03:
            genome.mutate_add_node(self.innovation_tracker)

