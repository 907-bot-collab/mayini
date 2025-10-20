"""Species management for NEAT"""
import numpy as np

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = []
        self.fitness_history = []
        self.average_fitness = 0.0
        self.staleness = 0

    def add_member(self, genome):
        self.members.append(genome)

    def calculate_average_fitness(self):
        if not self.members:
            self.average_fitness = 0.0
            return

        total_fitness = sum(m.fitness for m in self.members)
        self.average_fitness = total_fitness / len(self.members)

        # Track staleness
        if self.fitness_history and self.average_fitness <= max(self.fitness_history):
            self.staleness += 1
        else:
            self.staleness = 0

        self.fitness_history.append(self.average_fitness)

    def sort_members(self):
        self.members.sort(key=lambda x: x.fitness, reverse=True)

    def remove_weak(self, survival_rate=0.2):
        self.sort_members()
        cutoff = max(1, int(len(self.members) * survival_rate))
        self.members = self.members[:cutoff]

    def choose_representative(self):
        if self.members:
            self.representative = np.random.choice(self.members).copy()

