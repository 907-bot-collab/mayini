import numpy as np


class Species:
    """
    Species for grouping similar genomes

    Parameters
    ----------
    representative : Genome
        Representative genome for the species
    species_id : int
        Unique species identifier

    Attributes
    ----------
    members : list
        List of genomes in this species
    fitness_history : list
        Historical fitness values
    age : int
        Age of the species in generations

    Example
    -------
    >>> from mayini.neat import Species
    >>> species = Species(genome, species_id=0)
    """

    def __init__(self, representative, species_id):
        self.representative = representative
        self.species_id = species_id
        self.members = []
        self.fitness_history = []
        self.age = 0
        self.max_fitness = 0

    def add_member(self, genome):
        """Add a genome to this species"""
        self.members.append(genome)

    def calculate_adjusted_fitness(self):
        """
        Calculate adjusted fitness for all members

        Adjusted fitness accounts for species size to encourage diversity
        """
        if not self.members:
            return

        # Explicit fitness sharing
        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / len(self.members)

    def update_representative(self):
        """Update species representative"""
        if self.members:
            # Choose member with highest fitness as representative
            self.representative = max(self.members, key=lambda g: g.fitness)

    def get_average_fitness(self):
        """Get average fitness of species"""
        if not self.members:
            return 0
        return np.mean([g.fitness for g in self.members])

    def cull(self, survival_threshold=0.2):
        """
        Remove lower-performing members

        Parameters
        ----------
        survival_threshold : float, default=0.2
            Fraction of members to keep
        """
        if len(self.members) <= 2:
            return

        # Sort by fitness
        self.members.sort(key=lambda g: g.fitness, reverse=True)

        # Keep top performers
        cutoff = max(2, int(len(self.members) * survival_threshold))
        self.members = self.members[:cutoff]
