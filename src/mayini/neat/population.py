import numpy as np
from .genome import Genome
from .species import Species


class Population:
    """
    Population of NEAT genomes

    Parameters
    ----------
    config : Config
        NEAT configuration
    n_inputs : int
        Number of input nodes
    n_outputs : int
        Number of output nodes

    Attributes
    ----------
    genomes : list
        List of all genomes
    species : list
        List of species
    generation : int
        Current generation number
    best_genome : Genome
        Best genome found so far

    Example
    -------
    >>> from mayini.neat import Population, Config
    >>> config = Config(population_size=150)
    >>> pop = Population(config, n_inputs=2, n_outputs=1)
    """

    def __init__(self, config, n_inputs, n_outputs):
        self.config = config
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.genomes = []
        self.species = []
        self.generation = 0
        self.best_genome = None
        self.innovation_tracker = None

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population"""
        from .innovation import InnovationTracker

        self.innovation_tracker = InnovationTracker()

        for _ in range(self.config.population_size):
            genome = Genome(self.n_inputs, self.n_outputs)

            # Add initial connections (fully connected)
            for i in range(self.n_inputs):
                for j in range(self.n_outputs):
                    out_node = self.n_inputs + j
                    innovation = self.innovation_tracker.get_innovation(i, out_node)
                    weight = np.random.randn()
                    genome.add_connection(i, out_node, weight, innovation)

            self.genomes.append(genome)

    def speciate(self):
        """Assign genomes to species"""
        # Reset species members
        for species in self.species:
            species.members = []

        # Assign each genome to a species
        for genome in self.genomes:
            found_species = False

            for species in self.species:
                distance = genome.distance(
                    species.representative,
                    self.config.excess_coefficient,
                    self.config.disjoint_coefficient,
                    self.config.weight_coefficient,
                )

                if distance < self.config.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break

            if not found_species:
                # Create new species
                new_species = Species(genome, len(self.species))
                new_species.add_member(genome)
                self.species.append(new_species)

        # Remove empty species
        self.species = [s for s in self.species if s.members]

        # Update representatives
        for species in self.species:
            species.update_representative()

    def evolve(self):
        """Evolve population for one generation"""
        # Speciate
        self.speciate()

        # Calculate adjusted fitness
        for species in self.species:
            species.calculate_adjusted_fitness()

        # Calculate spawn amounts
        total_adjusted_fitness = sum(
            sum(g.adjusted_fitness for g in s.members) for s in self.species
        )

        if total_adjusted_fitness == 0:
            total_adjusted_fitness = 1

        spawn_amounts = []
        for species in self.species:
            species_fitness = sum(g.adjusted_fitness for g in species.members)
            spawn = int(
                (species_fitness / total_adjusted_fitness)
                * self.config.population_size
            )
            spawn_amounts.append(spawn)

        # Adjust spawn amounts to maintain population size
        while sum(spawn_amounts) < self.config.population_size:
            spawn_amounts[np.argmax(spawn_amounts)] += 1
        while sum(spawn_amounts) > self.config.population_size:
            spawn_amounts[np.argmax(spawn_amounts)] -= 1

        # Create new generation
        new_genomes = []

        for species, spawn_amount in zip(self.species, spawn_amounts):
            # Cull species
            species.cull(self.config.survival_threshold)

            # Reproduce
            for _ in range(spawn_amount):
                if len(species.members) == 0:
                    continue

                if len(species.members) == 1:
                    # Asexual reproduction
                    parent = species.members[0]
                    child = parent.copy()
                else:
                    # Sexual reproduction (crossover)
                    parent1 = np.random.choice(species.members)
                    parent2 = np.random.choice(species.members)
                    child = self._crossover(parent1, parent2)

                # Mutate
                self._mutate(child)
                new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1

    def _crossover(self, parent1, parent2):
        """
        Crossover two genomes

        Parameters
        ----------
        parent1 : Genome
            First parent
        parent2 : Genome
            Second parent

        Returns
        -------
        Genome
            Child genome
        """
        # Select more fit parent
        if parent1.fitness >= parent2.fitness:
            better_parent = parent1
            worse_parent = parent2
        else:
            better_parent = parent2
            worse_parent = parent1

        child = Genome(self.n_inputs, self.n_outputs)

        # Inherit nodes from better parent
        for node_id, node in better_parent.nodes.items():
            child.nodes[node_id] = node.copy()

        # Inherit connections
        for innov, conn in better_parent.connections.items():
            if innov in worse_parent.connections:
                # Matching gene - randomly choose
                if np.random.rand() < 0.5:
                    child.connections[innov] = conn.copy()
                else:
                    child.connections[innov] = worse_parent.connections[innov].copy()
            else:
                # Excess/disjoint from better parent
                child.connections[innov] = conn.copy()

        return child

    def _mutate(self, genome):
        """Apply mutations to genome"""
        # Mutate weights
        if np.random.rand() < self.config.weight_mutation_rate:
            genome.mutate_weights(
                self.config.mutation_rate, self.config.weight_mutation_power
            )

        # Add connection
        if np.random.rand() < self.config.add_connection_rate:
            genome.mutate_add_connection(self.innovation_tracker)

        # Add node
        if np.random.rand() < self.config.add_node_rate:
            genome.mutate_add_node(self.innovation_tracker)

    def get_best_genome(self):
        """Get best genome in population"""
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)
