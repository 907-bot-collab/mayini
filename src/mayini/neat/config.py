class Config:
    """
    NEAT Configuration

    Parameters
    ----------
    population_size : int, default=150
        Number of genomes in population
    fitness_threshold : float, default=None
        Fitness threshold for termination
    max_generations : int, default=None
        Maximum generations
    activation_function : str, default='sigmoid'
        Default activation function
    mutation_rate : float, default=0.8
        Probability of mutation
    crossover_rate : float, default=0.75
        Probability of crossover
    add_node_rate : float, default=0.03
        Probability of adding node
    add_connection_rate : float, default=0.05
        Probability of adding connection
    weight_mutation_rate : float, default=0.8
        Probability of weight mutation
    weight_mutation_power : float, default=0.5
        Standard deviation for weight perturbation
    compatibility_threshold : float, default=3.0
        Threshold for species compatibility
    excess_coefficient : float, default=1.0
        Weight for excess genes in distance
    disjoint_coefficient : float, default=1.0
        Weight for disjoint genes in distance
    weight_coefficient : float, default=0.4
        Weight for weight difference in distance
    survival_threshold : float, default=0.2
        Fraction of species that survives

    Example
    -------
    >>> from mayini.neat import Config
    >>> config = Config(population_size=150, max_generations=100)
    """

    def __init__(
        self,
        population_size=150,
        fitness_threshold=None,
        max_generations=None,
        activation_function="sigmoid",
        mutation_rate=0.8,
        crossover_rate=0.75,
        add_node_rate=0.03,
        add_connection_rate=0.05,
        weight_mutation_rate=0.8,
        weight_mutation_power=0.5,
        compatibility_threshold=3.0,
        excess_coefficient=1.0,
        disjoint_coefficient=1.0,
        weight_coefficient=0.4,
        survival_threshold=0.2,
    ):
        self.population_size = population_size
        self.fitness_threshold = fitness_threshold
        self.max_generations = max_generations
        self.activation_function = activation_function
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.weight_mutation_power = weight_mutation_power
        self.compatibility_threshold = compatibility_threshold
        self.excess_coefficient = excess_coefficient
        self.disjoint_coefficient = disjoint_coefficient
        self.weight_coefficient = weight_coefficient
        self.survival_threshold = survival_threshold

    def __repr__(self):
        """String representation"""
        return (
            f"Config(population_size={self.population_size}, "
            f"max_generations={self.max_generations})"
        )
