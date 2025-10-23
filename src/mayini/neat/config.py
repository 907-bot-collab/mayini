class Config:
    """
    NEAT algorithm configuration
    
    Parameters for evolution, mutation, and speciation
    """
    
    def __init__(self):
        # Population parameters
        self.population_size = 150
        
        # Genome parameters
        self.n_inputs = 2
        self.n_outputs = 1
        
        # Mutation rates
        self.weight_mutation_rate = 0.8
        self.weight_mutation_power = 0.5
        self.weight_replace_rate = 0.1
        
        self.add_connection_rate = 0.05
        self.add_node_rate = 0.03
        
        self.enable_connection_rate = 0.01
        self.disable_connection_rate = 0.01
        
        # Speciation parameters
        self.compatibility_threshold = 3.0
        self.compatibility_disjoint_coefficient = 1.0
        self.compatibility_weight_coefficient = 0.4
        
        # Species parameters
        self.species_elitism = 2
        self.survival_threshold = 0.2
        self.stagnation_threshold = 15
        
        # Reproduction parameters
        self.crossover_rate = 0.75
        self.interspecies_crossover_rate = 0.001
        
        # Activation functions
        self.activation_default = 'sigmoid'
        self.activation_options = ['sigmoid', 'tanh', 'relu', 'identity']
        
        # Fitness
        self.fitness_criterion = 'max'  # 'max' or 'min'
        self.fitness_threshold = None
        
        # Network parameters
        self.bias_mutation_rate = 0.7
        self.bias_mutation_power = 0.5
        self.bias_replace_rate = 0.1
        
        # Recurrent connections
        self.allow_recurrent = False
        
    def validate(self):
        """Validate configuration parameters"""
        assert self.population_size > 0, "Population size must be positive"
        assert self.n_inputs > 0, "Number of inputs must be positive"
        assert self.n_outputs > 0, "Number of outputs must be positive"
        assert 0 <= self.weight_mutation_rate <= 1, "Mutation rate must be between 0 and 1"
        assert self.compatibility_threshold > 0, "Compatibility threshold must be positive"
        return True
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config.validate()
        return config

