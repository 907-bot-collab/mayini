import numpy as np
from .network import NeuralNetwork


class Evaluator:
    """
    Base evaluator class for fitness evaluation
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def evaluate_genome(self, genome, fitness_function):
        """
        Evaluate a single genome
        
        Parameters:
        -----------
        genome : Genome
            Genome to evaluate
        fitness_function : callable
            Function that takes network and returns fitness
        
        Returns:
        --------
        float : Fitness value
        """
        network = NeuralNetwork(genome)
        fitness = fitness_function(network)
        genome.fitness = fitness
        return fitness
    
    def evaluate_population(self, population, fitness_function, parallel=False):
        """
        Evaluate entire population
        
        Parameters:
        -----------
        population : Population
            Population to evaluate
        fitness_function : callable
            Fitness function
        parallel : bool
            Whether to use parallel evaluation
        
        Returns:
        --------
        list : List of fitness values
        """
        if parallel:
            return self._evaluate_parallel(population, fitness_function)
        else:
            return self._evaluate_sequential(population, fitness_function)
    
    def _evaluate_sequential(self, population, fitness_function):
        """Sequential evaluation"""
        fitnesses = []
        for genome in population.genomes:
            fitness = self.evaluate_genome(genome, fitness_function)
            fitnesses.append(fitness)
        
        # Track best genome
        best_genome = max(population.genomes, key=lambda g: g.fitness)
        if population.best_genome is None or best_genome.fitness > population.best_genome.fitness:
            population.best_genome = best_genome.copy()
        
        return fitnesses
    
    def _evaluate_parallel(self, population, fitness_function):
        """Parallel evaluation (simple multiprocessing)"""
        from multiprocessing import Pool, cpu_count
        
        n_workers = min(cpu_count(), len(population.genomes))
        
        with Pool(n_workers) as pool:
            args = [(genome, fitness_function) for genome in population.genomes]
            fitnesses = pool.starmap(self._eval_helper, args)
        
        for genome, fitness in zip(population.genomes, fitnesses):
            genome.fitness = fitness
        
        # Track best genome
        best_genome = max(population.genomes, key=lambda g: g.fitness)
        if population.best_genome is None or best_genome.fitness > population.best_genome.fitness:
            population.best_genome = best_genome.copy()
        
        return fitnesses
    
    @staticmethod
    def _eval_helper(genome, fitness_function):
        """Helper for parallel evaluation"""
        network = NeuralNetwork(genome)
        return fitness_function(network)


class XORFitnessEvaluator(Evaluator):
    """
    Example: XOR problem fitness evaluator
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.xor_outputs = np.array([0, 1, 1, 0])
    
    def evaluate_xor(self, network):
        """Evaluate network on XOR problem"""
        error = 0.0
        for xi, xo in zip(self.xor_inputs, self.xor_outputs):
            output = network.activate(xi)[0]
            error += (output - xo) ** 2
        
        # Fitness is inverse of error
        fitness = (4.0 - error) ** 2
        return fitness

