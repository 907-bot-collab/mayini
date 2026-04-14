import numpy as np


class Evaluator:
    """
    Evaluates fitness of NEAT genomes

    Parameters
    ----------
    fitness_function : callable
        Function that takes a genome and returns fitness score

    Example
    -------
    >>> from mayini.neat import Evaluator
    >>> def fitness_fn(genome):
    ...     # Evaluate genome
    ...     return score
    >>> evaluator = Evaluator(fitness_fn)
    >>> evaluator.evaluate_population(population)
    """

    def __init__(self, fitness_function):
        self.fitness_function = fitness_function

    def evaluate_genome(self, genome):
        """
        Evaluate a single genome

        Parameters
        ----------
        genome : Genome
            Genome to evaluate

        Returns
        -------
        float
            Fitness score
        """
        fitness = self.fitness_function(genome)
        genome.fitness = fitness
        return fitness

    def evaluate_population(self, population):
        """
        Evaluate all genomes in population

        Parameters
        ----------
        population : Population
            Population to evaluate

        Returns
        -------
        float
            Average fitness of population
        """
        if not population.genomes:
            return 0

        total_fitness = 0

        for genome in population.genomes:
            fitness = self.evaluate_genome(genome)
            total_fitness += fitness

        avg_fitness = total_fitness / len(population.genomes)

        # Update best genome
        best = population.get_best_genome()
        if (
            population.best_genome is None
            or best.fitness > population.best_genome.fitness
        ):
            population.best_genome = best.copy()

        return avg_fitness


class XORFitnessEvaluator(Evaluator):
    """Evaluator for XOR problem"""

    def __init__(self):
        super().__init__(self.xor_fitness)

    def xor_fitness(self, genome):
        # XOR inputs and outputs
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        targets = [0, 1, 1, 0]

        from .network import Network

        net = Network(genome)

        error = 0
        for input_data, target in zip(inputs, targets):
            output = net.activate(input_data)
            error += (output[0] - target) ** 2

        # Fitness = 4 - SSE (max fitness is 4)
        return max(0.0, 4.0 - error)

    def evaluate_xor(self, network):
        """Evaluate XOR for a network instance (as in tests)"""
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        targets = [0, 1, 1, 0]
        error = 0
        for input_data, target in zip(inputs, targets):
            output = network.activate(input_data)
            error += (output[0] - target) ** 2
        return max(0.0, 4.0 - error)
