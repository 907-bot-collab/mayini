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
        total_fitness = 0

        for genome in population.genomes:
            fitness = self.evaluate_genome(genome)
            total_fitness += fitness

        avg_fitness = total_fitness / len(population.genomes)

        # Update best genome
        best = population.get_best_genome()
        if population.best_genome is None or best.fitness > population.best_genome.fitness:
            population.best_genome = best.copy()

        return avg_fitness
