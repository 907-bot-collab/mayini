import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
import pytest


def test_node_gene():
    """Test NodeGene"""
    from mayini.neat.gene import NodeGene
    
    node = NodeGene(0, 'input', 'sigmoid')
    node_copy = node.copy()
    
    assert node.id == node_copy.id
    assert node.type == node_copy.type
    print("✅ NodeGene passed")


def test_connection_gene():
    """Test ConnectionGene"""
    from mayini.neat.gene import ConnectionGene
    
    conn = ConnectionGene(0, 1, 0.5, 0, True)
    conn_copy = conn.copy()
    
    assert conn.in_node == conn_copy.in_node
    assert conn.weight == conn_copy.weight
    print("✅ ConnectionGene passed")


def test_genome():
    """Test Genome"""
    from mayini.neat.genome import Genome
    from mayini.neat.innovation import InnovationTracker
    
    genome = Genome(n_inputs=2, n_outputs=1)
    innovation_tracker = InnovationTracker()
    
    # Add connections
    for i in range(2):
        innov = innovation_tracker.get_innovation(i, 2)
        genome.add_connection(i, 2, np.random.randn(), innov)
    
    assert len(genome.nodes) == 3
    assert len(genome.connections) == 2
    print("✅ Genome passed")


def test_innovation_tracker():
    """Test InnovationTracker"""
    from mayini.neat.innovation import InnovationTracker
    
    tracker = InnovationTracker()
    
    innov1 = tracker.get_innovation(0, 1)
    innov2 = tracker.get_innovation(0, 1)  # Same connection
    innov3 = tracker.get_innovation(0, 2)  # Different connection
    
    assert innov1 == innov2
    assert innov1 != innov3
    print("✅ InnovationTracker passed")


def test_neural_network():
    """Test NeuralNetwork"""
    from mayini.neat.genome import Genome
    from mayini.neat.network import NeuralNetwork
    from mayini.neat.innovation import InnovationTracker
    
    genome = Genome(n_inputs=2, n_outputs=1)
    innovation_tracker = InnovationTracker()
    
    # Add connections
    for i in range(2):
        innov = innovation_tracker.get_innovation(i, 2)
        genome.add_connection(i, 2, 1.0, innov)
    
    network = NeuralNetwork(genome)
    output = network.activate([1.0, 0.0])
    
    assert len(output) == 1
    assert 0 <= output[0] <= 1
    print("✅ NeuralNetwork passed")


def test_population():
    """Test Population"""
    from mayini.neat.population import Population
    
    pop = Population(n_inputs=2, n_outputs=1, pop_size=10)
    
    assert len(pop.genomes) == 10
    assert pop.generation == 0
    print("✅ Population passed")


def test_species():
    """Test Species"""
    from mayini.neat.species import Species
    from mayini.neat.genome import Genome
    
    genome = Genome(n_inputs=2, n_outputs=1)
    species = Species(genome)
    
    species.add_member(genome)
    species.calculate_average_fitness()
    
    assert len(species.members) == 1
    print("✅ Species passed")


def test_config():
    """Test Config"""
    from mayini.neat.config import Config
    
    config = Config()
    
    assert config.population_size == 150
    assert config.validate()
    print("✅ Config passed")


def test_activation_functions():
    """Test ActivationFunctions"""
    from mayini.neat.activation import ActivationFunctions
    
    x = np.array([0.0, 1.0, -1.0])
    
    y_sigmoid = ActivationFunctions.sigmoid(x)
    assert np.all((y_sigmoid >= 0) & (y_sigmoid <= 1))
    
    y_tanh = ActivationFunctions.tanh(x)
    assert np.all((y_tanh >= -1) & (y_tanh <= 1))
    
    y_relu = ActivationFunctions.relu(x)
    assert np.all(y_relu >= 0)
    
    print("✅ ActivationFunctions passed")


def test_evaluator():
    """Test Evaluator"""
    from mayini.neat.evaluator import Evaluator, XORFitnessEvaluator
    from mayini.neat.genome import Genome
    from mayini.neat.network import NeuralNetwork
    from mayini.neat.innovation import InnovationTracker
    
    evaluator = XORFitnessEvaluator()
    
    genome = Genome(n_inputs=2, n_outputs=1)
    innovation_tracker = InnovationTracker()
    
    for i in range(2):
        innov = innovation_tracker.get_innovation(i, 2)
        genome.add_connection(i, 2, 1.0, innov)
    
    network = NeuralNetwork(genome)
    fitness = evaluator.evaluate_xor(network)
    
    assert fitness >= 0
    print("✅ Evaluator passed")


if __name__ == '__main__':
    print("\\n" + "="*60)
    print("Testing NEAT Module")
    print("="*60 + "\\n")
    
    test_node_gene()
    test_connection_gene()
    test_genome()
    test_innovation_tracker()
    test_neural_network()
    test_population()
    test_species()
    test_config()
    test_activation_functions()
    test_evaluator()
    
    print("\\n" + "="*60)
    print("✅ All NEAT tests passed!")
    print("="*60)
