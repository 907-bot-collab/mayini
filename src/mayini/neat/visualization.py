
import numpy as np


class NEATVisualizer:
    """
    Visualize NEAT genomes and networks
    """
    
    @staticmethod
    def print_genome(genome):
        """Print genome structure"""
        print(f"\\nGenome (Fitness: {genome.fitness:.4f})")
        print(f"Inputs: {genome.n_inputs}, Outputs: {genome.n_outputs}")
        print(f"Nodes: {len(genome.nodes)}, Connections: {len(genome.connections)}")
        
        print("\\nNodes:")
        for node_id, node in genome.nodes.items():
            print(f"  {node_id}: {node.type} (bias: {node.bias:.3f})")
        
        print("\\nConnections:")
        for innov, conn in genome.connections.items():
            status = "enabled" if conn.enabled else "disabled"
            print(f"  {conn.in_node} -> {conn.out_node}: {conn.weight:.3f} ({status}, innov: {innov})")
    
    @staticmethod
    def print_population_stats(population):
        """Print population statistics"""
        print(f"\\n{'='*60}")
        print(f"Generation {population.generation}")
        print(f"{'='*60}")
        
        fitnesses = [g.fitness for g in population.genomes]
        print(f"Population size: {len(population.genomes)}")
        print(f"Species count: {len(population.species)}")
        print(f"\\nFitness Statistics:")
        print(f"  Max: {np.max(fitnesses):.4f}")
        print(f"  Mean: {np.mean(fitnesses):.4f}")
        print(f"  Min: {np.min(fitnesses):.4f}")
        print(f"  Std: {np.std(fitnesses):.4f}")
        
        if population.best_genome:
            print(f"\\nBest Genome Ever:")
            print(f"  Fitness: {population.best_genome.fitness:.4f}")
            print(f"  Nodes: {len(population.best_genome.nodes)}")
            print(f"  Connections: {len(population.best_genome.connections)}")
        
        print(f"\\nSpecies:")
        for i, species in enumerate(population.species):
            print(f"  Species {i}: {len(species.members)} members, "
                  f"avg fitness: {species.average_fitness:.4f}, "
                  f"staleness: {species.staleness}")
    
    @staticmethod
    def plot_fitness_history(fitness_history, save_path=None):
        """
        Plot fitness over generations
        
        Requires matplotlib (optional dependency)
        """
        try:
            import matplotlib.pyplot as plt
            
            generations = range(len(fitness_history))
            max_fitness = [max(gen) for gen in fitness_history]
            avg_fitness = [np.mean(gen) for gen in fitness_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, max_fitness, label='Max Fitness', linewidth=2)
            plt.plot(generations, avg_fitness, label='Avg Fitness', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('NEAT Evolution Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
    
    @staticmethod
    def export_network_graphviz(genome, filename='network.dot'):
        """
        Export network to GraphViz DOT format
        
        Can be visualized with: dot -Tpng network.dot -o network.png
        """
        lines = ['digraph G {']
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=circle];')
        
        # Add nodes
        for node_id, node in genome.nodes.items():
            if node.type == 'input':
                lines.append(f'  {node_id} [label="I{node_id}", style=filled, fillcolor=lightblue];')
            elif node.type == 'output':
                lines.append(f'  {node_id} [label="O{node_id}", style=filled, fillcolor=lightgreen];')
            else:
                lines.append(f'  {node_id} [label="H{node_id}"];')
        
        # Add connections
        for conn in genome.connections.values():
            if conn.enabled:
                color = 'green' if conn.weight > 0 else 'red'
                width = min(abs(conn.weight), 3.0)
                lines.append(f'  {conn.in_node} -> {conn.out_node} '
                           f'[color={color}, penwidth={width:.1f}, '
                           f'label="{conn.weight:.2f}"];')
        
        lines.append('}')
        
        with open(filename, 'w') as f:
            f.write('\\n'.join(lines))
        
        print(f"Network exported to {filename}")
        print("Visualize with: dot -Tpng {filename} -o network.png")
