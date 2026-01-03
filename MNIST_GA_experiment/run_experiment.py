"""
Main entry point for MNIST GA SNN experiments.

This script orchestrates the complete experiment workflow:
1. Load configuration
2. Setup data
3. Create network structure
4. Precompute spike trains
5. Run GA evolution
6. Save results

Usage:
    python run_experiment.py
    python run_experiment.py --arch small_2class
    python run_experiment.py --name my_experiment --generations 200
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import random
import multiprocessing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.experiment_config import get_config
from config.network_architectures import get_architecture, list_architectures
from mnist_ga.network_builder import create_snn_structure
from mnist_ga.encoding import setup_mnist_data, precompute_spike_trains
from mnist_ga.evaluation import evaluate_chromosome_fitness
from mnist_ga.visualization import plot_ga_progress
from LIF_objects.GeneticAlgorithm import GeneticAlgorithm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MNIST GA SNN Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--arch', type=str, default=None,
                       help='Pre-defined architecture name (e.g., small_2class, standard_5class)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (used for output directory)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of GA generations')
    parser.add_argument('--population', type=int, default=None,
                       help='GA population size')
    parser.add_argument('--list-archs', action='store_true',
                       help='List available architectures and exit')

    return parser.parse_args()


def setup_experiment(args):
    """
    Setup experiment configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        ExperimentConfig instance
    """
    # Start with architecture if specified
    if args.arch:
        arch_config = get_architecture(args.arch)
        cfg = get_config(**arch_config)
        print(f"Using architecture: {args.arch}")
    else:
        cfg = get_config()

    # Override with command line arguments
    if args.name:
        cfg.name = args.name
    if args.generations:
        cfg.num_generations = args.generations
    if args.population:
        cfg.population_size = args.population

    return cfg


def save_experiment_results(cfg, structure, best_chromosome, best_fitness):
    """
    Save all experiment results to disk.

    Args:
        cfg: ExperimentConfig
        structure: NetworkStructure
        best_chromosome: Best weight vector found
        best_fitness: Best fitness achieved
    """
    output_dir = os.path.join(cfg.experiment_output_dir, "final_network")
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"best_snn_{cfg.n_classes}class"

    print(f"\n=== Saving Results to {output_dir} ===")

    try:
        # Save weights
        weights_path = os.path.join(output_dir, f"{base_name}_weights.npy")
        np.save(weights_path, best_chromosome)
        print(f"✓ Saved weights: {weights_path}")

        # Save connection map
        map_path = os.path.join(output_dir, f"{base_name}_connection_map.npy")
        np.save(map_path, np.array(structure.connection_map, dtype=object))
        print(f"✓ Saved connection map: {map_path}")

        # Save delays matrix
        delays_path = os.path.join(output_dir, f"{base_name}_delays.npy")
        np.save(delays_path, structure.network.delays)
        print(f"✓ Saved delays: {delays_path}")

        # Save inhibitory array
        inhib_path = os.path.join(output_dir, f"{base_name}_inhibitory.npy")
        np.save(inhib_path, structure.network.is_inhibitory)
        print(f"✓ Saved inhibitory status: {inhib_path}")

        # Save positions
        pos_path = os.path.join(output_dir, f"{base_name}_positions.npy")
        np.save(pos_path, structure.positions)
        print(f"✓ Saved positions: {pos_path}")

        # Save configuration - COMPLETE version with all parameters
        config_path = os.path.join(output_dir, f"{base_name}_config.json")
        config_dict = {
            # Experiment metadata
            "experiment_name": cfg.name,
            "random_seed": cfg.random_seed,

            # Network architecture
            "target_classes": cfg.target_classes,
            "hidden_layers": cfg.hidden_layers,
            "layer_config": cfg.layer_config,
            "n_neurons": structure.n_neurons,
            "n_connections": structure.n_connections,
            "inhibitory_fraction": cfg.inhibitory_fraction,

            # Encoding parameters
            "encoding_mode": cfg.encoding_mode,
            "downsample_factor": cfg.downsample_factor,
            "conv_weights_path": cfg.conv_weights_path,
            "conv_feature_count": cfg.conv_feature_count,

            # Simulation parameters
            "sim_duration_ms": cfg.sim_duration_ms,
            "sim_dt_ms": cfg.sim_dt_ms,
            "mnist_stim_duration_ms": cfg.mnist_stim_duration_ms,
            "max_freq_hz": cfg.max_freq_hz,
            "stim_strength": cfg.stim_strength,

            # Network structure parameters
            "base_transmission_delay": cfg.base_transmission_delay,
            "connection_probs": cfg.connection_probs,
            "neuron_params": cfg.neuron_params,
            "output_mutual_inhibition_weight": cfg.output_mutual_inhibition_weight,
            "output_mutual_inhibition_delay": cfg.output_mutual_inhibition_delay,

            # GA parameters
            "ga_generations": cfg.num_generations,
            "ga_population": cfg.population_size,
            "mutation_rate": cfg.mutation_rate,
            "mutation_strength": cfg.mutation_strength,
            "crossover_rate": cfg.crossover_rate,
            "elitism_count": cfg.elitism_count,
            "tournament_size": cfg.tournament_size,
            "fitness_eval_examples": cfg.fitness_eval_examples,

            # Weight bounds
            "weight_min": cfg.weight_min,
            "weight_max": cfg.weight_max,

            # Results
            "best_fitness": float(best_fitness),
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"✓ Saved config: {config_path}")

        print("=" * 50)

    except Exception as e:
        print(f"Error saving results: {e}")


def run_experiment(cfg):
    """
    Run complete GA SNN experiment.

    Args:
        cfg: ExperimentConfig instance
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"MNIST GA SNN Experiment: {cfg.name}")
    print("=" * 70)
    print(f"Target classes: {cfg.target_classes}")
    print(f"Network: {' → '.join(map(str, cfg.layer_config))}")
    print(f"GA: {cfg.population_size} pop × {cfg.num_generations} gen")
    print(f"Cores: {cfg.n_cores}")
    print("=" * 70)

    # Set random seeds
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    # Create output directory
    os.makedirs(cfg.experiment_output_dir, exist_ok=True)
    plots_dir = os.path.join(cfg.experiment_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Setup data
    data = setup_mnist_data(cfg)

    # Create network structure
    structure = create_snn_structure(cfg)

    # Precompute spike trains
    spike_trains = precompute_spike_trains(data, cfg)

    # Setup fitness function arguments
    fitness_args_base = (
        structure.network,
        structure.connection_map,
        structure.layer_indices,
        cfg.n_classes,
        data,
        data.label_map,
        None,  # Placeholder for eval_indices
        spike_trains,
        cfg
    )
    EVAL_INDICES_ARG_INDEX = 6

    # Initialize GA
    print("\n=== Initializing Genetic Algorithm ===")
    ga = GeneticAlgorithm(
        population_size=cfg.population_size,
        chromosome_length=structure.n_connections,
        fitness_func=evaluate_chromosome_fitness,
        fitness_func_args=fitness_args_base,
        mutation_rate=cfg.mutation_rate,
        mutation_strength=cfg.mutation_strength,
        crossover_rate=cfg.crossover_rate,
        elitism_count=cfg.elitism_count,
        tournament_size=cfg.tournament_size,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max
    )

    # Run GA evolution
    print(f"\n=== Starting GA Evolution ===")
    best_fitness_history = []
    avg_fitness_history = []

    try:
        for generation in range(cfg.num_generations):
            gen_start = time.time()
            print(f"\n--- Generation {generation + 1}/{cfg.num_generations} ---")

            # Sample evaluation examples for this generation
            eval_indices = np.random.choice(
                data.train_indices,
                min(cfg.fitness_eval_examples, len(data.train_indices)),
                replace=False
            )

            # Update eval indices in fitness args
            current_fitness_args = list(ga.fitness_func_args)
            current_fitness_args[EVAL_INDICES_ARG_INDEX] = eval_indices
            ga.fitness_func_args = tuple(current_fitness_args)

            # Evaluate population
            ga.evaluate_population(n_cores=cfg.n_cores, show_progress=True)

            # Process fitness scores
            if np.all(np.isneginf(ga.fitness_scores)):
                best_gen = -np.inf
                avg_gen = -np.inf
                print("All fitness evaluations failed!")
            else:
                valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
                best_gen = np.nanmax(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf
                avg_gen = np.nanmean(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf

            best_fitness_history.append(best_gen)
            avg_fitness_history.append(avg_gen)

            gen_time = time.time() - gen_start
            print(f"Best: {best_gen:.4f} | Avg: {avg_gen:.4f} | Time: {gen_time:.1f}s")

            # Plot progress
            plot_path = os.path.join(plots_dir, f"gen_{generation+1:03d}.png")
            plot_ga_progress(generation + 1, best_fitness_history, avg_fitness_history,
                           plot_path, eval_examples=cfg.fitness_eval_examples)

            # Evolve to next generation
            if generation < cfg.num_generations - 1:
                ga.run_generation()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    # Save final results
    if len(ga.fitness_scores) > 0 and np.any(np.isfinite(ga.fitness_scores)):
        valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
        if np.any(np.isfinite(valid_scores)):
            best_idx = np.nanargmax(valid_scores)
            best_chromosome = ga.population[best_idx]
            best_fitness = valid_scores[best_idx]

            print(f"\n=== Evolution Complete ===")
            print(f"Best fitness: {best_fitness:.4f}")

            save_experiment_results(cfg, structure, best_chromosome, best_fitness)

            # Final plot
            final_plot = os.path.join(cfg.experiment_output_dir, "final_fitness_evolution.png")
            plot_ga_progress(len(best_fitness_history), best_fitness_history,
                           avg_fitness_history, final_plot,
                           eval_examples=cfg.fitness_eval_examples)

    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {cfg.experiment_output_dir}")


def main():
    """Main entry point."""
    multiprocessing.freeze_support()

    args = parse_args()

    # List architectures if requested
    if args.list_archs:
        list_architectures()
        return

    # Setup and run experiment
    cfg = setup_experiment(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
