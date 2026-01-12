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
from mnist_ga.evaluation import evaluate_chromosome_fitness, calculate_prediction
from mnist_ga.simulation import run_snn_simulation
from mnist_ga.visualization import plot_ga_progress
from LIF_objects.GeneticAlgorithm import GeneticAlgorithm

# Import evaluation functions
from evaluate_network import TrainedNetwork, evaluate_network, visualize_single_example


def quick_accuracy_check(chromosome, structure, data, spike_trains, cfg, n_samples=50):
    """
    Quick accuracy check on a small sample (for progress monitoring).

    Args:
        chromosome: Weight vector to evaluate
        structure: NetworkStructure instance
        data: MNISTDataset
        spike_trains: Precomputed spike trains
        cfg: ExperimentConfig
        n_samples: Number of samples to test (default: 50 for speed)

    Returns:
        Tuple of (accuracy, mean_spike_count):
            - accuracy: Accuracy as a float (0.0 to 1.0)
            - mean_spike_count: Average total spikes per sample
    """
    # Set weights
    structure.network.reset_all()
    structure.network.set_weights_sparse(chromosome, structure.connection_map)

    # Sample random test examples
    test_sample_indices = np.random.choice(
        data.test_indices,
        min(n_samples, len(data.test_indices)),
        replace=False
    )

    correct = 0
    total = 0
    total_spikes = 0

    for idx in test_sample_indices:
        try:
            true_label = data.get_mapped_label(idx)

            # Run simulation
            structure.network.reset_all()
            activity = run_snn_simulation(
                structure.network,
                mnist_input_spikes=spike_trains[idx],
                cfg=cfg,
                show_progress=False
            )

            # Count total spikes in this trial
            # activity is a list of arrays, one per timestep, containing neuron indices that spiked
            sample_spike_count = sum(len(active_indices) for active_indices in activity)
            total_spikes += sample_spike_count

            # Get prediction
            pred_label, _ = calculate_prediction(
                activity,
                structure.layer_indices,
                cfg.n_classes
            )

            if pred_label == true_label:
                correct += 1
            total += 1
        except:
            continue

    accuracy = correct / total if total > 0 else 0.0
    mean_spike_count = total_spikes / total if total > 0 else 0.0
    return accuracy, mean_spike_count


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
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip automatic evaluation after training')
    parser.add_argument('--eval-samples', type=int, default=None,
                       help='Number of test samples for evaluation (default: all test samples)')
    parser.add_argument('--animate', action='store_true',
                       help='Generate activity animation GIF during evaluation (slow)')

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

    # Save experiment config to output directory
    config_save_path = os.path.join(cfg.experiment_output_dir, "experiment_config.json")
    config_dict = {
        # Experiment metadata
        "experiment_name": cfg.name,
        "random_seed": cfg.random_seed,
        "output_dir": cfg.output_dir,

        # Target classes
        "target_classes": cfg.target_classes,

        # Encoding parameters
        "encoding_mode": cfg.encoding_mode,
        "downsample_factor": cfg.downsample_factor,
        "conv_weights_path": cfg.conv_weights_path,
        "conv_feature_count": cfg.conv_feature_count,

        # Network architecture
        "hidden_layers": cfg.hidden_layers,
        "layer_config": cfg.layer_config,
        "inhibitory_fraction": cfg.inhibitory_fraction,

        # Connection probabilities
        "connection_probs": cfg.connection_probs,
        "base_transmission_delay": cfg.base_transmission_delay,

        # Neuron parameters
        "neuron_params": cfg.neuron_params,

        # Simulation parameters
        "sim_duration_ms": cfg.sim_duration_ms,
        "sim_dt_ms": cfg.sim_dt_ms,
        "mnist_stim_duration_ms": cfg.mnist_stim_duration_ms,
        "max_freq_hz": cfg.max_freq_hz,
        "stim_strength": cfg.stim_strength,

        # GA parameters
        "population_size": cfg.population_size,
        "num_generations": cfg.num_generations,
        "mutation_rate": cfg.mutation_rate,
        "mutation_strength": cfg.mutation_strength,
        "crossover_rate": cfg.crossover_rate,
        "elitism_count": cfg.elitism_count,
        "tournament_size": cfg.tournament_size,
        "fitness_eval_examples": cfg.fitness_eval_examples,
        "test_eval_examples": cfg.test_eval_examples,
        "fitness_alpha": cfg.fitness_alpha,

        # Weight parameters
        "weight_min": cfg.weight_min,
        "weight_max": cfg.weight_max,
        "init_weight_mode": cfg.init_weight_mode,
        "init_weight_std": cfg.init_weight_std,

        # Computational
        "n_cores": cfg.n_cores,

        # Output mutual inhibition
        "output_mutual_inhibition_weight": cfg.output_mutual_inhibition_weight,
        "output_mutual_inhibition_delay": cfg.output_mutual_inhibition_delay,
    }
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Saved experiment config: {config_save_path}")

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
        weight_max=cfg.weight_max,
        init_weight_mode=cfg.init_weight_mode,
        init_weight_std=cfg.init_weight_std
    )

    # Run GA evolution
    print(f"\n=== Starting GA Evolution ===")
    best_fitness_history = []
    avg_fitness_history = []

    # Track evolution metrics for free energy analysis
    evolution_data = {
        'generation': [],
        'accuracy': [],
        'spike_count': [],
        'weight_variance': []
    }

    # Track overall best across all generations
    overall_best_fitness = -np.inf
    overall_best_chromosome = None

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

            # Update overall best if this generation has a better solution
            if best_gen > overall_best_fitness:
                overall_best_fitness = best_gen
                best_idx = np.nanargmax(valid_scores)
                overall_best_chromosome = ga.population[best_idx].copy()

            # Quick accuracy check on test set (every generation)
            test_accuracy = 0.0
            mean_spike_count = 0.0
            if overall_best_chromosome is not None:
                test_accuracy, mean_spike_count = quick_accuracy_check(
                    overall_best_chromosome, structure, data, spike_trains, cfg, n_samples=cfg.test_eval_examples
                )

            # Track evolution metrics every generation
            if overall_best_chromosome is not None:
                evolution_data['generation'].append(generation + 1)
                evolution_data['accuracy'].append(test_accuracy)  # Actual test accuracy
                evolution_data['spike_count'].append(mean_spike_count)  # Mean spike count from test samples
                evolution_data['weight_variance'].append(np.var(overall_best_chromosome))

            gen_time = time.time() - gen_start
            print(f"Fitness: {best_gen:.4f} | Avg: {avg_gen:.4f} | Acc: {test_accuracy:.1%} | Spikes: {mean_spike_count:.1f} | Wt Var: {np.var(overall_best_chromosome) if overall_best_chromosome is not None else 0:.6f} | Time: {gen_time:.1f}s")

            # Plot progress
            plot_path = os.path.join(plots_dir, f"gen_{generation+1:03d}.png")
            plot_ga_progress(generation + 1, best_fitness_history, avg_fitness_history,
                           plot_path, eval_examples=cfg.fitness_eval_examples)

            # Evolve to next generation
            if generation < cfg.num_generations - 1:
                ga.run_generation()

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    # Save final results using overall best across all generations
    if overall_best_chromosome is not None and np.isfinite(overall_best_fitness):
        print(f"\n=== Evolution Complete ===")
        print(f"Best fitness: {overall_best_fitness:.4f}")

        save_experiment_results(cfg, structure, overall_best_chromosome, overall_best_fitness)

        # Save evolution data for free energy analysis
        evolution_data_path = os.path.join(cfg.experiment_output_dir, "evolution_data.json")
        with open(evolution_data_path, 'w') as f:
            json.dump(evolution_data, f, indent=4)
        print(f"✓ Saved evolution data: {evolution_data_path}")

        # Final plot
        final_plot = os.path.join(cfg.experiment_output_dir, "final_fitness_evolution.png")
        plot_ga_progress(len(best_fitness_history), best_fitness_history,
                       avg_fitness_history, final_plot,
                       eval_examples=cfg.fitness_eval_examples)

        # Plot evolution trajectory (if we have data)
        if len(evolution_data['generation']) > 0:
            from evaluate_network import plot_evolution_trajectory
            trajectory_plot = os.path.join(cfg.experiment_output_dir, "evolution_trajectory.png")
            plot_evolution_trajectory(evolution_data, trajectory_plot)

    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {cfg.experiment_output_dir}")

    return overall_best_fitness if overall_best_chromosome is not None else None


def run_evaluation(network_dir: str, n_samples: int = None, animate: bool = False):
    """
    Run full evaluation on trained network.

    Args:
        network_dir: Path to final_network directory
        n_samples: Number of test samples to evaluate (None = all)
        animate: Whether to generate activity animation GIF
    """
    print("\n" + "=" * 70)
    print("AUTOMATIC POST-TRAINING EVALUATION")
    print("=" * 70)

    eval_start = time.time()

    try:
        # Load trained network
        trained_net = TrainedNetwork(network_dir)

        # Determine output directory
        output_dir = os.path.join(os.path.dirname(network_dir), "evaluation")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Output directory: {output_dir}")

        # Determine number of samples
        if n_samples is None:
            # Use all test samples
            from mnist_ga.encoding import setup_mnist_data
            data = setup_mnist_data(trained_net.cfg)
            n_samples = len(data.test_indices)
            print(f"Evaluating on all {n_samples} test samples")
        else:
            print(f"Evaluating on {n_samples} test samples")

        # Run evaluation
        results = evaluate_network(
            trained_net,
            n_samples=n_samples,
            output_dir=output_dir
        )

        # Generate visualizations
        visualize_single_example(
            trained_net,
            output_dir=output_dir,
            animate=animate
        )

        eval_time = time.time() - eval_start
        print(f"\n{'=' * 70}")
        print(f"Evaluation complete in {eval_time/60:.1f} minutes")
        print(f"Test Accuracy: {results['accuracy']:.2%}")
        print(f"Cohen's Kappa: {results['kappa']:.4f}")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    best_fitness = run_experiment(cfg)

    # Run automatic evaluation if training completed successfully
    if best_fitness is not None and not args.skip_eval:
        network_dir = os.path.join(cfg.experiment_output_dir, "final_network")
        if os.path.exists(network_dir):
            run_evaluation(
                network_dir=network_dir,
                n_samples=args.eval_samples,
                animate=args.animate
            )
        else:
            print(f"\nWarning: Network directory not found: {network_dir}")
            print("Skipping evaluation.")


if __name__ == "__main__":
    main()
