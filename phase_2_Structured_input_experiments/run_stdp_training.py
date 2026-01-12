"""
Main entry point for MNIST STDP online learning experiments.

This script implements supervised spike-timing-dependent plasticity (STDP)
for learning MNIST digit classification through online presentation of examples.

Usage:
    python run_stdp_training.py
    python run_stdp_training.py --name 2class_stdp_test --epochs 10
    python run_stdp_training.py --arch small_2class --epochs 5
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.stdp_config import get_stdp_config
from config.network_architectures import get_architecture, list_architectures
from mnist_ga.network_builder import create_snn_structure
from mnist_ga.encoding import setup_mnist_data, precompute_spike_trains
from mnist_ga.simulation import run_stdp_phased_simulation
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MNIST STDP Online Learning Experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--arch', type=str, default=None,
                       help='Pre-defined architecture name (e.g., small_2class)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (used for output directory)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--list-archs', action='store_true',
                       help='List available architectures and exit')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Do not shuffle examples each epoch')
    parser.add_argument('--no-homeostatic', action='store_true',
                       help='Disable homeostatic normalization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def setup_stdp_experiment(args):
    """
    Setup STDP experiment configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        STDPConfig instance
    """
    # Start with architecture if specified
    if args.arch:
        arch_config = get_architecture(args.arch)
        cfg = get_stdp_config(**arch_config)
        print(f"Using architecture: {args.arch}")
    else:
        cfg = get_stdp_config()

    # Override with command line arguments
    if args.name:
        cfg.name = args.name
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.no_shuffle:
        cfg.shuffle_each_epoch = False
    if args.no_homeostatic:
        cfg.homeostatic_normalization = False
    if args.seed:
        cfg.random_seed = args.seed

    return cfg


def save_checkpoint(cfg, structure, epoch, example_idx, metrics):
    """
    Save network checkpoint during training.

    Args:
        cfg: STDPConfig
        structure: NetworkStructure
        epoch: Current epoch number
        example_idx: Current example index
        metrics: Training metrics dictionary
    """
    checkpoint_dir = os.path.join(cfg.experiment_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = f"checkpoint_epoch_{epoch:03d}_ex_{example_idx:06d}"

    # Save weights
    weights = structure.network.get_weights_sparse(structure.connection_map)
    np.save(os.path.join(checkpoint_dir, f"{checkpoint_name}_weights.npy"), weights)

    # Save metrics
    metrics_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = value.item()
            else:
                metrics_serializable[key] = value
        json.dump(metrics_serializable, f, indent=2)


def evaluate_on_validation(cfg, structure, data, spike_trains, n_samples=200):
    """
    Quick validation accuracy check.

    Args:
        cfg: STDPConfig
        structure: NetworkStructure
        data: MNISTDataset
        spike_trains: Precomputed spike trains
        n_samples: Number of validation samples

    Returns:
        Validation accuracy (float)
    """
    if len(data.test_indices) == 0:
        return 0.0

    # Sample validation examples
    val_indices = np.random.choice(
        data.test_indices,
        min(n_samples, len(data.test_indices)),
        replace=False
    )

    correct = 0
    total = 0

    # Temporarily disable STDP
    structure.network.stdp_enabled = False
    structure.network.learning_phase = False

    for idx in val_indices:
        try:
            true_label = data.get_mapped_label(idx)

            # Reset and run simulation
            structure.network.reset_transient_state()

            result = run_stdp_phased_simulation(
                structure.network,
                mnist_input_spikes=spike_trains[idx],
                true_label=true_label,
                cfg=cfg,
                layer_indices=structure.layer_indices,
                n_classes=cfg.n_classes,
                show_progress=False
            )

            if result['predicted_class'] == true_label:
                correct += 1
            total += 1

        except Exception as e:
            print(f"Error evaluating example {idx}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def run_stdp_training(cfg):
    """
    Run complete STDP online learning experiment.

    Args:
        cfg: STDPConfig instance
    """
    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"MNIST STDP Online Learning: {cfg.name}")
    print("=" * 70)
    print(f"Target classes: {cfg.target_classes}")
    print(f"Network: {' → '.join(map(str, cfg.layer_config))}")
    print(f"Training: {cfg.num_epochs} epochs")
    print(cfg.get_stdp_summary())
    print("=" * 70)

    # Set random seeds
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)

    # Create output directory
    os.makedirs(cfg.experiment_output_dir, exist_ok=True)

    # Save configuration
    config_save_path = os.path.join(cfg.experiment_output_dir, "stdp_config.json")
    config_dict = {
        'experiment_name': cfg.name,
        'random_seed': cfg.random_seed,
        'target_classes': cfg.target_classes,
        'layer_config': cfg.layer_config,
        'num_epochs': cfg.num_epochs,
        'stdp_a_plus': cfg.stdp_a_plus,
        'stdp_a_minus': cfg.stdp_a_minus,
        'stdp_tau_pre': cfg.stdp_tau_pre,
        'stdp_tau_post': cfg.stdp_tau_post,
        'phase_durations': {
            'input': cfg.stdp_input_period,
            'readout': cfg.stdp_readout_period,
            'learning': cfg.stdp_learning_period,
            'rest': cfg.stdp_rest_period
        },
        'homeostatic_normalization': cfg.homeostatic_normalization,
        'shuffle_each_epoch': cfg.shuffle_each_epoch
    }
    with open(config_save_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"✓ Saved config: {config_save_path}")

    # Setup data
    print("\n=== Loading MNIST Data ===")
    data = setup_mnist_data(cfg)

    # Create network structure
    print("\n=== Creating Network Structure ===")
    structure = create_snn_structure(cfg)

    # Set STDP parameters on network
    structure.network.set_stdp_params(
        a_plus=cfg.stdp_a_plus,
        a_minus=cfg.stdp_a_minus,
        tau_pre=cfg.stdp_tau_pre,
        tau_post=cfg.stdp_tau_post,
        w_min=cfg.weight_min,
        w_max=cfg.weight_max
    )

    # Precompute spike trains
    print("\n=== Precomputing Spike Trains ===")
    spike_trains = precompute_spike_trains(data, cfg)

    # Training metrics
    training_metrics = {
        'epoch': [],
        'example_count': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'mean_weight': [],
        'weight_std': [],
        'correct_count': [],
        'total_count': []
    }

    # Start training
    print(f"\n=== Starting STDP Online Training ===")
    total_examples_seen = 0
    rolling_window = []  # For tracking recent accuracy

    try:
        for epoch in range(cfg.num_epochs):
            epoch_start = time.time()
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
            print(f"{'='*70}")

            # Get training indices
            train_indices = data.train_indices.copy()

            # Shuffle if enabled
            if cfg.shuffle_each_epoch:
                np.random.shuffle(train_indices)

            # Epoch statistics
            epoch_correct = 0
            epoch_total = 0

            # Progress bar for epoch
            pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}", ncols=100)

            for example_idx, data_idx in enumerate(pbar):
                try:
                    true_label = data.get_mapped_label(data_idx)

                    # Reset transient state
                    structure.network.reset_transient_state()

                    # Run phased simulation with STDP learning
                    result = run_stdp_phased_simulation(
                        structure.network,
                        mnist_input_spikes=spike_trains[data_idx],
                        true_label=true_label,
                        cfg=cfg,
                        layer_indices=structure.layer_indices,
                        n_classes=cfg.n_classes,
                        show_progress=False
                    )

                    # Update statistics
                    if result['is_correct']:
                        epoch_correct += 1
                    epoch_total += 1
                    total_examples_seen += 1

                    # Rolling window accuracy (last 100 examples)
                    rolling_window.append(1 if result['is_correct'] else 0)
                    if len(rolling_window) > 100:
                        rolling_window.pop(0)
                    rolling_acc = np.mean(rolling_window) if rolling_window else 0.0

                    # Apply homeostatic normalization
                    if cfg.homeostatic_normalization and cfg.normalize_every_example:
                        structure.network.apply_homeostatic_normalization(
                            connection_map=structure.connection_map,
                            target_sum_per_neuron=cfg.homeostatic_target_sum
                        )

                    # Update progress bar
                    pbar.set_postfix({
                        'acc': f"{rolling_acc:.3f}",
                        'correct': f"{epoch_correct}/{epoch_total}"
                    })

                    # Validation check
                    if (example_idx + 1) % cfg.validation_frequency == 0:
                        val_acc = evaluate_on_validation(
                            cfg, structure, data, spike_trains,
                            n_samples=cfg.validation_samples
                        )

                        # Track metrics
                        weights = structure.network.get_weights_sparse(structure.connection_map)
                        training_metrics['epoch'].append(epoch + 1)
                        training_metrics['example_count'].append(total_examples_seen)
                        training_metrics['train_accuracy'].append(rolling_acc)
                        training_metrics['val_accuracy'].append(val_acc)
                        training_metrics['mean_weight'].append(float(np.mean(np.abs(weights))))
                        training_metrics['weight_std'].append(float(np.std(weights)))
                        training_metrics['correct_count'].append(epoch_correct)
                        training_metrics['total_count'].append(epoch_total)

                        print(f"\n  [Ex {total_examples_seen}] "
                              f"Train: {rolling_acc:.3f} | Val: {val_acc:.3f} | "
                              f"Weights: μ={np.mean(np.abs(weights)):.4f}, "
                              f"σ={np.std(weights):.4f}")

                except Exception as e:
                    print(f"\nError processing example {data_idx}: {e}")
                    continue

            # End of epoch
            epoch_time = time.time() - epoch_start
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Accuracy: {epoch_accuracy:.4f} ({epoch_correct}/{epoch_total})")
            print(f"  Time: {epoch_time:.1f}s")

            # Save checkpoint
            if cfg.save_checkpoints and cfg.checkpoint_every_epoch:
                save_checkpoint(cfg, structure, epoch + 1, total_examples_seen, {
                    'epoch': epoch + 1,
                    'accuracy': epoch_accuracy,
                    'examples_seen': total_examples_seen
                })
                print(f"  ✓ Checkpoint saved")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save final trained network
    print(f"\n{'='*70}")
    print("=== Saving Final Trained Network ===")
    print(f"{'='*70}")

    final_dir = os.path.join(cfg.experiment_output_dir, "final_network")
    os.makedirs(final_dir, exist_ok=True)

    base_name = f"stdp_trained_{cfg.n_classes}class"

    # Save weights
    final_weights = structure.network.get_weights_sparse(structure.connection_map)
    np.save(os.path.join(final_dir, f"{base_name}_weights.npy"), final_weights)
    print(f"✓ Saved weights")

    # Save network structure (same format as GA for compatibility)
    np.save(os.path.join(final_dir, f"{base_name}_connection_map.npy"),
            np.array(structure.connection_map, dtype=object))
    np.save(os.path.join(final_dir, f"{base_name}_delays.npy"), structure.network.delays)
    np.save(os.path.join(final_dir, f"{base_name}_inhibitory.npy"), structure.network.is_inhibitory)
    np.save(os.path.join(final_dir, f"{base_name}_positions.npy"), structure.positions)
    print(f"✓ Saved network structure")

    # Save configuration
    final_config = {
        'experiment_name': cfg.name,
        'target_classes': cfg.target_classes,
        'layer_config': cfg.layer_config,
        'n_neurons': structure.n_neurons,
        'n_connections': structure.n_connections,
        'training_method': 'stdp',
        'num_epochs': cfg.num_epochs,
        'total_examples_seen': total_examples_seen,
        'stdp_params': {
            'a_plus': cfg.stdp_a_plus,
            'a_minus': cfg.stdp_a_minus,
            'tau_pre': cfg.stdp_tau_pre,
            'tau_post': cfg.stdp_tau_post
        }
    }
    with open(os.path.join(final_dir, f"{base_name}_config.json"), 'w') as f:
        json.dump(final_config, f, indent=4)
    print(f"✓ Saved configuration")

    # Save training metrics
    metrics_path = os.path.join(cfg.experiment_output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    print(f"✓ Saved training metrics")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total examples: {total_examples_seen}")
    print(f"Output: {cfg.experiment_output_dir}")
    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    args = parse_args()

    # List architectures if requested
    if args.list_archs:
        list_architectures()
        return

    # Setup and run experiment
    cfg = setup_stdp_experiment(args)
    run_stdp_training(cfg)


if __name__ == "__main__":
    main()
