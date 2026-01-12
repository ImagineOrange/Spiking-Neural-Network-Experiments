"""
Evaluate and visualize trained SNN networks.

This script loads a trained network from run_experiment.py and performs:
1. Full test set evaluation with confusion matrix and metrics
2. Comprehensive visualizations of network structure and activity
3. Single-example deep dive with all possible visualizations

Usage:
    python evaluate_network.py outputs/5class_mnist_ga/final_network
    python evaluate_network.py outputs/5class_mnist_ga/final_network --samples 5000 --animate
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import refactored modules
from config.experiment_config import ExperimentConfig
from mnist_ga.encoding import setup_mnist_data, precompute_spike_trains
from mnist_ga.network_builder import NetworkStructure
from mnist_ga.simulation import run_snn_simulation
from mnist_ga.evaluation import calculate_prediction

# Import legacy utilities
from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized
from LIF_utils.network_vis_utils import Layered_plot_network_connections_sparse, Layered_visualize_activity_layout_grid
from LIF_utils.activity_vis_utils import Layered_plot_activity_and_layer_psth, Layered_plot_layer_wise_raster, Layered_visualize_distance_dependences
from MNIST_utils.MNIST_stimulation_encodings import SNNStimulator

# Import visualization functions
from mnist_ga.vis_utils import (
    plot_evaluation_results,
    plot_mnist_input_with_feature_map,
    plot_activity_summary_and_heatmap,
    plot_weight_distribution_by_source
)

plt.style.use('dark_background')


def plot_pareto_frontier(prediction_errors, spike_counts, save_path):
    """
    Plot Pareto frontier: prediction error vs spike count.

    Good systems lie on the Pareto frontier (optimal tradeoff).

    Args:
        prediction_errors: Binary array (1 if error, 0 if correct)
        spike_counts: Spike counts per sample
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')

    # Scatter plot with transparency
    correct_mask = (prediction_errors == 0)
    incorrect_mask = (prediction_errors == 1)

    if correct_mask.any():
        ax.scatter(spike_counts[correct_mask], prediction_errors[correct_mask],
                  alpha=0.3, s=20, c='cyan', label='Correct', edgecolors='none')

    if incorrect_mask.any():
        ax.scatter(spike_counts[incorrect_mask], prediction_errors[incorrect_mask],
                  alpha=0.3, s=20, c='red', label='Error', edgecolors='none')

    # Calculate binned statistics for Pareto analysis
    n_bins = 20
    spike_bins = np.linspace(spike_counts.min(), spike_counts.max(), n_bins)
    bin_indices = np.digitize(spike_counts, spike_bins)

    bin_centers = []
    bin_error_rates = []

    for i in range(1, n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_centers.append(spike_bins[i-1:i+1].mean())
            bin_error_rates.append(prediction_errors[mask].mean())

    if bin_centers:
        ax.plot(bin_centers, bin_error_rates, 'o-', color='yellow', linewidth=2,
               markersize=8, label='Binned Error Rate', zorder=5)

    # Styling
    ax.set_xlabel('Spike Count (Energy)', fontsize=14, color='white')
    ax.set_ylabel('Prediction Error Rate', fontsize=14, color='white')
    ax.set_title('Pareto Frontier: Surprise vs Energy Tradeoff', fontsize=16,
                fontweight='bold', color='white')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.8, fontsize=11)
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
    plt.close(fig)
    print(f"Saved Pareto frontier plot to {save_path}")


def plot_evolution_trajectory(evolution_data, save_path):
    """
    Plot evolution trajectory over generations.

    Tracks: accuracy ↑, spike count ↓, weight variance ↓
    This shows free energy descent path.

    Args:
        evolution_data: Dict with keys 'generation', 'accuracy', 'spike_count', 'weight_variance'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='#1a1a1a')

    generations = evolution_data['generation']

    # Plot 1: Accuracy over time
    ax1 = axes[0]
    ax1.plot(generations, evolution_data['accuracy'], 'o-', color='cyan',
            linewidth=2, markersize=6, label='Test Accuracy (50 samples)')
    ax1.set_ylabel('Accuracy', fontsize=12, color='white')
    ax1.set_title('Evolution Trajectory: Free Energy Descent', fontsize=16,
                 fontweight='bold', color='white')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='best', framealpha=0.8)
    ax1.set_facecolor('#1a1a1a')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')

    # Plot 2: Spike count over time
    ax2 = axes[1]
    ax2.plot(generations, evolution_data['spike_count'], 'o-', color='orange',
            linewidth=2, markersize=6, label='Mean Spike Count')
    ax2.set_ylabel('Spike Count (Energy)', fontsize=12, color='white')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', framealpha=0.8)
    ax2.set_facecolor('#1a1a1a')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')

    # Plot 3: Weight variance over time
    ax3 = axes[2]
    ax3.plot(generations, evolution_data['weight_variance'], 'o-', color='lime',
            linewidth=2, markersize=6, label='Weight Variance')
    ax3.set_xlabel('Generation', fontsize=12, color='white')
    ax3.set_ylabel('Weight Variance', fontsize=12, color='white')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='best', framealpha=0.8)
    ax3.set_facecolor('#1a1a1a')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
    plt.close(fig)
    print(f"Saved evolution trajectory plot to {save_path}")


class TrainedNetwork:
    """Container for a trained SNN with all its components."""

    def __init__(self, network_dir: str):
        """
        Load trained network from directory.

        Args:
            network_dir: Path to final_network directory from run_experiment.py
        """
        self.network_dir = network_dir
        self._load_config()
        self._load_network_files()
        self._reconstruct_network()

    def _load_config(self):
        """Load experiment configuration - COMPLETE reconstruction from saved JSON."""
        config_path = self._find_config_file()

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct config with ALL saved parameters
        self.cfg = ExperimentConfig()

        # Network architecture
        self.cfg.target_classes = config_dict['target_classes']
        self.cfg.hidden_layers = config_dict.get('hidden_layers', config_dict['layer_config'][1:-1])
        self.cfg.inhibitory_fraction = config_dict.get('inhibitory_fraction', 0.2)

        # Encoding parameters
        self.cfg.encoding_mode = config_dict.get('encoding_mode', 'intensity_to_neuron')
        self.cfg.downsample_factor = config_dict.get('downsample_factor', 4)
        self.cfg.conv_weights_path = config_dict.get('conv_weights_path', '../MNIST_utils/conv_model_weights/conv_model_weights.pth')
        self.cfg.conv_feature_count = config_dict.get('conv_feature_count', 49)

        # Simulation parameters
        self.cfg.sim_duration_ms = config_dict.get('sim_duration_ms', 70.0)
        self.cfg.sim_dt_ms = config_dict.get('sim_dt_ms', 0.1)
        self.cfg.mnist_stim_duration_ms = config_dict.get('mnist_stim_duration_ms', 50.0)
        self.cfg.max_freq_hz = config_dict.get('max_freq_hz', 200.0)
        self.cfg.stim_strength = config_dict.get('stim_strength', 25.0)

        # Network structure parameters
        self.cfg.base_transmission_delay = config_dict.get('base_transmission_delay', 1.0)
        self.cfg.connection_probs = config_dict.get('connection_probs', self.cfg.connection_probs)
        self.cfg.neuron_params = config_dict.get('neuron_params', self.cfg.neuron_params)
        self.cfg.output_mutual_inhibition_weight = config_dict.get('output_mutual_inhibition_weight', -0.05)
        self.cfg.output_mutual_inhibition_delay = config_dict.get('output_mutual_inhibition_delay', 0.1)

        # GA parameters
        self.cfg.population_size = config_dict.get('ga_population', 100)
        self.cfg.num_generations = config_dict.get('ga_generations', 150)
        self.cfg.mutation_rate = config_dict.get('mutation_rate', 0.05)
        self.cfg.mutation_strength = config_dict.get('mutation_strength', 0.01)
        self.cfg.crossover_rate = config_dict.get('crossover_rate', 0.7)
        self.cfg.elitism_count = config_dict.get('elitism_count', 2)
        self.cfg.tournament_size = config_dict.get('tournament_size', 3)
        self.cfg.fitness_eval_examples = config_dict.get('fitness_eval_examples', 1000)

        # Weight bounds
        self.cfg.weight_min = config_dict.get('weight_min', 0.002)
        self.cfg.weight_max = config_dict.get('weight_max', 0.35)

        # Other metadata
        self.cfg.random_seed = config_dict.get('random_seed', 42)
        self.cfg.name = config_dict.get('experiment_name', 'loaded_experiment')

        # Network size info (computed, not part of config)
        self.n_neurons = config_dict['n_neurons']
        self.n_connections = config_dict['n_connections']
        self.best_fitness = config_dict.get('best_fitness', None)

        print(f"Loaded config: {self.cfg.n_classes} classes, architecture: {self.cfg.layer_config}")
        print(f"Encoding: {self.cfg.encoding_mode} (downsample={self.cfg.downsample_factor})")
        print(f"Simulation: {self.cfg.sim_duration_ms}ms @ {self.cfg.sim_dt_ms}ms timestep")
        print(f"Best fitness: {self.best_fitness:.4f}" if self.best_fitness is not None else "Best fitness: N/A")

    def _find_config_file(self) -> str:
        """Find the config JSON file in the network directory."""
        for fname in os.listdir(self.network_dir):
            if fname.endswith('_config.json'):
                return os.path.join(self.network_dir, fname)
        raise FileNotFoundError(f"No config file found in {self.network_dir}")

    def _find_network_files(self, suffix: str) -> str:
        """Find network file with given suffix."""
        for fname in os.listdir(self.network_dir):
            if fname.endswith(suffix):
                return os.path.join(self.network_dir, fname)
        raise FileNotFoundError(f"No file with suffix '{suffix}' found in {self.network_dir}")

    def _load_network_files(self):
        """Load all network state files."""
        print("Loading network files...")

        self.weights_path = self._find_network_files('_weights.npy')
        self.connection_map_path = self._find_network_files('_connection_map.npy')
        self.delays_path = self._find_network_files('_delays.npy')
        self.inhibitory_path = self._find_network_files('_inhibitory.npy')
        self.positions_path = self._find_network_files('_positions.npy')

        # Load arrays
        self.weights = np.load(self.weights_path)
        connection_map_obj = np.load(self.connection_map_path, allow_pickle=True)

        # Convert connection map to list of tuples
        if connection_map_obj.ndim > 0 and isinstance(connection_map_obj[0], np.ndarray):
            self.connection_map = [tuple(pair) for pair in connection_map_obj]
        else:
            self.connection_map = list(connection_map_obj)

        self.delays = np.load(self.delays_path)
        self.is_inhibitory = np.load(self.inhibitory_path)
        self.positions = np.load(self.positions_path, allow_pickle=True).item()

        print(f"Loaded: {len(self.weights)} weights, {len(self.connection_map)} connections")

    def _reconstruct_network(self):
        """Reconstruct the network object."""
        print("Reconstructing network...")

        # Create network with loaded parameters
        self.network = LayeredNeuronalNetworkVectorized(
            n_neurons=self.n_neurons,
            is_inhibitory=self.is_inhibitory,
            **self.cfg.neuron_params
        )

        # Apply delays and weights
        self.network.delays = self.delays.copy()
        self.network.set_weights_sparse(self.weights, self.connection_map)

        # Rebuild graph for visualization
        self.layer_indices = []
        start_idx = 0
        for size in self.cfg.layer_config:
            self.layer_indices.append((start_idx, start_idx + size))
            start_idx += size

        # Add nodes to graph
        self.network.graph.clear()
        for i in range(self.n_neurons):
            is_inhib = self.is_inhibitory[i]
            layer_num = -1
            for lyr_idx, (start, end) in enumerate(self.layer_indices):
                if start <= i < end:
                    layer_num = lyr_idx + 1
                    break
            self.network.graph.add_node(i, is_inhibitory=is_inhib, layer=layer_num)

        # Add edges
        for u, v in self.connection_map:
            if u < self.n_neurons and v < self.n_neurons:
                weight = self.network.weights[u, v]
                delay = self.network.delays[u, v]
                self.network.graph.add_edge(u, v, weight=weight, delay=delay)

        print(f"Network reconstructed: {self.network.graph.number_of_nodes()} nodes, "
              f"{self.network.graph.number_of_edges()} edges")


def evaluate_network(trained_net: TrainedNetwork, n_samples: int = 3000,
                     output_dir: str = None) -> dict:
    """
    Evaluate network on test set with detailed metrics for free energy analysis.

    Args:
        trained_net: TrainedNetwork instance
        n_samples: Number of test samples to evaluate
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics including energy and confidence data
    """
    print(f"\n=== Evaluating Network on {n_samples} Test Samples ===")

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(trained_net.network_dir, "../evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST data
    data = setup_mnist_data(trained_net.cfg)

    # Precompute spike trains
    spike_trains = precompute_spike_trains(data, trained_net.cfg)

    # Evaluate on test set
    test_indices = data.test_indices[:min(n_samples, len(data.test_indices))]

    true_labels = []
    predicted_labels = []
    spike_counts = []  # Total spike count per sample
    confidences = []   # Prediction confidence per sample

    for idx in tqdm(test_indices, desc="Evaluating", ncols=80):
        # Get true label
        true_label = data.get_mapped_label(idx)
        true_labels.append(true_label)

        # Run simulation
        trained_net.network.reset_all()
        activity = run_snn_simulation(
            trained_net.network,
            mnist_input_spikes=spike_trains[idx],
            cfg=trained_net.cfg
        )

        # Get prediction and output spike counts
        pred_label, output_spike_counts = calculate_prediction(
            activity,
            trained_net.layer_indices,
            trained_net.cfg.n_classes
        )
        predicted_labels.append(pred_label)

        # Calculate total energy (all spikes in network)
        total_spikes = sum(len(step_spikes) for step_spikes in activity)
        spike_counts.append(total_spikes)

        # Calculate confidence (margin between correct and best incorrect)
        output_start_idx = trained_net.layer_indices[-1][0]
        if pred_label == true_label and pred_label != -1:
            S_true = output_spike_counts.get(output_start_idx + true_label, 0)
            S_other = 0
            for class_idx in range(trained_net.cfg.n_classes):
                if class_idx != true_label:
                    neuron_idx = output_start_idx + class_idx
                    S_other = max(S_other, output_spike_counts.get(neuron_idx, 0))
            confidence = (S_true - S_other) / (S_true + S_other + 1e-6)
            confidence = max(confidence, 0.0)
        else:
            confidence = 0.0
        confidences.append(confidence)

    # Calculate metrics
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    spike_counts = np.array(spike_counts)
    confidences = np.array(confidences)

    # Overall accuracy
    accuracy = np.mean(predicted_labels == true_labels)

    # Prediction error
    prediction_error = np.mean(predicted_labels != true_labels)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels,
                         labels=list(range(trained_net.cfg.n_classes)))

    # Cohen's kappa
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Prediction Error: {prediction_error:.2%}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Mean Spike Count: {spike_counts.mean():.1f} ± {spike_counts.std():.1f}")
    print(f"Mean Confidence: {confidences.mean():.3f} ± {confidences.std():.3f}")

    # Plot results
    plot_evaluation_results(
        accuracy=accuracy,
        kappa_score=kappa,
        confusion_mat=cm,
        class_labels=[str(c) for c in trained_net.cfg.target_classes],
        save_path=os.path.join(output_dir, "evaluation_summary.png")
    )

    # Plot free energy analysis: Pareto frontier
    plot_pareto_frontier(
        prediction_errors=(predicted_labels != true_labels).astype(int),
        spike_counts=spike_counts,
        save_path=os.path.join(output_dir, "pareto_frontier.png")
    )

    return {
        'accuracy': accuracy,
        'prediction_error': prediction_error,
        'kappa': kappa,
        'confusion_matrix': cm,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'spike_counts': spike_counts,
        'confidences': confidences
    }


def visualize_single_example(trained_net: TrainedNetwork, output_dir: str,
                            animate: bool = False):
    """
    Generate comprehensive visualizations for a single test example.

    Args:
        trained_net: TrainedNetwork instance
        output_dir: Directory to save visualizations
        animate: Whether to generate activity animation GIF
    """
    print(f"\n=== Generating Single Example Visualizations ===")

    # Load data
    data = setup_mnist_data(trained_net.cfg)

    # Create stimulator
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")

    stimulator = SNNStimulator(
        total_time_ms=trained_net.cfg.mnist_stim_duration_ms,
        max_freq_hz=trained_net.cfg.max_freq_hz,
        mode=trained_net.cfg.encoding_mode,
        conv_weights_path=trained_net.cfg.conv_weights_path if trained_net.cfg.encoding_mode == 'conv_feature_to_neuron' else None,
        device=device
    )

    # Pick first test example
    if len(data.test_indices) == 0:
        print("No test samples available for visualization")
        return

    test_idx = data.test_indices[0]
    true_label = data.get_mapped_label(test_idx)
    image = data.get_image(test_idx)

    print(f"Visualizing test sample {test_idx}, true label: {true_label}")

    # Generate spikes
    if trained_net.cfg.encoding_mode == 'intensity_to_neuron':
        from MNIST_utils.MNIST_stimulation_encodings import downsample_image
        if trained_net.cfg.downsample_factor > 1:
            prepared_image = downsample_image(image * 255.0, trained_net.cfg.downsample_factor)
        else:
            prepared_image = image * 255.0
    else:
        prepared_image = image * 255.0

    spike_times = stimulator.generate_spikes(prepared_image)

    # Extract feature map if in conv mode
    feature_map = None
    if trained_net.cfg.encoding_mode == 'conv_feature_to_neuron':
        if hasattr(stimulator, 'extract_feature_map'):
            feature_map = stimulator.extract_feature_map(image * 255.0)

    # Plot input with features
    plot_mnist_input_with_feature_map(
        image=image if trained_net.cfg.downsample_factor == 1 else prepared_image / 255.0,
        label=true_label,
        feature_map=feature_map,
        spike_times_list=spike_times,
        stim_duration_ms=trained_net.cfg.mnist_stim_duration_ms,
        save_path=os.path.join(output_dir, f"vis_input_digit_{true_label}_example.png")
    )

    # Run simulation
    trained_net.network.reset_all()
    activity = run_snn_simulation(
        trained_net.network,
        mnist_input_spikes=spike_times,
        cfg=trained_net.cfg
    )

    # Get prediction
    pred_label, spike_counts = calculate_prediction(
        activity,
        trained_net.layer_indices,
        trained_net.cfg.n_classes
    )

    print(f"Prediction: {pred_label}, True: {true_label}")
    print(f"Output spike counts: {spike_counts}")

    # Generate visualizations
    vis_prefix = os.path.join(output_dir,
                             f"vis_digit_{true_label}_pred_{pred_label}_mode_{trained_net.cfg.encoding_mode}")

    # Activity summary and heatmap
    if activity:
        plot_activity_summary_and_heatmap(
            activity_record=activity,
            n_neurons=trained_net.n_neurons,
            dt=trained_net.cfg.sim_dt_ms,
            total_duration_ms=trained_net.cfg.sim_duration_ms,
            pos=trained_net.positions,
            grid_resolution=(100, 150),
            save_path_prefix=f"{vis_prefix}_activity"
        )

    # Network structure
    print("Generating structure plot...")
    Layered_plot_network_connections_sparse(
        network=trained_net.network,
        pos=trained_net.positions,
        edge_percent=100,
        save_path=f"{vis_prefix}_structure.png"
    )

    # Weight distribution
    print("Generating weight distribution plot...")
    plot_weight_distribution_by_source(
        weights_vector=trained_net.weights,
        connection_map=trained_net.connection_map,
        inhibitory_status_array=trained_net.is_inhibitory,
        save_path=os.path.join(output_dir, "weight_distribution_by_source.png")
    )

    # Distance dependence plots
    if trained_net.positions and trained_net.network.graph.number_of_nodes() > 0:
        print("Generating distance dependence plots...")
        try:
            neuron_idx = trained_net.layer_indices[0][0]  # First neuron of first layer
            w_fig, d_fig = Layered_visualize_distance_dependences(
                network=trained_net.network,
                pos=trained_net.positions,
                neuron_idx=neuron_idx,
                base_transmission_delay=trained_net.cfg.base_transmission_delay,
                save_path_base=f"{vis_prefix}_neuron{neuron_idx}_dist"
            )
            if w_fig: plt.close(w_fig)
            if d_fig: plt.close(d_fig)
        except Exception as e:
            print(f"Warning: Could not generate distance plots: {e}")

    # Activity PSTH and raster
    if trained_net.layer_indices:
        print("Generating activity PSTH plot...")
        Layered_plot_activity_and_layer_psth(
            network=trained_net.network,
            activity_record=activity,
            layer_indices=trained_net.layer_indices,
            dt=trained_net.cfg.sim_dt_ms,
            save_path=f"{vis_prefix}_activity_psth.png"
        )

        print("Generating layer-wise raster plot...")
        Layered_plot_layer_wise_raster(
            network=trained_net.network,
            activity_record=activity,
            layer_indices=trained_net.layer_indices,
            dt=trained_net.cfg.sim_dt_ms,
            save_path=f"{vis_prefix}_raster.png"
        )

    # Animation (optional)
    if animate and trained_net.positions:
        print("Generating activity animation GIF (this may take a while)...")
        Layered_visualize_activity_layout_grid(
            network=trained_net.network,
            pos=trained_net.positions,
            activity_record=activity,
            dt=trained_net.cfg.sim_dt_ms,
            save_path=f"{vis_prefix}_animation.gif",
            fps=25
        )

    print("Single example visualizations complete")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate and visualize trained SNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('network_dir', type=str,
                       help='Path to final_network directory (e.g., outputs/5class_mnist_ga/final_network)')
    parser.add_argument('--samples', type=int, default=3000,
                       help='Number of test samples to evaluate')
    parser.add_argument('--animate', action='store_true',
                       help='Generate activity animation GIF (slow)')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip full evaluation, only generate visualizations')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: [network_dir]/../evaluation)')

    return parser.parse_args()


def main():
    """Main entry point."""
    start_time = time.time()

    args = parse_args()

    # Check that network directory exists
    if not os.path.exists(args.network_dir):
        print(f"Error: Network directory not found: {args.network_dir}")
        print("\nUsage: python evaluate_network.py outputs/experiment_name/final_network")
        return

    print("=" * 70)
    print("SNN Network Evaluation and Visualization")
    print("=" * 70)

    try:
        # Load trained network
        trained_net = TrainedNetwork(args.network_dir)

        # Determine output directory
        output_dir = args.output
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(args.network_dir), "evaluation")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Output directory: {output_dir}")

        # Run evaluation
        if not args.skip_eval:
            results = evaluate_network(
                trained_net,
                n_samples=args.samples,
                output_dir=output_dir
            )

        # Generate visualizations
        visualize_single_example(
            trained_net,
            output_dir=output_dir,
            animate=args.animate
        )

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Evaluation complete in {elapsed/60:.1f} minutes")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
