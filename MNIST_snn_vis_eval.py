# visualize_trained_snn_v2.py # Renamed for clarity
# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm # For better color mapping of probabilities
import os
import time
import pandas as pd

import random
import json
import traceback
import seaborn as sns
import torch # Import torch
from collections import deque, Counter # Added Counter for confusion matrix
from tqdm import tqdm
# Added cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from MNIST_utils.MNIST_stimulation_encodings import downsample_image


# --- Style ---
plt.style.use('dark_background')

# --- Required Custom Imports ---

# LIF Objects (Need the specific network class used for evaluation)
# kept these try blocks for debugging's sake... will likely remove later

try:
    # Assuming evaluation uses the Vectorized version
    from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized
    print("Imported LayeredNeuronalNetworkVectorized.")
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import 'LayeredNeuronalNetworkVectorized' from 'LIF_objects'.")
    print("Please ensure this file exists and the class name is correct.")
    exit()

try:
    # SNNStimulator now includes ConvNet definition needed if loading that mode
    # It's assumed SNNStimulator internally imports ConvNet if needed for its mode
    from MNIST_utils.MNIST_stimulation_encodings import MNIST_loader, SNNStimulator
    # Make ConvNet definition available if needed for type hints or direct use (unlikely needed here now)
    # from MNIST_utils.MNIST_stimulation_encodings import ConvNet # <-- Added import
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import from 'MNIST_stimulation_encodings.py'.")
    exit()

# Visualization Utilities
from LIF_utils.network_vis_utils import Layered_plot_network_connections_sparse, Layered_visualize_activity_layout_grid
from LIF_utils.activity_vis_utils import Layered_plot_activity_and_layer_psth, Layered_plot_layer_wise_raster, Layered_visualize_distance_dependences
print("Imported visualization functions from LIF_utils.")


# --- Functions Implemented Directly (Keep definitions as provided previously) ---

def load_trained_data(weights_file, connections_file, delays_file, inhib_status_file, pos_file):
    """Loads the necessary network state arrays from .npy files."""
    print("Loading trained data arrays...")
    try:
        weights_vector = np.load(weights_file)
        connection_map_obj = np.load(connections_file, allow_pickle=True)
        # Convert connection_map back to list of tuples if saved as object array
        if connection_map_obj.ndim > 0 and isinstance(connection_map_obj[0], np.ndarray):
             connection_map = [tuple(pair) for pair in connection_map_obj]
        else: # Assume it was saved in a way that loads as list or compatible
             connection_map = list(connection_map_obj)

        delays_matrix = np.load(delays_file) # Expecting full matrix now
        inhibitory_status_array = np.load(inhib_status_file)
        # Load positions dictionary
        pos_data = np.load(pos_file, allow_pickle=True).item() # Load positions as dictionary

        print(f"Loaded weights: {weights_vector.shape}, map: {len(connection_map)} pairs, delays: {delays_matrix.shape}, inhib: {inhibitory_status_array.shape}, pos: {len(pos_data)} entries")
        # Return loaded data including positions
        return weights_vector, connection_map, delays_matrix, inhibitory_status_array, pos_data
    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Check paths.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading trained data: {e}")
        traceback.print_exc()
        return None, None, None, None, None

def create_network_from_data_vectorized(n_neurons_total, neuron_params, dt,
                                       inhibitory_status_array, delay_matrix, # Fixed structural info
                                       connection_map, weights_vector, # Fixed topology + weights
                                       random_seed=None):
    """
    Creates a LayeredNeuronalNetworkVectorized instance from loaded configuration
    and state data. Applies the fixed structure and loaded weights.
    """
    print("Reconstructing network from loaded data...")
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed) # Ensure consistency if any random init remains

    try:
        # --- 1. Instantiate the network with fixed neuron properties ---
        # Ensure LAYERS_CONFIG is inside neuron_params when calling this
        network = LayeredNeuronalNetworkVectorized(
            n_neurons=n_neurons_total,
            is_inhibitory=inhibitory_status_array, # CRITICAL: Use loaded array
            **neuron_params # Pass other neuron parameters including layers_config
        )

        # --- 2. Apply the FIXED delays matrix ---
        if network.delays.shape != delay_matrix.shape:
             print(f"Warning: Loaded delay matrix shape {delay_matrix.shape} != network delay shape {network.delays.shape}. Trying to apply anyway.")
        network.delays = delay_matrix.copy() # Apply the loaded fixed delays

        # --- 3. Apply the FIXED weights using the connection map ---
        network.set_weights_sparse(weights_vector, connection_map)

        # --- 4. Rebuild graph for visualization consistency ---
        network.graph.clear_edges()
        print("Populating graph structure based on connection map (for visualization)...")
        layer_indices_from_config = []
        layers_config = neuron_params.get('layers_config', []) # Get layers from params
        if layers_config:
             start_idx = 0
             for size in layers_config:
                 layer_indices_from_config.append((start_idx, start_idx + size))
                 start_idx += size

        for i in range(n_neurons_total):
             is_inhib = inhibitory_status_array[i]
             layer_num = -1
             for lyr_idx, (start, end) in enumerate(layer_indices_from_config):
                  if start <= i < end: layer_num = lyr_idx + 1; break
             network.graph.add_node(i, is_inhibitory=is_inhib, layer=layer_num)

        for u, v in connection_map:
            if u < n_neurons_total and v < n_neurons_total:
                 weight = network.weights[u, v]
                 delay = network.delays[u, v]
                 network.graph.add_edge(u, v, weight=weight, delay=delay)
            else:
                 print(f"Warning: Invalid index ({u},{v}) in connection map during graph rebuild.")

        print(f"Network reconstruction complete. Graph has {network.graph.number_of_nodes()} nodes, {network.graph.number_of_edges()} edges.")
        return network, layer_indices_from_config

    except Exception as e:
        print(f"Error reconstructing network: {e}")
        traceback.print_exc()
        return None, []

def run_vectorized_simulation(network: LayeredNeuronalNetworkVectorized,
                                 duration=1000.0, dt=0.1,
                                 stim_interval_strength=10, # Renamed for clarity
                                 stim_pulse_duration_ms=1.0,
                                 mnist_input_spikes=None,
                                 show_progress=False):
    """
    Runs the network simulation using vectorized updates.
    Focuses on MNIST input, ignoring other stimulation types.
    """
    # Pre-process MNIST spike times into step-based dictionary
    mnist_spikes_by_step = {}
    if mnist_input_spikes is not None:
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                if 0 <= step_index < int(duration / dt):
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    mnist_spikes_by_step[step_index].append(neuron_idx)

    # Initialize simulation state
    n_steps = int(duration / dt)
    activity_record = []
    ongoing_stimulations = {} # {neuron_idx: end_time_ms}

    # Setup optional progress bar
    sim_loop_iterator = range(n_steps)
    if show_progress:
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step (Vis)", leave=False, ncols=80)

    # Vectorized array for external stimulus conductances
    current_stim_conductances = np.zeros(network.n_neurons)

    # --- Main Simulation Loop ---
    for step in sim_loop_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0) # Reset conductances
        newly_stimulated_indices_this_step = set()

        # Apply MNIST stimulation if spikes occur at this step
        if step in mnist_spikes_by_step:
            neurons_spiking_now = mnist_spikes_by_step[step]
            stim_end_time = current_time + stim_pulse_duration_ms
            for neuron_idx in neurons_spiking_now:
                if 0 <= neuron_idx < network.n_neurons:
                    if neuron_idx not in ongoing_stimulations:
                         ongoing_stimulations[neuron_idx] = stim_end_time
                         newly_stimulated_indices_this_step.add(neuron_idx)
                    # Apply conductance (will be maintained/decayed below)
                    current_stim_conductances[neuron_idx] = stim_interval_strength

        # Manage ongoing stimulation pulses (decay/end)
        expired_stims = set()
        for neuron_idx, end_time in list(ongoing_stimulations.items()): # Iterate over copy
            if current_time >= end_time:
                expired_stims.add(neuron_idx)
                if neuron_idx in ongoing_stimulations: # Check again before deleting
                    del ongoing_stimulations[neuron_idx]
                current_stim_conductances[neuron_idx] = 0.0 # Ensure conductance is off
            elif 0 <= neuron_idx < network.n_neurons: # If not expired
                 # Maintain conductance if pulse is active and wasn't just started
                 if neuron_idx not in newly_stimulated_indices_this_step:
                    current_stim_conductances[neuron_idx] = stim_interval_strength

        # Set external stimulus conductances on the network object
        network.external_stim_g[:] = current_stim_conductances
        # Update the network state and get spiking neurons
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

    # Cleanup after loop
    network.external_stim_g.fill(0.0)
    if show_progress and isinstance(sim_loop_iterator, tqdm):
         sim_loop_iterator.close()

    return activity_record

def classify_output_total_sim(activity_record, output_layer_indices, n_classes):
    """
    Classifies based on TOTAL spike counts in the output layer during the sim.
    MODIFIED: In case of a tie, selects the neuron with the lowest index
              among the tied neurons (similar to GA fitness evaluation).
    """
    if not output_layer_indices:
         print("Warning: Output layer indices not provided for classification.")
         return -1, {}
    output_start_idx, output_end_idx = output_layer_indices

    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}
    total_output_spikes = 0

    for step_spikes in activity_record:
        # Ensure step_spikes is iterable, handle potential non-iterable types if necessary
        if hasattr(step_spikes, '__iter__'):
            for neuron_idx in step_spikes:
                if output_start_idx <= neuron_idx < output_end_idx:
                    output_spike_counts[neuron_idx] += 1
                    total_output_spikes += 1

    predicted_label = -1
    if total_output_spikes > 0:
        # Find the neuron index with the maximum spikes.
        # If there's a tie, max() with a key returns the *first* key --- network may learn to spike earlier to classify
        # encountered with the maximum value (typically the lowest index here).
        predicted_neuron_idx = max(output_spike_counts, key=output_spike_counts.get)
        # Directly assign the label based on the result of max()
        predicted_label = predicted_neuron_idx - output_start_idx # Convert index to label


    return predicted_label, output_spike_counts

def plot_mnist_input_with_feature_map(image, label, feature_map, spike_times_list, stim_duration_ms, save_path):
    """Creates a 4-panel figure: MNIST image, 7x7 feature map, input spike raster."""
    print(f"Generating MNIST input + feature map visualization...")
    num_input_neurons = len(spike_times_list)
    total_spikes = sum(len(spikes) for spikes in spike_times_list)

    fig = plt.figure(figsize=(12, 9), facecolor='#1a1a1a') # Adjusted size
    # Create a 2x2 grid
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], hspace=0.4, wspace=0.3)

    # Panel 1: Input Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap='gray', interpolation='nearest')
    ax_img.set_title(f"Input Image (Label: {label})", color='white')
    ax_img.set_xticks([]); ax_img.set_yticks([])
    [spine.set_color('gray') for spine in ax_img.spines.values()]

    # Panel 2: Feature Map
    ax_feat = fig.add_subplot(gs[0, 1])
    if feature_map is not None and feature_map.shape == (7, 7):
        im = ax_feat.imshow(feature_map, cmap='viridis', interpolation='nearest')
        ax_feat.set_title("CNN 7x7 Feature Map", color='white')
        plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04) # Add colorbar
    else:
        ax_feat.text(0.5, 0.5, "Feature Map\nNot Available", color='grey', ha='center', va='center')
        ax_feat.set_title("CNN Feature Map", color='white')
    ax_feat.set_xticks([]); ax_feat.set_yticks([])
    [spine.set_color('gray') for spine in ax_feat.spines.values()]

    # Panel 3: Spike Raster (Spanning bottom row)
    ax_raster = fig.add_subplot(gs[1, :])
    raster_times = []
    raster_neurons = []
    for n_idx, spikes in enumerate(spike_times_list):
        if spikes:
             for t in spikes:
                 raster_times.append(t)
                 raster_neurons.append(n_idx)
    if raster_times: ax_raster.scatter(raster_times, raster_neurons, s=5, color='white', alpha=0.8, marker='|')
    ax_raster.set_title(f"Input Spike Encoding ({total_spikes} spikes)", color='white')
    ax_raster.set_xlabel("Time (ms)", color='white')
    ax_raster.set_ylabel(f"SNN Input Neuron Index ({num_input_neurons})", color='white')
    ax_raster.set_xlim(0, stim_duration_ms)
    ax_raster.set_ylim(-0.5, num_input_neurons - 0.5 if num_input_neurons > 0 else 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_facecolor('#1a1a1a')
    ax_raster.tick_params(colors='white')
    [spine.set_color('white') for spine in ax_raster.spines.values()]
    ax_raster.grid(True, alpha=0.15, axis='x')
    for t in np.arange(10, stim_duration_ms, 10): ax_raster.axvline(t, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle(f"MNIST Input Details (Class {label}) - Conv Feature Mode", fontsize=16, color='white', y=0.99)
    try: plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust rect
    except ValueError as e: print(f"Warning: tight_layout issue: {e}")
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved MNIST input details plot (with feature map) to {save_path}")
    except Exception as e: print(f"Error saving input plot {save_path}: {e}")
    plt.close(fig)


def plot_evaluation_results(accuracy, kappa_score, confusion_mat, class_labels, save_path):
    """
    Creates a more elegant plot for evaluation: overall accuracy, kappa,
    and a normalized confusion matrix.
    """
    # Ensure dark style is active (usually set globally)
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(12, 6), facecolor='#1a1a1a') # Slightly wider figure
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.4) # Adjust width ratio and spacing

    # --- Panel 1: Metrics ---
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.set_facecolor('#1a1a1a') # Ensure background is dark
    ax_metrics.axis('off') # Turn off axis lines and ticks

    # Format metrics text
    acc_text = f"Overall Accuracy: {accuracy:.2%}"
    kappa_text = f"Cohen's Kappa: {kappa_score:.4f}" if kappa_score is not None else "Cohen's Kappa: N/A"

    # Display text directly on the axis - centered, larger font
    ax_metrics.text(0.5, 0.6, acc_text, fontsize=18, color='limegreen', ha='center', va='center', weight='bold')
    ax_metrics.text(0.5, 0.4, kappa_text, fontsize=14, color='silver', ha='center', va='center')

    # --- Panel 2: Normalized Confusion Matrix ---
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_cm.set_facecolor('#1a1a1a') # Ensure background is dark

    if confusion_mat is not None and class_labels:
        # Normalize the confusion matrix by the true label counts (rows)
        # This shows recall rates (percentage correctly identified for each class)
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        # Handle potential NaN if a row sum is zero (no instances of that class)
        cm_normalized = np.nan_to_num(cm_normalized)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_labels)

        # Plot using a sequential colormap ('Blues' or 'viridis') and format values as percentages
        disp.plot(ax=ax_cm, cmap='viridis', colorbar=True, values_format=".1%") # Format as percentage

        # --- Improve Text Readability ---
        # Determine threshold for switching text color based on colormap range
        norm = Normalize(vmin=cm_normalized.min(), vmax=cm_normalized.max())
        threshold = norm.vmin + (norm.vmax - norm.vmin) / 2.0 # Midpoint of colormap range

        # Iterate through matrix cells and set text color
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax_cm.texts[i * cm_normalized.shape[1] + j].set_color(
                    'white' if cm_normalized[i, j] < threshold else 'black'
                )
        # --- End Text Readability Improvement ---

        ax_cm.set_title("Normalized Confusion Matrix (Recall)", color='white', fontsize=14) # Updated title
        ax_cm.tick_params(axis='x', colors='white', rotation=45, labelsize=10)
        ax_cm.tick_params(axis='y', colors='white', labelsize=10)
        ax_cm.xaxis.label.set_color('white'); ax_cm.xaxis.label.set_fontsize(12)
        ax_cm.yaxis.label.set_color('white'); ax_cm.yaxis.label.set_fontsize(12)

        # Style the color bar
        cbar = disp.im_.colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.set_label("Fraction of True Labels", color='white', fontsize=12) # Updated label for normalization

        # Style the spines
        for spine in ax_cm.spines.values():
            spine.set_color('gray')

    else:
        ax_cm.text(0.5, 0.5, "Confusion Matrix\nNot Available", color='white', ha='center', va='center')
        ax_cm.axis('off')

    # --- Overall Title and Layout ---
    plt.suptitle("Trained SNN Evaluation Results", fontsize=18, color='white', weight='bold', y=0.97) # Larger main title
    try:
        # Use tight_layout first, then adjust subplot parameters if needed
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect to prevent overlap with suptitle
    except ValueError as e:
        print(f"Warning: tight_layout issue: {e}")
        # Fallback adjustments if tight_layout fails
        plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, wspace=0.4)


    # --- Save Figure ---
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved updated evaluation results plot to {save_path}")
    except Exception as e:
        print(f"Error saving evaluation plot {save_path}: {e}")

    plt.close(fig) # Close the figure

def plot_weight_distribution_by_source(weights_vector, connection_map, inhibitory_status_array, save_path):
    """
    Visualizes the distribution of synaptic weights using a histogram and violin plot.
    Separates weights based on whether the SOURCE neuron is excitatory or inhibitory.

    Args:
        weights_vector (np.ndarray): 1D array of synaptic weight values (all positive).
        connection_map (list): List of (source_neuron_idx, target_neuron_idx) tuples.
        inhibitory_status_array (np.ndarray): Boolean array where True indicates an inhibitory neuron.
        save_path (str): Path to save the generated plot image.
    """
    if weights_vector is None or len(weights_vector) == 0:
        print("Warning: No weights provided for distribution plotting.")
        return
    if connection_map is None or len(connection_map) != len(weights_vector):
        print("Warning: Connection map is missing or length mismatch with weights vector.")
        return
    if inhibitory_status_array is None or len(inhibitory_status_array) == 0:
        print("Warning: Inhibitory status array is missing.")
        return

    # Separate weights based on source neuron type
    excitatory_source_weights = []
    inhibitory_source_weights = []

    print("Analyzing weights based on source neuron type...")
    max_neuron_index = inhibitory_status_array.shape[0] - 1

    for i, (u, v) in enumerate(connection_map):
        # Basic check for valid source index before accessing inhibitory_status_array
        if 0 <= u <= max_neuron_index:
            is_source_inhibitory = inhibitory_status_array[u]
            weight = weights_vector[i] # Assuming weight is positive magnitude

            if is_source_inhibitory:
                inhibitory_source_weights.append(weight)
            else:
                excitatory_source_weights.append(weight)
        else:
             # This case should ideally not happen if connection map is correct
             print(f"Warning: Source neuron index {u} from connection map is out of bounds for inhibitory status array (max index: {max_neuron_index}). Skipping connection {i}.")


    excitatory_source_weights = np.array(excitatory_source_weights)
    inhibitory_source_weights = np.array(inhibitory_source_weights)

    num_total = len(weights_vector)
    num_exc_source = len(excitatory_source_weights)
    num_inh_source = len(inhibitory_source_weights)

    print(f"--- Weight Distribution Analysis (by Source Neuron Type) ---")
    print(f"Total connections analyzed: {num_total}")
    print(f"  Connections from Excitatory Neurons: {num_exc_source} ({num_exc_source/num_total:.2%})")
    print(f"  Connections from Inhibitory Neurons: {num_inh_source} ({num_inh_source/num_total:.2%})")
    if num_total > 0:
         print(f"  Overall Weight Range (Magnitudes): [{np.min(weights_vector):.4f}, {np.max(weights_vector):.4f}]")
         if num_exc_source > 0: print(f"  Exc-> Weight Mean: {np.mean(excitatory_source_weights):.4f}, Std Dev: {np.std(excitatory_source_weights):.4f}")
         if num_inh_source > 0: print(f"  Inh-> Weight Mean: {np.mean(inhibitory_source_weights):.4f}, Std Dev: {np.std(inhibitory_source_weights):.4f}")
    print("-" * 60)


    plt.style.use('dark_background') # Ensure consistent style
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a1a', gridspec_kw={'width_ratios': [1.1, 0.9]})
    fig.suptitle("Synaptic Weight Distribution (by Source Neuron Type)", fontsize=16, color='white', y=0.98)

    # --- Panel 1: Histogram ---
    ax_hist = axes[0]
    min_w = np.min(weights_vector) if num_total > 0 else 0
    max_w = np.max(weights_vector) if num_total > 0 else 1
    bins = 'auto' if num_total > 10 else np.linspace(min_w, max_w, 10) # Use 'auto' or manual bins

    # Plot histograms overlaid
    ax_hist.hist(weights_vector, bins=bins, color='cyan', alpha=0.6, label=f'All ({num_total})', density=False)
    if num_exc_source > 0:
        ax_hist.hist(excitatory_source_weights, bins=bins, color='lime', alpha=0.7, label=f'From Excitatory Src ({num_exc_source})', density=False)
    if num_inh_source > 0:
        ax_hist.hist(inhibitory_source_weights, bins=bins, color='red', alpha=0.7, label=f'From Inhibitory Src ({num_inh_source})', density=False)

    ax_hist.set_title("Weight Value Histogram", color='white')
    ax_hist.set_xlabel("Synaptic Weight (Magnitude)", color='white')
    ax_hist.set_ylabel("Frequency Count", color='white')
    ax_hist.tick_params(colors='white')
    if num_total > 0: ax_hist.legend(facecolor='#333333', labelcolor='white', framealpha=0.8)
    ax_hist.grid(True, alpha=0.2, linestyle=':')
    [spine.set_color('gray') for spine in ax_hist.spines.values()]
    ax_hist.set_facecolor('#2a2a2a')
    # ax_hist.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5) # Line at zero might be less relevant now

    # --- Panel 2: Violin Plot ---
    ax_violin = axes[1]
    # Prepare data in a format suitable for seaborn (long format)
    plot_data = []
    categories = []

    if num_total > 0:
        plot_data.extend(weights_vector)
        categories.extend(['All'] * num_total)
    if num_exc_source > 0:
        plot_data.extend(excitatory_source_weights)
        categories.extend(['From Excitatory'] * num_exc_source)
    if num_inh_source > 0:
        plot_data.extend(inhibitory_source_weights)
        categories.extend(['From Inhibitory'] * num_inh_source)

    if plot_data:
        df = pd.DataFrame({'Weight Type': categories, 'Weight Value': plot_data})

        # Use seaborn for a nice violin plot showing distribution shape and quartiles
        sns.violinplot(data=df, x='Weight Type', y='Weight Value', ax=ax_violin,
                    palette={'All': 'cyan', 'From Excitatory': 'lime', 'From Inhibitory': 'red'},
                    inner='quartile', # Shows median and quartiles inside violins
                    linewidth=1.5, saturation=0.7)

        ax_violin.set_title("Weight Distribution Density", color='white')
        ax_violin.set_ylabel("Synaptic Weight (Magnitude)", color='white')
        ax_violin.set_xlabel("", color='white') # X-labels are clear enough
        ax_violin.tick_params(colors='white', axis='y')
        ax_violin.tick_params(colors='white', axis='x', rotation=10) # Slight rotation for readability
        ax_violin.grid(True, axis='y', alpha=0.2, linestyle=':')
        [spine.set_color('gray') for spine in ax_violin.spines.values()]
        ax_violin.set_facecolor('#2a2a2a')
        # ax_violin.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5) # Less relevant now
    else:
        ax_violin.text(0.5, 0.5, "No weights to plot", color='grey', ha='center', va='center')
        ax_violin.set_title("Weight Distribution Density", color='white')
        ax_violin.axis('off')

    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved weight distribution plot (by source) to {save_path}")
    except Exception as e:
        print(f"\nError saving weight distribution plot {save_path}: {e}")
        traceback.print_exc()
    plt.close(fig) # Close the figure to free memory


def plot_activity_summary_and_heatmap(activity_record, n_neurons, dt, total_duration_ms,
                                      pos, grid_resolution, save_path_prefix,
                                      smoothing_window_ms=5.0): # <-- Smoothing parameter
    """
    Generates two plots related to network activity sparsity:
    1. A plot showing a SMOOTHED activation percentage over time
       (with raw data faint in background). The smoothed line is RED.
    2. A separate heatmap showing the firing probability of each neuron mapped
       onto a spatial grid.

    Args:
        activity_record (list): List of spiking neuron indices per time step.
        n_neurons (int): Total number of neurons in the network.
        dt (float): Simulation time step (ms).
        total_duration_ms (float): The total duration of the simulation run (ms).
        pos (dict): Dictionary mapping neuron indices to (x, y) positions.
                    Required for the heatmap.
        grid_resolution (tuple): Dimensions (rows, cols) for the heatmap grid.
        save_path_prefix (str): Base path/filename for saving the generated plots
                                (e.g., "output/vis_activity"). Suffixes will be added.
        smoothing_window_ms (float): Time window (in ms) for the moving average smoothing
                                     of the activation percentage plot. Set to 0 or None
                                     to disable smoothing.
    """
    if n_neurons <= 0:
        print("Warning: Cannot plot activity summary, n_neurons is 0 or negative.")
        return
    n_steps = len(activity_record)
    if n_steps == 0:
        print("Warning: Cannot plot activity summary, activity_record is empty.")
        return

    print(f"Generating activity timeline plot (Smoothing window: {smoothing_window_ms}ms)...")
    plt.style.use('dark_background') # Ensure dark style

    # --- Calculate derived data ---
    time_axis_ms = np.arange(n_steps) * dt
    activation_percentage = np.array([(len(spikes) / n_neurons) * 100.0 for spikes in activity_record]) # Use numpy array
    avg_activation = np.mean(activation_percentage) if len(activation_percentage) > 0 else 0

    # --- Calculate Smoothed Activation Percentage ---
    smoothed_activation = None
    if smoothing_window_ms and smoothing_window_ms > dt and len(activation_percentage) > 0:
        window_size_steps = max(1, int(smoothing_window_ms / dt))
        activation_series = pd.Series(activation_percentage)
        # Use rolling mean, handle edges with min_periods=1, center the window
        smoothed_activation = activation_series.rolling(window=window_size_steps, center=True, min_periods=1).mean().to_numpy()

    # --- FIGURE 1: Smoothed Percentage Timeline ---
    # Create a figure with a single subplot
    fig1, ax_perc = plt.subplots(1, 1, figsize=(12, 6), facecolor='#1a1a1a')
    fig1.suptitle("Network Activation Percentage Over Time", fontsize=16, color='white', y=0.98)

    # Plot RAW data faintly (using a neutral blue/grey)
    ax_perc.plot(time_axis_ms, activation_percentage, color='#4a6a8a', linewidth=0.5, alpha=0.3, label='Raw Activation (%)') # Faint blue/grey

    # Plot SMOOTHED data clearly if calculated (in RED)
    if smoothed_activation is not None:
        ax_perc.plot(time_axis_ms, smoothed_activation, color='#e74c3c', linewidth=2.0, label=f'Smoothed ({smoothing_window_ms}ms avg)') # RED color
        plot_title = "Smoothed Activation Percentage Over Time"
        avg_text = f"Avg. Raw Activation: {avg_activation:.2f}%\nAvg. Smoothed: {np.mean(smoothed_activation):.2f}%"
    else:
        # If not smoothing, make raw data more prominent (using the original blue)
        ax_perc.lines[0].set_color('#3498db') # Use the original blue
        ax_perc.lines[0].set_linewidth(1.5)
        ax_perc.lines[0].set_alpha(0.8)
        plot_title = "Activation Percentage Over Time"
        avg_text = f"Avg. Activation: {avg_activation:.2f}%"

    # Style the plot
    # ax_perc.set_title(plot_title, color='white') # Title is now suptitle
    ax_perc.set_xlabel("Time (ms)", color='white', fontsize=12) # Add x-label back
    ax_perc.set_ylabel("Activation (% of Neurons)", color='white', fontsize=12)
    ax_perc.set_xlim(0, total_duration_ms)
    ax_perc.set_ylim(0, max(1, np.max(activation_percentage) * 1.1) if len(activation_percentage)>0 else 1) # Y limit based on raw max
    ax_perc.set_facecolor('#2a2a2a')
    ax_perc.grid(True, alpha=0.2, linestyle=':')
    ax_perc.tick_params(colors='white', axis='y')
    ax_perc.tick_params(colors='white', axis='x') # Show x-axis ticks/labels
    for spine in ax_perc.spines.values(): spine.set_color('gray')
    ax_perc.text(0.98, 0.95, avg_text,
                 transform=ax_perc.transAxes, color='silver', fontsize=10,
                 ha='right', va='top', bbox=dict(facecolor='#333333', alpha=0.7, boxstyle='round'))
    ax_perc.legend(loc='upper left', fontsize='small', facecolor='#333333', labelcolor='white', framealpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # Save Figure 1
    timeline_save_path = f"{save_path_prefix}_timeline_smoothed.png" # Updated filename
    try:
        plt.savefig(timeline_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved smoothed activity timeline plot to {timeline_save_path}")
    except Exception as e:
        print(f"\nError saving smoothed activity timeline plot {timeline_save_path}: {e}")
        traceback.print_exc()
    plt.close(fig1)

    # --- FIGURE 2: Activation Probability Heatmap (Code remains the same) ---
    print(f"Generating activation probability heatmap...")
    heatmap_save_path = f"{save_path_prefix}_heatmap.png"

    if pos is None or not isinstance(pos, dict) or len(pos) == 0:
        print("Warning: Neuron positions ('pos' dictionary) not provided or empty. Skipping heatmap generation.")
        return
    if grid_resolution is None or not isinstance(grid_resolution, tuple) or len(grid_resolution) != 2:
        print("Warning: Invalid 'grid_resolution' provided. Skipping heatmap generation.")
        return
    grid_rows, grid_cols = grid_resolution
    if grid_rows <= 0 or grid_cols <= 0:
        print("Warning: Invalid grid dimensions in 'grid_resolution'. Skipping heatmap generation.")
        return

    spike_counts = np.zeros(n_neurons)
    for step_spikes in activity_record:
        try:
            for idx in step_spikes:
                if 0 <= idx < n_neurons:
                    spike_counts[idx] += 1
        except TypeError:
             continue # Ignore non-iterable items

    firing_probability = spike_counts / n_steps if n_steps > 0 else np.zeros(n_neurons)

    valid_pos = {n: p for n, p in pos.items() if 0 <= n < n_neurons and isinstance(p, (tuple, list)) and len(p) == 2}
    if not valid_pos:
        print("Warning: No valid neuron positions found after filtering. Skipping heatmap.")
        return

    xs = [p[0] for p in valid_pos.values()]
    ys = [p[1] for p in valid_pos.values()]
    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    x_range = max_x - min_x; y_range = max_y - min_y
    x_margin = x_range * 0.01 if x_range > 1e-6 else 0.1
    y_margin = y_range * 0.01 if y_range > 1e-6 else 0.1
    min_x -= x_margin; max_x += x_margin
    min_y -= y_margin; max_y += y_margin
    x_range = max_x - min_x; y_range = max_y - min_y

    heatmap_prob_sum = np.zeros((grid_rows, grid_cols))
    heatmap_neuron_count = np.zeros((grid_rows, grid_cols), dtype=int)
    for neuron_idx, (x, y) in valid_pos.items():
        col = int(((x - min_x) / x_range) * (grid_cols - 1)) if x_range > 1e-9 else grid_cols // 2
        row = int(((max_y - y) / y_range) * (grid_rows - 1)) if y_range > 1e-9 else grid_rows // 2
        col = max(0, min(grid_cols - 1, col))
        row = max(0, min(grid_rows - 1, row))
        heatmap_prob_sum[row, col] += firing_probability[neuron_idx]
        heatmap_neuron_count[row, col] += 1

    heatmap_avg_prob = np.full((grid_rows, grid_cols), np.nan)
    valid_cells = heatmap_neuron_count > 0
    heatmap_avg_prob[valid_cells] = heatmap_prob_sum[valid_cells] / heatmap_neuron_count[valid_cells]

    fig2, ax_heatmap = plt.subplots(figsize=(max(8, grid_cols/grid_rows * 7), 7), facecolor='#1a1a1a')

    if np.any(valid_cells):
        min_prob_nz = np.nanmin(heatmap_avg_prob[heatmap_avg_prob > 0]) if np.any(heatmap_avg_prob > 0) else 1e-6
        max_prob = np.nanmax(heatmap_avg_prob) if np.any(valid_cells) else 1.0
        norm = LogNorm(vmin=max(1e-9, min_prob_nz), vmax=max(1e-8, max_prob)) if max_prob > 0 and max_prob / min_prob_nz > 100 else None
        cmap = plt.cm.hot
        cmap.set_bad(color='#1a1a1a')
        im = ax_heatmap.imshow(heatmap_avg_prob, cmap=cmap, interpolation='nearest', origin='upper', norm=norm, aspect='auto')
        cbar = plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
        cbar.set_label('Average Firing Probability per Grid Cell', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white', labelsize=9)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    else:
        ax_heatmap.text(0.5, 0.5, "No Activity Recorded\nCannot Generate Heatmap", color='grey', ha='center', va='center')

    ax_heatmap.set_title("Spatial Heatmap of Neuron Firing Probability", color='white', fontsize=14)
    ax_heatmap.set_xticks([]); ax_heatmap.set_yticks([])
    ax_heatmap.set_facecolor('#1a1a1a')
    for spine in ax_heatmap.spines.values(): spine.set_color('gray')

    plt.tight_layout()
    try:
        plt.savefig(heatmap_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved activation probability heatmap to {heatmap_save_path}")
    except Exception as e:
        print(f"\nError saving activation heatmap plot {heatmap_save_path}: {e}")
        traceback.print_exc()
    plt.close(fig2)




# --- Main Execution Block ---
if __name__ == "__main__":
    start_overall_time = time.time()

    # --- Configuration Loading ---
    CONFIG_FILE = "ga_mnist_snn_vectorized_precomputed_output/best_snn_3class_fixed_structure_precomputed_config.json"
    SAVED_STATE_DIR = os.path.dirname(CONFIG_FILE)

    # keep these for debugging purposes
    if not os.path.exists(CONFIG_FILE): print(f"FATAL ERROR: Config file not found at {CONFIG_FILE}"); exit()
    if not os.path.isdir(SAVED_STATE_DIR): print(f"FATAL ERROR: Saved state directory not found at {SAVED_STATE_DIR}"); exit()

    # load block
    try:
        print(f"Loading configuration from: {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f: config = json.load(f)

        # --- Load encoding mode and related params ---
        LOADED_ENCODING_MODE = config.get("encoding_mode_used")
        # Use the specific downsample factor saved for intensity mode, if available
        DOWNSAMPLE_FACTOR_INTENSITY = config.get("downsample_factor_intensity_mode", config.get("downsample_factor"))
        CONV_FEATURE_COUNT = 49 # Expected feature count
        # adjust this line for convnet path weights
        CONV_WEIGHTS_PATH = '/Users/ethancrouse/Desktop/Spiking-Neural-Network-Experiments/MNIST_utils/conv_model_weights/conv_model_weights.pth' # Path to CNN weights

        if LOADED_ENCODING_MODE not in ['intensity_to_neuron', 'conv_feature_to_neuron']:
            raise ValueError(f"Unknown encoding mode loaded from config: '{LOADED_ENCODING_MODE}'")
        print(f"Loaded model was trained using encoding mode: '{LOADED_ENCODING_MODE}'")

        # --- Dependency check for conv mode ---
        if LOADED_ENCODING_MODE == 'conv_feature_to_neuron':
             if not os.path.exists(CONV_WEIGHTS_PATH):
                 print(f"FATAL ERROR: Encoding mode is '{LOADED_ENCODING_MODE}' but required ConvNet weights file not found at:")
                 print(f"'{CONV_WEIGHTS_PATH}'")
                 exit()
             else:
                 print(f"Required ConvNet weights for stimulator found: {CONV_WEIGHTS_PATH}")

        # Load other parameters from config
        N_CLASSES = config.get("n_classes")
        LAYERS_CONFIG = config.get("layers_config") # Should have correct input size stored
        NEURON_CONFIG = config.get("neuron_config")
        SIM_DT = config.get("simulation_dt")
        RANDOM_SEED = config.get("random_seed", None)
        MNIST_STIM_DURATION_MS = config.get("mnist_stim_duration_ms")
        MAX_FREQ_HZ = config.get("max_frequency_hz")
        BASE_DELAY = config.get("base_transmission_delay_used")
        N_NEURONS_TOTAL = config.get("n_neurons_total")
        TARGET_CLASSES_FROM_CONFIG = config.get("target_classes", list(range(N_CLASSES)))

        # Validate essential parameters
        if None in [N_CLASSES, LAYERS_CONFIG, NEURON_CONFIG, SIM_DT, DOWNSAMPLE_FACTOR_INTENSITY,
                    MNIST_STIM_DURATION_MS, MAX_FREQ_HZ, BASE_DELAY, N_NEURONS_TOTAL, LOADED_ENCODING_MODE]:
            raise ValueError("One or more critical parameters missing from config file.")
        if not isinstance(NEURON_CONFIG, dict):
            raise ValueError("'neuron_config' in config file must be a dictionary.")

        # --- Configuration for this Eval/Vis script ---
        TOTAL_SIMULATION_DURATION_MS = 90
        STIM_CONFIG = {'strength': 25.0, 'pulse_duration_ms': SIM_DT}
        ANIMATE_ACTIVITY = True # Set to True to generate activity GIF
        EVALUATION_SAMPLES = 3000 # Number of test samples to evaluate
        VIS_OUTPUT_DIR = "evaluation_and_visualization_output"
        if not os.path.exists(VIS_OUTPUT_DIR): os.makedirs(VIS_OUTPUT_DIR)











        # Construct filenames based on loaded config
        BASE_FILENAME = f"best_snn_{N_CLASSES}class_fixed_structure_precomputed"
        WEIGHTS_FILE = os.path.join(SAVED_STATE_DIR, f"{BASE_FILENAME}_weights.npy")
        CONNECTIONS_FILE = os.path.join(SAVED_STATE_DIR, f"{BASE_FILENAME}_connection_map.npy")
        DELAYS_FILE = os.path.join(SAVED_STATE_DIR, f"{BASE_FILENAME}_delays_matrix.npy")
        INHIB_STATUS_FILE = os.path.join(SAVED_STATE_DIR, f"{BASE_FILENAME}_inhibitory_array.npy")
        POS_FILE = os.path.join(SAVED_STATE_DIR, f"{BASE_FILENAME}_positions.npy")
        MNIST_TRAIN_SIZE = 60000

        print(f"Config loaded for {N_CLASSES} classes (Targets: {TARGET_CLASSES_FROM_CONFIG}).")
        print(f"Will evaluate on {EVALUATION_SAMPLES} test samples.")
        print(f"Will save outputs to: {VIS_OUTPUT_DIR}")

    except KeyError as e: print(f"FATAL ERROR: Missing key '{e}' in config file {CONFIG_FILE}"); exit()
    except ValueError as e: print(f"FATAL ERROR: Invalid value in config file: {e}"); exit()
    except Exception as e: print(f"FATAL ERROR loading configuration: {e}"); traceback.print_exc(); exit()

    if RANDOM_SEED is not None: print(f"Setting random seed: {RANDOM_SEED}"); random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    # --- Select Device ---
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Load the Trained Network State ---
    print("Loading trained network state...")
    network_obj = None; layer_indices = []; pos_loaded = {}
    try:
        vis_weights, vis_map, vis_delays_matrix, vis_inhib_status, pos_loaded = load_trained_data(
            WEIGHTS_FILE, CONNECTIONS_FILE, DELAYS_FILE, INHIB_STATUS_FILE, POS_FILE
        )
        if vis_weights is None or pos_loaded is None: raise ValueError("Failed to load one or more network state files.")

        # Pass the LAYERS_CONFIG loaded from the config file into neuron params
        temp_neuron_config = NEURON_CONFIG.copy(); temp_neuron_config['layers_config'] = LAYERS_CONFIG
        network_obj, layer_indices = create_network_from_data_vectorized(
            n_neurons_total=N_NEURONS_TOTAL, neuron_params=temp_neuron_config, dt=SIM_DT,
            inhibitory_status_array=vis_inhib_status, delay_matrix=vis_delays_matrix,
            connection_map=vis_map, weights_vector=vis_weights, random_seed=RANDOM_SEED
        )
        if network_obj is None: raise ValueError("Failed to create network object from loaded data.")
        print("Network object created successfully.")
    except Exception as e: print(f"FATAL ERROR loading/creating network: {e}"); traceback.print_exc(); exit()

    # --- Load MNIST for Evaluation ---
    print(f"Loading MNIST & preparing evaluation set...")
    try:
        mnist_loader = MNIST_loader()
        all_labels = mnist_loader.labels.astype(int)
        all_images = mnist_loader.images # Loader provides normalized 0-1 images
        # Find indices for the TEST set belonging to the target classes
        test_mask = (np.arange(len(all_labels)) >= MNIST_TRAIN_SIZE) & (np.isin(all_labels, TARGET_CLASSES_FROM_CONFIG))
        test_indices = np.where(test_mask)[0]

        if len(test_indices) == 0:
             print(f"Error: No test images found for target classes {TARGET_CLASSES_FROM_CONFIG}.")
             exit()

        # Select evaluation samples
        actual_eval_samples = min(EVALUATION_SAMPLES, len(test_indices))
        if actual_eval_samples < EVALUATION_SAMPLES:
             print(f"Warning: Only {actual_eval_samples} test samples available for target classes, requested {EVALUATION_SAMPLES}.")
        eval_indices = np.random.choice(test_indices, actual_eval_samples, replace=False)
        print(f"Selected {actual_eval_samples} indices for evaluation.")

    except Exception as e: print(f"Error loading/preparing MNIST for evaluation: {e}"); traceback.print_exc(); exit()

    # --- Perform Evaluation ---
    print(f"\n--- Starting Evaluation ({actual_eval_samples} samples) ---")
    eval_start_time = time.time()
    predictions = []
    true_labels = []
    correct_predictions = 0
    try:
        # --- Initialize stimulator based on loaded mode ---
        print(f"Initializing SNNStimulator for evaluation (mode: '{LOADED_ENCODING_MODE}')...")
        mnist_stimulator_eval = SNNStimulator(
            total_time_ms=MNIST_STIM_DURATION_MS,
            max_freq_hz=MAX_FREQ_HZ,
            mode=LOADED_ENCODING_MODE,         # Use loaded mode
            conv_weights_path=CONV_WEIGHTS_PATH, # Pass path (used only if mode is conv)
            device=device                      # Pass device (used only if mode is conv)
        )

        eval_loop_iterator = tqdm(eval_indices, desc="Evaluating Network", ncols=80)

        for idx in eval_loop_iterator:
            try:
                # Get original 28x28 image (scaled 0-1 by loader)
                mnist_original_image_0_1 = all_images[idx].reshape(28, 28)
                true_label = all_labels[idx]

                # --- Prepare image based on loaded mode ---
                # Pass 0-255 range image to stimulator's generate_spikes
                if LOADED_ENCODING_MODE == 'intensity_to_neuron':
                    if DOWNSAMPLE_FACTOR_INTENSITY > 1:
                        image_for_stimulator = downsample_image(mnist_original_image_0_1 * 255.0, DOWNSAMPLE_FACTOR_INTENSITY)
                    else:
                        image_for_stimulator = mnist_original_image_0_1 * 255.0
                elif LOADED_ENCODING_MODE == 'conv_feature_to_neuron':
                    # Conv mode stimulator needs the original 28x28
                    image_for_stimulator = mnist_original_image_0_1 * 255.0
                else:
                     # Should not happen due to earlier check
                     raise ValueError(f"Invalid encoding mode during evaluation loop: {LOADED_ENCODING_MODE}")
                # --- End Image Preparation ---

                # Generate spikes using the correct mode and prepared image
                mnist_spike_times = mnist_stimulator_eval.generate_spikes(image_for_stimulator)

                network_obj.reset_all() # Reset network state for each sample
                activity_record = run_vectorized_simulation(
                    network_obj, duration=TOTAL_SIMULATION_DURATION_MS, dt=SIM_DT,
                    mnist_input_spikes=mnist_spike_times,
                    stim_interval_strength=STIM_CONFIG['strength'],
                    stim_pulse_duration_ms=STIM_CONFIG['pulse_duration_ms'],
                    show_progress=False
                )
                predicted_label, _ = classify_output_total_sim(
                    activity_record, layer_indices[-1] if layer_indices else None, N_CLASSES
                )
                predictions.append(predicted_label)
                true_labels.append(true_label)

                # Check for correctness based on valid predictions only
                if predicted_label != -1 and predicted_label == true_label:
                     correct_predictions += 1

                # Update tqdm description with running accuracy
                if len(predictions) > 0:
                    valid_preds_so_far = [p for p in predictions if p != -1]
                    correct_so_far = sum(1 for i, p in enumerate(predictions) if p != -1 and p == true_labels[i])
                    running_acc = correct_so_far / len(valid_preds_so_far) if len(valid_preds_so_far) > 0 else 0.0
                    eval_loop_iterator.set_postfix_str(f"Acc: {running_acc:.3f}", refresh=True)

            except Exception as e:
                print(f"\nWarning: Error during evaluation for index {idx}: {e}")
                # Log traceback for debugging individual sample errors if needed
                # traceback.print_exc()
                predictions.append(-1) # Indicate error/no prediction
                true_labels.append(all_labels[idx]) # Still record true label

        eval_end_time = time.time()
        print(f"\nEvaluation complete in {eval_end_time - eval_start_time:.2f}s.")

        # Calculate Final Metrics
        overall_accuracy = 0.0
        kappa_score = None
        cm = None
        cm_labels = [str(l) for l in TARGET_CLASSES_FROM_CONFIG]

        if actual_eval_samples > 0:
            valid_preds_mask = [p != -1 for p in predictions]
            num_valid_predictions = sum(valid_preds_mask)

            if num_valid_predictions > 0:
                 true_labels_valid = np.array(true_labels)[valid_preds_mask]
                 predictions_valid = np.array(predictions)[valid_preds_mask]

                 correct_valid_predictions = sum(1 for i, p in enumerate(predictions) if p != -1 and p == true_labels[i])
                 overall_accuracy = correct_valid_predictions / num_valid_predictions
                 print(f"Overall Accuracy (on {num_valid_predictions} valid predictions): {overall_accuracy:.4f}")
                 print("Percentage of valid predictions:", num_valid_predictions / actual_eval_samples * 100.0)

                 try:
                     kappa_score = cohen_kappa_score(true_labels_valid, predictions_valid, labels=TARGET_CLASSES_FROM_CONFIG)
                     print(f"Cohen's Kappa Score: {kappa_score:.4f}")
                 except ValueError as e:
                     print(f"Could not calculate Cohen's Kappa: {e}")
                     kappa_score = None

                 cm = confusion_matrix(true_labels_valid, predictions_valid, labels=TARGET_CLASSES_FROM_CONFIG)
            else:
                print("No valid predictions were made during evaluation.")
        else:
            print("No samples were evaluated.")

        # --- Plot Evaluation Results ---
        if actual_eval_samples > 0:
            plot_evaluation_results(
                overall_accuracy, kappa_score, cm, cm_labels,
                os.path.join(VIS_OUTPUT_DIR, "evaluation_summary.png")
            )
        else:
            print("Skipping evaluation plotting.")

    except Exception as e: print(f"FATAL ERROR during evaluation: {e}"); traceback.print_exc(); exit()


    # --- (Optional) Generate Visualizations for ONE Example ---
    print("\n(Optional) Generating detailed visualizations for one example...")
    vis_example_index = eval_indices[0] if len(eval_indices) > 0 else -1 # Use first evaluated sample
    TARGET_VIS_CLASS = all_labels[vis_example_index] if vis_example_index != -1 else -1

    if vis_example_index != -1:
        print(f"Visualizing example index: {vis_example_index} (True Label: {TARGET_VIS_CLASS})")
        try:
            # Get original 28x28 image (0-1)
            vis_image_orig_0_1 = all_images[vis_example_index].reshape(28, 28)

            # --- Initialize stimulator and prepare image based on loaded mode ---
            print(f"Initializing SNNStimulator for visualization (mode: '{LOADED_ENCODING_MODE}')...")
            vis_stimulator = SNNStimulator(
                total_time_ms=MNIST_STIM_DURATION_MS,
                max_freq_hz=MAX_FREQ_HZ,
                mode=LOADED_ENCODING_MODE,
                conv_weights_path=CONV_WEIGHTS_PATH,
                device=device
            )
            
        
            # Prepare image for the stimulator (pass 0-255)
            if LOADED_ENCODING_MODE == 'intensity_to_neuron':
                if DOWNSAMPLE_FACTOR_INTENSITY > 1:
                    image_for_stimulator_vis = downsample_image(vis_image_orig_0_1 * 255.0, DOWNSAMPLE_FACTOR_INTENSITY)
                else:
                    image_for_stimulator_vis = vis_image_orig_0_1 * 255.0
            
                # Image to plot is the one used by stimulator
                image_to_plot_final = image_for_stimulator_vis / 255.0 # Normalize for plotting
            elif LOADED_ENCODING_MODE == 'conv_feature_to_neuron':
                # Stimulator needs 28x28 (0-255)
                image_for_stimulator_vis = vis_image_orig_0_1 * 255.0
                # Image to plot is still the original 28x28
                image_to_plot_final = vis_image_orig_0_1
            else:
                raise ValueError(f"Invalid encoding mode during visualization: {LOADED_ENCODING_MODE}")

            # Generate spikes
            vis_spike_times = vis_stimulator.generate_spikes(image_for_stimulator_vis)

            # --- Extract feature map if in conv mode ---
            vis_feature_map = None
            if LOADED_ENCODING_MODE == 'conv_feature_to_neuron':
                # Use the *original 28x28* image (0-255) for feature map extraction
                # Ensure the extract_feature_map method exists in SNNStimulator
                if hasattr(vis_stimulator, 'extract_feature_map'):
                     vis_feature_map = vis_stimulator.extract_feature_map(vis_image_orig_0_1 * 255.0)
                else:
                     print("Warning: SNNStimulator does not have 'extract_feature_map' method. Cannot visualize feature map.")
            # --- End feature map extraction ---

            # --- Call the new plotting function ---
            plot_mnist_input_with_feature_map(
                image_to_plot_final,         # The image to display in the top-left panel
                TARGET_VIS_CLASS,
                vis_feature_map,             # Pass the extracted feature map (or None)
                vis_spike_times,
                MNIST_STIM_DURATION_MS,
                os.path.join(VIS_OUTPUT_DIR, f"vis_input_digit_{TARGET_VIS_CLASS}_example_with_features.png") # New filename
            )
            # --- End plotting call modification ---

            # Run simulation with these spikes
            network_obj.reset_all()
            vis_activity_record = run_vectorized_simulation(
                network_obj, duration=TOTAL_SIMULATION_DURATION_MS, dt=SIM_DT,
                mnist_input_spikes=vis_spike_times,
                stim_interval_strength=STIM_CONFIG['strength'],
                stim_pulse_duration_ms=STIM_CONFIG['pulse_duration_ms'],
                show_progress=True
            )

            # Re-classify
            vis_predicted_label, vis_spike_counts = classify_output_total_sim(
                vis_activity_record, layer_indices[-1] if layer_indices else None, N_CLASSES
            )
            print(f"Single example classification: Predicted={vis_predicted_label}, True={TARGET_VIS_CLASS}")
            print(f"Single example output spike counts: {vis_spike_counts}")

            # --- Generate other visualization plots ---
            vis_output_prefix = os.path.join(VIS_OUTPUT_DIR, f"vis_digit_{TARGET_VIS_CLASS}_pred_{vis_predicted_label}_mode_{LOADED_ENCODING_MODE}_example") # Add mode to filename
            pos_for_vis = pos_loaded
            if not isinstance(pos_for_vis, dict): pos_for_vis = None

            #activity sparsity plots
            if vis_activity_record: # Check if simulation produced activity
                activity_plot_prefix = f"{vis_output_prefix}_activity" # Use prefix
                # Define grid resolution (adjust as needed, maybe match GIF?)
                heatmap_grid_res = (100, 150)
                plot_activity_summary_and_heatmap(
                    activity_record=vis_activity_record,
                    n_neurons=network_obj.n_neurons,
                    dt=SIM_DT,
                    total_duration_ms=TOTAL_SIMULATION_DURATION_MS,
                    pos=pos_for_vis,                # Pass the loaded positions
                    grid_resolution=heatmap_grid_res, # Pass the desired heatmap grid size
                    save_path_prefix=activity_plot_prefix # Pass the prefix
                )

            print("Generating structure plot...")
            # Ensure stimulated/connected neurons are handled if needed by plot function
            Layered_plot_network_connections_sparse(network=network_obj, pos=pos_for_vis, edge_percent=100, save_path=f"{vis_output_prefix}_structure.png")

            print("\nGenerating weight distribution plot by source neuron type...")
            weight_dist_save_path = os.path.join(VIS_OUTPUT_DIR, f"{BASE_FILENAME}_weight_distribution_by_source.png") # New filename
            # Call the NEW function with the required arguments
            plot_weight_distribution_by_source(
                weights_vector=vis_weights,         # The loaded weights vector
                connection_map=vis_map,             # The loaded connection map
                inhibitory_status_array=vis_inhib_status, # The loaded inhibitory status array
                save_path=weight_dist_save_path     # The path to save the plot
)

            if pos_for_vis and network_obj.graph.number_of_nodes() > 0:
                print("Generating distance dependence plots...")
                try:
                     neuron_for_dist_plot = 0
                     if layer_indices and len(layer_indices) > 0:
                          first_layer_start, _ = layer_indices[0]
                          if first_layer_start < network_obj.n_neurons:
                               neuron_for_dist_plot = first_layer_start
                     w_fig, d_fig = Layered_visualize_distance_dependences(network=network_obj, pos=pos_for_vis, neuron_idx=neuron_for_dist_plot, base_transmission_delay=BASE_DELAY, save_path_base=f"{vis_output_prefix}_neuron{neuron_for_dist_plot}_dist")
                     if w_fig: plt.close(w_fig)
                     if d_fig: plt.close(d_fig)
                except Exception as dist_e:
                    print(f"Warning: Could not generate distance dependence plots: {dist_e}")

            if layer_indices:
                 print("Generating activity PSTH plot...")
                 Layered_plot_activity_and_layer_psth(network=network_obj, activity_record=vis_activity_record, layer_indices=layer_indices, dt=SIM_DT, save_path=f"{vis_output_prefix}_activity_psth.png")
                 print("Generating layer-wise raster plot...")
                 Layered_plot_layer_wise_raster(network=network_obj, activity_record=vis_activity_record, layer_indices=layer_indices, dt=SIM_DT, save_path=f"{vis_output_prefix}_raster.png")

            if ANIMATE_ACTIVITY and pos_for_vis:
                print("Generating activity animation GIF (this may take a while)...")
                Layered_visualize_activity_layout_grid(network=network_obj, pos=pos_for_vis, activity_record=vis_activity_record, dt=SIM_DT, save_path=f"{vis_output_prefix}_animation.gif", fps=25)
            elif ANIMATE_ACTIVITY and not pos_for_vis:
                print("Skipping animation: Positions (pos) data not loaded correctly.")

            print("Single example visualization generated.")

        except Exception as e: print(f"Error during single example visualization: {e}"); traceback.print_exc()
    else: print("Skipping single example visualization (no valid evaluation samples).")


    overall_end_time = time.time()
    print(f"\n--- Script Finished --- Total Time: {overall_end_time - start_overall_time:.2f}s")
    plt.close('all') # Close any remaining figures
