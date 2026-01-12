"""
Visualization utilities for trained SNN networks.

This module provides plotting functions for:
- MNIST input encoding with CNN feature maps
- Evaluation results with confusion matrices
- Weight distribution analysis
- Network activity summaries and heatmaps

These functions are used by evaluate_network.py for comprehensive
visualization of trained networks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LogNorm
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import traceback


def plot_mnist_input_with_feature_map(image, label, feature_map, spike_times_list, stim_duration_ms, save_path):
    """Creates a 4-panel figure: MNIST image, 7x7 feature map, input spike raster."""
    print(f"Generating MNIST input + feature map visualization...")
    num_input_neurons = len(spike_times_list)
    total_spikes = sum(len(spikes) for spikes in spike_times_list)

    fig = plt.figure(figsize=(12, 9), facecolor='#1a1a1a')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], hspace=0.4, wspace=0.3)

    # Panel 1: Input Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap='gray', interpolation='nearest')
    ax_img.set_title(f"Input Image (Label: {label})", color='white')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    [spine.set_color('gray') for spine in ax_img.spines.values()]

    # Panel 2: Feature Map
    ax_feat = fig.add_subplot(gs[0, 1])
    if feature_map is not None and feature_map.shape == (7, 7):
        im = ax_feat.imshow(feature_map, cmap='viridis', interpolation='nearest')
        ax_feat.set_title("CNN 7x7 Feature Map", color='white')
        plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
    else:
        ax_feat.text(0.5, 0.5, "Feature Map\nNot Available", color='grey', ha='center', va='center')
        ax_feat.set_title("CNN Feature Map", color='white')
    ax_feat.set_xticks([])
    ax_feat.set_yticks([])
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
    if raster_times:
        ax_raster.scatter(raster_times, raster_neurons, s=5, color='white', alpha=0.8, marker='|')
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
    for t in np.arange(10, stim_duration_ms, 10):
        ax_raster.axvline(t, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    fig.suptitle(f"MNIST Input Details (Class {label}) - Conv Feature Mode", fontsize=16, color='white', y=0.99)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    except ValueError as e:
        print(f"Warning: tight_layout issue: {e}")
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved MNIST input details plot (with feature map) to {save_path}")
    except Exception as e:
        print(f"Error saving input plot {save_path}: {e}")
    plt.close(fig)


def plot_evaluation_results(accuracy, kappa_score, confusion_mat, class_labels, save_path):
    """
    Creates a more elegant plot for evaluation: overall accuracy, kappa,
    and a normalized confusion matrix.
    """
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(12, 6), facecolor='#1a1a1a')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.4)

    # --- Panel 1: Metrics ---
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.set_facecolor('#1a1a1a')
    ax_metrics.axis('off')

    # Format metrics text
    acc_text = f"Overall Accuracy: {accuracy:.2%}"
    kappa_text = f"Cohen's Kappa: {kappa_score:.4f}" if kappa_score is not None else "Cohen's Kappa: N/A"

    # Display text directly on the axis
    ax_metrics.text(0.5, 0.6, acc_text, fontsize=18, color='limegreen', ha='center', va='center', weight='bold')
    ax_metrics.text(0.5, 0.4, kappa_text, fontsize=14, color='silver', ha='center', va='center')

    # --- Panel 2: Normalized Confusion Matrix ---
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_cm.set_facecolor('#1a1a1a')

    if confusion_mat is not None and class_labels:
        # Normalize the confusion matrix by the true label counts (rows)
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_labels)
        disp.plot(ax=ax_cm, cmap='viridis', colorbar=True, values_format=".1%")

        # Improve text readability
        norm = Normalize(vmin=cm_normalized.min(), vmax=cm_normalized.max())
        threshold = norm.vmin + (norm.vmax - norm.vmin) / 2.0

        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax_cm.texts[i * cm_normalized.shape[1] + j].set_color(
                    'white' if cm_normalized[i, j] < threshold else 'black'
                )

        ax_cm.set_title("Normalized Confusion Matrix (Recall)", color='white', fontsize=14)
        ax_cm.tick_params(axis='x', colors='white', rotation=45, labelsize=10)
        ax_cm.tick_params(axis='y', colors='white', labelsize=10)
        ax_cm.xaxis.label.set_color('white')
        ax_cm.xaxis.label.set_fontsize(12)
        ax_cm.yaxis.label.set_color('white')
        ax_cm.yaxis.label.set_fontsize(12)

        # Style the color bar
        cbar = disp.im_.colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color='white', labelsize=10)
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.set_label("Fraction of True Labels", color='white', fontsize=12)

        for spine in ax_cm.spines.values():
            spine.set_color('gray')
    else:
        ax_cm.text(0.5, 0.5, "Confusion Matrix\nNot Available", color='white', ha='center', va='center')
        ax_cm.axis('off')

    # Overall title and layout
    plt.suptitle("Trained SNN Evaluation Results", fontsize=18, color='white', weight='bold', y=0.97)
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    except ValueError as e:
        print(f"Warning: tight_layout issue: {e}")
        plt.subplots_adjust(top=0.90, bottom=0.1, left=0.1, right=0.9, wspace=0.4)

    # Save figure
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved updated evaluation results plot to {save_path}")
    except Exception as e:
        print(f"Error saving evaluation plot {save_path}: {e}")

    plt.close(fig)


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
        if 0 <= u <= max_neuron_index:
            is_source_inhibitory = inhibitory_status_array[u]
            weight = weights_vector[i]

            if is_source_inhibitory:
                inhibitory_source_weights.append(weight)
            else:
                excitatory_source_weights.append(weight)
        else:
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
        if num_exc_source > 0:
            print(f"  Exc-> Weight Mean: {np.mean(excitatory_source_weights):.4f}, Std Dev: {np.std(excitatory_source_weights):.4f}")
        if num_inh_source > 0:
            print(f"  Inh-> Weight Mean: {np.mean(inhibitory_source_weights):.4f}, Std Dev: {np.std(inhibitory_source_weights):.4f}")
    print("-" * 60)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#1a1a1a', gridspec_kw={'width_ratios': [1.1, 0.9]})
    fig.suptitle("Synaptic Weight Distribution (by Source Neuron Type)", fontsize=16, color='white', y=0.98)

    # --- Panel 1: Histogram ---
    ax_hist = axes[0]
    min_w = np.min(weights_vector) if num_total > 0 else 0
    max_w = np.max(weights_vector) if num_total > 0 else 1
    bins = 'auto' if num_total > 10 else np.linspace(min_w, max_w, 10)

    ax_hist.hist(weights_vector, bins=bins, color='cyan', alpha=0.6, label=f'All ({num_total})', density=False)
    if num_exc_source > 0:
        ax_hist.hist(excitatory_source_weights, bins=bins, color='lime', alpha=0.7, label=f'From Excitatory Src ({num_exc_source})', density=False)
    if num_inh_source > 0:
        ax_hist.hist(inhibitory_source_weights, bins=bins, color='red', alpha=0.7, label=f'From Inhibitory Src ({num_inh_source})', density=False)

    ax_hist.set_title("Weight Value Histogram", color='white')
    ax_hist.set_xlabel("Synaptic Weight (Magnitude)", color='white')
    ax_hist.set_ylabel("Frequency Count", color='white')
    ax_hist.tick_params(colors='white')
    if num_total > 0:
        ax_hist.legend(facecolor='#333333', labelcolor='white', framealpha=0.8)
    ax_hist.grid(True, alpha=0.2, linestyle=':')
    [spine.set_color('gray') for spine in ax_hist.spines.values()]
    ax_hist.set_facecolor('#2a2a2a')

    # --- Panel 2: Violin Plot ---
    ax_violin = axes[1]
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

        sns.violinplot(data=df, x='Weight Type', y='Weight Value', ax=ax_violin,
                      palette={'All': 'cyan', 'From Excitatory': 'lime', 'From Inhibitory': 'red'},
                      inner='quartile',
                      linewidth=1.5, saturation=0.7)

        ax_violin.set_title("Weight Distribution Density", color='white')
        ax_violin.set_ylabel("Synaptic Weight (Magnitude)", color='white')
        ax_violin.set_xlabel("", color='white')
        ax_violin.tick_params(colors='white', axis='y')
        ax_violin.tick_params(colors='white', axis='x', rotation=10)
        ax_violin.grid(True, axis='y', alpha=0.2, linestyle=':')
        [spine.set_color('gray') for spine in ax_violin.spines.values()]
        ax_violin.set_facecolor('#2a2a2a')
    else:
        ax_violin.text(0.5, 0.5, "No weights to plot", color='grey', ha='center', va='center')
        ax_violin.set_title("Weight Distribution Density", color='white')
        ax_violin.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved weight distribution plot (by source) to {save_path}")
    except Exception as e:
        print(f"\nError saving weight distribution plot {save_path}: {e}")
        traceback.print_exc()
    plt.close(fig)


def plot_activity_summary_and_heatmap(activity_record, n_neurons, dt, total_duration_ms,
                                      pos, grid_resolution, save_path_prefix,
                                      smoothing_window_ms=5.0):
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
    plt.style.use('dark_background')

    # Calculate derived data
    time_axis_ms = np.arange(n_steps) * dt
    activation_percentage = np.array([(len(spikes) / n_neurons) * 100.0 for spikes in activity_record])
    avg_activation = np.mean(activation_percentage) if len(activation_percentage) > 0 else 0

    # Calculate smoothed activation percentage
    smoothed_activation = None
    if smoothing_window_ms and smoothing_window_ms > dt and len(activation_percentage) > 0:
        window_size_steps = max(1, int(smoothing_window_ms / dt))
        activation_series = pd.Series(activation_percentage)
        smoothed_activation = activation_series.rolling(window=window_size_steps, center=True, min_periods=1).mean().to_numpy()

    # --- FIGURE 1: Smoothed Percentage Timeline ---
    fig1, ax_perc = plt.subplots(1, 1, figsize=(12, 6), facecolor='#1a1a1a')
    fig1.suptitle("Network Activation Percentage Over Time", fontsize=16, color='white', y=0.98)

    # Plot raw data faintly
    ax_perc.plot(time_axis_ms, activation_percentage, color='#4a6a8a', linewidth=0.5, alpha=0.3, label='Raw Activation (%)')

    # Plot smoothed data clearly if calculated
    if smoothed_activation is not None:
        ax_perc.plot(time_axis_ms, smoothed_activation, color='#e74c3c', linewidth=2.0, label=f'Smoothed ({smoothing_window_ms}ms avg)')
        avg_text = f"Avg. Raw Activation: {avg_activation:.2f}%\nAvg. Smoothed: {np.mean(smoothed_activation):.2f}%"
    else:
        ax_perc.lines[0].set_color('#3498db')
        ax_perc.lines[0].set_linewidth(1.5)
        ax_perc.lines[0].set_alpha(0.8)
        avg_text = f"Avg. Activation: {avg_activation:.2f}%"

    # Style the plot
    ax_perc.set_xlabel("Time (ms)", color='white', fontsize=12)
    ax_perc.set_ylabel("Activation (% of Neurons)", color='white', fontsize=12)
    ax_perc.set_xlim(0, total_duration_ms)
    ax_perc.set_ylim(0, max(1, np.max(activation_percentage) * 1.1) if len(activation_percentage) > 0 else 1)
    ax_perc.set_facecolor('#2a2a2a')
    ax_perc.grid(True, alpha=0.2, linestyle=':')
    ax_perc.tick_params(colors='white', axis='y')
    ax_perc.tick_params(colors='white', axis='x')
    for spine in ax_perc.spines.values():
        spine.set_color('gray')
    ax_perc.text(0.98, 0.95, avg_text,
                 transform=ax_perc.transAxes, color='silver', fontsize=10,
                 ha='right', va='top', bbox=dict(facecolor='#333333', alpha=0.7, boxstyle='round'))
    ax_perc.legend(loc='upper left', fontsize='small', facecolor='#333333', labelcolor='white', framealpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save Figure 1
    timeline_save_path = f"{save_path_prefix}_timeline_smoothed.png"
    try:
        plt.savefig(timeline_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved smoothed activity timeline plot to {timeline_save_path}")
    except Exception as e:
        print(f"\nError saving smoothed activity timeline plot {timeline_save_path}: {e}")
        traceback.print_exc()
    plt.close(fig1)

    # --- FIGURE 2: Activation Probability Heatmap ---
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
            continue

    firing_probability = spike_counts / n_steps if n_steps > 0 else np.zeros(n_neurons)

    valid_pos = {n: p for n, p in pos.items() if 0 <= n < n_neurons and isinstance(p, (tuple, list)) and len(p) == 2}
    if not valid_pos:
        print("Warning: No valid neuron positions found after filtering. Skipping heatmap.")
        return

    xs = [p[0] for p in valid_pos.values()]
    ys = [p[1] for p in valid_pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_range = max_x - min_x
    y_range = max_y - min_y
    x_margin = x_range * 0.01 if x_range > 1e-6 else 0.1
    y_margin = y_range * 0.01 if y_range > 1e-6 else 0.1
    min_x -= x_margin
    max_x += x_margin
    min_y -= y_margin
    max_y += y_margin
    x_range = max_x - min_x
    y_range = max_y - min_y

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
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_facecolor('#1a1a1a')
    for spine in ax_heatmap.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    try:
        plt.savefig(heatmap_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved activation probability heatmap to {heatmap_save_path}")
    except Exception as e:
        print(f"\nError saving activation heatmap plot {heatmap_save_path}: {e}")
        traceback.print_exc()
    plt.close(fig2)
