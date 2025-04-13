# visualize_trained_snn.py
# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import random
import json
import traceback
from collections import deque, Counter # Added Counter for confusion matrix
from tqdm import tqdm
# Added cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score

# --- Style ---
plt.style.use('dark_background')

# --- Required Custom Imports ---

# LIF Objects (Need the specific network class used for evaluation)
try:
    # Assuming evaluation uses the Vectorized version
    from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized
    print("Imported LayeredNeuronalNetworkVectorized.")
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import 'LayeredNeuronalNetworkVectorized' from 'LIF_objects'.")
    print("Please ensure this file exists and the class name is correct.")
    exit()

# MNIST Utilities
try:
    from skimage.transform import downscale_local_mean
except ImportError:
    print("Warning: scikit-image not found. Downsampling will fail if factor > 1.")
    # Dummy function if skimage is missing
    def downscale_local_mean(image, factors):
        print("Error: skimage.transform.downscale_local_mean required. Using basic mean.")
        M,N=image.shape; m,n=factors; R,C=M//m, N//n
        res = np.zeros((R,C));
        for i in range(R):
            for j in range(C): res[i,j]=np.mean(image[i*m:(i+1)*m, j*n:(j+1)*n])
        return res
try:
    from MNIST_utils.MNIST_stimulation_encodings import MNIST_loader,SNNStimulator, downsample_image
except ImportError:
    print("\n--- FATAL ERROR ---")
    print("Could not import from 'MNIST_stimulation_encodings.py'.")
    exit()

# Visualization Utilities
try:
    from LIF_utils.network_vis_utils import Layered_plot_network_connections_sparse, Layered_visualize_activity_layout_grid
    from LIF_utils.activity_vis_utils import Layered_plot_activity_and_layer_psth, Layered_plot_layer_wise_raster, Layered_visualize_distance_dependences
    print("Imported visualization functions from LIF_utils.")
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import all visualization functions from 'LIF_utils'.")
    print("Define dummy functions to avoid errors, but plots may fail.")
    def Layered_plot_network_connections_sparse(*args, **kwargs): print("Dummy: Layered_plot_network_connections_sparse")
    def Layered_visualize_activity_layout_grid(*args, **kwargs): print("Dummy: Layered_visualize_activity_layout_grid")
    def Layered_plot_activity_and_layer_psth(*args, **kwargs): print("Dummy: Layered_plot_activity_and_layer_psth")
    def Layered_plot_layer_wise_raster(*args, **kwargs): print("Dummy: Layered_plot_layer_wise_raster")
    def Layered_visualize_distance_dependences(*args, **kwargs): print("Dummy: Layered_visualize_distance_dependences"); return None, None

# --- Functions Implemented Directly (Replacing mnist_snn_evaluation.py dependency) ---

def load_trained_data(weights_file, connections_file, delays_file, inhib_status_file, pos_file): # Added pos_file argument
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
        network = LayeredNeuronalNetworkVectorized(
            n_neurons=n_neurons_total,
            is_inhibitory=inhibitory_status_array, # CRITICAL: Use loaded array
            **neuron_params # Pass other neuron parameters
        )

        # --- 2. Apply the FIXED delays matrix ---
        if network.delays.shape != delay_matrix.shape:
             print(f"Warning: Loaded delay matrix shape {delay_matrix.shape} != network delay shape {network.delays.shape}. Trying to apply anyway.")
        network.delays = delay_matrix.copy() # Apply the loaded fixed delays

        # --- 3. Apply the FIXED weights using the connection map ---
        network.set_weights_sparse(weights_vector, connection_map)

        # --- 4. (Optional but recommended) Rebuild graph for visualization consistency ---
        network.graph.clear_edges()
        print("Populating graph structure based on connection map (for visualization)...")
        # Determine layer indices from config['layers_config'] if available
        layer_indices_from_config = []
        layers_config = neuron_params.get('layers_config', []) # Get layers from params if passed
        if layers_config:
             start_idx = 0
             for size in layers_config:
                 layer_indices_from_config.append((start_idx, start_idx + size))
                 start_idx += size

        # Add nodes with attributes
        for i in range(n_neurons_total):
             is_inhib = inhibitory_status_array[i]
             layer_num = -1
             for idx, (start, end) in enumerate(layer_indices_from_config):
                  if start <= i < end: layer_num = idx + 1; break
             network.graph.add_node(i, is_inhibitory=is_inhib, layer=layer_num) # Add minimal node info

        # Add edges specified by the map
        for u, v in connection_map:
            if u < n_neurons_total and v < n_neurons_total:
                 weight = network.weights[u, v]
                 delay = network.delays[u, v]
                 network.graph.add_edge(u, v, weight=weight, delay=delay)
            else:
                 print(f"Warning: Invalid index ({u},{v}) in connection map during graph rebuild.")

        print(f"Network reconstruction complete. Graph has {network.graph.number_of_nodes()} nodes, {network.graph.number_of_edges()} edges.")
        # Return network, layer indices, but NOT pos (as it's loaded separately)
        return network, layer_indices_from_config

    except Exception as e:
        print(f"Error reconstructing network: {e}")
        traceback.print_exc()
        return None, []

# --- Simulation Function (Copied & Adapted from GA script) ---
def run_vectorized_simulation(network: LayeredNeuronalNetworkVectorized,
                                 duration=1000.0, dt=0.1,
                                 stim_interval_strength=10, # Renamed for clarity
                                 stim_pulse_duration_ms=1.0,
                                 mnist_input_spikes=None,
                                 show_progress=False):
    """
    Runs the network simulation using vectorized updates.
    Adapted from Layered_run_unified_simulation for visualization context.
    Focuses on MNIST input, ignoring other stimulation types.
    """
    mnist_spikes_by_step = {}
    if mnist_input_spikes is not None:
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                if 0 <= step_index < int(duration / dt):
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    mnist_spikes_by_step[step_index].append(neuron_idx)

    n_steps = int(duration / dt)
    activity_record = []
    ongoing_stimulations = {} # Tracks {neuron_idx: end_time}

    sim_loop_iterator = range(n_steps)
    if show_progress:
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step (Vis)", leave=False, ncols=80)

    current_stim_conductances = np.zeros(network.n_neurons)

    for step in sim_loop_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0)
        newly_stimulated_indices_this_step = set()

        # --- Apply MNIST Input Stimulation ---
        if step in mnist_spikes_by_step:
            neurons_spiking_now = mnist_spikes_by_step[step]
            stim_end_time = current_time + stim_pulse_duration_ms
            for neuron_idx in neurons_spiking_now:
                if 0 <= neuron_idx < network.n_neurons:
                    if neuron_idx not in ongoing_stimulations:
                         ongoing_stimulations[neuron_idx] = stim_end_time
                         newly_stimulated_indices_this_step.add(neuron_idx)
                    current_stim_conductances[neuron_idx] = stim_interval_strength

        # --- Update Ongoing Stimulations ---
        expired_stims = set()
        for neuron_idx, end_time in ongoing_stimulations.items():
            if current_time >= end_time:
                expired_stims.add(neuron_idx)
                current_stim_conductances[neuron_idx] = 0.0 # Ensure it's off if expired
            else:
                 # Apply conductance if pulse is still active
                 if 0 <= neuron_idx < network.n_neurons:
                      if neuron_idx not in newly_stimulated_indices_this_step:
                           current_stim_conductances[neuron_idx] = stim_interval_strength

        for neuron_idx in expired_stims:
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Set External Stimulus ---
        network.external_stim_g[:] = current_stim_conductances

        # --- Update Network State ---
        active_indices = network.update_network(dt) # Vectorized update
        activity_record.append(active_indices)

    # --- Cleanup ---
    network.external_stim_g.fill(0.0)
    if show_progress and isinstance(sim_loop_iterator, tqdm):
         sim_loop_iterator.close()

    return activity_record

# --- Classification Function (Optional - Modified to handle total sim) ---
def classify_output_total_sim(activity_record, output_layer_indices, n_classes):
    """Classifies based on TOTAL spike counts in the output layer during the sim."""
    if not output_layer_indices:
         print("Warning: Output layer indices not provided for classification.")
         return -1, {}
    output_start_idx, output_end_idx = output_layer_indices

    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}
    total_output_spikes = 0

    for step_spikes in activity_record:
        for neuron_idx in step_spikes:
            if output_start_idx <= neuron_idx < output_end_idx:
                output_spike_counts[neuron_idx] += 1
                total_output_spikes += 1

    predicted_label = -1
    if total_output_spikes > 0:
        max_spikes = -1
        tied_labels = []
        for neuron_idx, count in output_spike_counts.items():
             current_label = neuron_idx - output_start_idx # Convert neuron index to class label (0, 1, 2...)
             if count > max_spikes:
                  max_spikes = count
                  predicted_label = current_label
                  tied_labels = [current_label]
             elif count == max_spikes:
                  tied_labels.append(current_label)
        # Handle ties: If multiple neurons have the max spike count, prediction is ambiguous
        if len(tied_labels) > 1:
             # print(f"Ambiguous prediction: Labels {tied_labels} tied with {max_spikes} spikes.") # Optional debug
             predicted_label = -1 # Ambiguous prediction

    # Handle case where no output neurons spiked at all
    elif total_output_spikes == 0:
        predicted_label = -1 # No prediction possible

    return predicted_label, output_spike_counts
# --- End of Directly Implemented Functions ---


# --- Function to Visualize Input ---
def plot_mnist_input_details(image, label, spike_times_list, stim_duration_ms, save_path):
    """Creates a 3-panel figure: MNIST image, intensity histogram, input spike raster."""
    print(f"Generating MNIST input visualization...")
    num_input_neurons = len(spike_times_list); total_spikes = sum(len(spikes) for spikes in spike_times_list)
    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a1a'); gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    ax_img = fig.add_subplot(gs[0, 0]); ax_img.imshow(image, cmap='gray', interpolation='nearest'); ax_img.set_title(f"Input Image (Label: {label}, {image.shape[0]}x{image.shape[1]})", color='white'); ax_img.set_xticks([]); ax_img.set_yticks([]); [spine.set_color('gray') for spine in ax_img.spines.values()]
    ax_hist = fig.add_subplot(gs[0, 1]); img_flat = image.flatten(); max_val = img_flat.max(); img_flat = img_flat / max_val if max_val > 1.0 else img_flat; ax_hist.hist(img_flat, bins=10, range=(0, 1), color='skyblue', edgecolor='white', alpha=0.8); ax_hist.set_title("Intensity Distribution", color='white'); ax_hist.set_xlabel("Pixel Intensity", color='white'); ax_hist.set_ylabel("Count", color='white'); ax_hist.set_facecolor('#1a1a1a'); ax_hist.tick_params(colors='white'); [spine.set_color('white') for spine in ax_hist.spines.values()]; ax_hist.grid(True, alpha=0.2, axis='y')
    ax_raster = fig.add_subplot(gs[1, :]); raster_times = []; raster_neurons = []
    for n_idx, spikes in enumerate(spike_times_list):
        for t in spikes: raster_times.append(t); raster_neurons.append(n_idx)
    if raster_times: ax_raster.scatter(raster_times, raster_neurons, s=4, color='white', alpha=0.8, marker='|') # Slightly thicker marker
    ax_raster.set_title(f"Input Spike Encoding ({total_spikes} spikes)", color='white'); ax_raster.set_xlabel("Time (ms)", color='white'); ax_raster.set_ylabel("Input Neuron Index", color='white'); ax_raster.set_xlim(0, stim_duration_ms); ax_raster.set_ylim(-0.5, num_input_neurons - 0.5); ax_raster.invert_yaxis(); ax_raster.set_facecolor('#1a1a1a'); ax_raster.tick_params(colors='white'); [spine.set_color('white') for spine in ax_raster.spines.values()]; ax_raster.grid(True, alpha=0.15, axis='x')
    for t in np.arange(10, stim_duration_ms, 10): ax_raster.axvline(t, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)
    fig.suptitle(f"MNIST Input Stimulus Details (Class {label})", fontsize=16, color='white', y=0.98);
    try:
         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    except ValueError as e:
         print(f"Warning: tight_layout issue: {e}") # Catch potential tight_layout errors
    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved MNIST input details plot to {save_path}")
    except Exception as e:
        print(f"Error saving input plot {save_path}: {e}")
    plt.close(fig)

# --- Function to Plot Evaluation Results ---
# <<< MODIFIED: Added kappa_score argument and plotting >>>
def plot_evaluation_results(accuracy, kappa_score, confusion_mat, class_labels, save_path):
    """Creates plots for evaluation: overall accuracy, kappa, and confusion matrix."""
    fig = plt.figure(figsize=(10, 5), facecolor='#1a1a1a')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # Plot Accuracy and Kappa Text
    ax_acc = fig.add_subplot(gs[0, 0])
    # <<< MODIFIED: Included Kappa score in the text box >>>
    kappa_text = f"Cohen's Kappa: {kappa_score:.4f}" if kappa_score is not None else "Cohen's Kappa: N/A"
    acc_text = f"Overall Accuracy: {accuracy:.2%}\n{kappa_text}"
    ax_acc.text(0.5, 0.5, acc_text,
                fontsize=16, color='lime', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))
    ax_acc.set_title("Evaluation Metrics", color='white') # Renamed title
    ax_acc.axis('off')

    # Plot Confusion Matrix
    ax_cm = fig.add_subplot(gs[0, 1])
    if confusion_mat is not None and class_labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=class_labels)
        disp.plot(ax=ax_cm, cmap='viridis', colorbar=True, text_kw={'color': 'black', 'ha': 'center', 'va': 'center'}) # Set text color for visibility on viridis
        ax_cm.set_title("Confusion Matrix", color='white')
        ax_cm.tick_params(axis='x', colors='white', rotation=45)
        ax_cm.tick_params(axis='y', colors='white')
        ax_cm.xaxis.label.set_color('white')
        ax_cm.yaxis.label.set_color('white')
        # Make colorbar ticks white
        cbar = disp.im_.colorbar
        if cbar:
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            cbar.set_label(cbar.ax.get_ylabel(), color='white') # Set label color
    else:
        ax_cm.text(0.5, 0.5, "Confusion Matrix\nNot Available", color='white', ha='center', va='center')
        ax_cm.axis('off')


    plt.suptitle("Trained SNN Evaluation Results", fontsize=16, color='white', y=0.98)
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    except ValueError as e:
        print(f"Warning: tight_layout issue: {e}")

    try:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved evaluation results plot to {save_path}")
    except Exception as e:
        print(f"Error saving evaluation plot {save_path}: {e}")
    plt.close(fig)


# --- Main Execution Block ---
if __name__ == "__main__":
    start_overall_time = time.time()

    # --- Configuration Loading ---
    CONFIG_FILE = "ga_mnist_snn_vectorized_precomputed_output/best_snn_2class_fixed_structure_precomputed_config.json" # <<< ADJUST PATH
    SAVED_STATE_DIR = os.path.dirname(CONFIG_FILE) # Directory containing .npy files

    if not os.path.exists(CONFIG_FILE): print(f"FATAL ERROR: Config file not found at {CONFIG_FILE}"); exit()
    if not os.path.isdir(SAVED_STATE_DIR): print(f"FATAL ERROR: Saved state directory not found at {SAVED_STATE_DIR}"); exit()

    try:
        print(f"Loading configuration from: {CONFIG_FILE}")
        with open(CONFIG_FILE, 'r') as f: config = json.load(f)
        N_CLASSES = config.get("n_classes")
        LAYERS_CONFIG = config.get("layers_config")
        NEURON_CONFIG = config.get("neuron_config")
        SIM_DT = config.get("simulation_dt")
        RANDOM_SEED = config.get("random_seed", None)
        DOWNSAMPLE_FACTOR = config.get("downsample_factor")
        MNIST_STIM_DURATION_MS = config.get("mnist_stim_duration_ms")
        MAX_FREQ_HZ = config.get("max_frequency_hz")
        BASE_DELAY = config.get("base_transmission_delay_used")
        N_NEURONS_TOTAL = config.get("n_neurons_total")
        TARGET_CLASSES_FROM_CONFIG = config.get("target_classes", list(range(N_CLASSES))) # Load target classes if saved, else default

        if None in [N_CLASSES, LAYERS_CONFIG, NEURON_CONFIG, SIM_DT, DOWNSAMPLE_FACTOR,
                    MNIST_STIM_DURATION_MS, MAX_FREQ_HZ, BASE_DELAY, N_NEURONS_TOTAL]:
            raise ValueError("One or more critical parameters missing from config file.")
        if not isinstance(NEURON_CONFIG, dict):
            raise ValueError("'neuron_config' in config file must be a dictionary.")

        TOTAL_SIMULATION_DURATION_MS = 150
        STIM_CONFIG = {'strength': 25.0, 'pulse_duration_ms': SIM_DT}
        ANIMATE_ACTIVITY = False # <<< SET TO False TO SPEED UP, True for GIF
        EVALUATION_SAMPLES = 4000 # <<< REDUCED for faster testing, increase as needed
        VIS_OUTPUT_DIR = "evaluation_and_visualization_output" # New dir name
        if not os.path.exists(VIS_OUTPUT_DIR): os.makedirs(VIS_OUTPUT_DIR)

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

    # --- Load the Trained Network State ---
    print("Loading trained network state...")
    network_obj = None; layer_indices = []; pos_loaded = {}
    try:
        vis_weights, vis_map, vis_delays_matrix, vis_inhib_status, pos_loaded = load_trained_data(
            WEIGHTS_FILE, CONNECTIONS_FILE, DELAYS_FILE, INHIB_STATUS_FILE, POS_FILE
        )
        if vis_weights is None or pos_loaded is None: raise ValueError("Failed to load one or more network state files.")
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
        all_images = mnist_loader.images
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
        mnist_stimulator_eval = SNNStimulator(total_time_ms=MNIST_STIM_DURATION_MS, max_freq_hz=MAX_FREQ_HZ)
        eval_loop_iterator = tqdm(eval_indices, desc="Evaluating Network", ncols=80)

        for idx in eval_loop_iterator:
            try:
                mnist_original_image = all_images[idx].reshape(28, 28)
                true_label = all_labels[idx]
                if DOWNSAMPLE_FACTOR > 1:
                    mnist_image_to_stimulate = downsample_image(mnist_original_image, DOWNSAMPLE_FACTOR)
                else:
                    mnist_image_to_stimulate = mnist_original_image

                mnist_spike_times = mnist_stimulator_eval.generate_spikes(mnist_image_to_stimulate)
                network_obj.reset_all() # Reset network state for each sample
                activity_record = run_vectorized_simulation(
                    network_obj, duration=TOTAL_SIMULATION_DURATION_MS, dt=SIM_DT,
                    mnist_input_spikes=mnist_spike_times,
                    stim_interval_strength=STIM_CONFIG['strength'],
                    stim_pulse_duration_ms=STIM_CONFIG['pulse_duration_ms'],
                    show_progress=False # Progress bar is handled by tqdm above
                )
                predicted_label, _ = classify_output_total_sim( # Use total sim classification
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
                predictions.append(-1) # Indicate error/no prediction
                true_labels.append(all_labels[idx]) # Still record true label

        eval_end_time = time.time()
        print(f"\nEvaluation complete in {eval_end_time - eval_start_time:.2f}s.")

        # Calculate Final Metrics
        overall_accuracy = 0.0
        kappa_score = None # Initialize kappa
        cm = None # Initialize confusion matrix
        cm_labels = [str(l) for l in TARGET_CLASSES_FROM_CONFIG]

        if actual_eval_samples > 0:
            # Filter out -1 predictions for metric calculations
            valid_preds_mask = [p != -1 for p in predictions]
            num_valid_predictions = sum(valid_preds_mask)

            if num_valid_predictions > 0:
                 true_labels_valid = np.array(true_labels)[valid_preds_mask]
                 predictions_valid = np.array(predictions)[valid_preds_mask]

                 # Accuracy (based on valid predictions)
                 correct_valid_predictions = sum(1 for i, p in enumerate(predictions) if p != -1 and p == true_labels[i])
                 overall_accuracy = correct_valid_predictions / num_valid_predictions
                 print(f"Overall Accuracy (on {num_valid_predictions} valid predictions): {overall_accuracy:.4f}")

                 # <<< ADDED: Cohen's Kappa Calculation >>>
                 try:
                     kappa_score = cohen_kappa_score(true_labels_valid, predictions_valid, labels=TARGET_CLASSES_FROM_CONFIG)
                     print(f"Cohen's Kappa Score: {kappa_score:.4f}")
                 except ValueError as e:
                     print(f"Could not calculate Cohen's Kappa: {e}") # e.g., if only one class predicted
                     kappa_score = None

                 # Confusion Matrix
                 cm = confusion_matrix(true_labels_valid, predictions_valid, labels=TARGET_CLASSES_FROM_CONFIG)
            else:
                print("No valid predictions were made during evaluation. Accuracy and Kappa cannot be calculated.")
                overall_accuracy = 0.0
                kappa_score = None
                cm = None
        else:
            print("No samples were evaluated.")


        # --- Plot Evaluation Results ---
        # <<< MODIFIED: Pass kappa_score to the plotting function >>>
        if actual_eval_samples > 0:
            plot_evaluation_results(
                overall_accuracy, kappa_score, cm, cm_labels, # Pass kappa
                os.path.join(VIS_OUTPUT_DIR, "evaluation_summary.png")
            )
        else:
            print("Skipping evaluation plotting as no samples were evaluated.")

    except Exception as e: print(f"FATAL ERROR during evaluation: {e}"); traceback.print_exc(); exit()


    # --- (Optional) Generate Visualizations for ONE Example ---
    # You can choose a specific example or the first one from the eval set
    print("\n(Optional) Generating detailed visualizations for one example...")
    vis_example_index = eval_indices[0] if len(eval_indices) > 0 else -1 # Use first evaluated sample
    TARGET_VIS_CLASS = all_labels[vis_example_index] if vis_example_index != -1 else -1

    if vis_example_index != -1:
        print(f"Visualizing example index: {vis_example_index} (True Label: {TARGET_VIS_CLASS})")
        try:
            # Regenerate spikes/run sim just for this one example to ensure clean state/record
            vis_image_orig = all_images[vis_example_index].reshape(28, 28)
            if DOWNSAMPLE_FACTOR > 1: vis_image_stim = downsample_image(vis_image_orig, DOWNSAMPLE_FACTOR)
            else: vis_image_stim = vis_image_orig
            vis_stimulator = SNNStimulator(total_time_ms=MNIST_STIM_DURATION_MS, max_freq_hz=MAX_FREQ_HZ)
            vis_spike_times = vis_stimulator.generate_spikes(vis_image_stim)
            plot_mnist_input_details(vis_image_stim, TARGET_VIS_CLASS, vis_spike_times, MNIST_STIM_DURATION_MS, os.path.join(VIS_OUTPUT_DIR, f"vis_input_digit_{TARGET_VIS_CLASS}_example.png"))

            network_obj.reset_all()
            vis_activity_record = run_vectorized_simulation(
                network_obj, duration=TOTAL_SIMULATION_DURATION_MS, dt=SIM_DT,
                mnist_input_spikes=vis_spike_times, stim_interval_strength=STIM_CONFIG['strength'],
                stim_pulse_duration_ms=STIM_CONFIG['pulse_duration_ms'], show_progress=True # Show progress for single long sim
            )

            # Re-classify just this example for confirmation
            vis_predicted_label, vis_spike_counts = classify_output_total_sim(
                vis_activity_record, layer_indices[-1] if layer_indices else None, N_CLASSES
            )
            print(f"Single example classification: Predicted={vis_predicted_label}, True={TARGET_VIS_CLASS}")
            print(f"Single example output spike counts: {vis_spike_counts}")


            # Generate visualization plots using vis_activity_record
            vis_output_prefix = os.path.join(VIS_OUTPUT_DIR, f"vis_digit_{TARGET_VIS_CLASS}_pred_{vis_predicted_label}_example")
            pos_for_vis = pos_loaded
            if not isinstance(pos_for_vis, dict): pos_for_vis = None

            print("Generating structure plot...")
            Layered_plot_network_connections_sparse(network=network_obj, pos=pos_for_vis, edge_percent=100, save_path=f"{vis_output_prefix}_structure.png")

            if pos_for_vis and network_obj.graph.number_of_nodes() > 0:
                print("Generating distance dependence plots...")
                try:
                     # Pick a neuron from the first layer for distance plots if possible
                     neuron_for_dist_plot = 0
                     if layer_indices and len(layer_indices) > 0:
                          first_layer_start, first_layer_end = layer_indices[0]
                          if first_layer_start < network_obj.n_neurons:
                               neuron_for_dist_plot = first_layer_start # Use the first neuron of the first layer

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