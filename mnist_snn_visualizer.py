# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background') # Apply style early

# Required for MNIST loading and processing (Ensure MNIST_stimulation_encodings.py is available)
try:
    from skimage.transform import downscale_local_mean
except ImportError:
    print("Warning: scikit-image not installed. Downsampling will fail if factor > 1.")
    # Define a dummy function if skimage is not available
    def downscale_local_mean(image, factors):
        print("Error: skimage.transform.downscale_local_mean is required for downsampling.")
        # Simple averaging as a fallback, might not be equivalent
        M, N = image.shape
        m, n = factors
        col_extent = N // n
        row_extent = M // m
        result = np.zeros((row_extent, col_extent))
        for i in range(row_extent):
            for j in range(col_extent):
                 result[i, j] = np.mean(image[i*m:(i+1)*m, j*n:(j+1)*n])
        return result

try:
    # Attempt to import MNIST utilities
    from MNIST_stimulation_encodings import MNIST_loader, SNNStimulator, downsample_image
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import from 'MNIST_stimulation_encodings.py'.")
    print("Please ensure this file exists and is in the correct path.")
    print("Using dummy classes/functions as placeholders.")
    print("---------------")
    # Define dummy classes/functions if the import fails
    class MNIST_loader:
        def __init__(self):
            print("Using dummy MNIST_loader.")
            # Create dummy data resembling MNIST
            self.images = np.random.rand(100, 784) # 100 dummy images
            self.labels = np.random.randint(0, 10, 100) # 100 dummy labels
            # Ensure some 0s and 1s exist for testing the filter
            self.labels[0] = 0
            self.labels[1] = 1
        def get_image(self, index):
            print(f"Dummy MNIST: Getting image {index}")
            if 0 <= index < len(self.images):
                return self.images[index].reshape(28, 28)
            return np.zeros((28,28))
        def get_label(self, index):
            print(f"Dummy MNIST: Getting label {index}")
            if 0 <= index < len(self.labels):
                return self.labels[index]
            return -1
        def get_data_size(self): # Add method for dummy loader
            return len(self.labels)


    class SNNStimulator:
         def __init__(self, total_time_ms, max_freq_hz):
             print("Using dummy SNNStimulator.")
             self.total_time_ms = total_time_ms
             self.max_freq_hz = max_freq_hz
         def generate_spikes(self, image):
             print("Dummy SNNStimulator: Generating random spikes.")
             num_pixels = image.shape[0] * image.shape[1]
             spike_times = []
             for i in range(num_pixels):
                 # Generate a few random spike times for each pixel
                 num_spikes = np.random.randint(0, 5)
                 spikes = np.random.uniform(0, self.total_time_ms, num_spikes)
                 spike_times.append(list(np.sort(spikes)))
             return spike_times

    def downsample_image(image, factor):
        print(f"Dummy downsample_image called with factor {factor}.")
        if factor > 1:
             # Use the basic downscale_local_mean (either skimage's or the dummy)
             try:
                  return downscale_local_mean(image, (factor, factor))
             except NameError: # If downscale_local_mean is not defined at all
                  print("Error: Downsampling function not available.")
                  return image
        return image


import time # To measure execution time
import matplotlib.gridspec as gridspec # For creating complex subplot layouts
import random
import inspect # Import inspect module for introspection
from collections import deque # Used in LayeredNeuronalNetwork
from tqdm import tqdm # Progress bar for simulation

# --- Import custom LIF objects ---
# Ensure these paths are correct for your project structure
try:
    from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal
    from LIF_objects.LayeredNeuronalNetwork import LayeredNeuronalNetwork
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import LIF neuron/network classes from 'LIF_objects'.")
    print("Please ensure this folder and the required files exist.")
    # Define dummy classes if import fails
    class Layered_LIFNeuronWithReversal:
        def __init__(self, *args, **kwargs): self.is_inhibitory = kwargs.get('is_inhibitory', False); self.v=0; self.g_e=0; self.g_i=0; self.e_reversal=0; self.i_reversal=0; self.external_stim_g=0; self.layer=None
        def reset(self): pass
        def update(self, dt): return False # Never spikes
        def receive_spike(self, weight): pass
        def apply_external_stimulus(self, g): self.external_stim_g = g # Dummy implementation
    class LayeredNeuronalNetwork:
        def __init__(self, n_neurons, inhibitory_fraction, **kwargs): self.n_neurons = n_neurons; self.neurons = [None]*n_neurons; self.graph = None; self.inhibitory_fraction = inhibitory_fraction; self.weights=np.zeros((n_neurons, n_neurons)); self.delays=np.zeros((n_neurons, n_neurons)); self.spike_queue=deque(); self.network_activity=[]; self.neuron_params=kwargs; self.neuron_grid_positions={}; import networkx as nx; self.graph = nx.DiGraph()
        def add_neuron(self, neuron, node_id, pos, layer): self.neurons[node_id]=neuron; self.graph.add_node(node_id, is_inhibitory=neuron.is_inhibitory, layer=layer, pos=pos); self.neuron_grid_positions[node_id]=pos
        def add_connection(self, u, v, weight, delay): self.graph.add_edge(u,v,weight=weight,delay=delay); self.weights[u,v]=weight; self.delays[u,v]=delay
        def reset_all(self): self.spike_queue=deque(); self.network_activity=[]
        def update_network(self, dt): # Simulate basic decay for dummy neurons
            active = []
            for n in self.neurons:
                if n: n.g_e *=0.9; n.g_i *= 0.9; n.external_stim_g *= 0.9
            return active # No spikes
    print("Using dummy LIF classes.")
    print("---------------")


# --- Import ACTUAL Visualization Utils ---
# Ensure these paths are correct for your project structure
try:
    from LIF_utils.network_vis_utils import (
        Layered_plot_network_connections_sparse,
        Layered_visualize_activity_layout_grid,
    )
    from LIF_utils.activity_vis_utils import (
        Layered_plot_activity_and_layer_psth,
        Layered_plot_layer_wise_raster,
        Layered_visualize_distance_dependences
    )
    # ** Simulation function is defined below **
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import visualization or simulation utilities from 'LIF_utils'.")
    print("Please ensure this folder and the required files exist.")
    print("Using dummy functions as placeholders. Visualizations will likely fail.")
    # Define dummy functions if import fails
    def Layered_plot_network_connections_sparse(*args, **kwargs): print("Dummy plot_network_connections_sparse called."); return plt.figure()
    def Layered_visualize_activity_layout_grid(*args, **kwargs): print("Dummy visualize_activity_layout_grid called."); return None
    def Layered_visualize_distance_dependences(*args, **kwargs): print("Dummy visualize_distance_dependences called."); return plt.figure(), plt.figure()
    def Layered_plot_activity_and_layer_psth(*args, **kwargs): print("Dummy plot_activity_and_layer_psth called."); return plt.figure()
    def Layered_plot_layer_wise_raster(*args, **kwargs): print("Dummy plot_layer_wise_raster called."); return plt.figure()
    print("---------------")


# --- Simulation Function (Copied from mnist_snn.py to handle mnist_input_spikes) ---
def Layered_run_unified_simulation(network, duration=1000.0, dt=0.1,
                                 # Original stim params (will be disabled if mnist_input_spikes is used)
                                 stim_interval=None, stim_interval_strength=10,
                                 stim_fraction=0.01, stim_target_indices=None,
                                 stim_pulse_duration_ms=1.0,
                                 # New MNIST input param
                                 mnist_input_spikes=None,
                                 # Other params
                                 track_neurons=None, stochastic_stim=False,
                                 no_stimulation=False):
    """
    Runs the network simulation handling various stimulation types and tracking.
    MODIFIED: Accepts pre-generated MNIST spike times (`mnist_input_spikes`).
              If provided, it overrides other stimulation methods.
    """
    # print(f"--- Running Simulation ({duration}ms) ---") # Reduced print output
    if mnist_input_spikes is not None:
        # print("INFO: Using pre-generated MNIST spike input. Other stimulation settings ignored.") # Reduced print output
        # Override other stimulation flags if MNIST input is used
        no_stimulation = True # MNIST provides the stimulation
        stochastic_stim = False
        stim_interval = None
        num_input_neurons = len(mnist_input_spikes)
        # Pre-process MNIST spikes for quick lookup during simulation
        # Create a dictionary: {time_step: [list of input neurons spiking]}
        mnist_spikes_by_step = {}
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                if step_index >= 0: # Only consider non-negative steps
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    mnist_spikes_by_step[step_index].append(neuron_idx)

    n_steps = int(duration / dt)
    activity_record = []
    # Ensure 'times' key exists even if no stimulation occurs
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms, 'times': []}
    neuron_data = {} # Not tracking detailed neuron data for this visualization script

    ongoing_stimulations = {} # Tracks end times for active pulses {neuron_idx: end_time}
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None
    # Ensure stimulation population uses target_indices if provided, even if using MNIST later
    stimulation_population = list(stim_target_indices) if stim_target_indices is not None else list(range(network.n_neurons))

    # --- Simulation Loop ---
    sim_loop_iterator = tqdm(range(n_steps), desc="Simulation Step", leave=False) # Use progress bar

    for step in sim_loop_iterator:
        current_time = step * dt
        newly_stimulated_indices = [] # Reset for this step

        # --- Apply MNIST Input Stimulation (if provided) ---
        if mnist_input_spikes is not None:
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                stim_end_time = current_time + stim_pulse_duration_ms # Duration of effect
                for idx in neurons_spiking_now:
                    # Apply stimulus to the corresponding neuron in the first layer
                    if idx < len(network.neurons) and network.neurons[idx] is not None:
                        if idx not in ongoing_stimulations: # Start new pulse
                            ongoing_stimulations[idx] = stim_end_time
                            newly_stimulated_indices.append(idx)
                            # Use apply_external_stimulus method if it exists
                            if hasattr(network.neurons[idx], 'apply_external_stimulus'):
                                network.neurons[idx].apply_external_stimulus(stim_interval_strength)
                            else: # Fallback if method is missing (e.g., using different Neuron class)
                                print(f"Warning: Neuron {idx} missing apply_external_stimulus method.")

                # Record this MNIST-triggered stimulation event if new neurons were pulsed
                if newly_stimulated_indices:
                    stimulation_record['pulse_starts'].append(current_time)
                    stimulation_record['times'] = stimulation_record['pulse_starts'] # Keep 'times' key updated
                    stimulation_record['neurons'].append(list(newly_stimulated_indices)) # Store list copy

        # --- Apply Original Stimulation (Only if MNIST input is NOT used) ---
        # This block is effectively disabled when using MNIST input due to no_stimulation=True
        elif not no_stimulation and stimulation_population:
            num_to_stimulate = max(1, int(len(stimulation_population) * stim_fraction))
            apply_new_stim_pulse = False
            if stochastic_stim and random.random() < (dt / 100): # Example rate
                apply_new_stim_pulse = True
            elif stim_interval_steps and (step > 0 and step % stim_interval_steps == 0):
                apply_new_stim_pulse = True

            if apply_new_stim_pulse:
                target_neurons_for_pulse = random.sample(stimulation_population, min(num_to_stimulate, len(stimulation_population)))
                stim_end_time = current_time + stim_pulse_duration_ms
                for idx in target_neurons_for_pulse:
                    if idx < len(network.neurons) and network.neurons[idx] is not None:
                        if idx not in ongoing_stimulations:
                            ongoing_stimulations[idx] = stim_end_time
                            newly_stimulated_indices.append(idx)
                            if hasattr(network.neurons[idx], 'apply_external_stimulus'):
                                network.neurons[idx].apply_external_stimulus(stim_interval_strength)
                if newly_stimulated_indices:
                    stimulation_record['pulse_starts'].append(current_time)
                    stimulation_record['times'] = stimulation_record['pulse_starts']
                    stimulation_record['neurons'].append(list(newly_stimulated_indices))

        # --- Update Ongoing Stimulations (Applies conductance if pulse is active) ---
        expired_stims = []
        # We iterate over a copy of keys because ongoing_stimulations might be modified
        for neuron_idx, end_time in list(ongoing_stimulations.items()):
            if current_time >= end_time:
                expired_stims.append(neuron_idx)
            else:
                # This re-application ensures the conductance is held high for the duration.
                if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
                     if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                        network.neurons[neuron_idx].apply_external_stimulus(stim_interval_strength)

        # Remove expired stimulations and reset their external conductance
        for neuron_idx in expired_stims:
            if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
                 if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                     network.neurons[neuron_idx].apply_external_stimulus(0.0) # Turn off stimulus
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Update Network State ---
        active_indices = network.update_network(dt)
        activity_record.append(active_indices) # Store indices of spiking neurons


    # --- Cleanup After Loop ---
    for neuron_idx in list(ongoing_stimulations.keys()): # Ensure all stims turned off
        if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
            if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                 network.neurons[neuron_idx].apply_external_stimulus(0.0)
        if neuron_idx in ongoing_stimulations: # Check existence before deleting
             del ongoing_stimulations[neuron_idx]


    # print(f"--- Simulation Finished ---") # Reduced print output
    # print(f"Recorded {len(stimulation_record['pulse_starts'])} stimulation pulse starts.") # Reduced print output
    return activity_record, neuron_data, stimulation_record


# --- Network Creation Function (Adapted from genetic_experiment.py) ---
def create_visual_snn_structure(n_layers_list, inhibitory_fraction, connection_probs,
                                neuron_params, weight_min, weight_max,
                                base_transmission_delay, dt, random_seed=None):
    """
    Creates the SNN structure based on GA parameters.
    Initializes connections based on probabilities with random weights.
    Returns network, layer_indices, pos.
    """
    print("--- Creating SNN Structure for Visualization ---")
    # Parameter Extraction and Setup
    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Extract connection probabilities with defaults
    conn_prob_defaults = {
        'exc_recurrent': 0.08, 'inh_recurrent': 0.15,
        'feedforward_1': 0.45, 'feedforward_2': 0.15,
        'feedback_1': 0.06, 'feedback_2': 0.02,
        'long_feedforward': 0.01, 'long_feedback': 0.005
    }
    conn_probs_actual = {k: connection_probs.get(k, conn_prob_defaults[k]) for k in conn_prob_defaults}

    # Create Network Instance
    network = LayeredNeuronalNetwork(n_neurons=total_neurons,
                                     inhibitory_fraction=inhibitory_fraction,
                                     **neuron_params) # Pass neuron params

    # Create Neurons and Assign Positions/Layers
    pos = {} # Stores {neuron_idx: (x, y)}
    x_coords = np.linspace(0.1, 0.9, num_layers) # Horizontal positions for layers
    horizontal_spread = 0.04 # Jitter within layer x-axis
    vertical_spread = max(0.5, total_neurons / 200.0) # Jitter within layer y-axis
    layer_indices = [] # Stores [(start_idx, end_idx), ...] for each layer
    start_idx = 0

    # Filter neuron_params for constructor compatibility
    # Use try-except block in case Layered_LIFNeuronWithReversal is a dummy class
    try:
        neuron_constructor_sig = inspect.signature(Layered_LIFNeuronWithReversal)
        valid_neuron_keys = {k for k in neuron_constructor_sig.parameters if k != 'self' and k != 'is_inhibitory'}
        filtered_neuron_params = {k: v for k, v in neuron_params.items() if k in valid_neuron_keys}
    except ValueError: # Handles case where inspect.signature fails on dummy class
        print("Warning: Could not inspect LIF neuron signature. Using all neuron_params.")
        filtered_neuron_params = neuron_params

    # Create neurons layer by layer
    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num - 1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))
        print(f"  Layer {layer_num}: Neurons {start_idx} to {end_idx-1} ({n_layer} neurons)")
        for current_node_index in range(start_idx, end_idx):
             is_inhib = random.random() < network.inhibitory_fraction
             neuron = Layered_LIFNeuronWithReversal(is_inhibitory=is_inhib, **filtered_neuron_params)
             neuron.layer = layer_num # Assign layer attribute
             # Assign position with jitter
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                         random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[current_node_index] = node_pos # Store position dict for plotting
             # Add the created neuron to the network object
             network.add_neuron(neuron, current_node_index, node_pos, layer_num)
        start_idx = end_idx # Update start index for the next layer

    # Calculate Max Distance for Delay Normalization
    max_possible_dist = 1.0 # Default value
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        if all_x and all_y: # Check if lists are not empty
             dist_sq = (max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2
             if dist_sq > 1e-9: # Avoid sqrt of tiny number
                 max_possible_dist = np.sqrt(dist_sq)

    min_delay = max(0.1, dt) # Minimum delay is one time step or 0.1ms

    # Add Connections with Distance-Dependent Delays and Random Weights
    print(f"Creating connections with distance-dependent delays (Base: {base_transmission_delay} ms)...")
    connection_count = 0
    for i in range(total_neurons):
        # Check if source neuron exists and has needed attributes
        if i >= len(network.neurons) or network.neurons[i] is None or i not in network.graph.nodes or i not in pos: continue
        is_source_inhibitory = network.neurons[i].is_inhibitory
        layer_i = network.graph.nodes[i].get('layer', -1)
        pos_i = pos[i]

        for j in range(total_neurons):
            # Check if target neuron exists, not self, and has position
            if i == j or j not in network.graph.nodes or j not in pos: continue
            layer_j = network.graph.nodes[j].get('layer', -1)
            pos_j = pos[j]
            prob = 0.0; connect = False; weight = 0.0 # Reset flags

            # Determine connection probability based on layers and neuron types
            if layer_i != -1 and layer_j != -1: # Ensure layers are defined
                 layer_diff = layer_j - layer_i
                 if is_source_inhibitory:
                     # Inhibitory connections (simplified logic from genetic_experiment)
                     if layer_diff == 0: prob = conn_probs_actual['inh_recurrent']
                     elif abs(layer_diff) == 1: prob = conn_probs_actual['feedforward_1'] * 0.2 # Example scaling
                     if random.random() < prob:
                         weight = -1.0 * random.uniform(weight_min, weight_max) # Negative weight
                         connect = True
                 else: # Excitatory connections
                     if layer_diff == 0: prob = conn_probs_actual['exc_recurrent']
                     elif layer_diff == 1: prob = conn_probs_actual['feedforward_1']
                     elif layer_diff == 2: prob = conn_probs_actual['feedforward_2']
                     elif layer_diff == -1: prob = conn_probs_actual['feedback_1']
                     elif layer_diff == -2: prob = conn_probs_actual['feedback_2']
                     elif layer_diff > 2: prob = conn_probs_actual['long_feedforward']
                     elif layer_diff < -2: prob = conn_probs_actual['long_feedback']
                     if random.random() < prob:
                         weight = 1.0 * random.uniform(weight_min, weight_max) # Positive weight
                         connect = True

            # If connection should be made based on probability and weight is significant
            if connect and abs(weight) > 1e-9: # Use a small epsilon to check non-zero
                # Calculate distance and delay
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                # Linear delay scaling (ensure max_possible_dist is > 0)
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist)) if max_possible_dist > 1e-9 else base_transmission_delay * 0.5
                delay = max(min_delay, delay) # Enforce minimum delay
                # Add connection with the calculated weight and delay
                network.add_connection(i, j, weight=weight, delay=delay)
                connection_count += 1

    print(f"Created structure with {connection_count} connections.")
    return network, layer_indices, pos

# --- NEW Visualization Function for MNIST Input ---
def plot_mnist_input_details(image, label, spike_times_list, stim_duration_ms, save_path):
    """
    Creates a 3-panel figure showing the MNIST image, pixel intensity histogram,
    and input spike raster plot.

    Args:
        image (np.array): The (potentially downsampled) MNIST image.
        label (int): The true label of the MNIST digit.
        spike_times_list (list): List of lists, where each inner list contains spike times (ms)
                                 for the corresponding input neuron (pixel).
        stim_duration_ms (float): The total duration for which spikes were generated.
        save_path (str): Path to save the figure.
    """
    print(f"Generating MNIST input visualization...")
    num_input_neurons = len(spike_times_list)
    total_spikes = sum(len(spikes) for spikes in spike_times_list)

    fig = plt.figure(figsize=(10, 8), facecolor='#1a1a1a')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

    # --- Panel 1: MNIST Image ---
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap='gray', interpolation='nearest')
    ax_img.set_title(f"MNIST Input Image (Label: {label}, {image.shape[0]}x{image.shape[1]})", color='white') # Added shape
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values(): spine.set_color('gray')

    # --- Panel 2: Intensity Histogram ---
    ax_hist = fig.add_subplot(gs[0, 1])
    # Normalize pixel values to 0-1 range if they aren't already
    img_flat = image.flatten()
    max_val = img_flat.max()
    if max_val > 1.0: img_flat = img_flat / max_val # Normalize intensities
    ax_hist.hist(img_flat, bins=10, range=(0, 1), color='skyblue', edgecolor='white', alpha=0.8)
    ax_hist.set_title("Input Intensity Distribution", color='white')
    ax_hist.set_xlabel("Pixel Intensity", color='white')
    ax_hist.set_ylabel("Count", color='white')
    ax_hist.set_facecolor('#1a1a1a')
    ax_hist.tick_params(colors='white')
    for spine in ax_hist.spines.values(): spine.set_color('white')
    ax_hist.grid(True, alpha=0.2, axis='y')

    # --- Panel 3: Input Spike Raster ---
    ax_raster = fig.add_subplot(gs[1, :]) # Span both columns at the bottom
    raster_times = []
    raster_neurons = []
    for neuron_idx, spikes in enumerate(spike_times_list):
        for spike_time in spikes:
            raster_times.append(spike_time)
            raster_neurons.append(neuron_idx)

    if raster_times:
        ax_raster.scatter(raster_times, raster_neurons, s=2, color='white', alpha=0.7, marker='|')
    ax_raster.set_title(f"Input Spike Encoding ({total_spikes} spikes)", color='white')
    ax_raster.set_xlabel("Time (ms)", color='white')
    ax_raster.set_ylabel("Input Neuron Index (Pixel)", color='white')
    ax_raster.set_xlim(0, stim_duration_ms)
    ax_raster.set_ylim(-0.5, num_input_neurons - 0.5) # Adjust limits based on number of input neurons
    ax_raster.invert_yaxis() # Often MNIST rasters have pixel 0 at the top
    ax_raster.set_facecolor('#1a1a1a')
    ax_raster.tick_params(colors='white')
    for spine in ax_raster.spines.values(): spine.set_color('white')
    ax_raster.grid(True, alpha=0.15, axis='x')

    # Optional: Add vertical lines similar to example (e.g., every 10ms)
    for t in np.arange(10, stim_duration_ms, 10):
        ax_raster.axvline(t, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

    # Overall figure title
    fig.suptitle(f"MNIST Input Stimulus Details (Class {label})", fontsize=16, color='white', y=0.98) # Indicate class
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title

    # Save the figure
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"Saved MNIST input details plot to {save_path}")
    plt.close(fig) # Close figure to free memory


# --- Main Execution Block ---
if __name__ == "__main__":
    start_overall_time = time.time()
    # -------------------------------------
    # --- Configuration (from genetic_experiment.py) ---
    # MNIST Input Parameters
    # MNIST_EXAMPLE_INDEX = 42 # <<< NO LONGER USED - We search for 0 or 1
    TARGET_MNIST_CLASSES = [0, 1] # <<< FILTER: Only use images with these labels
    MNIST_STIM_DURATION_MS = 50  # Duration used for generating MNIST spikes (ms)
    MAX_FREQ_HZ = 200           # Max frequency for MNIST encoding (Hz)
    DOWNSAMPLE_FACTOR = 4         # <<< MODIFIED: Factor to downsample MNIST image (e.g., 4 -> 7x7)

    # Simulation Duration
    TOTAL_SIMULATION_DURATION_MS = 150 # Total duration for the network to run (ms)

    # SNN Network Structure Parameters (from genetic_experiment.py)
    # Input neurons calculation is now dynamic based on downsampling
    input_neurons = (28 // DOWNSAMPLE_FACTOR) ** 2 # Calculate input neurons (e.g., 49 for factor 4)
    # Using GA structure, output layer is 2 for MNIST digits 0/1 visualization
    layers_config = [input_neurons, 30, 20, 2] # Input, Hidden, Output layers (2 output neurons)
    inhib_frac = 0.2 # Fraction of inhibitory neurons

    # SNN Connection Probability Parameters (from genetic_experiment.py)
    conn_probs = {
        'exc_recurrent': 0.05, 'inh_recurrent': 0.1,
        'feedforward_1': 0.3, 'feedforward_2': 0.15,
        'feedback_1': 0.06, 'feedback_2': 0.02, # Allowing FB2 unlike GA strict config
        'long_feedforward': 0.01, 'long_feedback': 0.005 # Allowing long range unlike GA strict config
    }

    # SNN Neuron Parameters (from genetic_experiment.py)
    neuron_config = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 10.0, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0, # Adjusted from GA's -80.0 to match layered_exp
        'v_noise_amp': 0.0, 'i_noise_amp': 0.00,
        'adaptation_increment': 0.3, 'tau_adaptation': 120,
        'weight_scale': 0.1 # Conceptual scale for weight range
    }

    # Weight Range (from genetic_experiment.py)
    weight_min = 0.002 # Min absolute weight for random generation
    weight_max = 0.067  # Max absolute weight for random generation

    # Base Transmission Delay (using 1.0 as in create_snn_structure call in GA)
    base_delay = .1 # ms

    # Simulation Time Step (from genetic_experiment.py)
    sim_dt = 0.1 # ms

    # Stimulation parameters used within simulation (from genetic_experiment.py)
    stim_config = {
        'strength': 25,            # Conductance applied per MNIST spike event
        'pulse_duration_ms': sim_dt # How long conductance is held (e.g., one time step)
    }
    # --- Animation Flag for Activity Grid ---
    animate_activity = True # Set to True to generate the GIF

    # --- Random Seed ---
    seed = 123 # For reproducibility
    # -------------------------------------

    # --- Load MNIST Data ---
    print("Loading MNIST dataset...")
    found_image = False
    mnist_original_image = None
    mnist_original_label = -1
    mnist_index_used = -1

    try:
        mnist_loader = MNIST_loader()
        # <<< MODIFIED: Search for an image with label 0 or 1 >>>
        # Check if the real loader has a method to get total size, otherwise assume 60000
        try:
            num_images_to_check = mnist_loader.get_data_size() # Use method if available
        except AttributeError:
            num_images_to_check = 60000 # Default for standard MNIST training set

        print(f"Searching for first image with label in {TARGET_MNIST_CLASSES}...")
        for index in range(num_images_to_check):
            label = mnist_loader.get_label(index)
            if label in TARGET_MNIST_CLASSES:
                mnist_original_image = mnist_loader.get_image(index)
                mnist_original_label = label
                mnist_index_used = index
                found_image = True
                print(f"Found image with Label {mnist_original_label} at Index {mnist_index_used}.")
                break # Stop searching once found

        if not found_image:
             print(f"Error: Could not find an MNIST image with label in {TARGET_MNIST_CLASSES} within the first {num_images_to_check} images.")
             exit()

    except Exception as e:
        print(f"Error loading or searching MNIST data: {e}. Exiting.")
        exit()

    # --- Prepare Input ---
    try:
        if DOWNSAMPLE_FACTOR > 1:
            # Use the downsample_image function (real or dummy)
            mnist_image_to_stimulate = downsample_image(mnist_original_image, DOWNSAMPLE_FACTOR)
            print(f"Downsampled image from 28x28 to {mnist_image_to_stimulate.shape} using factor {DOWNSAMPLE_FACTOR}")
        else:
            mnist_image_to_stimulate = mnist_original_image
            print("Using original 28x28 image (no downsampling).")

        img_h, img_w = mnist_image_to_stimulate.shape
        num_input_neurons_actual = img_h * img_w

        # Ensure first layer matches actual input size after downsampling
        # Recalculate input_neurons based on actual downsampled size
        layers_config[0] = num_input_neurons_actual
        print(f"Network input layer size set to: {layers_config[0]} neurons")

        # Initialize the SNNStimulator
        mnist_stimulator = SNNStimulator(
            total_time_ms=MNIST_STIM_DURATION_MS,
            max_freq_hz=MAX_FREQ_HZ
        )
        # Generate the spike times list for the chosen image
        mnist_spike_times = mnist_stimulator.generate_spikes(mnist_image_to_stimulate)
        mnist_total_spikes = sum(len(spikes) for spikes in mnist_spike_times)
        print(f"Generated {mnist_total_spikes} total input spikes for image index {mnist_index_used} (Label: {mnist_original_label}).")
    except Exception as e:
        print(f"Error preparing MNIST input: {e}. Exiting.")
        exit()

    # --- Visualize Input Details --- *BEFORE* creating the network
    try:
        plot_mnist_input_details(
            image=mnist_image_to_stimulate,
            label=mnist_original_label,
            spike_times_list=mnist_spike_times,
            stim_duration_ms=MNIST_STIM_DURATION_MS,
            save_path="mnist_input_stimulus_visualization.png"
        )
    except Exception as e:
        print(f"Error generating input visualization: {e}")


    # --- Create Network Structure ---
    # Use the adapted function to create the network, getting pos and layer_indices
    try:
        network_obj, layer_indices, pos = create_visual_snn_structure(
            n_layers_list=layers_config, # This now uses the updated input layer size
            inhibitory_fraction=inhib_frac,
            connection_probs=conn_probs,
            neuron_params=neuron_config,
            weight_min=weight_min,
            weight_max=weight_max,
            base_transmission_delay=base_delay,
            dt=sim_dt,
            random_seed=seed
        )
    except Exception as e:
        print(f"Error creating network structure: {e}. Exiting.")
        exit()

    # --- Run Simulation for Single Input ---
    print(f"\nStarting simulation for {TOTAL_SIMULATION_DURATION_MS} ms...")
    network_obj.reset_all() # Reset network state

    try:
        # Use the Layered_run_unified_simulation DEFINED IN THIS SCRIPT
        activity_record, neuron_data, stimulation_record = Layered_run_unified_simulation(
            network_obj,
            duration=TOTAL_SIMULATION_DURATION_MS,
            dt=sim_dt,
            # Pass MNIST input spikes
            mnist_input_spikes=mnist_spike_times,
            # Stimulation parameters for the spikes
            stim_interval_strength=stim_config['strength'],
            stim_pulse_duration_ms=stim_config['pulse_duration_ms'],
            # Other parameters (mostly disabled by MNIST input)
            stim_target_indices=list(range(layers_config[0])), # Target is the first layer
            stim_interval=None, # Not used with MNIST input override
            stim_fraction=0.0, # Not used
            track_neurons=None, # Can add specific neurons to track if needed
            stochastic_stim=False, # Not used
            no_stimulation=False # Let the function handle overriding based on mnist_input_spikes
        )
        print("Simulation complete.")
    except Exception as e:
        print(f"Error during simulation: {e}. Exiting.")
        exit()


    # --- Generate Network Visualizations ---
    print("\nGenerating network visualizations...")
    vis_start_time = time.time()

    try:
        # Ensure network_obj has the graph attribute initialized correctly
        if not hasattr(network_obj, 'graph') or network_obj.graph is None:
             print("Error: network_obj.graph is missing. Cannot generate visualizations.")
        else:
            # 1. Network Structure Plot
            Layered_plot_network_connections_sparse(
                network=network_obj,
                pos=pos,
                stimulated_neurons=list(range(layers_config[0])), # Highlight input layer
                edge_percent=100, # Show 5% of edges
                save_path="mnist_vis_network_structure.png"
            )

            # 2. Distance Dependence Plots (for neuron 0, if it exists and has position)
            if 0 in network_obj.graph.nodes() and 0 in pos:
                 weight_fig, delay_fig = Layered_visualize_distance_dependences(
                      network=network_obj,
                      pos=pos,
                      neuron_idx=0,
                      base_transmission_delay=base_delay,
                      save_path_base="mnist_vis_neuron0_dist_dependence"
                 )
                 # Close figures immediately after saving to free memory
                 if weight_fig: plt.close(weight_fig)
                 if delay_fig: plt.close(delay_fig)
            else:
                 print("Skipping distance dependence plots (Neuron 0 not found or lacks position).")

            # 3. Overall Activity and Layer PSTH Plot
            Layered_plot_activity_and_layer_psth(
                 network=network_obj,
                 activity_record=activity_record,
                 layer_indices=layer_indices,
                 dt=sim_dt,
                 stim_times=stimulation_record.get('pulse_starts', []), # Get actual pulse starts from record
                 bin_width_ms=2.0, # Use 2ms bins
                 save_path="mnist_vis_activity_psth.png"
            )

            # 4. Layer-wise Raster Plot
            Layered_plot_layer_wise_raster(
                network=network_obj,
                activity_record=activity_record,
                layer_indices=layer_indices,
                dt=sim_dt,
                stim_times=stimulation_record.get('pulse_starts', []), # Use actual pulse starts
                save_path="mnist_vis_layer_raster.png"
            )

            # 5. Activity Grid Animation (if enabled)
            if animate_activity:
                Layered_visualize_activity_layout_grid(
                    network=network_obj,
                    pos=pos,
                    activity_record=activity_record,
                    dt=sim_dt,
                    stim_record=stimulation_record, # Pass full stim record for pulse visualization
                    grid_resolution=(120, 180), # Adjust grid size as needed based on layout density
                    save_path="mnist_vis_activity_animation.gif",
                    max_frames=int(TOTAL_SIMULATION_DURATION_MS / sim_dt), # Use all frames for this short sim
                    fps=25 # Animation speed
                )
            else:
                print("Skipping activity grid animation generation.")

    except Exception as e:
        print(f"Error during visualization generation: {e}")

    vis_end_time = time.time()
    print(f"Visualizations generated in {vis_end_time - vis_start_time:.2f}s.")

    # --- Final ---
    overall_end_time = time.time()
    print(f"\n--- Script Finished ---")
    print(f"Total Time: {overall_end_time - start_overall_time:.2f}s")

    # Close any remaining plot figures
    plt.close('all')