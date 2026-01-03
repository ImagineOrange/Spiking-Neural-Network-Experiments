# [Previous code remains the same up to the end of imports]
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import inspect # Import inspect module for introspection (e.g., getting function signatures)


# Set dark style for all plots for consistency
plt.style.use('dark_background')
sns.set_style("darkgrid", {"axes.grid": False}) # Use darkgrid style but disable grid lines by default
plt.rcParams['grid.alpha'] = 0.2 # Make grid lines subtle if they are enabled later

from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal
from LIF_objects.LayeredNeuronalNetwork import LayeredNeuronalNetwork

from LIF_utils.simulation_utils import Layered_run_unified_simulation
from LIF_utils.network_vis_utils import Layered_plot_network_connections_sparse, Layered_visualize_activity_layout_grid
from LIF_utils.activity_vis_utils import Layered_plot_activity_and_layer_psth, Layered_plot_layer_wise_raster, Layered_visualize_distance_dependences




# --- Main Experiment Function ---
def run_6_layer_experiment(
    n_layers_list,          # List with number of neurons per layer (e.g., [100, 150, ...])
    inhibitory_fraction,    # Overall fraction of inhibitory neurons
    connection_probs,       # Dictionary defining connection probabilities between/within layers
    duration,               # Simulation duration (ms)
    dt,                     # Simulation time step (ms)
    neuron_params,          # Dictionary of parameters for LIFNeuronWithReversal
    stimulation_params,     # Dictionary defining stimulation protocol
    pruning_threshold,      # Threshold below which to prune weak connections
    weight_min, weight_max, # Range for random synaptic weights
    base_transmission_delay,# Base delay used for distance-dependent calculation
    random_seed=42,         # Seed for reproducibility
    animate=False):         # Flag to enable/disable GIF animation generation
    """
    Sets up and runs a simulation of a multi-layered network with specific
    connectivity rules and distance-dependent delays, then generates visualizations.
    """
    print("--- Running 6-Layer Network Experiment (Distance-Dependent Delay) ---")
    # Set random seeds for numpy and random modules
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # --- Parameter Extraction and Setup ---
    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    # Extract connection probabilities from dictionary
    default_exc_rec = connection_probs['exc_recurrent'] # Excitatory recurrent
    default_inh_rec = connection_probs['inh_recurrent'] # Inhibitory recurrent
    default_ff_1 = connection_probs['feedforward_1']    # Feedforward +1 layer
    default_ff_2 = connection_probs['feedforward_2']    # Feedforward +2 layers
    default_fb_1 = connection_probs['feedback_1']       # Feedback -1 layer
    default_fb_2 = connection_probs['feedback_2']       # Feedback -2 layers
    long_ff_prob = connection_probs.get('long_feedforward', 0.01) # Long-range feedforward (>2 layers)
    long_fb_prob = connection_probs.get('long_feedback', 0.005) # Long-range feedback (< -2 layers)

    # Extract stimulation parameters
    stochastic_stim = stimulation_params['stochastic']
    no_stimulation = stimulation_params['none']
    stim_interval = stimulation_params['interval_ms']
    stim_interval_strength = stimulation_params['strength']
    stim_fraction = stimulation_params['fraction']
    stim_pulse_duration_ms = stimulation_params.get('pulse_duration_ms', dt) # Duration of stim pulse

    # --- 1. Create Network Instance ---
    print(f"Initializing network object for {total_neurons} neurons...")
    # Pass neuron_params (like noise, reversal potentials) to the network constructor
    network = LayeredNeuronalNetwork(n_neurons=total_neurons,
                                                inhibitory_fraction=inhibitory_fraction,
                                                **neuron_params)
    network.graph.graph['num_layers'] = num_layers # Store number of layers in graph metadata

    # --- 2. Create Neurons and Assign Positions/Layers ---
    print("Creating neurons and assigning layers/original positions...")
    pos = {} # Dictionary to store neuron positions {neuron_id: (x, y)}
    # Define x-coordinates for layers (spread across horizontal axis)
    x_coords = np.linspace(0.1, 0.9, num_layers)
    horizontal_spread = 0.04 # Small horizontal jitter within a layer
    vertical_spread = total_neurons / 20.0 # Vertical spread (adjust based on total neurons)
    layer_indices = [] # List to store (start_idx, end_idx) for each layer
    start_idx = 0 # Running index for neuron IDs

    # Filter neuron_params to only include valid arguments for LIFNeuronWithReversal constructor
    neuron_init_params = inspect.signature(Layered_LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_params if k != 'self' and k!= 'is_inhibitory'}
    filtered_neuron_params = {k: network.neuron_params[k] for k in valid_neuron_keys if k in network.neuron_params}

    # Iterate through layers defined in n_layers_list
    for layer_num, n_layer in enumerate(n_layers_list, 1): # Start layer numbering from 1
        x_layer = x_coords[layer_num-1] # Base x-coordinate for this layer
        end_idx = start_idx + n_layer # End index for neurons in this layer
        layer_indices.append((start_idx, end_idx)) # Store index range for the layer
        print(f"  Layer {layer_num}: Neurons {start_idx} to {end_idx-1}")
        # Create neurons for the current layer
        for current_node_index in range(start_idx, end_idx):
             # Randomly determine if neuron is inhibitory based on overall fraction
             is_inhib = random.random() < network.inhibitory_fraction
             # Create neuron instance with filtered parameters
             neuron = Layered_LIFNeuronWithReversal(is_inhibitory=is_inhib, **filtered_neuron_params)
             neuron.layer = layer_num # Assign layer number to neuron object
             # Assign position with jitter
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                         random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[current_node_index] = node_pos # Store position
             # Add the created neuron to the network object
             network.add_neuron(neuron, current_node_index, node_pos, layer_num)
        start_idx = end_idx # Update start index for the next layer

    # Identify indices for the first layer (potential stimulation target)
    first_layer_start, first_layer_end = layer_indices[0]
    first_layer_indices = list(range(first_layer_start, first_layer_end))
    print(f"Targeting stimulation to Layer 1 indices: {first_layer_start}-{first_layer_end-1}")

    # --- Calculate Max Distance for Delay Normalization ---
    # (Repeated from visualize_distance_dependences, could be refactored)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    max_possible_dist = np.sqrt((max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2) if len(pos)>1 else 1.0
    if max_possible_dist < 1e-6: max_possible_dist = 1.0

    # --- 3. Add Connections with Distance-Dependent Delays ---
    print(f"Creating initial connections with distance-dependent delays (Base: {base_transmission_delay} ms)...")
    connection_count = 0
    min_delay = 0.1 # Minimum allowed transmission delay (ms)

    # Iterate through all possible pairs of neurons (i -> j)
    for i in range(total_neurons):
        # Skip if source neuron doesn't exist or lacks position data
        if i >= len(network.neurons) or not network.neurons[i] or i not in network.graph.nodes or i not in pos: continue
        is_source_inhibitory = network.neurons[i].is_inhibitory
        layer_i = network.graph.nodes[i]['layer']
        pos_i = pos[i]

        for j in range(total_neurons):
            # Skip self-connections or if target neuron doesn't exist/lack position
            if i == j or j not in network.graph.nodes or j not in pos: continue
            layer_j = network.graph.nodes[j]['layer']
            pos_j = pos[j]
            prob = 0.0; weight = 0.0; connect = False # Initialize connection variables

            # Determine connection probability and potential weight based on neuron types and layers
            if is_source_inhibitory: # Source is inhibitory
                if layer_i == layer_j: # Recurrent inhibition
                    prob = default_inh_rec
                    if random.random() < prob:
                        weight = -1.0 * random.uniform(weight_min, weight_max) # Negative weight
                        connect = True
            else: # Source is excitatory
                layer_diff = layer_j - layer_i # Calculate layer difference
                # Assign probability based on layer difference
                if layer_diff == 0: prob = default_exc_rec  # Recurrent excitation
                elif layer_diff == 1: prob = default_ff_1   # Short feedforward (+1)
                elif layer_diff == 2: prob = default_ff_2   # Medium feedforward (+2)
                elif layer_diff == -1: prob = default_fb_1  # Short feedback (-1)
                elif layer_diff == -2: prob = default_fb_2  # Medium feedback (-2)
                elif layer_diff > 2: prob = long_ff_prob    # Long feedforward
                elif layer_diff < -2: prob = long_fb_prob   # Long feedback

                # Connect based on probability
                if random.random() < prob:
                    weight = 1.0 * random.uniform(weight_min, weight_max) # Positive weight
                    connect = True

            # If connection is made, calculate delay and add to network
            if connect:
                # Calculate Euclidean distance between neurons
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                # Calculate delay: linearly scales with distance from min_delay up to base_delay*1.0
                # delay = base * (min_factor + scale_factor * normalized_distance)
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist))
                delay = max(min_delay, delay) # Enforce minimum delay
                # Add connection to network object
                network.add_connection(i, j, weight, delay=delay)
                connection_count += 1

    print(f"Initially added {connection_count} connections.")
    print(f"Initial edge count in graph: {network.graph.number_of_edges()}.")

    # --- 4. Prune Weak Connections ---
    # Remove connections with weights too small to have significant effect
    print(f"Pruning connections with |weight| < {pruning_threshold}...")
    network.prune_weak_connections(threshold=pruning_threshold)
    print(f"Edge count after pruning: {network.graph.number_of_edges()}.")

    # --- Visualize Network Structure ---
    print("Visualizing network structure (sparse)...")
    # Select a small subset of stimulated neurons for cleaner visualization
    stimulated_targets_for_vis = first_layer_indices[:min(5, len(first_layer_indices))]
    connected_for_vis = [] # Placeholder if not tracking specific downstream neurons

    # Call the sparse plotting function
    Layered_plot_network_connections_sparse(
         network=network,
         pos=pos,
         stimulated_neurons=stimulated_targets_for_vis,
         connected_neurons=connected_for_vis,
         edge_percent=5, # Show only 5% of edges
         save_path=f"{num_layers}_layer_network_structure.png"
    )

    # --- 5. Visualize Distance Dependencies ---
    print("Visualizing distance dependencies...")
    neuron_to_vis = first_layer_start # Choose first neuron of layer 1
    # Check if neuron exists and has position before visualizing
    if neuron_to_vis in network.graph.nodes() and neuron_to_vis in pos:
         # Call visualization function
         vis_figs = Layered_visualize_distance_dependences(
              network=network,
              pos=pos,
              neuron_idx=neuron_to_vis,
              base_transmission_delay=base_transmission_delay,
              save_path_base=f"{num_layers}_layer_neuron{neuron_to_vis}_dist_dependence"
         )
         # Close figures immediately after saving to prevent display issues in non-interactive runs
         if vis_figs:
              for fig in vis_figs:
                   if fig: plt.close(fig)
    else:
         print(f"Warning: Cannot visualize distance for neuron {neuron_to_vis}, not found.")

    # --- 6. Run Simulation ---
    print(f"\nStarting simulation for {duration} ms...")
    track_neurons = [] # Currently disabling detailed tracking for performance
    network.reset_all() # Reset network state before simulation
    # Run the simulation using the unified function
    activity_record, neuron_data, stimulation_record = Layered_run_unified_simulation(
        network, duration=duration, dt=dt,
        stim_interval=stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stim_target_indices=first_layer_indices, # Stimulate only layer 1
        stim_pulse_duration_ms=stim_pulse_duration_ms,
        track_neurons=track_neurons,
        stochastic_stim=stochastic_stim,
        no_stimulation=no_stimulation
    )

    # --- 7. Post-Simulation Visualizations ---
    print("\nGenerating post-simulation visualizations...")
    # Get stimulation times (pulse starts) for marking on plots
    stim_times = stimulation_record.get('pulse_starts', []) if not no_stimulation else None

    # Plot overall activity and layer PSTHs
    Layered_plot_activity_and_layer_psth(
         network=network, activity_record=activity_record, layer_indices=layer_indices,
         dt=dt, stim_times=stim_times, bin_width_ms=2.0, # Use 2ms bins for PSTH
         save_path=f"{num_layers}_layer_activity_psth.png"
    )

    # Plot layer-wise raster plots
    Layered_plot_layer_wise_raster(
        network=network, activity_record=activity_record, layer_indices=layer_indices,
        dt=dt, stim_times=stim_times,
        save_path=f"{num_layers}_layer_raster.png"
    )

    # Generate activity animation GIF if requested
    if animate:
        Layered_visualize_activity_layout_grid(
            network=network, pos=pos, activity_record=activity_record, dt=dt,
            stim_record=stimulation_record, # Pass full stim record for pulse duration info
            grid_resolution=(120, 180), # Resolution of the animation grid
            save_path=f"{num_layers}_layer_layout_grid_animation.gif",
            max_frames=int(duration*10), # Limit frames based on duration
            fps=25 # Animation frames per second
        )
    else: # If animation is disabled
        print("Skipping layout grid animation generation (animate=False).")

    print("\n--- Experiment Complete ---")
    # Return simulation results
    return network, activity_record, neuron_data, stimulation_record


# --- Run the Experiment ---
# This block executes only when the script is run directly (not imported)
if __name__ == "__main__":
    # -------------------------------------
    # --- Experiment Configuration ---
    # --- Network Parameters ---
    layers_config = [196, 147, 116, 90, 70, 9] # Number of neurons in each of the 6 layers
    inhib_frac = 0.45 # Overall fraction of inhibitory neurons

    # --- Connection Probabilities (Dictionary) ---
    conn_probs = {
        'exc_recurrent': 0.05,    # Excitatory within-layer connection probability
        'inh_recurrent': 0.1,     # Inhibitory within-layer connection probability
        'feedforward_1': 0.40,    # Excitatory connection probability to layer +1
        'feedforward_2': 0.1,    # Excitatory connection probability to layer +2
        'feedback_1': 0.05,       # Excitatory connection probability to layer -1
        'feedback_2': 0.01,       # Excitatory connection probability to layer -2
        'long_feedforward': 0.005,# Excitatory connection probability to layer > +2
        'long_feedback': 0.002    # Excitatory connection probability to layer < -2
    }

    # --- Neuron Parameters (Dictionary) ---
    neuron_config = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0, # Basic LIF params (mV)
        'tau_m': 10.0, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0, # Time constants (ms)
        'e_reversal': 0.0, 'i_reversal': -70.0,                 # Reversal potentials (mV)
        'v_noise_amp': 0.32, 'i_noise_amp': 0.04,              # Noise amplitudes
        'adaptation_increment': 0.5, 'tau_adaptation': 100,    # Adaptation params
        'weight_scale': 0.1, # Base scale factor for weights (might be overridden by weight_config)
        # STD (Short-Term Depression) parameters
        'std_enabled': False,  # Enable Short-Term Synaptic Depression (default: False)
        'U': 0.3,              # STD utilization factor (fraction of resources released per spike)
        'tau_d': 400.0         # STD recovery time constant in ms
    }

    # --- Synaptic Weight Range ---
    weight_config = {
        'min': 0.003, # Minimum absolute weight for random generation
        'max': 0.02   # Maximum absolute weight for random generation
    }
    

    # --- Connection Pruning ---
    prune_thresh = 0.00 # Threshold for removing weak connections (0 means no pruning)

    # --- Transmission Delay ---
    base_delay = .1 # Base delay in ms used for distance-dependent calculation

    # --- Simulation Parameters ---
    sim_duration = 300 # Total simulation time (ms)
    sim_dt = 0.1       # Simulation time step (ms)

    # --- Stimulation Parameters (Dictionary) ---
    stim_config = {
        'stochastic': False,        # Use regular interval stimulation?
        'none': False,              # Turn stimulation off completely?
        'interval_ms': 30,          # Interval between stim pulses (ms) if not stochastic/none
        'strength': 50,            # Conductance strength of stimulation pulse
        'fraction': 1,            # Fraction of target population to stimulate per pulse
        'pulse_duration_ms': 5     # How long each stimulation pulse lasts (ms)
    }
    # --- Animation Flag ---
    animate = True # Generate the activity GIF?

    # --- Random Seed ---
    seed = 123 # For reproducibility
    # -------------------------------------

    # Call the main experiment function with the configured parameters
    network_obj, activity, tracked_data, stim_info = run_6_layer_experiment(
        n_layers_list=layers_config,
        inhibitory_fraction=inhib_frac,
        connection_probs=conn_probs,
        duration=sim_duration,
        dt=sim_dt,
        neuron_params=neuron_config,
        stimulation_params=stim_config,
        pruning_threshold=prune_thresh,
        weight_min=weight_config['min'],
        weight_max=weight_config['max'],
        base_transmission_delay = base_delay, # Pass the base delay
        random_seed=seed,
        animate=animate # Pass the animation flag
    )

    # Close all matplotlib figures after saving (useful in scripts)
    plt.close('all')

    # plt.show() # Typically not needed if figures are saved and running non-interactively
