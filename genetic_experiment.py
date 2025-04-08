# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background') # Apply style early
import time # To measure execution time
import matplotlib.gridspec as gridspec # For creating complex subplot layouts
import random
import inspect # Import inspect module for introspection
from collections import deque # Used in LayeredNeuronalNetwork
from tqdm import tqdm # Progress bar
import copy # For deep copying chromosomes/networks
import os # For creating directories
import multiprocessing # For parallelization

# --- Import custom modules ---
# Assume these are available in the same directory or PYTHONPATH
# You might need to adjust these import paths based on your project structure
try:
    from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal
    from LIF_objects.LayeredNeuronalNetwork import LayeredNeuronalNetwork # Base class
    from MNIST_stimulation_encodings import MNIST_loader, SNNStimulator, downsample_image
except ImportError:
    print("\n--- WARNING ---")
    print("Could not import LIF objects or MNIST utilities.")
    print("Please ensure 'LIF_objects' and 'MNIST_stimulation_encodings.py' are accessible.")
    print("Using dummy classes/functions as placeholders.")
    print("---------------")
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
        def update_network(self, dt): return [] # No spikes
        def get_weights_sparse(self, connection_map): return np.zeros(len(connection_map))
        def set_weights_sparse(self, weights, connection_map): pass
    class MNIST_loader:
        def __init__(self): print("Using dummy MNIST_loader."); self.images = np.random.rand(100, 784); self.labels = np.random.randint(0, 10, 100)
        def get_image(self, index): return self.images[index].reshape(28, 28) if 0 <= index < len(self.images) else np.zeros((28,28))
        def get_label(self, index): return self.labels[index] if 0 <= index < len(self.labels) else -1
    class SNNStimulator:
         def __init__(self, total_time_ms, max_freq_hz): self.total_time_ms = total_time_ms; self.max_freq_hz = max_freq_hz
         def generate_spikes(self, image): num_pixels = image.size; return [list(np.sort(np.random.uniform(0, self.total_time_ms, np.random.randint(0,5)))) for _ in range(num_pixels)]
    def downsample_image(image, factor): print(f"Dummy downsample_image called with factor {factor}."); return image


# --- Add necessary methods to LayeredNeuronalNetwork ---
# Monkey-patch the methods onto the imported class if they don't exist
# Ideally, modify the class definition in LayeredNeuronalNetwork.py directly.
if not hasattr(LayeredNeuronalNetwork, 'get_weights_sparse'):
    def get_weights_sparse(self, connection_map):
        sparse_weights = np.zeros(len(connection_map))
        for i, (u, v) in enumerate(connection_map):
            if u < self.n_neurons and v < self.n_neurons:
                 sparse_weights[i] = self.weights[u, v]
            else:
                 print(f"Warning: Invalid index ({u},{v}) in connection_map during get_weights_sparse.")
        return sparse_weights
    LayeredNeuronalNetwork.get_weights_sparse = get_weights_sparse

# --- Add necessary methods to LayeredNeuronalNetwork ---
# (Assuming this function is correctly placed or monkey-patched)
# ... (keep the get_weights_sparse function as is) ...

# MODIFY THIS FUNCTION:
if not hasattr(LayeredNeuronalNetwork, 'set_weights_sparse'):
    # Define the function (or redefine if it exists)
    def set_weights_sparse(self, sparse_weights_vector, connection_map):
        """
        Sets network weights from a sparse vector, applying sign based on
        the source neuron's inhibitory status.
        Assumes sparse_weights_vector contains POSITIVE weight magnitudes.
        """
        if len(sparse_weights_vector) != len(connection_map):
            raise ValueError(f"Sparse weight vector length {len(sparse_weights_vector)} "
                             f"does not match connection map length {len(connection_map)}")

        for i, (u, v) in enumerate(connection_map):
             # Check if source (u) and target (v) neuron indices are valid
             # Also check if the neuron object exists at index u
             if u < self.n_neurons and v < self.n_neurons and u < len(self.neurons) and self.neurons[u] is not None:
                 weight_magnitude = sparse_weights_vector[i] # Get magnitude from GA vector

                 # --- START MODIFICATION ---
                 # Check if the SOURCE neuron 'u' is inhibitory
                 if self.neurons[u].is_inhibitory:
                     # Apply negative sign for inhibitory source neurons
                     final_weight = -abs(weight_magnitude) # Ensure magnitude is used and made negative
                 else:
                     # Use positive sign for excitatory source neurons
                     final_weight = abs(weight_magnitude) # Ensure magnitude is used and is positive
                 # --- END MODIFICATION ---

                 # Store the final signed weight in the matrix
                 self.weights[u, v] = final_weight
                 # Update the graph attribute as well if the graph exists and has the edge
                 if self.graph is not None and self.graph.has_edge(u, v):
                     self.graph[u][v]['weight'] = final_weight
             else:
                 # Print warning if indices are invalid or neuron doesn't exist
                 print(f"Warning: Invalid index ({u},{v}) or missing neuron during set_weights_sparse.")

    # Assign the modified function to the class
    LayeredNeuronalNetwork.set_weights_sparse = set_weights_sparse
# --- End of LayeredNeuronalNetwork Modifications ---
# --- End of LayeredNeuronalNetwork Modifications ---


# --- Dummy plotting functions (placeholders if LIF_utils are not used/available) ---
# These are not strictly needed for the GA logic but might be called if plots were enabled.
def Layered_plot_network_connections_sparse(*args, **kwargs): pass
def Layered_visualize_activity_layout_grid(*args, **kwargs): pass
def Layered_visualize_distance_dependences(*args, **kwargs): return None, None
def Layered_plot_activity_and_layer_psth(*args, **kwargs): pass
def Layered_plot_layer_wise_raster(*args, **kwargs): pass

# --- Simulation Function (Handles MNIST input, copied from previous script) ---
# Note: This uses the function definition from visualize_mnist_single.py
def Layered_run_unified_simulation(network, duration=1000.0, dt=0.1,
                                 # Original stim params (will be disabled if mnist_input_spikes is used)
                                 stim_interval=None, stim_interval_strength=10,
                                 stim_fraction=0.01, stim_target_indices=None,
                                 stim_pulse_duration_ms=1.0,
                                 # New MNIST input param
                                 mnist_input_spikes=None,
                                 # Other params
                                 track_neurons=None, stochastic_stim=False,
                                 no_stimulation=False, show_progress=False): # Added show_progress flag
    """
    Runs the network simulation handling various stimulation types and tracking.
    MODIFIED: Accepts pre-generated MNIST spike times (`mnist_input_spikes`).
              If provided, it overrides other stimulation methods.
              Includes optional progress bar control.
    """
    if mnist_input_spikes is not None:
        no_stimulation = True # MNIST provides the stimulation
        stochastic_stim = False
        stim_interval = None
        num_input_neurons = len(mnist_input_spikes)
        mnist_spikes_by_step = {}
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                if step_index >= 0:
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    mnist_spikes_by_step[step_index].append(neuron_idx)

    n_steps = int(duration / dt)
    activity_record = []
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms, 'times': []}
    neuron_data = {} # Not tracking detailed data for GA

    ongoing_stimulations = {}
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None
    stimulation_population = list(stim_target_indices) if stim_target_indices is not None else list(range(network.n_neurons))

    # Simulation Loop Setup
    sim_loop_iterator = range(n_steps)
    if show_progress: # Control tqdm display
         # leave=False prevents multiple bars sticking around in parallel execution
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step", leave=False, ncols=80)

    for step in sim_loop_iterator:
        current_time = step * dt
        newly_stimulated_indices = []

        # --- Apply MNIST Input Stimulation (if provided) ---
        if mnist_input_spikes is not None:
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                stim_end_time = current_time + stim_pulse_duration_ms
                for idx in neurons_spiking_now:
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

        # --- Update Ongoing Stimulations ---
        expired_stims = []
        for neuron_idx, end_time in list(ongoing_stimulations.items()):
            if current_time >= end_time:
                expired_stims.append(neuron_idx)
            else:
                if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
                     if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                        network.neurons[neuron_idx].apply_external_stimulus(stim_interval_strength)

        # Remove expired stimulations
        for neuron_idx in expired_stims:
            if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
                 if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                     network.neurons[neuron_idx].apply_external_stimulus(0.0)
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Update Network State ---
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

    # --- Cleanup After Loop ---
    for neuron_idx in list(ongoing_stimulations.keys()):
        if neuron_idx < len(network.neurons) and network.neurons[neuron_idx] is not None:
            if hasattr(network.neurons[neuron_idx], 'apply_external_stimulus'):
                 network.neurons[neuron_idx].apply_external_stimulus(0.0)
        if neuron_idx in ongoing_stimulations:
             del ongoing_stimulations[neuron_idx]

    return activity_record, neuron_data, stimulation_record


# --- Refactored Network Creation (Builds structure & connection map) ---
def create_snn_structure(n_layers_list, inhibitory_fraction, connection_probs,
                        neuron_params, weight_min=0.0, weight_max=1.0,
                        base_transmission_delay=1.0, dt=0.1, random_seed=None, n_classes=5): # Added n_classes
    """
    Creates the SNN structure (neurons, layers, connections topology).
    Weights are initially 0. Returns network, layer indices, positions, and connection_map.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    # Ensure last layer size matches required classes
    if n_layers_list[-1] != n_classes:
         print(f"Structure Creation Warning: Adjusting last layer from {n_layers_list[-1]} to {n_classes}.")
         n_layers_list[-1] = n_classes
         total_neurons = sum(n_layers_list) # Recalculate

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
    try:
        neuron_constructor_sig = inspect.signature(Layered_LIFNeuronWithReversal)
        valid_neuron_keys = {k for k in neuron_constructor_sig.parameters if k != 'self' and k != 'is_inhibitory'}
        filtered_neuron_params = {k: v for k, v in neuron_params.items() if k in valid_neuron_keys}
    except ValueError:
        print("Warning: Could not inspect LIF neuron signature. Using all neuron_params.")
        filtered_neuron_params = neuron_params


    # Create neurons layer by layer
    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num - 1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))
        for current_node_index in range(start_idx, end_idx):
             is_inhib = random.random() < network.inhibitory_fraction
             neuron = Layered_LIFNeuronWithReversal(is_inhibitory=is_inhib, **filtered_neuron_params)
             neuron.layer = layer_num # Assign layer attribute
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                         random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[current_node_index] = node_pos
             network.add_neuron(neuron, current_node_index, node_pos, layer_num)
        start_idx = end_idx

    # Calculate Max Distance for Delay Scaling
    max_possible_dist = 1.0
    if pos:
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        if all_x and all_y:
             dist_sq = (max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2
             if dist_sq > 1e-9: max_possible_dist = np.sqrt(dist_sq)

    min_delay = max(0.1, dt) # Minimum delay

    # Add Connections based on Probability and Build Connection Map
    connection_count = 0
    connection_map = [] # Stores [(u, v), ...] for existing connections
    for i in range(total_neurons):
        if i >= len(network.neurons) or network.neurons[i] is None or i not in network.graph.nodes or i not in pos: continue
        is_source_inhibitory = network.neurons[i].is_inhibitory
        layer_i = network.graph.nodes[i].get('layer', -1)
        pos_i = pos[i]

        for j in range(total_neurons):
            if i == j or j not in network.graph.nodes or j not in pos: continue
            layer_j = network.graph.nodes[j].get('layer', -1)
            pos_j = pos[j]
            prob = 0.0; connect = False

            # Determine connection probability based on layers and neuron types
            if layer_i != -1 and layer_j != -1: # Ensure layers are defined
                 layer_diff = layer_j - layer_i
                 if is_source_inhibitory:
                     if layer_diff == 0: prob = conn_probs_actual['inh_recurrent']
                     elif abs(layer_diff) == 1: prob = conn_probs_actual['feedforward_1'] * 0.2 # Example scaling
                     if random.random() < prob: connect = True
                 else: # Excitatory connections
                     if layer_diff == 0: prob = conn_probs_actual['exc_recurrent']
                     elif layer_diff == 1: prob = conn_probs_actual['feedforward_1']
                     elif layer_diff == 2: prob = conn_probs_actual['feedforward_2']
                     elif layer_diff == -1: prob = conn_probs_actual['feedback_1']
                     elif layer_diff == -2: prob = conn_probs_actual['feedback_2']
                     elif layer_diff > 2: prob = conn_probs_actual['long_feedforward']
                     elif layer_diff < -2: prob = conn_probs_actual['long_feedback']
                     if random.random() < prob: connect = True

            # If connection should be made based on probability
            if connect:
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist)) if max_possible_dist > 1e-9 else base_transmission_delay * 0.5
                delay = max(min_delay, delay) # Enforce minimum delay
                # Add connection with weight 0 (GA sets actual weight)
                network.add_connection(i, j, weight=0.0, delay=delay)
                connection_map.append((i, j))
                connection_count += 1

    return network, layer_indices, pos, connection_map


# --- Refactored Simulation Runner (Only runs simulation) ---
def run_snn_simulation(network, duration, dt, mnist_input_spikes, stimulation_params):
    """Runs the SNN simulation for a given network (weights assumed set) and input."""
    stim_strength = stimulation_params['strength']
    stim_pulse_duration = stimulation_params.get('pulse_duration_ms', dt)
    first_layer_indices = list(range(len(mnist_input_spikes))) if mnist_input_spikes else []

    network.reset_all() # Reset neuron states before simulation run

    # Use the Layered_run_unified_simulation defined in this script
    activity_record, _, _ = Layered_run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        mnist_input_spikes=mnist_input_spikes,
        stim_interval_strength=stim_strength,
        stim_target_indices=first_layer_indices,
        stim_pulse_duration_ms=stim_pulse_duration,
        track_neurons=None, # No detailed tracking needed for GA fitness
        no_stimulation=False, # Let function override based on mnist_input_spikes
        show_progress=False # Disable internal progress bar for fitness evals
    )
    return activity_record


# --- Prediction Calculation (MODIFIED to ignore stimulus period) ---
# Add dt and stim_duration_ms to the function signature
def calculate_prediction(activity_record, layer_indices, dt, stim_duration_ms, n_classes=5): # Added n_classes
    """
    Calculates the predicted class index (0 to N_CLASSES-1) based on output layer
    spike counts, considering only spikes AFTER the stimulation period.
    """
    if not layer_indices: return -1 # No layer data

    output_start_idx, output_end_idx = layer_indices[-1] # Get indices for the last layer
    num_output_neurons = output_end_idx - output_start_idx

    # Verify output layer size matches expected classes
    if num_output_neurons != n_classes:
        # print(f"Prediction Warning: Output layer size {num_output_neurons} != {n_classes}")
        return -1 # Mismatch, cannot make valid prediction

    # --- MODIFICATION START ---
    # Calculate the simulation step index where the stimulus period ends
    start_step = int(round(stim_duration_ms / dt))
    if start_step >= len(activity_record):
        print(f"Warning: stim_duration_ms ({stim_duration_ms}ms) is longer than or equal to simulation duration. No post-stimulus spikes to count.")
        return -1 # Or handle as appropriate, maybe count all spikes?

    # Only consider activity record *after* the stimulus period
    post_stimulus_activity = activity_record[start_step:]
    # --- MODIFICATION END ---


    # Count spikes for each neuron in the output layer *during the post-stimulus period*
    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}
    # --- MODIFICATION: Iterate over the post_stimulus_activity ---
    for step_spikes in post_stimulus_activity:
        for neuron_idx in step_spikes:
            if output_start_idx <= neuron_idx < output_end_idx:
                output_spike_counts[neuron_idx] += 1

    # Determine prediction based on highest spike count
    if sum(output_spike_counts.values()) > 0:
        predicted_neuron_idx = max(output_spike_counts, key=output_spike_counts.get)
        # Predicted label is the index WITHIN the output layer (0 to n_classes-1)
        predicted_label = predicted_neuron_idx - output_start_idx
    else:
        # print("Warning: No output spikes detected AFTER the stimulus period.") # Optional warning
        predicted_label = -1 # No prediction if no output spikes post-stimulus

    return predicted_label


# --- Genetic Algorithm Implementation ---
class GeneticAlgorithm:
    """Implements the Genetic Algorithm logic."""
    def __init__(self, population_size, chromosome_length, fitness_func, fitness_func_args, # Pass fixed args
                 mutation_rate=0.05, mutation_strength=0.01, crossover_rate=0.7,
                 elitism_count=2, tournament_size=5,
                 weight_min=-0.02, weight_max=0.02):
        self.population_size = population_size
        self.chromosome_length = chromosome_length # Should be len(connection_map)
        self.fitness_func = fitness_func # Function handle
        self.fitness_func_args = fitness_func_args # Tuple of fixed arguments for fitness func
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.population = self._initialize_population()
        # Initialize fitness scores to indicate they haven't been calculated yet
        self.fitness_scores = np.full(population_size, -np.inf) # Use -inf or np.nan

    def _initialize_population(self):
        """Creates the initial population of sparse weight vectors."""
        population = []
        # print(f"Initializing population with chromosome length: {self.chromosome_length}") # Optional print
        for _ in range(self.population_size):
            # Chromosome contains weights only for actual connections
            chromosome = np.random.uniform(self.weight_min, self.weight_max, self.chromosome_length)
            population.append(chromosome)
        return population

    def evaluate_population(self, n_cores, show_progress=True): # Changed default to True
        """ Calculates fitness for the entire population using multiprocessing. """
        if n_cores <= 0:
             n_cores = multiprocessing.cpu_count()

        # Tasks = [(chromosome, fixed_arg1, fixed_arg2, ...)]
        # Note: The fitness_func_args tuple already contains the fixed arguments.
        # We just prepend the chromosome to each task definition.
        tasks = [(self.population[i],) + self.fitness_func_args for i in range(self.population_size)]


        print(f"Evaluating population using {min(n_cores, self.population_size)} cores...")
        start_eval_time = time.time()

        pool = None
        try:
            # Limit pool size to population size if fewer individuals than cores
            actual_processes = min(n_cores, self.population_size)
            pool = multiprocessing.Pool(processes=actual_processes)

            results_iterator = pool.starmap(self.fitness_func, tasks)

            # Add tqdm wrapper for overall progress
            if show_progress:
                results_iterator = tqdm(results_iterator, total=self.population_size, desc="Evaluating Pop", ncols=80, leave=False)

            new_fitness_scores = list(results_iterator) # Collect results

            if any(score is None for score in new_fitness_scores):
                 print("Warning: Received None fitness score(s). Check fitness function error handling.")
                 new_fitness_scores = [score if score is not None else -np.inf for score in new_fitness_scores]
            self.fitness_scores = np.array(new_fitness_scores)

        except Exception as e:
            print(f"FATAL Error during parallel fitness evaluation: {e}")
            # Optionally, re-raise the exception if you want the program to stop
            # raise e
            # Or, try to recover by setting all fitness to -inf
            print("Setting all fitness scores to -inf due to error.")
            self.fitness_scores = np.full(self.population_size, -np.inf)

        finally:
            if pool:
                pool.close()
                pool.join()

        eval_time = time.time() - start_eval_time
        # print(f"Evaluation finished in {eval_time:.2f}s") # Can uncomment if needed

    def _tournament_selection(self):
        """Selects a parent using tournament selection based on fitness scores."""
        best_idx = -1
        best_fitness = -np.inf
        # Handle case where population size is smaller than tournament size
        k = min(self.population_size, self.tournament_size)
        if k <= 0: # Should not happen if population_size > 0
             return self.population[np.random.choice(len(self.population))] if self.population else None

        competitor_indices = np.random.choice(self.population_size, k, replace=False)

        for idx in competitor_indices:
            if idx < len(self.fitness_scores) and self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]
                best_idx = idx

        if best_idx == -1 or best_idx >= len(self.population):
             # Fallback: If all competitors had -inf fitness or index issue, pick random from competitors
             best_idx = np.random.choice(competitor_indices) if competitor_indices.size > 0 else np.random.choice(len(self.population))

        return self.population[best_idx]


    def _crossover(self, parent1, parent2):
        """Performs single-point crossover on weight vectors."""
        if random.random() < self.crossover_rate:
            if self.chromosome_length > 1:
                 crossover_point = random.randint(1, self.chromosome_length - 1)
                 child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                 child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                 return child1, child2
            else: # If chromosome length is 1, no crossover possible
                 return parent1.copy(), parent2.copy()
        else:
            return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        """Applies Gaussian mutation and clamps weights."""
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                mutation_val = np.random.normal(0, self.mutation_strength)
                mutated_chromosome[i] += mutation_val
                # Clamp the mutated weight to the allowed range
                mutated_chromosome[i] = np.clip(mutated_chromosome[i], self.weight_min, self.weight_max)
        return mutated_chromosome

    def run_generation(self):
         """Creates the next generation via elitism, selection, crossover, mutation."""
         new_population = []

         # 1. Elitism
         actual_elitism_count = min(self.elitism_count, self.population_size)
         if actual_elitism_count > 0 and len(self.fitness_scores) == self.population_size and np.any(np.isfinite(self.fitness_scores)):
              try:
                   elite_indices = np.argsort(self.fitness_scores)[-actual_elitism_count:]
                   for idx in elite_indices:
                       if idx < len(self.population):
                          new_population.append(self.population[idx].copy())
              except IndexError:
                   print("Warning: Error getting elite indices. Skipping elitism.")
         else:
             # Handle cases where fitness scores might be all -inf or invalid
             pass # Simply don't add elites if scores are bad


         # 2. Generate remaining individuals
         while len(new_population) < self.population_size:
             parent1 = self._tournament_selection()
             parent2 = self._tournament_selection()
             # Ensure parents were selected successfully
             if parent1 is None or parent2 is None:
                  print("Warning: Parent selection failed. Using random individuals.")
                  # Fallback to random initialization if selection fails
                  if len(self.population) > 0:
                       parent1 = self.population[np.random.choice(len(self.population))] if parent1 is None else parent1
                       parent2 = self.population[np.random.choice(len(self.population))] if parent2 is None else parent2
                  else: # Cannot proceed if population is empty
                       break

             child1, child2 = self._crossover(parent1, parent2)
             child1 = self._mutate(child1)
             child2 = self._mutate(child2)

             if len(new_population) < self.population_size:
                  new_population.append(child1)
             if len(new_population) < self.population_size:
                  new_population.append(child2)

         # Handle potential size mismatch (shouldn't happen with checks)
         if len(new_population) > self.population_size:
             new_population = new_population[:self.population_size]

         self.population = new_population
         # Reset fitness scores for the new generation
         self.fitness_scores = np.full(self.population_size, -np.inf)


# --- Fitness Function (MODIFIED call to calculate_prediction) ---
# Note: Arguments prefixed with 'p_' to denote they are passed parameters within the parallel context.
def evaluate_chromosome_fitness(chromosome_weights,       # The weights to evaluate
                                # Network Structure Params (passed to recreate network)
                                p_layers_config, p_inhib_frac, p_conn_probs,
                                p_neuron_config, p_base_delay, p_dt, p_seed,
                                p_connection_map, p_layer_indices,
                                p_n_classes, # Pass number of classes
                                # Data Params
                                p_filtered_images, p_filtered_labels, p_label_map,
                                p_eval_indices, # Specific indices for this evaluation run
                                # Simulation Params
                                p_sim_duration, p_stim_config,
                                p_downsample_factor, p_mnist_stim_duration, p_max_freq_hz):
    """
    Fitness function designed for parallel execution.
    It recreates the network structure locally to ensure process isolation.
    MODIFIED: Passes dt and stimulus duration to calculate_prediction.
    """
    # --- 1. Recreate network structure within the worker process ---
    try:
        # Pass n_classes to ensure correct output layer size
        eval_network, _, _, _ = create_snn_structure(
            n_layers_list=list(p_layers_config), # Pass copy
            inhibitory_fraction=p_inhib_frac,
            connection_probs=p_conn_probs,
            neuron_params=p_neuron_config,
            base_transmission_delay=p_base_delay,
            dt=p_dt,
            random_seed=p_seed, # Use same seed for consistent topology
            n_classes=p_n_classes # Ensure output layer is correct size
        )
        eval_network.reset_all() # Ensure clean state
    except Exception as e:
        print(f"Worker Error: creating SNN structure: {e}")
        return 0.0 # Cannot evaluate

    # --- 2. Set weights for this evaluation using the sparse method ---
    try:
        eval_network.set_weights_sparse(chromosome_weights, p_connection_map)
    except ValueError as e:
        print(f"Worker Error: setting sparse weights: {e}")
        return 0.0

    # --- 3. Create SNN Stimulator locally ---
    mnist_stimulator = SNNStimulator(
        total_time_ms=p_mnist_stim_duration,
        max_freq_hz=p_max_freq_hz
    )

    # --- 4. Run simulation on the specified subset of filtered data ---
    accuracies = []
    if len(p_filtered_labels) == 0: return 0.0 # Handle empty data case

    for idx in p_eval_indices: # Use the specific indices passed for this evaluation
        # Basic check for index validity
        if idx >= len(p_filtered_labels):
             # print(f"Worker Warning: Index {idx} out of bounds. Skipping.")
             continue
        try:
            # Get data using the index relative to the filtered arrays
            mnist_original_image = p_filtered_images[idx].reshape(28,28)
            true_original_label = p_filtered_labels[idx]
            # Map original label (e.g., 4) to network output index (e.g., 3)
            true_mapped_label = p_label_map[true_original_label]

            # Downsample image if needed
            if p_downsample_factor > 1:
                mnist_image = downsample_image(mnist_original_image, p_downsample_factor)
            else:
                mnist_image = mnist_original_image

            # Generate input spikes
            mnist_spike_times = mnist_stimulator.generate_spikes(mnist_image)

            # Run the SNN simulation
            activity_record = run_snn_simulation(
                eval_network,
                duration=p_sim_duration,
                dt=p_dt,
                mnist_input_spikes=mnist_spike_times,
                stimulation_params=p_stim_config
            )

            # --- MODIFICATION: Call calculate_prediction with dt and stim duration ---
            predicted_label = calculate_prediction(
                activity_record,
                p_layer_indices, # Pass layer indices
                dt=p_dt, # Pass dt
                stim_duration_ms=p_mnist_stim_duration, # Pass stimulus duration
                n_classes=p_n_classes # Pass number of classes
            )
            # --- END MODIFICATION ---

            # Record accuracy if a valid prediction was made
            if predicted_label != -1:
                 accuracies.append(1 if predicted_label == true_mapped_label else 0)

        except KeyError: # Handle cases where label might not be in map (shouldn't happen with filtering)
             # print(f"Worker Warning: Label {true_original_label} not in map. Skipping.")
             continue
        except Exception as e:
             # Catch other potential errors during simulation/prediction
             # print(f"Worker Warning: Error during sim for index {idx}: {e}. Skipping.")
             continue

    # --- 5. Return average accuracy as fitness ---
    fitness = np.mean(accuracies) if accuracies else 0.0
    return fitness


# --- Plotting Function for GA Progress ---
def plot_ga_progress(generations, best_fitness_history, avg_fitness_history, final_test_accuracy, filename, fitness_eval_examples):
    """Plots the GA fitness history and saves the figure."""
    fig_ga, ax_ga = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    gen_axis = range(1, generations + 1) # Generation numbers start from 1
    # Plot best and average fitness trends
    ax_ga.plot(gen_axis, best_fitness_history, marker='o', linestyle='-', color='cyan', markersize=4, label='Best Fitness (Eval Set)')
    ax_ga.plot(gen_axis, avg_fitness_history, marker='x', linestyle='--', color='orange', markersize=4, label='Average Fitness (Eval Set)')
    # Styling
    ax_ga.set_title(f'GA Fitness - Generation {generations}', color='white')
    ax_ga.set_xlabel('Generation', color='white')
    ax_ga.set_ylabel(f'Fitness (Accuracy on {fitness_eval_examples} Examples)') # Use param in label
    ax_ga.set_ylim(0, 1.05) # Y-axis from 0% to 105%
    ax_ga.grid(True, alpha=0.3) # Add subtle grid
    ax_ga.legend(loc='lower right', framealpha=0.7)
    ax_ga.set_facecolor('#1a1a1a') # Dark background
    ax_ga.tick_params(colors='white') # White tick labels
    for spine in ax_ga.spines.values(): spine.set_color('white') # White axes lines
    # Add text annotation for final test accuracy if provided
    if final_test_accuracy is not None:
        ax_ga.text(0.95, 0.1, f'Test Acc: {final_test_accuracy:.3f}', transform=ax_ga.transAxes,
                    ha='right', color='lime', bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout() # Adjust layout
    plt.savefig(filename, dpi=150, facecolor='#1a1a1a') # Save the plot
    plt.close(fig_ga) # Close the figure to free memory


# --- Main Execution Block ---
if __name__ == "__main__":
    # Required for multiprocessing spawning compatibility on some OS
    multiprocessing.freeze_support()

    start_overall_time = time.time() # Start overall timer

    # --- Configuration ---
    # MNIST Classes and Mapping
    TARGET_CLASSES = [0, 1] # The specific digits we want to classify
    N_CLASSES = len(TARGET_CLASSES)
    # Create mapping: {Original MNIST Label: New Index (0 to N_CLASSES-1)}
    label_map_global = {original_label: new_index for new_index, original_label in enumerate(TARGET_CLASSES)}
    print(f"--- Running {N_CLASSES}-Class MNIST GA SNN ---")
    print(f"Target Digits: {TARGET_CLASSES} -> Mapped Indices: {list(range(N_CLASSES))}")

    # GA Parameters
    POPULATION_SIZE = 100       # Number of individuals (weight sets) in the population
    NUM_GENERATIONS = 200       # How many generations to run the evolution
    MUTATION_RATE = 0.10       # Probability of a single weight being mutated
    MUTATION_STRENGTH = 0.005  # Std dev of noise added during mutation
    CROSSOVER_RATE = 0.7       # Probability of two parents performing crossover
    ELITISM_COUNT = 4          # Number of best individuals carried directly to next gen
    TOURNAMENT_SIZE = 5        # Number of individuals competing in selection tournament
    FITNESS_EVAL_EXAMPLES = 100 # Number of examples per fitness evaluation (passed to func)
    TEST_SET_EXAMPLES = 100    # Number of examples for final testing

    # Parallelization Config
    N_CORES = 11 # Number of CPU cores for parallel fitness evaluation (adjust as needed, 0 or -1 uses all)
    if N_CORES <= 0: N_CORES = multiprocessing.cpu_count()
    print(f"Using {N_CORES} cores for parallel evaluation.")

    # MNIST Input Parameters (used by SNNStimulator)
    mnist_stim_duration_global = 50  # ms
    max_freq_hz_global = 200.0       # Hz
    downsample_factor_global = 4     # Downsample 28x28 -> 14x14 (196 inputs)

    # SNN Simulation Parameters
    sim_duration_global = 150 # ms (Total simulation time per example)
    sim_dt_global = 0.1      # ms (Simulation time step)

    # SNN Network Structure Parameters
    input_neurons = (28 // downsample_factor_global) ** 2 # Calculate input neurons (196)
    layers_config_global = [input_neurons, 30, 20,  N_CLASSES] # Input, Hidden, Output layers
    inhib_frac_global = 0.2 # Fraction of inhibitory neurons

    # SNN Connection Probability Parameters
    conn_probs_global = {
        'exc_recurrent': 0.12, 'inh_recurrent': 0.15,
        'feedforward_1': 0.3, 'feedforward_2': 0.15, # FF to hidden, FF to output
        'feedback_1': 0.06, 'feedback_2': 0.0,    # FB from hidden only
        'long_feedforward': 0.0, 'long_feedback': 0.0 # No long range needed
    }

    # SNN Neuron Parameters (passed to LIFNeuron class)
    neuron_config_global = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 10.0, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0, # Consistent with visualizer
        'v_noise_amp': 0.0, 'i_noise_amp': 0.00, # Noise levels
        'adaptation_increment': 0.3, 'tau_adaptation': 120, # Adaptation params
        'weight_scale': 0.1 # Only used conceptually for weight range setting below
    }

    # GA Weight Initialization/Clamping Range
    weight_min_ga_global = 0.002
    weight_max_ga_global = 0.067

    # Stimulation parameters used within run_snn_simulation (passed to Layered_run_unified_simulation)
    stim_config_global = {
        'strength': 25,            # Conductance applied per MNIST spike event
        'pulse_duration_ms': sim_dt_global # How long conductance is held
    }

    # Random Seed for reproducibility
    master_seed = 42
    np.random.seed(master_seed)
    random.seed(master_seed)

    # Create Output Directory for results
    output_dir = "ga_mnist_snn_5class_sparse_parallel_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    # -------------------------------------

    # --- Load and Filter MNIST Data ---
    print("Loading MNIST dataset...")
    try:
        mnist_loader = MNIST_loader()
        print(f"Filtering dataset for classes: {TARGET_CLASSES}...")
        mask = np.isin(mnist_loader.labels, TARGET_CLASSES) # Boolean mask for target classes
        filtered_images_global = mnist_loader.images[mask]  # Apply mask to images
        filtered_labels_global = mnist_loader.labels[mask]  # Apply mask to labels
        num_filtered = len(filtered_labels_global)
        print(f"Filtered dataset size: {num_filtered} examples.")

        # Split filtered data indices into training/evaluation pool and test pool
        split_idx = int(num_filtered * 0.8) # Use 80% for GA training/evaluation phase
        train_eval_indices_pool = np.arange(split_idx) # Indices within filtered set for GA
        test_indices_pool = np.arange(split_idx, num_filtered) # Indices within filtered set for final test
        print(f"Using {len(train_eval_indices_pool)} examples for GA eval sampling, {len(test_indices_pool)} for final testing.")
    except Exception as e:
        print(f"Error loading or filtering MNIST data: {e}. Exiting.")
        exit()

    # --- Create Base Network Structure AND Connection Map ---
    print("Creating base SNN structure and connection map...")
    try:
        network_creation_args = { # Store args needed to recreate the structure
            "n_layers_list": list(layers_config_global), # Pass list copy
            "inhibitory_fraction": inhib_frac_global,
            "connection_probs": conn_probs_global,
            "neuron_params": neuron_config_global,
            "base_transmission_delay": 1.0, # Example value from visualizer script
            "dt": sim_dt_global,
            "random_seed": master_seed, # Ensure consistent topology
            "n_classes": N_CLASSES # Pass number of classes
        }
        # Create the initial network to get the map, indices etc.
        _, layer_indices_global, _, connection_map_global = create_snn_structure(**network_creation_args)
        chromosome_len = len(connection_map_global)
        print(f"Network structure defined. Actual connections: {chromosome_len} -> Chromosome length: {chromosome_len}")
    except Exception as e:
        print(f"Error creating base network structure: {e}. Exiting.")
        exit()


    # --- Prepare Fixed Arguments Tuple for Parallel Fitness Function ---
    # These arguments are constant across all fitness evaluations.
    fitness_args_tuple_base = (
        # Network Structure Creation Params
        layers_config_global, inhib_frac_global, conn_probs_global,
        neuron_config_global, network_creation_args['base_transmission_delay'], sim_dt_global, master_seed,
        # Map, Indices, Classes
        connection_map_global, layer_indices_global, N_CLASSES,
        # Data Params
        filtered_images_global, filtered_labels_global, label_map_global,
        None, # Placeholder for p_eval_indices - filled in loop below
        # Simulation Params
        sim_duration_global, stim_config_global,
        downsample_factor_global, mnist_stim_duration_global, max_freq_hz_global
    )
    EVAL_INDICES_ARG_INDEX = 13 # Index of the eval_indices placeholder in the tuple above (count carefully)

    # --- Initialize Genetic Algorithm ---
    print("Initializing Genetic Algorithm...")
    try:
        ga = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            chromosome_length=chromosome_len, # Use sparse length
            fitness_func=evaluate_chromosome_fitness, # Pass function handle
            fitness_func_args=fitness_args_tuple_base, # Pass base tuple (indices added later)
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT,
            tournament_size=TOURNAMENT_SIZE,
            weight_min=weight_min_ga_global,
            weight_max=weight_max_ga_global
        )
        print(f"GA Initialized: Pop={POPULATION_SIZE}, Gens={NUM_GENERATIONS}")
    except Exception as e:
        print(f"Error initializing GA: {e}. Exiting.")
        exit()

    # --- Run Genetic Algorithm ---
    best_fitness_history = []
    avg_fitness_history = []
    print(f"\n--- Starting GA Evolution for {NUM_GENERATIONS} Generations ({N_CORES} cores) ---")
    final_test_accuracy = None

    try: # Wrap the GA loop in a try-except block
        for generation in tqdm(range(NUM_GENERATIONS), desc="Generations"):
            gen_start_time = time.time()

            # Prepare specific eval indices for this generation
            eval_indices_this_gen = np.random.choice(
                train_eval_indices_pool,
                min(FITNESS_EVAL_EXAMPLES, len(train_eval_indices_pool)),
                replace=False
            )
            # Update the arguments tuple for this generation's evaluations
            current_fitness_args = list(ga.fitness_func_args)
            current_fitness_args[EVAL_INDICES_ARG_INDEX] = eval_indices_this_gen
            ga.fitness_func_args = tuple(current_fitness_args) # Update tuple in GA object

            # Evaluate population fitness in parallel
            ga.evaluate_population(n_cores=N_CORES, show_progress=True)

            # Track and report fitness
            if np.all(np.isneginf(ga.fitness_scores)): # Check if all scores are -inf
                 best_gen_fitness = -np.inf
                 avg_gen_fitness = -np.inf
                 print(f"Generation {generation + 1}/{NUM_GENERATIONS} | All fitness evaluations failed!")
            else:
                 # Use nanmax/nanmean to ignore -inf values if some evaluations failed
                 best_gen_fitness = np.nanmax(np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores))
                 avg_gen_fitness = np.nanmean(np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores))

            best_fitness_history.append(best_gen_fitness)
            avg_fitness_history.append(avg_gen_fitness)
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation + 1}/{NUM_GENERATIONS} | Best Fitness: {best_gen_fitness:.4f} | Avg Fitness: {avg_gen_fitness:.4f} | Time: {gen_time:.2f}s")

            # Save plot for this generation
            plot_filename = os.path.join(output_dir, f"ga_fitness_gen_{generation+1:03d}.png")
            plot_ga_progress(generation + 1, best_fitness_history, avg_fitness_history, final_test_accuracy, plot_filename, FITNESS_EVAL_EXAMPLES)

            # Evolve to the next generation
            if generation < NUM_GENERATIONS - 1:
                 ga.run_generation()

    except KeyboardInterrupt:
        print("\nGA execution interrupted by user.")
    except Exception as e:
        print(f"\nError during GA evolution: {e}")

    # --- End of GA Loop ---
    ga_end_time = time.time()
    print(f"\n--- GA Evolution Complete (or interrupted) ---")
    print(f"Total GA time: {ga_end_time - start_overall_time:.2f}s")

    # --- Get Best Chromosome Found ---
    best_chromosome = None
    best_fitness_final = -np.inf
    if len(ga.fitness_scores) > 0 and np.any(np.isfinite(ga.fitness_scores)):
         # Use nanargmax to handle potential NaNs from failed evaluations
         valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
         if np.any(np.isfinite(valid_scores)): # Check if there are any valid scores
              final_best_idx = np.nanargmax(valid_scores)
              if final_best_idx < len(ga.population):
                  best_chromosome = ga.population[final_best_idx]
                  best_fitness_final = valid_scores[final_best_idx] # Use the valid score
                  print(f"Best fitness found during evolution (on eval set): {best_fitness_final:.4f}")

                  # Save Best SPARSE Weights and the Connection Map
                  weights_save_path = os.path.join(output_dir, "best_snn_weights_5class_sparse.npy")
                  map_save_path = os.path.join(output_dir, "connection_map_5class.npy")
                  try:
                      np.save(weights_save_path, best_chromosome)
                      np.save(map_save_path, np.array(connection_map_global, dtype=object))
                      print(f"Saved best sparse weights to {weights_save_path}")
                      print(f"Saved connection map to {map_save_path}")
                  except Exception as e:
                      print(f"Error saving weights or map: {e}")
              else:
                  print("Warning: Best fitness index is out of bounds for population.")
         else:
             print("Warning: All fitness scores were invalid (-inf or NaN). Cannot determine best chromosome.")
    else:
         print("Warning: No valid fitness scores available. Cannot determine best chromosome.")

    # --- Evaluate Best Chromosome on Test Set (MODIFIED call to calculate_prediction) ---
    if best_chromosome is not None:
         print(f"\n--- Evaluating best weights on {TEST_SET_EXAMPLES} filtered test examples ---")
         try:
             # Recreate the network structure for testing
             test_network, test_layer_indices, _, test_connection_map = create_snn_structure(**network_creation_args)
             test_network.reset_all()
             test_network.set_weights_sparse(best_chromosome, test_connection_map) # Use original map

             test_accuracies = []
             # Sample indices from the designated test pool
             if len(test_indices_pool) > 0:
                 actual_test_examples = min(TEST_SET_EXAMPLES, len(test_indices_pool))
                 test_indices_final = np.random.choice(test_indices_pool, actual_test_examples, replace=False)
             else:
                 print("Warning: No test examples available in the pool.")
                 test_indices_final = []
                 actual_test_examples = 0

             # Create stimulator for testing
             mnist_stimulator_test = SNNStimulator(total_time_ms=mnist_stim_duration_global, max_freq_hz=max_freq_hz_global)
             test_loop_iterator = tqdm(test_indices_final, desc="Testing Best Weights", ncols=80)
             correct_test_predictions = 0

             # Test loop
             for filt_idx in test_loop_iterator:
                 try:
                      # Get data, downsample, encode spikes (as before)
                      mnist_original_image = filtered_images_global[filt_idx].reshape(28,28)
                      true_original_label = filtered_labels_global[filt_idx]
                      true_mapped_label = label_map_global[true_original_label]

                      if downsample_factor_global > 1: mnist_image = downsample_image(mnist_original_image, downsample_factor_global)
                      else: mnist_image = mnist_original_image
                      mnist_spike_times = mnist_stimulator_test.generate_spikes(mnist_image)

                      # Run simulation
                      activity_record = run_snn_simulation(
                          test_network, sim_duration_global, sim_dt_global, mnist_spike_times, stim_config_global
                      )

                      # --- MODIFICATION: Call calculate_prediction with dt and stim duration ---
                      predicted_label = calculate_prediction(
                          activity_record,
                          test_layer_indices, # Use layer indices from test network
                          dt=sim_dt_global, # Pass global dt
                          stim_duration_ms=mnist_stim_duration_global, # Pass global stim duration
                          n_classes=N_CLASSES # Pass number of classes
                      )
                      # --- END MODIFICATION ---

                      # Check accuracy
                      if predicted_label != -1:
                          is_correct = (predicted_label == true_mapped_label)
                          if is_correct: correct_test_predictions += 1
                 except Exception as e:
                      # Catch errors during testing loop for a single example
                      print(f"\nWarning: Error during test sim for index {filt_idx}: {e}. Skipping example.")
                      continue # Skip to the next test example

             # Calculate final test accuracy
             final_test_accuracy = correct_test_predictions / actual_test_examples if actual_test_examples > 0 else 0.0
             print(f"\nFinal Accuracy on Filtered Test Set ({actual_test_examples} examples): {final_test_accuracy:.4f}")

         except Exception as e:
             print(f"\nError during final test evaluation: {e}")
             final_test_accuracy = None
    else:
         print("\nSkipping final test evaluation as no best chromosome was found.")
         final_test_accuracy = None


    overall_end_time = time.time() # End overall timer
    print(f"Total script execution time: {overall_end_time - start_overall_time:.2f}s")

    # --- Plot Final GA Fitness History ---
    if best_fitness_history: # Ensure history exists before plotting
         print("\nPlotting final GA fitness history...")
         final_plot_filename = os.path.join(output_dir, "mnist_snn_ga_fitness_plot_final_5class_sparse.png")
         plot_ga_progress(len(best_fitness_history), best_fitness_history, avg_fitness_history, final_test_accuracy, final_plot_filename, FITNESS_EVAL_EXAMPLES)
         print(f"Saved final GA fitness plot to {final_plot_filename}")

    print("--- Script Finished ---")
    # Optional: Close all matplotlib figures at the very end if running interactively
    # plt.close('all')