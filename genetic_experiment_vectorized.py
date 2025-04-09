# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background') # Apply style early
import time # To measure execution time
import random
from collections import deque # Used in LayeredNeuronalNetwork
from tqdm import tqdm # Progress bar
import os # For creating directories
import multiprocessing # For parallelization

# --- Import custom modules ---
# Assume these are available in the same directory or PYTHONPATH
# You might need to adjust these import paths based on your project structure
try:
    # Import the VECTORIZED network class and the neuron class (for parameter inspection)
    from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized # MODIFIED IMPORT
    from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal # Keep for param inspection
    from MNIST_stimulation_encodings import MNIST_loader, SNNStimulator, downsample_image
except ImportError as e:
    print("\n--- WARNING ---")
    print(f"Could not import required modules: {e}")
    print("Please ensure 'LIF_objects' and 'MNIST_stimulation_encodings.py' are accessible.")
    print("Using dummy classes/functions as placeholders.")
    print("---------------")
    # Define dummy classes if import fails
    class LayeredNeuronalNetworkVectorized: # MODIFIED DUMMY CLASS NAME
        def __init__(self, n_neurons, **kwargs):
             self.n_neurons = n_neurons
             self.graph = None; self.weights=np.zeros((n_neurons, n_neurons)); self.delays=np.zeros((n_neurons, n_neurons))
             self.spike_queue=deque(); self.network_activity=[]; self.neuron_params=kwargs; self.neuron_grid_positions={}
             import networkx as nx; self.graph = nx.DiGraph()
             # Dummy state arrays
             self.v = np.zeros(n_neurons); self.g_e = np.zeros(n_neurons); self.g_i = np.zeros(n_neurons)
             self.adaptation = np.zeros(n_neurons); self.t_since_spike = np.zeros(n_neurons)
             self.is_inhibitory = np.zeros(n_neurons, dtype=bool) # Important for set_weights_sparse

        def add_connection(self, u, v, weight, delay):
             if u < self.n_neurons and v < self.n_neurons:
                  if self.graph is None: import networkx as nx; self.graph = nx.DiGraph()
                  if u not in self.graph: self.graph.add_node(u) # Ensure nodes exist
                  if v not in self.graph: self.graph.add_node(v)
                  self.graph.add_edge(u,v,weight=weight,delay=delay)
                  self.weights[u,v]=weight; self.delays[u,v]=delay
        def reset_all(self): self.spike_queue=deque(); self.network_activity=[]
        def update_network(self, dt): return [] # No spikes
        def get_weights_sparse(self, connection_map): return np.zeros(len(connection_map))
        # Define set_weights_sparse directly in the dummy class for basic functionality
        def set_weights_sparse(self, sparse_weights_vector, connection_map):
             if len(sparse_weights_vector) != len(connection_map):
                  raise ValueError("Length mismatch")
             for i, (u, v) in enumerate(connection_map):
                  if u < self.n_neurons and v < self.n_neurons:
                       weight_magnitude = abs(sparse_weights_vector[i])
                       final_weight = -weight_magnitude if self.is_inhibitory[u] else weight_magnitude
                       self.weights[u, v] = final_weight
                       if self.graph is not None and self.graph.has_edge(u, v):
                           self.graph[u][v]['weight'] = final_weight
        # Dummy method for vectorized external stimulation
        def set_external_stimulus(self, indices, values): pass # Placeholder

    class Layered_LIFNeuronWithReversal: # Keep dummy for inspection if needed
        def __init__(self, *args, **kwargs): pass # Minimal placeholder

    class MNIST_loader:
        def __init__(self): print("Using dummy MNIST_loader."); self.images = np.random.rand(100, 784); self.labels = np.random.randint(0, 10, 100)
        def get_image(self, index): return self.images[index].reshape(28, 28) if 0 <= index < len(self.images) else np.zeros((28,28))
        def get_label(self, index): return self.labels[index] if 0 <= index < len(self.labels) else -1
    class SNNStimulator:
         def __init__(self, total_time_ms, max_freq_hz): self.total_time_ms = total_time_ms; self.max_freq_hz = max_freq_hz
         def generate_spikes(self, image): num_pixels = image.size; return [list(np.sort(np.random.uniform(0, self.total_time_ms, np.random.randint(0,5)))) for _ in range(num_pixels)]
    def downsample_image(image, factor): print(f"Dummy downsample_image called with factor {factor}."); return image


# --- REMOVED Monkey-patching section for get/set_weights_sparse ---
# These methods should now be part of LayeredNeuronalNetworkVectorized


# --- Dummy plotting functions (placeholders if LIF_utils are not used/available) ---
# These are not strictly needed for the GA logic but might be called if plots were enabled.
def Layered_plot_network_connections_sparse(*args, **kwargs): pass
def Layered_visualize_activity_layout_grid(*args, **kwargs): pass
def Layered_visualize_distance_dependences(*args, **kwargs): return None, None
def Layered_plot_activity_and_layer_psth(*args, **kwargs): pass
def Layered_plot_layer_wise_raster(*args, **kwargs): pass


# --- Simulation Function (MODIFIED for vectorized external stimulation) ---
def Layered_run_unified_simulation(network: LayeredNeuronalNetworkVectorized, # Type hint updated
                                 duration=1000.0, dt=0.1,
                                 # Original stim params (may be overridden by MNIST input)
                                 stim_interval=None, stim_interval_strength=10,
                                 stim_fraction=0.01, stim_target_indices=None,
                                 stim_pulse_duration_ms=1.0,
                                 # New MNIST input param
                                 mnist_input_spikes=None,
                                 # Other params
                                 track_neurons=None, # Note: Tracking detailed neuron data needs rework for vectorized state
                                 stochastic_stim=False,
                                 no_stimulation=False, show_progress=False):
    """
    Runs the network simulation handling various stimulation types and tracking.
    MODIFIED: Accepts pre-generated MNIST spike times (`mnist_input_spikes`).
              Uses vectorized network update and stimulation.
              Includes optional progress bar control.
              NOTE: Detailed neuron data tracking (neuron_data) is DISABLED for vectorized network.
    """
    # --- Pre-process MNIST spike times (Same as before) ---
    mnist_spikes_by_step = {}
    if mnist_input_spikes is not None:
        # print("Using MNIST spike input. Disabling other stimulation.") # Optional print
        no_stimulation = True
        stochastic_stim = False
        stim_interval = None
        num_input_neurons = len(mnist_input_spikes)
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                if 0 <= step_index < int(duration / dt):
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    mnist_spikes_by_step[step_index].append(neuron_idx)
    # --- End Pre-processing ---

    n_steps = int(duration / dt)
    activity_record = []
    # Stimulation record structure kept for potential compatibility/analysis
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms, 'times': []}
    # --- Detailed Neuron Data Tracking DISABLED ---
    neuron_data = {} # Keep empty for compatibility, but won't be populated

    ongoing_stimulations = {} # Tracks {neuron_idx: end_time} for managing pulse durations
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None
    stimulation_population = set(stim_target_indices) if stim_target_indices is not None else set(range(network.n_neurons))
    if not stimulation_population: no_stimulation = True

    # --- Simulation Loop ---
    sim_loop_iterator = range(n_steps)
    if show_progress:
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step", leave=False, ncols=80)

    # Array to hold conductance values for external stimulation in the current step
    current_stim_conductances = np.zeros(network.n_neurons)

    for step in sim_loop_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0) # Reset external stimulus for this step

        # --- Determine which neurons should be stimulated THIS STEP ---
        newly_stimulated_indices_this_step = set() # Track neurons receiving new pulse this step

        # --- Apply MNIST Input Stimulation (if provided) ---
        if mnist_input_spikes is not None:
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                stim_end_time = current_time + stim_pulse_duration_ms
                for neuron_idx in neurons_spiking_now:
                    if 0 <= neuron_idx < network.n_neurons:
                        # If not already being stimulated, start the pulse timer
                        if neuron_idx not in ongoing_stimulations:
                             ongoing_stimulations[neuron_idx] = stim_end_time
                             newly_stimulated_indices_this_step.add(neuron_idx)
                        # Mark for conductance application below
                        current_stim_conductances[neuron_idx] = stim_interval_strength
        # --- Apply Non-MNIST Stimulation (if not using MNIST input) ---
        elif not no_stimulation and stimulation_population:
            num_to_stimulate = max(1, int(len(stimulation_population) * stim_fraction))
            apply_new_stim_pulse = False
            if stochastic_stim and random.random() < (dt / 100): apply_new_stim_pulse = True
            elif stim_interval_steps and (step % stim_interval_steps == 0): apply_new_stim_pulse = True

            if apply_new_stim_pulse:
                 target_neurons_for_pulse = random.sample(stimulation_population, min(num_to_stimulate, len(stimulation_population)))
                 stim_end_time = current_time + stim_pulse_duration_ms
                 for idx in target_neurons_for_pulse:
                     if idx not in ongoing_stimulations:
                         ongoing_stimulations[idx] = stim_end_time
                         newly_stimulated_indices_this_step.add(idx)
                     # Mark for conductance application below
                     current_stim_conductances[idx] = stim_interval_strength
                 # Record stimulation event (optional)
                 # if newly_stimulated_indices_this_step:
                 #     stimulation_record['pulse_starts'].append(current_time)
                 #     stimulation_record['times'] = stimulation_record['pulse_starts']
                 #     stimulation_record['neurons'].append(list(newly_stimulated_indices_this_step))

        # --- Update Ongoing Stimulations (Turns off expired, applies conductance for active) ---
        expired_stims = set()
        for neuron_idx, end_time in ongoing_stimulations.items():
            if current_time >= end_time:
                expired_stims.add(neuron_idx) # Mark for removal
                # Ensure conductance is set to 0 if it expires this step
                current_stim_conductances[neuron_idx] = 0.0
            else:
                 # Apply conductance if pulse is still active
                 # (Handles cases where pulse started earlier and continues)
                 # Ensure index is valid before setting conductance
                 if 0 <= neuron_idx < network.n_neurons:
                      # Use the value set previously by MNIST or non-MNIST logic if pulse just started,
                      # otherwise re-apply the standard strength
                      if neuron_idx not in newly_stimulated_indices_this_step:
                           current_stim_conductances[neuron_idx] = stim_interval_strength

        # Remove expired stimulation timers
        for neuron_idx in expired_stims:
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Set External Stimulus on Network Object ---
        # Apply the calculated conductances for this step to the network's state
        # Assuming LayeredNeuronalNetworkVectorized now handles this internally via a method or direct array access
        # Option 1: Direct array access (if external_stim_g is public)
        network.external_stim_g[:] = current_stim_conductances
        # Option 2: Using a dedicated method (preferred)
        # network.set_external_stimulus(np.arange(network.n_neurons), current_stim_conductances) # Use if method exists

        # --- Update Network State (Now calls the vectorized update) ---
        active_indices = network.update_network(dt) # Returns NumPy array of spiking indices
        activity_record.append(active_indices) # Store indices

        # Detailed Neuron Data Tracking is skipped for vectorized implementation

    # --- Cleanup After Loop ---
    # Ensure final external stimulus is zero
    current_stim_conductances.fill(0.0)
    network.external_stim_g[:] = current_stim_conductances
    # network.set_external_stimulus(np.arange(network.n_neurons), current_stim_conductances) # Alt method call

    if show_progress and isinstance(sim_loop_iterator, tqdm):
         sim_loop_iterator.close()

    return activity_record, neuron_data, stimulation_record


# --- Refactored Network Creation (MODIFIED for vectorized network) ---
def create_snn_structure(n_layers_list, inhibitory_fraction, connection_probs,
                        neuron_params_dict, # Renamed for clarity
                        base_transmission_delay=1.0, dt=0.1,
                        random_seed=None, n_classes=5):
    """
    Creates the SNN structure (topology, delays) and initializes a
    LayeredNeuronalNetworkVectorized instance with appropriate parameter arrays.
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

    # --- Determine inhibitory status for all neurons ---
    is_inhibitory_array = np.random.rand(total_neurons) < inhibitory_fraction

    # --- Prepare parameters dictionary for vectorized network constructor ---
    # Filter neuron_params_dict for constructor compatibility if possible
    # (Assuming LayeredNeuronalNetworkVectorized constructor accepts **kwargs)
    network_constructor_params = neuron_params_dict.copy()
    # Add the crucial is_inhibitory array
    network_constructor_params['is_inhibitory'] = is_inhibitory_array

    # Create Vectorized Network Instance
    network = LayeredNeuronalNetworkVectorized(n_neurons=total_neurons,
                                               **network_constructor_params) # Pass params dict

    # Extract connection probabilities with defaults
    conn_prob_defaults = {
        'exc_recurrent': 0.08, 'inh_recurrent': 0.15,
        'feedforward_1': 0.45, 'feedforward_2': 0.15,
        'feedback_1': 0.06, 'feedback_2': 0.02,
        'long_feedforward': 0.01, 'long_feedback': 0.005
    }
    conn_probs_actual = {k: connection_probs.get(k, conn_prob_defaults[k]) for k in conn_prob_defaults}

    # Create Positions and Assign Layers to Graph Nodes
    pos = {} # Stores {neuron_idx: (x, y)} for visualization
    x_coords = np.linspace(0.1, 0.9, num_layers)
    horizontal_spread = 0.04
    vertical_spread = max(0.5, total_neurons / 200.0)
    layer_indices = [] # Stores [(start_idx, end_idx), ...]
    start_idx = 0

    # Add nodes to graph with layer and position info
    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num - 1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))
        for current_node_index in range(start_idx, end_idx):
             # Get inhibitory status from the pre-generated array
             is_inhib = network.is_inhibitory[current_node_index]
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                         random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[current_node_index] = node_pos
             # Add node to the graph AFTER network init
             network.graph.add_node(current_node_index, is_inhibitory=is_inhib, layer=layer_num, pos=node_pos)
             network.neuron_grid_positions[current_node_index] = node_pos # Store pos
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
        # Ensure node exists in graph before checking properties
        if i not in network.graph.nodes or i not in pos: continue
        # Get properties directly from network arrays/graph
        is_source_inhibitory = network.is_inhibitory[i]
        layer_i = network.graph.nodes[i].get('layer', -1)
        pos_i = pos[i]

        for j in range(total_neurons):
            if i == j or j not in network.graph.nodes or j not in pos: continue
            layer_j = network.graph.nodes[j].get('layer', -1)
            pos_j = pos[j]
            prob = 0.0; connect = False

            # Determine connection probability (same logic as before)
            if layer_i != -1 and layer_j != -1:
                 layer_diff = layer_j - layer_i
                 # Simplified probability check (can reuse original logic block)
                 if is_source_inhibitory:
                     if layer_diff == 0: prob = conn_probs_actual['inh_recurrent']
                     elif abs(layer_diff) == 1: prob = conn_probs_actual['feedforward_1'] * 0.2
                 else: # Excitatory connections
                     if layer_diff == 0: prob = conn_probs_actual['exc_recurrent']
                     elif layer_diff == 1: prob = conn_probs_actual['feedforward_1']
                     elif layer_diff == 2: prob = conn_probs_actual['feedforward_2']
                     elif layer_diff == -1: prob = conn_probs_actual['feedback_1']
                     elif layer_diff == -2: prob = conn_probs_actual['feedback_2']
                     elif layer_diff > 2: prob = conn_probs_actual['long_feedforward']
                     elif layer_diff < -2: prob = conn_probs_actual['long_feedback']
                 # Connect based on probability
                 if random.random() < prob: connect = True

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
def run_snn_simulation(network: LayeredNeuronalNetworkVectorized, # Type hint updated
                     duration, dt, mnist_input_spikes, stimulation_params):
    """Runs the SNN simulation for a given network (weights assumed set) and input."""
    stim_strength = stimulation_params['strength']
    stim_pulse_duration = stimulation_params.get('pulse_duration_ms', dt)
    # first_layer_indices no longer directly needed for stim application with vectorized approach

    network.reset_all() # Reset neuron states before simulation run

    # Use the modified Layered_run_unified_simulation defined above
    activity_record, _, _ = Layered_run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        mnist_input_spikes=mnist_input_spikes,
        stim_interval_strength=stim_strength, # Passed to simulation loop
        # stim_target_indices=first_layer_indices, # Less critical now
        stim_pulse_duration_ms=stim_pulse_duration, # Passed to simulation loop
        track_neurons=None, # Tracking disabled
        no_stimulation=False, # Let function override based on mnist_input_spikes
        show_progress=False # Disable internal progress bar for fitness evals
    )
    return activity_record


# --- Prediction Calculation (Unchanged, relies only on activity_record and layer_indices) ---
def calculate_prediction(activity_record, layer_indices, dt, stim_duration_ms, n_classes=5):
    """
    Calculates the predicted class index based on output layer spike counts,
    considering only spikes AFTER the stimulation period. (No changes needed here)
    """
    if not layer_indices: return -1
    output_start_idx, output_end_idx = layer_indices[-1]
    num_output_neurons = output_end_idx - output_start_idx
    if num_output_neurons != n_classes: return -1

    start_step = int(round(stim_duration_ms / dt))
    if start_step >= len(activity_record): return -1

    post_stimulus_activity = activity_record[start_step:]
    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}

    for step_spikes in post_stimulus_activity:
        # step_spikes is now potentially a NumPy array
        for neuron_idx in step_spikes:
            if output_start_idx <= neuron_idx < output_end_idx:
                output_spike_counts[neuron_idx] += 1

    if sum(output_spike_counts.values()) > 0:
        predicted_neuron_idx = max(output_spike_counts, key=output_spike_counts.get)
        predicted_label = predicted_neuron_idx - output_start_idx
    else:
        predicted_label = -1

    return predicted_label


# --- Genetic Algorithm Implementation (Largely Unchanged) ---
# Class GA structure remains the same, relies on fitness_func
class GeneticAlgorithm:
    """Implements the Genetic Algorithm logic. (No changes needed here)"""
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
        self.fitness_scores = np.full(population_size, -np.inf)

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.uniform(self.weight_min, self.weight_max, self.chromosome_length)
            population.append(chromosome)
        return population

    def evaluate_population(self, n_cores, show_progress=True):
         tasks = [(self.population[i],) + self.fitness_func_args for i in range(self.population_size)]
         # print(f"Evaluating population using {min(n_cores, self.population_size)} cores...") # Verbose
         start_eval_time = time.time()
         pool = None
         try:
             actual_processes = min(n_cores, self.population_size)
             pool = multiprocessing.Pool(processes=actual_processes)
             results_iterator = pool.starmap(self.fitness_func, tasks)
             if show_progress:
                 results_iterator = tqdm(results_iterator, total=self.population_size, desc="Evaluating Pop", ncols=80, leave=False)
             new_fitness_scores = list(results_iterator)
             if any(score is None for score in new_fitness_scores):
                  print("Warning: Received None fitness score(s). Check fitness function error handling.")
                  new_fitness_scores = [score if score is not None else -np.inf for score in new_fitness_scores]
             self.fitness_scores = np.array(new_fitness_scores)
         except Exception as e:
             print(f"FATAL Error during parallel fitness evaluation: {e}")
             self.fitness_scores = np.full(self.population_size, -np.inf)
             # raise e # Optional: re-raise to stop execution
         finally:
             if pool: pool.close(); pool.join()
         # eval_time = time.time() - start_eval_time # Verbose
         # print(f"Evaluation finished in {eval_time:.2f}s") # Verbose

    def _tournament_selection(self):
        best_idx = -1; best_fitness = -np.inf
        k = min(self.population_size, self.tournament_size)
        if k <= 0: return self.population[np.random.choice(len(self.population))] if self.population else None
        competitor_indices = np.random.choice(self.population_size, k, replace=False)
        for idx in competitor_indices:
            if idx < len(self.fitness_scores) and self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]; best_idx = idx
        if best_idx == -1 or best_idx >= len(self.population):
             best_idx = np.random.choice(competitor_indices) if competitor_indices.size > 0 else np.random.choice(len(self.population))
        return self.population[best_idx]

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate and self.chromosome_length > 1:
             crossover_point = random.randint(1, self.chromosome_length - 1)
             child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
             child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
             return child1, child2
        else: return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                mutation_val = np.random.normal(0, self.mutation_strength)
                mutated_chromosome[i] += mutation_val
                mutated_chromosome[i] = np.clip(mutated_chromosome[i], self.weight_min, self.weight_max)
        return mutated_chromosome

    def run_generation(self):
         new_population = []
         actual_elitism_count = min(self.elitism_count, self.population_size)
         if actual_elitism_count > 0 and len(self.fitness_scores) == self.population_size and np.any(np.isfinite(self.fitness_scores)):
              try:
                   elite_indices = np.argsort(self.fitness_scores)[-actual_elitism_count:]
                   for idx in elite_indices:
                       if idx < len(self.population): new_population.append(self.population[idx].copy())
              except IndexError: print("Warning: Error getting elite indices. Skipping elitism.")
         while len(new_population) < self.population_size:
             parent1 = self._tournament_selection(); parent2 = self._tournament_selection()
             if parent1 is None or parent2 is None:
                  print("Warning: Parent selection failed. Using random individuals."); break # Or handle differently
             child1, child2 = self._crossover(parent1, parent2)
             child1 = self._mutate(child1); child2 = self._mutate(child2)
             if len(new_population) < self.population_size: new_population.append(child1)
             if len(new_population) < self.population_size: new_population.append(child2)
         if len(new_population) > self.population_size: new_population = new_population[:self.population_size]
         self.population = new_population
         self.fitness_scores = np.full(self.population_size, -np.inf)


# --- Fitness Function (MODIFIED call to create_snn_structure) ---
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
    It recreates the network structure locally using the MODIFIED create_snn_structure.
    """
    # --- 1. Recreate network structure using MODIFIED function ---
    try:
        # This now returns a LayeredNeuronalNetworkVectorized instance
        eval_network, _, _, _ = create_snn_structure(
            n_layers_list=list(p_layers_config),
            inhibitory_fraction=p_inhib_frac, # Used to create is_inhibitory array inside
            connection_probs=p_conn_probs,
            neuron_params_dict=p_neuron_config, # Pass the dict of params
            base_transmission_delay=p_base_delay,
            dt=p_dt,
            random_seed=p_seed,
            n_classes=p_n_classes
        )
        eval_network.reset_all() # Reset state arrays
    except Exception as e:
        print(f"Worker Error: creating SNN structure: {e}")
        return 0.0 # Cannot evaluate if network creation fails

    # --- 2. Set weights (No change needed here, uses integrated method) ---
    try:
        eval_network.set_weights_sparse(chromosome_weights, p_connection_map)
    except ValueError as e:
        print(f"Worker Error: setting sparse weights: {e}")
        return 0.0

    # --- 3. Create SNN Stimulator locally (No change needed) ---
    mnist_stimulator = SNNStimulator(
        total_time_ms=p_mnist_stim_duration,
        max_freq_hz=p_max_freq_hz
    )

    # --- 4. Run simulation on the specified subset of filtered data (No change needed here) ---
    # Relies on run_snn_simulation, which internally calls the modified simulation loop
    accuracies = []
    if len(p_filtered_labels) == 0: return 0.0

    for idx in p_eval_indices:
        if idx >= len(p_filtered_labels): continue
        try:
            mnist_original_image = p_filtered_images[idx].reshape(28,28)
            true_original_label = p_filtered_labels[idx]
            true_mapped_label = p_label_map[true_original_label]

            if p_downsample_factor > 1:
                mnist_image = downsample_image(mnist_original_image, p_downsample_factor)
            else: mnist_image = mnist_original_image

            mnist_spike_times = mnist_stimulator.generate_spikes(mnist_image)

            # This call now uses the modified simulation loop internally
            activity_record = run_snn_simulation(
                eval_network,
                duration=p_sim_duration,
                dt=p_dt,
                mnist_input_spikes=mnist_spike_times,
                stimulation_params=p_stim_config
            )

            # Prediction calculation unchanged
            predicted_label = calculate_prediction(
                activity_record, p_layer_indices, dt=p_dt,
                stim_duration_ms=p_mnist_stim_duration, n_classes=p_n_classes
            )

            if predicted_label != -1:
                 accuracies.append(1 if predicted_label == true_mapped_label else 0)

        except KeyError: continue
        except Exception as e:
             # print(f"Worker Warning: Error during sim for index {idx}: {e}. Skipping.") # Optional verbose warning
             continue

    # --- 5. Return average accuracy as fitness ---
    fitness = np.mean(accuracies) if accuracies else 0.0
    return fitness


# --- Plotting Function for GA Progress (Unchanged) ---
def plot_ga_progress(generations, best_fitness_history, avg_fitness_history, final_test_accuracy, filename, fitness_eval_examples):
    """Plots the GA fitness history and saves the figure."""
    fig_ga, ax_ga = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    gen_axis = range(1, generations + 1)
    ax_ga.plot(gen_axis, best_fitness_history, marker='o', linestyle='-', color='cyan', markersize=4, label='Best Fitness (Eval Set)')
    ax_ga.plot(gen_axis, avg_fitness_history, marker='x', linestyle='--', color='orange', markersize=4, label='Average Fitness (Eval Set)')
    ax_ga.set_title(f'GA Fitness - Generation {generations}', color='white')
    ax_ga.set_xlabel('Generation', color='white')
    ax_ga.set_ylabel(f'Fitness (Accuracy on {fitness_eval_examples} Examples)')
    ax_ga.set_ylim(0, 1.05)
    ax_ga.grid(True, alpha=0.3)
    ax_ga.legend(loc='lower right', framealpha=0.7)
    ax_ga.set_facecolor('#1a1a1a')
    ax_ga.tick_params(colors='white')
    for spine in ax_ga.spines.values(): spine.set_color('white')
    if final_test_accuracy is not None:
        ax_ga.text(0.95, 0.1, f'Test Acc: {final_test_accuracy:.3f}', transform=ax_ga.transAxes,
                    ha='right', color='lime', bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='#1a1a1a')
    plt.close(fig_ga)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Required for multiprocessing spawning compatibility on some OS
    multiprocessing.freeze_support()

    start_overall_time = time.time() # Start overall timer

    # --- Configuration (Mostly Unchanged) ---
    TARGET_CLASSES = [0, 1] # Example: Classify digits 0 and 1
    N_CLASSES = len(TARGET_CLASSES)
    label_map_global = {original_label: new_index for new_index, original_label in enumerate(TARGET_CLASSES)}
    print(f"--- Running {N_CLASSES}-Class MNIST GA SNN (Vectorized) ---") # Updated print
    print(f"Target Digits: {TARGET_CLASSES} -> Mapped Indices: {list(range(N_CLASSES))}")

    POPULATION_SIZE = 100
    NUM_GENERATIONS = 200 # Reduced for quick testing, use more for real runs (e.g., 200)
    MUTATION_RATE = 0.10
    MUTATION_STRENGTH = 0.001
    CROSSOVER_RATE = 0.7
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 7
    FITNESS_EVAL_EXAMPLES = 100 # Reduced for quick testing (e.g., 100)
    TEST_SET_EXAMPLES = 1000

    N_CORES = os.cpu_count() - 1 if os.cpu_count() > 1 else 1 # Use almost all cores
    print(f"Using {N_CORES} cores for parallel evaluation.")

    mnist_stim_duration_global = 50  # ms
    max_freq_hz_global = 200.0       # Hz
    downsample_factor_global = 4     # 28x28 -> 7x7 = 49 inputs
    sim_duration_global = 150        # ms
    sim_dt_global = 0.1              # ms

    input_neurons = (28 // downsample_factor_global) ** 2
    layers_config_global = [input_neurons, 30, 20, N_CLASSES]
    inhib_frac_global = 0.2 # Passed to create_snn_structure

    conn_probs_global = {
        'exc_recurrent': 0.12, 'inh_recurrent': 0.15, 'feedforward_1': 0.3,
        'feedforward_2': 0.15, 'feedback_1': 0.06, 'feedback_2': 0.0,
        'long_feedforward': 0.0, 'long_feedback': 0.0
    }

    # Neuron parameters (passed as a dictionary)
    neuron_config_global = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 10.0, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0,
        'v_noise_amp': 0.0, 'i_noise_amp': 0.0, # Noise levels (keep low/zero for deterministic testing if needed)
        'adaptation_increment': 0.3, 'tau_adaptation': 120,
        # 'is_inhibitory' is NOT set here, it's determined in create_snn_structure
    }

    weight_min_ga_global = 0.002
    weight_max_ga_global = 0.067

    stim_config_global = {'strength': 25, 'pulse_duration_ms': sim_dt_global}
    master_seed = 42
    np.random.seed(master_seed)
    random.seed(master_seed)
    output_dir = "ga_mnist_snn_vectorized_output" # Changed output dir name
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # -------------------------------------

    # --- Load and Filter MNIST Data (Unchanged) ---
    print("Loading MNIST dataset...")
    try:
        mnist_loader = MNIST_loader()
        print(f"Filtering dataset for classes: {TARGET_CLASSES}...")
        mask = np.isin(mnist_loader.labels, TARGET_CLASSES)
        filtered_images_global = mnist_loader.images[mask]
        filtered_labels_global = mnist_loader.labels[mask]
        num_filtered = len(filtered_labels_global)
        print(f"Filtered dataset size: {num_filtered} examples.")
        split_idx = int(num_filtered * 0.8)
        train_eval_indices_pool = np.arange(split_idx)
        test_indices_pool = np.arange(split_idx, num_filtered)
        print(f"Using {len(train_eval_indices_pool)} examples for GA eval sampling, {len(test_indices_pool)} for final testing.")
    except Exception as e: print(f"Error loading/filtering MNIST: {e}. Exiting."); exit()

    # --- Create Base Network Structure using MODIFIED function ---
    print("Creating base SNN structure (vectorized)...")
    try:
        network_creation_args = {
            "n_layers_list": layers_config_global,
            "inhibitory_fraction": inhib_frac_global, # Will be used inside
            "connection_probs": conn_probs_global,
            "neuron_params_dict": neuron_config_global, # Pass the dict
            "base_transmission_delay": 1.0,
            "dt": sim_dt_global,
            "random_seed": master_seed,
            "n_classes": N_CLASSES
        }
        # This now returns a LayeredNeuronalNetworkVectorized instance
        base_network, layer_indices_global, _, connection_map_global = create_snn_structure(**network_creation_args)
        chromosome_len = len(connection_map_global)
        print(f"Network structure defined. Actual connections: {chromosome_len} -> Chromosome length: {chromosome_len}")
    except Exception as e: print(f"Error creating base network structure: {e}. Exiting."); exit()


    # --- Prepare Fixed Arguments Tuple for Parallel Fitness Function (Unchanged) ---
    fitness_args_tuple_base = (
        layers_config_global, inhib_frac_global, conn_probs_global,
        neuron_config_global, network_creation_args['base_transmission_delay'], sim_dt_global, master_seed,
        connection_map_global, layer_indices_global, N_CLASSES,
        filtered_images_global, filtered_labels_global, label_map_global,
        None, # Placeholder for p_eval_indices
        sim_duration_global, stim_config_global,
        downsample_factor_global, mnist_stim_duration_global, max_freq_hz_global
    )
    EVAL_INDICES_ARG_INDEX = 13

    # --- Initialize Genetic Algorithm (Unchanged) ---
    print("Initializing Genetic Algorithm...")
    try:
        ga = GeneticAlgorithm(
            population_size=POPULATION_SIZE, chromosome_length=chromosome_len,
            fitness_func=evaluate_chromosome_fitness, fitness_func_args=fitness_args_tuple_base,
            mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH,
            crossover_rate=CROSSOVER_RATE, elitism_count=ELITISM_COUNT,
            tournament_size=TOURNAMENT_SIZE, weight_min=weight_min_ga_global,
            weight_max=weight_max_ga_global
        )
        print(f"GA Initialized: Pop={POPULATION_SIZE}, Gens={NUM_GENERATIONS}")
    except Exception as e: print(f"Error initializing GA: {e}. Exiting."); exit()

    # --- Run Genetic Algorithm (Main loop unchanged) ---
    best_fitness_history = []
    avg_fitness_history = []
    print(f"\n--- Starting GA Evolution for {NUM_GENERATIONS} Generations ({N_CORES} cores) ---")
    final_test_accuracy = None

    try:
        for generation in tqdm(range(NUM_GENERATIONS), desc="Generations"):
            gen_start_time = time.time()

            eval_indices_this_gen = np.random.choice(
                train_eval_indices_pool,
                min(FITNESS_EVAL_EXAMPLES, len(train_eval_indices_pool)),
                replace=False
            )
            current_fitness_args = list(ga.fitness_func_args)
            current_fitness_args[EVAL_INDICES_ARG_INDEX] = eval_indices_this_gen
            ga.fitness_func_args = tuple(current_fitness_args)

            ga.evaluate_population(n_cores=N_CORES, show_progress=True) # Set show_progress based on preference

            if np.all(np.isneginf(ga.fitness_scores)):
                 best_gen_fitness = -np.inf; avg_gen_fitness = -np.inf
                 print(f"Generation {generation + 1}/{NUM_GENERATIONS} | All fitness evaluations failed!")
            else:
                 valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
                 best_gen_fitness = np.nanmax(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf
                 avg_gen_fitness = np.nanmean(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf

            best_fitness_history.append(best_gen_fitness)
            avg_fitness_history.append(avg_gen_fitness)
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation + 1}/{NUM_GENERATIONS} | Best Fitness: {best_gen_fitness:.4f} | Avg Fitness: {avg_gen_fitness:.4f} | Time: {gen_time:.2f}s")

            plot_filename = os.path.join(output_dir, f"ga_fitness_gen_{generation+1:03d}.png")
            plot_ga_progress(generation + 1, best_fitness_history, avg_fitness_history, final_test_accuracy, plot_filename, FITNESS_EVAL_EXAMPLES)

            if generation < NUM_GENERATIONS - 1:
                 ga.run_generation()

    except KeyboardInterrupt: print("\nGA execution interrupted by user.")
    except Exception as e: print(f"\nError during GA evolution: {e}")

    # --- End of GA Loop ---
    ga_end_time = time.time()
    print(f"\n--- GA Evolution Complete (or interrupted) ---")
    print(f"Total GA time: {ga_end_time - start_overall_time:.2f}s")

    # --- Get Best Chromosome Found (Unchanged) ---
    best_chromosome = None; best_fitness_final = -np.inf
    if len(ga.fitness_scores) > 0 and np.any(np.isfinite(ga.fitness_scores)):
         valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
         if np.any(np.isfinite(valid_scores)):
              final_best_idx = np.nanargmax(valid_scores)
              if final_best_idx < len(ga.population):
                  best_chromosome = ga.population[final_best_idx]
                  best_fitness_final = valid_scores[final_best_idx]
                  print(f"Best fitness found during evolution (on eval set): {best_fitness_final:.4f}")
                  weights_save_path = os.path.join(output_dir, f"best_snn_weights_{N_CLASSES}class_sparse.npy") # Generic name
                  map_save_path = os.path.join(output_dir, f"connection_map_{N_CLASSES}class.npy")
                  try:
                      np.save(weights_save_path, best_chromosome); np.save(map_save_path, np.array(connection_map_global, dtype=object))
                      print(f"Saved best sparse weights to {weights_save_path}"); print(f"Saved connection map to {map_save_path}")
                  except Exception as e: print(f"Error saving weights or map: {e}")
              else: print("Warning: Best fitness index is out of bounds.")
         else: print("Warning: All fitness scores were invalid. Cannot determine best chromosome.")
    else: print("Warning: No valid fitness scores available. Cannot determine best chromosome.")

    # --- Evaluate Best Chromosome on Test Set (MODIFIED call to create_snn_structure) ---
    if best_chromosome is not None:
         print(f"\n--- Evaluating best weights on {TEST_SET_EXAMPLES} filtered test examples ---")
         try:
             # Recreate the network structure (will be vectorized)
             test_network, test_layer_indices, _, test_connection_map = create_snn_structure(**network_creation_args)
             test_network.reset_all()
             test_network.set_weights_sparse(best_chromosome, test_connection_map)

             test_accuracies = []
             if len(test_indices_pool) > 0:
                 actual_test_examples = min(TEST_SET_EXAMPLES, len(test_indices_pool))
                 test_indices_final = np.random.choice(test_indices_pool, actual_test_examples, replace=False)
             else: print("Warning: No test examples available."); test_indices_final = []; actual_test_examples = 0

             mnist_stimulator_test = SNNStimulator(total_time_ms=mnist_stim_duration_global, max_freq_hz=max_freq_hz_global)
             test_loop_iterator = tqdm(test_indices_final, desc="Testing Best Weights", ncols=80)
             correct_test_predictions = 0

             for filt_idx in test_loop_iterator:
                 try:
                      mnist_original_image = filtered_images_global[filt_idx].reshape(28,28)
                      true_original_label = filtered_labels_global[filt_idx]
                      true_mapped_label = label_map_global[true_original_label]
                      if downsample_factor_global > 1: mnist_image = downsample_image(mnist_original_image, downsample_factor_global)
                      else: mnist_image = mnist_original_image
                      mnist_spike_times = mnist_stimulator_test.generate_spikes(mnist_image)

                      activity_record = run_snn_simulation(
                          test_network, sim_duration_global, sim_dt_global, mnist_spike_times, stim_config_global
                      )

                      predicted_label = calculate_prediction(
                          activity_record, test_layer_indices, dt=sim_dt_global,
                          stim_duration_ms=mnist_stim_duration_global, n_classes=N_CLASSES
                      )

                      if predicted_label != -1:
                          is_correct = (predicted_label == true_mapped_label)
                          if is_correct: correct_test_predictions += 1
                 except Exception as e: print(f"\nWarning: Error during test sim for index {filt_idx}: {e}. Skipping."); continue

             final_test_accuracy = correct_test_predictions / actual_test_examples if actual_test_examples > 0 else 0.0
             print(f"\nFinal Accuracy on Filtered Test Set ({actual_test_examples} examples): {final_test_accuracy:.4f}")

         except Exception as e: print(f"\nError during final test evaluation: {e}"); final_test_accuracy = None
    else: print("\nSkipping final test evaluation as no best chromosome was found."); final_test_accuracy = None

    overall_end_time = time.time()
    print(f"Total script execution time: {overall_end_time - start_overall_time:.2f}s")

    # --- Plot Final GA Fitness History (Unchanged) ---
    if best_fitness_history:
         print("\nPlotting final GA fitness history...")
         final_plot_filename = os.path.join(output_dir, f"mnist_snn_ga_fitness_plot_final_{N_CLASSES}class_sparse.png")
         plot_ga_progress(len(best_fitness_history), best_fitness_history, avg_fitness_history, final_test_accuracy, final_plot_filename, FITNESS_EVAL_EXAMPLES)
         print(f"Saved final GA fitness plot to {final_plot_filename}")

    print("--- Script Finished ---")
    # plt.close('all') # Optional: Close figures if running interactively
