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
import json # For saving config

# --- Import custom modules ---
# Assume these are available in the same directory or PYTHONPATH

from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized
from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal # Keep for param inspection
from MNIST_utils.MNIST_stimulation_encodings import MNIST_loader,SNNStimulator, downsample_image


# --- Structure Creation Function (Only called ONCE now) ---
def create_snn_structure(n_layers_list, inhibitory_fraction, connection_probs,
                        neuron_params_dict, # Renamed for clarity
                        base_transmission_delay=1.0, dt=0.1,
                        random_seed=None, n_classes=5):
    """
    Creates the SNN structure (topology, delays) and initializes a
    LayeredNeuronalNetworkVectorized instance with appropriate parameter arrays.
    Weights are initially 0. Returns network, layer indices, positions, and connection_map.
    MODIFIED: This function is now intended to be called only ONCE.
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

    # --- Determine inhibitory status for all neurons (FIXED) ---
    is_inhibitory_array = np.random.rand(total_neurons) < inhibitory_fraction

    # --- Prepare parameters dictionary for vectorized network constructor ---
    network_constructor_params = neuron_params_dict.copy()
    network_constructor_params['is_inhibitory'] = is_inhibitory_array # Add the fixed array

    # Create Vectorized Network Instance (This holds the final structure)
    network = LayeredNeuronalNetworkVectorized(n_neurons=total_neurons,
                                               **network_constructor_params)

    # Extract connection probabilities with defaults
    conn_prob_defaults = {
        'exc_recurrent': 0.08, 'inh_recurrent': 0.15,
        'feedforward_1': 0.45, 'feedforward_2': 0.15,
        'feedback_1': 0.06, 'feedback_2': 0.0,
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
             # Use the fixed is_inhibitory status
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

    # Add Connections based on Probability and Build Connection Map (FIXED TOPOLOGY)
    connection_count = 0
    connection_map = [] # Stores [(u, v), ...] for existing connections
    for i in range(total_neurons):
        if i not in network.graph.nodes or i not in pos: continue
        # Use fixed inhibitory status and layer info
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
                 if is_source_inhibitory:
                     if layer_diff == 0: prob = conn_probs_actual['inh_recurrent']
                     elif abs(layer_diff) == 1: prob = conn_probs_actual['feedforward_1'] * 0.2 # Example Inhibitory FF
                 else: # Excitatory connections
                     if layer_diff == 0: prob = conn_probs_actual['exc_recurrent']
                     elif layer_diff == 1: prob = conn_probs_actual['feedforward_1']
                     elif layer_diff == 2: prob = conn_probs_actual['feedforward_2']
                     elif layer_diff == -1: prob = conn_probs_actual['feedback_1']
                     elif layer_diff == -2: prob = conn_probs_actual['feedback_2']
                     elif layer_diff > 2: prob = conn_probs_actual['long_feedforward']
                     elif layer_diff < -2: prob = conn_probs_actual['long_feedback']
                 # Connect based on probability (THIS DETERMINES THE FIXED TOPOLOGY)
                 if random.random() < prob: connect = True

            if connect:
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist)) if max_possible_dist > 1e-9 else base_transmission_delay * 0.5
                delay = max(min_delay, delay) # Enforce minimum delay
                # Add connection with weight 0 AND THE FIXED DELAY
                # This populates the network.weights and network.delays matrices
                network.add_connection(i, j, weight=0.0, delay=delay)
                connection_map.append((i, j)) # Record that this connection exists
                connection_count += 1

    print(f"Fixed SNN Structure Created: {total_neurons} neurons, {connection_count} potential connections.")
    return network, layer_indices, pos, connection_map


# --- Simulation Function (Unchanged from previous modification) ---
def Layered_run_unified_simulation(network: LayeredNeuronalNetworkVectorized, # Type hint updated
                                 duration=1000.0, dt=0.1,
                                 stim_interval=None, stim_interval_strength=10,
                                 stim_fraction=0.01, stim_target_indices=None,
                                 stim_pulse_duration_ms=1.0,
                                 mnist_input_spikes=None,
                                 track_neurons=None, # Note: Tracking detailed neuron data needs rework for vectorized state
                                 stochastic_stim=False,
                                 no_stimulation=False, show_progress=False):
    """
    Runs the network simulation handling various stimulation types and tracking.
    Uses vectorized network update and stimulation.
    NOTE: Detailed neuron data tracking (neuron_data) is DISABLED for vectorized network.
    """
    mnist_spikes_by_step = {}
    if mnist_input_spikes is not None:
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

    n_steps = int(duration / dt)
    activity_record = []
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms, 'times': []}
    neuron_data = {} # Detailed tracking disabled

    ongoing_stimulations = {}
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None
    stimulation_population = set(stim_target_indices) if stim_target_indices is not None else set(range(network.n_neurons))
    if not stimulation_population: no_stimulation = True

    sim_loop_iterator = range(n_steps)
    if show_progress:
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step", leave=False, ncols=80)

    current_stim_conductances = np.zeros(network.n_neurons)

    for step in sim_loop_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0)

        newly_stimulated_indices_this_step = set()

        if mnist_input_spikes is not None:
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                stim_end_time = current_time + stim_pulse_duration_ms
                for neuron_idx in neurons_spiking_now:
                    if 0 <= neuron_idx < network.n_neurons:
                        if neuron_idx not in ongoing_stimulations:
                             ongoing_stimulations[neuron_idx] = stim_end_time
                             newly_stimulated_indices_this_step.add(neuron_idx)
                        current_stim_conductances[neuron_idx] = stim_interval_strength
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
                     current_stim_conductances[idx] = stim_interval_strength

        expired_stims = set()
        for neuron_idx, end_time in ongoing_stimulations.items():
            if current_time >= end_time:
                expired_stims.add(neuron_idx)
                current_stim_conductances[neuron_idx] = 0.0
            else:
                 if 0 <= neuron_idx < network.n_neurons:
                      if neuron_idx not in newly_stimulated_indices_this_step:
                           current_stim_conductances[neuron_idx] = stim_interval_strength

        for neuron_idx in expired_stims:
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Set External Stimulus on Network Object ---
        # This assumes network.external_stim_g is the array holding external conductances
        network.external_stim_g[:] = current_stim_conductances

        # --- Update Network State (Vectorized) ---
        active_indices = network.update_network(dt) # Returns NumPy array
        activity_record.append(active_indices) # Store indices

    # --- Cleanup After Loop ---
    network.external_stim_g.fill(0.0)

    if show_progress and isinstance(sim_loop_iterator, tqdm):
         sim_loop_iterator.close()

    return activity_record, neuron_data, stimulation_record


# --- Refactored Simulation Runner (Unchanged) ---
def run_snn_simulation(network: LayeredNeuronalNetworkVectorized, # Type hint updated
                     duration, dt, mnist_input_spikes, stimulation_params):
    """Runs the SNN simulation for a given network (weights assumed set) and input."""
    stim_strength = stimulation_params['strength']
    stim_pulse_duration = stimulation_params.get('pulse_duration_ms', dt)

    network.reset_all() # Reset neuron states before simulation run

    activity_record, _, _ = Layered_run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        mnist_input_spikes=mnist_input_spikes,
        stim_interval_strength=stim_strength,
        stim_pulse_duration_ms=stim_pulse_duration,
        track_neurons=None, # Tracking disabled
        no_stimulation=False, # Let function override based on mnist_input_spikes
        show_progress=False
    )
    return activity_record


# In genetic_experiment_vectorized.py
def calculate_prediction(activity_record, layer_indices, dt, stim_duration_ms, n_classes=5):
    """
    Calculates the predicted class index based on the TOTAL spike counts
    from output layer neurons throughout the entire simulation.
    MODIFIED: Counts all spikes, not just post-stimulus. Also returns counts dict.
    """
    # --- Input Validation ---
    if not layer_indices:
        print("Warning: Layer indices not provided.")
        return -1, {}  # Return empty dict for counts
    output_start_idx, output_end_idx = layer_indices[-1]
    num_output_neurons = output_end_idx - output_start_idx
    if num_output_neurons != n_classes:
        print(f"Warning: Output layer size ({num_output_neurons}) != n_classes ({n_classes}).")
        return -1, {} # Return empty dict for counts

    # --- Initialize Spike Counts ---
    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}

    # --- Count Spikes over ENTIRE Simulation ---
    for step_spikes in activity_record:
        for neuron_idx in step_spikes:
            if output_start_idx <= neuron_idx < output_end_idx:
                output_spike_counts[neuron_idx] += 1

    # --- Determine Prediction ---
    total_output_spikes = sum(output_spike_counts.values())
    predicted_label = -1
    if total_output_spikes > 0:
        predicted_neuron_idx = max(output_spike_counts, key=output_spike_counts.get)
        predicted_label = predicted_neuron_idx - output_start_idx
    # else: predicted_label remains -1

    # Return both the label and the counts dictionary
    return predicted_label, output_spike_counts


# --- Genetic Algorithm Implementation (Unchanged) ---
class GeneticAlgorithm:
    """Implements the Genetic Algorithm logic."""
    def __init__(self, population_size, chromosome_length, fitness_func, fitness_func_args, # Pass fixed args
                 mutation_rate=0.05, mutation_strength=0.01, crossover_rate=0.7,
                 elitism_count=2, tournament_size=5,
                 weight_min=-0.02, weight_max=0.02):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
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
        # Initialize weights with small random values
        population = []
        for _ in range(self.population_size):
             # Consider initializing closer to zero, or using a distribution
             # chromosome = np.random.uniform(self.weight_min, self.weight_max, self.chromosome_length)
             chromosome = np.random.normal(0, (self.weight_max - self.weight_min)/4 , self.chromosome_length)
             chromosome = np.clip(chromosome, self.weight_min, self.weight_max)
             population.append(chromosome)
        return population

    def evaluate_population(self, n_cores, show_progress=True):
         tasks = [(self.population[i],) + self.fitness_func_args for i in range(self.population_size)]
         start_eval_time = time.time()
         pool = None
         try:
             actual_processes = min(n_cores, self.population_size)
             # Use 'spawn' context for potentially better cross-platform compatibility
             # mp_context = multiprocessing.get_context('spawn')
             # pool = mp_context.Pool(processes=actual_processes)
             pool = multiprocessing.Pool(processes=actual_processes)

             results_iterator = pool.starmap(self.fitness_func, tasks)
             if show_progress:
                 results_iterator = tqdm(results_iterator, total=self.population_size, desc="Evaluating Pop", ncols=80, leave=False)

             new_fitness_scores = list(results_iterator)

             if any(score is None for score in new_fitness_scores):
                  print("\nWarning: Received None fitness score(s). Check fitness function.")
                  new_fitness_scores = [score if score is not None else -np.inf for score in new_fitness_scores]

             self.fitness_scores = np.array(new_fitness_scores)

         except Exception as e:
             print(f"\nFATAL Error during parallel fitness evaluation: {e}")
             # Consider how to handle this - stop? set all fitness to -inf?
             self.fitness_scores = np.full(self.population_size, -np.inf)
             # traceback.print_exc() # Print traceback for debugging
             # raise e # Optional: re-raise to stop execution
         finally:
             if pool: pool.close(); pool.join()
         # eval_time = time.time() - start_eval_time # Verbose
         # print(f"Eval finished: {eval_time:.2f}s") # Verbose

    def _tournament_selection(self):
        best_idx = -1; best_fitness = -np.inf
        k = min(self.population_size, self.tournament_size)
        if k <= 0: return self.population[np.random.choice(len(self.population))] if self.population else None
        competitor_indices = np.random.choice(self.population_size, k, replace=False)
        for idx in competitor_indices:
            if idx < len(self.fitness_scores) and self.fitness_scores[idx] > best_fitness:
                best_fitness = self.fitness_scores[idx]; best_idx = idx
        # Handle cases where all selected competitors have -inf fitness
        if best_idx == -1:
             best_idx = np.random.choice(competitor_indices) if competitor_indices.size > 0 else np.random.choice(len(self.population))
        # Ensure index is valid before returning
        if best_idx >= len(self.population):
             best_idx = np.random.choice(len(self.population)) # Fallback to random choice

        return self.population[best_idx]

    def _crossover(self, parent1, parent2):
        # Uniform crossover might explore the space better for weights
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < self.crossover_rate:
            for i in range(self.chromosome_length):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            return child1, child2
        else:
            return child1, child2 # Return copies even if no crossover

    def _mutate(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(self.chromosome_length):
            if random.random() < self.mutation_rate:
                 # Additive Gaussian mutation
                 mutation_val = np.random.normal(0, self.mutation_strength)
                 mutated_chromosome[i] += mutation_val
                 # Ensure weights stay within bounds
                 mutated_chromosome[i] = np.clip(mutated_chromosome[i], self.weight_min, self.weight_max)
        return mutated_chromosome

    def run_generation(self):
         new_population = []
         actual_elitism_count = min(self.elitism_count, self.population_size)

         # Elitism: Copy best individuals directly
         if actual_elitism_count > 0 and len(self.fitness_scores) == self.population_size and np.any(np.isfinite(self.fitness_scores)):
              try:
                   # Use nanargsort if NaNs are possible, otherwise argsort
                   valid_scores = np.where(np.isneginf(self.fitness_scores), np.nan, self.fitness_scores)
                   if np.any(np.isfinite(valid_scores)):
                        elite_indices = np.argsort(self.fitness_scores)[-actual_elitism_count:] # Indices of highest fitness
                        for idx in elite_indices:
                            if idx < len(self.population):
                                 new_population.append(self.population[idx].copy())
                   else: print("Warning: No finite scores for elitism.")
              except IndexError: print("Warning: Error getting elite indices. Skipping elitism.")

         # Generate remaining individuals through selection, crossover, mutation
         while len(new_population) < self.population_size:
             parent1 = self._tournament_selection(); parent2 = self._tournament_selection()
             if parent1 is None or parent2 is None:
                  print("Warning: Parent selection failed. Breaking generation loop."); break
             child1, child2 = self._crossover(parent1, parent2)
             child1 = self._mutate(child1); child2 = self._mutate(child2)
             if len(new_population) < self.population_size: new_population.append(child1)
             if len(new_population) < self.population_size: new_population.append(child2)

         # Handle population size discrepancies (e.g., if odd number needed)
         if len(new_population) > self.population_size:
              new_population = new_population[:self.population_size]
         elif len(new_population) < self.population_size:
              print(f"Warning: New population size ({len(new_population)}) < target ({self.population_size}). This might indicate issues.")
              # Optionally fill remaining spots? (e.g., with random individuals or copies)
              while len(new_population) < self.population_size:
                   new_population.append(self._initialize_population()[0]) # Add a random new individual

         self.population = new_population
         self.fitness_scores = np.full(self.population_size, -np.inf) # Reset fitness for next gen


# In genetic_experiment_vectorized.py

# --- Assume calculate_prediction is modified to return: (predicted_label, output_spike_counts) ---
# def calculate_prediction(activity_record, layer_indices, dt, stim_duration_ms, n_classes=5):
#    ... # Modified logic from previous turn
#    return predicted_label, output_spike_counts


# --- MODIFIED Fitness Function with Correct Formatting ---
def evaluate_chromosome_fitness(chromosome_weights,       # The weights to evaluate
                                # Network Structure Params (FIXED - Passed from main)
                                p_total_neurons, p_neuron_config, p_layer_indices,
                                p_fixed_inhib_array, p_fixed_delays_matrix, p_connection_map,
                                p_n_classes, # Pass number of classes
                                # Data Params
                                p_filtered_images, p_filtered_labels, p_label_map,
                                p_eval_indices, # Specific indices for this evaluation run
                                p_precomputed_spikes, # *** ADDED: Parameter to accept the argument ***
                                # Simulation Params
                                p_sim_duration, p_dt, p_stim_config,
                                p_downsample_factor, p_mnist_stim_duration, p_max_freq_hz):
    """
    Fitness function designed for parallel execution.
    MODIFIED: Creates a minimal network instance, populates its graph, and applies weights.
              (Accepts precomputed spikes argument, but internal logic may still generate them).
    """
    # --- 1. Create a MINIMAL network instance for evaluation ---
    try:
        # Pass essential parameters, including the fixed inhibitory array
        eval_network = LayeredNeuronalNetworkVectorized(
            n_neurons=p_total_neurons,
            is_inhibitory=p_fixed_inhib_array, # CRITICAL: Use fixed array
            **p_neuron_config # Pass other neuron parameters (taus, thresholds, etc.)
        )
        # Apply the FIXED delays
        eval_network.delays = p_fixed_delays_matrix.copy() # Use copy for safety

        eval_network.reset_all() # Reset state arrays (v, g_e, g_i, etc.)

        # --- 2. Set weights (using the fixed connection map) ---
        # This sets the weight matrix but NOT the graph edges yet
        eval_network.set_weights_sparse(chromosome_weights, p_connection_map)

        # --- 3. *** POPULATE GRAPH FOR EVAL_NETWORK *** ---
        # print("Populating graph for eval_network...") # Optional debug print
        # Add nodes first (optional but good practice)
        for i in range(p_total_neurons):
            eval_network.graph.add_node(i, is_inhibitory=p_fixed_inhib_array[i])

        # Add edges based on the connection map
        for u, v in p_connection_map:
            if u < p_total_neurons and v < p_total_neurons:
                weight = eval_network.weights[u, v] # Get the weight just set
                delay = eval_network.delays[u, v]   # Get the delay already set
                eval_network.graph.add_edge(u, v, weight=weight, delay=delay)
            # else: print(f"Warning: Invalid indices ({u}, {v}) in p_connection_map skipped.") # Optional debug
        # print("Graph populated for eval_network.") # Optional debug print
        # --- *** END OF GRAPH POPULATION *** ---

    except ValueError as e: # Catch specific error from set_weights_sparse if needed
        # print(f"Worker Error: setting sparse weights: {e}") # Suppress print
        return -np.inf
    except Exception as e: # Catch other errors during network creation/population
        # print(f"Worker Error: creating/populating eval SNN instance: {e}") # Suppress print
        return -np.inf # Use -inf to signal failure clearly


    # --- 4. Create SNN Stimulator locally ---
    # NOTE: This part still generates spikes on the fly, ignoring p_precomputed_spikes
    # To use precomputed spikes, this block and the call inside the loop should be removed,
    # and mnist_spike_times should be retrieved from p_precomputed_spikes.
    try:
        mnist_stimulator = SNNStimulator(
            total_time_ms=p_mnist_stim_duration,
            max_freq_hz=p_max_freq_hz
        )
    except Exception as e:
         # print(f"Worker Error: creating SNNStimulator: {e}") # Suppress print
         return -np.inf

    # --- 5. Run simulation on the specified subset of filtered data ---
    accuracies = []
    # print_limit = 5 # Limit debug prints to first few examples per worker
    # print_count = 0

    # Check if evaluation indices are valid for the filtered data
    if not hasattr(p_eval_indices, '__len__') or len(p_eval_indices) == 0:
         # print("Worker Warning: No evaluation indices provided.") # Suppress print
         return 0.0
    if len(p_filtered_labels) == 0:
         # print("Worker Warning: Filtered labels list is empty.") # Suppress print
         return 0.0

    valid_eval_indices = [idx for idx in p_eval_indices if 0 <= idx < len(p_filtered_labels)]
    if not valid_eval_indices:
         # print("Worker Warning: No valid evaluation indices found.") # Suppress print
         return 0.0

    for idx in valid_eval_indices: # Iterate only through valid indices
        try:
            # print_count += 1
            # do_print = print_count <= print_limit # Control printing

            mnist_original_image = p_filtered_images[idx].reshape(28,28)
            true_original_label = p_filtered_labels[idx]

            # Check label map validity
            if true_original_label not in p_label_map:
                 # print(f"Worker Warning: Label {true_original_label} not in label_map. Skipping index {idx}.") # Optional verbose
                 continue
            true_mapped_label = p_label_map[true_original_label]

            # if do_print:
            #     print(f"\n--- [Worker] Evaluating Example Index (filtered): {idx} ---")
            #     print(f"True Original Label: {true_original_label} -> Mapped Target Label: {true_mapped_label}")

            if p_downsample_factor > 1:
                mnist_image = downsample_image(mnist_original_image, p_downsample_factor)
            else: mnist_image = mnist_original_image

            # *** NOTE: Still generating spikes here, not using p_precomputed_spikes ***
            mnist_spike_times = mnist_stimulator.generate_spikes(mnist_image)

            # Run simulation using the network with fixed structure & current weights
            activity_record = run_snn_simulation(
                eval_network, # Use the eval_network with populated graph
                duration=p_sim_duration,
                dt=p_dt,
                mnist_input_spikes=mnist_spike_times,
                stimulation_params=p_stim_config
            )

            # Get prediction AND counts (ensure calculate_prediction returns both)
            predicted_label, output_counts = calculate_prediction(
                activity_record, p_layer_indices, dt=p_dt,
                stim_duration_ms=p_mnist_stim_duration, n_classes=p_n_classes
            )

            # if do_print:
            #     print(f"Output Layer Spike Counts: {output_counts}")
            #     print(f"Predicted Label: {predicted_label}")

            # Check if prediction is valid (not -1) before calculating accuracy
            if predicted_label != -1:
                 is_correct = (predicted_label == true_mapped_label)
                 accuracies.append(1 if is_correct else 0)
                 # if do_print: print(f"Comparison: Correct? {is_correct}")
            # If prediction is -1 (e.g., no output spikes or tie if handled that way),
            # accuracy is implicitly 0 as nothing is appended as 1.
            # elif do_print:
                 # print("Comparison: No prediction made or tie occurred (predicted_label == -1)")

            # if do_print: print(f"--- [Worker] End Example {idx} ---\n")

        except KeyError as e:
             # print(f"Worker Warning: KeyError for index {idx} (likely label map): {e}. Skipping.") # Optional verbose
             continue
        except IndexError as e:
             # print(f"Worker Warning: IndexError for index {idx} (likely image/label access): {e}. Skipping.") # Suppress print
             continue
        except Exception as e:
             # print(f"Worker Error: Unhandled exception during sim for index {idx}: {e}. Skipping.") # Suppress print
             # Consider logging traceback here for debugging difficult errors
             # import traceback
             # traceback.print_exc()
             continue

    # --- 6. Return average accuracy as fitness ---
    # print(f"\n--- [Worker] Final Accuracies List for Chromosome: {accuracies} ---") # Print the whole list
    fitness = np.mean(accuracies) if accuracies else 0.0
    # Ensure fitness is not NaN
    if np.isnan(fitness):
        # print("Worker Warning: Calculated fitness is NaN. Returning 0.0.") # Suppress print
        fitness = 0.0

    # print(f"--- [Worker] Calculated Fitness: {fitness:.4f} ---") # Suppress print
    return fitness

# --- Plotting Function for GA Progress (Unchanged) ---
def plot_ga_progress(generations, best_fitness_history, avg_fitness_history, final_test_accuracy, filename, fitness_eval_examples):
    """Plots the GA fitness history and saves the figure."""
    fig_ga, ax_ga = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')
    gen_axis = range(1, generations + 1)
    # Filter out -inf before plotting
    best_plot = [f if np.isfinite(f) else np.nan for f in best_fitness_history]
    avg_plot = [f if np.isfinite(f) else np.nan for f in avg_fitness_history]

    ax_ga.plot(gen_axis, best_plot, marker='o', linestyle='-', color='cyan', markersize=4, label='Best Fitness (Eval Set)')
    ax_ga.plot(gen_axis, avg_plot, marker='x', linestyle='--', color='orange', markersize=4, label='Average Fitness (Eval Set)')
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
    try:
        plt.savefig(filename, dpi=150, facecolor='#1a1a1a')
    except Exception as e:
        print(f"Error saving GA progress plot {filename}: {e}")
    plt.close(fig_ga)


# --- Main Execution Block (MODIFIED FOR PRECOMPUTATION) ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # For compatibility

    start_overall_time = time.time()

    # --- Configuration (Mostly Unchanged) ---
    TARGET_CLASSES = [0, 1, 2, 3] # Example: Classify digits 0, 1, 2
    N_CLASSES = len(TARGET_CLASSES)
    label_map_global = {original_label: new_index for new_index, original_label in enumerate(TARGET_CLASSES)}
    print(f"--- Running {N_CLASSES}-Class MNIST GA SNN (Vectorized - Fixed Structure, Precomputed Spikes) ---") # Updated print
    print(f"Target Digits: {TARGET_CLASSES} -> Mapped Indices: {list(range(N_CLASSES))}")

    POPULATION_SIZE = 100
    NUM_GENERATIONS = 5
    MUTATION_RATE = 0.01
    MUTATION_STRENGTH = 0.001
    CROSSOVER_RATE = 0.7
    ELITISM_COUNT = 2
    TOURNAMENT_SIZE = 5
    FITNESS_EVAL_EXAMPLES = 100 # Number of examples per fitness evaluation
    TEST_SET_EXAMPLES = 1000 # Number of examples for final test
    N_CORES = max(1, os.cpu_count() - 1 if os.cpu_count() else 1) # Use almost all cores
    print(f"Using {N_CORES} cores for parallel evaluation.")

    mnist_stim_duration_global = 50  # ms
    max_freq_hz_global = 200.0       # Hz
    downsample_factor_global = 4     # 28x28 -> 7x7 = 49 inputs
    sim_duration_global = 100        # ms
    sim_dt_global = 0.1              # ms

    input_neurons = (28 // downsample_factor_global) ** 2
    layers_config_global = [input_neurons, 30, 20,  N_CLASSES]
    inhib_frac_global = 0.2

    conn_probs_global = { # Keep using probabilities for initial structure generation
        'exc_recurrent': 0.12, 'inh_recurrent': 0.15, 'feedforward_1': 0.3,
        'feedforward_2': 0.15, 'feedback_1': 0.06, 'feedback_2': 0.0,
        'long_feedforward': 0.0, 'long_feedback': 0.0
    }

    neuron_config_global = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 10.0, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0,
        'v_noise_amp': 0.0, 'i_noise_amp': 0.005,
        'adaptation_increment': 0.3, 'tau_adaptation': 120,
        # 'is_inhibitory' will be generated in create_snn_structure
    }

    weight_min_ga_global = 0.002 # Allow inhibitory weights directly
    weight_max_ga_global = 0.4

    stim_config_global = {'strength': 25, 'pulse_duration_ms': sim_dt_global}
    master_seed = 42
    np.random.seed(master_seed)
    random.seed(master_seed)
    output_dir = "ga_mnist_snn_vectorized_precomputed_output" # Changed output dir name
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
        # Simple 80/20 split for GA eval / final test
        split_idx = int(num_filtered * 0.8)
        if split_idx == 0 or split_idx == num_filtered:
             raise ValueError("Could not create a valid train/test split. Check filtered data.")
        train_eval_indices_pool = np.arange(split_idx)
        test_indices_pool = np.arange(split_idx, num_filtered)
        print(f"Using {len(train_eval_indices_pool)} examples for GA eval sampling, {len(test_indices_pool)} for final testing.")
        if len(train_eval_indices_pool) < FITNESS_EVAL_EXAMPLES:
             print(f"Warning: Pool for GA eval ({len(train_eval_indices_pool)}) is smaller than requested eval size ({FITNESS_EVAL_EXAMPLES}). Using pool size.")
        if len(test_indices_pool) < TEST_SET_EXAMPLES:
             print(f"Warning: Pool for final test ({len(test_indices_pool)}) is smaller than requested test size ({TEST_SET_EXAMPLES}). Using pool size.")

    except Exception as e: print(f"Error loading/filtering MNIST: {e}. Exiting."); exit()

    # --- Create FIXED Base Network Structure ONCE ---
    print("Creating FIXED base SNN structure...")
    try:
        network_creation_args = {
            "n_layers_list": layers_config_global,
            "inhibitory_fraction": inhib_frac_global,
            "connection_probs": conn_probs_global,
            "neuron_params_dict": neuron_config_global,
            "base_transmission_delay": 1.0, # Example value
            "dt": sim_dt_global,
            "random_seed": master_seed,
            "n_classes": N_CLASSES
        }
        # Call structure creation ONCE
        base_network, layer_indices_global, pos_global, connection_map_global = \
            create_snn_structure(**network_creation_args)

        # Extract FIXED structural info
        fixed_inhib_array_global = base_network.is_inhibitory.copy()
        fixed_delays_matrix_global = base_network.delays.copy()
        chromosome_len_global = len(connection_map_global)
        total_neurons_global = base_network.n_neurons

        # Sanity check
        if fixed_inhib_array_global.shape[0] != total_neurons_global:
             raise ValueError("Inhibitory array size mismatch")
        if fixed_delays_matrix_global.shape != (total_neurons_global, total_neurons_global):
             raise ValueError("Delays matrix size mismatch")
        if chromosome_len_global == 0:
             print("Warning: No connections were created in the fixed structure. Check connection probabilities.")

        print(f"FIXED Network structure defined. Neurons: {total_neurons_global}, Connections: {chromosome_len_global}")

    except Exception as e: print(f"Error creating base network structure: {e}. Exiting."); exit()

    # +++ START PRECOMPUTATION +++
    print("\n--- Precomputing MNIST Spike Trains ---")
    precomputed_spike_trains_global = {} # Dictionary to store {index: spike_trains}
    mnist_stimulator_precompute = SNNStimulator(
        total_time_ms=mnist_stim_duration_global,
        max_freq_hz=max_freq_hz_global
    )
    # Precompute for all filtered examples (both potential eval and test)
    indices_to_precompute = np.arange(num_filtered)
    precompute_start_time = time.time()
    for idx in tqdm(indices_to_precompute, desc="Precomputing Spikes", ncols=80):
        try:
            mnist_original_image = filtered_images_global[idx].reshape(28,28)
            if downsample_factor_global > 1:
                mnist_image = downsample_image(mnist_original_image, downsample_factor_global)
            else:
                mnist_image = mnist_original_image
            # Generate and store
            precomputed_spike_trains_global[idx] = mnist_stimulator_precompute.generate_spikes(mnist_image)
        except Exception as e:
            print(f"\nWarning: Error precomputing spikes for index {idx}: {e}. Skipping.")
            precomputed_spike_trains_global[idx] = None # Mark as failed

    precompute_end_time = time.time()
    print(f"Finished precomputing {len(precomputed_spike_trains_global)} spike trains in {precompute_end_time - precompute_start_time:.2f}s.")
    # +++ END PRECOMPUTATION +++

    # --- Prepare Fixed Arguments Tuple for Parallel Fitness Function (MODIFIED) ---
    # Add the precomputed spikes dictionary
    fitness_args_tuple_base = (
        # Network Structure Params (FIXED)
        total_neurons_global, neuron_config_global, layer_indices_global,
        fixed_inhib_array_global, fixed_delays_matrix_global, connection_map_global,
        N_CLASSES,
        # Data Params
        filtered_images_global, filtered_labels_global, label_map_global,
        None, # Placeholder for p_eval_indices
        precomputed_spike_trains_global, # *** ADDED PRECOMPUTED SPIKES ***
        # Simulation Params
        sim_duration_global, sim_dt_global, stim_config_global,
        downsample_factor_global, mnist_stim_duration_global, max_freq_hz_global
    )
    # Update the index for eval_indices placeholder
    EVAL_INDICES_ARG_INDEX = 10
    PRECOMPUTED_SPIKES_ARG_INDEX = 11 # New index for the spikes dict

    # --- Initialize Genetic Algorithm (Unchanged, uses correct chromosome length) ---
    print("Initializing Genetic Algorithm...")
    try:
        ga = GeneticAlgorithm(
            population_size=POPULATION_SIZE, chromosome_length=chromosome_len_global,
            fitness_func=evaluate_chromosome_fitness, # Use the modified fitness func
            fitness_func_args=fitness_args_tuple_base, # Pass the tuple with fixed structure info
            mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH,
            crossover_rate=CROSSOVER_RATE, elitism_count=ELITISM_COUNT,
            tournament_size=TOURNAMENT_SIZE,
            weight_min=weight_min_ga_global, weight_max=weight_max_ga_global
        )
        print(f"GA Initialized: Pop={POPULATION_SIZE}, Gens={NUM_GENERATIONS}, Chromosome Length={chromosome_len_global}")
    except Exception as e: print(f"Error initializing GA: {e}. Exiting."); exit()

    # --- Run Genetic Algorithm (Main loop unchanged) ---
    best_fitness_history = []
    avg_fitness_history = []
    print(f"\n--- Starting GA Evolution for {NUM_GENERATIONS} Generations ({N_CORES} cores, FIXED Structure) ---")
    final_test_accuracy = None # Initialize final test accuracy

    try:
        for generation in range(NUM_GENERATIONS):
            gen_start_time = time.time()
            print(f"\n--- Generation {generation + 1}/{NUM_GENERATIONS} ---")

            # Select subset of training examples for fitness evaluation this generation
            eval_indices_this_gen = np.random.choice(
                train_eval_indices_pool,
                min(FITNESS_EVAL_EXAMPLES, len(train_eval_indices_pool)), # Use min to handle small pools
                replace=False # Sample without replacement
            )
            # Update the evaluation indices in the arguments tuple
            current_fitness_args_list = list(ga.fitness_func_args)
            current_fitness_args_list[EVAL_INDICES_ARG_INDEX] = eval_indices_this_gen
            ga.fitness_func_args = tuple(current_fitness_args_list)

            # Evaluate population in parallel
            ga.evaluate_population(n_cores=N_CORES, show_progress=True) # Control progress bar display

            # Process fitness scores (handle potential -inf)
            if np.all(np.isneginf(ga.fitness_scores)):
                 best_gen_fitness = -np.inf; avg_gen_fitness = -np.inf
                 print(f"Generation {generation + 1} | All fitness evaluations failed!")
            else:
                 # Use nanmax/nanmean to ignore -inf scores safely
                 valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
                 best_gen_fitness = np.nanmax(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf
                 avg_gen_fitness = np.nanmean(valid_scores) if np.any(np.isfinite(valid_scores)) else -np.inf

            best_fitness_history.append(best_gen_fitness)
            avg_fitness_history.append(avg_gen_fitness)
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation + 1} | Best Fitness: {best_gen_fitness:.4f} | Avg Fitness: {avg_gen_fitness:.4f} | Time: {gen_time:.2f}s")

            # Plot progress
            plot_filename = os.path.join(output_dir, f"ga_fitness_gen_{generation+1:03d}.png")
            plot_ga_progress(generation + 1, best_fitness_history, avg_fitness_history, final_test_accuracy, plot_filename, FITNESS_EVAL_EXAMPLES)

            # Evolve to next generation (unless it's the last one)
            if generation < NUM_GENERATIONS - 1:
                 ga.run_generation()

    except KeyboardInterrupt: print("\nGA execution interrupted by user.")
    except Exception as e: print(f"\nError during GA evolution: {e}") # import traceback; traceback.print_exc() # Uncomment for detailed traceback

    # --- End of GA Loop ---
    ga_end_time = time.time()
    print(f"\n--- GA Evolution Complete (or interrupted) ---")
    print(f"Total GA time: {ga_end_time - start_overall_time:.2f}s")

   # --- Get Best Chromosome Found ---
    best_chromosome = None
    best_fitness_final = -np.inf
    if len(ga.fitness_scores) > 0 and np.any(np.isfinite(ga.fitness_scores)):
         valid_scores = np.where(np.isneginf(ga.fitness_scores), np.nan, ga.fitness_scores)
         if np.any(np.isfinite(valid_scores)): # Check if there are any valid scores
              final_best_idx = np.nanargmax(valid_scores)
              if final_best_idx < len(ga.population):
                  best_chromosome = ga.population[final_best_idx]
                  best_fitness_final = valid_scores[final_best_idx]
                  print(f"\nBest fitness found during evolution (on eval set): {best_fitness_final:.4f} at index {final_best_idx}")

                  # --- SAVE ALL PERTINENT NETWORK INFORMATION ---
                  print(f"\n--- Saving Best Network State to Directory: {output_dir} ---")
                  base_filename = f"best_snn_{N_CLASSES}class_fixed_structure_precomputed" # Updated filename
                  weights_save_path = os.path.join(output_dir, f"{base_filename}_weights.npy")
                  map_save_path = os.path.join(output_dir, f"{base_filename}_connection_map.npy")
                  delays_save_path = os.path.join(output_dir, f"{base_filename}_delays_matrix.npy") # Save full matrix
                  inhib_save_path = os.path.join(output_dir, f"{base_filename}_inhibitory_array.npy") # Save full array
                  config_save_path = os.path.join(output_dir, f"{base_filename}_config.json")
                  pos_save_path = os.path.join(output_dir, f"{base_filename}_positions.npy") # File path for positions
                  # Optionally save precomputed spikes if needed for reloading later
                  # spikes_save_path = os.path.join(output_dir, f"{base_filename}_precomputed_spikes.npy")

                  try:
                      # 1. Save Best Weights (Chromosome)
                      np.save(weights_save_path, best_chromosome)
                      print(f"Saved best weights vector to {weights_save_path}")

                      # 2. Save Connection Map (List of tuples)
                      np.save(map_save_path, np.array(connection_map_global, dtype=object))
                      print(f"Saved connection map to {map_save_path}")

                      # 3. Save the FIXED Delays Matrix
                      np.save(delays_save_path, fixed_delays_matrix_global)
                      print(f"Saved fixed delays matrix to {delays_save_path}")

                      # 4. Save the FIXED Inhibitory Status Array
                      np.save(inhib_save_path, fixed_inhib_array_global)
                      print(f"Saved fixed inhibitory status array to {inhib_save_path}")

                      # 5. Save the FIXED Positions Dictionary
                      if 'pos_global' in locals() and isinstance(pos_global, dict):
                           np.save(pos_save_path, pos_global)
                           print(f"Saved fixed positions dictionary to {pos_save_path}")
                      else:
                           print("Error: 'pos_global' dictionary not found. Cannot save positions.")

                     # 6. Save Key Configuration Parameters
                      base_delay_used = network_creation_args.get('base_transmission_delay', 1.0) if 'network_creation_args' in locals() else 1.0
                      config_to_save = {
                          "layers_config": layers_config_global,
                          "inhibitory_fraction_used": inhib_frac_global,
                          "neuron_config": neuron_config_global,
                          "connection_probabilities_used": conn_probs_global,
                          "base_transmission_delay_used": base_delay_used,
                          "simulation_dt": sim_dt_global,
                          "random_seed": master_seed,
                          "n_neurons_total": total_neurons_global,
                          "n_connections": chromosome_len_global,
                          "n_classes": N_CLASSES,
                          "downsample_factor": downsample_factor_global,
                          "mnist_stim_duration_ms": mnist_stim_duration_global,
                          "max_frequency_hz": max_freq_hz_global,
                      }
                      with open(config_save_path, 'w') as f:
                          json.dump(config_to_save, f, indent=4)
                      print(f"Saved configuration parameters to {config_save_path}")

                      # Optionally save precomputed spikes
                      # np.save(spikes_save_path, precomputed_spike_trains_global)
                      # print(f"Saved precomputed spike trains to {spikes_save_path}")

                  except Exception as e:
                      print(f"Error saving network state files: {e}")
                  # --- END SAVING BLOCK ---

              else:
                  print("Warning: Best fitness index is out of bounds for population.")
         else:
             print("Warning: All fitness scores were invalid (-inf or NaN). Cannot determine best chromosome.")
    else:
         print("Warning: No valid fitness scores available. Cannot determine best chromosome.")


    overall_end_time = time.time()
    print(f"\nTotal script execution time: {overall_end_time - start_overall_time:.2f}s")

    # --- Plot Final GA Fitness History ---
    if best_fitness_history:
         print("\nPlotting final GA fitness history...")
         final_plot_filename = os.path.join(output_dir, f"mnist_snn_ga_fitness_plot_final_{N_CLASSES}class_precomputed.png") # Updated name
         plot_ga_progress(len(best_fitness_history), best_fitness_history, avg_fitness_history, final_test_accuracy, final_plot_filename, FITNESS_EVAL_EXAMPLES)
         print(f"Saved final GA fitness plot to {final_plot_filename}")

    print("\n--- Script Finished ---")
    # plt.close('all') # Optional: Close figures if running interactively

 











#precalculate delays
#precalculate input spieks in mnist
#change connections to json 