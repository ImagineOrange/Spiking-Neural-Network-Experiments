from collections import deque
import numpy as np
import networkx as nx
# Note: Removed direct dependency on individual neuron class

class LayeredNeuronalNetworkVectorized:
    """
    Represents a network of LIF neurons (potentially with multiple layers).
    This version uses NumPy arrays for vectorized state updates, improving performance.
    It includes spike propagation with delays and basic avalanche tracking.
    Initialization requires setting parameters and initial states directly into arrays,
    likely via a helper function.
    """
    def __init__(self, n_neurons=100, **kwargs):
        """
        Initializes the vectorized neuronal network structure.

        Args:
            n_neurons (int): Total number of neurons in the network.
            **kwargs: Dictionary containing neuron parameters. Expects keys like:
                      'v_rest', 'v_threshold', 'v_reset', 'tau_m', 'tau_ref',
                      'tau_e', 'tau_i', 'e_reversal', 'i_reversal',
                      'v_noise_amp', 'i_noise_amp', 'adaptation_increment',
                      'tau_adaptation', 'is_inhibitory' (as a boolean array/list).
                      Can contain scalars (uniform parameters) or NumPy arrays
                      of size n_neurons (heterogeneous parameters).
        """
        self.n_neurons = n_neurons

        # --- Network Structure ---
        self.graph = nx.DiGraph()
        self.weights = np.zeros((n_neurons, n_neurons))
        self.delays = np.zeros((n_neurons, n_neurons))
        self.neuron_grid_positions = {} # Store positions if needed for visualization

        # --- Neuron Parameters (Store as NumPy arrays) ---
        # Fetch parameters from kwargs, defaulting to typical values or raising errors
        # Allows for uniform (scalar) or heterogeneous (array) parameters
        def get_param(key, default):
            val = kwargs.get(key, default)
            # Convert scalar to array if needed, ensure correct size
            if np.isscalar(val):
                return np.full(n_neurons, val, dtype=float)
            elif isinstance(val, (list, np.ndarray)) and len(val) == n_neurons:
                return np.array(val, dtype=float)
            else:
                raise ValueError(f"Parameter '{key}' must be scalar or array of size {n_neurons}")

        self.v_rest = get_param('v_rest', -65.0)
        self.v_threshold = get_param('v_threshold', -55.0)
        self.v_reset = get_param('v_reset', -75.0)
        self.tau_m = get_param('tau_m', 10.0)
        self.tau_ref = get_param('tau_ref', 2.0)
        self.tau_e = get_param('tau_e', 3.0)
        self.tau_i = get_param('tau_i', 7.0)
        self.e_reversal = get_param('e_reversal', 0.0)
        self.i_reversal = get_param('i_reversal', -70.0)
        self.v_noise_amp = get_param('v_noise_amp', 0.5)
        self.i_noise_amp = get_param('i_noise_amp', 0.05)
        self.adaptation_increment = get_param('adaptation_increment', 0.5)
        self.tau_adaptation = get_param('tau_adaptation', 100.0)

        # Inhibitory status (must be provided as a boolean array/list)
        is_inhib_param = kwargs.get('is_inhibitory', None)
        if is_inhib_param is None or len(is_inhib_param) != n_neurons:
             raise ValueError(f"'is_inhibitory' must be provided as array/list of size {n_neurons}")
        self.is_inhibitory = np.array(is_inhib_param, dtype=bool)

        # --- Neuron State Variables (Initialize as NumPy arrays) ---
        self.v = np.full(n_neurons, self.v_rest, dtype=float)
        self.g_e = np.zeros(n_neurons, dtype=float)
        self.g_i = np.zeros(n_neurons, dtype=float)
        self.adaptation = np.zeros(n_neurons, dtype=float)
        # Initialize outside refractory period
        self.t_since_spike = np.full(n_neurons, self.tau_ref + 1e-5, dtype=float)
        self.external_stim_g = np.zeros(n_neurons, dtype=float) # For direct stimulation

        # --- Simulation State ---
        self.spike_queue = deque()
        self.network_activity = [] # Records indices of active neurons per time step

        # --- Avalanche Tracking ---
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None


    def add_connection(self, u, v, weight, delay=1.0):
         """
         Adds a directed connection from neuron 'u' to neuron 'v'.
         Updates graph, weights, and delays matrices.
         """
         if u < self.n_neurons and v < self.n_neurons and u in self.graph and v in self.graph:
              delay = max(0.1, delay) # Ensure minimum delay
              self.graph.add_edge(u, v, weight=weight, delay=delay)
              self.weights[u, v] = weight
              self.delays[u, v] = delay
         #else:
         #    print(f"Warning: Cannot add connection ({u}->{v}). Node missing or index out of bounds.")

    # Note: add_neuron is removed as state/params are now arrays.
    # Network structure (nodes, parameters, connections) should be set up
    # before simulation, likely by the function creating this network instance.

    def reset_all(self):
        """Resets all neuron states and clears simulation variables."""
        self.v[:] = self.v_rest # Reset voltage to resting potential
        self.g_e.fill(0.0)
        self.g_i.fill(0.0)
        self.adaptation.fill(0.0)
        self.t_since_spike[:] = self.tau_ref + 1e-5 # Reset refractory timer
        self.external_stim_g.fill(0.0)

        # Reset network-level simulation variables
        self.network_activity = []
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None
        self.spike_queue = deque()

    def set_external_stimulus(self, neuron_indices, conductance_values):
        """Applies external stimulus conductance to specified neurons."""
        # Ensure inputs are NumPy arrays for efficient assignment
        indices = np.array(neuron_indices, dtype=int)
        values = np.array(conductance_values, dtype=float)
        if len(indices) != len(values):
            raise ValueError("Number of indices must match number of conductance values.")
        # Reset previous stimuli first? Or assume additive? Let's reset.
        # self.external_stim_g.fill(0.0)
        # Apply new stimuli, ensuring conductance is non-negative
        self.external_stim_g[indices] = np.maximum(0, values)


    def update_network(self, dt):
        """
        Updates the entire network state for one time step 'dt' using vectorized operations.

        Args:
            dt (float): The simulation time step (ms).

        Returns:
            np.ndarray: An array of indices of neurons that spiked in this time step.
        """
        current_time = len(self.network_activity) * dt

        # --- 0. Pre-calculate Decay Factors (if not done at init) ---
        # If dt can change, recalculate decay factors here
        # Optimized: Calculate only once if dt is constant
        exp_decay_m = np.exp(-dt / np.where(self.tau_m > 1e-9, self.tau_m, 1e-9))
        exp_decay_e = np.exp(-dt / np.where(self.tau_e > 1e-9, self.tau_e, 1e-9))
        exp_decay_i = np.exp(-dt / np.where(self.tau_i > 1e-9, self.tau_i, 1e-9))
        exp_decay_adapt = np.exp(-dt / np.where(self.tau_adaptation > 1e-9, self.tau_adaptation, 1e-9))

        # --- 1. Process Delayed Spikes (Update Conductances) ---
        delta_g_e = np.zeros(self.n_neurons, dtype=float)
        delta_g_i = np.zeros(self.n_neurons, dtype=float)
        delivered_spikes = 0

        while self.spike_queue and self.spike_queue[0][0] <= current_time:
            delivery_time, source_idx, target_idx, weight = self.spike_queue.popleft()
            if target_idx < self.n_neurons: # Check if target index is valid
                if weight > 0:
                    # Accumulate excitatory conductance change efficiently
                    np.add.at(delta_g_e, target_idx, weight)
                else:
                     # Accumulate inhibitory conductance change efficiently
                    np.add.at(delta_g_i, target_idx, -weight) # Use positive value for g_i
                delivered_spikes += 1

        # Apply accumulated conductance changes
        self.g_e += delta_g_e
        self.g_i += delta_g_i

        # --- 2. Update Neuron States (Vectorized) ---

        # Identify neurons in refractory period
        refractory_mask = self.t_since_spike < self.tau_ref
        non_refractory_mask = ~refractory_mask

        # Update time since last spike for all neurons
        self.t_since_spike += dt

        # Update non-refractory neurons
        if np.any(non_refractory_mask):
             # Calculate synaptic currents for non-refractory neurons
             i_e = self.g_e[non_refractory_mask] * (self.e_reversal[non_refractory_mask] - self.v[non_refractory_mask])
             i_i = self.g_i[non_refractory_mask] * (self.i_reversal[non_refractory_mask] - self.v[non_refractory_mask])
             i_stim = self.external_stim_g[non_refractory_mask] * (self.e_reversal[non_refractory_mask] - self.v[non_refractory_mask]) # Assuming excitatory stim
             i_syn = i_e + i_i + i_stim

             # Calculate membrane potential change (dv)
             dv = (-(self.v[non_refractory_mask] - self.v_rest[non_refractory_mask]) / self.tau_m[non_refractory_mask] + i_syn - self.adaptation[non_refractory_mask]) * dt

             # Add membrane noise (scaled by sqrt(dt))
             v_noise = np.random.normal(0, self.v_noise_amp[non_refractory_mask] * np.sqrt(dt))
             # Update voltage
             self.v[non_refractory_mask] += dv + v_noise

        # Clamp voltage for refractory neurons
        self.v[refractory_mask] = self.v_reset[refractory_mask]

        # --- 3. Decay Conductances and Adaptation (Vectorized) ---
        self.g_e *= exp_decay_e
        self.g_i *= exp_decay_i
        self.adaptation *= exp_decay_adapt

        # Apply synaptic noise (after decay, before spike check)
        if np.any(self.i_noise_amp > 0):
            # Calculate noise scaled by sqrt(dt)
            noise_scale_e = self.i_noise_amp * np.sqrt(dt)
            noise_scale_i = self.i_noise_amp * np.sqrt(dt)
            # Add noise only where amp > 0 to avoid unnecessary random calls
            e_noise_mask = self.i_noise_amp > 0
            i_noise_mask = self.i_noise_amp > 0
            self.g_e[e_noise_mask] += np.random.normal(0, noise_scale_e[e_noise_mask])
            self.g_i[i_noise_mask] += np.random.normal(0, noise_scale_i[i_noise_mask])
            # Ensure conductances remain non-negative
            np.maximum(self.g_e, 0, out=self.g_e)
            np.maximum(self.g_i, 0, out=self.g_i)


        # --- 4. Spike Detection & Post-Spike Updates (Vectorized) ---
        # Neurons spike if voltage reaches threshold AND they are not refractory
        spiked_mask = (self.v >= self.v_threshold) & non_refractory_mask
        active_indices = np.where(spiked_mask)[0]

        if active_indices.size > 0:
             # Reset voltage for spiking neurons
             self.v[spiked_mask] = self.v_reset[spiked_mask]
             # Reset refractory timer
             self.t_since_spike[spiked_mask] = 0.0
             # Update adaptation for spiking neurons
             self.adaptation[spiked_mask] += self.adaptation_increment[spiked_mask]

        # --- 5. Queue New Spikes ---
        if active_indices.size > 0:
            new_spikes_to_queue = []
            for i in active_indices:
                 # Check graph successors efficiently
                 # Note: NetworkX iteration isn't vectorized, but happens only for spiking neurons
                 if i in self.graph:
                      for j in self.graph.successors(i):
                           if j < self.n_neurons and self.weights[i, j] != 0: # Ensure target valid & weight non-zero
                                weight = self.weights[i, j]
                                # Ensure delay is at least dt
                                delay = max(dt, self.delays[i, j]) if self.delays[i,j] > 0 else dt
                                delivery_time = current_time + delay
                                new_spikes_to_queue.append((delivery_time, i, j, weight))

            # Extend the deque and re-sort ONLY if new spikes were added
            if new_spikes_to_queue:
                self.spike_queue.extend(new_spikes_to_queue)
                self.spike_queue = deque(sorted(self.spike_queue, key=lambda x: x[0]))


        # --- 6. Update Network Activity and Avalanche Tracking ---
        activity_level = active_indices.size
        self.network_activity.append(active_indices) # Store spiking indices

        if activity_level > 0:
            if self.current_avalanche_start is None:
                self.current_avalanche_start = current_time
                self.current_avalanche_size = activity_level
            else:
                self.current_avalanche_size += activity_level
        elif self.current_avalanche_start is not None:
             # Use time at the START of the current step for duration calculation
            duration = current_time - self.current_avalanche_start
            if duration >= dt: # Only record if duration is at least one step
                self.avalanche_sizes.append(self.current_avalanche_size)
                self.avalanche_durations.append(duration)
            self.current_avalanche_start = None
            self.current_avalanche_size = 0

        return active_indices

    # --- Weight Accessor/Mutator Methods (Integrated) ---

    def get_weights_sparse(self, connection_map):
        """
        Retrieves weights for connections specified in the connection_map.

        Args:
            connection_map (list of tuples): List of (source_idx, target_idx) tuples.

        Returns:
            np.ndarray: Array of weights corresponding to the connection_map.
        """
        sparse_weights = np.zeros(len(connection_map))
        for i, (u, v) in enumerate(connection_map):
            # Check bounds to prevent errors with the weights matrix
            if 0 <= u < self.n_neurons and 0 <= v < self.n_neurons:
                 sparse_weights[i] = self.weights[u, v]
            else:
                 print(f"Warning: Invalid index ({u},{v}) in connection_map during get_weights_sparse.")
                 # Handle appropriately, e.g., assign NaN or 0, or raise error
                 sparse_weights[i] = 0.0 # Or np.nan
        return sparse_weights

    def set_weights_sparse(self, sparse_weights_vector, connection_map):
        """
        Sets network weights from a sparse vector, applying sign based on
        the source neuron's inhibitory status (read from self.is_inhibitory array).
        Assumes sparse_weights_vector contains POSITIVE weight magnitudes.

        Args:
            sparse_weights_vector (np.ndarray): Vector of positive weight magnitudes.
            connection_map (list of tuples): List of (source_idx, target_idx) tuples.
        """
        if len(sparse_weights_vector) != len(connection_map):
            raise ValueError(f"Sparse weight vector length {len(sparse_weights_vector)} "
                             f"does not match connection map length {len(connection_map)}")

        for i, (u, v) in enumerate(connection_map):
             # Check indices are within bounds
             if 0 <= u < self.n_neurons and 0 <= v < self.n_neurons:
                 weight_magnitude = abs(sparse_weights_vector[i]) # Ensure positive magnitude

                 # Check if the SOURCE neuron 'u' is inhibitory using the boolean array
                 if self.is_inhibitory[u]:
                     final_weight = -weight_magnitude # Apply negative sign
                 else:
                     final_weight = weight_magnitude # Use positive sign

                 # Store the final signed weight in the matrix
                 self.weights[u, v] = final_weight
                 # Update the graph attribute as well if the graph exists and has the edge
                 if self.graph.has_edge(u, v):
                     self.graph[u][v]['weight'] = final_weight
             else:
                 print(f"Warning: Invalid index ({u},{v}) during set_weights_sparse.")