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

        # --- Inhibitory Pairing Metadata (for 1:1 WTA) ---
        self.inhibitory_pairing = {}  # {exc_idx: inh_idx} mapping
        self.excitatory_neurons_indices = []  # List of excitatory neuron indices
        self.dedicated_inhibitory_indices = []  # List of dedicated inhibitory neuron indices

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
        self.adaptation_increment = get_param('adaptation_increment', 0.3)
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

        # --- Adaptive Threshold State (for preventing runaway activity) ---
        self.theta = np.zeros(n_neurons, dtype=float)  # Adaptive threshold offset
        self.tau_theta = np.full(n_neurons, 1e7, dtype=float)  # Very slow decay time constant (ms)
        self.theta_plus = np.full(n_neurons, 0.05, dtype=float)  # Threshold increment per spike (mV)

        # --- Simulation State ---
        self.spike_queue = deque()
        self.network_activity = [] # Records indices of active neurons per time step

        # --- Avalanche Tracking ---
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        # --- STDP State Variables ---
        self.trace_pre = np.zeros(n_neurons, dtype=float)  # Pre-synaptic traces
        self.trace_post = np.zeros(n_neurons, dtype=float)  # Post-synaptic traces
        self.stdp_enabled = False  # Flag to enable/disable STDP
        self.learning_phase = False  # Flag to track if in learning window
        self.current_target_class = None  # Target class for supervised learning
        self.last_prediction_correct = None  # Whether last prediction was correct
        self.stdp_weight_deltas = {}  # Accumulator for weight changes: {(u,v): delta_w}

        # --- Eligibility Traces (for reward-modulated STDP) ---
        self.eligibility_traces = {}  # {(pre, post): trace_value} - potential weight changes
        self.tau_eligibility = 20.0  # Eligibility trace decay time constant (ms)
        self.reward_signal = 0.0  # Current reward signal (+1 correct, -1 incorrect, 0 neutral)

        # STDP parameters (can be set externally)
        self.stdp_a_plus = 0.001  # LTP learning rate
        self.stdp_a_minus = 0.0012  # LTD learning rate (slightly larger)
        self.stdp_tau_pre = 20.0  # Pre-synaptic trace time constant (ms)
        self.stdp_tau_post = 20.0  # Post-synaptic trace time constant (ms)
        self.stdp_w_min = 0.002  # Minimum weight (match GA bounds)
        self.stdp_w_max = 0.35  # Maximum weight (match GA bounds)
        self.stdp_x_target = 0.0  # CHANGED TO 0.0 FOR DIEHL PURE HEBBIAN STDP (was 0.1)


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

        # Reset STDP state
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)
        self.stdp_weight_deltas = {}
        self.stdp_enabled = False
        self.learning_phase = False

        # Reset adaptive thresholds
        self.theta.fill(0.0)

        # Reset eligibility traces
        self.eligibility_traces = {}
        self.reward_signal = 0.0

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

        # --- 3.5. Decay Adaptive Thresholds (Vectorized) ---
        # Decay with very slow time constant (tau_theta = 1e7 ms ~ 166 minutes)
        exp_decay_theta = np.exp(-dt / np.where(self.tau_theta > 1e-9, self.tau_theta, 1e-9))
        self.theta *= exp_decay_theta

        # --- 3.6. Decay STDP Traces (Vectorized) ---
        if hasattr(self, 'trace_pre'):  # Check if STDP variables exist
            exp_decay_pre = np.exp(-dt / max(1e-9, self.stdp_tau_pre))
            exp_decay_post = np.exp(-dt / max(1e-9, self.stdp_tau_post))
            self.trace_pre *= exp_decay_pre
            self.trace_post *= exp_decay_post

        # --- 3.7. Decay Eligibility Traces (Vectorized) ---
        if hasattr(self, 'eligibility_traces') and self.eligibility_traces:
            exp_decay_elig = np.exp(-dt / max(1e-9, self.tau_eligibility))

            # Decay all eligibility traces
            decayed_traces = {}
            for key, value in self.eligibility_traces.items():
                decayed_value = value * exp_decay_elig
                if abs(decayed_value) > 1e-9:  # Keep only significant traces
                    decayed_traces[key] = decayed_value

            self.eligibility_traces = decayed_traces

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
        # Neurons spike if voltage reaches ADAPTIVE threshold AND they are not refractory
        # Effective threshold = base threshold + adaptive component
        effective_threshold = self.v_threshold + self.theta
        spiked_mask = (self.v >= effective_threshold) & non_refractory_mask
        active_indices = np.where(spiked_mask)[0]

        if active_indices.size > 0:
             # Reset voltage for spiking neurons
             self.v[spiked_mask] = self.v_reset[spiked_mask]
             # Reset refractory timer
             self.t_since_spike[spiked_mask] = 0.0
             # Update adaptation for spiking neurons
             self.adaptation[spiked_mask] += self.adaptation_increment[spiked_mask]

             # Increase adaptive threshold for spiking neurons (prevents runaway activity)
             self.theta[spiked_mask] += self.theta_plus[spiked_mask]

             # --- STDP: Update traces and accumulate weight changes ---
             if hasattr(self, 'learning_phase') and self.learning_phase:
                 # Update pre-synaptic traces for spiking neurons (always during learning_phase)
                 self.trace_pre[active_indices] += 1.0

                 # DEBUG: Track accumulation (only first example in session)
                 if not hasattr(self, '_debug_accumulation_shown'):
                     self._debug_accumulation_shown = False
                 n_elig_before = len(self.eligibility_traces)

                 # Accumulate STDP weight changes based on current traces
                 # This happens during input/readout when neurons are spiking
                 self._accumulate_stdp_updates(active_indices)

                 # DEBUG: Show first accumulation
                 if not self._debug_accumulation_shown:
                     n_elig_after = len(self.eligibility_traces)
                     if n_elig_after > n_elig_before:
                         print(f"[DEBUG ACCUMULATE] First eligibility trace accumulation: "
                               f"{n_elig_before} → {n_elig_after} traces, "
                               f"{len(active_indices)} neurons spiked")
                         self._debug_accumulation_shown = True

                 # Update post-synaptic traces for spiking neurons (for future pre-synaptic events)
                 self.trace_post[active_indices] += 1.0

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

    # --- STDP Methods ---

    def _accumulate_eligibility_traces(self, spiking_neurons):
        """
        Accumulate eligibility traces (potential weight changes) when neurons spike.

        REWARD-MODULATED STDP:
        Eligibility traces record what WOULD change based on spike timing,
        but actual weight changes only occur when a reward signal is applied.

        This implements the three-factor learning rule:
        ΔW = reward × eligibility × weight_dependence

        Args:
            spiking_neurons (np.ndarray): Indices of neurons that just spiked
        """
        if not self.learning_phase:
            return

        # Get learning rates
        a_plus = self.stdp_a_plus
        a_minus = self.stdp_a_minus
        x_tar = self.stdp_x_target  # Pre-synaptic trace target

        for post_idx in spiking_neurons:
            # Post-synaptic neuron just spiked
            # LTP eligibility: proportional to (pre-synaptic trace - target)
            if post_idx in self.graph:
                for pre_idx in self.graph.predecessors(post_idx):
                    if pre_idx < self.n_neurons:
                        # LTP rule with pre-synaptic target: Δw ∝ (x_pre - x_tar)
                        # This provides stability - synapses from inactive pre-neurons are weakened
                        if self.trace_pre[pre_idx] > 0 or x_tar > 0:
                            eligibility_ltp = a_plus * (self.trace_pre[pre_idx] - x_tar)

                            # Accumulate eligibility (NOT weight change yet!)
                            key = (pre_idx, post_idx)
                            self.eligibility_traces[key] = self.eligibility_traces.get(key, 0.0) + eligibility_ltp

            # Pre-synaptic neuron just spiked
            # LTD eligibility: proportional to post-synaptic trace
            if post_idx in self.graph:
                for target_idx in self.graph.successors(post_idx):
                    if target_idx < self.n_neurons:
                        # LTD: standard depression based on post-synaptic trace
                        if self.trace_post[target_idx] > 0:
                            eligibility_ltd = -a_minus * self.trace_post[target_idx]

                            # Accumulate eligibility (NOT weight change yet!)
                            key = (post_idx, target_idx)
                            self.eligibility_traces[key] = self.eligibility_traces.get(key, 0.0) + eligibility_ltd

    def _accumulate_stdp_updates(self, spiking_neurons):
        """
        DEPRECATED: Legacy name for backward compatibility.
        Now delegates to _accumulate_eligibility_traces().
        """
        self._accumulate_eligibility_traces(spiking_neurons)

    def apply_reward_modulated_stdp(self, reward_signal):
        """
        Apply accumulated eligibility traces modulated by reward signal.

        REWARD-MODULATED STDP (Three-Factor Learning):
        ΔW = reward × eligibility × weight_dependence

        This implements dopamine-modulated plasticity where:
        - Correct prediction (reward=+1): Strengthen connections that contributed
        - Incorrect prediction (reward=-1): Weaken (anti-STDP) those connections
        - No prediction/neutral (reward=0): No learning

        Args:
            reward_signal (float): +1.0 (correct), -1.0 (incorrect), 0.0 (neutral)
        """
        if not self.eligibility_traces:
            return

        for (pre_idx, post_idx), eligibility in self.eligibility_traces.items():
            if 0 <= pre_idx < self.n_neurons and 0 <= post_idx < self.n_neurons:
                current_weight = self.weights[pre_idx, post_idx]
                is_inhibitory_conn = self.is_inhibitory[pre_idx]
                current_magnitude = abs(current_weight)

                # Apply weight-dependent factor (multiplicative STDP)
                if eligibility > 0:  # LTP component
                    weight_dependent_factor = (self.stdp_w_max - current_magnitude)
                else:  # LTD component
                    weight_dependent_factor = (current_magnitude - self.stdp_w_min)

                # Final weight change: reward × eligibility × weight_dependence
                actual_delta = reward_signal * eligibility * weight_dependent_factor

                # Update magnitude
                new_magnitude = current_magnitude + actual_delta
                new_magnitude = np.clip(new_magnitude, self.stdp_w_min, self.stdp_w_max)

                # Restore sign based on source neuron type
                if is_inhibitory_conn:
                    new_weight = -new_magnitude
                else:
                    new_weight = new_magnitude

                # Update weight
                self.weights[pre_idx, post_idx] = new_weight

                # Update graph if edge exists
                if self.graph.has_edge(pre_idx, post_idx):
                    self.graph[pre_idx][post_idx]['weight'] = new_weight

        # Clear eligibility traces after applying reward
        self.eligibility_traces = {}

    def apply_unsupervised_stdp(self):
        """
        Apply DIRECT unsupervised STDP updates (Diehl & Cook style).

        Unlike reward-modulated STDP, this applies weight changes immediately
        based on accumulated eligibility traces WITHOUT a reward signal.
        This is classical Hebbian STDP where timing alone determines plasticity.

        Diehl & Cook use ADDITIVE STDP where eligibility directly becomes weight change:
        ΔW = eligibility (already includes A+/A- learning rates)

        The weight-dependence is implicitly handled by homeostatic normalization
        which rescales weights after each update.

        Used in: Diehl & Cook 2015 unsupervised learning
        """
        if not self.eligibility_traces:
            return

        for (pre_idx, post_idx), eligibility in self.eligibility_traces.items():
            if 0 <= pre_idx < self.n_neurons and 0 <= post_idx < self.n_neurons:
                current_weight = self.weights[pre_idx, post_idx]
                is_inhibitory_conn = self.is_inhibitory[pre_idx]
                current_magnitude = abs(current_weight)

                # ADDITIVE STDP: eligibility directly becomes weight change
                # (eligibility already contains A+ or A- learning rate)
                actual_delta = eligibility

                # Update magnitude
                new_magnitude = current_magnitude + actual_delta
                new_magnitude = np.clip(new_magnitude, self.stdp_w_min, self.stdp_w_max)

                # Restore sign based on source neuron type
                if is_inhibitory_conn:
                    new_weight = -new_magnitude
                else:
                    new_weight = new_magnitude

                # Update weight
                self.weights[pre_idx, post_idx] = new_weight

                # Update graph if edge exists
                if self.graph.has_edge(pre_idx, post_idx):
                    self.graph[pre_idx][post_idx]['weight'] = new_weight

        # Clear eligibility traces after applying
        self.eligibility_traces = {}

    def apply_stdp_updates(self, true_label=None, output_layer_range=None):
        """
        DEPRECATED: Legacy method for backward compatibility.

        For new code, use apply_reward_modulated_stdp() instead.
        This method converts the old API to the new reward-modulated approach.

        Args:
            true_label: Ground truth class label (optional, for supervised learning)
            output_layer_range: Tuple (start_idx, end_idx) of output layer neurons
        """
        # Legacy compatibility: use old stdp_weight_deltas if they exist
        if self.stdp_weight_deltas:
            # Old-style STDP - apply deltas directly with reward=+1
            for (pre_idx, post_idx), delta_w in self.stdp_weight_deltas.items():
                if 0 <= pre_idx < self.n_neurons and 0 <= post_idx < self.n_neurons:
                    current_weight = self.weights[pre_idx, post_idx]
                    is_inhibitory_conn = self.is_inhibitory[pre_idx]
                    current_magnitude = abs(current_weight)

                    new_magnitude = current_magnitude + delta_w
                    new_magnitude = np.clip(new_magnitude, self.stdp_w_min, self.stdp_w_max)

                    if is_inhibitory_conn:
                        new_weight = -new_magnitude
                    else:
                        new_weight = new_magnitude

                    self.weights[pre_idx, post_idx] = new_weight
                    if self.graph.has_edge(pre_idx, post_idx):
                        self.graph[pre_idx][post_idx]['weight'] = new_weight

            self.stdp_weight_deltas = {}
        else:
            # New-style: use reward-modulated STDP
            # Assume correct if called without explicit reward signal
            self.apply_reward_modulated_stdp(reward_signal=1.0)

    def apply_anti_stdp_updates(self, true_label=None, predicted_label=None, output_layer_range=None):
        """
        Legacy method for backward compatibility.
        Now delegates to apply_stdp_updates with supervised learning parameters.

        This method is kept for compatibility but the supervised logic is now
        in apply_stdp_updates() which handles both correct and incorrect cases.

        Note: predicted_label parameter is ignored in simplified version.
        """
        # Just call the unified supervised STDP method (predicted_label no longer used)
        self.apply_stdp_updates(true_label, output_layer_range)

    def apply_homeostatic_normalization(self, connection_map=None, target_sum_per_neuron=None):
        """
        Apply homeostatic normalization: normalize incoming weight sum per neuron.
        Implements synaptic scaling - a biologically inspired mechanism.

        Args:
            connection_map (list of tuples): List of (source, target) connections.
                                             If None, uses all non-zero weights.
            target_sum_per_neuron (float or dict): Target sum of incoming weights.
                                                   If float, same for all neurons.
                                                   If dict, maps neuron_idx -> target_sum.
                                                   If None, maintains current average sum.
        """
        # Build incoming connections per neuron
        incoming_weights = {i: [] for i in range(self.n_neurons)}

        if connection_map is not None:
            # Use provided connection map
            for pre_idx, post_idx in connection_map:
                if 0 <= pre_idx < self.n_neurons and 0 <= post_idx < self.n_neurons:
                    w = self.weights[pre_idx, post_idx]
                    if w != 0:
                        incoming_weights[post_idx].append((pre_idx, abs(w)))
        else:
            # Use all non-zero weights
            for post_idx in range(self.n_neurons):
                for pre_idx in range(self.n_neurons):
                    w = self.weights[pre_idx, post_idx]
                    if w != 0:
                        incoming_weights[post_idx].append((pre_idx, abs(w)))

        # Normalize each neuron's incoming weights
        for post_idx in range(self.n_neurons):
            connections = incoming_weights[post_idx]
            if not connections:
                continue

            # Calculate current sum
            current_sum = sum(w for _, w in connections)
            if current_sum < 1e-9:
                continue

            # Determine target sum
            if target_sum_per_neuron is None:
                # Maintain current sum
                target_sum = current_sum
            elif isinstance(target_sum_per_neuron, dict):
                target_sum = target_sum_per_neuron.get(post_idx, current_sum)
            else:
                target_sum = float(target_sum_per_neuron)

            # Calculate scaling factor
            scale_factor = target_sum / current_sum

            # Apply scaling to all incoming weights
            for pre_idx, _ in connections:
                current_weight = self.weights[pre_idx, post_idx]
                is_inhibitory_conn = self.is_inhibitory[pre_idx]

                # Scale magnitude
                new_magnitude = abs(current_weight) * scale_factor

                # Clip to bounds
                new_magnitude = np.clip(new_magnitude, self.stdp_w_min, self.stdp_w_max)

                # Restore sign
                if is_inhibitory_conn:
                    new_weight = -new_magnitude
                else:
                    new_weight = new_magnitude

                # Update weight
                self.weights[pre_idx, post_idx] = new_weight

                # Update graph if edge exists
                if self.graph.has_edge(pre_idx, post_idx):
                    self.graph[pre_idx][post_idx]['weight'] = new_weight

    def reset_transient_state(self):
        """
        Reset transient state between digit windows, preserving weights and structure.
        Resets: voltages, conductances, traces, spike queue, activity records.
        Preserves: weights, connections, positions, delays, is_inhibitory.
        """
        # Reset neuronal state
        self.v[:] = self.v_rest
        self.g_e.fill(0.0)
        self.g_i.fill(0.0)
        self.adaptation.fill(0.0)
        self.t_since_spike[:] = self.tau_ref + 1e-5
        self.external_stim_g.fill(0.0)

        # Reset spike queue and activity
        self.spike_queue = deque()
        self.network_activity = []

        # Reset STDP traces
        if hasattr(self, 'trace_pre'):
            self.trace_pre.fill(0.0)
            self.trace_post.fill(0.0)
            self.stdp_weight_deltas = {}

        # Reset avalanche tracking
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        # Note: Do NOT reset:
        # - self.weights
        # - self.delays
        # - self.graph
        # - self.is_inhibitory
        # - self.neuron_grid_positions
        # - self.avalanche_sizes / durations (cumulative metrics)

    def set_stdp_params(self, a_plus=None, a_minus=None, tau_pre=None, tau_post=None,
                        w_min=None, w_max=None):
        """
        Set STDP parameters.

        Args:
            a_plus (float): LTP learning rate
            a_minus (float): LTD learning rate
            tau_pre (float): Pre-synaptic trace time constant (ms)
            tau_post (float): Post-synaptic trace time constant (ms)
            w_min (float): Minimum weight magnitude
            w_max (float): Maximum weight magnitude
        """
        if a_plus is not None:
            self.stdp_a_plus = a_plus
        if a_minus is not None:
            self.stdp_a_minus = a_minus
        if tau_pre is not None:
            self.stdp_tau_pre = tau_pre
        if tau_post is not None:
            self.stdp_tau_post = tau_post
        if w_min is not None:
            self.stdp_w_min = w_min
        if w_max is not None:
            self.stdp_w_max = w_max