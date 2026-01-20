import numpy as np
import networkx as nx


class SphericalNeuronalNetworkVectorized:
    """
    Vectorized 3D spherical neuronal network using NumPy arrays for all state.

    This implementation achieves significant speedup over the object-based version by:
    1. Storing all neuron state in NumPy arrays (not individual objects)
    2. Using vectorized operations for membrane dynamics
    3. Using a heap-based spike queue instead of sorted deque
    4. Batch processing spike delivery with np.add.at()
    5. Precomputing exponential decay factors

    Mathematical model is identical to SphericalNeuronalNetwork:
    - Conductance-based synapses with reversal potentials
    - Spike-frequency adaptation with potassium reversal
    - Distance-weighted connectivity
    - Transmission delays
    """

    def __init__(self, n_neurons=100, connection_p=0.1, connection_probabilities=None,
                 weight_scale=1.0, weight_min=0.0, spatial=True, transmission_delay=1.0,
                 inhibitory_fraction=0.2, layout='sphere',
                 v_noise_amp=0.3, i_noise_amp=0.05,
                 e_reversal=0.0, i_reversal=-80.0, distance_lambda=1,
                 lambda_decay_ie=0.05,
                 jitter_v_rest=0.0, jitter_v_threshold=0.0, jitter_tau_m=0.0,
                 jitter_tau_ref=0.0, jitter_tau_e=0.0, jitter_tau_i=0.0,
                 jitter_adaptation_increment=0.0, jitter_tau_adaptation=0.0):
        """
        Initialize the vectorized 3D spherical network.

        Parameters match SphericalNeuronalNetwork for compatibility.
        """
        self.n_neurons = n_neurons
        self.weight_scale = weight_scale
        self.distance_lambda = distance_lambda
        self.lambda_decay_ie = lambda_decay_ie

        # Store jitter parameters
        self.jitter_params = {
            'v_rest': jitter_v_rest,
            'v_threshold': jitter_v_threshold,
            'tau_m': jitter_tau_m,
            'tau_ref': jitter_tau_ref,
            'tau_e': jitter_tau_e,
            'tau_i': jitter_tau_i,
            'adaptation_increment': jitter_adaptation_increment,
            'tau_adaptation': jitter_tau_adaptation
        }

        # Resolve per-connection-type probabilities
        self.connection_probabilities = self._resolve_connection_probabilities(
            connection_p, connection_probabilities
        )

        # Calculate sphere radius
        self.sphere_radius = (3 * n_neurons / (4 * np.pi)) ** (1/3) * 1.5

        # Create graph for connectivity analysis
        self.graph = nx.DiGraph()

        # ===== VECTORIZED STATE ARRAYS =====
        # These replace individual neuron objects

        # Determine which neurons are inhibitory
        self.is_inhibitory = np.random.rand(n_neurons) < inhibitory_fraction

        # Base parameter values
        base_v_rest = -65.0
        base_v_threshold = -55.0
        base_v_reset = -75.0
        base_tau_m = 10.0
        base_tau_ref_exc = 4.0
        base_tau_ref_inh = 2.5
        base_tau_e = 3.0
        base_tau_i = 7.0
        base_adaptation_increment = 0.2
        base_tau_adaptation = 100.0
        k_reversal = -90.0

        # Generate jittered parameters (vectorized)
        self.v_rest = self._jitter_gaussian(base_v_rest, jitter_v_rest, n_neurons)
        self.v_threshold = self._jitter_gaussian(base_v_threshold, jitter_v_threshold, n_neurons)
        self.v_reset = np.full(n_neurons, base_v_reset)
        self.tau_m = self._jitter_lognormal(base_tau_m, jitter_tau_m, n_neurons)
        self.tau_e = self._jitter_lognormal(base_tau_e, jitter_tau_e, n_neurons)
        self.tau_i = self._jitter_lognormal(base_tau_i, jitter_tau_i, n_neurons)
        self.adaptation_increment = self._jitter_lognormal(base_adaptation_increment, jitter_adaptation_increment, n_neurons)
        self.tau_adaptation = self._jitter_lognormal(base_tau_adaptation, jitter_tau_adaptation, n_neurons)

        # Refractory period depends on neuron type
        base_tau_ref = np.where(self.is_inhibitory, base_tau_ref_inh, base_tau_ref_exc)
        self.tau_ref = self._jitter_lognormal_array(base_tau_ref, jitter_tau_ref)

        # Apply biologically plausible clipping
        self.v_rest = np.clip(self.v_rest, -80.0, -55.0)
        self.v_threshold = np.clip(self.v_threshold, self.v_rest + 5.0, -40.0)
        self.tau_m = np.clip(self.tau_m, 3.0, 30.0)
        self.tau_ref = np.clip(self.tau_ref, 1.0, 10.0)
        self.tau_e = np.clip(self.tau_e, 0.5, 10.0)
        self.tau_i = np.clip(self.tau_i, 2.0, 20.0)
        self.adaptation_increment = np.clip(self.adaptation_increment, 0.0, 1.0)
        self.tau_adaptation = np.clip(self.tau_adaptation, 20.0, 300.0)

        # Reversal potentials (constant per neuron)
        self.e_reversal_arr = np.full(n_neurons, e_reversal)
        self.i_reversal_arr = np.full(n_neurons, i_reversal)
        self.k_reversal = np.full(n_neurons, k_reversal)

        # Noise amplitudes
        self.v_noise_amp_arr = np.full(n_neurons, v_noise_amp)
        self.i_noise_amp_arr = np.full(n_neurons, i_noise_amp)

        # Store scalar versions for compatibility
        self.e_reversal = e_reversal
        self.i_reversal = i_reversal
        self.v_noise_amp = v_noise_amp
        self.i_noise_amp = i_noise_amp

        # ===== DYNAMIC STATE VARIABLES =====
        self.v = self.v_rest.copy()
        self.g_e = np.zeros(n_neurons)
        self.g_i = np.zeros(n_neurons)
        self.adaptation = np.zeros(n_neurons)
        self.t_since_spike = self.tau_ref + 1e-5  # Start past refractory

        # Store neuron params dict for compatibility with analysis code
        self.neuron_params = {
            'v_rest': self.v_rest.copy(),
            'v_threshold': self.v_threshold.copy(),
            'tau_m': self.tau_m.copy(),
            'tau_ref': self.tau_ref.copy(),
            'tau_e': self.tau_e.copy(),
            'tau_i': self.tau_i.copy(),
            'adaptation_increment': self.adaptation_increment.copy(),
            'tau_adaptation': self.tau_adaptation.copy(),
            'is_inhibitory': self.is_inhibitory.copy()
        }

        # ===== 3D SPATIAL POSITIONS =====
        self.neuron_3d_positions = {}
        if spatial:
            self._generate_3d_positions(layout)

        # Add nodes to graph
        for i in range(n_neurons):
            if spatial:
                x, y, z = self.neuron_3d_positions[i]
                self.graph.add_node(i, pos_3d=(x, y, z), is_inhibitory=self.is_inhibitory[i])
            else:
                self.graph.add_node(i, is_inhibitory=self.is_inhibitory[i])

        # ===== CONNECTION MATRICES =====
        self.weights = np.zeros((n_neurons, n_neurons))
        self.delays = np.zeros((n_neurons, n_neurons))

        # ===== ACTIVITY TRACKING =====
        self.network_activity = []
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        # ===== SPIKE BUFFER (time-binned for vectorized delivery) =====
        # Dict mapping timestep -> {'targets': array, 'weights': array}
        self._spike_buffer = {}

        # ===== DECAY FACTOR CACHE =====
        self._cached_dt = None
        self._exp_decay_e = None
        self._exp_decay_i = None
        self._exp_decay_adapt = None

        # Create connections
        self._create_connections_with_distance_weighting(
            weight_scale, weight_min, transmission_delay, distance_lambda, lambda_decay_ie
        )

        # Print statistics
        self._print_connection_stats()
        if any(v > 0 for v in self.jitter_params.values()):
            self._print_jitter_stats()

        # Create a fake neurons list for compatibility with simulation_utils
        # This provides .is_inhibitory access pattern
        self.neurons = _NeuronAccessor(self)

    def _jitter_gaussian(self, base, sigma, n):
        """Apply Gaussian jitter to a base value."""
        if sigma > 0:
            return base + np.random.normal(0, sigma, n)
        return np.full(n, base)

    def _jitter_lognormal(self, base, cv, n):
        """Apply log-normal jitter with coefficient of variation cv."""
        if cv > 0:
            sigma_log = np.sqrt(np.log(1 + cv**2))
            mu_log = np.log(base) - sigma_log**2 / 2
            return np.random.lognormal(mu_log, sigma_log, n)
        return np.full(n, base)

    def _jitter_lognormal_array(self, base_array, cv):
        """Apply log-normal jitter to an array of base values."""
        if cv > 0:
            sigma_log = np.sqrt(np.log(1 + cv**2))
            mu_log = np.log(base_array) - sigma_log**2 / 2
            return np.random.lognormal(mu_log, sigma_log)
        return base_array.copy()

    def _resolve_connection_probabilities(self, connection_p, connection_probabilities):
        """Resolve per-connection-type probabilities."""
        defaults = {
            'ee': connection_p,
            'ei': connection_p,
            'ie': connection_p,
            'ii': connection_p
        }
        if connection_probabilities is None:
            return defaults

        valid_keys = {'ee', 'ei', 'ie', 'ii'}
        for key in connection_probabilities:
            if key not in valid_keys:
                raise ValueError(f"Invalid connection type key: '{key}'")
            prob = connection_probabilities[key]
            if not 0 <= prob <= 1:
                raise ValueError(f"Connection probability for '{key}' must be between 0 and 1")

        resolved = defaults.copy()
        resolved.update(connection_probabilities)
        return resolved

    def _generate_3d_positions(self, layout='sphere'):
        """Generate 3D spatial positions for neurons."""
        if layout == 'sphere-surface':
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(self.n_neurons):
                theta = 2 * np.pi * i / golden_ratio
                phi = np.arccos(1 - 2 * (i + 0.5) / self.n_neurons)
                x = self.sphere_radius * np.sin(phi) * np.cos(theta)
                y = self.sphere_radius * np.sin(phi) * np.sin(theta)
                z = self.sphere_radius * np.cos(phi)
                self.neuron_3d_positions[i] = (x, y, z)
        else:
            # Uniform distribution inside sphere volume
            positions = []
            while len(positions) < self.n_neurons:
                x = np.random.uniform(-self.sphere_radius, self.sphere_radius)
                y = np.random.uniform(-self.sphere_radius, self.sphere_radius)
                z = np.random.uniform(-self.sphere_radius, self.sphere_radius)
                if x**2 + y**2 + z**2 <= self.sphere_radius**2:
                    positions.append((x, y, z))
            for i in range(self.n_neurons):
                self.neuron_3d_positions[i] = positions[i]

    def _create_connections_with_distance_weighting(self, weight_scale, weight_min,
                                                     transmission_delay, distance_lambda, lambda_decay_ie):
        """Create connections with distance-weighted strengths."""
        attempted_connections = 0
        nonzero_connections = 0

        self.connection_attempts_by_type = {'ee': 0, 'ei': 0, 'ie': 0, 'ii': 0}
        self.connection_counts_by_type = {'ee': 0, 'ei': 0, 'ie': 0, 'ii': 0}

        sphere_diameter = 2 * self.sphere_radius

        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i == j:
                    continue

                src_inhib = self.is_inhibitory[i]
                tgt_inhib = self.is_inhibitory[j]

                if src_inhib and tgt_inhib:
                    conn_type = 'ii'
                elif src_inhib:
                    conn_type = 'ie'
                elif tgt_inhib:
                    conn_type = 'ei'
                else:
                    conn_type = 'ee'

                connection_p = self.connection_probabilities[conn_type]

                if np.random.random() < connection_p:
                    attempted_connections += 1
                    self.connection_attempts_by_type[conn_type] += 1

                    # Calculate 3D distance
                    if self.neuron_3d_positions:
                        pos_i = self.neuron_3d_positions[i]
                        pos_j = self.neuron_3d_positions[j]
                        dist = np.sqrt((pos_i[0] - pos_j[0])**2 +
                                      (pos_i[1] - pos_j[1])**2 +
                                      (pos_i[2] - pos_j[2])**2)

                        is_ie_connection = src_inhib and not tgt_inhib
                        effective_lambda = lambda_decay_ie if is_ie_connection else distance_lambda
                        distance_factor = np.exp(-effective_lambda * dist)
                    else:
                        dist = 0
                        distance_factor = 1.0

                    # Determine weight
                    if src_inhib:
                        base_weight = -np.random.uniform(weight_min, weight_scale * 4.0)
                    else:
                        base_weight = np.random.uniform(weight_min, weight_scale)

                    weight = base_weight * distance_factor

                    # Calculate delay
                    if self.neuron_3d_positions:
                        delay = transmission_delay * (0.5 + 0.5 * dist / sphere_diameter)
                    else:
                        delay = transmission_delay

                    # Prune zero weights
                    if weight > 0.01 or weight < -0.01:
                        self.weights[i, j] = weight
                        self.delays[i, j] = delay
                        self.graph.add_edge(i, j, weight=weight, delay=delay,
                                           distance=dist, distance_factor=distance_factor)
                        nonzero_connections += 1
                        self.connection_counts_by_type[conn_type] += 1

        self.attempted_connections = attempted_connections
        self.nonzero_connections = nonzero_connections

        # Precompute outgoing connections for fast spike queueing
        self._precompute_outgoing_connections()

    def _precompute_outgoing_connections(self):
        """Precompute outgoing targets, weights, and delays for each neuron."""
        print("Precomputing outgoing connections for fast spike queueing...")
        self._outgoing_targets = []
        self._outgoing_weights = []
        self._outgoing_delays = []

        for i in range(self.n_neurons):
            targets = list(self.graph.successors(i))
            if targets:
                targets_arr = np.array(targets, dtype=np.int32)
                weights_arr = np.array([self.weights[i, j] for j in targets], dtype=np.float64)
                delays_arr = np.array([self.delays[i, j] for j in targets], dtype=np.float64)
            else:
                targets_arr = np.array([], dtype=np.int32)
                weights_arr = np.array([], dtype=np.float64)
                delays_arr = np.array([], dtype=np.float64)

            self._outgoing_targets.append(targets_arr)
            self._outgoing_weights.append(weights_arr)
            self._outgoing_delays.append(delays_arr)

        total_outgoing = sum(len(t) for t in self._outgoing_targets)
        print(f"Precomputed {total_outgoing} outgoing connections")

    def _print_connection_stats(self):
        """Print connection statistics."""
        total_possible = self.n_neurons * (self.n_neurons - 1)
        nonzero_count = np.count_nonzero(self.weights)

        n_excitatory = np.sum(~self.is_inhibitory)
        n_inhibitory = np.sum(self.is_inhibitory)

        possible_ee = n_excitatory * (n_excitatory - 1)
        possible_ei = n_excitatory * n_inhibitory
        possible_ie = n_inhibitory * n_excitatory
        possible_ii = n_inhibitory * (n_inhibitory - 1)

        print("\n===== 3D Spherical Network (Vectorized) Connection Statistics =====")
        print(f"Neurons: {self.n_neurons} (E: {n_excitatory}, I: {n_inhibitory})")
        print(f"Sphere radius: {self.sphere_radius:.2f}")
        print(f"Total possible connections: {total_possible}")
        print(f"Attempted connections: {self.attempted_connections}")
        print(f"Non-zero weight connections: {nonzero_count}")
        print(f"Connection density: {nonzero_count / total_possible:.6f}")
        print(f"Zero-weights pruned: {self.attempted_connections - nonzero_count}")

        print("\n--- Per-Type Connection Breakdown ---")
        print(f"{'Type':<6} {'Config P':<10} {'Actual P':<10} {'Expected':<12} {'Actual':<12} {'Exp-Act':<12} {'Match'}")

        type_labels = {'ee': 'E→E', 'ei': 'E→I', 'ie': 'I→E', 'ii': 'I→I'}
        possible_counts = {'ee': possible_ee, 'ei': possible_ei, 'ie': possible_ie, 'ii': possible_ii}

        for conn_type in ['ee', 'ei', 'ie', 'ii']:
            config_p = self.connection_probabilities[conn_type]
            possible = possible_counts[conn_type]
            expected = int(possible * config_p)
            actual = self.connection_counts_by_type[conn_type]
            actual_p = actual / possible if possible > 0 else 0
            diff = expected - actual
            tolerance = 0.05
            match = "yes" if abs(actual_p - config_p) < tolerance else "no"
            print(f"{type_labels[conn_type]:<6} {config_p:<10.3f} {actual_p:<10.3f} {expected:<12,} {actual:<12,} {diff:<12,} {match}")

        print("=====================================================================")

    def _print_jitter_stats(self):
        """Print jitter statistics."""
        print("\n===== Neuron Parameter Jitter Statistics =====")
        print(f"{'Parameter':<25} {'Config':<10} {'Actual μ':<12} {'Actual σ':<12} {'Min':<10} {'Max':<10}")
        print("-" * 85)

        for param_name in ['v_rest', 'v_threshold', 'tau_m', 'tau_ref', 'tau_e', 'tau_i',
                          'adaptation_increment', 'tau_adaptation']:
            values = self.neuron_params[param_name]
            configured = self.jitter_params[param_name]
            actual_mean = np.mean(values)
            actual_std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{param_name:<25} {configured:<10.3f} {actual_mean:<12.2f} {actual_std:<12.2f} {min_val:<10.2f} {max_val:<10.2f}")

        print("=" * 85)

    def plot_neuron_parameter_distributions(self, save_path="neuron_parameter_distributions.png", darkstyle=True):
        """
        Generate histograms showing the distribution of jittered neuron parameters.
        """
        import matplotlib.pyplot as plt

        if darkstyle:
            plt.style.use('dark_background')

        v_rest_vals = self.neuron_params['v_rest']
        v_threshold_vals = self.neuron_params['v_threshold']
        tau_m_vals = self.neuron_params['tau_m']
        tau_ref_vals = self.neuron_params['tau_ref']
        tau_e_vals = self.neuron_params['tau_e']
        tau_i_vals = self.neuron_params['tau_i']
        adaptation_inc_vals = self.neuron_params['adaptation_increment']
        tau_adapt_vals = self.neuron_params['tau_adaptation']
        is_inhibitory = self.neuron_params['is_inhibitory']

        tau_ref_exc = tau_ref_vals[~is_inhibitory]
        tau_ref_inh = tau_ref_vals[is_inhibitory]

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle('Neuron Parameter Distributions (with Jitter)', fontsize=16, y=0.995)

        exc_color = '#ff6b6b'
        inh_color = '#4dabf7'
        mixed_color = '#69db7c'

        def add_stats(ax, vals, color='white'):
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8)
            ax.text(0.97, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nn = {len(vals)}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right', color=color,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax = axes[0, 0]
        ax.hist(v_rest_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('V_rest (mV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Resting Potential', fontsize=12)
        add_stats(ax, v_rest_vals)

        ax = axes[0, 1]
        ax.hist(v_threshold_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('V_threshold (mV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Spike Threshold', fontsize=12)
        add_stats(ax, v_threshold_vals)

        ax = axes[1, 0]
        ax.hist(tau_m_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_m (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Membrane Time Constant', fontsize=12)
        add_stats(ax, tau_m_vals)

        ax = axes[1, 1]
        if len(tau_ref_exc) > 0:
            ax.hist(tau_ref_exc, bins=20, color=exc_color, alpha=0.6, edgecolor='white', linewidth=0.5, label=f'Exc (n={len(tau_ref_exc)})')
        if len(tau_ref_inh) > 0:
            ax.hist(tau_ref_inh, bins=20, color=inh_color, alpha=0.6, edgecolor='white', linewidth=0.5, label=f'Inh (n={len(tau_ref_inh)})')
        ax.set_xlabel('τ_ref (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Refractory Period (E vs I)', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        if len(tau_ref_exc) > 0 and len(tau_ref_inh) > 0:
            ax.text(0.97, 0.72, f'E: μ={np.mean(tau_ref_exc):.2f}, σ={np.std(tau_ref_exc):.2f}\nI: μ={np.mean(tau_ref_inh):.2f}, σ={np.std(tau_ref_inh):.2f}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   horizontalalignment='right', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax = axes[2, 0]
        ax.hist(tau_e_vals, bins=30, color=exc_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_e (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Excitatory Synaptic Time Constant', fontsize=12)
        add_stats(ax, tau_e_vals, color=exc_color)

        ax = axes[2, 1]
        ax.hist(tau_i_vals, bins=30, color=inh_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_i (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Inhibitory Synaptic Time Constant', fontsize=12)
        add_stats(ax, tau_i_vals, color=inh_color)

        ax = axes[3, 0]
        ax.hist(adaptation_inc_vals, bins=30, color='#ffd43b', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Adaptation Increment', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Spike-Frequency Adaptation Increment', fontsize=12)
        add_stats(ax, adaptation_inc_vals, color='#ffd43b')

        ax = axes[3, 1]
        ax.hist(tau_adapt_vals, bins=30, color='#ffd43b', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_adaptation (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Adaptation Time Constant', fontsize=12)
        add_stats(ax, tau_adapt_vals, color='#ffd43b')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"Neuron parameter distributions saved to: {save_path}")

    def plot_connection_type_distribution(self, save_path="connection_type_distribution.png"):
        """Generate a bar chart comparing configured vs actual connection probabilities."""
        import matplotlib.pyplot as plt

        n_excitatory = int(np.sum(~self.is_inhibitory))
        n_inhibitory = int(np.sum(self.is_inhibitory))

        possible_counts = {
            'ee': n_excitatory * (n_excitatory - 1),
            'ei': n_excitatory * n_inhibitory,
            'ie': n_inhibitory * n_excitatory,
            'ii': n_inhibitory * (n_inhibitory - 1)
        }

        conn_types = ['ee', 'ei', 'ie', 'ii']
        type_labels = ['E→E', 'E→I', 'I→E', 'I→I']

        config_probs = [self.connection_probabilities[ct] for ct in conn_types]
        actual_probs = [
            self.connection_counts_by_type[ct] / possible_counts[ct]
            if possible_counts[ct] > 0 else 0
            for ct in conn_types
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(type_labels))
        width = 0.35

        bars1 = ax1.bar(x - width/2, config_probs, width, label='Configured', color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, actual_probs, width, label='Actual', color='darkorange', alpha=0.8)

        ax1.set_xlabel('Connection Type', fontsize=12)
        ax1.set_ylabel('Connection Probability', fontsize=12)
        ax1.set_title('Configured vs Actual Connection Probabilities', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(type_labels, fontsize=11)
        ax1.legend(fontsize=10)
        ax1.set_ylim(0, max(max(config_probs), max(actual_probs)) * 1.2)

        for bar, val in zip(bars1, config_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, actual_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        expected_counts = [int(possible_counts[ct] * self.connection_probabilities[ct]) for ct in conn_types]
        actual_counts = [self.connection_counts_by_type[ct] for ct in conn_types]

        bars3 = ax2.bar(x - width/2, expected_counts, width, label='Expected', color='steelblue', alpha=0.8)
        bars4 = ax2.bar(x + width/2, actual_counts, width, label='Actual', color='darkorange', alpha=0.8)

        ax2.set_xlabel('Connection Type', fontsize=12)
        ax2.set_ylabel('Number of Connections', fontsize=12)
        ax2.set_title('Expected vs Actual Connection Counts', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(type_labels, fontsize=11)
        ax2.legend(fontsize=10)

        for bar, val in zip(bars3, expected_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(expected_counts)*0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=8, rotation=45)
        for bar, val in zip(bars4, actual_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(expected_counts)*0.02,
                    f'{val:,}', ha='center', va='bottom', fontsize=8, rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Connection type distribution saved to: {save_path}")

    def _precompute_decay_factors(self, dt):
        """Precompute exponential decay factors for given dt."""
        self._exp_decay_e = np.exp(-dt / np.maximum(self.tau_e, 1e-9))
        self._exp_decay_i = np.exp(-dt / np.maximum(self.tau_i, 1e-9))
        self._exp_decay_adapt = np.exp(-dt / np.maximum(self.tau_adaptation, 1e-9))
        self._cached_dt = dt

    def update_network(self, dt):
        """
        Update the entire network for one time step (VECTORIZED).

        Returns indices of neurons that spiked.
        """
        # Precompute decay factors if needed
        if self._cached_dt != dt:
            self._precompute_decay_factors(dt)

        # ===== 1. PROCESS SPIKE QUEUE - deliver due spikes (VECTORIZED) =====
        current_step = len(self.network_activity)

        # Check if there are spikes to deliver at this timestep
        if current_step in self._spike_buffer:
            # Get all spikes for this timestep
            spike_data = self._spike_buffer.pop(current_step)
            targets = spike_data['targets']
            weights = spike_data['weights']

            # Vectorized delivery using np.add.at
            exc_mask = weights > 0
            inh_mask = weights < 0

            if np.any(exc_mask):
                np.add.at(self.g_e, targets[exc_mask], weights[exc_mask])
            if np.any(inh_mask):
                np.add.at(self.g_i, targets[inh_mask], -weights[inh_mask])

        # ===== 2. IDENTIFY REFRACTORY VS NON-REFRACTORY NEURONS =====
        refractory_mask = self.t_since_spike < self.tau_ref
        non_refractory_mask = ~refractory_mask

        # Update time since spike for all neurons
        self.t_since_spike += dt

        # ===== 3. UPDATE NON-REFRACTORY NEURONS (VECTORIZED) =====
        if np.any(non_refractory_mask):
            # Synaptic currents with reversal potentials
            v_nr = self.v[non_refractory_mask]
            i_e = self.g_e[non_refractory_mask] * (self.e_reversal_arr[non_refractory_mask] - v_nr)
            i_i = self.g_i[non_refractory_mask] * (self.i_reversal_arr[non_refractory_mask] - v_nr)

            # Adaptation current with potassium reversal
            i_adapt = self.adaptation[non_refractory_mask] * (self.k_reversal[non_refractory_mask] - v_nr)

            # Membrane potential update: dv = dt * ((-(v - v_rest)/tau_m) + i_e + i_i + i_adapt)
            dv = dt * ((-(v_nr - self.v_rest[non_refractory_mask]) / self.tau_m[non_refractory_mask])
                       + i_e + i_i + i_adapt)

            # Add membrane noise (vectorized)
            v_noise = np.random.normal(0, self.v_noise_amp_arr[non_refractory_mask])

            self.v[non_refractory_mask] += dv + v_noise

        # ===== 4. CLAMP REFRACTORY NEURONS =====
        self.v[refractory_mask] = self.v_reset[refractory_mask]

        # ===== 5. DECAY CONDUCTANCES (VECTORIZED) =====
        self.g_e *= self._exp_decay_e
        self.g_i *= self._exp_decay_i
        self.adaptation *= self._exp_decay_adapt

        # ===== 6. ADD SYNAPTIC NOISE (VECTORIZED) =====
        # Generate noise for all neurons at once
        e_noise = np.random.normal(0, self.i_noise_amp_arr)
        i_noise = np.random.normal(0, self.i_noise_amp_arr)
        # Only add positive noise (rectified)
        self.g_e += np.maximum(e_noise, 0)
        self.g_i += np.maximum(i_noise, 0)

        # ===== 7. SPIKE DETECTION (VECTORIZED) =====
        spiked_mask = (self.v >= self.v_threshold) & non_refractory_mask
        active_indices = np.where(spiked_mask)[0]

        # ===== 8. POST-SPIKE UPDATES (VECTORIZED) =====
        if active_indices.size > 0:
            self.v[spiked_mask] = self.v_reset[spiked_mask]
            self.t_since_spike[spiked_mask] = 0.0
            self.adaptation[spiked_mask] += self.adaptation_increment[spiked_mask]

        # ===== 9. TRACK NETWORK ACTIVITY =====
        self.network_activity.append(len(active_indices))
        activity_level = len(active_indices)

        # Avalanche tracking
        if activity_level > 0 and self.current_avalanche_start is None:
            self.current_avalanche_start = len(self.network_activity) - 1
            self.current_avalanche_size = activity_level
        elif activity_level > 0 and self.current_avalanche_start is not None:
            self.current_avalanche_size += activity_level
        elif activity_level == 0 and self.current_avalanche_start is not None:
            duration = len(self.network_activity) - 1 - self.current_avalanche_start
            self.avalanche_sizes.append(self.current_avalanche_size)
            self.avalanche_durations.append(duration)
            self.current_avalanche_start = None
            self.current_avalanche_size = 0

        # ===== 10. QUEUE NEW SPIKES (FULLY VECTORIZED) =====
        if active_indices.size > 0:
            # Collect all targets, weights, delays for spiking neurons
            all_targets = []
            all_weights = []
            all_delays = []

            for i in active_indices:
                targets = self._outgoing_targets[i]
                if len(targets) > 0:
                    all_targets.append(targets)
                    all_weights.append(self._outgoing_weights[i])
                    all_delays.append(self._outgoing_delays[i])

            if all_targets:
                # Concatenate into single arrays
                all_targets = np.concatenate(all_targets)
                all_weights = np.concatenate(all_weights)
                all_delays = np.concatenate(all_delays)

                # Convert delays to delivery timesteps
                delivery_steps = current_step + np.round(all_delays / dt).astype(np.int64)

                # Group by delivery timestep and store in buffer
                unique_steps = np.unique(delivery_steps)
                for step in unique_steps:
                    mask = delivery_steps == step
                    step_targets = all_targets[mask]
                    step_weights = all_weights[mask]

                    if step in self._spike_buffer:
                        # Append to existing
                        self._spike_buffer[step]['targets'] = np.concatenate([
                            self._spike_buffer[step]['targets'], step_targets
                        ])
                        self._spike_buffer[step]['weights'] = np.concatenate([
                            self._spike_buffer[step]['weights'], step_weights
                        ])
                    else:
                        self._spike_buffer[step] = {
                            'targets': step_targets,
                            'weights': step_weights
                        }

        return list(active_indices)

    def reset_all(self):
        """Reset all neurons to initial state (VECTORIZED)."""
        self.v[:] = self.v_rest
        self.g_e.fill(0.0)
        self.g_i.fill(0.0)
        self.adaptation.fill(0.0)
        self.t_since_spike[:] = self.tau_ref + 1e-5

        self.network_activity = []
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        self._spike_buffer = {}

    def stimulate_neuron(self, idx, current):
        """Stimulate a specific neuron with given current."""
        if 0 <= idx < self.n_neurons:
            if current > 0:
                self.g_e[idx] += current
            else:
                self.g_i[idx] += -current

    def stimulate_neurons(self, indices, currents):
        """Stimulate multiple neurons efficiently (VECTORIZED)."""
        indices = np.asarray(indices)
        currents = np.asarray(currents)

        exc_mask = currents > 0
        inh_mask = currents < 0

        if np.any(exc_mask):
            np.add.at(self.g_e, indices[exc_mask], currents[exc_mask])
        if np.any(inh_mask):
            np.add.at(self.g_i, indices[inh_mask], -currents[inh_mask])

    def stimulate_random(self, n_stimulated=1, current=10.0):
        """Stimulate random neurons."""
        if n_stimulated > self.n_neurons:
            n_stimulated = self.n_neurons

        indices = np.random.choice(self.n_neurons, size=n_stimulated, replace=False)
        for idx in indices:
            self.stimulate_neuron(idx, current)


class _NeuronAccessor:
    """
    Provides .neurons[idx].is_inhibitory access pattern for compatibility.
    """
    def __init__(self, network):
        self._network = network

    def __getitem__(self, idx):
        return _NeuronProxy(self._network, idx)

    def __len__(self):
        return self._network.n_neurons

    def __iter__(self):
        for i in range(self._network.n_neurons):
            yield _NeuronProxy(self._network, i)


class _NeuronProxy:
    """
    Proxy object that provides attribute access to vectorized neuron state.
    """
    def __init__(self, network, idx):
        self._network = network
        self._idx = idx

    @property
    def is_inhibitory(self):
        return self._network.is_inhibitory[self._idx]

    @property
    def v(self):
        return self._network.v[self._idx]

    @property
    def g_e(self):
        return self._network.g_e[self._idx]

    @property
    def g_i(self):
        return self._network.g_i[self._idx]

    @property
    def v_rest(self):
        return self._network.v_rest[self._idx]

    @property
    def v_threshold(self):
        return self._network.v_threshold[self._idx]

    @property
    def v_reset(self):
        return self._network.v_reset[self._idx]

    @property
    def tau_m(self):
        return self._network.tau_m[self._idx]

    @property
    def tau_ref(self):
        return self._network.tau_ref[self._idx]

    @property
    def tau_e(self):
        return self._network.tau_e[self._idx]

    @property
    def tau_i(self):
        return self._network.tau_i[self._idx]

    @property
    def e_reversal(self):
        return self._network.e_reversal_arr[self._idx]

    @property
    def i_reversal(self):
        return self._network.i_reversal_arr[self._idx]

    @property
    def k_reversal(self):
        return self._network.k_reversal[self._idx]

    @property
    def v_noise_amp(self):
        return self._network.v_noise_amp_arr[self._idx]

    @property
    def i_noise_amp(self):
        return self._network.i_noise_amp_arr[self._idx]

    @property
    def adaptation_increment(self):
        return self._network.adaptation_increment[self._idx]

    @property
    def tau_adaptation(self):
        return self._network.tau_adaptation[self._idx]
