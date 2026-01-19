import numpy as np
import networkx as nx
from collections import deque
from .LIFNeuronWithReversal import LIFNeuronWithReversal

class SphericalNeuronalNetwork:
    """
    A 3D spherical version of the CircularNeuronalNetwork class that:
    1. Positions neurons in 3D space within a sphere volume
    2. Uses 3D Euclidean distance for connection weight decay
    3. Supports reversal potentials for biologically plausible modeling
    4. Includes avalanche detection and analysis capabilities
    5. Uses transmission delays for spike propagation
    6. Prunes zero-weight connections for efficiency

    This provides a comprehensive 3D neural simulation framework.
    """
    def __init__(self, n_neurons=100, connection_p=0.1, connection_probabilities=None,
            weight_scale=1.0, weight_min=0.0, spatial=True, transmission_delay=1.0,
            inhibitory_fraction=0.2, layout='sphere',
            v_noise_amp=0.3, i_noise_amp=0.05,
            e_reversal=0.0, i_reversal=-80.0, distance_lambda=1,
            lambda_decay_ie=0.05,
            # Neuron parameter jitter (Gaussian std dev, 0 = no jitter)
            jitter_v_rest=0.0, jitter_v_threshold=0.0, jitter_tau_m=0.0,
            jitter_tau_ref=0.0, jitter_tau_e=0.0, jitter_tau_i=0.0,
            jitter_adaptation_increment=0.0, jitter_tau_adaptation=0.0):
        """
        Initialize the 3D spherical network with the given parameters.

        Parameters:
        -----------
        n_neurons : int
            Number of neurons in the network
        connection_p : float
            Connection probability between neurons (0-1)
        weight_scale : float
            Scale factor for synaptic weights
        weight_min : float
            Minimum weight value
        spatial : bool
            Whether to use spatial organization
        transmission_delay : float
            Base transmission delay in ms
        inhibitory_fraction : float
            Fraction of neurons that are inhibitory (0-1)
        layout : str
            Spatial layout: 'sphere' or 'sphere-surface'
        v_noise_amp : float
            Amplitude of membrane potential noise (mV)
        i_noise_amp : float
            Amplitude of synaptic current noise
        e_reversal : float
            Excitatory reversal potential (mV)
        i_reversal : float
            Inhibitory reversal potential (mV)
        distance_lambda : float
            Distance decay constant for synaptic weights (higher values mean faster decay)
        lambda_decay_ie : float
            Distance decay constant specifically for inhibitory→excitatory connections
            (default 0.05, meaning slower decay / longer range for I→E connections)
        connection_probabilities : dict or None
            Per-connection-type probabilities. If provided, overrides connection_p.
            Expected keys: 'ee', 'ei', 'ie', 'ii' (all optional).
            Missing keys default to connection_p value.
            Example: {'ee': 0.1, 'ei': 0.15, 'ie': 0.4, 'ii': 0.15}
            Biological defaults: E→E ~0.1, E→I ~0.15-0.20, I→E ~0.4-0.5, I→I ~0.1-0.2
        jitter_v_rest : float
            Gaussian std dev for resting potential jitter (mV). Default 0 (no jitter).
        jitter_v_threshold : float
            Gaussian std dev for threshold potential jitter (mV). Default 0 (no jitter).
        jitter_tau_m : float
            Coefficient of variation (CV) for membrane time constant (log-normal).
            CV = std/mean, e.g., 0.3 means ~30% variation. Default 0 (no jitter).
        jitter_tau_ref : float
            Coefficient of variation (CV) for refractory period (log-normal).
            Default 0 (no jitter).
        jitter_tau_e : float
            Coefficient of variation (CV) for excitatory synaptic time constant (log-normal).
            Default 0 (no jitter).
        jitter_tau_i : float
            Coefficient of variation (CV) for inhibitory synaptic time constant (log-normal).
            Default 0 (no jitter).
        jitter_adaptation_increment : float
            Coefficient of variation (CV) for adaptation increment (log-normal).
            Default 0 (no jitter).
        jitter_tau_adaptation : float
            Coefficient of variation (CV) for adaptation time constant (log-normal).
            Default 0 (no jitter).
        """
        # Store distance decay parameters
        self.distance_lambda = distance_lambda
        self.lambda_decay_ie = lambda_decay_ie
        self.n_neurons = n_neurons
        self.weight_scale = weight_scale

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

        # Calculate sphere radius based on volume scaling
        # Volume of sphere = (4/3) * pi * r^3
        # For n_neurons to fit with similar density: r = (3*n/(4*pi))^(1/3) * scale_factor
        self.sphere_radius = (3 * n_neurons / (4 * np.pi)) ** (1/3) * 1.5

        # Create graph - only connections with non-zero weights will be added
        self.graph = nx.DiGraph()

        # Create neurons
        self.neurons = []
        self.neuron_3d_positions = {}

        # Generate spatial positions if needed
        if spatial:
            self._generate_3d_positions(layout)

        # Initialize arrays to store jittered parameter values for all neurons
        self.neuron_params = {
            'v_rest': np.zeros(n_neurons),
            'v_threshold': np.zeros(n_neurons),
            'tau_m': np.zeros(n_neurons),
            'tau_ref': np.zeros(n_neurons),
            'tau_e': np.zeros(n_neurons),
            'tau_i': np.zeros(n_neurons),
            'adaptation_increment': np.zeros(n_neurons),
            'tau_adaptation': np.zeros(n_neurons),
            'is_inhibitory': np.zeros(n_neurons, dtype=bool)
        }

        # Create neurons and add to graph
        for i in range(n_neurons):
            # Determine if inhibitory based on fraction
            is_inhibitory = np.random.rand() < inhibitory_fraction

            # Apply Gaussian jitter to neuron parameters with biologically plausible clipping
            # Base values (LIFNeuronWithReversal defaults)
            base_v_rest = -65.0
            base_v_threshold = -55.0
            base_tau_m = 10.0
            base_tau_ref_exc = 4.0
            base_tau_ref_inh = 2.5
            base_tau_e = 3.0
            base_tau_i = 7.0
            base_adaptation_increment = 0.2
            base_tau_adaptation = 100.0

            # Generate jittered values
            # Voltage parameters use Gaussian (additive) jitter - can be negative relative to baseline
            jittered_v_rest = base_v_rest + np.random.normal(0, self.jitter_params['v_rest']) if self.jitter_params['v_rest'] > 0 else base_v_rest
            jittered_v_threshold = base_v_threshold + np.random.normal(0, self.jitter_params['v_threshold']) if self.jitter_params['v_threshold'] > 0 else base_v_threshold

            # Time constants use log-normal (multiplicative) jitter - naturally bounded > 0
            # For log-normal: if X ~ LogNormal(mu, sigma), then median(X) = exp(mu) and CV ≈ sigma
            # We want median = base_value, so mu = log(base_value)
            # The jitter param is interpreted as coefficient of variation (CV = sigma/mean)
            if self.jitter_params['tau_m'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['tau_m']**2))  # Convert CV to log-space sigma
                mu_log = np.log(base_tau_m) - sigma_log**2 / 2  # Adjust mu so mean = base_value
                jittered_tau_m = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_tau_m = base_tau_m

            if self.jitter_params['tau_e'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['tau_e']**2))
                mu_log = np.log(base_tau_e) - sigma_log**2 / 2
                jittered_tau_e = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_tau_e = base_tau_e

            if self.jitter_params['tau_i'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['tau_i']**2))
                mu_log = np.log(base_tau_i) - sigma_log**2 / 2
                jittered_tau_i = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_tau_i = base_tau_i

            if self.jitter_params['tau_adaptation'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['tau_adaptation']**2))
                mu_log = np.log(base_tau_adaptation) - sigma_log**2 / 2
                jittered_tau_adaptation = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_tau_adaptation = base_tau_adaptation

            if self.jitter_params['adaptation_increment'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['adaptation_increment']**2))
                mu_log = np.log(base_adaptation_increment) - sigma_log**2 / 2
                jittered_adaptation_increment = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_adaptation_increment = base_adaptation_increment

            # Jitter refractory period (different base for E vs I neurons) - also log-normal
            base_tau_ref = base_tau_ref_inh if is_inhibitory else base_tau_ref_exc
            if self.jitter_params['tau_ref'] > 0:
                sigma_log = np.sqrt(np.log(1 + self.jitter_params['tau_ref']**2))
                mu_log = np.log(base_tau_ref) - sigma_log**2 / 2
                jittered_tau_ref = np.random.lognormal(mu_log, sigma_log)
            else:
                jittered_tau_ref = base_tau_ref

            # Apply biologically plausible clipping
            jittered_v_rest = np.clip(jittered_v_rest, -80.0, -55.0)  # Rest can't be above threshold range
            jittered_v_threshold = np.clip(jittered_v_threshold, jittered_v_rest + 5.0, -40.0)  # Must be above rest
            jittered_tau_m = np.clip(jittered_tau_m, 3.0, 30.0)  # Biological range ~5-30ms
            jittered_tau_ref = np.clip(jittered_tau_ref, 1.0, 10.0)  # Biological range ~1-10ms
            jittered_tau_e = np.clip(jittered_tau_e, 0.5, 10.0)  # AMPA ~1-5ms, some slower
            jittered_tau_i = np.clip(jittered_tau_i, 2.0, 20.0)  # GABA_A ~5-15ms
            jittered_adaptation_increment = np.clip(jittered_adaptation_increment, 0.0, 1.0)  # Must be non-negative
            jittered_tau_adaptation = np.clip(jittered_tau_adaptation, 20.0, 300.0)  # Biological range ~50-200ms

            # Create neuron with jittered parameters
            neuron = LIFNeuronWithReversal(
                is_inhibitory=is_inhibitory,
                e_reversal=e_reversal,
                i_reversal=i_reversal,
                v_noise_amp=v_noise_amp,
                i_noise_amp=i_noise_amp,
                v_rest=jittered_v_rest,
                v_threshold=jittered_v_threshold,
                tau_m=jittered_tau_m,
                tau_ref=jittered_tau_ref,
                tau_e=jittered_tau_e,
                tau_i=jittered_tau_i,
                adaptation_increment=jittered_adaptation_increment,
                tau_adaptation=jittered_tau_adaptation,
            )

            self.neurons.append(neuron)

            # Store jittered parameter values for analysis
            self.neuron_params['v_rest'][i] = jittered_v_rest
            self.neuron_params['v_threshold'][i] = jittered_v_threshold
            self.neuron_params['tau_m'][i] = jittered_tau_m
            self.neuron_params['tau_ref'][i] = jittered_tau_ref
            self.neuron_params['tau_e'][i] = jittered_tau_e
            self.neuron_params['tau_i'][i] = jittered_tau_i
            self.neuron_params['adaptation_increment'][i] = jittered_adaptation_increment
            self.neuron_params['tau_adaptation'][i] = jittered_tau_adaptation
            self.neuron_params['is_inhibitory'][i] = is_inhibitory

            # Add to graph with position and properties
            if spatial:
                x, y, z = self.neuron_3d_positions[i]
                self.graph.add_node(i, pos_3d=(x, y, z), is_inhibitory=is_inhibitory)
            else:
                self.graph.add_node(i, is_inhibitory=is_inhibitory)

        # Initialize connection matrices - these will only store non-zero weights
        self.weights = np.zeros((n_neurons, n_neurons))
        self.delays = np.zeros((n_neurons, n_neurons))

        # Variables for tracking activity
        self.network_activity = []

        # Avalanche tracking
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        # Store noise parameters
        self.v_noise_amp = v_noise_amp
        self.i_noise_amp = i_noise_amp

        # Store reversal potentials
        self.e_reversal = e_reversal
        self.i_reversal = i_reversal

        # Queue for spikes with delay
        self.spike_queue = deque()

        # Create connections and prune zero weights automatically
        self._create_connections_with_distance_weighting(weight_scale, weight_min,
                                                         transmission_delay, distance_lambda, lambda_decay_ie)

        # Print connection statistics
        self._print_connection_stats()

        # Print jitter statistics if any jitter was applied
        if any(v > 0 for v in self.jitter_params.values()):
            self._print_jitter_stats()

    def _print_jitter_stats(self):
        """Print statistics about neuron parameter jitter."""
        print("\n===== Neuron Parameter Jitter Statistics =====")
        print("Voltage params use Gaussian (σ in mV), time constants use Log-Normal (CV = σ/μ)")
        print(f"{'Parameter':<25} {'Config':<10} {'Type':<8} {'Actual μ':<12} {'Actual σ':<12} {'Actual CV':<10} {'Min':<10} {'Max':<10}")
        print("-" * 105)

        # Define base values, units, and jitter type for reference
        param_info = {
            'v_rest': {'base': -65.0, 'unit': 'mV', 'type': 'gauss'},
            'v_threshold': {'base': -55.0, 'unit': 'mV', 'type': 'gauss'},
            'tau_m': {'base': 10.0, 'unit': 'ms', 'type': 'lognorm'},
            'tau_ref': {'base': '4.0/2.5', 'unit': 'ms', 'type': 'lognorm'},
            'tau_e': {'base': 3.0, 'unit': 'ms', 'type': 'lognorm'},
            'tau_i': {'base': 7.0, 'unit': 'ms', 'type': 'lognorm'},
            'adaptation_increment': {'base': 0.2, 'unit': '', 'type': 'lognorm'},
            'tau_adaptation': {'base': 100.0, 'unit': 'ms', 'type': 'lognorm'}
        }

        for param_name in ['v_rest', 'v_threshold', 'tau_m', 'tau_ref', 'tau_e', 'tau_i',
                          'adaptation_increment', 'tau_adaptation']:
            values = self.neuron_params[param_name]
            configured = self.jitter_params[param_name]
            actual_mean = np.mean(values)
            actual_std = np.std(values)
            actual_cv = actual_std / abs(actual_mean) if actual_mean != 0 else 0
            min_val = np.min(values)
            max_val = np.max(values)
            jitter_type = param_info[param_name]['type']

            print(f"{param_name:<25} {configured:<10.3f} {jitter_type:<8} {actual_mean:<12.2f} {actual_std:<12.2f} {actual_cv:<10.3f} {min_val:<10.2f} {max_val:<10.2f}")

        print("=" * 105)

    def _resolve_connection_probabilities(self, connection_p, connection_probabilities):
        """
        Resolve per-connection-type probabilities with backward compatibility.

        Parameters:
        -----------
        connection_p : float
            Default connection probability (used as fallback)
        connection_probabilities : dict or None
            Per-type probabilities dict with keys 'ee', 'ei', 'ie', 'ii'

        Returns:
        --------
        dict
            Resolved probabilities for all four connection types
        """
        # Default: use single probability for all types
        defaults = {
            'ee': connection_p,  # Excitatory → Excitatory
            'ei': connection_p,  # Excitatory → Inhibitory
            'ie': connection_p,  # Inhibitory → Excitatory
            'ii': connection_p   # Inhibitory → Inhibitory
        }

        if connection_probabilities is None:
            return defaults

        # Validate and merge with defaults
        valid_keys = {'ee', 'ei', 'ie', 'ii'}
        for key in connection_probabilities:
            if key not in valid_keys:
                raise ValueError(f"Invalid connection type key: '{key}'. "
                               f"Valid keys are: {valid_keys}")
            prob = connection_probabilities[key]
            if not 0 <= prob <= 1:
                raise ValueError(f"Connection probability for '{key}' must be "
                               f"between 0 and 1, got {prob}")

        # Merge: user values override defaults
        resolved = defaults.copy()
        resolved.update(connection_probabilities)

        return resolved

    def _generate_3d_positions(self, layout='sphere'):
        """
        Generate 3D spatial positions for neurons within or on a sphere.

        Parameters:
        -----------
        layout : str
            Layout type: 'sphere' or 'sphere-filled' (volume), 'sphere-surface' (surface only)
        """
        if layout == 'sphere-surface':
            # Create positions uniformly on the sphere surface using Fibonacci spiral
            positions = []
            golden_ratio = (1 + np.sqrt(5)) / 2

            for i in range(self.n_neurons):
                theta = 2 * np.pi * i / golden_ratio
                phi = np.arccos(1 - 2 * (i + 0.5) / self.n_neurons)

                x = self.sphere_radius * np.sin(phi) * np.cos(theta)
                y = self.sphere_radius * np.sin(phi) * np.sin(theta)
                z = self.sphere_radius * np.cos(phi)

                positions.append((x, y, z))

            # Assign positions to neurons
            for i in range(self.n_neurons):
                self.neuron_3d_positions[i] = positions[i]

        else:  # 'sphere' or 'sphere-filled' - uniform distribution inside sphere volume
            # Use rejection sampling to create uniform distribution in sphere
            positions = []
            while len(positions) < self.n_neurons:
                # Generate random position in cube that bounds the sphere
                x = np.random.uniform(-self.sphere_radius, self.sphere_radius)
                y = np.random.uniform(-self.sphere_radius, self.sphere_radius)
                z = np.random.uniform(-self.sphere_radius, self.sphere_radius)

                # Check if point is inside sphere
                if x**2 + y**2 + z**2 <= self.sphere_radius**2:
                    positions.append((x, y, z))

            # Assign positions to neurons
            for i in range(self.n_neurons):
                self.neuron_3d_positions[i] = positions[i]

    def _create_connections_with_distance_weighting(self, weight_scale, weight_min, transmission_delay, distance_lambda=0.2, lambda_decay_ie=0.05):
        """
        Create random connections between neurons with weights that decay with 3D distance.
        Zero-weight connections are automatically pruned.
        Uses per-connection-type probabilities from self.connection_probabilities.

        Parameters:
        -----------
        weight_scale : float
            Scale factor for synaptic weights
        weight_min : float
            Minimum weight value
        transmission_delay : float
            Base transmission delay in ms
        distance_lambda : float
            Distance decay constant (higher values mean faster decay)
        lambda_decay_ie : float
            Distance decay constant specifically for I→E connections (default 0.05)
        """
        # Keep track of connection statistics
        attempted_connections = 0
        nonzero_connections = 0

        # Track per-type connection attempts and successes
        self.connection_attempts_by_type = {'ee': 0, 'ei': 0, 'ie': 0, 'ii': 0}
        self.connection_counts_by_type = {'ee': 0, 'ei': 0, 'ie': 0, 'ii': 0}

        # Sphere diameter for delay scaling
        sphere_diameter = 2 * self.sphere_radius

        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i == j:
                    continue

                # Determine connection type based on source and target neuron types
                src_inhib = self.neurons[i].is_inhibitory
                tgt_inhib = self.neurons[j].is_inhibitory

                if src_inhib and tgt_inhib:
                    conn_type = 'ii'
                elif src_inhib:
                    conn_type = 'ie'
                elif tgt_inhib:
                    conn_type = 'ei'
                else:
                    conn_type = 'ee'

                # Get connection probability for this type
                connection_p = self.connection_probabilities[conn_type]

                if np.random.random() < connection_p:
                    attempted_connections += 1
                    self.connection_attempts_by_type[conn_type] += 1

                    # Calculate 3D distance between neurons
                    if hasattr(self, 'neuron_3d_positions'):
                        pos_i = self.neuron_3d_positions[i]
                        pos_j = self.neuron_3d_positions[j]
                        dist = np.sqrt((pos_i[0] - pos_j[0])**2 +
                                      (pos_i[1] - pos_j[1])**2 +
                                      (pos_i[2] - pos_j[2])**2)

                        # Apply distance-based weight scaling
                        # Use lambda_decay_ie for inhibitory→excitatory connections
                        is_ie_connection = self.neurons[i].is_inhibitory and not self.neurons[j].is_inhibitory
                        effective_lambda = lambda_decay_ie if is_ie_connection else distance_lambda
                        distance_factor = np.exp(-effective_lambda * dist)
                    else:
                        dist = 0
                        distance_factor = 1.0

                    # Determine weight based on neuron types and distance
                    if self.neurons[i].is_inhibitory:
                        # Inhibitory neurons have stronger negative weights
                        base_weight = -np.random.uniform(weight_min, weight_scale * 4.0)
                    else:
                        # Excitatory neurons have weaker positive weights
                        base_weight = np.random.uniform(weight_min, weight_scale)

                    # Apply distance scaling to weight
                    weight = base_weight * distance_factor

                    # Calculate delay based on distance
                    if hasattr(self, 'neuron_3d_positions'):
                        delay = transmission_delay * (0.5 + 0.5 * dist / sphere_diameter)
                    else:
                        delay = transmission_delay

                    # Prune zero weights - only store and add to graph if weight is non-zero
                    if weight > 0.01 or weight < -0.01:
                        # Store connection information
                        self.weights[i, j] = weight
                        self.delays[i, j] = delay

                        # Add to graph
                        self.graph.add_edge(i, j, weight=weight, delay=delay, distance=dist, distance_factor=distance_factor)
                        nonzero_connections += 1
                        self.connection_counts_by_type[conn_type] += 1

        self.attempted_connections = attempted_connections
        self.nonzero_connections = nonzero_connections

    def _print_connection_stats(self):
        """Print statistics about network connections with per-type breakdown."""
        total_possible = self.n_neurons * (self.n_neurons - 1)
        nonzero_count = np.count_nonzero(self.weights)

        # Count neurons by type
        n_excitatory = sum(1 for n in self.neurons if not n.is_inhibitory)
        n_inhibitory = sum(1 for n in self.neurons if n.is_inhibitory)

        # Calculate possible connections per type
        possible_ee = n_excitatory * (n_excitatory - 1)
        possible_ei = n_excitatory * n_inhibitory
        possible_ie = n_inhibitory * n_excitatory
        possible_ii = n_inhibitory * (n_inhibitory - 1)

        print("\n===== 3D Spherical Network Connection Statistics =====")
        print(f"Neurons: {self.n_neurons} (E: {n_excitatory}, I: {n_inhibitory})")
        print(f"Sphere radius: {self.sphere_radius:.2f}")
        print(f"Total possible connections: {total_possible}")
        print(f"Attempted connections: {self.attempted_connections}")
        print(f"Non-zero weight connections: {nonzero_count}")
        print(f"Connection density: {nonzero_count / total_possible:.6f}")
        print(f"Zero-weights pruned: {self.attempted_connections - nonzero_count}")

        print("\n--- Per-Type Connection Breakdown ---")
        print(f"{'Type':<6} {'Config P':<10} {'Actual P':<10} {'Expected':<12} {'Actual':<12} {'Match'}")

        type_labels = {'ee': 'E→E', 'ei': 'E→I', 'ie': 'I→E', 'ii': 'I→I'}
        possible_counts = {'ee': possible_ee, 'ei': possible_ei, 'ie': possible_ie, 'ii': possible_ii}

        for conn_type in ['ee', 'ei', 'ie', 'ii']:
            config_p = self.connection_probabilities[conn_type]
            possible = possible_counts[conn_type]
            expected = int(possible * config_p)
            actual = self.connection_counts_by_type[conn_type]
            actual_p = actual / possible if possible > 0 else 0

            # Check if actual is within 5% of expected (accounting for randomness)
            tolerance = 0.05
            match = "✓" if abs(actual_p - config_p) < tolerance else "✗"

            print(f"{type_labels[conn_type]:<6} {config_p:<10.3f} {actual_p:<10.3f} {expected:<12,} {actual:<12,} {match}")

        print("======================================================")

    def plot_connection_type_distribution(self, save_path="connection_type_distribution.png"):
        """
        Generate a bar chart comparing configured vs actual connection probabilities.

        Parameters:
        -----------
        save_path : str
            Path to save the figure
        """
        import matplotlib.pyplot as plt

        # Count neurons by type
        n_excitatory = sum(1 for n in self.neurons if not n.is_inhibitory)
        n_inhibitory = sum(1 for n in self.neurons if n.is_inhibitory)

        # Calculate possible connections per type
        possible_counts = {
            'ee': n_excitatory * (n_excitatory - 1),
            'ei': n_excitatory * n_inhibitory,
            'ie': n_inhibitory * n_excitatory,
            'ii': n_inhibitory * (n_inhibitory - 1)
        }

        conn_types = ['ee', 'ei', 'ie', 'ii']
        type_labels = ['E→E', 'E→I', 'I→E', 'I→I']

        # Get configured and actual probabilities
        config_probs = [self.connection_probabilities[ct] for ct in conn_types]
        actual_probs = [
            self.connection_counts_by_type[ct] / possible_counts[ct]
            if possible_counts[ct] > 0 else 0
            for ct in conn_types
        ]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Connection Probabilities
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

        # Add value labels on bars
        for bar, val in zip(bars1, config_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, actual_probs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Connection Counts
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

        # Add value labels on bars
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

    def plot_neuron_parameter_distributions(self, save_path="neuron_parameter_distributions.png", darkstyle=True):
        """
        Generate histograms showing the distribution of jittered neuron parameters.

        Parameters:
        -----------
        save_path : str
            Path to save the figure
        darkstyle : bool
            If True, use dark background style
        """
        import matplotlib.pyplot as plt

        if darkstyle:
            plt.style.use('dark_background')

        # Use stored neuron_params arrays (more efficient than iterating neurons)
        v_rest_vals = self.neuron_params['v_rest']
        v_threshold_vals = self.neuron_params['v_threshold']
        tau_m_vals = self.neuron_params['tau_m']
        tau_ref_vals = self.neuron_params['tau_ref']
        tau_e_vals = self.neuron_params['tau_e']
        tau_i_vals = self.neuron_params['tau_i']
        adaptation_inc_vals = self.neuron_params['adaptation_increment']
        tau_adapt_vals = self.neuron_params['tau_adaptation']
        is_inhibitory = self.neuron_params['is_inhibitory']

        # Separate by neuron type for tau_ref (since E and I have different baselines)
        tau_ref_exc = tau_ref_vals[~is_inhibitory]
        tau_ref_inh = tau_ref_vals[is_inhibitory]

        # Create figure with 4x2 subplots
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle('Neuron Parameter Distributions (with Gaussian Jitter)', fontsize=16, y=0.995)

        # Color scheme
        exc_color = '#ff6b6b'  # Red for excitatory
        inh_color = '#4dabf7'  # Blue for inhibitory
        mixed_color = '#69db7c'  # Green for mixed

        # Helper function to add stats annotation
        def add_stats(ax, vals, color='white'):
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f'μ={mean_val:.2f}')
            ax.text(0.97, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nn = {len(vals)}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   horizontalalignment='right', color=color,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # 1. V_rest
        ax = axes[0, 0]
        ax.hist(v_rest_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('V_rest (mV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Resting Potential', fontsize=12)
        add_stats(ax, v_rest_vals)

        # 2. V_threshold
        ax = axes[0, 1]
        ax.hist(v_threshold_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('V_threshold (mV)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Spike Threshold', fontsize=12)
        add_stats(ax, v_threshold_vals)

        # 3. Tau_m
        ax = axes[1, 0]
        ax.hist(tau_m_vals, bins=30, color=mixed_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_m (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Membrane Time Constant', fontsize=12)
        add_stats(ax, tau_m_vals)

        # 4. Tau_ref (separate E and I)
        ax = axes[1, 1]
        if len(tau_ref_exc) > 0:
            ax.hist(tau_ref_exc, bins=20, color=exc_color, alpha=0.6, edgecolor='white', linewidth=0.5, label=f'Exc (n={len(tau_ref_exc)})')
        if len(tau_ref_inh) > 0:
            ax.hist(tau_ref_inh, bins=20, color=inh_color, alpha=0.6, edgecolor='white', linewidth=0.5, label=f'Inh (n={len(tau_ref_inh)})')
        ax.set_xlabel('τ_ref (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Refractory Period (E vs I)', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        # Add combined stats
        ax.text(0.97, 0.72, f'E: μ={np.mean(tau_ref_exc):.2f}, σ={np.std(tau_ref_exc):.2f}\nI: μ={np.mean(tau_ref_inh):.2f}, σ={np.std(tau_ref_inh):.2f}',
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               horizontalalignment='right', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # 5. Tau_e
        ax = axes[2, 0]
        ax.hist(tau_e_vals, bins=30, color=exc_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_e (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Excitatory Synaptic Time Constant', fontsize=12)
        add_stats(ax, tau_e_vals, color=exc_color)

        # 6. Tau_i
        ax = axes[2, 1]
        ax.hist(tau_i_vals, bins=30, color=inh_color, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('τ_i (ms)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Inhibitory Synaptic Time Constant', fontsize=12)
        add_stats(ax, tau_i_vals, color=inh_color)

        # 7. Adaptation increment
        ax = axes[3, 0]
        ax.hist(adaptation_inc_vals, bins=30, color='#ffd43b', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Adaptation Increment', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Spike-Frequency Adaptation Increment', fontsize=12)
        add_stats(ax, adaptation_inc_vals, color='#ffd43b')

        # 8. Tau_adaptation
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

        # Return summary statistics
        return {
            'v_rest': {'mean': np.mean(v_rest_vals), 'std': np.std(v_rest_vals)},
            'v_threshold': {'mean': np.mean(v_threshold_vals), 'std': np.std(v_threshold_vals)},
            'tau_m': {'mean': np.mean(tau_m_vals), 'std': np.std(tau_m_vals)},
            'tau_ref_exc': {'mean': np.mean(tau_ref_exc), 'std': np.std(tau_ref_exc)} if len(tau_ref_exc) > 0 else None,
            'tau_ref_inh': {'mean': np.mean(tau_ref_inh), 'std': np.std(tau_ref_inh)} if len(tau_ref_inh) > 0 else None,
            'tau_e': {'mean': np.mean(tau_e_vals), 'std': np.std(tau_e_vals)},
            'tau_i': {'mean': np.mean(tau_i_vals), 'std': np.std(tau_i_vals)},
            'adaptation_increment': {'mean': np.mean(adaptation_inc_vals), 'std': np.std(adaptation_inc_vals)},
            'tau_adaptation': {'mean': np.mean(tau_adapt_vals), 'std': np.std(tau_adapt_vals)},
        }

    def reset_all(self):
        """Reset all neurons to initial state."""
        for neuron in self.neurons:
            neuron.reset()
        self.network_activity = []

        # Reset avalanche tracking
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None

        # Clear the spike queue
        self.spike_queue = deque()

    def stimulate_neuron(self, idx, current):
        """Stimulate a specific neuron with given current."""
        if 0 <= idx < self.n_neurons:
            self.neurons[idx].stimulate(current)

    def stimulate_random(self, n_stimulated=1, current=10.0):
        """
        Stimulate a random set of neurons with the given current.

        Parameters:
        -----------
        n_stimulated : int
            Number of neurons to stimulate
        current : float
            Current to apply to each neuron
        """
        if n_stimulated > self.n_neurons:
            n_stimulated = self.n_neurons

        # Select random neurons
        indices = np.random.choice(np.arange(self.n_neurons), size=n_stimulated, replace=False)

        # Apply stimulation
        for idx in indices:
            self.stimulate_neuron(idx, current)

    def update_network(self, dt):
        """
        Update the entire network for one time step.
        Returns indices of neurons that spiked in this time step.

        This method also tracks neuronal avalanches.

        Parameters:
        -----------
        dt : float
            Time step size in ms

        Returns:
        --------
        list
            Indices of neurons that spiked in this time step
        """
        # Collect spikes from this time step
        active_indices = []

        # Update each neuron
        for i, neuron in enumerate(self.neurons):
            # Update neuron and check if it spiked
            spiked = neuron.update(dt)

            if spiked:
                active_indices.append(i)

        # Track network-wide activity
        self.network_activity.append(len(active_indices))

        activity_level = len(active_indices)

        # Update avalanche tracking
        if activity_level > 0 and self.current_avalanche_start is None:
            # Start of a new avalanche
            self.current_avalanche_start = len(self.network_activity) - 1
            self.current_avalanche_size = activity_level
        elif activity_level > 0 and self.current_avalanche_start is not None:
            # Continue existing avalanche
            self.current_avalanche_size += activity_level
        elif activity_level == 0 and self.current_avalanche_start is not None:
            # End of an avalanche
            duration = len(self.network_activity) - 1 - self.current_avalanche_start

            # Record avalanche metrics
            self.avalanche_sizes.append(self.current_avalanche_size)
            self.avalanche_durations.append(duration)

            # Reset tracking
            self.current_avalanche_start = None
            self.current_avalanche_size = 0

        # Process pending spikes that have reached their target time
        current_time = len(self.network_activity) * dt
        delivered_spikes = 0

        # Process spikes that are due to be delivered at the current time
        while self.spike_queue and self.spike_queue[0][0] <= current_time:
            delivery_time, source_idx, target_idx, weight = self.spike_queue.popleft()

            # Deliver the spike to the target neuron
            self.neurons[target_idx].receive_spike(weight)
            delivered_spikes += 1

        # Queue new spikes from neurons that just fired, with appropriate delays
        for i in active_indices:
            # Use graph edges for propagation instead of iterating through all neurons
            for j in self.graph.successors(i):
                weight = self.weights[i, j]
                delay = self.delays[i, j]

                # Calculate delivery time
                delivery_time = current_time + delay

                # Add to queue
                self.spike_queue.append((delivery_time, i, j, weight))

        # Sort the queue by delivery time to ensure proper order
        if active_indices:
            self.spike_queue = deque(sorted(self.spike_queue, key=lambda x: x[0]))

        # Print some debug info occasionally
        if len(self.network_activity) % 1000 == 0 and self.spike_queue:
            print(f"Time: {current_time:.1f}ms, Queue size: {len(self.spike_queue)}, Delivered: {delivered_spikes}")

        return active_indices

    def prune_zero_weights(self):
        """
        Manually prune zero-weight connections from an existing network.

        Returns:
        --------
        int
            Number of connections pruned
        """
        nonzero_before = np.count_nonzero(self.weights)

        zero_indices = np.where(self.weights == 0)
        num_zeros = len(zero_indices[0])

        edges_removed = 0
        for i, j in zip(zero_indices[0], zero_indices[1]):
            if self.graph.has_edge(i, j):
                self.graph.remove_edge(i, j)
                edges_removed += 1

        print(f"Pruned {edges_removed} zero-weight connections from the network graph.")
        print(f"Matrix still contains {num_zeros} zero entries.")

        return edges_removed
