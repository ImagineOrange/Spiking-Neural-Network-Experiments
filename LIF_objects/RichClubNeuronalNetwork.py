import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from .LIFNeuronWithReversal import LIFNeuronWithReversal

class RichClubNeuronalNetwork:
    """
    A neuronal network with rich club connectivity.

    This network implements "rich club" organization where high-degree nodes
    (hubs with k=7 to k=15) preferentially connect to each other, exhibiting
    Φ_norm(k) > 1 for this degree range.

    Based on CircularNeuronalNetwork but with modified connectivity pattern.
    """
    def __init__(self, n_neurons=100, target_degree_range=(7, 15), connection_p=0.1,
                 weight_scale=1.0, weight_min=0.0, spatial=True, transmission_delay=1.0,
                 inhibitory_fraction=0.2, layout='grid',
                 v_noise_amp=0.3, i_noise_amp=0.05,
                 e_reversal=0.0, i_reversal=-80.0, distance_lambda=1,
                 rich_club_boost=3.0, std_enabled=False, U=0.3, tau_d=400.0):
        """
        Initialize the network with rich club connectivity.

        Parameters:
        -----------
        n_neurons : int
            Number of neurons in the network
        target_degree_range : tuple
            (min_degree, max_degree) for rich club nodes (default: 7-15)
        connection_p : float
            Base connection probability between neurons (0-1)
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
            Spatial layout: 'grid', 'circle-perimeter', or 'circle-filled'
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
        rich_club_boost : float
            Multiplier for connection probability between rich club nodes (default: 3.0)
        std_enabled : bool
            Enable Short-Term Synaptic Depression (Tsodyks-Markram model)
        U : float
            Utilization factor for STD (fraction of resources released per spike)
        tau_d : float
            Recovery time constant for STD in ms
        """
        # Store distance decay parameter
        self.distance_lambda = distance_lambda
        self.n_neurons = n_neurons
        self.weight_scale = weight_scale
        self.target_degree_range = target_degree_range
        self.rich_club_boost = rich_club_boost

        # STD state variables
        self.std_enabled = std_enabled
        self.U = U  # Utilization factor
        self.tau_d = tau_d  # Recovery time constant (ms)

        # Calculate grid dimensions (approximate square)
        self.side_length = int(np.ceil(np.sqrt(n_neurons)))

        # Create graph - only connections with non-zero weights will be added
        self.graph = nx.DiGraph()

        # Create neurons
        self.neurons = []
        self.neuron_grid_positions = {}

        # Generate spatial positions if needed
        if spatial:
            self._generate_spatial_positions(layout)

        # Create neurons and add to graph
        for i in range(n_neurons):
            # Determine if inhibitory based on fraction
            is_inhibitory = np.random.rand() < inhibitory_fraction

            # Create neuron with appropriate parameters
            if is_inhibitory:
                # Inhibitory neurons - stronger adaptation to prevent sustained firing
                neuron = LIFNeuronWithReversal(
                    v_rest=-65.0,
                    v_threshold=-55.0,
                    v_reset=-75.0,
                    tau_m=10.0,
                    tau_ref=np.random.uniform(1, 1.5),
                    tau_e=3.0,             # Faster excitatory time constant
                    tau_i=7.0,             # Shorter inhibitory time constant
                    is_inhibitory=True,
                    e_reversal=e_reversal,     # Excitatory reversal potential
                    i_reversal=i_reversal,     # Inhibitory reversal potential
                    v_noise_amp=v_noise_amp,   # Membrane potential noise
                    i_noise_amp=i_noise_amp,   # Synaptic input noise
                    adaptation_increment=0.1,  # Stronger adaptation for inhibitory neurons
                    tau_adaptation=40.0       # Adaptation time constant
                )
            else:
                # Excitatory neurons - moderate adaptation
                neuron = LIFNeuronWithReversal(
                    v_rest=-65.0,
                    v_threshold=-55.0,
                    v_reset=-75.0,
                    tau_m=10.0,
                    tau_ref=np.random.uniform(2, 2.5),
                    tau_e=3.0,              # Faster excitatory time constant
                    tau_i=7.0,              # Shorter inhibatory time constant
                    is_inhibitory=False,
                    e_reversal=e_reversal,      # Excitatory reversal potential
                    i_reversal=i_reversal,      # Inhibitory reversal potential
                    v_noise_amp=v_noise_amp,    # Membrane potential noise
                    i_noise_amp=i_noise_amp,    # Synaptic input noise
                    adaptation_increment=0.05,   # Standard adaptation for excitatory neurons
                    tau_adaptation=40.0        # Adaptation time constant
                )

            self.neurons.append(neuron)

            # Add to graph with position and properties
            if spatial:
                row, col = self.neuron_grid_positions[i]
                self.graph.add_node(i, pos=(col, self.side_length - row - 1),
                                   is_inhibitory=is_inhibitory)
            else:
                self.graph.add_node(i, is_inhibitory=is_inhibitory)

        # Initialize connection matrices - these will only store non-zero weights
        self.weights = np.zeros((n_neurons, n_neurons))
        self.delays = np.zeros((n_neurons, n_neurons))

        # Initialize STD synaptic resources if enabled
        if std_enabled:
            # All synapses start with full resources (x = 1.0)
            self.x_resources = np.ones((n_neurons, n_neurons))
        else:
            self.x_resources = None

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

        # Create connections with rich club structure
        self._create_rich_club_connectivity(connection_p, weight_scale, weight_min,
                                            transmission_delay, distance_lambda)

        # Print connection statistics
        self._print_connection_stats()

    def _generate_spatial_positions(self, layout='grid'):
        """
        Generate spatial positions for neurons.

        Parameters:
        -----------
        layout : str
            Layout type: 'grid' (default), 'circle-perimeter', or 'circle-filled'
        """
        if layout == 'circle-perimeter':
            # Create positions on the perimeter of a circle
            radius = self.n_neurons / (2 * np.pi)  # Scale radius based on number of neurons
            angles = np.linspace(0, 2*np.pi, self.n_neurons, endpoint=False)

            # Calculate positions
            for i in range(self.n_neurons):
                x = radius * np.cos(angles[i]) + radius  # Add radius to center in positive space
                y = radius * np.sin(angles[i]) + radius

                # We'll still store in row/col format for compatibility
                # but these aren't really rows and columns anymore
                row = int(y)
                col = int(x)
                self.neuron_grid_positions[i] = (row, col)

            # Update side_length to accommodate the circle
            self.side_length = int(2 * radius + 1)

        elif layout == 'circle-filled' or layout == 'circle':
            # Create positions filling a circle (evenly distributed)
            # Calculate the radius of the circle that can fit all neurons
            total_radius = np.sqrt(self.n_neurons / np.pi) * 1.5
            center_x = total_radius
            center_y = total_radius

            # Use rejection sampling to create uniform distribution in circle
            positions = []
            while len(positions) < self.n_neurons:
                # Generate random position in square that bounds the circle
                x = np.random.uniform(0, 2 * total_radius)
                y = np.random.uniform(0, 2 * total_radius)

                # Check if point is inside circle
                if (x - center_x)**2 + (y - center_y)**2 <= total_radius**2:
                    positions.append((int(y), int(x)))

            # Assign positions to neurons
            for i in range(self.n_neurons):
                self.neuron_grid_positions[i] = positions[i]

            # Update side_length to accommodate the circle
            self.side_length = int(2 * total_radius + 1)

        else:
            # Original grid layout
            for i in range(self.n_neurons):
                row = i // self.side_length
                col = i % self.side_length
                self.neuron_grid_positions[i] = (row, col)

    def _create_rich_club_connectivity(self, connection_p, weight_scale, weight_min,
                                       transmission_delay, distance_lambda=0.2):
        """
        Create rich club connectivity where high-degree nodes preferentially connect.

        Algorithm:
        1. First pass: Create baseline connections with connection_p
        2. Identify nodes with degrees in target range (k=7 to k=15)
        3. Second pass: Boost connection probability between rich club nodes
        4. Apply distance-based weight decay

        Parameters:
        -----------
        connection_p : float
            Base probability of connection between neurons
        weight_scale : float
            Scale factor for synaptic weights
        weight_min : float
            Minimum weight value
        transmission_delay : float
            Base transmission delay in ms
        distance_lambda : float
            Distance decay constant (higher values mean faster decay)
        """
        print("\n===== Creating Rich Club Network =====")

        # PHASE 1: Create baseline connectivity
        print("Phase 1: Creating baseline connectivity...")
        attempted_connections = 0
        baseline_connections = 0

        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and np.random.random() < connection_p:
                    attempted_connections += 1

                    # Calculate distance between neurons
                    if hasattr(self, 'neuron_grid_positions'):
                        pos_i = self.neuron_grid_positions[i]
                        pos_j = self.neuron_grid_positions[j]
                        dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                        distance_factor = np.exp(-distance_lambda * dist)
                    else:
                        dist = 0
                        distance_factor = 1.0

                    # Determine weight based on neuron types and distance
                    if self.neurons[i].is_inhibitory:
                        base_weight = -np.random.uniform(weight_min, weight_scale * 4.0)
                    else:
                        base_weight = np.random.uniform(weight_min, weight_scale)

                    # Apply distance scaling to weight
                    weight = base_weight * distance_factor

                    # Calculate delay based on distance
                    if hasattr(self, 'neuron_grid_positions'):
                        delay = transmission_delay * (0.5 + 0.5 * dist / self.side_length)
                    else:
                        delay = transmission_delay

                    # Only store if weight is non-zero
                    if weight > 0.01 or weight < -0.01:
                        self.weights[i, j] = weight
                        self.delays[i, j] = delay
                        self.graph.add_edge(i, j, weight=weight, delay=delay,
                                          distance=dist, distance_factor=distance_factor)
                        baseline_connections += 1

        print(f"Baseline: {baseline_connections} connections created")

        # PHASE 2: Identify potential rich club nodes based on current degree
        print(f"Phase 2: Identifying rich club nodes (degree range: {self.target_degree_range})...")
        degrees = dict(self.graph.degree())

        # Find nodes in the target degree range
        rich_club_nodes = [node for node, deg in degrees.items()
                          if self.target_degree_range[0] <= deg <= self.target_degree_range[1]]

        print(f"Found {len(rich_club_nodes)} potential rich club nodes")

        # PHASE 3: Boost connectivity between rich club nodes
        print(f"Phase 3: Boosting rich club connectivity (boost factor: {self.rich_club_boost})...")
        rich_club_connections = 0

        for i in rich_club_nodes:
            for j in rich_club_nodes:
                if i != j and not self.graph.has_edge(i, j):
                    # Boost connection probability for rich club
                    if np.random.random() < (connection_p * self.rich_club_boost):
                        # Calculate distance
                        if hasattr(self, 'neuron_grid_positions'):
                            pos_i = self.neuron_grid_positions[i]
                            pos_j = self.neuron_grid_positions[j]
                            dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                            distance_factor = np.exp(-distance_lambda * dist)
                        else:
                            dist = 0
                            distance_factor = 1.0

                        # Determine weight
                        if self.neurons[i].is_inhibitory:
                            base_weight = -np.random.uniform(weight_min, weight_scale * 4.0)
                        else:
                            base_weight = np.random.uniform(weight_min, weight_scale)

                        weight = base_weight * distance_factor

                        # Calculate delay
                        if hasattr(self, 'neuron_grid_positions'):
                            delay = transmission_delay * (0.5 + 0.5 * dist / self.side_length)
                        else:
                            delay = transmission_delay

                        # Add connection
                        if weight > 0.01 or weight < -0.01:
                            self.weights[i, j] = weight
                            self.delays[i, j] = delay
                            self.graph.add_edge(i, j, weight=weight, delay=delay,
                                              distance=dist, distance_factor=distance_factor,
                                              rich_club=True)
                            rich_club_connections += 1

        print(f"Added {rich_club_connections} rich club connections")

        # PHASE 4: Calculate final rich club coefficient
        self._calculate_rich_club_coefficient()

        self.attempted_connections = attempted_connections + rich_club_connections
        self.nonzero_connections = baseline_connections + rich_club_connections
        self.rich_club_nodes = rich_club_nodes
        self.baseline_connections = baseline_connections
        self.rich_club_connections = rich_club_connections

    def _calculate_rich_club_coefficient(self):
        """
        Calculate the rich club coefficient Φ(k) and normalized coefficient Φ_norm(k).

        Φ(k) measures the fraction of edges that exist between nodes with degree > k
        divided by the total possible edges between those nodes.

        Φ_norm(k) = Φ(k) / Φ_rand(k), where Φ_rand is from a random network.
        """
        print("\nCalculating rich club coefficients...")

        # Convert to undirected for rich club analysis
        G_undirected = self.graph.to_undirected()

        try:
            # Calculate rich club coefficient
            rc = nx.rich_club_coefficient(G_undirected, normalized=False)

            # Store results
            self.rich_club_coefficients = rc

            # Print coefficients for target range
            print("\nRich Club Coefficients Φ(k):")
            for k in sorted(rc.keys()):
                if self.target_degree_range[0] <= k <= self.target_degree_range[1]:
                    print(f"  k={k}: Φ(k) = {rc[k]:.4f}")

            # For normalized coefficient, we'd need to compare with random network
            # For now, just report the raw coefficient
            print("\nNote: Φ_norm(k) requires comparison with random network baseline")

        except Exception as e:
            print(f"Could not calculate rich club coefficient: {e}")
            self.rich_club_coefficients = {}

    def _print_connection_stats(self):
        """Print statistics about network connections."""
        total_possible = self.n_neurons * (self.n_neurons - 1)
        nonzero_count = np.count_nonzero(self.weights)

        print("\n===== Rich Club Network Statistics =====")
        print(f"Neurons: {self.n_neurons}")
        print(f"Total possible connections: {total_possible}")
        print(f"Non-zero weight connections: {nonzero_count}")
        print(f"  - Baseline connections: {self.baseline_connections}")
        print(f"  - Rich club connections: {self.rich_club_connections}")
        print(f"Connection density: {nonzero_count / total_possible:.6f}")
        print(f"Rich club nodes: {len(self.rich_club_nodes)} "
              f"(degrees {self.target_degree_range[0]}-{self.target_degree_range[1]})")
        print("========================================")

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

        # Reset STD resources
        if self.std_enabled:
            self.x_resources = np.ones((self.n_neurons, self.n_neurons))

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
            # Calculate duration in time steps
            duration = len(self.network_activity) - 1 - self.current_avalanche_start

            # Record avalanche metrics
            self.avalanche_sizes.append(self.current_avalanche_size)
            self.avalanche_durations.append(duration)

            # Reset tracking
            self.current_avalanche_start = None
            self.current_avalanche_size = 0

        # STD: Recover synaptic resources every timestep
        if self.std_enabled:
            # Exponential recovery: x(t+dt) = x(t) + (1 - x(t)) * (1 - exp(-dt/tau_d))
            recovery_factor = 1.0 - np.exp(-dt / self.tau_d)
            self.x_resources += (1.0 - self.x_resources) * recovery_factor

        # Process pending spikes that have reached their target time
        current_time = len(self.network_activity) * dt
        delivered_spikes = 0

        # Process spikes that are due to be delivered at the current time
        while self.spike_queue and self.spike_queue[0][0] <= current_time:
            # Get the first spike in the queue (spikes are sorted by delivery time)
            delivery_time, source_idx, target_idx, weight = self.spike_queue.popleft()

            # Deliver the spike to the target neuron
            self.neurons[target_idx].receive_spike(weight)
            delivered_spikes += 1

        # Queue new spikes from neurons that just fired, with appropriate delays
        for i in active_indices:
            # Use graph edges for propagation instead of iterating through all neurons
            # This efficiently skips zero-weight connections
            for j in self.graph.successors(i):
                weight_base = self.weights[i, j]
                delay = self.delays[i, j]

                # Apply STD if enabled
                if self.std_enabled:
                    x = self.x_resources[i, j]  # Current available resources
                    u = self.U  # Utilization (release probability)

                    # Compute effective weight with depression
                    weight_effective = weight_base * u * x

                    # Deplete resources: x ← x(1 - u)
                    self.x_resources[i, j] = x * (1.0 - u)

                    weight_to_deliver = weight_effective
                else:
                    weight_to_deliver = weight_base

                # Calculate delivery time
                delivery_time = current_time + delay

                # Add to queue (delivery_time, source, target, weight)
                self.spike_queue.append((delivery_time, i, j, weight_to_deliver))

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
        This is useful if you need to prune an already created network.

        Returns:
        --------
        int
            Number of connections pruned
        """
        # Count non-zero connections before pruning
        nonzero_before = np.count_nonzero(self.weights)

        # Find zero-weight connections
        zero_indices = np.where(self.weights == 0)
        num_zeros = len(zero_indices[0])

        # Remove edges from the graph
        edges_removed = 0
        for i, j in zip(zero_indices[0], zero_indices[1]):
            if self.graph.has_edge(i, j):
                self.graph.remove_edge(i, j)
                edges_removed += 1

        print(f"Pruned {edges_removed} zero-weight connections from the network graph.")
        print(f"Matrix still contains {num_zeros} zero entries.")

        return edges_removed
