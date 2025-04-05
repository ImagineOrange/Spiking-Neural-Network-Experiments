import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from LIFNeuronWithReversal import LIFNeuronWithReversal

class CircularNeuronalNetwork:
    """
    An enhanced version of the NeuronalNetwork class that combines:
    1. Reversal potentials for more biologically plausible modeling
    2. Avalanche detection and analysis capabilities
    3. Various network layouts
    4. Transmission delays for spike propagation
    5. Zero-weight pruning for efficiency
    
    This provides a comprehensive neural simulation framework.
    """
    def __init__(self, n_neurons=100, connection_p=0.1, weight_scale=1.0, 
            weight_min=0.0, spatial=True, transmission_delay=1.0, 
            inhibitory_fraction=0.2, layout='grid',
            v_noise_amp=0.3, i_noise_amp=0.05,
            e_reversal=0.0, i_reversal=-80.0, distance_lambda=1):
        """
        Initialize the network with the given parameters.
        
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
        """
        # Store distance decay parameter
        self.distance_lambda = distance_lambda
        self.n_neurons = n_neurons
        self.weight_scale = weight_scale
        
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
                    adaptation_increment=0.1,  # Stronger adaptation for inhibitory neurons was .8
                    tau_adaptation=40.0       # Adaptation time constant was 100
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
                    tau_i=7.0,              # Shorter inhibitory time constant
                    is_inhibitory=False,
                    e_reversal=e_reversal,      # Excitatory reversal potential
                    i_reversal=i_reversal,      # Inhibitory reversal potential  
                    v_noise_amp=v_noise_amp,    # Membrane potential noise
                    i_noise_amp=i_noise_amp,    # Synaptic input noise
                    adaptation_increment=0.05,   # Standard adaptation for excitatory neurons was .5
                    tau_adaptation=40.0        # Adaptation time constant was 100
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
        self._create_connections_with_distance_weighting(connection_p, weight_scale, weight_min, 
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
    
    def _create_connections_with_distance_weighting(self, connection_p, weight_scale, weight_min, transmission_delay, distance_lambda=0.2):
        """
        Create random connections between neurons with weights that decay with distance.
        Zero-weight connections are automatically pruned.
        
        Parameters:
        -----------
        connection_p : float
            Probability of connection between neurons
        weight_scale : float
            Scale factor for synaptic weights
        weight_min : float
            Minimum weight value
        transmission_delay : float
            Base transmission delay in ms
        distance_lambda : float
            Distance decay constant (higher values mean faster decay)
        """
        # Keep track of connection statistics
        attempted_connections = 0
        nonzero_connections = 0
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if i != j and np.random.random() < connection_p:
                    attempted_connections += 1
                    
                    # Calculate distance between neurons
                    if hasattr(self, 'neuron_grid_positions'):
                        pos_i = self.neuron_grid_positions[i]
                        pos_j = self.neuron_grid_positions[j]
                        dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                        
                        # Apply distance-based weight scaling
                        distance_factor = np.exp(-distance_lambda * dist)
                    else:
                        dist = 0  # No distance if spatial positioning is not enabled
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
                    
                    # Calculate delay based on distance if spatial
                    if hasattr(self, 'neuron_grid_positions'):
                        delay = transmission_delay * (0.5 + 0.5 * dist / self.side_length)
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
        
        self.attempted_connections = attempted_connections
        self.nonzero_connections = nonzero_connections
    
    def _print_connection_stats(self):
        """Print statistics about network connections."""
        total_possible = self.n_neurons * (self.n_neurons - 1)  # All possible connections
        nonzero_count = np.count_nonzero(self.weights)  # Actual connections with non-zero weights
        
        print("\n===== Network Connection Statistics =====")
        print(f"Neurons: {self.n_neurons}")
        print(f"Total possible connections: {total_possible}")
        print(f"Attempted connections: {self.attempted_connections}")
        print(f"Non-zero weight connections: {nonzero_count}")
        print(f"Connection density: {nonzero_count / total_possible:.6f}")
        print(f"Zero-weights pruned: {self.attempted_connections - nonzero_count}")
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
                weight = self.weights[i, j]
                delay = self.delays[i, j]
                
                # Calculate delivery time
                delivery_time = current_time + delay
                
                # Add to queue (delivery_time, source, target, weight)
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