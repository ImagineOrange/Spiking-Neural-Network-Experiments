from collections import deque
import numpy as np
import networkx as nx

class LayeredNeuronalNetwork:
    """
    Represents a network of LIF neurons (potentially with multiple layers).
    This version is designed to be built more manually using add_neuron and add_connection,
    compared to automatically generating structure based on parameters.
    It includes spike propagation with delays and basic avalanche tracking.
    """
    def __init__(self, n_neurons=100, inhibitory_fraction=0.2, **kwargs):
        """
        Initializes the neuronal network structure.

        Args:
            n_neurons (int): Total number of neurons the network can hold.
            inhibitory_fraction (float): Default fraction of inhibitory neurons (used if not specified per neuron).
            **kwargs: Additional parameters passed to neurons if created internally,
                      or stored for reference (e.g., noise levels, reversal potentials, STD parameters).
        """
        self.n_neurons = n_neurons # Maximum number of neurons
        self.inhibitory_fraction = inhibitory_fraction # Default fraction
        self.neurons = [] # List to hold LIFNeuron instances (sparse initially)

        # Network structure representation
        self.graph = nx.DiGraph() # Directed graph for connections using NetworkX
        self.weights = np.zeros((n_neurons, n_neurons)) # Matrix to store connection weights
        self.delays = np.zeros((n_neurons, n_neurons)) # Matrix to store connection delays

        # Optional spatial information
        self.neuron_grid_positions = {} # Dictionary mapping neuron index to (row, col) or (x, y) position

        # Simulation state
        self.spike_queue = deque() # Queue to manage delayed spike delivery
        self.network_activity = [] # Records active neuron indices per time step

        # Avalanche tracking state
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None # Time when the current avalanche began

        # Store neuron parameters for potential later use or reference
        self.neuron_params = kwargs

        # Parameters potentially derived or stored from kwargs
        self.side_length = int(np.ceil(np.sqrt(n_neurons))) # For grid layout visualization
        self.v_noise_amp = kwargs.get('v_noise_amp', 0.1)
        self.i_noise_amp = kwargs.get('i_noise_amp', 0.01)
        self.e_reversal = kwargs.get('e_reversal', 0.0)
        self.i_reversal = kwargs.get('i_reversal', -80.0)
        self.distance_lambda = kwargs.get('distance_lambda', 0.1) # Distance decay factor for weights/delays
        self.weight_scale = kwargs.get('weight_scale', 0.1) # Base scale for weights

        # STD (Short-Term Depression) parameters
        self.std_enabled = kwargs.get('std_enabled', False)
        self.U = kwargs.get('U', 0.3)  # Utilization factor
        self.tau_d = kwargs.get('tau_d', 400.0)  # Recovery time constant (ms)

        # Initialize STD synaptic resources if enabled
        if self.std_enabled:
            # All synapses start with full resources (x = 1.0)
            self.x_resources = np.ones((n_neurons, n_neurons))
        else:
            self.x_resources = None

    def add_neuron(self, neuron, node_id, pos, layer):
        """
        Adds a pre-initialized neuron object to the network.

        Args:
            neuron (LIFNeuronWithReversal): The neuron instance to add.
            node_id (int): The index (ID) for this neuron in the network (0 to n_neurons-1).
            pos (tuple): The (x, y) position of the neuron for visualization.
            layer (int): The layer number this neuron belongs to.
        """
        if node_id >= self.n_neurons:
            raise IndexError(f"Node ID {node_id} exceeds network size {self.n_neurons}")
        # Ensure the neurons list is large enough, padding with None if necessary
        while len(self.neurons) <= node_id:
            self.neurons.append(None)
        # Add the neuron instance
        self.neurons[node_id] = neuron
        # Store position (note the swapped order, likely for row-major grid indexing if used)
        self.neuron_grid_positions[node_id] = (pos[1], pos[0])
        # Add node to the NetworkX graph with attributes
        self.graph.add_node(node_id, is_inhibitory=neuron.is_inhibitory, layer=layer, pos=pos)

    def add_connection(self, u, v, weight, delay=1.0):
         """
         Adds a directed connection from neuron 'u' to neuron 'v'.

         Args:
             u (int): Index of the source neuron.
             v (int): Index of the target neuron.
             weight (float): Synaptic weight of the connection.
             delay (float): Transmission delay of the connection (ms). Must be >= 0.1 ms.
         """
         # Check if source and target indices are valid and nodes exist in the graph
         if u < self.n_neurons and v < self.n_neurons and u in self.graph and v in self.graph:
              # Ensure a minimum delay (e.g., one time step if dt=0.1)
              delay = max(0.1, delay)
              # Add edge to NetworkX graph with attributes
              self.graph.add_edge(u, v, weight=weight, delay=delay)
              # Store weight and delay in matrices for quick access during simulation
              self.weights[u, v] = weight
              self.delays[u, v] = delay

    def prune_weak_connections(self, threshold=0.03):
        """
        Removes connections from the network whose absolute weight is below a threshold.

        Args:
            threshold (float): The minimum absolute weight for a connection to be kept.
        """
        pruned_count = 0
        edges_to_remove = []
        # Iterate through all edges and identify those below the threshold
        for u, v, data in self.graph.edges(data=True):
            if abs(data.get('weight', 0)) < threshold:
                edges_to_remove.append((u, v))
        # Remove the identified edges from the graph and zero out matrix entries
        for u, v in edges_to_remove:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
                self.weights[u, v] = 0.0
                self.delays[u, v] = 0.0
                pruned_count += 1
        print(f"Pruned {pruned_count} connections with |weight| < {threshold}.")

    def reset_all(self):
        """Resets all neurons in the network and clears simulation state."""
        # Ensure neuron list has placeholders up to n_neurons
        while len(self.neurons) < self.n_neurons:
            self.neurons.append(None)
        # Reset each individual neuron if it exists
        for neuron in self.neurons:
             if neuron:
                 neuron.reset()
        # Reset network-level simulation variables
        self.network_activity = []
        self.avalanche_sizes = []
        self.avalanche_durations = []
        self.current_avalanche_size = 0
        self.current_avalanche_start = None
        self.spike_queue = deque() # Clear the spike queue

        # Reset STD resources if enabled
        if self.std_enabled and self.x_resources is not None:
            self.x_resources = np.ones((self.n_neurons, self.n_neurons))

    def update_network(self, dt):
        """
        Updates the entire network state for one time step 'dt'.

        Args:
            dt (float): The simulation time step (ms).

        Returns:
            list: A list of indices of neurons that spiked in this time step.
        """
        active_indices = [] # List to store neurons that spike in this step
        current_time = len(self.network_activity) * dt # Simulation time at the start of this step

        # --- 1. Update individual neuron states ---
        for i, neuron in enumerate(self.neurons):
            if neuron: # Check if neuron exists at this index
                spiked = neuron.update(dt)
                if spiked:
                    active_indices.append(i) # Record index if spiked

        # --- 2. Process delayed spikes ---
        delivered_spikes = 0
        # Check the spike queue for spikes scheduled to arrive at or before the current time
        while self.spike_queue and self.spike_queue[0][0] <= current_time:
            # Remove the earliest spike from the queue
            delivery_time, source_idx, target_idx, weight = self.spike_queue.popleft()
            # Deliver the spike to the target neuron if it exists
            if target_idx < len(self.neurons) and self.neurons[target_idx]:
                self.neurons[target_idx].receive_spike(weight)
                delivered_spikes += 1

        # --- 3. Queue new spikes for future delivery ---
        # For each neuron that spiked in the current step:
        for i in active_indices:
            # Check if the neuron index is valid and exists in the graph
            if i < self.n_neurons and i in self.graph:
                 # Iterate through its successors (neurons it connects to) in the graph
                 for j in self.graph.successors(i):
                     # Check if target neuron index is valid and connection exists with non-zero weight
                     if j < self.n_neurons and self.graph.has_edge(i, j) and self.weights[i, j] != 0:
                             weight = self.weights[i, j]

                             # Apply Short-Term Depression if enabled
                             if self.std_enabled and self.x_resources is not None:
                                 # Current available resources for this synapse
                                 x = self.x_resources[i, j]
                                 # Effective weight after depression
                                 weight = weight * self.U * x
                                 # Update resources: some are used, rest recover
                                 self.x_resources[i, j] = x - self.U * x

                             # Use stored delay, ensuring it's at least dt
                             delay = self.delays[i, j] if self.delays[i,j] > 0 else dt
                             delivery_time = current_time + delay # Calculate when spike arrives
                             # Add spike details to the queue
                             self.spike_queue.append((delivery_time, i, j, weight))

        # --- 4. Recover STD resources ---
        # If STD is enabled, allow synaptic resources to recover over time
        if self.std_enabled and self.x_resources is not None:
            # Recovery follows: dx/dt = (1 - x) / tau_d
            # Using forward Euler: x(t+dt) = x(t) + dt * (1 - x(t)) / tau_d
            recovery_rate = dt / self.tau_d
            self.x_resources += recovery_rate * (1.0 - self.x_resources)
            # Clamp to [0, 1] to avoid numerical issues
            self.x_resources = np.clip(self.x_resources, 0.0, 1.0)

        # --- 5. Maintain sorted spike queue ---
        # If new spikes were added, re-sort the queue by delivery time
        if active_indices:
            self.spike_queue = deque(sorted(self.spike_queue, key=lambda x: x[0]))

        # --- 6. Update network activity and avalanche tracking ---
        activity_level = len(active_indices) # Number of spikes in this time step
        self.network_activity.append(active_indices) # Record the spiking neurons

        # Avalanche detection logic:
        if activity_level > 0: # If there was activity
            if self.current_avalanche_start is None: # If no avalanche was ongoing
                self.current_avalanche_start = current_time # Start a new avalanche
                self.current_avalanche_size = activity_level # Initial size
            else: # If an avalanche was already ongoing
                self.current_avalanche_size += activity_level # Add to its size
        elif self.current_avalanche_start is not None: # If activity stopped and an avalanche was ongoing
            # Calculate duration (time from start to the beginning of the current silent step)
            duration = (len(self.network_activity) * dt) - self.current_avalanche_start
            # Ensure duration is at least one time step
            if duration < dt: duration = dt
            # Record the completed avalanche's size and duration
            self.avalanche_sizes.append(self.current_avalanche_size)
            self.avalanche_durations.append(duration)
            # Reset avalanche state
            self.current_avalanche_start = None
            self.current_avalanche_size = 0

        return active_indices

