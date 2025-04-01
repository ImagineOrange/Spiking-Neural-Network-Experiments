# [Previous code remains the same up to the end of imports]
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import seaborn as sns
from collections import deque
from tqdm import tqdm
import matplotlib.animation as animation # Ensure animation is imported
from matplotlib.gridspec import GridSpec # Needed for combined plot
import inspect # Import inspect module
import matplotlib.cm as cm             # For colormaps in visualization
import matplotlib.colors as mcolors    # For colormaps in visualization


# Set dark style for all plots
plt.style.use('dark_background')
sns.set_style("darkgrid", {"axes.grid": False}) # Disable grid lines by default for cleaner plots
plt.rcParams['grid.alpha'] = 0.2 # Keep grid lines subtle if enabled

# --- Class Definitions ---
# LIFNeuronWithReversal class (as defined previously)
class LIFNeuronWithReversal:
    # ... (rest of class definition as in previous correct version) ...
    def __init__(self, v_rest=-65.0, v_threshold=-55.0, v_reset=-75.0,
                 tau_m=10.0, tau_ref=2.0, tau_e=3.0, tau_i=7.0, is_inhibitory=False,
                 e_reversal=0.0, i_reversal=-70.0, v_noise_amp=0.5, i_noise_amp=0.05,
                 adaptation_increment=0.5, tau_adaptation=100.0):
        self.v_rest = v_rest; self.v_threshold = v_threshold; self.v_reset = v_reset
        self.tau_m = tau_m; self.tau_ref = tau_ref; self.tau_e = tau_e; self.tau_i = tau_i
        self.e_reversal = e_reversal; self.i_reversal = i_reversal
        self.v_noise_amp = v_noise_amp; self.i_noise_amp = i_noise_amp
        self.adaptation_increment = adaptation_increment; self.tau_adaptation = tau_adaptation
        self.v = v_rest; self.g_e = 0.0; self.g_i = 0.0; self.adaptation = 0.0
        self.t_since_spike = tau_ref; self.is_inhibitory = is_inhibitory
        self.layer = None; self.external_stim_g = 0.0
        self.v_history = []; self.g_e_history = []; self.g_i_history = []
        self.adaptation_history = []; self.i_syn_history = []; self.spike_times = []

    def reset(self):
        self.v = self.v_rest; self.g_e = 0.0; self.g_i = 0.0; self.adaptation = 0.0
        self.t_since_spike = self.tau_ref; self.v_history = []; self.g_e_history = []
        self.g_i_history = []; self.adaptation_history = []; self.i_syn_history = []
        self.spike_times = []; self.external_stim_g = 0.0

    def update(self, dt):
        self.v_history.append(self.v); self.g_e_history.append(self.g_e); self.g_i_history.append(self.g_i)
        self.adaptation_history.append(self.adaptation)
        i_syn_internal = self.g_e * (self.e_reversal - self.v) + self.g_i * (self.i_reversal - self.v)
        i_syn_external = self.external_stim_g * (self.e_reversal - self.v)
        i_syn = i_syn_internal + i_syn_external
        self.i_syn_history.append(i_syn); self.t_since_spike += dt
        if self.t_since_spike < self.tau_ref:
            self.v = self.v_reset
            self.g_e *= np.exp(-dt/self.tau_e); self.g_i *= np.exp(-dt/self.tau_i);
            self.adaptation *= np.exp(-dt/self.tau_adaptation)
            if self.i_noise_amp > 0:
                e_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt)); i_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
                self.g_e = max(0, self.g_e + e_noise); self.g_i = max(0, self.g_i + i_noise)
            return False
        v_noise = np.random.normal(0, self.v_noise_amp * np.sqrt(dt)) if self.v_noise_amp > 0 else 0
        dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_syn - self.adaptation) + v_noise;
        self.v += dv
        self.g_e *= np.exp(-dt/self.tau_e); self.g_i *= np.exp(-dt/self.tau_i);
        if self.i_noise_amp > 0:
            e_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt)); i_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
            self.g_e = max(0, self.g_e + e_noise); self.g_i = max(0, self.g_i + i_noise)
        self.adaptation *= np.exp(-dt/self.tau_adaptation)
        if self.v >= self.v_threshold:
            self.v = self.v_reset; self.t_since_spike = 0.0; self.adaptation += self.adaptation_increment; self.spike_times.append(len(self.v_history) * dt)
            return True
        return False

    def receive_spike(self, weight):
        if weight > 0: self.g_e += weight
        else: self.g_i += -weight

    def apply_external_stimulus(self, conductance_change):
        self.external_stim_g = max(0, conductance_change)


# ExtendedNeuronalNetworkWithReversal class (as defined previously)
class ExtendedNeuronalNetworkWithReversal:
    # ... (rest of class definition as in previous correct version) ...
    def __init__(self, n_neurons=100, inhibitory_fraction=0.2, **kwargs):
        self.n_neurons = n_neurons; self.inhibitory_fraction = inhibitory_fraction; self.neurons = []
        self.graph = nx.DiGraph(); self.weights = np.zeros((n_neurons, n_neurons)); self.delays = np.zeros((n_neurons, n_neurons))
        self.neuron_grid_positions = {}; self.spike_queue = deque(); self.network_activity = []
        self.avalanche_sizes = []; self.avalanche_durations = []; self.current_avalanche_size = 0; self.current_avalanche_start = None
        self.neuron_params = kwargs
        self.side_length = int(np.ceil(np.sqrt(n_neurons))); self.v_noise_amp = kwargs.get('v_noise_amp', 0.1); self.i_noise_amp = kwargs.get('i_noise_amp', 0.01)
        self.e_reversal = kwargs.get('e_reversal', 0.0); self.i_reversal = kwargs.get('i_reversal', -80.0); self.distance_lambda = kwargs.get('distance_lambda', 0.1); self.weight_scale = kwargs.get('weight_scale', 0.1)

    def add_neuron(self, neuron, node_id, pos, layer):
        if node_id >= self.n_neurons: raise IndexError("Node ID exceeds network size")
        while len(self.neurons) <= node_id: self.neurons.append(None)
        self.neurons[node_id] = neuron
        self.neuron_grid_positions[node_id] = (pos[1], pos[0])
        self.graph.add_node(node_id, is_inhibitory=neuron.is_inhibitory, layer=layer, pos=pos)

    def add_connection(self, u, v, weight, delay=1.0):
         if u < self.n_neurons and v < self.n_neurons and u in self.graph and v in self.graph:
              delay = max(0.1, delay)
              self.graph.add_edge(u, v, weight=weight, delay=delay)
              self.weights[u, v] = weight; self.delays[u, v] = delay

    def prune_weak_connections(self, threshold=0.03):
        pruned_count = 0; edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            if abs(data.get('weight', 0)) < threshold: edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
                self.weights[u, v] = 0.0; self.delays[u, v] = 0.0
                pruned_count += 1
        print(f"Pruned {pruned_count} connections with |weight| < {threshold}.")

    def reset_all(self):
        while len(self.neurons) < self.n_neurons: self.neurons.append(None)
        for neuron in self.neurons:
             if neuron: neuron.reset()
        self.network_activity = []; self.avalanche_sizes = []; self.avalanche_durations = []
        self.current_avalanche_size = 0; self.current_avalanche_start = None; self.spike_queue = deque()

    def update_network(self, dt):
        active_indices = []; current_time = len(self.network_activity) * dt
        for i, neuron in enumerate(self.neurons):
            if neuron:
                spiked = neuron.update(dt)
                if spiked: active_indices.append(i)
        delivered_spikes = 0
        while self.spike_queue and self.spike_queue[0][0] <= current_time:
            delivery_time, source_idx, target_idx, weight = self.spike_queue.popleft()
            if target_idx < len(self.neurons) and self.neurons[target_idx]:
                self.neurons[target_idx].receive_spike(weight); delivered_spikes += 1
        for i in active_indices:
            if i < self.n_neurons and i in self.graph:
                 for j in self.graph.successors(i):
                     if j < self.n_neurons and self.graph.has_edge(i, j) and self.weights[i, j] != 0:
                             weight = self.weights[i, j];
                             delay = self.delays[i, j] if self.delays[i,j] > 0 else dt
                             delivery_time = current_time + delay
                             self.spike_queue.append((delivery_time, i, j, weight))
        if active_indices: self.spike_queue = deque(sorted(self.spike_queue, key=lambda x: x[0]))
        activity_level = len(active_indices); self.network_activity.append(active_indices)
        if activity_level > 0:
            if self.current_avalanche_start is None: self.current_avalanche_start = current_time; self.current_avalanche_size = activity_level
            else: self.current_avalanche_size += activity_level
        elif self.current_avalanche_start is not None:
            duration = (len(self.network_activity) * dt) - self.current_avalanche_start
            if duration < dt: duration = dt
            self.avalanche_sizes.append(self.current_avalanche_size); self.avalanche_durations.append(duration)
            self.current_avalanche_start = None; self.current_avalanche_size = 0
        return active_indices


# --- Placeholder/Modified run_unified_simulation ---
def run_unified_simulation(network, duration=1000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
                         stim_fraction=0.01, stim_target_indices=None,
                         stim_pulse_duration_ms=1.0,
                         track_neurons=None, stochastic_stim=False,
                         no_stimulation=False):
    # ... (function definition remains the same as previous version) ...
    print(f"--- Running Simulation (Placeholder: {duration}ms) ---")
    n_steps = int(duration / dt)
    activity_record = []
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms}
    neuron_data = {idx: {'v_history': [], 'g_e_history': [], 'g_i_history': [], 'i_syn_history': [], 'spike_times': [], 'stim_times': [], 'is_inhibitory': network.neurons[idx].is_inhibitory if idx < len(network.neurons) and network.neurons[idx] else False} for idx in (track_neurons or [])}

    ongoing_stimulations = {}
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None

    stimulation_population = list(stim_target_indices) if stim_target_indices is not None else list(range(network.n_neurons))
    if not stimulation_population:
        print("Warning: Stimulation population is empty.")
        no_stimulation = True

    for step in tqdm(range(n_steps), desc="Simulation"):
        current_time = step * dt
        newly_stimulated_indices = []

        expired_stims = []
        currently_stimulated_set = set()
        for neuron_idx, end_time in ongoing_stimulations.items():
            if current_time >= end_time:
                expired_stims.append(neuron_idx)
            else:
                if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                    pulse_strength = stim_interval_strength
                    network.neurons[neuron_idx].apply_external_stimulus(pulse_strength)
                    currently_stimulated_set.add(neuron_idx)

        for neuron_idx in expired_stims:
            if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                 network.neurons[neuron_idx].apply_external_stimulus(0.0)
            if neuron_idx in ongoing_stimulations:
                del ongoing_stimulations[neuron_idx]

        if not no_stimulation and stimulation_population:
            num_to_stimulate = max(1, int(len(stimulation_population) * stim_fraction))
            apply_new_stim_pulse = False

            if stochastic_stim and random.random() < (dt / 100):
                 apply_new_stim_pulse = True
            elif stim_interval_steps and (step % stim_interval_steps == 0):
                 apply_new_stim_pulse = True

            if apply_new_stim_pulse:
                 target_neurons_for_pulse = random.sample(stimulation_population, min(num_to_stimulate, len(stimulation_population)))
                 stim_end_time = current_time + stim_pulse_duration_ms

                 for idx in target_neurons_for_pulse:
                     if idx not in ongoing_stimulations:
                         ongoing_stimulations[idx] = stim_end_time
                         newly_stimulated_indices.append(idx)
                         if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                            network.neurons[idx].apply_external_stimulus(stim_interval_strength)
                            currently_stimulated_set.add(idx)

                 if newly_stimulated_indices:
                     stimulation_record['pulse_starts'].append(current_time)
                     stimulation_record['times'] = stimulation_record['pulse_starts']
                     stimulation_record['neurons'].append(newly_stimulated_indices)

        if track_neurons:
            for idx in track_neurons:
                 if idx < len(network.neurons) and network.neurons[idx]:
                    neuron = network.neurons[idx]; neuron_data[idx]['v_history'].append(neuron.v); neuron_data[idx]['g_e_history'].append(neuron.g_e); neuron_data[idx]['g_i_history'].append(neuron.g_i)
                    i_syn_internal = neuron.g_e * (neuron.e_reversal - neuron.v) + neuron.g_i * (neuron.i_reversal - neuron.v)
                    i_syn_external = neuron.external_stim_g * (neuron.e_reversal - neuron.v)
                    neuron_data[idx]['i_syn_history'].append(i_syn_internal + i_syn_external)
                    if idx in newly_stimulated_indices:
                        neuron_data[idx]['stim_times'].append(current_time)

        active_indices = network.update_network(dt);
        activity_record.append(active_indices)
        if track_neurons:
             for idx in active_indices:
                  if idx in neuron_data: neuron_data[idx]['spike_times'].append(current_time)

    for neuron_idx in list(ongoing_stimulations.keys()):
        if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
             network.neurons[neuron_idx].apply_external_stimulus(0.0)
        del ongoing_stimulations[neuron_idx]

    print(f"--- Simulation Finished ---"); return activity_record, neuron_data, stimulation_record
# --- End Placeholder/Modified ---


# --- Import Visualization Utilities ---
# ... (placeholders as before) ...
def plot_psth_and_raster(*args, **kwargs): print(f"Placeholder: Plotting PSTH/Raster")
def plot_reversal_effects(*args, **kwargs): print(f"Placeholder: Plotting Reversal Effects")
def plot_enhanced_criticality_analysis(*args, **kwargs): print(f"Placeholder: Plotting Criticality Analysis")
def visualize_spatial_correlations(*args, **kwargs): print(f"Placeholder: Plotting Spatial Correlations")


# --- Visualization Functions ---

# plot_network_connections_sparse (as defined previously)
def plot_network_connections_sparse(network, pos, stimulated_neurons=None, connected_neurons=None,
                             edge_percent=10, save_path="formal_layered_network_sim.png"):
    # ... (function definition remains the same as previous version) ...
    graph = network.graph; n_neurons = network.n_neurons
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a1a')
    if stimulated_neurons is None: stimulated_neurons = []
    if connected_neurons is None: connected_neurons = []
    nodes_with_pos = [n for n in graph.nodes() if n in pos]
    if len(nodes_with_pos) != graph.number_of_nodes():
        print(f"Warning: Position data missing for {graph.number_of_nodes() - len(nodes_with_pos)} nodes.")

    node_list = list(nodes_with_pos);
    node_colors, node_sizes, node_shapes = [], [], []
    nodes_by_shape = {'s': [], 'o': []}; colors_by_shape = {'s': [], 'o': []}; sizes_by_shape = {'s': [], 'o': []}

    for node in node_list:
        is_inhibitory = graph.nodes[node].get('is_inhibitory', False); shape = 's' if is_inhibitory else 'o'; node_shapes.append(shape)
        color, size = '#a5b1c2', 20
        if node in stimulated_neurons: color, size = '#1dd1a1', 150
        elif node in connected_neurons: color, size = '#ffae42', 80
        node_colors.append(color); node_sizes.append(size)
        nodes_by_shape[shape].append(node); colors_by_shape[shape].append(color); sizes_by_shape[shape].append(size)

    for shape_marker, nodelist_for_shape in nodes_by_shape.items():
         if nodelist_for_shape: nx.draw_networkx_nodes(graph, pos, nodelist=nodelist_for_shape, node_size=sizes_by_shape[shape_marker], node_color=colors_by_shape[shape_marker], node_shape=shape_marker, ax=ax, alpha=0.9)

    all_edges = list(graph.edges(data=True)); random.shuffle(all_edges); target_edge_count = int(len(all_edges) * edge_percent / 100); edges_to_consider = all_edges[:target_edge_count]
    edges_to_draw_list, edge_colors, widths = [], [], []
    feedforward_color, feedback_color, recurrent_color = '#d62728', '#1f77b4', '#808080'
    for u, v, data in edges_to_consider:
        if u not in graph.nodes or v not in graph.nodes or u not in pos or v not in pos: continue
        edges_to_draw_list.append((u, v)); layer_u, layer_v = graph.nodes[u]['layer'], graph.nodes[v]['layer']
        if layer_v > layer_u: edge_colors.append(feedforward_color); widths.append(0.6)
        elif layer_v < layer_u: edge_colors.append(feedback_color); widths.append(0.6)
        else: edge_colors.append(recurrent_color); widths.append(0.4)

    if edges_to_draw_list:
        node_list_ordered_for_edges = list(graph.nodes())
        try:
            size_map = {node: node_sizes[node_list.index(node)] for node in node_list}
            node_size_list_ordered = [size_map.get(n, 20) for n in node_list_ordered_for_edges]
            nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw_list, edge_color=edge_colors, width=widths, alpha=0.5, arrows=True, arrowsize=8, node_size=node_size_list_ordered, ax=ax)
        except Exception as e:
             print(f"Error drawing edges: {e}. Ensure node sizes match nodes being drawn.")

    labels = {n: f"C{n}" for n in connected_neurons}
    labels_with_pos = {n: lbl for n, lbl in labels.items() if n in pos}
    if labels_with_pos: nx.draw_networkx_labels(graph, pos, labels=labels_with_pos, font_color='white', font_weight='bold', font_size=8, ax=ax)

    n_excitatory = sum(1 for n in graph.nodes() if not graph.nodes[n].get('is_inhibitory', False)); n_inhibitory = n_neurons - n_excitatory
    title = (f"Formal {network.graph.graph.get('num_layers', '?')}-Layer Network - {n_neurons} Neurons ({n_excitatory} Exc, {n_inhibitory} Inhib)\nShowing {len(edges_to_draw_list)} ({edge_percent:.1f}%) of {len(all_edges)} connections (Post Pruning)")
    ax.set_title(title, color='white', fontsize=14)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], color=feedforward_color, lw=1.5, label='Feedforward Conn.'),
        plt.Line2D([0], [0], color=feedback_color, lw=1.5, label='Feedback Conn.'),
        plt.Line2D([0], [0], color=recurrent_color, lw=1.5, label='Recurrent Conn.')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize='small'); ax.set_axis_off(); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight'); print(f"Saved network structure visualization to {save_path}")
    return fig


# visualize_activity_layout_grid (MODIFIED for showing pulse duration)
def visualize_activity_layout_grid(network, pos, activity_record, dt=0.1, stim_record=None,
                                   grid_resolution=(100, 150), save_path="3_layer_layout_grid_activity.gif",
                                   max_frames=1000, fps=30):
    # ... (function definition remains the same as previous version) ...
    print(f"Generating sparse grid activity animation (up to {max_frames} frames)...")
    n_neurons = network.n_neurons; total_steps = len(activity_record)
    if total_steps == 0: print("Warning: No activity recorded."); return None

    if not pos or not any(isinstance(p, (tuple, list)) and len(p) == 2 for p in pos.values()):
        print("Warning: Invalid or empty position data (pos), cannot generate layout grid animation.")
        return None

    grid_rows, grid_cols = grid_resolution; print(f"Mapping {n_neurons} neurons onto a {grid_rows}x{grid_cols} sparse grid...")
    all_nodes = list(network.graph.nodes());
    nodes_with_pos = [n for n in all_nodes if n in pos and isinstance(pos[n], (tuple, list)) and len(pos[n]) == 2]
    if not nodes_with_pos:
        print("Warning: No nodes found with valid position data."); return None

    xs = [pos[n][0] for n in nodes_with_pos]; ys = [pos[n][1] for n in nodes_with_pos]
    if not xs or not ys: print("Warning: Position data extraction failed."); return None

    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    x_range = (max_x - min_x); y_range = (max_y - min_y)
    x_margin = x_range * 0.05 if x_range > 1e-6 else 1.0
    y_margin = y_range * 0.05 if y_range > 1e-6 else 1.0
    min_x -= x_margin; max_x += x_margin
    min_y -= y_margin; max_y += y_margin
    x_range = max_x - min_x; y_range = max_y - min_y

    neuron_to_sparse_grid_pos = {}; grid_to_neuron_map = {}
    for i in nodes_with_pos:
         x, y = pos[i]
         col = int(((x - min_x) / x_range) * (grid_cols - 1)) if x_range > 1e-9 else grid_cols // 2
         row = int(((max_y - y) / y_range) * (grid_rows - 1)) if y_range > 1e-9 else grid_rows // 2
         col = max(0, min(grid_cols - 1, col)); row = max(0, min(grid_rows - 1, row))
         neuron_to_sparse_grid_pos[i] = (row, col); grid_coord = (row, col)
         if grid_coord not in grid_to_neuron_map: grid_to_neuron_map[grid_coord] = []
         grid_to_neuron_map[grid_coord].append(i)

    inhibitory_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    for i in all_nodes:
         if i < len(network.neurons) and network.neurons[i] and network.neurons[i].is_inhibitory:
              if i in neuron_to_sparse_grid_pos:
                  row, col = neuron_to_sparse_grid_pos[i]; inhibitory_mask[row, col] = True

    if total_steps > max_frames:
        indices = np.linspace(0, total_steps - 1, max_frames, dtype=int); sampled_activity = [activity_record[i] for i in indices]; sampled_times = [i * dt for i in indices]
    else: sampled_activity = activity_record; sampled_times = [i * dt for i in range(total_steps)]; indices = np.arange(total_steps)

    stim_active_at_step = {}
    any_stim_active_flag = [False] * total_steps
    if stim_record and 'pulse_starts' in stim_record and 'neurons' in stim_record:
        pulse_starts = stim_record['pulse_starts']
        neurons_per_start = stim_record['neurons']
        pulse_duration_ms = stim_record.get('pulse_duration_ms', dt)
        pulse_duration_steps = max(1, int(pulse_duration_ms / dt))

        for i, start_time in enumerate(pulse_starts):
             start_step = int(start_time / dt)
             end_step = start_step + pulse_duration_steps
             if i < len(neurons_per_start) and isinstance(neurons_per_start[i], (list, set)):
                 neurons_in_pulse = neurons_per_start[i]
                 for step_index in range(start_step, min(end_step, total_steps)):
                      if step_index not in stim_active_at_step:
                          stim_active_at_step[step_index] = set()
                      stim_active_at_step[step_index].update(neurons_in_pulse)
                      any_stim_active_flag[step_index] = True

    aspect_ratio = grid_cols / grid_rows if grid_rows > 0 else 1
    fig_height = 8; fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(max(8, fig_width), fig_height), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a'); ax.set_xticks([]); ax.set_yticks([])
    activity_grid = np.zeros((grid_rows, grid_cols)); activity_colors = np.zeros((grid_rows, grid_cols, 3))
    img = ax.imshow(activity_colors, interpolation='nearest', origin='upper', vmin=0, vmax=1, aspect='auto')
    title = ax.set_title(f"Time: 0.0 ms", color='white', fontsize=14)
    stim_text = ax.text(0.01, 0.98, "", transform=ax.transAxes, color='lime', fontsize=10, verticalalignment='top', fontweight='bold')
    prev_activity_grid = np.zeros((grid_rows, grid_cols))

    pbar = tqdm(total=len(sampled_activity), desc="Generating GIF Frames")

    def update_sparse_grid(frame_idx):
        nonlocal prev_activity_grid
        pbar.update(1)
        original_step_index = indices[frame_idx];
        active_neuron_indices_this_frame = set(sampled_activity[frame_idx])
        current_time = sampled_times[frame_idx]

        current_visual_intensity = prev_activity_grid * 0.75

        spike_intensity_grid = np.zeros((grid_rows, grid_cols))
        for idx in active_neuron_indices_this_frame:
             if idx in neuron_to_sparse_grid_pos:
                 row, col = neuron_to_sparse_grid_pos[idx]
                 spike_intensity_grid[row, col] = 1.0

        colors = np.zeros((grid_rows, grid_cols, 3))
        neurons_stimulated_this_frame = stim_active_at_step.get(original_step_index, set())

        for r in range(grid_rows):
             for c in range(grid_cols):
                 is_spiking = spike_intensity_grid[r, c] > 0.01
                 neuron_ids_at_coord = grid_to_neuron_map.get((r,c), [])
                 is_stimulated = any(nid in neurons_stimulated_this_frame for nid in neuron_ids_at_coord)

                 if is_spiking:
                     if is_stimulated:
                         colors[r, c, :] = [0.5, 1.0, 0.5]
                     elif inhibitory_mask[r, c]:
                         colors[r, c, 2] = 1.0
                     else:
                         colors[r, c, 0] = 1.0
                     current_visual_intensity[r, c] = 1.0
                 elif is_stimulated:
                     decayed_stim_intensity = max(0.1, current_visual_intensity[r, c])
                     colors[r, c, :] = [0.6 * decayed_stim_intensity, 0.6 * decayed_stim_intensity, 0.0]
                 else:
                      # Apply visual decay to non-stimulated, non-spiking cells that were previously active
                      colors[r,c,:] = prev_activity_grid[r,c] * 0.75 # Fade previous color? Needs careful handling
                      pass # Let black background show through decayed intensity

        img.set_array(np.clip(colors, 0, 1)); title.set_text(f"Time: {current_time:.1f} ms")
        stim_text.set_text("STIMULATION" if any_stim_active_flag[original_step_index] else "")

        prev_activity_grid = current_visual_intensity
        return [img, title, stim_text]

    anim = animation.FuncAnimation(fig, update_sparse_grid, frames=len(sampled_activity), interval=max(20, 1000//fps), blit=True)
    try:
        writer = animation.PillowWriter(fps=fps); anim.save(save_path, writer=writer, dpi=150); print(f"Successfully saved layout grid animation to {save_path}")
    except Exception as e:
        print(f"Error saving animation: {e}. PillowWriter might require Pillow installation (`pip install Pillow`)")
    finally:
        pbar.close()
        plt.close(fig)
    return anim

# --- Combined Activity and Layer PSTH Plot Function ---
def plot_activity_and_layer_psth(network, activity_record, layer_indices, dt=0.1, stim_times=None, bin_width_ms=5.0, save_path="activity_layer_psth.png"):
    # ... (function definition remains the same as previous version) ...
    print(f"Generating combined activity and layer PSTH plot...")
    num_layers = len(layer_indices)
    total_steps = len(activity_record)
    if total_steps == 0:
        print("Warning: No activity recorded.")
        return None

    times = np.arange(total_steps) * dt
    overall_activity = [len(spikes) for spikes in activity_record]
    fig = plt.figure(figsize=(12, 7), facecolor='#1a1a1a')
    height_ratios = [2] + [1] * num_layers
    gs = GridSpec(num_layers + 1, 1, height_ratios=height_ratios, hspace=0.3, figure=fig)

    ax_overall = fig.add_subplot(gs[0])
    ax_overall.plot(times, overall_activity, color='#ff7f0e', linewidth=1.0)
    ax_overall.set_title("Overall Network Activity", color='white')
    ax_overall.set_ylabel("Total Spikes", color='white')
    ax_overall.tick_params(axis='x', labelbottom=False)
    ax_overall.grid(True, alpha=0.2)
    if stim_times: # Should be pulse_starts
        for stim_time in stim_times:
            ax_overall.axvline(x=stim_time, color='lime', linestyle='--', linewidth=1.0, alpha=0.6)

    psth_axes = []
    bin_edges = np.arange(0, times[-1] + bin_width_ms, bin_width_ms)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for i in range(num_layers):
        ax_psth = fig.add_subplot(gs[i + 1], sharex=ax_overall)
        psth_axes.append(ax_psth)
        start_idx, end_idx = layer_indices[i]
        layer_neurons = list(range(start_idx, end_idx))
        num_neurons_in_layer = len(layer_neurons)

        if num_neurons_in_layer == 0:
             ax_psth.text(0.5, 0.5, "No neurons", ha='center', va='center', color='grey', transform=ax_psth.transAxes)
             ax_psth.set_ylabel(f"Layer {i+1}\nRate (Hz)", color='white')
             if i == num_layers - 1: ax_psth.set_xlabel("Time (ms)", color='white')
             else: ax_psth.tick_params(axis='x', labelbottom=False)
             continue

        layer_spike_times = []
        for t_step, active_indices in enumerate(activity_record):
            time_ms = t_step * dt
            spikes_in_layer = [idx for idx in active_indices if start_idx <= idx < end_idx]
            layer_spike_times.extend([time_ms] * len(spikes_in_layer))

        psth_counts, _ = np.histogram(layer_spike_times, bins=bin_edges)
        firing_rate = psth_counts / (bin_width_ms / 1000.0) / num_neurons_in_layer if num_neurons_in_layer > 0 else np.zeros_like(psth_counts)

        ax_psth.plot(bin_centers, firing_rate, color='#30a9de', linewidth=1.5)
        ax_psth.set_ylabel(f"Layer {i+1}\nRate (Hz)", color='white')
        ax_psth.grid(True, alpha=0.2)

        if stim_times: # Should be pulse_starts
            for stim_time in stim_times:
                ax_psth.axvline(x=stim_time, color='lime', linestyle='--', linewidth=1.0, alpha=0.6)

        if i == num_layers - 1: ax_psth.set_xlabel("Time (ms)", color='white')
        else: ax_psth.tick_params(axis='x', labelbottom=False)

    for ax in [ax_overall] + psth_axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(axis='y', colors='white', labelsize='small')
        ax.tick_params(axis='x', colors='white', labelsize='small')
        for spine in ax.spines.values():
            spine.set_color('#555555')

    fig.align_ylabels()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Network Activity and Layer-wise PSTH", color='white', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined activity and PSTH plot to {save_path}")
    return fig


# --- Layer-wise Raster Plot Function ---
def plot_layer_wise_raster(network, activity_record, layer_indices, dt=0.1, stim_times=None, save_path="layer_raster.png"):
    # ... (function definition remains the same as previous version) ...
    print(f"Generating layer-wise raster plot...")
    num_layers = len(layer_indices)
    total_steps = len(activity_record)
    if total_steps == 0:
        print("Warning: No activity recorded.")
        return None

    times_ms = np.arange(total_steps) * dt
    fig = plt.figure(figsize=(14, 2 + num_layers * 1.5), facecolor='#1a1a1a')
    gs = GridSpec(num_layers, 1, hspace=0.1, figure=fig)
    raster_axes = []
    max_neuron_id = network.n_neurons - 1

    for i in range(num_layers):
        sharex = raster_axes[0] if i > 0 else None
        ax_raster = fig.add_subplot(gs[i], sharex=sharex)
        raster_axes.append(ax_raster)
        start_idx, end_idx = layer_indices[i]
        layer_neurons = list(range(start_idx, end_idx))
        num_neurons_in_layer = len(layer_neurons)

        if num_neurons_in_layer == 0:
             ax_raster.text(0.5, 0.5, "No neurons", ha='center', va='center', color='grey', transform=ax_raster.transAxes)
             ax_raster.set_ylabel(f"Layer {i+1}", color='white')
             ax_raster.set_yticks([])
             if i == num_layers - 1: ax_raster.set_xlabel("Time (ms)", color='white')
             else: ax_raster.tick_params(axis='x', labelbottom=False)
             continue

        layer_spike_times = []
        layer_neuron_ids = []
        for t_step, active_indices in enumerate(activity_record):
            time_ms = t_step * dt
            for idx in active_indices:
                if start_idx <= idx < end_idx:
                    layer_spike_times.append(time_ms)
                    layer_neuron_ids.append(idx)

        if layer_spike_times:
            ax_raster.scatter(layer_spike_times, layer_neuron_ids, s=2, color='white', alpha=0.8, marker='|')
        ax_raster.set_ylabel(f"Layer {i+1}\nNeuron ID", color='white', fontsize=10)
        ax_raster.set_ylim(start_idx - 0.5, end_idx - 0.5)

        if stim_times: # Should be pulse_starts
            for stim_time in stim_times:
                ax_raster.axvline(x=stim_time, color='lime', linestyle='--', linewidth=0.8, alpha=0.6)

        ax_raster.grid(True, alpha=0.15, axis='x')
        if i == num_layers - 1: ax_raster.set_xlabel("Time (ms)", color='white')
        else: ax_raster.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        ax_raster.set_facecolor('#1a1a1a')
        ax_raster.tick_params(axis='y', colors='white', labelsize='small')
        ax_raster.tick_params(axis='x', colors='white', labelsize='small')
        for spine in ax_raster.spines.values():
            spine.set_color('#555555')

        if num_neurons_in_layer > 1:
             y_ticks = np.linspace(start_idx, end_idx -1, min(5, num_neurons_in_layer), dtype=int)
             ax_raster.set_yticks(y_ticks)
        elif num_neurons_in_layer == 1:
             ax_raster.set_yticks([start_idx])

    fig.align_ylabels(raster_axes)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.suptitle("Layer-wise Raster Plots", color='white', fontsize=16, y=0.99)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer-wise raster plot to {save_path}")
    return fig

# --- MODIFIED: Function to visualize distance relationships ---
def visualize_distance_dependences(network, pos, neuron_idx, base_transmission_delay,
                                   network_figsize=(10, 10), # Smaller default
                                   scatter_figsize=(10, 10), # Adjusted for 2 plots
                                   save_path_base="distance_dependence"):
    """
    Creates two separate figures showing how connection weights AND delays
    decay with distance for outgoing connections from a specific neuron.
    MODIFIED: Corrected axis limits for network plot.
    """
    print(f"Generating distance dependence plots for Neuron {neuron_idx}...")

    if neuron_idx not in pos:
        print(f"Error: Position for neuron {neuron_idx} not found.")
        return None, None, None

    center_pos = pos[neuron_idx]
    outgoing_data = [] # List of (target_idx, distance, weight, delay)

    # Find max distance for normalization (approximate using bounds of pos)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    max_possible_dist = np.sqrt((max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2) if len(pos)>1 else 1.0
    if max_possible_dist == 0: max_possible_dist = 1.0

    # Collect outgoing connection data
    for target_idx in network.graph.successors(neuron_idx):
        if target_idx in pos:
            target_pos = pos[target_idx]
            dist = np.sqrt((center_pos[0] - target_pos[0])**2 + (center_pos[1] - target_pos[1])**2)
            weight = network.weights[neuron_idx, target_idx]
            delay = network.delays[neuron_idx, target_idx]
            outgoing_data.append((target_idx, dist, weight, delay))

    

    # ========== FIGURE 2: WEIGHT VS DISTANCE ==========
    weight_scatter_fig, ax_weight = plt.subplots(figsize=scatter_figsize, facecolor='#1a1a1a')
    if outgoing_data:
        distances = [d for _, d, _, _ in outgoing_data]
        weights = [w for _, _, w, _ in outgoing_data]
        ax_weight.scatter(distances, weights, color='yellow', alpha=0.7, label='Outgoing Weights', s=30)
    ax_weight.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_weight.grid(True, alpha=0.2)
    ax_weight.set_xlabel('Distance', color='white', fontsize=12)
    ax_weight.set_ylabel('Synaptic Weight', color='white', fontsize=12)
    ax_weight.set_title(f'Weight vs. Distance (Neuron {neuron_idx})', color='white', fontsize=14)
    ax_weight.legend(loc='best', framealpha=0.7, fontsize=10)
    ax_weight.set_facecolor('#1a1a1a')
    ax_weight.tick_params(colors='white', labelsize=10)
    for spine in ax_weight.spines.values(): spine.set_color('white')
    weight_scatter_fig.tight_layout()
    if save_path_base:
        w_save_path = f"{save_path_base}_weight_scatter.png"
        weight_scatter_fig.savefig(w_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved weight-distance relationship plot to {w_save_path}")

    # ========== FIGURE 3: DELAY VS DISTANCE ==========
    delay_scatter_fig, ax_delay = plt.subplots(figsize=scatter_figsize, facecolor='#1a1a1a')
    if outgoing_data:
        distances = [d for _, d, _, _ in outgoing_data]
        delays = [dl for _, _, _, dl in outgoing_data]
        ax_delay.scatter(distances, delays, color='cyan', alpha=0.7, label='Outgoing Delays', s=30)
        dist_range = np.linspace(0, max(distances) if distances else 1, 100)
        theo_delays = base_transmission_delay * (0.5 + 0.5 * dist_range / max_possible_dist)
        ax_delay.plot(dist_range, theo_delays, '--', color='red', alpha=0.7, label='Theoretical Delay (Linear)')
    ax_delay.grid(True, alpha=0.2)
    ax_delay.set_xlabel('Distance', color='white', fontsize=12)
    ax_delay.set_ylabel('Synaptic Delay (ms)', color='white', fontsize=12)
    ax_delay.set_title(f'Delay vs. Distance (Neuron {neuron_idx})', color='white', fontsize=14)
    ax_delay.legend(loc='best', framealpha=0.7, fontsize=10)
    ax_delay.set_facecolor('#1a1a1a')
    ax_delay.tick_params(colors='white', labelsize=10)
    for spine in ax_delay.spines.values(): spine.set_color('white')
    delay_scatter_fig.tight_layout()
    if save_path_base:
        d_save_path = f"{save_path_base}_delay_scatter.png"
        delay_scatter_fig.savefig(d_save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
        print(f"Saved delay-distance relationship plot to {d_save_path}")

    return weight_scatter_fig, delay_scatter_fig



def plot_network_connections_sparse(network, pos, stimulated_neurons=None, connected_neurons=None,
                             edge_percent=10, save_path="formal_layered_network_sim.png"):
    """
    Visualize the layered network structure with a sparse sampling of connections.
    """
    print(f"Generating sparse network structure visualization ({edge_percent}% edges)...")
    graph = network.graph; n_neurons = network.n_neurons
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a') # Ensure axes background is dark

    if stimulated_neurons is None: stimulated_neurons = []
    if connected_neurons is None: connected_neurons = []

    # Ensure position data is valid
    nodes_with_pos = [n for n in graph.nodes() if n in pos and isinstance(pos[n], (tuple, list)) and len(pos[n]) == 2]
    if len(nodes_with_pos) != graph.number_of_nodes():
        print(f"Warning: Position data missing or invalid for {graph.number_of_nodes() - len(nodes_with_pos)} nodes.")
        # Use only nodes with valid positions for visualization
        graph_nodes_to_draw = nodes_with_pos
    else:
        graph_nodes_to_draw = list(graph.nodes()) # All nodes have valid pos

    if not graph_nodes_to_draw:
        print("Error: No nodes with valid positions found. Cannot draw network.")
        plt.close(fig)
        return None

    # Node properties
    node_list = list(graph_nodes_to_draw); # Use only nodes we can draw
    node_colors, node_sizes, node_shapes = [], [], []
    nodes_by_shape = {'s': [], 'o': []}; colors_by_shape = {'s': [], 'o': []}; sizes_by_shape = {'s': [], 'o': []}

    for node in node_list:
        is_inhibitory = graph.nodes[node].get('is_inhibitory', False); shape = 's' if is_inhibitory else 'o'; node_shapes.append(shape)
        color, size = '#a5b1c2', 20 # Default other node
        if node in stimulated_neurons: color, size = '#1dd1a1', 150 # Stimulated
        elif node in connected_neurons: color, size = '#ffae42', 80 # Connected (used?)
        node_colors.append(color); node_sizes.append(size)
        # Group by shape for drawing
        nodes_by_shape[shape].append(node); colors_by_shape[shape].append(color); sizes_by_shape[shape].append(size)

    # Draw nodes grouped by shape
    for shape_marker, nodelist_for_shape in nodes_by_shape.items():
         if nodelist_for_shape:
              # Extract sizes and colors corresponding to this shape's nodelist
              sizes = [sizes_by_shape[shape_marker][i] for i, n in enumerate(nodes_by_shape[shape_marker])]
              colors = [colors_by_shape[shape_marker][i] for i, n in enumerate(nodes_by_shape[shape_marker])]
              nx.draw_networkx_nodes(graph, pos, nodelist=nodelist_for_shape, node_size=sizes, node_color=colors, node_shape=shape_marker, ax=ax, alpha=0.9)


    # Edge properties and drawing
    all_edges = list(graph.edges(data=True)); random.shuffle(all_edges);
    target_edge_count = int(len(all_edges) * edge_percent / 100);
    edges_to_consider = all_edges[:target_edge_count]
    edges_to_draw_list, edge_colors, widths = [], [], []
    feedforward_color, feedback_color, recurrent_color = '#d62728', '#1f77b4', '#808080' # Red, Blue, Grey

    for u, v, data in edges_to_consider:
        # Ensure both nodes have positions before attempting to draw edge
        if u not in graph.nodes or v not in graph.nodes or u not in pos or v not in pos: continue
        edges_to_draw_list.append((u, v));
        layer_u = graph.nodes[u].get('layer', -1) # Default layer if missing
        layer_v = graph.nodes[v].get('layer', -1)
        # Determine edge type based on layer difference
        if layer_u != -1 and layer_v != -1: # Only color if layers are defined
            if layer_v > layer_u: edge_colors.append(feedforward_color); widths.append(0.6)  # Feedforward
            elif layer_v < layer_u: edge_colors.append(feedback_color); widths.append(0.6)   # Feedback
            else: edge_colors.append(recurrent_color); widths.append(0.4)                     # Recurrent
        else:
             edge_colors.append(recurrent_color); widths.append(0.4) # Default if layer info missing

    if edges_to_draw_list:
        try:
            # Create size map based on nodes that have position and were processed
            size_map = {node: node_sizes[node_list.index(node)] for node in node_list}
            # Ensure node_list_ordered_for_edges only contains nodes present in the graph
            nodes_in_graph = list(graph.nodes())
            # Map sizes, using a default for any potential mismatch (should ideally not happen)
            node_size_list_ordered = [size_map.get(n, 20) for n in nodes_in_graph]

            # Draw the edges
            nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw_list,
                                  edge_color=edge_colors, width=widths, alpha=0.5,
                                  arrows=True, arrowsize=8,
                                  node_size=node_size_list_ordered, # Pass node sizes for arrow clipping
                                  ax=ax)
        except Exception as e:
             print(f"Error drawing edges: {e}. Ensure node sizes map correctly to nodes in edgelist.")


    # Labels (optional, can be cluttered)
    # labels = {n: f"C{n}" for n in connected_neurons}
    # labels_with_pos = {n: lbl for n, lbl in labels.items() if n in pos}
    # if labels_with_pos: nx.draw_networkx_labels(graph, pos, labels=labels_with_pos, font_color='white', font_weight='bold', font_size=8, ax=ax)

    # Title and Legend
    n_excitatory = sum(1 for n in graph.nodes() if not graph.nodes[n].get('is_inhibitory', False)); n_inhibitory = n_neurons - n_excitatory
    title = (f"Formal {network.graph.graph.get('num_layers', '?')}-Layer Network - {n_neurons} Neurons ({n_excitatory} Exc, {n_inhibitory} Inhib)\nShowing {len(edges_to_draw_list)} ({edge_percent:.1f}%) of {len(all_edges)} connections (Post Pruning)")
    ax.set_title(title, color='white', fontsize=14)

    # Legend elements based on colors used
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Inhib)', linestyle='None'),
        # plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Exc)', linestyle='None'), # If using connected neurons
        # plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], color=feedforward_color, lw=1.5, label='Feedforward Conn.'),
        plt.Line2D([0], [0], color=feedback_color, lw=1.5, label='Feedback Conn.'),
        plt.Line2D([0], [0], color=recurrent_color, lw=1.5, label='Recurrent Conn.')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize='small');
    ax.set_axis_off(); # Turn off axis box and ticks
    plt.tight_layout() # Adjust layout

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight');
            print(f"Saved network structure visualization to {save_path}")
        except Exception as e:
            print(f"Error saving network structure plot: {e}")
    # plt.close(fig) # Keep open if running interactively, close otherwise
    return fig
# --- END RE-ADDED plot_network_connections_sparse ---


# --- Main Experiment Function ---
def run_6_layer_experiment(
    n_layers_list,
    inhibitory_fraction,
    connection_probs,
    duration,
    dt,
    neuron_params,
    stimulation_params,
    pruning_threshold,
    weight_min, weight_max,
    base_transmission_delay, # Use base delay
    random_seed=42,
    animate=False):
    """ Run experiment with distance-dependent delays. """
    print("--- Running 6-Layer Network Experiment (Distance-Dependent Delay) ---")
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    # Extract connection probabilities
    default_exc_rec = connection_probs['exc_recurrent']
    default_inh_rec = connection_probs['inh_recurrent']
    default_ff_1 = connection_probs['feedforward_1']
    default_ff_2 = connection_probs['feedforward_2']
    default_fb_1 = connection_probs['feedback_1']
    default_fb_2 = connection_probs['feedback_2']
    long_ff_prob = connection_probs.get('long_feedforward', 0.01)
    long_fb_prob = connection_probs.get('long_feedback', 0.005)

    # Extract stimulation parameters
    stochastic_stim = stimulation_params['stochastic']
    no_stimulation = stimulation_params['none']
    stim_interval = stimulation_params['interval_ms']
    stim_interval_strength = stimulation_params['strength']
    stim_fraction = stimulation_params['fraction']
    stim_pulse_duration_ms = stimulation_params.get('pulse_duration_ms', dt)

    # 1. Create Network Instance
    print(f"Initializing network object for {total_neurons} neurons...")
    network = ExtendedNeuronalNetworkWithReversal(n_neurons=total_neurons, inhibitory_fraction=inhibitory_fraction, **neuron_params)
    network.graph.graph['num_layers'] = num_layers

    # 2. Manually Create Neurons, Nodes, and Original Positions
    print("Creating neurons and assigning layers/original positions...")
    pos = {}
    x_coords = np.linspace(0.1, 0.9, num_layers)
    horizontal_spread = 0.04
    vertical_spread = total_neurons / 20.0
    layer_indices = []
    start_idx = 0

    neuron_init_params = inspect.signature(LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_params if k != 'self' and k!= 'is_inhibitory'}
    filtered_neuron_params = {k: network.neuron_params[k] for k in valid_neuron_keys if k in network.neuron_params}

    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num-1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))
        print(f"  Layer {layer_num}: Neurons {start_idx} to {end_idx-1}")
        for current_node_index in range(start_idx, end_idx):
             is_inhib = random.random() < network.inhibitory_fraction
             neuron = LIFNeuronWithReversal(is_inhibitory=is_inhib, **filtered_neuron_params)
             neuron.layer = layer_num
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread), random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[current_node_index] = node_pos; network.add_neuron(neuron, current_node_index, node_pos, layer_num)
        start_idx = end_idx

    first_layer_start, first_layer_end = layer_indices[0]
    first_layer_indices = list(range(first_layer_start, first_layer_end))
    print(f"Targeting stimulation to Layer 1 indices: {first_layer_start}-{first_layer_end-1}")

    # --- Calculate Max Distance for Delay Normalization ---
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    max_possible_dist = np.sqrt((max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2) if len(pos)>1 else 1.0
    if max_possible_dist < 1e-6: max_possible_dist = 1.0 # Avoid division by zero if all points coincide

    # 3. Manually Add Connections with Distance-Dependent Delays
    print(f"Creating initial connections with distance-dependent delays (Base: {base_transmission_delay} ms)...")
    connection_count = 0
    min_delay = 0.1 # Minimum possible delay
    for i in range(total_neurons):
        if i >= len(network.neurons) or not network.neurons[i] or i not in network.graph.nodes or i not in pos: continue
        is_source_inhibitory = network.neurons[i].is_inhibitory; layer_i = network.graph.nodes[i]['layer']
        pos_i = pos[i]
        for j in range(total_neurons):
            if i == j or j not in network.graph.nodes or j not in pos: continue
            layer_j = network.graph.nodes[j]['layer']
            pos_j = pos[j]
            prob = 0.0; weight = 0.0; connect = False

            if is_source_inhibitory:
                if layer_i == layer_j:
                    prob = default_inh_rec
                    if random.random() < prob: weight = -1.0 * random.uniform(weight_min, weight_max); connect = True
            else:
                layer_diff = layer_j - layer_i
                if layer_diff == 0: prob = default_exc_rec
                elif layer_diff == 1: prob = default_ff_1
                elif layer_diff == 2: prob = default_ff_2
                elif layer_diff == -1: prob = default_fb_1
                elif layer_diff == -2: prob = default_fb_2
                elif layer_diff > 2: prob = long_ff_prob
                elif layer_diff < -2: prob = long_fb_prob
                if random.random() < prob: weight = 1.0 * random.uniform(weight_min, weight_max); connect = True

            if connect:
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                # Linear scaling: delay = base * (min_factor + scale_factor * normalized_distance)
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist))
                delay = max(min_delay, delay) # Ensure minimum delay
                network.add_connection(i, j, weight, delay=delay)
                connection_count += 1

    print(f"Initially added {connection_count} connections."); print(f"Initial edge count in graph: {network.graph.number_of_edges()}.")

    # 4. Prune Weak Connections
    print(f"Pruning connections with |weight| < {pruning_threshold}...")
    network.prune_weak_connections(threshold=pruning_threshold)
    print(f"Edge count after pruning: {network.graph.number_of_edges()}.")


    print("Visualizing network structure (sparse)...")
    # Define placeholder lists if needed, or use actual stimulated/connected neurons if tracked
    stimulated_targets_for_vis = first_layer_indices[:min(5, len(first_layer_indices))] # Example: show first 5 targets
    connected_for_vis = [] # Define if you track specific connected neurons

    plot_network_connections_sparse(
         network=network,
         pos=pos,
         stimulated_neurons=stimulated_targets_for_vis, # Pass the list of stimulated neurons
         connected_neurons=connected_for_vis, # Pass list if tracking connected ones
         edge_percent=5, # Show 5% of edges
         save_path=f"{num_layers}_layer_network_structure.png"
    )

    # 5. Visualize Distance Dependences
    print("Visualizing distance dependencies...")
    neuron_to_vis = first_layer_start # Visualize for first neuron in Layer 1
    if neuron_to_vis in network.graph.nodes() and neuron_to_vis in pos:
         vis_figs = visualize_distance_dependences( # Store figure handles
              network=network,
              pos=pos,
              neuron_idx=neuron_to_vis,
              base_transmission_delay=base_transmission_delay,
              save_path_base=f"{num_layers}_layer_neuron{neuron_to_vis}_dist_dependence"
         )
         # Close the generated figures immediately after saving if desired
         if vis_figs:
              for fig in vis_figs:
                   if fig: plt.close(fig)
    else:
         print(f"Warning: Cannot visualize distance for neuron {neuron_to_vis}, not found.")


    # 6. Run Simulation
    print(f"\nStarting simulation for {duration} ms...")
    track_neurons = [] # Disable tracking for speed
    network.reset_all()
    activity_record, neuron_data, stimulation_record = run_unified_simulation(
        network, duration=duration, dt=dt,
        stim_interval=stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stim_target_indices=first_layer_indices,
        stim_pulse_duration_ms=stim_pulse_duration_ms,
        track_neurons=track_neurons,
        stochastic_stim=stochastic_stim,
        no_stimulation=no_stimulation
    )

    # 7. Post-Simulation Visualizations
    print("\nGenerating post-simulation visualizations...")
    stim_times = stimulation_record.get('pulse_starts', []) if not no_stimulation else None

    plot_activity_and_layer_psth(
         network=network, activity_record=activity_record, layer_indices=layer_indices,
         dt=dt, stim_times=stim_times, bin_width_ms=2.0,
         save_path=f"{num_layers}_layer_activity_psth.png"
    )

    plot_layer_wise_raster(
        network=network, activity_record=activity_record, layer_indices=layer_indices,
        dt=dt, stim_times=stim_times,
        save_path=f"{num_layers}_layer_raster.png"
    )

    if animate: # Add this line
        visualize_activity_layout_grid( # Indent this block
            network=network, pos=pos, activity_record=activity_record, dt=dt,
            stim_record=stimulation_record,
            grid_resolution=(120, 180),
            save_path=f"{num_layers}_layer_layout_grid_animation.gif",
            max_frames=int(duration*10), fps=25
        )
    else: # Optional: Add a print statement if not animating
        print("Skipping layout grid animation generation (animate=False).") # Optional

    print("\n--- Experiment Complete ---")
    return network, activity_record, neuron_data, stimulation_record












# --- Run the Experiment ---
if __name__ == "__main__":
    # -------------------------------------
    # --- Network Parameters ---
    layers_config = [150, 175, 175, 175, 175, 150]
    inhib_frac = 0.4 

    # --- Connection Probabilities ---
    conn_probs = {
        'exc_recurrent': 0.05,
        'inh_recurrent': 0.1,
        'feedforward_1': 0.20,
        'feedforward_2': 0.05,
        'feedback_1': 0.05,
        'feedback_2': 0.01,
        'long_feedforward': 0.005,
        'long_feedback': 0.002
    }

    # --- Neuron Parameters ---
    neuron_config = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 10.0, 'tau_ref': 2.0, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0,
        'v_noise_amp': 0.5, 'i_noise_amp': 0.0088, # Low noise
        'adaptation_increment': 0.5,
        'tau_adaptation': 100,
        'weight_scale': 0.1
    }

    # --- Weight Range ---
    weight_config = {
        'min': 0.004,
        'max': 0.02
    }

    # --- Pruning ---
    prune_thresh = 0.00

    # --- MODIFIED: Base Delay ---
    base_delay = .5 # ms (average delay for distance calculation)

    # --- Simulation Parameters ---
    sim_duration = 10_000
    sim_dt = 0.1

    # --- Stimulation Parameters ---
    stim_config = {
        'stochastic': False,
        'none': False, # Keep stim off for now
        'interval_ms': 50,
        'strength': 100,
        'fraction': 0.5,
        'pulse_duration_ms': 10
    }
    # --- Animate ---
    animate = False

    # --- Random Seed ---
    seed = 123
    # -------------------------------------

    # Call experiment function with all parameters
    network_obj, activity, tracked_data, stim_info = run_6_layer_experiment(
        n_layers_list=layers_config,
        inhibitory_fraction=inhib_frac,
        connection_probs=conn_probs,
        duration=sim_duration,
        dt=sim_dt,
        neuron_params=neuron_config,
        stimulation_params=stim_config,
        pruning_threshold=prune_thresh,
        weight_min=weight_config['min'],
        weight_max=weight_config['max'],
        # --- Pass base delay ---
        base_transmission_delay = base_delay,
        random_seed=seed,
        animate=animate
    )

    # Close figures generated by visualize_distance_dependences if needed
    plt.close('all') # Close all figures automatically after saving

    # No plt.show() needed if running non-interactively
    # plt.show()
