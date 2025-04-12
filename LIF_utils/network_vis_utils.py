import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import matplotlib.animation as animation # Ensure animation is imported
from tqdm import tqdm

# Set dark style for plots
plt.style.use('dark_background')

def plot_network_connections_sparse(network, stimulated_neurons=None, connected_neurons=None, 
                             edge_percent=1, save_path="network_connections_sparse.png"):
    """
    Visualize the network connectivity with focus on the stimulated and connected neurons,
    showing only a specified percentage of edges to reduce visual clutter.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#1a1a1a')
    
    # Get network graph from the network object
    G = network.graph
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # If no positions are available, use spring layout
    if not pos:
        pos = nx.spring_layout(G)
    
    # Set defaults if not provided
    if stimulated_neurons is None:
        stimulated_neurons = []
    if connected_neurons is None:
        connected_neurons = []
    
    # Draw all neurons with light color
    node_colors = []
    node_sizes = []
    node_shapes = []
    
    for node in G.nodes():
        if node in stimulated_neurons:
            node_colors.append('#1dd1a1')  # Green for stimulated
            node_sizes.append(200)
            shape = 's' if G.nodes[node]['is_inhibitory'] else 'o'
            node_shapes.append(shape)
        elif node in connected_neurons:
            node_colors.append('#ff6b6b')  # Red for connected
            node_sizes.append(100)
            shape = 's' if G.nodes[node]['is_inhibitory'] else 'o'
            node_shapes.append(shape)
        else:
            node_colors.append('#a5b1c2')  # Grey for others
            node_sizes.append(10)
            shape = 's' if G.nodes[node]['is_inhibitory'] else 'o'
            node_shapes.append(shape)
    
    # Draw nodes by groups so we can control shape by node
    for i, node in enumerate(G.nodes()):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], 
            node_color=[node_colors[i]], 
            node_size=node_sizes[i],
            node_shape=node_shapes[i],
            ax=ax
        )
    
    # Prepare edge colors based on excitatory/inhibitory
    all_edges = list(G.edges(data=True))
    
    # Keep all edges between stimulated and connected neurons
    important_edges = []
    other_edges = []
    
    for u, v, data in all_edges:
        if u in stimulated_neurons and v in connected_neurons:
            # Keep all important connections
            important_edges.append((u, v, data))
        else:
            # Sample from other connections
            other_edges.append((u, v, data))
    
    # Calculate how many additional edges to show beyond the important ones
    total_edges = len(all_edges)
    target_edges = int(total_edges * edge_percent / 100)
    
    # Make sure we don't exceed the total number of edges
    additional_edges_needed = min(target_edges - len(important_edges), len(other_edges))
    
    # Sample random edges if we need more
    if additional_edges_needed > 0:
        sampled_edges = random.sample(other_edges, additional_edges_needed)
    else:
        sampled_edges = []
    
    # Combine the important edges with the sampled ones
    edges_to_draw = important_edges + sampled_edges
    
    # Prepare edge colors and widths
    edge_colors = []
    widths = []
    
    # Create lists of edges for drawing
    edges_to_draw_list = []
    
    for u, v, data in edges_to_draw:
        edges_to_draw_list.append((u, v))
        
        if u in stimulated_neurons and v in connected_neurons:
            # Highlight connections between stimulated and tracked neurons
            if data['weight'] > 0:
                edge_colors.append('#1dd1a1')  # Green for excitatory
            else:
                edge_colors.append('#ff6b6b')  # Red for inhibitory
            widths.append(2.0)
        else:
            # Other connections with more subtle colors
            if data['weight'] > 0:
                edge_colors.append('#1f77b4')  # Blue for excitatory
            else:
                edge_colors.append('#d62728')  # Red for inhibitory
            widths.append(0.5)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        edgelist=edges_to_draw_list,
        edge_color=edge_colors,
        width=widths,
        alpha=0.6,
        ax=ax
    )
    
    # Add labels for only stimulated and connected neurons
    labels = {}
    for node in stimulated_neurons:
        labels[node] = f"S{node}"
    for node in connected_neurons:
        labels[node] = f"C{node}"
    
    nx.draw_networkx_labels(
        G, pos, 
        labels=labels,
        font_color='white',
        font_weight='bold',
        ax=ax
    )
    
    # Create a title with network information
    n_excitatory = sum(1 for n in G.nodes() if not G.nodes[n]['is_inhibitory'])
    n_inhibitory = network.n_neurons - n_excitatory
    
    title = (f"Network Connectivity - {network.n_neurons} Neurons "
             f"({n_excitatory} Excitatory, {n_inhibitory} Inhibitory)\n"
             f"Showing {len(edges_to_draw)} of {total_edges} connections ({edge_percent}%)")
    ax.set_title(title, color='white', fontsize=16)
    
    # Add a legend
    stimulated_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1dd1a1',
                                 markersize=15, label='Stimulated (Exc)')
    stimulated_inhib = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1dd1a1',
                                 markersize=15, label='Stimulated (Inhib)')
    connected_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff6b6b',
                               markersize=15, label='Connected (Exc)')
    connected_inhib = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff6b6b',
                               markersize=15, label='Connected (Inhib)')
    other_exc = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#a5b1c2',
                         markersize=15, label='Other (Exc)')
    other_inhib = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#a5b1c2',
                           markersize=15, label='Other (Inhib)')
    exc_conn = plt.Line2D([0], [0], color='#1f77b4', lw=1.5, label='Excitatory Connection')
    inh_conn = plt.Line2D([0], [0], color='#d62728', lw=1.5, label='Inhibitory Connection')
    
    ax.legend(handles=[stimulated_patch, stimulated_inhib, connected_patch, 
                      connected_inhib, other_exc, other_inhib,
                      exc_conn, inh_conn], 
             loc='upper right', framealpha=0.7)
    
    # Turn off axis
    ax.set_axis_off()
    
    # Save the network connectivity visualization
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sparse network connectivity visualization to {save_path}")
    
    return fig

def visualize_distance_weights(network, neuron_idx, network_figsize=(18, 18), 
                               scatter_figsize=(10, 8), save_path=None):
    """
    Creates two separate figures showing how connection weights decay with distance:
    1. A network visualization showing outgoing connections from a specific neuron
    2. A weight vs. distance scatter plot showing the distance-weight relationship
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object
    neuron_idx : int
        Index of the neuron to highlight
    network_figsize : tuple
        Figure size for the network visualization (width, height) in inches
    scatter_figsize : tuple
        Figure size for the weight vs. distance plot (width, height) in inches
    save_path : str or None
        Base path to save the visualizations, if provided.
        Will append "_network" and "_scatter" to the base name.
            
    Returns:
    --------
    tuple
        (network_fig, scatter_fig) - The two figure objects
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if not hasattr(network, 'neuron_grid_positions'):
        print("Error: This visualization requires spatial positioning of neurons.")
        return None, None
    
    # Get the neuron's position
    center_pos = network.neuron_grid_positions[neuron_idx]
    
    # Collect outgoing and incoming weights with distances
    outgoing_data = []
    incoming_data = []
    
    for j in range(network.n_neurons):
        if j != neuron_idx:
            # Outgoing connection
            if network.weights[neuron_idx, j] != 0:
                target_pos = network.neuron_grid_positions[j]
                dist = np.sqrt((center_pos[0] - target_pos[0])**2 + (center_pos[1] - target_pos[1])**2)
                outgoing_data.append((j, dist, network.weights[neuron_idx, j]))
            
            # Incoming connection
            if network.weights[j, neuron_idx] != 0:
                source_pos = network.neuron_grid_positions[j]
                dist = np.sqrt((center_pos[0] - source_pos[0])**2 + (center_pos[1] - source_pos[1])**2)
                incoming_data.append((j, dist, network.weights[j, neuron_idx]))
    
    # ========== FIGURE 1: NETWORK VISUALIZATION ==========
    # Create figure with zero padding to maximize space utilization
    network_fig = plt.figure(figsize=network_figsize, facecolor='#1a1a1a', constrained_layout=False)
    
    # Create subplot that exactly fills the figure - use the entire canvas area
    ax1 = network_fig.add_axes([0, 0, 1, 0.95], aspect='equal', frameon=False)  
    
    # Add title at the very top with minimal space consumption
    n_connections = len(outgoing_data)
    title_text = f"Neuron {neuron_idx} - {n_connections} Outgoing Connections"
    network_fig.text(0.5, 0.98, title_text, color='white', fontsize=16, ha='center', va='top')
    
    # Use network positions for circular layout
    pos = {i: (col, row) for i, (row, col) in network.neuron_grid_positions.items()}
    
    # Create a list of outgoing connections
    outgoing_targets = [j for j, _, _ in outgoing_data]
    
    # Split nodes into three groups
    source_nodes = [neuron_idx]
    connected_nodes = outgoing_targets
    other_nodes = [i for i in range(network.n_neurons) if i != neuron_idx and i not in outgoing_targets]
    
    # Draw other neurons first (small and gray)
    gray_color = '#a5b1c2'  # Light gray
    nx.draw_networkx_nodes(
        network.graph, pos,
        nodelist=other_nodes,
        node_color=[gray_color] * len(other_nodes),  # Same color for all
        node_size=15,  # Slightly larger for dedicated figure
        node_shape='o',
        alpha=0.5,
        ax=ax1
    )
    
    # Draw connected neurons
    nx.draw_networkx_nodes(
        network.graph, pos,
        nodelist=connected_nodes,
        node_color='#ff6b6b',  # Red
        node_size=40,  # Larger for dedicated figure
        node_shape='o',
        alpha=0.8,
        ax=ax1
    )
    
    # Draw source neuron last so it's on top
    nx.draw_networkx_nodes(
        network.graph, pos,
        nodelist=source_nodes,
        node_color='#1dd1a1',  # Green
        node_size=120,  # Larger for dedicated figure
        node_shape='o',
        alpha=1.0,
        ax=ax1
    )
    
    # Sort outgoing connections by absolute weight (weakest first, strongest last)
    sorted_outgoing_data = sorted(outgoing_data, key=lambda x: abs(x[2]))
    
    # Collect edges and prepare colors based on weight
    if sorted_outgoing_data:
        # Get the absolute weights for color mapping
        abs_weights = [abs(w) for _, _, w in sorted_outgoing_data]
        
        # Create a color mapping based on absolute weight values
        min_weight = min(abs_weights)
        max_weight = max(abs_weights)
        norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
        
        # Use plasma colormap (purple to yellow)
        cmap = cm.plasma
        
        # Draw connections one by one, weakest first
        for target, _, weight in sorted_outgoing_data:
            edge = [(neuron_idx, target)]
            
            # Color based on absolute weight strength
            abs_weight = abs(weight)
            edge_color = cmap(norm(abs_weight))
            
            # Width based on absolute weight
            edge_width = max(1.5, min(6.0, abs(weight) * 100))
            
            # Draw this edge
            nx.draw_networkx_edges(
                network.graph, pos,
                edgelist=edge,
                edge_color=[edge_color],
                width=edge_width,
                arrows=False,
                alpha=0.8,
                ax=ax1
            )
    
    # Ensure the plot is perfectly circular with maximum size
    all_x = [pos[i][0] for i in range(network.n_neurons)]
    all_y = [pos[i][1] for i in range(network.n_neurons)]
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # Find the most extreme value to create square boundaries
    # Calculate the center and the distance to the farthest point
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Find distance to furthest point
    max_dist = max([
        np.sqrt((x - center_x)**2 + (y - center_y)**2) 
        for x, y in zip(all_x, all_y)
    ])
    
    # Add just 1% margin to avoid cutting off nodes
    max_dist *= 1.01
    
    # Set identical limits for x and y to ensure perfect circle with maximum size
    ax1.set_xlim(center_x - max_dist, center_x + max_dist)
    ax1.set_ylim(center_y - max_dist, center_y + max_dist)
    
    ax1.set_facecolor('#1a1a1a')
    ax1.axis('off')
    
    # Save network figure if path provided
    if save_path:
        network_save_path = save_path.replace('.png', '_network.png') if '.png' in save_path else f"{save_path}_network.png"
        network_fig.savefig(network_save_path, dpi=300, facecolor='#1a1a1a', bbox_inches='tight', pad_inches=0)
        print(f"Saved network distance visualization to {network_save_path}")
    
    # ========== FIGURE 2: WEIGHT VS DISTANCE PLOT ==========
    scatter_fig = plt.figure(figsize=scatter_figsize, facecolor='#1a1a1a')
    ax2 = scatter_fig.add_subplot(111)
    
    outgoing_dist = [d for _, d, _ in outgoing_data]
    outgoing_weights = [w for _, _, w in outgoing_data]
    incoming_dist = [d for _, d, _ in incoming_data]
    incoming_weights = [w for _, _, w in incoming_data]
    
    # Plot outgoing connections
    ax2.scatter(outgoing_dist, outgoing_weights, color='yellow', alpha=0.7, label='Outgoing', s=50)
    # Plot incoming connections
    ax2.scatter(incoming_dist, incoming_weights, color='cyan', alpha=0.7, label='Incoming', s=50)
    
    # Plot exponential decay reference line
    if outgoing_data or incoming_data:
        all_dist = outgoing_dist + incoming_dist
        max_dist = max(all_dist) if all_dist else 10
        
        dist_range = np.linspace(0, max_dist * 1.05, 100)  # 5% margin
        
        # Use the class's distance_lambda parameter if it exists
        distance_lambda = getattr(network, 'distance_lambda', 0.2)
        
        # Plot excitatory reference line
        ref_weight_exc = network.weight_scale * np.exp(-distance_lambda * dist_range)
        ax2.plot(dist_range, ref_weight_exc, '--', color='green', alpha=0.7, label='Exc. Reference')
        
        # Plot inhibitory reference line
        ref_weight_inh = -network.weight_scale * 4.0 * np.exp(-distance_lambda * dist_range)
        ax2.plot(dist_range, ref_weight_inh, '--', color='red', alpha=0.7, label='Inh. Reference')
    
    # Determine optimal y-axis limits based on actual data
    all_weights = outgoing_weights + incoming_weights
    if all_weights:
        min_weight = min(all_weights)
        max_weight = max(all_weights)
        weight_range = max_weight - min_weight
        
        # Add a small buffer (5%) to avoid cutting off points
        y_min = min_weight - 0.05 * weight_range
        y_max = max_weight + 0.05 * weight_range
        
        # Also ensure the reference lines are visible
        if locals().get('ref_weight_exc') is not None and locals().get('ref_weight_inh') is not None:
            y_min = min(y_min, min(ref_weight_inh))
            y_max = max(y_max, max(ref_weight_exc))
        
        ax2.set_ylim(y_min, y_max)
    
    # Add horizontal line at y=0
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add grid
    ax2.grid(True, alpha=0.2)
    
    # Style the scatter plot - larger fonts for dedicated figure
    ax2.set_xlabel('Distance (grid units)', color='white', fontsize=14)
    ax2.set_ylabel('Synaptic Weight', color='white', fontsize=14)
    scatter_fig.suptitle('Weight vs. Distance', color='white', fontsize=16)
    
    # Add legend
    ax2.legend(loc='upper right', framealpha=0.7, fontsize=12)
    
    ax2.set_facecolor('#1a1a1a')
    ax2.tick_params(colors='white', labelsize=12)
    
    # Style spines
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')
    
    # Maximize the plot in the figure
    scatter_fig.tight_layout()
    
    # Save scatter figure if path provided
    if save_path:
        scatter_save_path = save_path.replace('.png', '_scatter.png') if '.png' in save_path else f"{save_path}_scatter.png"
        scatter_fig.savefig(scatter_save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved weight-distance relationship plot to {scatter_save_path}")
    
    # Return both figures
    return network_fig, scatter_fig




# Function to plot the network structure sparsely
def Layered_plot_network_connections_sparse(network, pos, stimulated_neurons=None, connected_neurons=None,
                             edge_percent=10, save_path="formal_layered_network_sim.png"):
    """
    Visualizes the network connectivity, highlighting specific neurons and showing only a
    percentage of edges to reduce clutter, colored by connection type (feedforward, feedback, recurrent).

    Args:
        network: The ExtendedNeuronalNetworkWithReversal object.
        pos (dict): Dictionary mapping neuron indices to (x, y) positions.
        stimulated_neurons (list, optional): Indices of neurons to highlight as stimulated.
        connected_neurons (list, optional): Indices of neurons to highlight as connected.
        edge_percent (float): Percentage of total connections to display.
        save_path (str, optional): File path to save the visualization.
    """
    graph = network.graph; n_neurons = network.n_neurons
    # Create figure and axes with dark background
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a') # Explicitly set axes background

    # Initialize lists if not provided
    if stimulated_neurons is None: stimulated_neurons = []
    if connected_neurons is None: connected_neurons = []

    # Check for nodes missing position data
    nodes_with_pos = [n for n in graph.nodes() if n in pos and isinstance(pos[n], (tuple, list)) and len(pos[n]) == 2]
    if len(nodes_with_pos) != graph.number_of_nodes():
        print(f"Warning: Position data missing or invalid for {graph.number_of_nodes() - len(nodes_with_pos)} nodes.")
    graph_nodes_to_draw = nodes_with_pos # Only draw nodes with valid positions

    if not graph_nodes_to_draw:
        print("Error: No nodes with valid positions found. Cannot draw network.")
        plt.close(fig)
        return None

    # --- Node Drawing Preparation ---
    node_list = list(graph_nodes_to_draw)
    node_colors, node_sizes, node_shapes = [], [], []
    # Group nodes by shape (inhibitory 's', excitatory 'o') for efficient drawing
    nodes_by_shape = {'s': [], 'o': []}
    colors_by_shape = {'s': [], 'o': []}
    sizes_by_shape = {'s': [], 'o': []}

    # Determine color, size, and shape for each node
    for node in node_list:
        is_inhibitory = graph.nodes[node].get('is_inhibitory', False)
        shape = 's' if is_inhibitory else 'o'
        node_shapes.append(shape)
        # Default style for "other" neurons
        color, size = '#a5b1c2', 20
        # Style for stimulated neurons
        if node in stimulated_neurons: color, size = '#1dd1a1', 150
        # Style for connected neurons (if used)
        elif node in connected_neurons: color, size = '#ffae42', 80 # Orange/Yellow
        node_colors.append(color)
        node_sizes.append(size)
        # Add node and its properties to the shape group
        nodes_by_shape[shape].append(node)
        colors_by_shape[shape].append(color)
        sizes_by_shape[shape].append(size)

    # --- Draw Nodes ---
    # Draw nodes shape by shape for correct marker type
    for shape_marker, nodelist_for_shape in nodes_by_shape.items():
         if nodelist_for_shape:
              # Get the corresponding sizes and colors for this group
              sizes = [sizes_by_shape[shape_marker][i] for i, n in enumerate(nodes_by_shape[shape_marker])]
              colors = [colors_by_shape[shape_marker][i] for i, n in enumerate(nodes_by_shape[shape_marker])]
              # Draw the nodes
              nx.draw_networkx_nodes(graph, pos, nodelist=nodelist_for_shape, node_size=sizes,
                                   node_color=colors, node_shape=shape_marker, ax=ax, alpha=0.9)

    # --- Edge Drawing Preparation ---
    all_edges = list(graph.edges(data=True))
    random.shuffle(all_edges) # Shuffle for random sampling
    # Calculate number of edges to draw based on percentage
    target_edge_count = int(len(all_edges) * edge_percent / 100)
    edges_to_consider = all_edges[:target_edge_count] # Select subset

    edges_to_draw_list, edge_colors, widths = [], [], []
    # Define colors for different connection types
    feedforward_color, feedback_color, recurrent_color = '#d62728', '#1f77b4', '#808080' # Red, Blue, Grey

    # Determine color and width for each edge to be drawn
    for u, v, data in edges_to_consider:
        # Skip if source or target node position is missing
        if u not in graph.nodes or v not in graph.nodes or u not in pos or v not in pos: continue
        edges_to_draw_list.append((u, v))
        # Get layer information
        layer_u = graph.nodes[u].get('layer', -1) # Use -1 if layer is not defined
        layer_v = graph.nodes[v].get('layer', -1)

        # Assign color and width based on layer difference
        if layer_u != -1 and layer_v != -1: # Only if layers are defined
            if layer_v > layer_u: # Feedforward (to higher layer)
                edge_colors.append(feedforward_color)
                widths.append(0.6)
            elif layer_v < layer_u: # Feedback (to lower layer)
                edge_colors.append(feedback_color)
                widths.append(0.6)
            else: # Recurrent (within same layer)
                edge_colors.append(recurrent_color)
                widths.append(0.4)
        else: # Default color if layer info is missing
             edge_colors.append(recurrent_color)
             widths.append(0.4)

    # --- Draw Edges ---
    if edges_to_draw_list:
        try:
            # Map node sizes for accurate arrow placement (avoids overlap with large nodes)
            size_map = {node: node_sizes[node_list.index(node)] for node in node_list}
            # Get sizes in the order networkx expects (order of graph.nodes())
            nodes_in_graph = list(graph.nodes())
            node_size_list_ordered = [size_map.get(n, 20) for n in nodes_in_graph] # Use default if mismatch

            # Draw the selected edges
            nx.draw_networkx_edges(graph, pos, edgelist=edges_to_draw_list,
                                  edge_color=edge_colors, width=widths, alpha=0.5,
                                  arrows=True, arrowsize=8, # Add arrows
                                  node_size=node_size_list_ordered, # Provide node sizes for clipping
                                  ax=ax)
        except Exception as e:
             # Catch potential errors during drawing (e.g., size mismatch)
             print(f"Error drawing edges: {e}. Ensure node sizes map correctly.")

    # --- Labels (Optional) ---
    # Labels can make the plot cluttered, uncomment if needed
    # labels = {n: f"C{n}" for n in connected_neurons} # Example: Label connected neurons
    # labels_with_pos = {n: lbl for n, lbl in labels.items() if n in pos}
    # if labels_with_pos: nx.draw_networkx_labels(graph, pos, labels=labels_with_pos, font_color='white', font_weight='bold', font_size=8, ax=ax)

    # --- Title and Legend ---
    # Calculate number of excitatory/inhibitory neurons
    n_excitatory = sum(1 for n in graph.nodes() if not graph.nodes[n].get('is_inhibitory', False))
    n_inhibitory = n_neurons - n_excitatory
    # Create plot title
    title = (f"Formal {network.graph.graph.get('num_layers', '?')}-Layer Network - {n_neurons} Neurons ({n_excitatory} Exc, {n_inhibitory} Inhib)\n"
             f"Showing {len(edges_to_draw_list)} ({edge_percent:.1f}%) of {len(all_edges)} connections (Post Pruning)")
    ax.set_title(title, color='white', fontsize=14)

    # Create legend elements manually for clarity
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1dd1a1', markersize=10, label='Stimulated Target (Inhib)', linestyle='None'),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Exc)', linestyle='None'), # Uncomment if using connected_neurons
        #plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffae42', markersize=8, label='Connected (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Exc)', linestyle='None'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#a5b1c2', markersize=6, label='Other (Inhib)', linestyle='None'),
        plt.Line2D([0], [0], color=feedforward_color, lw=1.5, label='Feedforward Conn.'),
        plt.Line2D([0], [0], color=feedback_color, lw=1.5, label='Feedback Conn.'),
        plt.Line2D([0], [0], color=recurrent_color, lw=1.5, label='Recurrent Conn.')
    ]
    # Add legend to plot
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7, fontsize='small')
    ax.set_axis_off() # Hide axis borders and ticks
    plt.tight_layout() # Adjust plot layout

    # --- Save Figure ---
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved network structure visualization to {save_path}")
        except Exception as e:
            print(f"Error saving network structure plot: {e}")
    # plt.close(fig) # Close the figure after saving if running in a script
    return fig
# --- END plot_network_connections_sparse ---


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Ensure plt.style.use('dark_background') is potentially set earlier in the file or calling script

# --- CORRECTED/DYNAMIC Function ---
def Layered_visualize_activity_layout_grid(network, pos, activity_record, dt=0.1, stim_record=None,
                                   grid_resolution=(100, 150), save_path="3_layer_layout_grid_activity.gif",
                                   max_frames=1000, fps=30):
    """
    Creates a GIF animation showing neural activity spreading across the network layout.
    Neurons are mapped to a sparse grid based on their positions.
    Colors indicate neuron type (inhibitory=blue, excitatory=red) and stimulation state (yellow/green).
    MODIFIED: Dynamically handles both vectorized (has network.is_inhibitory) and
              non-vectorized (has network.neurons list) network objects.

    Args:
        network: The network object (vectorized or non-vectorized).
        pos (dict): Dictionary mapping neuron indices to (x, y) positions.
        activity_record (list): List of spiking neuron indices per time step.
        dt (float): Simulation time step (ms).
        stim_record (dict, optional): Stimulation record containing 'pulse_starts', 'neurons', 'pulse_duration_ms'.
        grid_resolution (tuple): Dimensions (rows, cols) of the sparse grid for visualization.
        save_path (str): Path to save the output GIF file.
        max_frames (int): Maximum number of frames to include in the GIF (downsamples if needed).
        fps (int): Frames per second for the output GIF.
    """
    print(f"Generating sparse grid activity animation (up to {max_frames} frames)...")
    # Safely get total neuron count
    n_neurons = getattr(network, 'n_neurons', 0)
    total_steps = len(activity_record)
    if total_steps == 0:
        print("Warning: No activity recorded.")
        return None
    if n_neurons == 0:
        print("Warning: Network has 0 neurons.")
        return None

    # Validate position data
    if not pos or not isinstance(pos, dict) or not any(isinstance(p, (tuple, list)) and len(p) == 2 for p in pos.values()):
        print("Warning: Invalid or empty position data (pos), cannot generate layout grid animation.")
        return None

    grid_rows, grid_cols = grid_resolution
    print(f"Mapping {n_neurons} neurons onto a {grid_rows}x{grid_cols} sparse grid...")

    # --- Map neuron positions to sparse grid coordinates ---
    # Ensure we only try to map nodes that exist in the network graph if available
    nodes_to_map = list(getattr(network, 'graph', {}).nodes()) or list(range(n_neurons))
    nodes_with_pos = [n for n in nodes_to_map if n in pos and isinstance(pos[n], (tuple, list)) and len(pos[n]) == 2]

    if not nodes_with_pos:
        print("Warning: No nodes found with valid position data in the provided 'pos' dictionary.")
        return None

    # Get position ranges for normalization using only valid positions
    xs = [pos[n][0] for n in nodes_with_pos]
    ys = [pos[n][1] for n in nodes_with_pos]
    if not xs or not ys:
        print("Warning: Position data extraction failed.")
        return None

    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    x_range = (max_x - min_x); y_range = (max_y - min_y)
    # Add small margin to prevent mapping to exact edge
    x_margin = x_range * 0.05 if x_range > 1e-6 else 1.0
    y_margin = y_range * 0.05 if y_range > 1e-6 else 1.0
    min_x -= x_margin; max_x += x_margin
    min_y -= y_margin; max_y += y_margin
    x_range = max_x - min_x; y_range = max_y - min_y

    # Create mappings: neuron ID -> grid (row, col), and grid (row, col) -> list of neuron IDs
    neuron_to_sparse_grid_pos = {}
    grid_to_neuron_map = {}
    for i in nodes_with_pos: # Iterate only over nodes confirmed to have positions
         x, y = pos[i]
         # Normalize and scale position to grid coordinates
         col = int(((x - min_x) / x_range) * (grid_cols - 1)) if x_range > 1e-9 else grid_cols // 2
         row = int(((max_y - y) / y_range) * (grid_rows - 1)) if y_range > 1e-9 else grid_rows // 2 # Invert y for image origin
         col = max(0, min(grid_cols - 1, col))
         row = max(0, min(grid_rows - 1, row))
         neuron_to_sparse_grid_pos[i] = (row, col)
         grid_coord = (row, col)
         if grid_coord not in grid_to_neuron_map: grid_to_neuron_map[grid_coord] = []
         grid_to_neuron_map[grid_coord].append(i)

    # --- Dynamically determine inhibitory status and create mask ---
    inhibitory_mask = np.zeros((grid_rows, grid_cols), dtype=bool)
    is_vectorized = hasattr(network, 'is_inhibitory') and isinstance(network.is_inhibitory, np.ndarray)
    is_non_vectorized = hasattr(network, 'neurons') and isinstance(network.neurons, list)

    for i in range(n_neurons): # Iterate through all possible neuron indices
         is_inhib = False # Default
         try:
             if is_vectorized:
                 if i < len(network.is_inhibitory): is_inhib = network.is_inhibitory[i]
             elif is_non_vectorized:
                 if i < len(network.neurons) and network.neurons[i] is not None:
                      neuron_obj = network.neurons[i]
                      if hasattr(neuron_obj, 'is_inhibitory'): is_inhib = neuron_obj.is_inhibitory
             # else: Fallback handled by is_inhib = False

             if is_inhib and i in neuron_to_sparse_grid_pos:
                 row, col = neuron_to_sparse_grid_pos[i]
                 inhibitory_mask[row, col] = True
         except (IndexError, AttributeError) as e:
              # Silently handle potential errors during status check for robusteness
              # print(f"Warning: Error checking inhibitory status for neuron {i}: {e}")
              pass # Keep is_inhib as False

    # --- Frame Sampling ---
    if total_steps > max_frames:
        indices = np.linspace(0, total_steps - 1, max_frames, dtype=int)
        sampled_activity = [activity_record[i] for i in indices]
        sampled_times = [i * dt for i in indices]
    else:
        sampled_activity = activity_record
        sampled_times = [i * dt for i in range(total_steps)]
        indices = np.arange(total_steps)

    # --- Process Stimulation Record ---
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
                      if step_index not in stim_active_at_step: stim_active_at_step[step_index] = set()
                      stim_active_at_step[step_index].update(neurons_in_pulse)
                      any_stim_active_flag[step_index] = True

    # --- Animation Setup ---
    aspect_ratio = grid_cols / grid_rows if grid_rows > 0 else 1
    fig_height = 8; fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(max(8, fig_width), fig_height), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a'); ax.set_xticks([]); ax.set_yticks([])

    activity_colors = np.zeros((grid_rows, grid_cols, 3)) # RGB color array
    img = ax.imshow(activity_colors, interpolation='nearest', origin='upper', vmin=0, vmax=1, aspect='auto')
    title = ax.set_title(f"Time: 0.0 ms", color='white', fontsize=14)
    stim_text = ax.text(0.01, 0.98, "", transform=ax.transAxes, color='lime', fontsize=10, verticalalignment='top', fontweight='bold')
    prev_activity_grid = np.zeros((grid_rows, grid_cols)) # Visual intensity state

    # Progress bar
    pbar = tqdm(total=len(sampled_activity), desc="Generating GIF Frames", leave=False)

    # --- Animation Update Function ---
    def update_sparse_grid(frame_idx):
        """Updates the grid visualization for a single animation frame."""
        nonlocal prev_activity_grid
        pbar.update(1)

        original_step_index = indices[frame_idx]
        # Ensure activity record items are treated as sets for efficient lookup
        active_neuron_indices_this_frame = set(sampled_activity[frame_idx])
        current_time = sampled_times[frame_idx]

        current_visual_intensity = prev_activity_grid * 0.75 # Decay factor

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
                     if is_stimulated: colors[r, c, :] = [0.5, 1.0, 0.5] # Lime green
                     elif inhibitory_mask[r, c]: colors[r, c, 2] = 1.0 # Bright Blue
                     else: colors[r, c, 0] = 1.0 # Bright Red
                     current_visual_intensity[r, c] = 1.0 # Reset intensity
                 elif is_stimulated:
                     decayed_stim_intensity = max(0.1, current_visual_intensity[r, c])
                     colors[r, c, :] = [0.6 * decayed_stim_intensity, 0.6 * decayed_stim_intensity, 0.0] # Dim yellow
                 else:
                      # Apply faded color based on decayed intensity (e.g., fade white)
                      intensity = current_visual_intensity[r, c]
                      colors[r, c, :] = [intensity * 0.7, intensity * 0.7, intensity * 0.7] # Fade grey/white

        img.set_array(np.clip(colors, 0, 1))
        title.set_text(f"Time: {current_time:.1f} ms")
        stim_text.set_text("STIMULATION" if any_stim_active_flag[original_step_index] else "")
        prev_activity_grid = current_visual_intensity
        return [img, title, stim_text]

    # --- Create and Save Animation ---
    anim = animation.FuncAnimation(fig, update_sparse_grid, frames=len(sampled_activity),
                                 interval=max(20, 1000//fps), blit=True)
    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=150)
        print(f"Successfully saved layout grid animation to {save_path}")
    except Exception as e:
        print(f"Error saving animation: {e}. PillowWriter might require Pillow (`pip install Pillow`)")
    finally:
        pbar.close()
        plt.close(fig)
    return anim

# --- END CORRECTED/DYNAMIC Function ---

