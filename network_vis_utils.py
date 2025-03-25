import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

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