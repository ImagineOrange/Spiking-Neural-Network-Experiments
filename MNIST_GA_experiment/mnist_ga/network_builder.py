"""
SNN structure creation utilities.

This module handles creating the fixed network topology (connections, delays, positions)
that will be used throughout GA evolution. Weights start at 0 and are evolved by the GA.
"""

import numpy as np
import random
from typing import Tuple, Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LIF_objects.LayeredNeuronalNetworkVectorized import LayeredNeuronalNetworkVectorized


class NetworkStructure:
    """Container for network structural information."""

    def __init__(self, network, layer_indices, positions, connection_map):
        self.network = network
        self.layer_indices = layer_indices
        self.positions = positions
        self.connection_map = connection_map

        # Computed properties
        self.n_neurons = network.n_neurons
        self.n_connections = len(connection_map)
        self.output_layer_start, self.output_layer_end = layer_indices[-1]


def create_snn_structure(cfg) -> NetworkStructure:
    """
    Creates the fixed SNN structure (topology, delays) from configuration.

    This function is called ONCE at the start of an experiment to define
    the network structure. Weights are initialized to 0 and will be evolved
    by the genetic algorithm.

    Args:
        cfg: ExperimentConfig instance with all parameters

    Returns:
        NetworkStructure containing network object and structural info

    Note:
        Includes hardcoded mutual inhibition between output layer neurons
        for winner-take-all dynamics.
    """
    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)

    # Get layer configuration
    n_layers_list = cfg.layer_config
    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    print(f"\n=== Creating Fixed SNN Structure ===")
    print(f"Architecture: {' → '.join(map(str, n_layers_list))}")
    print(f"Total neurons: {total_neurons}")

    # --- Determine Inhibitory Status (FIXED throughout evolution) ---
    is_inhibitory_array = np.random.rand(total_neurons) < cfg.inhibitory_fraction

    # --- Create Network Instance ---
    network_constructor_params = cfg.neuron_params.copy()
    network_constructor_params['is_inhibitory'] = is_inhibitory_array

    network = LayeredNeuronalNetworkVectorized(
        n_neurons=total_neurons,
        **network_constructor_params
    )

    # --- Create Spatial Positions and Layer Assignments ---
    positions, layer_indices = _create_positions_and_layers(
        network, n_layers_list, num_layers, total_neurons
    )

    # --- Calculate Max Distance for Delay Scaling ---
    max_possible_dist = _calculate_max_distance(positions)

    # --- Add Probabilistic Connections ---
    connection_map = _add_probabilistic_connections(
        network, positions, layer_indices, total_neurons,
        cfg.connection_probs, cfg.base_transmission_delay,
        max_possible_dist, cfg.sim_dt_ms
    )

    # --- Add Hardcoded Output Mutual Inhibition ---
    _add_output_mutual_inhibition(
        network, layer_indices, positions, connection_map,
        cfg.output_mutual_inhibition_weight,
        cfg.output_mutual_inhibition_delay
    )

    print(f"Structure created: {len(connection_map)} connections")
    print("=" * 50)

    return NetworkStructure(network, layer_indices, positions, connection_map)


def _create_positions_and_layers(network, n_layers_list, num_layers, total_neurons):
    """Create neuron positions and assign layers to graph nodes."""
    positions = {}
    layer_indices = []

    # Spread layers horizontally
    x_coords = np.linspace(0.1, 0.9, num_layers)
    horizontal_spread = 0.04
    vertical_spread = max(0.5, total_neurons / 200.0)

    start_idx = 0
    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num - 1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))

        for neuron_idx in range(start_idx, end_idx):
            # Get inhibitory status from network's fixed array
            is_inhib = network.is_inhibitory[neuron_idx]

            # Assign random position within layer's spatial region
            node_pos = (
                x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread)
            )
            positions[neuron_idx] = node_pos

            # Add node to network graph
            network.graph.add_node(
                neuron_idx,
                is_inhibitory=is_inhib,
                layer=layer_num,
                pos=node_pos
            )
            network.neuron_grid_positions[neuron_idx] = node_pos

        start_idx = end_idx

    return positions, layer_indices


def _calculate_max_distance(positions):
    """Calculate maximum possible distance between neurons for delay normalization."""
    if not positions or len(positions) < 2:
        return 1.0

    all_pos_vals = list(positions.values())
    all_x = [p[0] for p in all_pos_vals]
    all_y = [p[1] for p in all_pos_vals]

    # Check if all points are identical
    if max(all_x) == min(all_x) and max(all_y) == min(all_y):
        return 1.0

    dist_sq = (max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2
    return np.sqrt(dist_sq) if dist_sq > 1e-9 else 1.0


def _add_probabilistic_connections(network, positions, layer_indices, total_neurons,
                                   connection_probs, base_delay, max_dist, dt):
    """Add connections between neurons based on layer-dependent probabilities."""
    connection_map = []
    min_delay = max(0.1, dt)

    # Extract connection probabilities
    conn_probs = {
        'exc_recurrent': connection_probs.get('exc_recurrent', 0.08),
        'inh_recurrent': connection_probs.get('inh_recurrent', 0.15),
        'feedforward_1': connection_probs.get('feedforward_1', 0.45),
        'feedforward_2': connection_probs.get('feedforward_2', 0.15),
        'feedback_1': connection_probs.get('feedback_1', 0.06),
        'feedback_2': connection_probs.get('feedback_2', 0.0),
        'long_feedforward': connection_probs.get('long_feedforward', 0.01),
        'long_feedback': connection_probs.get('long_feedback', 0.005),
    }

    connection_count = 0

    # Iterate through all possible neuron pairs
    for i in range(total_neurons):
        if i not in network.graph.nodes or i not in positions:
            continue

        is_source_inhibitory = network.is_inhibitory[i]
        layer_i = network.graph.nodes[i].get('layer', -1)
        pos_i = positions[i]

        for j in range(total_neurons):
            if i == j or j not in network.graph.nodes or j not in positions:
                continue

            layer_j = network.graph.nodes[j].get('layer', -1)
            pos_j = positions[j]

            # Determine connection probability based on layer difference
            prob = _get_connection_probability(
                is_source_inhibitory, layer_i, layer_j, conn_probs
            )

            # Randomly connect based on probability
            if random.random() < prob:
                # Calculate distance-dependent delay
                distance = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                delay = base_delay * (0.5 + 0.5 * (distance / max_dist if max_dist > 1e-9 else 0))
                delay = max(min_delay, np.round(delay / dt) * dt if dt > 0 else delay)

                # Add connection with weight=0 (to be set by GA)
                network.add_connection(i, j, weight=0.0, delay=delay)
                connection_map.append((i, j))
                connection_count += 1

    print(f"  Probabilistic connections: {connection_count}")
    return connection_map


def _get_connection_probability(is_source_inhibitory, layer_i, layer_j, conn_probs):
    """Determine connection probability based on neuron type and layer difference."""
    if layer_i == -1 or layer_j == -1:
        return 0.0

    layer_diff = layer_j - layer_i

    if is_source_inhibitory:
        # Inhibitory connections
        if layer_diff == 0:
            return conn_probs['inh_recurrent']
        elif abs(layer_diff) == 1:
            # Inhibitory feedforward/feedback with reduced probability
            return conn_probs['feedforward_1'] * 0.2
        else:
            return 0.0
    else:
        # Excitatory connections
        if layer_diff == 0:
            return conn_probs['exc_recurrent']
        elif layer_diff == 1:
            return conn_probs['feedforward_1']
        elif layer_diff == 2:
            return conn_probs['feedforward_2']
        elif layer_diff == -1:
            return conn_probs['feedback_1']
        elif layer_diff == -2:
            return conn_probs['feedback_2']
        elif layer_diff > 2:
            return conn_probs['long_feedforward']
        elif layer_diff < -2:
            return conn_probs['long_feedback']

    return 0.0


def _add_output_mutual_inhibition(network, layer_indices, positions, connection_map,
                                  inhibitory_weight, delay):
    """
    Add hardcoded mutual inhibition between all output layer neurons.

    This creates winner-take-all dynamics where output neurons compete.
    These connections are NOT evolved by the GA - they remain fixed.
    """
    output_start, output_end = layer_indices[-1]
    output_size = output_end - output_start

    if output_size < 2:
        print("  Output layer has < 2 neurons, skipping mutual inhibition")
        return

    manual_connections = 0
    print(f"  Adding mutual inhibition between output neurons {output_start}-{output_end-1}")

    # Add inhibitory connection from each output neuron to all others
    for i in range(output_start, output_end):
        if i not in positions:
            continue

        for j in range(output_start, output_end):
            if i == j or j not in positions:
                continue

            # Check if connection already exists
            connection_exists = (i, j) in connection_map

            # Add/update connection with fixed inhibitory weight
            network.add_connection(i, j, weight=inhibitory_weight, delay=delay)
            manual_connections += 1

            if not connection_exists:
                connection_map.append((i, j))

    print(f"  Mutual inhibition connections: {manual_connections}")


if __name__ == "__main__":
    # Demo: Create a small network structure
    import sys
    sys.path.append('..')
    from config.experiment_config import get_config

    cfg = get_config()
    cfg.target_classes = [0, 1]  # 2-class for demo
    cfg.hidden_layers = [10, 5]

    structure = create_snn_structure(cfg)
    print(f"\nCreated network with {structure.n_neurons} neurons and {structure.n_connections} connections")
