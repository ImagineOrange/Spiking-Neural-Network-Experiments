"""
Pre-defined network architectures for different experiment scenarios.

Use these to quickly configure experiments for common use cases like
quick testing, different class counts, or deep networks.
"""

from typing import Dict, Any


# ========== Pre-defined Architectures ==========

# Dense connection probabilities - nearly all feedforward connections made
DENSE_CONNECTIONS = {
    'exc_recurrent': 0.0,      # No excitatory recurrent
    'inh_recurrent': 0.30,     # Keep inhibitory recurrent for dynamics
    'feedforward_1': 1.0,      # Dense to next layer
    'feedforward_2': 1.0,      # Dense skip connections
    'feedback_1': 0.06,        # Sparse feedback for dynamics
    'feedback_2': 0.0,         # No long feedback
    'long_feedforward': 0.0,   # No long feedforward (handled by skip)
    'long_feedback': 0.0       # No long feedback
}

ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": "Original pre-refactor configuration (5-digit classification)",
        "name": "default_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "tiny_2class": {
        "description": "Minimal network for quick testing (2 digits)",
        "name": "tiny_2class",
        "target_classes": [0, 1],
        "hidden_layers": [15, 10],
        "population_size": 30,
        "num_generations": 50,
        "fitness_eval_examples": 500,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "small_2class": {
        "description": "Small network for 2-digit classification",
        "name": "small_2class",
        "target_classes": [0, 1],
        "hidden_layers": [20, 15],
        "population_size": 50,
        "num_generations": 100,
        "fitness_eval_examples": 800,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "medium_3class": {
        "description": "Medium network for 3-digit classification",
        "name": "medium_3class",
        "target_classes": [0, 1, 2],
        "hidden_layers": [30, 25],
        "population_size": 10,
        "num_generations": 120,
        "fitness_eval_examples": 100,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "standard_5class": {
        "description": "Standard configuration for 5-digit classification",
        "name": "standard_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "large_10class": {
        "description": "Large network for full 10-digit classification",
        "name": "large_10class",
        "target_classes": list(range(10)),
        "hidden_layers": [80, 60, 40],
        "population_size": 150,
        "num_generations": 200,
        "fitness_eval_examples": 1500,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "deep_5class": {
        "description": "Deeper network for 5-digit classification",
        "name": "deep_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [50, 40, 30, 20],
        "population_size": 120,
        "num_generations": 180,
        "fitness_eval_examples": 1000,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "wide_5class": {
        "description": "Wider shallow network for 5-digit classification",
        "name": "wide_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [80, 60],
        "population_size": 120,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "conv_features_5class": {
        "description": "5-class using CNN feature encoding",
        "name": "conv_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "encoding_mode": "conv_feature_to_neuron",
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
        "connection_probs": DENSE_CONNECTIONS,
    },

    "debug": {
        "description": "Ultra-fast config for debugging code",
        "name": "debug_2class",
        "target_classes": [0, 1],
        "hidden_layers": [20, 10],
        "population_size": 10,
        "num_generations": 5,
        "fitness_eval_examples": 100,
        "sim_duration_ms": 30.0,  # Shorter simulation
        "mnist_stim_duration_ms": 20.0,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",  # New parameter: small random weights
        "init_weight_std": 0.1,  # Std dev for random initialization
    },

    "random_init_3class": {
        "description": "3-class with small random weight initialization",
        "name": "random_init_3class",
        "target_classes": [0, 1, 2],
        "hidden_layers": [30, 25],
        "population_size": 10,
        "num_generations": 120,
        "fitness_eval_examples": 100,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",  # New parameter: small random weights
        "init_weight_std": 0.02,  # Std dev for random initialization
    },

    # ========== Diehl-style STDP Architectures (Refactored) ==========

    "stdp_diehl_2class": {
        "description": "Compact 2-class STDP: 49→20→10 (5 output neurons per class)",
        "name": "stdp_diehl_2class",
        "target_classes": [0, 1],
        "hidden_layers": [20],  # Single small hidden layer
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,  # 7x7 input = 49 neurons
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",
        "init_weight_std": 0.078,  # Diehl & Cook 2015
        # STDP-specific (not used by GA, but passed through)
        "neurons_per_class": 5,  # 5 neurons per class = 10 output total
    },

    "stdp_diehl_2class_large": {
        "description": "Larger Diehl-style network for 2-class STDP (400 hidden neurons)",
        "name": "stdp_diehl_2class_large",
        "target_classes": [0, 1],
        "hidden_layers": [400],  # Large single hidden layer
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",
        "init_weight_std": 0.15,
        "neurons_per_class": 5,
    },

    "stdp_diehl_3class": {
        "description": "Diehl-style network for 3-class STDP",
        "name": "stdp_diehl_3class",
        "target_classes": [0, 1, 2],
        "hidden_layers": [150],  # Slightly larger for 3 classes
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",
        "init_weight_std": 0.15,
        "neurons_per_class": 5,  # 15 output neurons total
    },

    "stdp_diehl_5class": {
        "description": "Diehl-style network for 5-class STDP",
        "name": "stdp_diehl_5class",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [250],  # Larger for 5 classes
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",
        "init_weight_std": 0.15,
        "neurons_per_class": 10,  # 50 output neurons total
    },

    "stdp_test": {
        "description": "Quick test config for refactored STDP system",
        "name": "stdp_test",
        "target_classes": [0, 1],
        "hidden_layers": [50],  # Smaller for quick testing
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
        "connection_probs": DENSE_CONNECTIONS,
        "init_weight_mode": "random_small",
        "init_weight_std": 0.15,
        "neurons_per_class": 5,
        "sim_duration_ms": 100.0,  # Total simulation window
        "mnist_stim_duration_ms": 50.0,  # Input period
    },
}


def get_architecture(name: str) -> Dict[str, Any]:
    """
    Get pre-defined architecture configuration by name.

    Args:
        name: Architecture name (e.g., 'standard_5class', 'tiny_2class')

    Returns:
        Dictionary of configuration parameters to override

    Raises:
        ValueError: If architecture name is not found

    Example:
        >>> from config.experiment_config import get_config
        >>> from config.network_architectures import get_architecture
        >>>
        >>> arch = get_architecture('small_2class')
        >>> cfg = get_config(**arch)
    """
    if name not in ARCHITECTURES:
        available = ', '.join(ARCHITECTURES.keys())
        raise ValueError(
            f"Unknown architecture: '{name}'\n"
            f"Available architectures: {available}"
        )

    # Copy and remove 'description' field - it's for documentation only
    arch_config = ARCHITECTURES[name].copy()
    arch_config.pop('description', None)
    return arch_config


def list_architectures() -> None:
    """Print all available architectures with descriptions."""
    print("\nAvailable Network Architectures:")
    print("=" * 70)

    for name, config in ARCHITECTURES.items():
        desc = config.get('description', 'No description')
        n_classes = len(config.get('target_classes', []))
        layers = config.get('hidden_layers', [])

        print(f"\n{name}:")
        print(f"  Description: {desc}")
        print(f"  Classes: {n_classes}")
        print(f"  Hidden layers: {layers}")
        print(f"  Population: {config.get('population_size', '?')}")
        print(f"  Generations: {config.get('num_generations', '?')}")


if __name__ == "__main__":
    # Demo: List all available architectures
    list_architectures()

    print("\n" + "=" * 70)
    print("\nExample usage:")
    print(">>> from config.network_architectures import get_architecture")
    print(">>> arch = get_architecture('standard_5class')")
    print(">>> cfg = get_config(**arch)")
