"""
Pre-defined network architectures for different experiment scenarios.

Use these to quickly configure experiments for common use cases like
quick testing, different class counts, or deep networks.
"""

from typing import Dict, Any


# ========== Pre-defined Architectures ==========

ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": "Original pre-refactor configuration (5-digit classification)",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
        "encoding_mode": "intensity_to_neuron",
        "downsample_factor": 4,
    },

    "tiny_2class": {
        "description": "Minimal network for quick testing (2 digits)",
        "target_classes": [0, 1],
        "hidden_layers": [15, 10],
        "population_size": 30,
        "num_generations": 50,
        "fitness_eval_examples": 500,
    },

    "small_2class": {
        "description": "Small network for 2-digit classification",
        "target_classes": [0, 1],
        "hidden_layers": [20, 15],
        "population_size": 50,
        "num_generations": 100,
        "fitness_eval_examples": 800,
    },

    "medium_3class": {
        "description": "Medium network for 3-digit classification",
        "target_classes": [0, 1, 2],
        "hidden_layers": [30, 25],
        "population_size": 75,
        "num_generations": 120,
        "fitness_eval_examples": 900,
    },

    "standard_5class": {
        "description": "Standard configuration for 5-digit classification",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
    },

    "large_10class": {
        "description": "Large network for full 10-digit classification",
        "target_classes": list(range(10)),
        "hidden_layers": [80, 60, 40],
        "population_size": 150,
        "num_generations": 200,
        "fitness_eval_examples": 1500,
    },

    "deep_5class": {
        "description": "Deeper network for 5-digit classification",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [50, 40, 30, 20],
        "population_size": 120,
        "num_generations": 180,
        "fitness_eval_examples": 1000,
    },

    "wide_5class": {
        "description": "Wider shallow network for 5-digit classification",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [80, 60],
        "population_size": 120,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
    },

    "conv_features_5class": {
        "description": "5-class using CNN feature encoding",
        "target_classes": [0, 1, 2, 3, 4],
        "hidden_layers": [40, 30],
        "encoding_mode": "conv_feature_to_neuron",
        "population_size": 100,
        "num_generations": 150,
        "fitness_eval_examples": 1000,
    },

    "debug": {
        "description": "Ultra-fast config for debugging code",
        "target_classes": [0, 1],
        "hidden_layers": [10, 5],
        "population_size": 10,
        "num_generations": 5,
        "fitness_eval_examples": 100,
        "sim_duration_ms": 30.0,  # Shorter simulation
        "mnist_stim_duration_ms": 20.0,
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

    return ARCHITECTURES[name].copy()


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
