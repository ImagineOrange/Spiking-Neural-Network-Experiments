"""
Central configuration for MNIST GA SNN experiments.

This module provides a clean, type-safe configuration system using dataclasses.
All experiment parameters are defined here for easy modification and tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import os


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration for MNIST GA SNN.

    This dataclass contains all parameters needed to run a complete
    GA evolution experiment for MNIST classification.
    """

    # ========== Experiment Metadata ==========
    name: str = "5class_mnist_ga"
    """Experiment name - used for output directory naming"""

    random_seed: int = 42
    """Random seed for reproducibility"""

    output_dir: str = "outputs"
    """Base directory for all outputs"""

    # ========== Target Classes ==========
    target_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    """MNIST digit classes to train on (e.g., [0, 1, 2, 3, 4] for 5-class)"""

    # ========== Input Encoding ==========
    encoding_mode: str = 'intensity_to_neuron'
    """
    Encoding mode for converting MNIST images to spikes.
    Options:
        - 'intensity_to_neuron': Direct pixel intensity -> firing rate
        - 'conv_feature_to_neuron': CNN features -> firing rate
    """

    downsample_factor: int = 4
    """
    Downsampling factor for intensity_to_neuron mode.
    28x28 image becomes (28//downsample_factor)^2 pixels.
    Only used when encoding_mode='intensity_to_neuron'
    """

    conv_weights_path: str = '../MNIST_utils/conv_model_weights/conv_model_weights.pth'
    """Path to pre-trained CNN weights (only used for conv_feature_to_neuron mode)"""

    conv_feature_count: int = 49
    """Number of CNN features (only used for conv_feature_to_neuron mode)"""

    # ========== Network Architecture ==========
    hidden_layers: List[int] = field(default_factory=lambda: [40, 30])
    """
    Hidden layer sizes (not including input/output layers).
    Example: [40, 30] creates a 49->40->30->5 network for 5-class
    """

    inhibitory_fraction: float = 0.2
    """Fraction of neurons that are inhibitory (0.0 to 1.0)"""

    # ========== Connection Probabilities ==========
    connection_probs: Dict[str, float] = field(default_factory=lambda: {
        'exc_recurrent': 0.0,      # Excitatory recurrent within layer
        'inh_recurrent': 0.30,     # Inhibitory recurrent within layer
        'feedforward_1': 0.3,      # Excitatory to next layer (+1)
        'feedforward_2': 0.15,     # Excitatory skip connection (+2)
        'feedback_1': 0.06,        # Excitatory feedback to previous layer (-1)
        'feedback_2': 0.0,         # Excitatory feedback skip connection (-2)
        'long_feedforward': 0.0,   # Long-range feedforward (>+2)
        'long_feedback': 0.0       # Long-range feedback (<-2)
    })
    """Connection probabilities between layers based on layer difference"""

    base_transmission_delay: float = 1.0
    """Base synaptic delay in ms (actual delays are distance-dependent)"""

    # ========== Neuron Parameters ==========
    neuron_params: Dict[str, float] = field(default_factory=lambda: {
        # Basic LIF parameters
        'v_rest': -65.0,           # Resting potential (mV)
        'v_threshold': -55.0,      # Spike threshold (mV)
        'v_reset': -75.0,          # Reset potential after spike (mV)

        # Time constants
        'tau_m': 10.0,             # Membrane time constant (ms)
        'tau_ref': 1.5,            # Refractory period (ms)
        'tau_e': 3.0,              # Excitatory conductance decay (ms)
        'tau_i': 7.0,              # Inhibitory conductance decay (ms)

        # Reversal potentials
        'e_reversal': 0.0,         # Excitatory reversal potential (mV)
        'i_reversal': -70.0,       # Inhibitory reversal potential (mV)

        # Noise
        'v_noise_amp': 0.0,        # Membrane potential noise amplitude (mV)
        'i_noise_amp': 0.0005,     # Synaptic current noise amplitude

        # Adaptation
        'adaptation_increment': 0.3,  # Adaptation increase per spike
        'tau_adaptation': 120,        # Adaptation decay time constant (ms)
    })
    """LIF neuron parameters"""

    # ========== Simulation Parameters ==========
    sim_duration_ms: float = 70.0
    """Total simulation duration per MNIST example (ms)"""

    sim_dt_ms: float = 0.1
    """Simulation timestep (ms)"""

    mnist_stim_duration_ms: float = 50.0
    """Duration of MNIST spike presentation (ms)"""

    max_freq_hz: float = 200.0
    """Maximum firing frequency for spike encoding (Hz)"""

    stim_strength: float = 25.0
    """Conductance strength for external stimulation"""

    # ========== Genetic Algorithm Parameters ==========
    population_size: int = 100
    """Number of individuals in GA population"""

    num_generations: int = 150
    """Number of generations to evolve"""

    mutation_rate: float = 0.05
    """Probability of mutating each weight (0.0 to 1.0)"""

    mutation_strength: float = 0.01
    """Standard deviation of mutation (fraction of weight range)"""

    crossover_rate: float = 0.7
    """Probability of crossover between parents (0.0 to 1.0)"""

    elitism_count: int = 2
    """Number of top individuals to preserve unchanged"""

    tournament_size: int = 3
    """Number of individuals in tournament selection"""

    fitness_eval_examples: int = 1000
    """Number of MNIST examples to evaluate per generation"""

    # ========== Weight Bounds ==========
    weight_min: float = 0.002
    """Minimum absolute synaptic weight"""

    weight_max: float = 0.35
    """Maximum absolute synaptic weight"""

    # ========== Computational ==========
    n_cores: int = field(default_factory=lambda: max(1, os.cpu_count() - 1 if os.cpu_count() else 1))
    """Number of CPU cores to use for parallel evaluation"""

    # ========== Output Mutual Inhibition ==========
    output_mutual_inhibition_weight: float = -0.05
    """Fixed inhibitory weight between output neurons (negative)"""

    output_mutual_inhibition_delay: float = 0.1
    """Fixed delay for output mutual inhibition (ms)"""

    # ========== Computed Properties ==========
    @property
    def n_classes(self) -> int:
        """Number of target classes"""
        return len(self.target_classes)

    @property
    def input_layer_size(self) -> int:
        """
        Size of SNN input layer based on encoding mode.

        Returns:
            Number of input neurons
        """
        if self.encoding_mode == 'intensity_to_neuron':
            return (28 // self.downsample_factor) ** 2
        elif self.encoding_mode == 'conv_feature_to_neuron':
            return self.conv_feature_count
        else:
            raise ValueError(f"Unknown encoding mode: {self.encoding_mode}")

    @property
    def layer_config(self) -> List[int]:
        """
        Complete layer configuration including input and output.

        Returns:
            List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        return [self.input_layer_size] + self.hidden_layers + [self.n_classes]

    @property
    def experiment_output_dir(self) -> str:
        """Full path to this experiment's output directory"""
        return os.path.join(self.output_dir, self.name)

    @property
    def label_map(self) -> Dict[int, int]:
        """
        Mapping from original MNIST labels to 0-indexed class labels.

        Returns:
            Dict mapping original label -> class index
            Example: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4} for [0,1,2,3,4]
        """
        return {original_label: idx for idx, original_label in enumerate(self.target_classes)}


def get_config(name: str = None, **kwargs) -> ExperimentConfig:
    """
    Get experiment configuration with optional overrides.

    Args:
        name: Optional experiment name
        **kwargs: Any config parameter to override

    Returns:
        ExperimentConfig instance

    Example:
        >>> cfg = get_config(name="2class_test", target_classes=[0, 1], num_generations=50)
    """
    cfg = ExperimentConfig()

    if name:
        cfg.name = name

    # Apply any keyword argument overrides
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return cfg


if __name__ == "__main__":
    # Demo: Print default configuration
    cfg = get_config()
    print(f"Experiment: {cfg.name}")
    print(f"Target classes: {cfg.target_classes}")
    print(f"Network architecture: {cfg.layer_config}")
    print(f"Input encoding: {cfg.encoding_mode}")
    print(f"GA: {cfg.population_size} pop × {cfg.num_generations} gen")
    print(f"Output directory: {cfg.experiment_output_dir}")
