"""
Configuration for STDP-based online learning experiments.

Extends the base ExperimentConfig with STDP-specific parameters.
"""

from dataclasses import dataclass, field
from typing import List
from .experiment_config import ExperimentConfig


@dataclass
class STDPConfig(ExperimentConfig):
    """
    Configuration for STDP online learning experiments.

    Inherits all parameters from ExperimentConfig and adds STDP-specific settings.
    """

    # ========== STDP Learning Parameters ==========
    stdp_a_plus: float = 0.001
    """LTP (potentiation) learning rate"""

    stdp_a_minus: float = 0.0012
    """LTD (depression) learning rate (slightly larger for depression bias)"""

    stdp_tau_pre: float = 20.0
    """Pre-synaptic trace time constant (ms)"""

    stdp_tau_post: float = 20.0
    """Post-synaptic trace time constant (ms)"""

    # ========== Phase Timing Parameters ==========
    stdp_input_period: float = 50.0
    """Duration of input/stimulus period (ms) - matches mnist_stim_duration_ms"""

    stdp_readout_period: float = 30.0
    """Duration of readout period for prediction (ms)"""

    stdp_learning_period: float = 50.0
    """Duration of STDP learning period (ms)"""

    stdp_rest_period: float = 20.0
    """Duration of rest period before reset (ms)"""

    @property
    def stdp_total_window(self) -> float:
        """Total duration of one digit presentation window (ms)"""
        return (self.stdp_input_period + self.stdp_readout_period +
                self.stdp_learning_period + self.stdp_rest_period)

    # ========== Training Parameters ==========
    num_epochs: int = 5
    """Number of training epochs (passes through dataset)"""

    shuffle_each_epoch: bool = True
    """Whether to shuffle training examples each epoch"""

    validation_frequency: int = 500
    """Evaluate on validation set every N examples"""

    validation_samples: int = 200
    """Number of validation samples to use for progress monitoring"""

    # ========== Homeostatic Normalization ==========
    homeostatic_normalization: bool = True
    """Whether to apply homeostatic normalization after learning"""

    homeostatic_target_sum: float = None
    """Target sum for incoming weights per neuron. If None, maintains current sum."""

    normalize_every_example: bool = True
    """If True, normalize after each example. If False, normalize every N examples."""

    normalize_frequency: int = 1
    """Normalize every N examples (only if normalize_every_example=False)"""

    # ========== Checkpointing ==========
    save_checkpoints: bool = True
    """Whether to save network checkpoints during training"""

    checkpoint_every_epoch: bool = True
    """Save checkpoint at end of each epoch"""

    checkpoint_every_n_examples: int = None
    """Save checkpoint every N examples (None = disabled)"""

    # ========== Progress Monitoring ==========
    print_every_n_examples: int = 100
    """Print training progress every N examples"""

    track_weight_stats: bool = True
    """Whether to track weight statistics during training"""

    track_spike_stats: bool = True
    """Whether to track spike statistics during training"""

    # ========== Memory Management ==========
    reset_state_between_examples: bool = True
    """Whether to reset transient state between digit windows"""

    reset_traces: bool = True
    """Whether to hard reset traces or let them decay naturally"""

    # ========== Experiment Naming ==========
    experiment_type: str = "stdp"
    """Experiment type identifier (used in output naming)"""

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure input period matches stimulus duration
        if abs(self.stdp_input_period - self.mnist_stim_duration_ms) > 0.1:
            print(f"Warning: stdp_input_period ({self.stdp_input_period}ms) "
                  f"!= mnist_stim_duration_ms ({self.mnist_stim_duration_ms}ms)")

        # Update experiment name to reflect STDP
        if "stdp" not in self.name.lower():
            self.name = f"{self.name}_stdp"

    def get_stdp_summary(self) -> str:
        """Get a summary string of STDP configuration."""
        return (
            f"STDP Config:\n"
            f"  Learning: A+={self.stdp_a_plus}, A-={self.stdp_a_minus}\n"
            f"  Traces: τ+={self.stdp_tau_pre}ms, τ-={self.stdp_tau_post}ms\n"
            f"  Phases: Input={self.stdp_input_period}ms, "
            f"Readout={self.stdp_readout_period}ms, "
            f"Learning={self.stdp_learning_period}ms, "
            f"Rest={self.stdp_rest_period}ms\n"
            f"  Total window: {self.stdp_total_window}ms\n"
            f"  Training: {self.num_epochs} epochs, "
            f"shuffle={self.shuffle_each_epoch}\n"
            f"  Homeostatic: {self.homeostatic_normalization}, "
            f"every_example={self.normalize_every_example}"
        )


def get_stdp_config(name: str = None, **kwargs) -> STDPConfig:
    """
    Get STDP configuration with optional overrides.

    Args:
        name: Optional experiment name
        **kwargs: Any config parameter to override

    Returns:
        STDPConfig instance

    Example:
        >>> cfg = get_stdp_config(name="2class_stdp_test",
        ...                        target_classes=[0, 1],
        ...                        num_epochs=10)
    """
    cfg = STDPConfig()

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
    # Demo: Print default STDP configuration
    cfg = get_stdp_config()
    print(f"Experiment: {cfg.name}")
    print(f"Target classes: {cfg.target_classes}")
    print(f"Network architecture: {cfg.layer_config}")
    print(cfg.get_stdp_summary())
    print(f"\nOutput directory: {cfg.experiment_output_dir}")
