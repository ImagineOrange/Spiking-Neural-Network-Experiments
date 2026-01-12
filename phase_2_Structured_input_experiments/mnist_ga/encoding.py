"""
MNIST data loading and spike encoding utilities.

This module handles:
1. Loading and filtering MNIST data
2. Converting images to spike trains (precomputation for efficiency)
3. Managing train/test splits
"""

import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MNIST_utils.MNIST_stimulation_encodings import MNIST_loader, SNNStimulator, downsample_image


class MNISTDataset:
    """
    Wrapper for MNIST data with filtering and train/test splitting.

    This class handles loading MNIST, filtering to target classes,
    and creating train/test splits with proper label mapping.
    """

    def __init__(self, target_classes: list, train_split: float = 0.8):
        """
        Initialize MNIST dataset.

        Args:
            target_classes: List of MNIST digits to include (e.g., [0, 1, 2, 3, 4])
            train_split: Fraction of data to use for training (default: 0.8)
        """
        self.target_classes = target_classes
        self.train_split = train_split

        # Load and process data
        print(f"Loading MNIST dataset...")
        self.loader = MNIST_loader()
        self._filter_and_split()

        print(f"Filtered to {len(self.labels)} examples from classes {target_classes}")
        print(f"Train: {len(self.train_indices)} examples, Test: {len(self.test_indices)} examples")

    def _filter_and_split(self):
        """Filter to target classes and split into train/test sets."""
        # Filter to only include target classes
        mask = np.isin(self.loader.labels, self.target_classes)
        self.images = self.loader.images[mask]
        self.labels = self.loader.labels[mask]

        # Create train/test split
        n_total = len(self.labels)
        split_idx = int(n_total * self.train_split)

        if split_idx == 0 or split_idx == n_total:
            raise ValueError(
                f"Invalid train/test split. Total examples: {n_total}, "
                f"train_split: {self.train_split}"
            )

        self.train_indices = np.arange(split_idx)
        self.test_indices = np.arange(split_idx, n_total)

        # Create label mapping (original label -> 0-indexed class)
        self.label_map = {
            original: idx
            for idx, original in enumerate(self.target_classes)
        }

    def get_mapped_label(self, index: int) -> int:
        """
        Get the 0-indexed class label for an example.

        Args:
            index: Dataset index

        Returns:
            Class label (0 to n_classes-1)
        """
        original_label = self.labels[index]
        return self.label_map[original_label]

    def get_image(self, index: int) -> np.ndarray:
        """
        Get image at given index.

        Args:
            index: Dataset index

        Returns:
            28x28 image array (values 0-1)
        """
        return self.images[index].reshape(28, 28)

    @property
    def n_classes(self) -> int:
        """Number of classes in the dataset."""
        return len(self.target_classes)

    def __len__(self) -> int:
        """Total number of examples."""
        return len(self.labels)


def setup_mnist_data(cfg) -> MNISTDataset:
    """
    Load and prepare MNIST data from configuration.

    Args:
        cfg: ExperimentConfig instance

    Returns:
        MNISTDataset instance

    Example:
        >>> from config.experiment_config import get_config
        >>> cfg = get_config()
        >>> data = setup_mnist_data(cfg)
    """
    return MNISTDataset(
        target_classes=cfg.target_classes,
        train_split=0.8  # Hardcoded 80/20 split
    )


def precompute_spike_trains(data: MNISTDataset, cfg, device: Optional[torch.device] = None) -> Dict[int, np.ndarray]:
    """
    Precompute spike trains for all MNIST examples.

    This dramatically speeds up GA evolution by encoding each image
    to spikes only once, rather than re-encoding on every evaluation.

    Args:
        data: MNISTDataset instance
        cfg: ExperimentConfig with encoding parameters
        device: Optional torch device for CNN (if using conv features)

    Returns:
        Dictionary mapping dataset index -> spike times array

    Example:
        >>> spike_trains = precompute_spike_trains(data, cfg)
        >>> spikes_for_img_0 = spike_trains[0]  # List of spike times per neuron
    """
    print(f"\n=== Precomputing Spike Trains ===")
    print(f"Encoding mode: {cfg.encoding_mode}")
    print(f"Stimulus duration: {cfg.mnist_stim_duration_ms}ms")
    print(f"Max frequency: {cfg.max_freq_hz}Hz")

    # Determine device if not provided
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create stimulator
    stimulator = _create_stimulator(cfg, device)

    # Precompute spikes for all examples
    spike_trains = {}
    n_failed = 0

    for idx in tqdm(range(len(data)), desc="Encoding images", ncols=80):
        try:
            image = data.get_image(idx)
            prepared_image = _prepare_image_for_encoding(image, cfg)
            spike_trains[idx] = stimulator.generate_spikes(prepared_image)
        except Exception as e:
            print(f"\nWarning: Failed to encode image {idx}: {e}")
            spike_trains[idx] = None
            n_failed += 1

    if n_failed > 0:
        print(f"Warning: Failed to encode {n_failed}/{len(data)} images")

    print(f"Successfully precomputed {len(spike_trains) - n_failed} spike trains")
    print("=" * 50)

    return spike_trains


def _create_stimulator(cfg, device) -> SNNStimulator:
    """Create SNNStimulator with appropriate configuration."""
    # Check if CNN weights are needed and exist
    conv_weights_path = None
    if cfg.encoding_mode == 'conv_feature_to_neuron':
        conv_weights_path = cfg.conv_weights_path
        if not os.path.exists(conv_weights_path):
            raise FileNotFoundError(
                f"Conv feature encoding requires CNN weights at: {conv_weights_path}\n"
                "Please train the CNN first or switch to 'intensity_to_neuron' mode."
            )
        print(f"Using CNN weights from: {conv_weights_path}")

    return SNNStimulator(
        total_time_ms=cfg.mnist_stim_duration_ms,
        max_freq_hz=cfg.max_freq_hz,
        mode=cfg.encoding_mode,
        conv_weights_path=conv_weights_path,
        device=device
    )


def _prepare_image_for_encoding(image: np.ndarray, cfg) -> np.ndarray:
    """
    Prepare image for spike encoding based on encoding mode.

    Args:
        image: 28x28 MNIST image (values 0-1)
        cfg: ExperimentConfig

    Returns:
        Prepared image for SNNStimulator (values 0-255)
    """
    if cfg.encoding_mode == 'intensity_to_neuron':
        # Downsample if needed
        if cfg.downsample_factor > 1:
            return downsample_image(image * 255.0, cfg.downsample_factor)
        else:
            return image * 255.0

    elif cfg.encoding_mode == 'conv_feature_to_neuron':
        # CNN expects full 28x28 image
        return image * 255.0

    else:
        raise ValueError(f"Unknown encoding mode: {cfg.encoding_mode}")


if __name__ == "__main__":
    # Demo: Load data and precompute spike trains
    import sys
    sys.path.append('..')
    from config.experiment_config import get_config

    # Create small config for demo
    cfg = get_config()
    cfg.target_classes = [0, 1]
    cfg.encoding_mode = 'intensity_to_neuron'

    # Load data
    data = setup_mnist_data(cfg)
    print(f"\nLoaded {len(data)} examples")
    print(f"Train: {len(data.train_indices)}, Test: {len(data.test_indices)}")

    # Precompute a few spike trains (not all for demo)
    print("\nPrecomputing spike trains for first 10 examples...")
    spike_trains = {}
    stimulator = SNNStimulator(
        total_time_ms=cfg.mnist_stim_duration_ms,
        max_freq_hz=cfg.max_freq_hz,
        mode=cfg.encoding_mode
    )

    for idx in range(10):
        image = data.get_image(idx)
        if cfg.downsample_factor > 1:
            image = downsample_image(image * 255.0, cfg.downsample_factor)
        else:
            image = image * 255.0
        spike_trains[idx] = stimulator.generate_spikes(image)

    print(f"Precomputed {len(spike_trains)} spike trains")
    print(f"Example: Image 0 has spike trains for {len(spike_trains[0])} input neurons")
