"""
Fitness evaluation and prediction logic for GA-evolved SNNs.

This module handles:
1. Prediction from network activity (winner-take-all)
2. Fitness calculation (balanced accuracy)
3. Chromosome evaluation for GA
"""

import numpy as np
from typing import Tuple, Dict
from .simulation import run_snn_simulation


def calculate_prediction(activity_record, layer_indices, n_classes) -> Tuple[int, Dict[int, int]]:
    """
    Calculate predicted class from network activity using winner-take-all.

    The output neuron that fires the MOST during the entire simulation
    determines the predicted class.

    Args:
        activity_record: List of active neuron indices per timestep
        layer_indices: List of (start_idx, end_idx) for each layer
        n_classes: Number of output classes

    Returns:
        Tuple of (predicted_label, output_spike_counts)
        - predicted_label: 0 to n_classes-1, or -1 if no spikes
        - output_spike_counts: Dict mapping output neuron index -> spike count

    Example:
        >>> pred, counts = calculate_prediction(activity, layers, 5)
        >>> # counts might be {10: 5, 11: 2, 12: 15, 13: 1, 14: 0}
        >>> # pred would be 2 (neuron 12 = output index 2 had most spikes)
    """
    if not layer_indices:
        return -1, {}

    # Get output layer indices
    output_start_idx, output_end_idx = layer_indices[-1]
    num_output_neurons = output_end_idx - output_start_idx

    if num_output_neurons != n_classes:
        print(f"Warning: Output layer size ({num_output_neurons}) != n_classes ({n_classes})")
        return -1, {}

    # Count spikes for each output neuron
    output_spike_counts = {i: 0 for i in range(output_start_idx, output_end_idx)}

    for step_spikes in activity_record:
        for neuron_idx in step_spikes:
            if output_start_idx <= neuron_idx < output_end_idx:
                output_spike_counts[neuron_idx] += 1

    # Winner-take-all: neuron with most spikes wins
    total_output_spikes = sum(output_spike_counts.values())
    predicted_label = -1

    if total_output_spikes > 0:
        winner_neuron_idx = max(output_spike_counts, key=output_spike_counts.get)
        predicted_label = winner_neuron_idx - output_start_idx

    return predicted_label, output_spike_counts


def calculate_balanced_accuracy(true_labels: np.ndarray,
                                predicted_labels: np.ndarray,
                                n_classes: int) -> float:
    """
    Calculate balanced accuracy (average per-class recall).

    Balanced accuracy is better than standard accuracy for imbalanced datasets
    because it weights all classes equally. Only averages over classes that
    are actually present in the batch.

    Args:
        true_labels: Array of true class labels (0 to n_classes-1)
        predicted_labels: Array of predicted class labels
        n_classes: Number of classes

    Returns:
        Balanced accuracy score (0.0 to 1.0)

    Example:
        >>> true = np.array([0, 0, 1, 1, 2, 2])
        >>> pred = np.array([0, 1, 1, 1, 2, 0])
        >>> balanced_acc = calculate_balanced_accuracy(true, pred, 3)
        >>> # Class 0: 1/2 correct, Class 1: 2/2 correct, Class 2: 1/2 correct
        >>> # Balanced acc = (0.5 + 1.0 + 0.5) / 3 = 0.667
    """
    if len(true_labels) == 0:
        return 0.0

    # Only calculate recall for classes that are present
    class_recalls = []

    for cls_label in range(n_classes):
        # Mask for this class
        true_mask = (true_labels == cls_label)
        num_true_cls = np.sum(true_mask)

        if num_true_cls > 0:
            # Count correct predictions for this class
            correct_cls_count = np.sum(predicted_labels[true_mask] == cls_label)
            recall = correct_cls_count / num_true_cls
            class_recalls.append(recall)
        # If class not present, skip it (don't assign 1.0)

    # Average only over present classes
    if len(class_recalls) == 0:
        return 0.0

    balanced_accuracy = np.mean(class_recalls)

    # Ensure no NaN
    return balanced_accuracy if not np.isnan(balanced_accuracy) else 0.0


def evaluate_chromosome_fitness(chromosome_weights, network, connection_map,
                                layer_indices, n_classes, data, label_map,
                                eval_indices, spike_trains, cfg) -> float:
    """
    Evaluate fitness of a chromosome using confidence/energy tradeoff.

    This implements a free energy principle-based fitness function with
    blended accuracy and margin components:

    1. Sets the chromosome's weights in the network
    2. Runs simulations on a batch of MNIST examples
    3. For each sample:
       - Calculate margin = (S_true - S_pred) / (S_true + S_pred + eps)
       - Calculate is_correct = 1 if predicted correctly, else 0
       - Blend: confidence = 0.7 * is_correct + 0.3 * max(margin, 0)
       - Sample score = confidence / (1 + alpha * energy)
    4. Returns mean sample score

    The blended approach:
    - Provides strong signal for correct predictions (70% weight)
    - Gives partial credit via margin even when wrong (30% weight)
    - Creates gradient for GA to improve even on incorrect samples
    - Energy penalty encourages metabolically efficient solutions

    Args:
        chromosome_weights: Array of synaptic weights to evaluate
        network: LayeredNeuronalNetworkVectorized instance (persistent)
        connection_map: List of (src, tgt) tuples defining connections
        layer_indices: List of (start, end) for each layer
        n_classes: Number of output classes
        data: MNISTDataset instance
        label_map: Dict mapping original labels to 0-indexed classes
        eval_indices: Indices of examples to evaluate on
        spike_trains: Dict of precomputed spike trains
        cfg: ExperimentConfig

    Returns:
        Fitness score (0.0 to ~0.7, trading off confidence vs energy)
        Returns -inf on error

    Note:
        This function is called in parallel by the GA, so it must be
        thread-safe and use the persistent network object correctly.
    """
    try:
        # Reset and configure network
        network.reset_all()
        network.set_weights_sparse(chromosome_weights, connection_map)
    except Exception as e:
        # Return lowest fitness on setup error
        return -np.inf

    # Validate evaluation indices
    if not hasattr(eval_indices, '__len__') or len(eval_indices) == 0:
        return 0.0

    # Filter to valid indices
    valid_eval_indices = [idx for idx in eval_indices if 0 <= idx < len(data.labels)]
    if not valid_eval_indices:
        return 0.0

    # Accumulate fitness across samples
    fitness_sum = 0.0
    num_samples = 0
    eps = 1e-6  # Prevent division by zero
    alpha = cfg.fitness_alpha  # Energy penalty coefficient

    # Evaluate on each example
    for idx in valid_eval_indices:
        try:
            # Get true label (mapped to 0..n_classes-1)
            true_original_label = data.labels[idx]
            if true_original_label not in label_map:
                continue  # Skip if not a target class
            true_mapped_label = label_map[true_original_label]

            # Get precomputed spikes
            mnist_spike_times = spike_trains.get(idx, None)
            if mnist_spike_times is None:
                continue  # Skip if encoding failed

            # Run simulation
            activity_record = run_snn_simulation(
                network,
                mnist_spike_times,
                cfg,
                show_progress=False
            )

            # Get prediction and spike counts
            predicted_label, output_spike_counts = calculate_prediction(
                activity_record,
                layer_indices,
                n_classes
            )

            # Get output layer boundaries
            output_start_idx, output_end_idx = layer_indices[-1]

            # Calculate confidence margin (ALWAYS, even when wrong)
            # This gives partial credit and provides gradient for improvement
            S_true = output_spike_counts.get(output_start_idx + true_mapped_label, 0)

            # Find max spike count among ALL incorrect classes (not just predicted)
            # This is crucial: when correct, gives margin; when wrong, still gives gradient
            S_other = 0
            for class_idx in range(n_classes):
                if class_idx != true_mapped_label:
                    neuron_idx = output_start_idx + class_idx
                    S_other = max(S_other, output_spike_counts.get(neuron_idx, 0))

            # Margin: how much stronger is true class vs best wrong class?
            # When CORRECT and S_true >> S_other: large positive margin (good!)
            # When CORRECT but S_true ≈ S_other: small positive margin (risky)
            # When WRONG: negative margin (true class lost to runner-up)
            margin = (S_true - S_other) / (S_true + S_other + eps)

            # Binary accuracy component (1 if correct, 0 if wrong)
            is_correct = 1.0 if predicted_label == true_mapped_label else 0.0

            # Blended confidence:
            # - 70% from binary accuracy (strong signal)
            # - 30% from margin (provides gradient even when wrong)
            # When correct with large margin: 0.7 + 0.3 = 1.0 (perfect)
            # When correct with small margin: 0.7 + ~0 = 0.7 (risky)
            # When wrong but S_true strong: 0.0 + up to 0.3 (partial credit)
            # When wrong and S_true weak: 0.0 + 0.0 (no credit)
            confidence = 0.7 * is_correct + 0.3 * max(margin, 0.0)
            confidence = max(confidence, 0.0)  # Ensure non-negative

            # Calculate total energy (all network spikes)
            total_spikes = sum(len(step_spikes) for step_spikes in activity_record)

            # Sample score: confidence divided by energy cost
            sample_score = confidence / (1 + alpha * total_spikes)
            fitness_sum += sample_score
            num_samples += 1

        except Exception as e:
            # Skip example on error
            continue

    # Return mean fitness
    if num_samples == 0:
        return 0.0

    return fitness_sum / num_samples


if __name__ == "__main__":
    # Demo: Test prediction and balanced accuracy calculation
    print("Testing prediction logic...")

    # Mock activity record where output neurons are indices 10-14 (5 classes)
    layer_indices = [(0, 10), (10, 15)]  # Input, output
    activity_record = [
        [10, 10, 12],     # Output neuron 0 fires twice, neuron 2 once
        [12, 12, 12],     # Output neuron 2 fires three times
        [10, 14],         # Output neurons 0 and 4 fire
        [12]              # Output neuron 2 fires
    ]

    pred, counts = calculate_prediction(activity_record, layer_indices, 5)
    print(f"Prediction: {pred}")
    print(f"Spike counts: {counts}")
    print(f"Expected: neuron 12 (class 2) should win with 5 spikes")

    # Test balanced accuracy
    print("\nTesting balanced accuracy...")
    true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1])

    acc = calculate_balanced_accuracy(true, pred, 3)
    print(f"Balanced accuracy: {acc:.3f}")
    print(f"Class 0: 2/3 correct = 0.667")
    print(f"Class 1: 2/3 correct = 0.667")
    print(f"Class 2: 2/3 correct = 0.667")
    print(f"Average: 0.667")
