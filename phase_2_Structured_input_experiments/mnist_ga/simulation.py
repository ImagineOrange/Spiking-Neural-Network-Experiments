"""
SNN simulation utilities for MNIST classification.

Handles running the vectorized SNN with MNIST spike input.
Includes phase-aware simulation for STDP training.
"""

import numpy as np
from collections import deque
from tqdm import tqdm
import random


def run_snn_simulation(network, mnist_input_spikes, cfg, show_progress=False):
    """
    Run SNN simulation with precomputed MNIST spike input.

    Args:
        network: LayeredNeuronalNetworkVectorized instance
        mnist_input_spikes: Precomputed spike times for each input neuron
        cfg: ExperimentConfig with simulation parameters
        show_progress: Whether to show progress bar

    Returns:
        activity_record: List of active neuron indices per timestep

    Example:
        >>> activity = run_snn_simulation(network, spike_trains[0], cfg)
        >>> # activity[i] contains indices of neurons that fired at timestep i
    """
    # Reset network state
    network.reset_all()

    # Run simulation
    activity_record = _run_vectorized_simulation(
        network,
        duration=cfg.sim_duration_ms,
        dt=cfg.sim_dt_ms,
        mnist_input_spikes=mnist_input_spikes,
        stim_strength=cfg.stim_strength,
        stim_pulse_duration_ms=cfg.sim_dt_ms,
        show_progress=show_progress
    )

    return activity_record


def _run_vectorized_simulation(network, duration, dt, mnist_input_spikes,
                               stim_strength, stim_pulse_duration_ms,
                               show_progress=False):
    """
    Core vectorized simulation loop.

    Extracted from original MNIST_GA_experiment.py
    """
    # Preprocess MNIST spikes into step-indexed dictionary
    mnist_spikes_by_step = {}
    for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
        for time_ms in spike_list_ms:
            step_index = int(round(time_ms / dt))
            if 0 <= step_index < int(duration / dt):
                if step_index not in mnist_spikes_by_step:
                    mnist_spikes_by_step[step_index] = []
                mnist_spikes_by_step[step_index].append(neuron_idx)

    # Initialize simulation
    n_steps = int(duration / dt)
    activity_record = []
    ongoing_stimulations = {}  # {neuron_idx: end_time_ms}
    current_stim_conductances = np.zeros(network.n_neurons)

    # Setup progress bar if requested
    sim_iterator = range(n_steps)
    if show_progress:
        sim_iterator = tqdm(sim_iterator, desc="Sim", leave=False, ncols=80)

    # Main simulation loop
    for step in sim_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0)
        newly_stimulated_indices = set()

        # Apply MNIST input spikes
        if step in mnist_spikes_by_step:
            neurons_spiking_now = mnist_spikes_by_step[step]
            stim_end_time = current_time + stim_pulse_duration_ms

            for neuron_idx in neurons_spiking_now:
                if 0 <= neuron_idx < network.n_neurons:
                    if neuron_idx not in ongoing_stimulations:
                        ongoing_stimulations[neuron_idx] = stim_end_time
                        newly_stimulated_indices.add(neuron_idx)
                    current_stim_conductances[neuron_idx] = stim_strength

        # Manage ongoing stimulation pulses
        expired_stims = set()
        for neuron_idx, end_time in list(ongoing_stimulations.items()):
            if current_time >= end_time:
                expired_stims.add(neuron_idx)
                current_stim_conductances[neuron_idx] = 0.0
            else:
                if neuron_idx not in newly_stimulated_indices and 0 <= neuron_idx < network.n_neurons:
                    current_stim_conductances[neuron_idx] = stim_strength

        # Remove expired stimulations
        for neuron_idx in expired_stims:
            if neuron_idx in ongoing_stimulations:
                del ongoing_stimulations[neuron_idx]

        # Apply stimulation to network
        network.external_stim_g[:] = current_stim_conductances

        # Update network state
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

    # Cleanup
    network.external_stim_g.fill(0.0)
    if show_progress and isinstance(sim_iterator, tqdm):
        sim_iterator.close()

    return activity_record


def run_stdp_phased_simulation(network, mnist_input_spikes, true_label, cfg,
                                 layer_indices, n_classes, show_progress=False):
    """
    Run phase-aware SNN simulation with STDP learning.

    Phases:
    1. Input period (50ms): Present MNIST spikes, STDP disabled
    2. Readout period (30ms): Network reverberation, count output spikes for prediction
    3. Learning period (50ms): Apply STDP or anti-STDP based on prediction correctness
    4. Rest period (20ms): Let network settle before reset

    Args:
        network: LayeredNeuronalNetworkVectorized instance with STDP capabilities
        mnist_input_spikes: Precomputed spike times for input neurons
        true_label: Ground truth class label for this example
        cfg: Config object with phase timing parameters
        layer_indices: List of (start_idx, end_idx) for each layer
        n_classes: Number of output classes
        show_progress: Whether to show progress bar

    Returns:
        dict containing:
            - 'activity': Full activity record
            - 'predicted_class': Network prediction
            - 'is_correct': Whether prediction matches true label
            - 'output_spike_counts': Spike counts per output neuron
    """
    # Phase durations (ms)
    INPUT_PERIOD = getattr(cfg, 'stdp_input_period', 50.0)
    READOUT_PERIOD = getattr(cfg, 'stdp_readout_period', 30.0)
    LEARNING_PERIOD = getattr(cfg, 'stdp_learning_period', 50.0)
    REST_PERIOD = getattr(cfg, 'stdp_rest_period', 20.0)
    TOTAL_DURATION = INPUT_PERIOD + READOUT_PERIOD + LEARNING_PERIOD + REST_PERIOD

    dt = cfg.sim_dt_ms
    n_steps = int(TOTAL_DURATION / dt)

    # Phase boundaries (in steps)
    input_end_step = int(INPUT_PERIOD / dt)
    readout_end_step = int((INPUT_PERIOD + READOUT_PERIOD) / dt)
    learning_end_step = int((INPUT_PERIOD + READOUT_PERIOD + LEARNING_PERIOD) / dt)

    # Preprocess MNIST spikes (only apply during input period)
    mnist_spikes_by_step = {}
    for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
        for time_ms in spike_list_ms:
            step_index = int(round(time_ms / dt))
            # Only include spikes within input period
            if 0 <= step_index < input_end_step:
                if step_index not in mnist_spikes_by_step:
                    mnist_spikes_by_step[step_index] = []
                mnist_spikes_by_step[step_index].append(neuron_idx)

    # Initialize simulation state
    activity_record = []
    ongoing_stimulations = {}
    current_stim_conductances = np.zeros(network.n_neurons)
    stim_strength = cfg.stim_strength
    stim_pulse_duration_ms = dt

    # STDP state
    network.stdp_enabled = False
    network.learning_phase = False
    predicted_class = None
    output_spike_counts = None

    # Progress bar
    sim_iterator = range(n_steps)
    if show_progress:
        sim_iterator = tqdm(sim_iterator, desc="STDP Sim", leave=False, ncols=80)

    # Main simulation loop
    for step in sim_iterator:
        current_time = step * dt
        current_stim_conductances.fill(0.0)
        newly_stimulated_indices = set()

        # ========== PHASE 1: INPUT PERIOD ==========
        if step < input_end_step:
            # Apply MNIST input spikes
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                stim_end_time = current_time + stim_pulse_duration_ms

                for neuron_idx in neurons_spiking_now:
                    if 0 <= neuron_idx < network.n_neurons:
                        if neuron_idx not in ongoing_stimulations:
                            ongoing_stimulations[neuron_idx] = stim_end_time
                            newly_stimulated_indices.add(neuron_idx)
                        current_stim_conductances[neuron_idx] = stim_strength

            # Manage ongoing stimulations
            expired_stims = set()
            for neuron_idx, end_time in list(ongoing_stimulations.items()):
                if current_time >= end_time:
                    expired_stims.add(neuron_idx)
                    current_stim_conductances[neuron_idx] = 0.0
                else:
                    if neuron_idx not in newly_stimulated_indices:
                        current_stim_conductances[neuron_idx] = stim_strength

            for neuron_idx in expired_stims:
                if neuron_idx in ongoing_stimulations:
                    del ongoing_stimulations[neuron_idx]

        # ========== PHASE 2: READOUT PERIOD ==========
        elif step < readout_end_step:
            # No new input, let network reverberate
            # At the END of readout period, make prediction
            if step == readout_end_step - 1:
                # Count spikes in output layer during input + readout
                output_start, output_end = layer_indices[-1]
                output_spike_counts = np.zeros(n_classes)

                for timestep_activity in activity_record:
                    for neuron_idx in timestep_activity:
                        if output_start <= neuron_idx < output_end:
                            class_idx = neuron_idx - output_start
                            if class_idx < n_classes:
                                output_spike_counts[class_idx] += 1

                # Predict class with most spikes
                predicted_class = int(np.argmax(output_spike_counts))
                is_correct = (predicted_class == true_label)

                # Store prediction result in network
                network.last_prediction_correct = is_correct

        # ========== PHASE 3: LEARNING PERIOD ==========
        elif step < learning_end_step:
            # Enable STDP during learning period
            if step == readout_end_step:
                network.stdp_enabled = True
                network.learning_phase = True

            # At END of learning period, apply weight updates
            if step == learning_end_step - 1:
                if network.last_prediction_correct:
                    # Correct prediction: apply standard STDP
                    network.apply_stdp_updates()
                else:
                    # Incorrect prediction: apply anti-STDP (reversed)
                    network.apply_anti_stdp_updates()

                # Disable STDP for rest period
                network.stdp_enabled = False
                network.learning_phase = False

        # ========== PHASE 4: REST PERIOD ==========
        else:
            # Just let network settle, no special actions
            pass

        # Apply stimulation and update network
        network.external_stim_g[:] = current_stim_conductances
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

    # Cleanup
    network.external_stim_g.fill(0.0)
    if show_progress and isinstance(sim_iterator, tqdm):
        sim_iterator.close()

    # Return results
    return {
        'activity': activity_record,
        'predicted_class': predicted_class,
        'is_correct': network.last_prediction_correct,
        'output_spike_counts': output_spike_counts
    }
