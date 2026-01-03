"""
SNN simulation utilities for MNIST classification.

Handles running the vectorized SNN with MNIST spike input.
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
