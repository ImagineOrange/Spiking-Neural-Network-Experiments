import numpy as np
from tqdm import tqdm
import random

# Keep the original run_unified_simulation if needed, or remove if unused
def run_unified_simulation(network, duration=1000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
                         stim_fraction=0.01, stim_neuron=None, track_neurons=None, stochastic_stim=False,
                         no_stimulation=False):
    """
    Original simulation function (kept for reference or other uses).
    Does NOT handle mnist_input_spikes.
    """
    n_steps = int(duration / dt)
    activity_record = []
    stimulation_record = {'times': [], 'neurons': []}
    neuron_data = {
        idx: {
            'v_history': [], 'g_e_history': [], 'g_i_history': [], 'i_syn_history': [],
            'spike_times': [], 'stim_times': [],
            'is_inhibitory': network.neurons[idx].is_inhibitory if idx < len(network.neurons) and network.neurons[idx] else False
        } for idx in (track_neurons or [])
    }
    progress_bar = tqdm(total=n_steps, desc="Sim (Original)", unit="steps")
    network.reset_all()
    # --- Original stimulation logic ---
    if stochastic_stim and not no_stimulation:
        initial_stim_neurons = np.random.choice(network.n_neurons, size=5, replace=False)
        stimulation_record['times'].append(0.0)
        stimulation_record['neurons'].append(list(initial_stim_neurons))
        for idx in initial_stim_neurons: network.stimulate_neuron(idx, current=15)
    # --- Loop ---
    for step in range(n_steps):
        time = step * dt
        if not no_stimulation:
             if stochastic_stim and np.random.random() < (dt / 150):
                 n_stim = np.random.randint(1, 6)
                 current_strength = 20.0 + 10.0 * np.random.random()
                 stim_neurons = np.random.choice(network.n_neurons, size=int(network.n_neurons*stim_fraction), replace=False)
                 stimulation_record['times'].append(time); stimulation_record['neurons'].append(list(stim_neurons))
                 for idx in stim_neurons: network.stimulate_neuron(idx, current=current_strength)
             elif stim_interval and step % int(stim_interval / dt) == 0 and not step == 0:
                 stim_neurons = np.random.choice(network.n_neurons, size=int(network.n_neurons*stim_fraction), replace=False)
                 stimulation_record['times'].append(time); stimulation_record['neurons'].append(list(stim_neurons))
                 for idx in stim_neurons: network.stimulate_neuron(idx, current=stim_interval_strength)
             if stim_neuron is not None and step % int(stim_interval / dt) == 0:
                 network.stimulate_neuron(stim_neuron, current=10)
                 stimulation_record['times'].append(time); stimulation_record['neurons'].append([stim_neuron])

        # Record state BEFORE update
        if track_neurons:
            for idx in track_neurons:
                 if idx in neuron_data:
                     neuron = network.neurons[idx]
                     neuron_data[idx]['v_history'].append(neuron.v)
                     neuron_data[idx]['g_e_history'].append(neuron.g_e)
                     neuron_data[idx]['g_i_history'].append(neuron.g_i)
                     i_syn = (neuron.g_e * (neuron.e_reversal - neuron.v) +
                              neuron.g_i * (neuron.i_reversal - neuron.v))
                     neuron_data[idx]['i_syn_history'].append(i_syn)
        # Update network
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)
        # Record spikes
        if track_neurons:
            for idx in track_neurons:
                 if idx in neuron_data and idx in active_indices:
                     neuron_data[idx]['spike_times'].append(time)
        progress_bar.update(1)
    progress_bar.close()
    return activity_record, neuron_data, stimulation_record


# --- CORRECTED VERSION of Layered_run_unified_simulation ---
def Layered_run_unified_simulation(network, duration=1000.0, dt=0.1,
                                 # Original stim params (will be disabled if mnist_input_spikes is used)
                                 stim_interval=None, stim_interval_strength=10,
                                 stim_fraction=0.01, stim_target_indices=None,
                                 stim_pulse_duration_ms=1.0,
                                 # *** ADDED MNIST INPUT PARAM ***
                                 mnist_input_spikes=None,
                                 # Other params
                                 track_neurons=None, stochastic_stim=False,
                                 no_stimulation=False, show_progress=False): # Added show_progress flag
    """
    Runs the network simulation handling various stimulation types and tracking.
    MODIFIED: Accepts pre-generated MNIST spike times (`mnist_input_spikes`).
              If provided, it overrides other stimulation methods.
              Includes optional progress bar control.
    """
    print(f"--- Running Simulation ({duration}ms, dt={dt}ms) ---")
    mnist_spikes_by_step = {} # Initialize dictionary to store MNIST spikes by step index

    # --- Pre-process MNIST spike times ---
    if mnist_input_spikes is not None:
        print("Using MNIST spike input. Disabling other stimulation.")
        no_stimulation = True # MNIST provides the stimulation
        stochastic_stim = False
        stim_interval = None
        num_input_neurons = len(mnist_input_spikes) # Should match the size of the first layer

        # Convert MNIST spike times (ms) to simulation step indices
        for neuron_idx, spike_list_ms in enumerate(mnist_input_spikes):
            for time_ms in spike_list_ms:
                step_index = int(round(time_ms / dt))
                # Only consider spikes within the simulation duration
                if step_index >= 0 and step_index < int(duration / dt):
                    if step_index not in mnist_spikes_by_step:
                        mnist_spikes_by_step[step_index] = []
                    # Store the index of the neuron that spikes at this step
                    mnist_spikes_by_step[step_index].append(neuron_idx)
    # --- End Pre-processing ---

    n_steps = int(duration / dt)
    activity_record = []
    # Stimulation record structure needs slight adjustment if using MNIST input
    # Let's keep the original structure but note it might be less relevant with MNIST input
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms, 'times': []}

    # Initialize neuron data tracking
    neuron_data = {
        idx: {
            'v_history': [], 'g_e_history': [], 'g_i_history': [], 'i_syn_history': [],
            'spike_times': [], 'stim_times': [],
            'is_inhibitory': network.neurons[idx].is_inhibitory if idx < len(network.neurons) and network.neurons[idx] else False
        } for idx in (track_neurons or [])
    }

    ongoing_stimulations = {} # Tracks {neuron_idx: end_time} for non-MNIST stim pulses
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None
    stimulation_population = list(stim_target_indices) if stim_target_indices is not None else list(range(network.n_neurons))
    if not stimulation_population: no_stimulation = True # Force no stimulation if population empty


    # --- Simulation Loop ---
    sim_loop_iterator = range(n_steps)
    if show_progress: # Optional progress bar
         sim_loop_iterator = tqdm(range(n_steps), desc="Sim Step", leave=False, ncols=80)

    for step in sim_loop_iterator:
        current_time = step * dt
        newly_stimulated_indices = [] # Track neurons stimulated *in this specific step*

        # --- Apply MNIST Input Stimulation (if provided) ---
        if mnist_input_spikes is not None:
            if step in mnist_spikes_by_step:
                neurons_spiking_now = mnist_spikes_by_step[step]
                # For MNIST input, treat each spike as a brief pulse of conductance
                # Use stim_interval_strength as the magnitude of the conductance change
                # Use stim_pulse_duration_ms to determine how long it lasts
                stim_end_time = current_time + stim_pulse_duration_ms
                for neuron_idx in neurons_spiking_now:
                    # Ensure the neuron index is valid and the neuron exists
                    if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                        if neuron_idx not in ongoing_stimulations: # Apply only if not already active from a previous step pulse
                            ongoing_stimulations[neuron_idx] = stim_end_time # Track pulse end time
                            newly_stimulated_indices.append(neuron_idx) # Mark as newly stimulated this step
                        # Apply the conductance change (will be applied below in the ongoing check)
                        network.neurons[neuron_idx].apply_external_stimulus(stim_interval_strength)

                # Record this MNIST "stimulation" event (optional, might clutter record)
                # if newly_stimulated_indices:
                #     stimulation_record['pulse_starts'].append(current_time)
                #     stimulation_record['times'] = stimulation_record['pulse_starts'] # Compatibility
                #     stimulation_record['neurons'].append(list(newly_stimulated_indices))
        # --- End MNIST Input ---

        # --- Apply Non-MNIST Stimulation (if not using MNIST input) ---
        elif not no_stimulation and stimulation_population:
            num_to_stimulate = max(1, int(len(stimulation_population) * stim_fraction))
            apply_new_stim_pulse = False
            if stochastic_stim and random.random() < (dt / 100): apply_new_stim_pulse = True
            elif stim_interval_steps and (step % stim_interval_steps == 0): apply_new_stim_pulse = True

            if apply_new_stim_pulse:
                 target_neurons_for_pulse = random.sample(stimulation_population, min(num_to_stimulate, len(stimulation_population)))
                 stim_end_time = current_time + stim_pulse_duration_ms
                 for idx in target_neurons_for_pulse:
                     if idx not in ongoing_stimulations:
                         ongoing_stimulations[idx] = stim_end_time
                         newly_stimulated_indices.append(idx)
                         # Apply conductance change immediately for non-MNIST stim
                         if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                             network.neurons[idx].apply_external_stimulus(stim_interval_strength)
                 if newly_stimulated_indices:
                     stimulation_record['pulse_starts'].append(current_time)
                     stimulation_record['times'] = stimulation_record['pulse_starts']
                     stimulation_record['neurons'].append(list(newly_stimulated_indices))
        # --- End Non-MNIST Stimulation ---


        # --- Update Ongoing Stimulations (Handles decay/turn-off) ---
        expired_stims = []
        currently_stimulated_set = set() # Neurons actively stimulated this step
        for neuron_idx, end_time in list(ongoing_stimulations.items()):
            if current_time >= end_time:
                expired_stims.append(neuron_idx)
            else:
                 # Ensure neuron exists before applying stimulus
                 if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                    # Re-apply stimulus conductance for active pulses
                    # (Neuron's internal state handles conductance decay if needed)
                    network.neurons[neuron_idx].apply_external_stimulus(stim_interval_strength)
                    currently_stimulated_set.add(neuron_idx)

        # Remove expired stimulations and turn off stimulus conductance
        for neuron_idx in expired_stims:
            if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                 network.neurons[neuron_idx].apply_external_stimulus(0.0) # Turn off external stim
            if neuron_idx in ongoing_stimulations:
                 del ongoing_stimulations[neuron_idx]

        # --- Record Tracked Neuron Data (Before Update) ---
        if track_neurons:
            for idx in track_neurons:
                 if 0 <= idx < len(network.neurons) and network.neurons[idx] and idx in neuron_data:
                    neuron = network.neurons[idx]
                    neuron_data[idx]['v_history'].append(neuron.v)
                    neuron_data[idx]['g_e_history'].append(neuron.g_e)
                    neuron_data[idx]['g_i_history'].append(neuron.g_i)
                    # Calculate total synaptic current
                    i_syn_internal = neuron.g_e * (neuron.e_reversal - neuron.v) + neuron.g_i * (neuron.i_reversal - neuron.v)
                    i_syn_external = neuron.external_stim_g * (neuron.e_reversal - neuron.v)
                    neuron_data[idx]['i_syn_history'].append(i_syn_internal + i_syn_external)
                    # Record if stimulated *this step*
                    if idx in currently_stimulated_set:
                         # Avoid adding duplicate times if pulse spans multiple steps
                         if not neuron_data[idx]['stim_times'] or neuron_data[idx]['stim_times'][-1] != current_time:
                              neuron_data[idx]['stim_times'].append(current_time)

        # --- Update Network State ---
        active_indices = network.update_network(dt) # Update all neurons, get spikes
        activity_record.append(active_indices) # Store indices of spiking neurons

        # Record spike times for tracked neurons
        if track_neurons:
             for idx in active_indices:
                  if idx in neuron_data:
                      neuron_data[idx]['spike_times'].append(current_time)

    # --- Cleanup After Loop ---
    for neuron_idx in list(ongoing_stimulations.keys()):
        if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
             network.neurons[neuron_idx].apply_external_stimulus(0.0)
        if neuron_idx in ongoing_stimulations:
             del ongoing_stimulations[neuron_idx]

    if show_progress and isinstance(sim_loop_iterator, tqdm):
         sim_loop_iterator.close() # Close progress bar if used

    print(f"--- Simulation Finished ---")
    return activity_record, neuron_data, stimulation_record


# Keep the criticality check function if needed
def check_criticality_during_run(network, step, window=1000):
    """Check criticality markers every N steps without breaking progress bar"""
    if step % window == 0 and len(network.avalanche_sizes) > 20:
        sizes = network.avalanche_sizes[-100:] if len(network.avalanche_sizes) > 100 else network.avalanche_sizes
        if not sizes: return
        # Simple power-law check: plot log count vs log size in bins
        min_s, max_s = min(sizes), max(sizes)
        if min_s <= 0 or max_s <= 0 or min_s == max_s: return # Avoid log errors
        size_bins = np.logspace(np.log10(min_s), np.log10(max_s), 10)
        hist, _ = np.histogram(sizes, bins=size_bins)
        valid_hist = hist[hist > 0]
        if sum(valid_hist) > 0 and len(valid_hist) > 3:
            # Check if bins follow approximate power-law (roughly linear in log-log)
            log_counts = np.log10(valid_hist)
            valid_bin_centers = (size_bins[:-1] + size_bins[1:]) / 2
            log_bins = np.log10(valid_bin_centers[hist > 0])
            # Calculate Pearson correlation - high negative correlation suggests power law
            if len(log_counts) > 3:
                status_lines = []
                try:
                    corr = np.corrcoef(log_bins, log_counts)[0,1]
                    status_lines.append(f"\nTime {step*0.1:.1f}ms: Power-law correlation: {corr:.3f} (target: ~ -1)")
                except Exception: # Catch potential errors in corrcoef
                    status_lines.append(f"\nTime {step*0.1:.1f}ms: Could not compute power-law correlation.")

                # Also check distribution exponent
                if len(sizes) > 20:
                    try:
                        from scipy import stats
                        # Filter sizes for fitting (needs positive values)
                        fit_sizes = [s for s in sizes if s > 0]
                        if fit_sizes:
                            alpha_size, _, _ = stats.powerlaw.fit(fit_sizes, floc=0) # Force location=0
                            status_lines.append(f"Current size exponent: α = {alpha_size:.3f} (target: 1.3-1.7)")
                            if 1.2 <= alpha_size <= 1.8: status_lines.append("*** Potential Criticality Detected! ***")
                    except Exception as e: pass # Ignore fit errors silently

                # Print status with trailing newline to avoid breaking progress bar
                from tqdm import tqdm
                tqdm.write("\n".join(status_lines))
