import numpy as np
from tqdm import tqdm
import random 

def run_unified_simulation(network, duration=1000.0, dt=0.1, stim_interval=None, stim_interval_strength=10, 
                         stim_fraction=0.01, stim_neuron=None, track_neurons=None, stochastic_stim=False,
                         no_stimulation=False):
    """
    Run simulation for specified duration, with options for both regular and stochastic stimulation,
    optionally tracking specific neurons.
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network to simulate
    duration : float
        Duration of simulation in ms
    dt : float
        Time step size in ms
    stim_interval : float or None
        Interval for regular stimulation in ms
    stim_neuron : int or None
        Index of neuron to stimulate regularly
    track_neurons : list or None
        List of neuron indices to track in detail
    stochastic_stim : bool
        Whether to use stochastic stimulation (random timing and strength)
    no_stimulation : bool
        If True, no external stimulation will be applied
    
    Returns:
    --------
    activity_record : list
        List of active neuron indices at each time step
    neuron_data : dict
        Dictionary of tracked neuron data (if track_neurons provided)
    stimulation_record : dict
        Dictionary recording which neurons were stimulated at which times
    """
    n_steps = int(duration / dt)
    activity_record = []
    
    # New: Record which neurons were stimulated and when
    stimulation_record = {
        'times': [],          # Times of stimulation (in ms)
        'neurons': []         # List of lists of stimulated neurons at each time
    }
    
    # Initialize tracking for specific neurons if requested
    neuron_data = {}
    if track_neurons is not None:
        for idx in track_neurons:
            if 0 <= idx < network.n_neurons:
                # Clear any existing history
                network.neurons[idx].v_history = []
                network.neurons[idx].g_e_history = []
                network.neurons[idx].g_i_history = []
                network.neurons[idx].i_syn_history = []
                network.neurons[idx].spike_times = []
                
                # Initialize tracking dict
                neuron_data[idx] = {
                    'v_history': [],
                    'g_e_history': [],
                    'g_i_history': [],
                    'i_syn_history': [],
                    'spike_times': [],
                    'stim_times': [],  # Track when this neuron was specifically stimulated
                    'is_inhibitory': network.neurons[idx].is_inhibitory
                }
    
    # Create progress bar
    progress_bar = tqdm(
        total=n_steps,
        desc="Simulation Progress",
        unit="steps",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} steps [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Reset network
    network.reset_all()
    
    # Initial random stimulation to jump-start activity - skip if no_stimulation is True
    if stochastic_stim and not no_stimulation:
        # Get 5 random neurons to stimulate
        initial_stim_neurons = np.random.choice(network.n_neurons, size=5, replace=False)
        print('Random 5 neurons stimulated at initiation:', initial_stim_neurons)
        
        # Record this stimulation
        stimulation_record['times'].append(0.0)
        stimulation_record['neurons'].append(list(initial_stim_neurons))
        
        for idx in initial_stim_neurons:
            network.stimulate_neuron(idx, current=15)
            
            # Record stimulation time if this neuron is being tracked
            if track_neurons is not None and idx in neuron_data:
                neuron_data[idx]['stim_times'].append(0.0)  # At time 0
    
    for step in range(n_steps):
        time = step * dt
        
        # Skip all stimulation logic if no_stimulation is True
        if not no_stimulation:
            # Stimulation logic
            if stochastic_stim and np.random.random() < (dt / 150):  # ~1 per 150ms on average
                # Random number of neurons (1-5) with random current strength
                n_stim = np.random.randint(1, 6)
                current_strength = 20.0 + 10.0 * np.random.random()  # Random current 20-30
                
                # Select random neurons to stimulate
                stim_neurons = np.random.choice(network.n_neurons, size=int(network.n_neurons*stim_fraction), replace=False)
                
                # Record this stimulation event
                stimulation_record['times'].append(time)
                stimulation_record['neurons'].append(list(stim_neurons))
                
                # Apply stimulation to each selected neuron
                for idx in stim_neurons:
                    network.stimulate_neuron(idx, current=current_strength)
                    
                    # Record stimulation time if this neuron is being tracked
                    if track_neurons is not None and idx in neuron_data:
                        neuron_data[idx]['stim_times'].append(time)
            
            # Regular stimulation if specified
            elif stim_interval and step % int(stim_interval / dt) == 0 and not step == 0:
                # Select random neurons to stimulate
                stim_neurons = np.random.choice(network.n_neurons, size=int(network.n_neurons*stim_fraction), replace=False)
                
                # Record this stimulation event
                stimulation_record['times'].append(time)
                stimulation_record['neurons'].append(list(stim_neurons))
                
                # Apply stimulation
                for idx in stim_neurons:
                    network.stimulate_neuron(idx, current=stim_interval_strength)
                    
                    # Record stimulation time if this neuron is being tracked
                    if track_neurons is not None and idx in neuron_data:
                        neuron_data[idx]['stim_times'].append(time)
                
            if stim_neuron is not None and step % int(stim_interval / dt) == 0:
                network.stimulate_neuron(stim_neuron, current=10)
                
                # Record this specific stimulation
                stimulation_record['times'].append(time)
                stimulation_record['neurons'].append([stim_neuron])
                
                # Record stimulation time if this neuron is being tracked
                if track_neurons is not None and stim_neuron in neuron_data:
                    neuron_data[stim_neuron]['stim_times'].append(time)
        
        # Save neuron state variables BEFORE updating
        if track_neurons is not None:
            for idx in track_neurons:
                if idx in neuron_data:
                    neuron = network.neurons[idx]
                    neuron_data[idx]['v_history'].append(neuron.v)
                    neuron_data[idx]['g_e_history'].append(neuron.g_e)
                    neuron_data[idx]['g_i_history'].append(neuron.g_i)
                    
                    # Calculate total synaptic current for backward compatibility
                    i_syn = (neuron.g_e * (neuron.e_reversal - neuron.v) + 
                            neuron.g_i * (neuron.i_reversal - neuron.v))
                    neuron_data[idx]['i_syn_history'].append(i_syn)
        
        # Update network
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)
        
        # Record any spikes that occurred in this time step
        if track_neurons is not None:
            for idx in track_neurons:
                if idx in neuron_data and idx in active_indices:
                    neuron_data[idx]['spike_times'].append(time)
        
        # Update progress bar
        progress_bar.update(1)
        
        # Check for criticality every 1000 steps
        if step % 1000 == 0 and len(network.avalanche_sizes) > 20:
            check_criticality_during_run(network, step)
    
    # Close progress bar
    progress_bar.close()
    
    # Print spike detection summary if tracking
    if track_neurons is not None:
        print("\nSpike detection summary:")
        for idx in track_neurons:
            if idx in neuron_data:
                num_spikes = len(neuron_data[idx]['spike_times'])
                print(f"Neuron {idx}: {num_spikes} spikes detected")
    
    # Print stimulation summary
    print(f"\nTotal stimulation events: {len(stimulation_record['times'])}")
    if stimulation_record['times']:
        print(f"First few stimulation times: {stimulation_record['times'][:5]}")
    
    # Count unique stimulated neurons
    all_stim_neurons = set()
    for neurons in stimulation_record['neurons']:
        all_stim_neurons.update(neurons)
    
    if all_stim_neurons:
        print(f"Total unique neurons stimulated: {len(all_stim_neurons)}")
    elif no_stimulation:
        print("No neurons were stimulated (no-stimulation mode)")
    else:
        print("No neurons were stimulated during this simulation")
    
    return activity_record, neuron_data, stimulation_record

def Layered_run_unified_simulation(network, duration=1000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
                         stim_fraction=0.01, stim_target_indices=None,
                         stim_pulse_duration_ms=1.0,
                         track_neurons=None, stochastic_stim=False,
                         no_stimulation=False):
    """
    Runs the network simulation handling various stimulation types and tracking.

    Args:
        network (ExtendedNeuronalNetworkWithReversal): The network object to simulate.
        duration (float): Total simulation time (ms).
        dt (float): Simulation time step (ms).
        stim_interval (float, optional): Interval between regular stimulation pulses (ms).
        stim_interval_strength (float): Conductance change applied during stimulation.
        stim_fraction (float): Fraction of target neurons to stimulate in each pulse.
        stim_target_indices (list, optional): Indices of neurons eligible for stimulation. If None, all neurons are eligible.
        stim_pulse_duration_ms (float): Duration for which a stimulation pulse stays active (ms).
        track_neurons (list, optional): Indices of specific neurons to record detailed state history for.
        stochastic_stim (bool): If True, applies stimulation pulses at random times (average rate defined internally).
        no_stimulation (bool): If True, overrides other stimulation settings and applies no external stimulus.

    Returns:
        tuple: (activity_record, neuron_data, stimulation_record)
            - activity_record (list): List of lists, where each inner list contains indices of neurons spiking at that time step.
            - neuron_data (dict): Dictionary mapping tracked neuron indices to their state histories (voltage, conductances, etc.).
            - stimulation_record (dict): Records details of applied stimulation pulses (start times, neurons stimulated, duration).
    """
    print(f"--- Running Simulation ({duration}ms) ---")
    n_steps = int(duration / dt) # Total number of simulation steps
    activity_record = [] # To store spiking neuron indices per step
    # Record stimulation events: start times, stimulated neurons per pulse, and pulse duration
    stimulation_record = {'pulse_starts': [], 'neurons': [], 'pulse_duration_ms': stim_pulse_duration_ms}
    # Initialize data structure for tracked neurons
    neuron_data = {
        idx: {
            'v_history': [], 'g_e_history': [], 'g_i_history': [], 'i_syn_history': [],
            'spike_times': [], 'stim_times': [],
            'is_inhibitory': network.neurons[idx].is_inhibitory if idx < len(network.neurons) and network.neurons[idx] else False
        } for idx in (track_neurons or []) # Create entries only if tracking is requested
    }

    # --- Simulation Setup ---
    ongoing_stimulations = {} # Dictionary to track active stimulus pulses: {neuron_idx: end_time}
    # Calculate stimulation interval in steps for regular stimulation
    stim_interval_steps = int(stim_interval / dt) if stim_interval is not None else None

    # Determine the population of neurons eligible for stimulation
    stimulation_population = list(stim_target_indices) if stim_target_indices is not None else list(range(network.n_neurons))
    if not stimulation_population:
        print("Warning: Stimulation population is empty. No stimulation will be applied.")
        no_stimulation = True # Force no stimulation if population is empty

    # --- Simulation Loop ---
    for step in tqdm(range(n_steps), desc="Simulation"):
        current_time = step * dt # Current simulation time in ms
        newly_stimulated_indices = [] # Neurons receiving a *new* stimulus pulse in this step

        # --- Update Ongoing Stimulations ---
        expired_stims = [] # Neurons whose stimulation pulse ends in this step
        currently_stimulated_set = set() # Set of neurons actively stimulated *in this step*
        # Check each neuron currently under stimulation
        for neuron_idx, end_time in ongoing_stimulations.items():
            if current_time >= end_time: # If pulse duration has elapsed
                expired_stims.append(neuron_idx)
            else: # If pulse is still active
                # Ensure neuron exists before applying stimulus
                if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                    pulse_strength = stim_interval_strength
                    # Apply the stimulus conductance change to the neuron
                    network.neurons[neuron_idx].apply_external_stimulus(pulse_strength)
                    currently_stimulated_set.add(neuron_idx) # Mark as actively stimulated

        # Remove expired stimulations and turn off stimulus conductance
        for neuron_idx in expired_stims:
            # Ensure neuron exists before turning off stimulus
            if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
                 network.neurons[neuron_idx].apply_external_stimulus(0.0) # Set external conductance to zero
            # Remove from the ongoing tracking dictionary
            if neuron_idx in ongoing_stimulations:
                del ongoing_stimulations[neuron_idx]

        # --- Apply New Stimulation Pulses ---
        if not no_stimulation and stimulation_population: # Check if stimulation is enabled and possible
            # Determine number of neurons to stimulate per pulse
            num_to_stimulate = max(1, int(len(stimulation_population) * stim_fraction))
            apply_new_stim_pulse = False # Flag to indicate if a new pulse should start

            # Check conditions for starting a new pulse
            # Stochastic: random chance based on dt (approx. 1 pulse per 100ms average)
            if stochastic_stim and random.random() < (dt / 100):
                 apply_new_stim_pulse = True
            # Regular interval: check if current step matches the interval
            elif stim_interval_steps and (step % stim_interval_steps == 0):
                 apply_new_stim_pulse = True

            # If a new pulse should start
            if apply_new_stim_pulse:
                 # Randomly select target neurons from the eligible population
                 target_neurons_for_pulse = random.sample(stimulation_population, min(num_to_stimulate, len(stimulation_population)))
                 # Calculate the end time for this new pulse
                 stim_end_time = current_time + stim_pulse_duration_ms

                 # Apply the pulse to selected neurons
                 for idx in target_neurons_for_pulse:
                     if idx not in ongoing_stimulations: # Avoid re-stimulating if already active
                         ongoing_stimulations[idx] = stim_end_time # Track the pulse end time
                         newly_stimulated_indices.append(idx) # Record as newly stimulated
                         # Ensure neuron exists before applying stimulus
                         if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                            network.neurons[idx].apply_external_stimulus(stim_interval_strength) # Apply conductance
                            currently_stimulated_set.add(idx) # Mark as actively stimulated

                 # Record the stimulation event if any neurons were newly stimulated
                 if newly_stimulated_indices:
                     stimulation_record['pulse_starts'].append(current_time)
                     # Compatibility with plotting functions expecting 'times' key
                     stimulation_record['times'] = stimulation_record['pulse_starts']
                     stimulation_record['neurons'].append(newly_stimulated_indices)

        # --- Record Tracked Neuron Data (Before Update) ---
        if track_neurons:
            for idx in track_neurons:
                 # Ensure neuron exists before accessing data
                 if idx < len(network.neurons) and network.neurons[idx]:
                    neuron = network.neurons[idx]
                    # Record state variables
                    neuron_data[idx]['v_history'].append(neuron.v)
                    neuron_data[idx]['g_e_history'].append(neuron.g_e)
                    neuron_data[idx]['g_i_history'].append(neuron.g_i)
                    # Calculate and record total synaptic current for this neuron
                    i_syn_internal = neuron.g_e * (neuron.e_reversal - neuron.v) + neuron.g_i * (neuron.i_reversal - neuron.v)
                    i_syn_external = neuron.external_stim_g * (neuron.e_reversal - neuron.v)
                    neuron_data[idx]['i_syn_history'].append(i_syn_internal + i_syn_external)
                    # Record if this neuron was newly stimulated in this step
                    if idx in newly_stimulated_indices:
                        neuron_data[idx]['stim_times'].append(current_time)

        # --- Update Network State ---
        active_indices = network.update_network(dt) # Update all neurons and get spikes
        activity_record.append(active_indices) # Store indices of spiking neurons

        # Record spike times for tracked neurons
        if track_neurons:
             for idx in active_indices:
                  if idx in neuron_data:
                      neuron_data[idx]['spike_times'].append(current_time)

    # --- Cleanup After Loop ---
    # Ensure any lingering stimulations are turned off
    for neuron_idx in list(ongoing_stimulations.keys()):
        if 0 <= neuron_idx < len(network.neurons) and network.neurons[neuron_idx]:
             network.neurons[neuron_idx].apply_external_stimulus(0.0)
        del ongoing_stimulations[neuron_idx]

    print(f"--- Simulation Finished ---")
    return activity_record, neuron_data, stimulation_record



def check_criticality_during_run(network, step, window=1000):
    """Check criticality markers every N steps without breaking progress bar"""
    if step % window == 0 and len(network.avalanche_sizes) > 20:
        sizes = network.avalanche_sizes[-100:] if len(network.avalanche_sizes) > 100 else network.avalanche_sizes
        # Simple power-law check: plot log count vs log size in bins
        size_bins = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 10)
        hist, _ = np.histogram(sizes, bins=size_bins)
        if sum(hist) > 0:
            # Check if bins follow approximate power-law (roughly linear in log-log)
            log_counts = np.log10(hist + 1)  # Add 1 to avoid log(0)
            log_bins = np.log10(size_bins[:-1])
            # Calculate Pearson correlation - high negative correlation suggests power law
            if len(log_counts) > 3:
                # Build status message
                status_lines = []
                corr = np.corrcoef(log_bins, log_counts)[0,1]
                status_lines.append(f"\nTime {step*0.1:.1f}ms: Power-law correlation: {corr:.3f} (closer to -1 is better)")
                
                # Also check distribution exponent
                if len(sizes) > 20:
                    try:
                        from scipy import stats
                        alpha_size, _, _ = stats.powerlaw.fit(sizes)
                        status_lines.append(f"Current size exponent: α = {alpha_size:.3f} (target: 1.3-1.7)")
                        
                        if 1.2 <= alpha_size <= 1.8:
                            status_lines.append("*** GETTING CLOSE TO CRITICALITY! ***")
                    except Exception as e:
                        pass
                
                # Print status with trailing newline to avoid breaking progress bar
                from tqdm import tqdm
                tqdm.write("\n".join(status_lines))