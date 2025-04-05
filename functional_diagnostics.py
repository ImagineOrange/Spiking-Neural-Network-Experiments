# functional_diagnostics.py (Corrected, Resized, 3 Traces, Layered Exp Params, Save Figs)

import numpy as np
import matplotlib.pyplot as plt
import random
import inspect
import os # Import os for path joining
import networkx as nx # Import networkx for graph operations
from tqdm import tqdm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# --- Import Classes from LIF_objects subdirectory ---
from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal
from LIF_objects.LayeredNeuronalNetwork import LayeredNeuronalNetwork
print("Successfully imported Layered LIF objects.")


# --- Plotting Style ---
plt.style.use('dark_background')
DEFAULT_FIG_SIZE = (12, 7) # Define default figure size
SAVE_DIR = "functional_diagnostic_figures" # Directory to save plots
os.makedirs(SAVE_DIR, exist_ok=True) # Create directory if it doesn't exist

# --- Default Simulation Parameters ---
DEFAULT_DT = 0.1
DEFAULT_DURATION = 200.0 # ms

# --- Default Neuron Parameters (Placeholder - will be overwritten in main block) ---
DEFAULT_NEURON_PARAMS = {
     'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
     'tau_m': 10.0, 'tau_ref': 2.0, 'tau_e': 3.0, 'tau_i': 7.0,
     'is_inhibitory': False, 'e_reversal': 0.0, 'i_reversal': -70.0,
     'v_noise_amp': 0.5, 'i_noise_amp': 0.05,
     'adaptation_increment': 0.5, 'tau_adaptation': 100.0
}
# Note: This DEFAULT_NEURON_PARAMS is just a placeholder now.
# The actual parameters used will be defined in the `if __name__ == "__main__":` block.


# ==============================================================================
# SECTION 1: SINGLE NEURON DIAGNOSTICS (from neuron_diagnostics_extended.py)
# Using Layered_LIFNeuronWithReversal
# ==============================================================================
print("\n--- Running Section 1: Single Neuron Behavior Plots ---")

# (Function definitions remain the same as previous version)
def run_single_neuron_extended_diagnostics(neuron_params, dt=DEFAULT_DT, save_dir=SAVE_DIR): # Removed default for neuron_params, added save_dir
    if neuron_params is None: # Should not happen if called from main
        print("Error: neuron_params not provided to run_single_neuron_extended_diagnostics")
        return

    # Filter parameters valid for the neuron constructor
    neuron_init_sig = inspect.signature(Layered_LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_sig if k != 'self'}
    base_params = {k: neuron_params[k] for k in valid_neuron_keys if k in neuron_params}

    # --- 1.1 Small Synaptic Input (Subthreshold) ---
    print("  Plotting: Subthreshold Response")
    neuron1_params = {**base_params, 'v_noise_amp': 0.05, 'i_noise_amp': 0.0}
    neuron1 = Layered_LIFNeuronWithReversal(**{k: neuron1_params[k] for k in valid_neuron_keys if k in neuron1_params})
    T_sub = 50
    time_points_sub = np.arange(0, T_sub, dt)
    input_time_sub = 10.0
    input_weight_sub = 0.05 # This is a conductance jump, not current

    neuron1.reset()
    for i, t in enumerate(time_points_sub):
        if abs(t - input_time_sub) < dt / 2:
            neuron1.receive_spike(input_weight_sub) # receive_spike adds to conductance
        neuron1.update(dt)

    fig1_1 = plt.figure(figsize=DEFAULT_FIG_SIZE) # Use default size
    plt.plot(time_points_sub, neuron1.v_history, label='Membrane Potential (V)')
    plt.axhline(neuron1.v_rest, color='gray', linestyle='--', label=f'Rest ({neuron1.v_rest:.1f}mV)')
    plt.axhline(neuron1.v_threshold, color='red', linestyle='--', label=f'Threshold ({neuron1.v_threshold:.1f}mV)')
    plt.axvline(input_time_sub, color='cyan', linestyle=':', label=f'Input (w={input_weight_sub})')
    plt.title('1.1: Subthreshold Response')
    plt.xlabel('Time (ms)'); plt.ylabel('Membrane Potential (mV)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(neuron1.v_reset - 5, neuron1.v_threshold + 5)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_1_1_subthreshold.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig1_1) # Close figure after saving/showing

    # --- 1.2 Action Potential ---
    print("  Plotting: Action Potential Generation")
    neuron2_params = {**base_params, 'v_noise_amp': 0.02, 'i_noise_amp': 0.0}
    neuron2 = Layered_LIFNeuronWithReversal(**{k: neuron2_params[k] for k in valid_neuron_keys if k in neuron2_params})
    T_ap = 150
    time_points_ap = np.arange(0, T_ap, dt)
    input_time_ap = 10.0
    input_weight_ap = 0.2 # Stronger weight (conductance jump)

    neuron2.reset()
    spike_occurred_time = -1
    for i, t in enumerate(time_points_ap):
        if abs(t - input_time_ap) < dt / 2:
            neuron2.receive_spike(input_weight_ap)
        spiked = neuron2.update(dt)
        if spiked and spike_occurred_time < 0:
             spike_occurred_time = t + dt

    fig1_2 = plt.figure(figsize=DEFAULT_FIG_SIZE) # Use default size
    plt.plot(time_points_ap, neuron2.v_history, label='Membrane Potential (V)')
    plt.axhline(neuron2.v_rest, color='gray', linestyle='--', label='Rest')
    plt.axhline(neuron2.v_threshold, color='red', linestyle='--', label='Threshold')
    plt.axvline(input_time_ap, color='cyan', linestyle=':', label=f'Input (w={input_weight_ap})')
    if spike_occurred_time > 0:
        plt.axvline(spike_occurred_time, color='lime', linestyle='-', alpha=0.7, label='Spike Occurred')
        if hasattr(neuron2, 'tau_ref'): # Check if attribute exists
             plt.axvspan(spike_occurred_time, spike_occurred_time + neuron2.tau_ref,
                         color='yellow', alpha=0.15, label=f'Refractory ({neuron2.tau_ref}ms)')
    plt.title('1.2: Action Potential Generation')
    plt.xlabel('Time (ms)'); plt.ylabel('Membrane Potential (mV)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(neuron2.v_reset - 5, neuron2.v_threshold + 15)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_1_2_action_potential.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig1_2)

    # --- 1.3 Membrane Potential Decay (Leak) ---
    print("  Plotting: Passive Membrane Decay")
    neuron3_params = {**base_params, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    neuron3 = Layered_LIFNeuronWithReversal(**{k: neuron3_params[k] for k in valid_neuron_keys if k in neuron3_params})
    T_decay = 5 * neuron3.tau_m # Simulate for 5 time constants
    time_points_decay = np.arange(0, T_decay, dt)
    initial_v_decay = neuron3.v_rest + 10.0 # Start 10mV above rest

    neuron3.reset()
    neuron3.v = initial_v_decay # Set initial voltage manually
    for t in time_points_decay:
        neuron3.update(dt) # No input, just let it decay

    fig1_3 = plt.figure(figsize=DEFAULT_FIG_SIZE) # Use default size
    plt.plot(time_points_decay, neuron3.v_history, label='Membrane Potential (V)', color='lightblue')
    plt.axhline(neuron3.v_rest, color='gray', linestyle='--', label=f'Resting Potential ({neuron3.v_rest:.1f} mV)')
    plt.plot(0, initial_v_decay, 'ro', markersize=8, label=f'Initial V ({initial_v_decay:.1f} mV)')
    # Theoretical decay curve
    theoretical_decay = neuron3.v_rest + (initial_v_decay - neuron3.v_rest) * np.exp(-time_points_decay / neuron3.tau_m)
    plt.plot(time_points_decay, theoretical_decay, 'w:', alpha=0.7, label=f'Theoretical Decay (τm={neuron3.tau_m})')
    plt.title('1.3: Passive Membrane Decay (Leak)')
    plt.xlabel('Time (ms)'); plt.ylabel('Membrane Potential (mV)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylim(min(neuron3.v_history + [neuron3.v_rest]) - 2, max(neuron3.v_history) + 2)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_1_3_decay.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig1_3)

    # --- 1.4 Adaptation Build-up and Decay ---
    print("  Plotting: Spike-Frequency Adaptation")
    neuron4_params = {**base_params,
                      'adaptation_increment': 1.5, # Stronger adaptation for demo
                      'tau_adaptation': 150.0,     # Longer adaptation tau for demo
                      'v_noise_amp': 0.1, 'i_noise_amp': 0.01} # Keep some noise from original script
    neuron4 = Layered_LIFNeuronWithReversal(**{k: neuron4_params[k] for k in valid_neuron_keys if k in neuron4_params})
    T_adapt = 400.0
    time_points_adapt = np.arange(0, T_adapt, dt)
    stimulus_start = 50.0
    stimulus_end = 250.0
    stimulus_conductance = 0.8 # Constant drive (conductance value)

    neuron4.reset()
    spike_times_adapt = []
    # Apply constant stimulus conductance during the window
    neuron4.external_stim_g = 0 # Start with no external stimulus
    for i, t in enumerate(time_points_adapt):
        if stimulus_start <= t < stimulus_end:
            neuron4.external_stim_g = stimulus_conductance # Apply the conductance
        else:
            neuron4.external_stim_g = 0 # Turn off stimulus outside window

        spiked = neuron4.update(dt)
        if spiked:
            spike_times_adapt.append(t + dt)

    fig1_4 = plt.figure(figsize=DEFAULT_FIG_SIZE) # Use default size
    # Plot membrane potential to see effect of adaptation
    ax_adapt1 = plt.subplot(2, 1, 1)
    ax_adapt1.plot(time_points_adapt, neuron4.v_history, label='Vm', color='cyan', lw=1)
    for spike_t in spike_times_adapt:
        ax_adapt1.axvline(spike_t, color='lime', linestyle=':', alpha=0.6, linewidth=0.8)
    ax_adapt1.axhline(neuron4.v_threshold, color='red', linestyle='--', label='Threshold', lw=1)
    ax_adapt1.axvspan(stimulus_start, stimulus_end, color='grey', alpha=0.2, label='Stimulus Period')
    ax_adapt1.set_ylabel('Potential (mV)')
    adapt_inc = neuron4_params.get('adaptation_increment', '?')
    tau_adapt = neuron4_params.get('tau_adaptation', '?')
    ax_adapt1.set_title(f'1.4: Adaptation Effect (Δw={adapt_inc}, τw={tau_adapt}ms)')
    ax_adapt1.legend(loc='upper right', fontsize='small')
    ax_adapt1.grid(True, alpha=0.3)
    plt.setp(ax_adapt1.get_xticklabels(), visible=False)

    # Plot adaptation variable
    ax_adapt2 = plt.subplot(2, 1, 2, sharex=ax_adapt1)
    ax_adapt2.plot(time_points_adapt, neuron4.adaptation_history, label='Adaptation Current (w)', color='magenta')
    for spike_t in spike_times_adapt:
        ax_adapt2.axvline(spike_t, color='lime', linestyle=':', alpha=0.6, linewidth=0.8)
    ax_adapt2.axvspan(stimulus_start, stimulus_end, color='grey', alpha=0.2)
    ax_adapt2.set_xlabel('Time (ms)'); ax_adapt2.set_ylabel('Adaptation (w)')
    ax_adapt2.legend(loc='upper right', fontsize='small')
    ax_adapt2.grid(True, alpha=0.3)
    plt.text(0.95, 0.9, f'{len(spike_times_adapt)} spikes', ha='right', va='top', transform=ax_adapt2.transAxes, color='lime')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_1_4_adaptation.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig1_4)


    # --- 1.5 Combined Conductance Dynamics ---
    print("  Plotting: Combined Conductance Dynamics")
    neuron_cond_params = {**base_params, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    neuron_cond = Layered_LIFNeuronWithReversal(**{k: neuron_cond_params[k] for k in valid_neuron_keys if k in neuron_cond_params})
    T_cond = 80.0
    time_points_cond = np.arange(0, T_cond, dt)
    inputs = {10.0: 2.0, 20.0: 1.5, 30.0: -1.8, 40.0: -2.2} # time: weight

    neuron_cond.reset()
    for t in time_points_cond:
        if any(abs(t - input_t) < dt / 2 for input_t in inputs):
            weight = inputs[min(inputs.keys(), key=lambda k: abs(k-t))] # Find closest input time's weight
            neuron_cond.receive_spike(weight)
        neuron_cond.update(dt)

    fig1_5, axes_cond = plt.subplots(2, 1, figsize=DEFAULT_FIG_SIZE, sharex=True) # Use default size
    fig1_5.suptitle('1.5: Synaptic Conductance Dynamics')
    ax_g_e, ax_g_i = axes_cond
    ax_g_e.plot(time_points_cond, neuron_cond.g_e_history, label='g_e', color='orange')
    ax_g_i.plot(time_points_cond, neuron_cond.g_i_history, label='g_i', color='lightblue')
    # Add vertical lines for inputs, checking which axis to add to
    input_labels_e = []
    input_labels_i = []
    for t_in, w_in in inputs.items():
        color = 'lime' if w_in > 0 else 'tomato'
        linestyle = ':' if abs(w_in) < 2 else '--'
        label = f'E Input (w={w_in})' if w_in > 0 else f'I Input (w={w_in})'
        if w_in > 0:
             if label not in input_labels_e: # Avoid duplicate labels
                 ax_g_e.axvline(t_in, color=color, linestyle=linestyle, label=label)
                 input_labels_e.append(label)
             else:
                 ax_g_e.axvline(t_in, color=color, linestyle=linestyle)
        else:
             if label not in input_labels_i: # Avoid duplicate labels
                 ax_g_i.axvline(t_in, color=color, linestyle=linestyle, label=label)
                 input_labels_i.append(label)
             else:
                  ax_g_i.axvline(t_in, color=color, linestyle=linestyle)

    tau_e = base_params.get('tau_e', '?')
    tau_i = base_params.get('tau_i', '?')
    ax_g_e.set_title(f'Excitatory Conductance (τe = {tau_e} ms)'); ax_g_e.set_ylabel('g_e'); ax_g_e.legend(fontsize='small')
    ax_g_i.set_title(f'Inhibitory Conductance (τi = {tau_i} ms)'); ax_g_i.set_ylabel('g_i'); ax_g_i.legend(fontsize='small')
    ax_g_i.set_xlabel('Time (ms)')
    ax_g_e.grid(True, alpha=0.3); ax_g_i.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "diag_1_5_conductance.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig1_5)


# ==============================================================================
# SECTION 2: NEURON TRACES UNDER CONSTANT STIMULATION (from neuron_diagnostics.py)
# Using Layered_LIFNeuronWithReversal
# ==============================================================================
print("\n--- Running Section 2: Neuron Traces Under Constant Stimulation ---")

# (Function definitions run_neuron_simulation_trace and plot_neuron_trace remain the same)
def run_neuron_simulation_trace(neuron, duration, dt, stim_conductance=0.0):
    """Runs a simulation for a single neuron with constant stimulation conductance."""
    n_steps = int(duration / dt)
    neuron.reset()
    time_hist = np.arange(n_steps) * dt
    v_hist, g_e_hist, g_i_hist, adapt_hist = [], [], [], []
    spike_times_sim = []

    for i in range(n_steps):
        # Apply constant stimulation via excitatory conductance increase
        neuron.apply_external_stimulus(stim_conductance)

        # Record state *before* update for this step's history
        v_hist.append(neuron.v)
        g_e_hist.append(neuron.g_e)
        g_i_hist.append(neuron.g_i)
        adapt_hist.append(neuron.adaptation)

        # Update neuron state
        spiked = neuron.update(dt)
        if spiked:
            spike_times_sim.append(time_hist[i])
            # Optional: mark spike visually in voltage trace (can clip plot)
            # v_hist[-1] = neuron.v_threshold + 10 # Artificial peak for viz

    return time_hist, v_hist, g_e_hist, g_i_hist, adapt_hist, spike_times_sim

def plot_neuron_trace(fig, gs_row, time, v, g_e, g_i, adaptation, spikes, stim_level_text, neuron_params_dict):
    """Plots the simulation results for one stimulation level using GridSpec."""
    inner_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_row, hspace=0.1, height_ratios=[2, 1, 1, 1])
    ax_v = fig.add_subplot(inner_gs[0]); ax_g = fig.add_subplot(inner_gs[1], sharex=ax_v)
    ax_a = fig.add_subplot(inner_gs[2], sharex=ax_v); ax_spikes = fig.add_subplot(inner_gs[3], sharex=ax_v)
    label_fontsize = 'small'

    # Panel 1: Membrane Potential
    ax_v.plot(time, v, label='Vm (mV)', color='cyan', linewidth=1)
    ax_v.axhline(neuron_params_dict.get('v_threshold', -55), color='red', linestyle='--', label='Thr', alpha=0.7) # Use .get with default
    ax_v.axhline(neuron_params_dict.get('v_rest', -65), color='gray', linestyle=':', label='Rest', alpha=0.7)
    ax_v.axhline(neuron_params_dict.get('e_reversal', 0), color='orange', linestyle=':', label='Erev', alpha=0.7)
    ax_v.axhline(neuron_params_dict.get('i_reversal', -70), color='lightblue', linestyle=':', label='Irev', alpha=0.7)
    ax_v.set_ylabel('Vm (mV)', fontsize=label_fontsize)
    y_min_v = min(neuron_params_dict.get('v_reset', -75) - 5, neuron_params_dict.get('i_reversal', -70) - 5)
    y_max_v = max(neuron_params_dict.get('v_threshold', -55) + 15, neuron_params_dict.get('e_reversal', 0) + 5)
    ax_v.set_ylim(y_min_v, y_max_v)
    ax_v.legend(loc='upper right', fontsize='x-small', ncol=2)
    ax_v.text(0.02, 0.95, stim_level_text, transform=ax_v.transAxes, color='white', fontsize=9, va='top')
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # Panel 2: Conductances
    ax_g.plot(time, g_e, label='g_e', color='orange', linewidth=1)
    ax_g.plot(time, g_i, label='g_i', color='lightblue', linewidth=1)
    max_g = max(np.max(g_e) if len(g_e)>0 else 0, np.max(g_i) if len(g_i)>0 else 0)
    if max_g > 1e-9: ax_g.set_ylim(-0.05 * max_g, max_g * 1.1)
    ax_g.set_ylabel('g', fontsize=label_fontsize); ax_g.legend(loc='upper right', fontsize='x-small')
    plt.setp(ax_g.get_xticklabels(), visible=False)

    # Panel 3: Adaptation Current
    ax_a.plot(time, adaptation, label='Adapt (w)', color='magenta', linewidth=1)
    max_a = np.max(adaptation) if len(adaptation)>0 else 0
    if max_a > 1e-9: ax_a.set_ylim(-0.05 * max_a, max_a * 1.1)
    ax_a.set_ylabel('w', fontsize=label_fontsize); ax_a.legend(loc='upper right', fontsize='x-small')
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # Panel 4: Spikes
    if spikes: ax_spikes.eventplot(spikes, orientation='horizontal', colors='yellow', lineoffsets=0.5, linelengths=0.8)
    ax_spikes.set_yticks([]); ax_spikes.set_ylabel('Spikes', fontsize=label_fontsize)
    ax_spikes.set_xlabel('Time (ms)', fontsize=label_fontsize) # Only x-label on bottom plot

    for sub_ax in [ax_v, ax_g, ax_a, ax_spikes]:
        sub_ax.tick_params(axis='y', labelsize='x-small'); sub_ax.tick_params(axis='x', labelsize='small')
        for spine in sub_ax.spines.values(): spine.set_color('white')
        sub_ax.grid(True, which='major', axis='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)


def run_constant_stim_traces(neuron_params, duration=500, dt=DEFAULT_DT, stim_conductances=None, save_dir=SAVE_DIR): # Removed default, added save_dir
    if neuron_params is None:
        print("Error: neuron_params not provided to run_constant_stim_traces")
        return
    if stim_conductances is None:
        stim_conductances = [0.0, 0.5, 1.0] # Example conductance levels

    # Use the passed neuron_params directly
    trace_neuron_params = neuron_params.copy()
    # Filter params just in case before initializing
    neuron_init_sig = inspect.signature(Layered_LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_sig if k != 'self'}
    init_params = {k: trace_neuron_params[k] for k in valid_neuron_keys if k in trace_neuron_params}

    neuron = Layered_LIFNeuronWithReversal(**init_params)
    num_traces = len(stim_conductances)
    results = []
    print("  Running constant stimulation simulations...")
    for stim_g in stim_conductances:
        print(f"    Stimulation conductance: {stim_g:.2f}")
        time_hist, v_hist, g_e_hist, g_i_hist, adapt_hist, spikes = run_neuron_simulation_trace(
            neuron, duration, dt, stim_conductance=stim_g
        )
        results.append({
            'time': time_hist, 'v': v_hist, 'g_e': g_e_hist, 'g_i': g_i_hist,
            'adaptation': adapt_hist, 'spikes': spikes, 'stim_level': stim_g
        })

    print("  Generating trace plot...")
    # Set figure size directly
    fig_traces = plt.figure(figsize=DEFAULT_FIG_SIZE, facecolor='#1a1a1a') # Use default size
    # Adjust GridSpec layout based on fixed figure size and number of traces
    # Ensure enough height for each trace group within the fixed figure height
    main_gs = GridSpec(num_traces, 1, figure=fig_traces, hspace=0.35) # Increased hspace slightly

    fig_traces.suptitle('2: Neuron Traces Under Constant Stimulus Conductance', color='white', fontsize=14)
    for i, res in enumerate(results):
        stim_text = f"Stim g = {res['stim_level']:.2f}"
        # Pass the params used for this specific simulation run
        plot_neuron_trace(
            fig_traces, main_gs[i], res['time'], res['v'], res['g_e'], res['g_i'],
            res['adaptation'], res['spikes'], stim_text, init_params
        )
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    save_path = os.path.join(save_dir, "diag_2_traces.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig_traces)


# ==============================================================================
# SECTION 3: STIMULUS-RESPONSE CURVES (from stim_response_curves.py)
# Using Layered_LIFNeuronWithReversal and LayeredNeuronalNetwork
# ==============================================================================
print("\n--- Running Section 3: Stimulus-Response Curves ---")

# (Function definitions analyze_and_plot_isolated_neuron_frequency, run_network_stim_response_sim,
# setup_layered_network, analyze_and_plot_network_average remain the same as previous version, just add saving)
# --- Function to run and plot for an ISOLATED neuron (FREQUENCY ONLY) ---
def analyze_and_plot_isolated_neuron_frequency(neuron_config, sim_duration_curve, dt,
                                               frequency_range, fixed_magnitude,
                                               stim_pulse_duration, save_dir=SAVE_DIR):
    """Creates one isolated neuron, runs frequency stim-response analysis, and plots results."""
    print("  Analyzing Single ISOLATED Neuron (Frequency Response)...")
    # Filter params for constructor
    neuron_init_sig = inspect.signature(Layered_LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_sig if k != 'self'}
    init_params = {k: neuron_config[k] for k in valid_neuron_keys if k in neuron_config}
    init_params['is_inhibitory'] = False # Ensure excitatory for this test

    neuron = Layered_LIFNeuronWithReversal(**init_params)
    n_steps = int(sim_duration_curve / dt)
    sim_duration_sec = sim_duration_curve / 1000.0

    frequency_responses_single = []
    for freq in tqdm(frequency_range, desc="    Isolated Freq Steps", leave=False):
        neuron.reset()
        stim_strength = fixed_magnitude
        stim_interval_ms = 1000.0 / freq if freq > 0 else float('inf')
        stim_interval_steps = int(stim_interval_ms / dt) if stim_interval_ms != float('inf') else None
        stim_end_time = -1.0
        neuron.external_stim_g = 0 # Ensure starts at 0

        for step in range(n_steps):
            current_time = step * dt
            # Turn off stimulus if pulse duration ended
            if current_time >= stim_end_time:
                neuron.apply_external_stimulus(0.0) # Turn off
            # Apply new stimulus pulse if interval is met
            if stim_interval_steps and (step % stim_interval_steps == 0):
                 neuron.apply_external_stimulus(stim_strength) # Turn on
                 stim_end_time = current_time + stim_pulse_duration
            neuron.update(dt) # Update neuron state

        # *** CORRECTED: Use len(neuron.spike_times) instead of neuron.spike_count ***
        spike_count = len(neuron.spike_times)
        rate = spike_count / sim_duration_sec if sim_duration_sec > 0 else 0
        frequency_responses_single.append(rate)

    # Plotting
    fig_single, ax = plt.subplots(1, 1, figsize=DEFAULT_FIG_SIZE, facecolor='#1a1a1a') # Use default size
    ax.plot(frequency_range, frequency_responses_single, color='#1f77b4', linestyle='-')
    ax.set_xlabel('Stimulus Frequency (Hz)'); ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'3.1: Isolated Neuron Freq Response (Stim Mag: {fixed_magnitude}, Pulse: {stim_pulse_duration}ms)')
    ax.grid(True, alpha=0.3)
    max_rate_single = max(frequency_responses_single) if frequency_responses_single else 0
    ax.axhline(y=max_rate_single, color='grey', linestyle=':', label=f'Max Rate: {max_rate_single:.1f} Hz')
    ax.legend(fontsize='small')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_3_1_isolated_freq_response.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig_single)

# --- Simulation Function (For NETWORK simulations - Stimulates ALL Layer 1) ---
def run_network_stim_response_sim(network, duration, dt, stimulus_params, layer_indices, analysis_target_neurons):
    """Runs NETWORK simulation, applies stimulus to ALL Layer 1 neurons, returns firing rates."""
    network.reset_all()
    n_steps = int(duration / dt)
    stim_type = stimulus_params['type']
    pulse_duration_ms = stimulus_params.get('pulse_duration_ms', dt)

    if stim_type != 'frequency':
         raise ValueError("This version only supports 'frequency' stimulus type for network sim")

    stim_strength = stimulus_params['fixed_magnitude']
    freq_hz = stimulus_params['value']
    stim_interval_ms = 1000.0 / freq_hz if freq_hz > 0 else float('inf')
    stim_interval_steps = int(stim_interval_ms / dt) if stim_interval_ms != float('inf') else None
    ongoing_stimulations = {} # {neuron_idx: end_time}

    # Get indices for ALL neurons in Layer 1
    if not layer_indices: return {idx: 0 for idx in analysis_target_neurons} # No layers defined
    l1_start, l1_end = layer_indices[0]
    first_layer_indices = list(range(l1_start, l1_end))
    if not first_layer_indices: return {idx: 0 for idx in analysis_target_neurons} # Layer 1 empty

    # --- Simulation Loop ---
    for step in range(n_steps):
        current_time = step * dt
        # Update ongoing stimulations
        expired_stims = [idx for idx, end_t in ongoing_stimulations.items() if current_time >= end_t]
        for idx in expired_stims:
            if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                network.neurons[idx].apply_external_stimulus(0.0)
            if idx in ongoing_stimulations: # Check if key exists before deleting
                del ongoing_stimulations[idx]
        # Apply stimulus for active ones
        for idx in ongoing_stimulations:
             if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                 network.neurons[idx].apply_external_stimulus(stim_strength)

        # Apply new stimulation pulses to ALL Layer 1 neurons
        if stim_interval_steps and (step % stim_interval_steps == 0):
            stim_end_time = current_time + pulse_duration_ms
            for idx in first_layer_indices:
                if idx not in ongoing_stimulations:
                    ongoing_stimulations[idx] = stim_end_time
                    if 0 <= idx < len(network.neurons) and network.neurons[idx]:
                        network.neurons[idx].apply_external_stimulus(stim_strength)
        # Update network
        network.update_network(dt)
    # Turn off any remaining stims
    for idx in list(ongoing_stimulations.keys()):
        if 0 <= idx < len(network.neurons) and network.neurons[idx]:
             network.neurons[idx].apply_external_stimulus(0.0)

    # Calculate firing rates for target neurons
    firing_rates = {}
    sim_duration_sec = duration / 1000.0
    for idx in analysis_target_neurons:
        if idx < len(network.neurons) and network.neurons[idx]:
            # *** CORRECTED: Use len(neuron.spike_times) instead of neuron.spike_count ***
            spike_count = len(network.neurons[idx].spike_times)
            firing_rates[idx] = spike_count / sim_duration_sec if sim_duration_sec > 0 else 0
        else: firing_rates[idx] = 0
    return firing_rates

# --- Setup Network Function (Adapted for LayeredNeuronalNetwork) ---
def setup_layered_network(n_layers_list, inhibitory_fraction, connection_probs,
                          neuron_params, weight_min, weight_max, base_transmission_delay,
                          pruning_threshold=0.0):
    """Creates and configures the LayeredNeuronalNetwork."""
    print("  Setting up Layered Network...")
    num_layers = len(n_layers_list)
    total_neurons = sum(n_layers_list)

    # Extract connection probabilities
    conn_keys = ['exc_recurrent', 'inh_recurrent', 'feedforward_1', 'feedforward_2',
                 'feedback_1', 'feedback_2', 'long_feedforward', 'long_feedback']
    probs = {k: connection_probs.get(k, 0.0) for k in conn_keys} # Get probs, default 0

    # Initialize LayeredNeuronalNetwork
    # Filter neuron_params for network constructor (might accept different params)
    network_init_sig = inspect.signature(LayeredNeuronalNetwork).parameters
    valid_network_keys = {k for k in network_init_sig if k != 'self'}
    network_init_params = {k: neuron_params[k] for k in valid_network_keys if k in neuron_params}
    # Pass n_neurons and inhibitory_fraction explicitly
    network_init_params['n_neurons'] = total_neurons
    network_init_params['inhibitory_fraction'] = inhibitory_fraction

    network = LayeredNeuronalNetwork(**network_init_params)
    network.graph.graph['num_layers'] = num_layers # Store layer count

    pos = {} # Neuron positions
    x_coords = np.linspace(0.1, 0.9, num_layers) # X-coord per layer
    horizontal_spread = 0.04; vertical_spread = total_neurons / 20.0
    layer_indices = [] # Store (start_idx, end_idx) for each layer
    start_idx = 0

    # Filter neuron_params for Layered_LIFNeuronWithReversal constructor
    neuron_init_sig = inspect.signature(Layered_LIFNeuronWithReversal).parameters
    valid_neuron_keys = {k for k in neuron_init_sig if k != 'self' and k != 'is_inhibitory'}
    filtered_neuron_params = {k: neuron_params[k] for k in valid_neuron_keys if k in neuron_params}

    # Create neurons layer by layer
    for layer_num, n_layer in enumerate(n_layers_list, 1):
        x_layer = x_coords[layer_num-1]
        end_idx = start_idx + n_layer
        layer_indices.append((start_idx, end_idx))
        for node_idx in range(start_idx, end_idx):
             is_inhib = random.random() < inhibitory_fraction # Use passed inhibitory_fraction
             neuron = Layered_LIFNeuronWithReversal(is_inhibitory=is_inhib, **filtered_neuron_params)
             neuron.layer = layer_num
             node_pos = (x_layer + random.uniform(-horizontal_spread, horizontal_spread),
                         random.uniform(0.5 - vertical_spread, 0.5 + vertical_spread))
             pos[node_idx] = node_pos
             network.add_neuron(neuron, node_idx, node_pos, layer_num) # Use network's add method
        start_idx = end_idx

    # Calculate max distance for delay normalization
    all_x = [p[0] for p in pos.values()]; all_y = [p[1] for p in pos.values()]
    max_possible_dist = np.sqrt((max(all_x)-min(all_x))**2 + (max(all_y)-min(all_y))**2) if len(pos)>1 else 1.0
    if max_possible_dist < 1e-6: max_possible_dist = 1.0
    min_delay = 0.1

    # Add connections
    connection_count = 0
    for i in range(total_neurons):
        if i not in network.graph.nodes or i not in pos: continue # Skip if node/pos missing
        is_source_inhib = network.graph.nodes[i].get('is_inhibitory', False)
        layer_i = network.graph.nodes[i].get('layer', -1)
        pos_i = pos[i]
        for j in range(total_neurons):
            if i == j or j not in network.graph.nodes or j not in pos: continue # Skip self or missing node/pos
            layer_j = network.graph.nodes[j].get('layer', -1)
            pos_j = pos[j]
            prob = 0.0; weight = 0.0; connect = False

            # Determine connection probability based on type and layers
            if is_source_inhib: # Source Inhibitory
                # Inhibitory connections are typically local or follow specific patterns
                # Example: Recurrent inhibition only
                if layer_i == layer_j: prob = probs['inh_recurrent']
                # Add other inhibitory patterns if needed (e.g., layer_diff == 1)
            else: # Source Excitatory
                if layer_i != -1 and layer_j != -1: # Check if layers are defined
                    layer_diff = layer_j - layer_i
                    if layer_diff == 0: prob = probs['exc_recurrent']
                    elif layer_diff == 1: prob = probs['feedforward_1']
                    elif layer_diff == 2: prob = probs['feedforward_2']
                    elif layer_diff == -1: prob = probs['feedback_1']
                    elif layer_diff == -2: prob = probs['feedback_2']
                    elif layer_diff > 2: prob = probs['long_feedforward']
                    elif layer_diff < -2: prob = probs['long_feedback']
                else: # Default if layer info is missing
                     if random.random() < 0.05: # Low default probability
                         prob = 1.0 # Force connection if random check passes

            # Connect based on probability
            if random.random() < prob:
                 weight_sign = -1.0 if is_source_inhib else 1.0
                 weight = weight_sign * random.uniform(weight_min, weight_max)
                 connect = True

            # Calculate delay and add connection
            if connect:
                distance = np.sqrt((pos_i[0]-pos_j[0])**2 + (pos_i[1]-pos_j[1])**2)
                # Simplified linear delay scaling
                delay = base_transmission_delay * (0.5 + 0.5 * (distance / max_possible_dist))
                delay = max(min_delay, delay)
                network.add_connection(i, j, weight, delay=delay) # Use network's add method
                connection_count += 1

    # Prune weak connections if requested
    if pruning_threshold > 0:
        network.prune_weak_connections(threshold=pruning_threshold) # Use network's prune method

    print(f"  Layered Network setup complete. {network.graph.number_of_edges()} connections.")
    return network, pos, layer_indices


# --- Function to Analyze and Plot Network Average Response ---
def analyze_and_plot_network_average(network_config, sim_duration_curve, dt,
                                    frequency_range, fixed_magnitude, stim_pulse_duration,
                                    n_neurons_to_average=10, save_dir=SAVE_DIR):
    """Sets up the layered network, runs frequency response analysis for avg of N neurons, plots."""
    print(f"  Analyzing Network Average (Frequency Response) for {n_neurons_to_average} Neurons...")
    # Setup network using the provided config
    network, pos, layer_indices = setup_layered_network(**network_config)

    if network.n_neurons == 0:
         print("  Error: Network has no neurons. Skipping average analysis.")
         return

    # Select random neurons for averaging (ensure they exist)
    available_indices = [i for i, n in enumerate(network.neurons) if n is not None]
    if len(available_indices) < n_neurons_to_average:
        print(f"  Warning: Network size ({len(available_indices)}) < {n_neurons_to_average}. Analyzing all available.")
        sampled_indices = available_indices
        n_neurons_to_average = len(available_indices)
    else:
        sampled_indices = random.sample(available_indices, n_neurons_to_average)

    if not sampled_indices:
         print("  Error: No neurons available for average analysis.")
         return

    print(f"    Selected neurons for averaging: {sampled_indices}")

    # Run frequency sweep for the network
    frequency_responses_avg = {idx: [] for idx in sampled_indices}
    for freq in tqdm(frequency_range, desc="    Network Avg Freq Steps", leave=False):
        stim_params = {'type': 'frequency', 'value': freq,
                       'fixed_magnitude': fixed_magnitude,
                       'pulse_duration_ms': stim_pulse_duration}
        # Run sim and get firing rates for the sampled neurons
        firing_rates = run_network_stim_response_sim(network, sim_duration_curve, dt,
                                                   stim_params, layer_indices, sampled_indices)
        for idx in sampled_indices:
            frequency_responses_avg[idx].append(firing_rates.get(idx, 0))

    # Calculate average response
    avg_frequency_response = np.mean(list(frequency_responses_avg.values()), axis=0) if sampled_indices else np.zeros_like(frequency_range)

    # Plotting
    fig_avg, ax_avg = plt.subplots(1, 1, figsize=DEFAULT_FIG_SIZE, facecolor='#1a1a1a') # Use default size
    # Plot individual neuron traces lightly
    for idx in sampled_indices:
          ax_avg.plot(frequency_range, frequency_responses_avg[idx], '--', color='grey', alpha=0.4, lw=0.8)
    # Plot average response boldly
    ax_avg.plot(frequency_range, avg_frequency_response, color='#1f77b4', label=f'Avg Response ({n_neurons_to_average} neurons)', lw=2)
    ax_avg.set_xlabel('Stimulus Frequency (Hz)'); ax_avg.set_ylabel('Average Firing Rate (Hz)')
    ax_avg.set_title('3.2: Network Avg Frequency Response (Stimulating L1)')
    ax_avg.grid(True, alpha=0.3)
    max_rate_avg = max(avg_frequency_response) if avg_frequency_response.size > 0 else 0
    ax_avg.axhline(y=max_rate_avg, color='grey', linestyle=':', label=f'Max Avg Rate: {max_rate_avg:.1f} Hz')
    ax_avg.legend(fontsize='small')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "diag_3_2_network_avg_freq_response.png")
    plt.savefig(save_path)
    print(f"    Saved figure to {save_path}")
    plt.show()
    plt.close(fig_avg)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    print("Starting Consolidated Neuron Analysis Script")

    # --- Configure Parameters ---
    # Simulation
    sim_dt = 0.1
    sim_duration_short = 500 # Increased duration for response curves
    sim_duration_long = 500.0 # For trace plots

    # --- Neuron Parameters from layered_network_experiment.py ---
    main_neuron_params = {
        'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
        'tau_m': 5, 'tau_ref': 1.5, 'tau_e': 3.0, 'tau_i': 7.0,
        'e_reversal': 0.0, 'i_reversal': -70.0,
        'v_noise_amp': 0.32, 'i_noise_amp': 0.04,
        'adaptation_increment': 0.5, 'tau_adaptation': 100.0,
        # 'weight_scale': 0.1 # This is used in network setup, not neuron directly
    }
    print(f"Using Neuron Params from layered_network_experiment.py: {main_neuron_params}")

    # --- Network Parameters from layered_network_experiment.py ---
    layers_config = [196, 147, 116, 90, 70, 9] # From layers_config
    inhib_frac = 0.45                           # From inhib_frac
    conn_probs = {                              # From conn_probs
        'exc_recurrent': 0.05, 'inh_recurrent': 0.1,
        'feedforward_1': 0.20, 'feedforward_2': 0.05,
        'feedback_1': 0.05, 'feedback_2': 0.01,
        'long_feedforward': 0.005, 'long_feedback': 0.002
    }
    weight_config = {'min': 0.004, 'max': 0.02} # From weight_config
    prune_thresh = 0.00                         # From prune_thresh
    base_delay = 0.1                            # From base_delay

    # Create the network_config dictionary for Section 3.2
    network_config = {
        'n_layers_list': layers_config,
        'inhibitory_fraction': inhib_frac,
        'connection_probs': conn_probs,
        'neuron_params': main_neuron_params, # Use the same neuron params
        'weight_min': weight_config['min'],
        'weight_max': weight_config['max'],
        'base_transmission_delay': base_delay,
        'pruning_threshold': prune_thresh
    }
    print(f"Using Network Params from layered_network_experiment.py: {network_config}")


    # --- Other Analysis Parameters ---
    # Stimulus Response Curve Specific Params
    stim_freq_range = np.linspace(1, 300, 300) # Freq range (Hz), number of steps
    stim_fixed_magnitude = 0.7 # Conductance magnitude for freq response
    stim_pulse_duration_ms = 5.0 # Duration of each stimulus pulse
    num_neurons_to_average = 30 # Number of neurons for network average plot

    # Constant Stimulation Trace Params (for Section 2)
    stim_conductances_for_traces = [0.0, 0.04, 1.2] # Conductance levels (3 levels)


    # --- Run Analysis Sections ---
    # Section 1: Single Neuron Behaviors
    run_single_neuron_extended_diagnostics(neuron_params=main_neuron_params, dt=sim_dt, save_dir=SAVE_DIR)

    # Section 2: Constant Stimulation Traces
    run_constant_stim_traces(neuron_params=main_neuron_params, duration=sim_duration_long, dt=sim_dt, stim_conductances=stim_conductances_for_traces, save_dir=SAVE_DIR)

    # Section 3: Stimulus-Response Curves
    # 3.1: Isolated Neuron Frequency Response
    analyze_and_plot_isolated_neuron_frequency(
         neuron_config=main_neuron_params, # Pass the correct config
         sim_duration_curve=sim_duration_short, dt=sim_dt,
         frequency_range=stim_freq_range, fixed_magnitude=stim_fixed_magnitude,
         stim_pulse_duration=stim_pulse_duration_ms, save_dir=SAVE_DIR)

    # 3.2: Network Average Frequency Response
    analyze_and_plot_network_average(
         network_config=network_config, # Pass the whole network config dict
         sim_duration_curve=sim_duration_short, dt=sim_dt,
         frequency_range=stim_freq_range, fixed_magnitude=stim_fixed_magnitude,
         stim_pulse_duration=stim_pulse_duration_ms,
         n_neurons_to_average=num_neurons_to_average, save_dir=SAVE_DIR)

    print(f"\n--- Consolidated Analysis Script Finished ---")
    print(f"Plots saved to directory: {SAVE_DIR}")