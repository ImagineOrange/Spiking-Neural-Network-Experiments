# neuron_diagnostics.py

import numpy as np
import matplotlib.pyplot as plt
# Assuming the file is in the same directory or accessible path
from lif_neuron_with_reversal import LIFNeuronWithReversal
# Import GridSpec and GridSpecFromSubplotSpec for better layout control
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Apply Matplotlib dark style
plt.style.use('dark_background')

def run_neuron_simulation(neuron, duration, dt, stim_current=0.0):
    """Runs a simulation for a single neuron with constant stimulation."""
    n_steps = int(duration / dt)
    neuron.reset() # Ensure neuron starts from a clean state

    # Pre-allocate arrays for efficiency
    time_hist = np.arange(n_steps) * dt
    v_hist = np.zeros(n_steps)
    g_e_hist = np.zeros(n_steps)
    g_i_hist = np.zeros(n_steps)
    adaptation_hist = np.zeros(n_steps)
    spike_times_sim = []

    for i in range(n_steps):
        # Apply constant stimulation via excitatory conductance increase
        if stim_current > 0:
             neuron.stimulate(stim_current * dt)

        # Record state *before* update
        v_hist[i] = neuron.v
        g_e_hist[i] = neuron.g_e
        g_i_hist[i] = neuron.g_i
        adaptation_hist[i] = neuron.adaptation

        # Update neuron state
        spiked = neuron.update(dt)
        if spiked:
            spike_times_sim.append(time_hist[i])
            v_hist[i] = 0 # Represent spike peak for visualization

    return time_hist, v_hist, g_e_hist, g_i_hist, adaptation_hist, spike_times_sim

def plot_neuron_trace(fig, gs_row, time, v, g_e, g_i, adaptation, spikes, stim_level_text, neuron_params):
    """Plots the simulation results for one stimulation level using GridSpec."""
    # Keep height ratios giving Vm slightly more space
    inner_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_row,
                                       hspace=0.1, height_ratios=[2, 1, 1, 1]) # Keep internal ratios

    ax_v = fig.add_subplot(inner_gs[0])
    ax_g = fig.add_subplot(inner_gs[1], sharex=ax_v)
    ax_a = fig.add_subplot(inner_gs[2], sharex=ax_v)
    ax_spikes = fig.add_subplot(inner_gs[3], sharex=ax_v)

    label_fontsize = 'small'

    # Panel 1: Membrane Potential
    ax_v.plot(time, v, label='Vm (mV)', color='cyan', linewidth=1)
    ax_v.axhline(neuron_params['v_threshold'], color='red', linestyle='--', label='Threshold', alpha=0.7)
    ax_v.axhline(neuron_params['v_rest'], color='gray', linestyle=':', label='Rest', alpha=0.7)
    ax_v.axhline(neuron_params['e_reversal'], color='orange', linestyle=':', label='E_rev', alpha=0.7)
    ax_v.axhline(neuron_params['i_reversal'], color='lightblue', linestyle=':', label='I_rev', alpha=0.7)
    ax_v.set_ylabel('Potential (mV)', fontsize=label_fontsize)
    y_min_v = min(neuron_params['v_reset'] - 10, neuron_params['i_reversal'] - 5)
    y_max_v = max(neuron_params['v_threshold'] + 30, neuron_params['e_reversal'] + 5)
    ax_v.set_ylim(y_min_v, y_max_v)
    ax_v.legend(loc='upper right', fontsize='x-small')
    # Remove individual title, add stimulation level as text annotation instead
    # ax_v.set_title(title, color='white', fontsize=10, loc='left')
    ax_v.text(0.02, 0.95, stim_level_text, transform=ax_v.transAxes,
              color='white', fontsize=9, verticalalignment='top')
    plt.setp(ax_v.get_xticklabels(), visible=False)

    # Panel 2: Conductances
    ax_g.plot(time, g_e, label='g_e', color='orange', linewidth=1)
    ax_g.plot(time, g_i, label='g_i', color='lightblue', linewidth=1)
    max_g = max(np.max(g_e) if len(g_e)>0 else 0, np.max(g_i) if len(g_i)>0 else 0)
    if max_g > 1e-9:
        ax_g.set_ylim(-0.05 * max_g, max_g * 1.1)
    ax_g.legend(loc='upper right', fontsize='x-small')
    plt.setp(ax_g.get_xticklabels(), visible=False)

    # Panel 3: Adaptation Current
    ax_a.plot(time, adaptation, label='Adaptation', color='magenta', linewidth=1)
    max_a = np.max(adaptation) if len(adaptation)>0 else 0
    if max_a > 1e-9:
         ax_a.set_ylim(-0.05 * max_a, max_a * 1.1)
    ax_a.legend(loc='upper right', fontsize='x-small')
    plt.setp(ax_a.get_xticklabels(), visible=False)

    # Panel 4: Spikes
    if spikes:
        ax_spikes.eventplot(spikes, orientation='horizontal', colors='yellow', lineoffsets=0.5, linelengths=0.8)
    ax_spikes.set_yticks([])

    # Apply styling consistent with dark_background theme
    for sub_ax in [ax_v, ax_g, ax_a, ax_spikes]:
        sub_ax.tick_params(axis='y', labelsize='x-small')
        sub_ax.tick_params(axis='x', labelsize='small')
        for spine in sub_ax.spines.values():
            spine.set_color('white')
        sub_ax.grid(True, which='major', axis='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

# --- Simulation Parameters ---
duration = 750
dt = 0.1
stimulation_levels = [0.0, 0.05, 0.08]
num_traces = len(stimulation_levels)

# --- Neuron Parameters ---
neuron_params = {
    'v_rest': -65.0, 'v_threshold': -55.0, 'v_reset': -75.0,
    'tau_m': 10.0, 'tau_ref': 2.0, 'tau_e': 3.0, 'tau_i': 7.0,
    'e_reversal': 0.0, 'i_reversal': -80.0,
    'v_noise_amp': 0.2,
    'i_noise_amp': 0.01,
    'adaptation_increment': 0.5,
    'tau_adaptation': 100.0
}

# Create neuron instance
neuron = LIFNeuronWithReversal(**neuron_params)

# --- Run Simulations ---
results = []
print("Running simulations...")
for stim in stimulation_levels:
    print(f"  Stimulation level: {stim}")
    time_hist, v_hist, g_e_hist, g_i_hist, adapt_hist, spikes = run_neuron_simulation(
        neuron, duration, dt, stim_current=stim
    )
    results.append({
        'time': time_hist, 'v': v_hist, 'g_e': g_e_hist, 'g_i': g_i_hist,
        'adaptation': adapt_hist, 'spikes': spikes, 'stim_level': stim
    })

# --- Plotting ---
print("Generating plot...")
# *** KEEP Original Figure Size ***
fig = plt.figure(figsize=(13, 8), facecolor='#1a1a1a')

# *** REDUCE HSPACE Drastically between trace groups ***
# This forces the groups closer together, giving more height to each plot
main_gs = GridSpec(num_traces, 1, figure=fig, hspace=0.2) # Significantly reduced from 0.9


for i, res in enumerate(results):
    # Pass stimulation level text instead of title
    stim_text = f" "
    plot_neuron_trace(
        fig, main_gs[i], res['time'], res['v'], res['g_e'], res['g_i'],
        res['adaptation'], res['spikes'], stim_text, neuron_params
    )

# Save and show
output_filename = 'neuron_diagnostics_reduced_spacing.png'
plt.savefig(output_filename, dpi=150, facecolor=fig.get_facecolor())
print(f"Plot saved as {output_filename}")
plt.show()