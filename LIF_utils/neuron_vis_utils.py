import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Default: do NOT set dark style globally - let functions handle it based on darkstyle parameter


def plot_reversal_effects(network, neuron_data, selected_neuron, stim_times=None, dt=0.1,
                           save_path="reversal_effects.png", darkstyle=True):
    """
    Create a detailed visualization showing how reversal potentials affect synaptic currents
    for a selected neuron. This function highlights the reversal potential mechanism.

    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network containing neuron models
    neuron_data : dict
        Dictionary of tracked neuron data
    selected_neuron : int
        Index of the neuron to visualize
    stim_times : list or None
        List of stimulation time points (in ms)
    dt : float
        Time step size in ms
    save_path : str
        Path to save the visualization
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)
    """
    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        zero_line_color = 'white'
        rest_line_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        zero_line_color = 'black'
        rest_line_color = 'gray'
    if selected_neuron not in neuron_data:
        print(f"Error: Neuron {selected_neuron} not found in neuron_data")
        return None
        
    # Get neuron data
    data = neuron_data[selected_neuron]
    v_history = np.array(data['v_history'])
    g_e_history = np.array(data['g_e_history'])
    g_i_history = np.array(data['g_i_history'])
    i_syn_history = np.array(data['i_syn_history'])
    spike_times = data['spike_times']
    
    # Get neuron parameters
    neuron = network.neurons[selected_neuron]
    v_threshold = neuron.v_threshold
    v_rest = neuron.v_rest
    e_reversal = neuron.e_reversal
    i_reversal = neuron.i_reversal
    is_inhibitory = neuron.is_inhibitory
    
    # Create time axis
    n_steps = len(v_history)
    time = np.arange(n_steps) * dt
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(14, 10), dpi=150, facecolor=bg_color)
    gs = GridSpec(4, 1, height_ratios=[3, 2, 2, 2], hspace=0.3, figure=fig)
    
    # Define colors
    v_color = '#00a8ff'           # Bright blue for membrane potential
    g_e_color = '#ff9f43'         # Orange for excitatory conductance
    g_i_color = '#0abde3'         # Cyan for inhibitory conductance
    i_e_color = '#ff9f43'         # Orange for excitatory current
    i_i_color = '#0abde3'         # Cyan for inhibitory current
    i_syn_color = '#ff6b6b'       # Red for total synaptic current
    ap_color = '#FFD700'          # Gold for action potentials
    
    neuron_type = "Inhibitory" if is_inhibitory else "Excitatory"
    fig.suptitle(f"Reversal Potential Effects - Neuron {selected_neuron} ({neuron_type})",
                fontsize=16, color=text_color)
    
    # PANEL 1: Membrane Potential
    ax1 = fig.add_subplot(gs[0])
    
    # Plot membrane potential
    ax1.plot(time, v_history, color=v_color, linewidth=1.5, label="Membrane Potential")
    
    # Plot threshold and rest potentials
    ax1.axhline(y=v_threshold, color='purple', linestyle='--', 
              linewidth=1.0, alpha=0.7, label="Threshold")
    ax1.axhline(y=v_rest, color=rest_line_color, linestyle=':',
              linewidth=1.0, alpha=0.4, label="Rest")
    
    # Plot reversal potentials
    ax1.axhline(y=e_reversal, color=g_e_color, linestyle=':', 
              linewidth=1.0, alpha=0.7, label="E-reversal")
    ax1.axhline(y=i_reversal, color=g_i_color, linestyle=':', 
              linewidth=1.0, alpha=0.7, label="I-reversal")
    
    # Mark spikes
    for spike_time in spike_times:
        ax1.axvline(x=spike_time, color=ap_color, linewidth=1.0, 
                  alpha=0.8, zorder=5)
    
    # Mark stimulation times if provided
    if stim_times:
        for stim_time in stim_times:
            ax1.axvline(x=stim_time, color='green', linestyle='-.',
                      linewidth=0.8, alpha=0.15)
    
    ax1.set_ylabel("Membrane Potential (mV)", color=text_color)
    ax1.set_title("Membrane Potential and Reversal Potentials", color=text_color)
    ax1.legend(loc='upper right', framealpha=0.7)
    
    # PANEL 2: Synaptic Conductances
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Plot synaptic conductances
    scale_factor = 20  # Adjust based on data
    g_e_scaled = g_e_history * scale_factor
    g_i_scaled = g_i_history * scale_factor
    
    ax2.plot(time, g_e_scaled, color=g_e_color, linewidth=1.5,
            label="Excitatory Conductance (g_e)")
    ax2.plot(time, g_i_scaled, color=g_i_color, linewidth=1.5,
            label="Inhibitory Conductance (g_i)")
    ax2.set_ylabel("Conductance (scaled)", color=text_color)
    ax2.set_title("Synaptic Conductances", color=text_color)
    ax2.legend(loc='upper right', framealpha=0.7)
    
    # PANEL 3: Synaptic Currents
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Calculate excitatory and inhibitory currents
    i_e = g_e_history * (e_reversal - v_history)
    i_i = g_i_history * (i_reversal - v_history)
    
    # Scale for visualization
    i_e_scaled = i_e * 0.1
    i_i_scaled = i_i * 0.1
    i_total_scaled = i_syn_history * 0.1
    
    # Plot currents
    ax3.plot(time, i_e_scaled, color=i_e_color, linewidth=1.5, 
            label="Excitatory Current (scaled)")
    ax3.plot(time, i_i_scaled, color=i_i_color, linewidth=1.5, 
            label="Inhibitory Current (scaled)")
    ax3.plot(time, i_total_scaled, color=i_syn_color, linewidth=1.5, 
            label="Total Current (scaled)")
    
    ax3.axhline(y=0, color=zero_line_color, linestyle='-',
              linewidth=0.5, alpha=0.5)
    
    ax3.set_ylabel("Current (scaled)", color=text_color)
    ax3.set_title("Synaptic Currents", color=text_color)
    ax3.legend(loc='upper right', framealpha=0.7)
    
    # PANEL 4: Current-Voltage Relationship
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    scale = .5
    # Calculate true driving forces without absolute value
    e_driving_force = e_reversal - v_history
    i_driving_force = i_reversal - v_history

    # Scale for visualization but preserve sign
    e_driving_scaled = e_driving_force * scale
    i_driving_scaled = i_driving_force * scale

    # Plot the correct driving forces with their true signs
    ax4.plot(time, e_driving_scaled, color=i_e_color, linewidth=1.5, 
            label="E-channel Driving Force (scaled)")
    ax4.plot(time, i_driving_scaled, color=i_i_color, linewidth=1.5, 
            label="I-channel Driving Force (scaled)")
    
    ax4.axhline(y=0, color=zero_line_color, linestyle='-',
              linewidth=0.5, alpha=0.5)
    
    ax4.set_xlabel("Time (ms)", color=text_color)
    ax4.set_ylabel("Driving Force (scaled)", color=text_color)
    ax4.set_title("Synaptic Driving Forces", color=text_color)
    ax4.legend(loc='upper right', framealpha=0.7)

    # Apply theme styling to all panels based on darkstyle
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)
        ax.grid(True, alpha=0.3)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved reversal potential effects to {save_path}")
    
    return fig