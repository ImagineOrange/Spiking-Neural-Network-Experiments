# article_mathematics_visualizations.py
# Educational visualizations for the SNN article mathematics
# Uses white background for publication-quality figures

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.gridspec import GridSpec

# --- Add parent directory to path for imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# --- Import LIF Neuron Class ---
from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal
print("Successfully imported LIF neuron class.")

# --- Plotting Style: Clean white background for article ---
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# Define color palette for consistency
COLORS = {
    'membrane': '#2563eb',      # Blue
    'threshold': '#dc2626',     # Red
    'rest': '#6b7280',          # Gray
    'reset': '#7c3aed',         # Purple
    'excitatory': '#ea580c',    # Orange
    'inhibitory': '#0891b2',    # Cyan
    'adaptation': '#c026d3',    # Magenta
    'spike': '#16a34a',         # Green
    'theoretical': '#333333',   # Dark gray
    'annotation': '#1f2937',    # Very dark gray
}

DEFAULT_FIG_SIZE = (10, 6)
SAVE_DIR = "article_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Default Neuron Parameters (matching article values) ---
NEURON_PARAMS = {
    'v_rest': -70.0,       # Resting potential (mV) - article mentions ~-70mV
    'v_threshold': -55.0,  # Threshold potential (mV) - article mentions ~-55mV
    'v_reset': -75.0,      # Reset potential (mV) - article mentions ~-75mV
    'tau_m': 10.0,         # Membrane time constant (ms)
    'tau_ref': None,       # Use class default based on is_inhibitory (4.0ms excit, 2.5ms inhib)
    'tau_e': 3.0,          # Excitatory time constant (ms) - article mentions te < ti
    'tau_i': 7.0,          # Inhibitory time constant (ms)
    'e_reversal': 0.0,     # Excitatory reversal potential (Ee)
    'i_reversal': -80.0,   # Inhibitory reversal potential (Ei)
    'v_noise_amp': 0.0,    # No noise for clean mathematical demonstrations
    'i_noise_amp': 0.0,
    'adaptation_increment': 0.2,   # Tuned for ~2:1 adaptation ratio with tau_ref=4.0ms
    'tau_adaptation': 100.0,
}


# ==============================================================================
# FIGURE 1: THE MEMBRANE LEAK TERM - -(V - Vrest) / tau_m
# Demonstrates exponential decay to resting potential
# ==============================================================================
def plot_membrane_leak():
    """
    Visualizes the passive membrane decay (leak) term from the LIF equation:
    dV/dt = -(V - V_rest) / tau_m

    Shows how membrane potential decays exponentially to rest with time constant tau_m.
    """
    print("\n--- Figure 1: Membrane Leak (Passive Decay) ---")

    neuron = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)
    dt = 0.1
    T = 80  # ms - enough time to see full decay
    time = np.arange(0, T, dt)

    # Test multiple initial conditions to show decay from different starting points
    # Note: Keep all starting points below threshold to show pure passive decay
    initial_voltages = [-57.0, -63.0, -80.0, -90.0]  # Various starting points (all below threshold)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Simulated decay from multiple starting points
    for v_init in initial_voltages:
        neuron.reset()
        neuron.v = v_init
        for t in time:
            neuron.update(dt)

        # Plot simulated trajectory
        label = f'V(0) = {v_init:.0f} mV'
        ax1.plot(time, neuron.v_history, linewidth=2, label=label)

    # Add reference lines
    ax1.axhline(neuron.v_rest, color=COLORS['rest'], linestyle='--',
                linewidth=1.5, label=f'$V_{{rest}}$ = {neuron.v_rest:.0f} mV')
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle=':',
                linewidth=1.5, label=f'$V_{{thresh}}$ = {neuron.v_threshold:.0f} mV')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Passive Membrane Decay')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True)
    ax1.set_xlim(0, T)

    # Right panel: Show the exponential decay formula with two time constants
    # V(t) = V_rest + (V_0 - V_rest) * exp(-t/tau_m)
    v_init_demo = -57.0  # Below threshold to show pure decay
    v_rest = neuron.v_rest
    tau_m_1 = 10.0  # First time constant
    tau_m_2 = 20.0  # Second time constant for comparison

    theoretical_1 = v_rest + (v_init_demo - v_rest) * np.exp(-time / tau_m_1)
    theoretical_2 = v_rest + (v_init_demo - v_rest) * np.exp(-time / tau_m_2)

    ax2.plot(time, theoretical_1, color=COLORS['membrane'], linewidth=2.5,
             label=f'$\\tau_m$ = {tau_m_1:.0f} ms')
    ax2.plot(time, theoretical_2, color=COLORS['excitatory'], linewidth=2.5,
             label=f'$\\tau_m$ = {tau_m_2:.0f} ms')
    ax2.axhline(v_rest, color=COLORS['rest'], linestyle='--', linewidth=1.5,
                label=f'$V_{{rest}}$ = {v_rest:.0f} mV')

    # Mark the time constants for both curves
    v_at_tau_1 = v_rest + (v_init_demo - v_rest) * np.exp(-1)  # V at t=tau_m_1
    v_at_tau_2 = v_rest + (v_init_demo - v_rest) * np.exp(-1)  # V at t=tau_m_2
    ax2.axvline(tau_m_1, color=COLORS['membrane'], linestyle=':', alpha=0.7)
    ax2.axvline(tau_m_2, color=COLORS['excitatory'], linestyle=':', alpha=0.7)
    ax2.plot(tau_m_1, v_at_tau_1, 'o', color=COLORS['membrane'], markersize=10, zorder=5)
    ax2.plot(tau_m_2, v_at_tau_2, 'o', color=COLORS['excitatory'], markersize=10, zorder=5)

    # Annotate both time constants together (nearby labels for easy comparison)
    ax2.annotate(f'$\\tau$ = {tau_m_1:.0f} ms',
                 xy=(tau_m_1, v_at_tau_1), xytext=(tau_m_1 + 18, v_at_tau_1 + 3),
                 fontsize=9, color=COLORS['membrane'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['membrane']))

    ax2.annotate(f'$\\tau$ = {tau_m_2:.0f} ms',
                 xy=(tau_m_2, v_at_tau_2), xytext=(tau_m_1 + 18, v_at_tau_1 - 1),
                 fontsize=9, color=COLORS['excitatory'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['excitatory']))

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.set_title('Exponential Decay with Time Constant $\\tau_m$')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True)
    ax2.set_xlim(0, T)

    # Add equation annotation
    equation_text = (
        "Membrane Leak Equation:\n"
        "$\\frac{dV}{dt} = \\frac{-(V - V_{rest})}{\\tau_m}$"
    )
    fig.text(0.5, 0.02, equation_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig1_membrane_leak.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 2: CONDUCTANCE DYNAMICS - g(t+dt) = g(t) * exp(-dt/tau)
# Shows exponential decay of synaptic conductances
# ==============================================================================
def plot_conductance_dynamics():
    """
    Visualizes the conductance decay dynamics from the article:
    g(t + Δt) = g(t) * e^(-Δt/τ)

    Shows both excitatory (tau_e = 3ms) and inhibitory (tau_i = 7ms) time constants,
    demonstrating GABA's more prolonged effect as mentioned in the article.

    Uses the actual LIF neuron class to simulate conductance decay.
    """
    print("\n--- Figure 2: Conductance Dynamics ---")

    dt = 0.1
    T = 40  # ms
    time = np.arange(0, T, dt)

    tau_e = NEURON_PARAMS['tau_e']
    tau_i = NEURON_PARAMS['tau_i']

    # Create two neurons - one to receive excitatory input, one for inhibitory
    neuron_e = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)
    neuron_i = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)

    # Initial conductance values (simulating a spike input at t=0)
    g_e_init = 1.0
    g_i_init = 1.0

    # Apply initial spike to set conductances
    neuron_e.reset()
    neuron_i.reset()
    neuron_e.receive_spike(g_e_init)      # Excitatory input (positive weight)
    neuron_i.receive_spike(-g_i_init)     # Inhibitory input (negative weight)

    # Run simulation and record conductance decay
    for t in time:
        neuron_e.update(dt)
        neuron_i.update(dt)

    # Get simulated conductances from neuron histories
    g_e_simulated = neuron_e.g_e_history
    g_i_simulated = neuron_i.g_i_history

    # Calculate theoretical decay for comparison
    g_e_theoretical = g_e_init * np.exp(-time / tau_e)
    g_i_theoretical = g_i_init * np.exp(-time / tau_i)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Both conductances from actual neuron simulation
    ax1.plot(time, g_e_simulated, color=COLORS['excitatory'], linewidth=2.5,
             label=f'$g_e(t)$ simulated: $\\tau_e$ = {tau_e:.0f} ms')
    ax1.plot(time, g_i_simulated, color=COLORS['inhibitory'], linewidth=2.5,
             label=f'$g_i(t)$ simulated: $\\tau_i$ = {tau_i:.0f} ms')

    # Overlay theoretical curves (dashed)
    ax1.plot(time, g_e_theoretical, '--', color=COLORS['excitatory'], linewidth=1.5,
             alpha=0.5, label='Theoretical $g_e$')
    ax1.plot(time, g_i_theoretical, '--', color=COLORS['inhibitory'], linewidth=1.5,
             alpha=0.5, label='Theoretical $g_i$')

    # Mark time constants
    ax1.axvline(tau_e, color=COLORS['excitatory'], linestyle=':', alpha=0.6)
    ax1.axvline(tau_i, color=COLORS['inhibitory'], linestyle=':', alpha=0.6)

    # 36.8% markers (at t = tau)
    g_at_tau_e = g_e_init * np.exp(-1)
    ax1.axhline(g_at_tau_e, color=COLORS['annotation'], linestyle='--', alpha=0.4)
    ax1.text(T - 2, g_at_tau_e + 0.02, '36.8%', fontsize=9, ha='right',
             color=COLORS['annotation'])

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Conductance (normalized)')
    ax1.set_title('Synaptic Conductance Decay (LIF Neuron Simulation)')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax1.grid(True)
    ax1.set_xlim(0, T)
    ax1.set_ylim(0, 1.1)

    # Right panel: Detailed comparison showing actual neuron vs theoretical
    # Run a fresh simulation with more detail
    neuron_detail = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)
    neuron_detail.reset()
    neuron_detail.receive_spike(1.0)  # Initial excitatory spike

    steps = 100
    time_calc = np.arange(0, steps * dt, dt)

    for t in time_calc:
        neuron_detail.update(dt)

    g_neuron = neuron_detail.g_e_history
    g_theoretical = 1.0 * np.exp(-time_calc / tau_e)

    ax2.plot(time_calc, g_neuron, 'o-', color=COLORS['excitatory'],
             markersize=2, linewidth=1.5, label='LIF Neuron simulation')
    ax2.plot(time_calc, g_theoretical, '--', color=COLORS['theoretical'],
             linewidth=2, alpha=0.7, label='Theoretical: $g \\cdot e^{-t/\\tau}$')

    # Show the exact values for the first few steps in inset
    ax2_inset = ax2.inset_axes([0.5, 0.5, 0.45, 0.45])
    ax2_inset.plot(time_calc[:6], g_neuron[:6], 'o-', color=COLORS['excitatory'],
                   markersize=6, linewidth=2, label='Simulated')
    ax2_inset.plot(time_calc[:6], g_theoretical[:6], 's--', color=COLORS['theoretical'],
                   markersize=4, linewidth=1.5, alpha=0.7, label='Theoretical')
    ax2_inset.set_xlim(-0.05, 0.55)
    ax2_inset.set_ylim(0.94, 1.01)
    ax2_inset.grid(True, alpha=0.5)
    ax2_inset.set_title('First 5 steps (zoomed)', fontsize=9)
    ax2_inset.legend(fontsize=7, loc='lower left')

    # Annotate the calculation
    calc_text = (
        f"$g_e(t + 0.1) = 1.0 \\times e^{{-0.1/3}}$\n"
        f"$= 1.0 \\times e^{{-0.0333}}$\n"
        f"$\\approx 0.967$ (3.33% drop)"
    )
    ax2.text(5, 0.6, calc_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#fef3c7', edgecolor='#f59e0b'))

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Conductance (normalized)')
    ax2.set_title(f'Neuron vs Theory ($\\tau_e$ = {tau_e} ms, $\\Delta t$ = {dt} ms)')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True)

    # Add equation
    equation_text = "Conductance Decay: $g(t + \\Delta t) = g(t) \\cdot e^{-\\Delta t / \\tau}$"
    fig.text(0.5, 0.02, equation_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig2_conductance_dynamics.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 3: SYNAPTIC CURRENT WITH DRIVING FORCE - I_syn = g * (E - V)
# Demonstrates reversal potentials and driving force
# ==============================================================================
def plot_driving_force():
    """
    Visualizes the synaptic current calculation with driving force:
    I_e = g_e * (E_e - V)
    I_i = g_i * (E_i - V)
    I_syn = I_e + I_i

    Shows how the driving force modulates synaptic current based on membrane potential.
    Uses the actual LIF neuron to demonstrate the driving force effect.
    """
    print("\n--- Figure 3: Synaptic Current and Driving Force ---")

    E_e = NEURON_PARAMS['e_reversal']    # 0 mV
    E_i = NEURON_PARAMS['i_reversal']    # -80 mV
    V_rest = NEURON_PARAMS['v_rest']     # -70 mV
    V_thresh = NEURON_PARAMS['v_threshold']  # -55 mV

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Range of membrane potentials - extend well beyond reversal potentials
    V = np.linspace(-110, 30, 300)

    # Fixed conductance for demonstration
    g_e = 0.5
    g_i = 0.5

    # Calculate currents using same formula as neuron class
    I_e = g_e * (E_e - V)
    I_i = g_i * (E_i - V)

    # Panel 1: Excitatory current I-V curve
    ax1 = axes[0]
    ax1.plot(V, I_e, color=COLORS['excitatory'], linewidth=2.5,
             label=f'$I_e = g_e \\cdot (E_e - V)$')
    ax1.axhline(0, color=COLORS['annotation'], linestyle='-', linewidth=0.5)
    ax1.axvline(E_e, color=COLORS['excitatory'], linestyle='--', linewidth=1.5,
                label=f'$E_e$ = {E_e:.0f} mV (reversal)')
    ax1.axvline(V_rest, color=COLORS['rest'], linestyle=':', linewidth=1.5,
                label=f'$V_{{rest}}$ = {V_rest:.0f} mV')

    # Depolarizing region (orange) and hyperpolarizing region (cyan - like inhibitory)
    ax1.fill_between(V, 0, I_e, where=(I_e > 0), alpha=0.2, color=COLORS['excitatory'])
    ax1.fill_between(V, 0, I_e, where=(I_e < 0), alpha=0.2, color=COLORS['inhibitory'])

    ax1.set_xlabel('Membrane Potential V (mV)', fontsize=12)
    ax1.set_ylabel('Current (arbitrary units)', fontsize=12)
    ax1.set_title('Excitatory Current ($g_e$ = 0.5)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(-110, 30)
    ax1.tick_params(axis='both', labelsize=11)

    ax1.annotate('Depolarizing\n(V < $E_e$)', xy=(-50, 20), fontsize=11,
                 ha='center', va='center', color=COLORS['excitatory'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['excitatory']))
    ax1.annotate('Hyperpolarizing\n(V > $E_e$)', xy=(15, -6), fontsize=11,
                 ha='center', va='center', color=COLORS['inhibitory'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['inhibitory']))

    # Panel 2: Inhibitory current I-V curve
    ax2 = axes[1]
    ax2.plot(V, I_i, color=COLORS['inhibitory'], linewidth=2.5,
             label=f'$I_i = g_i \\cdot (E_i - V)$')
    ax2.axhline(0, color=COLORS['annotation'], linestyle='-', linewidth=0.5)
    ax2.axvline(E_i, color=COLORS['inhibitory'], linestyle='--', linewidth=1.5,
                label=f'$E_i$ = {E_i:.0f} mV (reversal)')
    ax2.axvline(V_rest, color=COLORS['rest'], linestyle=':', linewidth=1.5,
                label=f'$V_{{rest}}$ = {V_rest:.0f} mV')

    # Hyperpolarizing region (cyan) and depolarizing region (orange - like excitatory)
    ax2.fill_between(V, 0, I_i, where=(I_i < 0), alpha=0.2, color=COLORS['inhibitory'])
    ax2.fill_between(V, 0, I_i, where=(I_i > 0), alpha=0.2, color=COLORS['excitatory'])

    ax2.set_xlabel('Membrane Potential V (mV)', fontsize=12)
    ax2.set_ylabel('Current (arbitrary units)', fontsize=12)
    ax2.set_title('Inhibitory Current ($g_i$ = 0.5)', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(-110, 30)
    ax2.tick_params(axis='both', labelsize=11)

    ax2.annotate('Hyperpolarizing\n(V > $E_i$)', xy=(-20, -25), fontsize=11,
                 ha='center', va='center', color=COLORS['inhibitory'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['inhibitory']))
    ax2.annotate('Depolarizing\n(V < $E_i$)', xy=(-95, 6), fontsize=11,
                 ha='center', va='center', color=COLORS['excitatory'],
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['excitatory']))

    # Add main equation
    equation_text = (
        "$I_e = g_e \\cdot (E_e - V)$ | "
        "$I_i = g_i \\cdot (E_i - V)$ | "
        "$I_{syn} = I_e + I_i$"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig3_driving_force.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 4: FORWARD EULER INTEGRATION - V(t+dt) = V(t) + dt * dV/dt
# Demonstrates numerical integration of the LIF equation
# ==============================================================================
def plot_euler_integration():
    """
    Visualizes the forward Euler method used to numerically integrate the LIF equation:
    V(t + Δt) = V(t) + Δt * dV/dt

    Compares the actual LIF neuron class (which uses Euler) against analytical solution.
    Shows how different timestep sizes affect accuracy.
    """
    print("\n--- Figure 4: Forward Euler Integration ---")

    V_rest = NEURON_PARAMS['v_rest']
    tau_m = NEURON_PARAMS['tau_m']
    V_init = -57.0  # Starting voltage (below threshold to show pure decay)
    T = 50  # ms

    # Analytical solution for comparison
    t_analytical = np.linspace(0, T, 1000)
    V_analytical = V_rest + (V_init - V_rest) * np.exp(-t_analytical / tau_m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Compare LIF neuron simulation at different timesteps
    ax1 = axes[0]
    ax1.plot(t_analytical, V_analytical, 'k-', linewidth=3, alpha=0.3,
             label='Analytical solution')

    # Test different timesteps using actual LIF neuron
    timesteps = [0.1, 1.0, 5.0]
    colors_dt = ['#2563eb', '#16a34a', '#dc2626']

    for dt, color in zip(timesteps, colors_dt):
        # Create neuron and run simulation with this timestep
        neuron = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)
        neuron.reset()
        neuron.v = V_init  # Set initial voltage

        time = np.arange(0, T + dt, dt)
        for _ in time:
            neuron.update(dt)

        # Plot neuron's voltage history
        ax1.plot(time, neuron.v_history, 'o-', color=color, linewidth=1.5, markersize=3,
                 label=f'LIF Neuron: $\\Delta t$ = {dt} ms')

    ax1.axhline(V_rest, color=COLORS['rest'], linestyle='--', linewidth=1,
                label=f'$V_{{rest}}$ = {V_rest:.0f} mV')

    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('LIF Neuron Simulation at Different Timesteps')
    ax1.legend(loc='best', framealpha=0.9, fontsize=8)
    ax1.grid(True)
    ax1.set_xlim(0, T)

    # Right panel: Detailed view showing neuron's internal Euler steps
    ax2 = axes[1]

    dt_demo = 5.0
    T_demo = 25

    # Run LIF neuron with large timestep
    neuron_demo = Layered_LIFNeuronWithReversal(**NEURON_PARAMS)
    neuron_demo.reset()
    neuron_demo.v = V_init

    time_demo = np.arange(0, T_demo + dt_demo, dt_demo)
    for _ in time_demo:
        neuron_demo.update(dt_demo)

    # Plot analytical solution
    t_fine = np.linspace(0, T_demo, 500)
    V_fine = V_rest + (V_init - V_rest) * np.exp(-t_fine / tau_m)
    ax2.plot(t_fine, V_fine, 'k-', linewidth=2, alpha=0.4, label='Analytical solution')

    # Plot neuron simulation with Euler step visualization
    V_neuron = neuron_demo.v_history

    for i in range(len(time_demo) - 1):
        # Draw tangent line segment (the Euler step)
        t_tangent = np.array([time_demo[i], time_demo[i+1]])
        V_tangent = np.array([V_neuron[i], V_neuron[i+1]])
        ax2.plot(t_tangent, V_tangent, '-', color=COLORS['excitatory'], linewidth=2)

        # Mark the point
        ax2.plot(time_demo[i], V_neuron[i], 'o', color=COLORS['membrane'], markersize=10, zorder=5)

        # Draw error arrow showing difference from analytical
        if i < len(time_demo) - 2:
            analytical_at_next = V_rest + (V_init - V_rest) * np.exp(-time_demo[i+1] / tau_m)
            ax2.annotate('', xy=(time_demo[i+1], V_neuron[i+1]),
                        xytext=(time_demo[i+1], analytical_at_next),
                        arrowprops=dict(arrowstyle='<->', color=COLORS['threshold'], lw=1.5))

    # Final point
    ax2.plot(time_demo[-1], V_neuron[-1], 'o', color=COLORS['membrane'], markersize=10, zorder=5)

    # Annotate showing what neuron computes internally
    dVdt_initial = -(V_init - V_rest) / tau_m
    ax2.annotate(f'LIF Neuron computes:\n$V(0) = {V_init:.0f}$\n$\\frac{{dV}}{{dt}} = {dVdt_initial:.1f}$\n$V(\\Delta t) = V + \\Delta t \\cdot \\frac{{dV}}{{dt}}$',
                 xy=(0, V_init), xytext=(8, V_init + 2),
                 fontsize=8, color=COLORS['annotation'],
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['annotation']))

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.set_title(f'LIF Neuron Euler Steps ($\\Delta t$ = {dt_demo} ms)')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True)
    ax2.set_xlim(-1, T_demo + 1)

    # Add equation
    equation_text = "Forward Euler (used by LIF Neuron): $V(t + \\Delta t) = V(t) + \\Delta t \\cdot \\frac{dV}{dt}$"
    fig.text(0.5, 0.02, equation_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig4_euler_integration.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 5: ACTION POTENTIAL AND THRESHOLD - Complete spike mechanism
# Demonstrates the full LIF neuron dynamics including spike and reset
# ==============================================================================
def plot_action_potential():
    """
    Visualizes the complete action potential mechanism:
    1. Spike occurs when V >= V_threshold
    2. V is reset to V_reset
    3. Refractory period blocks integration

    Shows the threshold detection and reset behavior.
    """
    print("\n--- Figure 5: Action Potential Mechanism ---")

    neuron_params = {**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    neuron = Layered_LIFNeuronWithReversal(**neuron_params)

    dt = 0.1
    T = 100  # ms
    time = np.arange(0, T, dt)

    # Apply strong constant excitation to induce spiking
    constant_excitation = 1.2

    neuron.reset()
    spike_times_recorded = []

    for i, t in enumerate(time):
        neuron.apply_external_stimulus(constant_excitation)
        spiked = neuron.update(dt)
        if spiked:
            spike_times_recorded.append(t)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.3)

    # Panel 1: Membrane potential
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, neuron.v_history, color=COLORS['membrane'], linewidth=1.5,
             label='$V(t)$')

    # Reference lines
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=2, label=f'$V_{{thresh}}$ = {neuron.v_threshold:.0f} mV')
    ax1.axhline(neuron.v_rest, color=COLORS['rest'], linestyle=':',
                linewidth=1.5, label=f'$V_{{rest}}$ = {neuron.v_rest:.0f} mV')
    ax1.axhline(neuron.v_reset, color=COLORS['reset'], linestyle='--',
                linewidth=1.5, label=f'$V_{{reset}}$ = {neuron.v_reset:.0f} mV')

    # Mark spikes
    for spike_t in spike_times_recorded[:5]:  # First few spikes
        ax1.axvline(spike_t, color=COLORS['spike'], linestyle='-', alpha=0.5, linewidth=1)

    # Annotate the first spike
    if spike_times_recorded:
        first_spike = spike_times_recorded[0]
        ax1.annotate('Spike!\n$V \\geq V_{thresh}$\n$V \\leftarrow V_{reset}$',
                     xy=(first_spike, neuron.v_threshold),
                     xytext=(first_spike + 10, neuron.v_threshold + 5),
                     fontsize=9, color=COLORS['spike'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['spike']),
                     bbox=dict(boxstyle='round', facecolor='#dcfce7', edgecolor=COLORS['spike']))

        # Mark refractory period
        ax1.axvspan(first_spike, first_spike + neuron.tau_ref,
                    color=COLORS['reset'], alpha=0.15, label=f'Refractory ({neuron.tau_ref} ms)')

    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title('Membrane Potential with Action Potentials')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.grid(True)
    ax1.set_xlim(0, T)
    ax1.set_ylim(neuron.v_reset - 5, neuron.v_threshold + 10)

    # Panel 2: Excitatory conductance
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(time, neuron.g_e_history, color=COLORS['excitatory'], linewidth=1.5,
             label='$g_e(t)$ (external stimulus)')
    ax2.set_ylabel('$g_e$')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Panel 3: Spike raster
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if spike_times_recorded:
        ax3.eventplot(spike_times_recorded, orientation='horizontal',
                      colors=COLORS['spike'], lineoffsets=0.5, linelengths=0.8)
    ax3.set_yticks([])
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Spikes')
    ax3.set_title(f'Spike Times ({len(spike_times_recorded)} spikes)', fontsize=10)

    # Add spike mechanism equations
    equation_text = (
        "Spike Mechanism: "
        "$V \\geq V_{thresh} \\Rightarrow$ Spike | "
        "$V \\leftarrow V_{reset}$ | "
        "$t_{spike} \\leftarrow 0$ | "
        "If $t_{spike} < \\tau_{ref}$: integration blocked"
    )
    fig.text(0.5, 0.02, equation_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig5_action_potential.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 6: SPIKE-FREQUENCY ADAPTATION - I_adapt dynamics
# Demonstrates adaptation current buildup and effect on firing rate
# ==============================================================================
def plot_adaptation():
    """
    Visualizes spike-frequency adaptation (I_adapt):
    - Adaptation current increases with each spike
    - Decays exponentially with time constant tau_adaptation
    - Reduces firing rate over time under constant stimulation
    """
    print("\n--- Figure 6: Spike-Frequency Adaptation ---")

    # Uses default adaptation_increment=0.1, with faster tau for clearer visualization
    neuron_params = {
        **NEURON_PARAMS,
        'tau_adaptation': 80.0,  # Faster than default (100ms) for clearer demo
        'v_noise_amp': 0.0,
        'i_noise_amp': 0.0
    }
    neuron = Layered_LIFNeuronWithReversal(**neuron_params)

    dt = 0.1
    T = 500  # ms
    time = np.arange(0, T, dt)

    # Constant strong stimulation
    stim_start = 50
    stim_end = 350
    stim_strength = 1.5

    neuron.reset()
    spike_times_adapt = []
    adaptation_current_history = []  # Track the actual current, not just conductance

    for i, t in enumerate(time):
        if stim_start <= t < stim_end:
            neuron.apply_external_stimulus(stim_strength)
        else:
            neuron.apply_external_stimulus(0.0)

        spiked = neuron.update(dt)
        if spiked:
            spike_times_adapt.append(t)

        # Calculate actual adaptation current: I_adapt = g_adapt * (E_K - V)
        e_k = getattr(neuron, 'e_k_reversal', -90.0)
        i_adapt = neuron.adaptation * (e_k - neuron.v)
        adaptation_current_history.append(i_adapt)

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[2, 1.5, 1, 1], hspace=0.3)

    # Panel 1: Membrane potential
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, neuron.v_history, color=COLORS['membrane'], linewidth=1,
             label='$V(t)$')
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1.5, label='$V_{thresh}$')
    ax1.axvspan(stim_start, stim_end, color=COLORS['excitatory'], alpha=0.1,
                label='Stimulus period')

    ax1.set_ylabel('V (mV)')
    ax1.set_title('Membrane Potential with Spike-Frequency Adaptation')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim(0, T)

    # Panel 2: Adaptation current (conductance-based: I_adapt = g_adapt * (E_K - V))
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(time, adaptation_current_history, color=COLORS['adaptation'], linewidth=2,
             label=f'$I_{{adapt}} = g_{{adapt}} \\cdot (E_K - V)$')
    ax2.axvspan(stim_start, stim_end, color=COLORS['excitatory'], alpha=0.1)

    # Mark spike-induced jumps
    for i, spike_t in enumerate(spike_times_adapt[:10]):
        idx = int(spike_t / dt)
        if idx < len(adaptation_current_history) - 1:
            ax2.plot(spike_t, adaptation_current_history[idx], 'o',
                    color=COLORS['spike'], markersize=4)

    ax2.set_ylabel('$I_{adapt}$ (a.u.)')
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Annotate adaptation effect
    if spike_times_adapt:
        first_spike_idx = int(spike_times_adapt[0] / dt)
        ax2.annotate(f'$\\Delta g_{{adapt}}$ = {neuron_params["adaptation_increment"]}\nper spike',
                     xy=(spike_times_adapt[0], adaptation_current_history[first_spike_idx]),
                     xytext=(stim_start + 50, min(adaptation_current_history) * 0.3),
                     fontsize=9, color=COLORS['adaptation'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['adaptation']),
                     bbox=dict(boxstyle='round', facecolor='#fdf4ff', edgecolor=COLORS['adaptation']))

    # Panel 3: Instantaneous firing rate
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    if len(spike_times_adapt) > 1:
        # Calculate inter-spike intervals and instantaneous rate
        isis = np.diff(spike_times_adapt)
        inst_rates = 1000.0 / isis  # Convert to Hz
        rate_times = spike_times_adapt[1:]

        ax3.plot(rate_times, inst_rates, 'o-', color=COLORS['spike'],
                 linewidth=1.5, markersize=4, label='Instantaneous rate')
        ax3.axhline(np.mean(inst_rates), color=COLORS['annotation'], linestyle='--',
                    alpha=0.5, label=f'Mean: {np.mean(inst_rates):.1f} Hz')

    ax3.axvspan(stim_start, stim_end, color=COLORS['excitatory'], alpha=0.1)
    ax3.set_ylabel('Rate (Hz)')
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax3.grid(True)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Panel 4: Spike raster
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if spike_times_adapt:
        ax4.eventplot(spike_times_adapt, orientation='horizontal',
                      colors=COLORS['spike'], lineoffsets=0.5, linelengths=0.8)
    ax4.axvspan(stim_start, stim_end, color=COLORS['excitatory'], alpha=0.1)
    ax4.set_yticks([])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Spikes')
    ax4.set_title(f'Total: {len(spike_times_adapt)} spikes', fontsize=10)

    # Add equation
    equation_text = (
        "Conductance-based adaptation: $I_{adapt} = g_{adapt} \\cdot (E_K - V)$ | "
        "On spike: $g_{adapt} \\leftarrow g_{adapt} + \\Delta g$ | "
        "Decay: $g_{adapt}(t+\\Delta t) = g_{adapt}(t) \\cdot e^{-\\Delta t / \\tau_w}$"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig6_adaptation.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 7: COMPLETE LIF EQUATION VISUALIZATION
# Shows all terms working together in the full dV/dt equation
# ==============================================================================
def plot_complete_lif_equation():
    """
    Comprehensive visualization of the complete LIF equation:
    dV/dt = -(V - V_rest)/tau_m + I_syn - I_adapt

    Shows each term's contribution over time during neuron activity.
    """
    print("\n--- Figure 7: Complete LIF Equation Components ---")

    neuron_params = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.0,
        'i_noise_amp': 0.0
    }  # Uses default adaptation_increment=0.1, tau_adaptation=100.0 from NEURON_PARAMS
    neuron = Layered_LIFNeuronWithReversal(**neuron_params)

    dt = 0.1
    T = 200  # ms
    time = np.arange(0, T, dt)

    # Variable stimulation pattern
    neuron.reset()

    # Record each component of dV/dt
    leak_term = []
    syn_term = []
    adapt_term = []
    total_dvdt = []
    spike_times_complete = []

    for i, t in enumerate(time):
        # Apply stimulus: strong excitation 40-100ms, weak 120-160ms
        if 40 <= t < 100:
            neuron.apply_external_stimulus(1.0)
        elif 120 <= t < 160:
            neuron.apply_external_stimulus(0.5)
        else:
            neuron.apply_external_stimulus(0.0)

        # Record components before update
        leak = -(neuron.v - neuron.v_rest) / neuron.tau_m
        i_syn = (neuron.g_e * (neuron.e_reversal - neuron.v) +
                 neuron.g_i * (neuron.i_reversal - neuron.v) +
                 neuron.external_stim_g * (neuron.e_reversal - neuron.v))
        adapt = neuron.adaptation

        leak_term.append(leak)
        syn_term.append(i_syn)
        adapt_term.append(adapt)
        total_dvdt.append(leak + i_syn - adapt)

        spiked = neuron.update(dt)
        if spiked:
            spike_times_complete.append(t)

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[2, 1.5, 1, 1, 1], hspace=0.25)

    # Panel 1: Membrane potential
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, neuron.v_history, color=COLORS['membrane'], linewidth=1.5,
             label='$V(t)$')
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1.5, label='$V_{thresh}$')
    ax1.axhline(neuron.v_rest, color=COLORS['rest'], linestyle=':',
                linewidth=1, label='$V_{rest}$')

    # Mark stimulus periods
    ax1.axvspan(40, 100, color=COLORS['excitatory'], alpha=0.15, label='Strong stim')
    ax1.axvspan(120, 160, color=COLORS['excitatory'], alpha=0.08, label='Weak stim')

    ax1.set_ylabel('V (mV)')
    ax1.set_title('Complete LIF Neuron Dynamics')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(0, T)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # Panel 2: dV/dt components
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(time, leak_term, color=COLORS['rest'], linewidth=1.5, alpha=0.8,
             label='Leak: $-(V-V_{rest})/\\tau_m$')
    ax2.plot(time, syn_term, color=COLORS['excitatory'], linewidth=1.5, alpha=0.8,
             label='Synaptic: $I_{syn}$')
    ax2.plot(time, [-a for a in adapt_term], color=COLORS['adaptation'], linewidth=1.5, alpha=0.8,
             label='Adaptation: $-I_{adapt}$')
    ax2.axhline(0, color='black', linewidth=0.5)

    ax2.set_ylabel('dV/dt\ncomponents')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax2.grid(True, alpha=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Panel 3: Total dV/dt
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(time, 0, total_dvdt, where=[t > 0 for t in total_dvdt],
                     color=COLORS['spike'], alpha=0.3, label='Depolarizing')
    ax3.fill_between(time, 0, total_dvdt, where=[t < 0 for t in total_dvdt],
                     color=COLORS['inhibitory'], alpha=0.3, label='Hyperpolarizing')
    ax3.plot(time, total_dvdt, color=COLORS['membrane'], linewidth=1.5)
    ax3.axhline(0, color='black', linewidth=0.5)

    ax3.set_ylabel('Total\ndV/dt')
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax3.grid(True, alpha=0.5)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Panel 4: Conductances
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(time, neuron.g_e_history, color=COLORS['excitatory'], linewidth=1.5,
             label='$g_e$')
    ax4.plot(time, neuron.g_i_history, color=COLORS['inhibitory'], linewidth=1.5,
             label='$g_i$')

    ax4.set_ylabel('g')
    ax4.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax4.grid(True, alpha=0.5)
    plt.setp(ax4.get_xticklabels(), visible=False)

    # Panel 5: Spike raster
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    if spike_times_complete:
        ax5.eventplot(spike_times_complete, orientation='horizontal',
                      colors=COLORS['spike'], lineoffsets=0.5, linelengths=0.8)
    ax5.set_yticks([])
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Spikes')

    # Add the complete equation
    equation_text = (
        "Complete LIF Equation: "
        "$\\frac{dV}{dt} = \\frac{-(V - V_{rest})}{\\tau_m} + I_{syn} - I_{adapt}$"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig7_complete_lif.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 8: WEIGHT-BASED CONDUCTANCE UPDATE
# Demonstrates how synaptic weights affect conductance
# ==============================================================================
def plot_weight_conductance():
    """
    Visualizes how synaptic weights translate to conductance changes:
    g_e <- g_e + w  (if w > 0)
    g_i <- g_i + |w|  (if w < 0)
    """
    print("\n--- Figure 8: Synaptic Weight to Conductance ---")

    neuron_params = {**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    neuron = Layered_LIFNeuronWithReversal(**neuron_params)

    dt = 0.1
    T = 100  # ms
    time = np.arange(0, T, dt)

    # Sequence of inputs with different weights
    inputs = [
        (15.0, 0.5, 'E'),    # Excitatory, medium
        (25.0, 1.0, 'E'),    # Excitatory, strong
        (40.0, 0.3, 'E'),    # Excitatory, weak
        (55.0, -0.6, 'I'),   # Inhibitory, medium
        (70.0, -1.2, 'I'),   # Inhibitory, strong
        (85.0, 0.8, 'E'),    # Excitatory, strong
    ]

    neuron.reset()
    for t in time:
        for input_t, weight, _ in inputs:
            if abs(t - input_t) < dt / 2:
                neuron.receive_spike(weight)
        neuron.update(dt)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 1: Membrane potential
    ax1 = axes[0]
    ax1.plot(time, neuron.v_history, color=COLORS['membrane'], linewidth=1.5,
             label='$V(t)$')
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, label='$V_{thresh}$')
    ax1.axhline(neuron.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1)

    # Mark input times
    for input_t, weight, input_type in inputs:
        color = COLORS['excitatory'] if input_type == 'E' else COLORS['inhibitory']
        ax1.axvline(input_t, color=color, linestyle=':', alpha=0.7)

    ax1.set_ylabel('V (mV)')
    ax1.set_title('Effect of Synaptic Inputs on Neuron Dynamics')
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.5)

    # Panel 2: Excitatory conductance
    ax2 = axes[1]
    ax2.plot(time, neuron.g_e_history, color=COLORS['excitatory'], linewidth=2,
             label='$g_e$')

    # Annotate excitatory inputs
    for input_t, weight, input_type in inputs:
        if input_type == 'E':
            idx = int(input_t / dt)
            if idx < len(neuron.g_e_history):
                ax2.annotate(f'w={weight}', xy=(input_t, neuron.g_e_history[idx]),
                            xytext=(input_t + 3, neuron.g_e_history[idx] + 0.1),
                            fontsize=8, color=COLORS['excitatory'])
                ax2.axvline(input_t, color=COLORS['excitatory'], linestyle=':', alpha=0.5)

    ax2.set_ylabel('$g_e$')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.5)

    # Panel 3: Inhibitory conductance
    ax3 = axes[2]
    ax3.plot(time, neuron.g_i_history, color=COLORS['inhibitory'], linewidth=2,
             label='$g_i$')

    # Annotate inhibitory inputs
    for input_t, weight, input_type in inputs:
        if input_type == 'I':
            idx = int(input_t / dt)
            if idx < len(neuron.g_i_history):
                ax3.annotate(f'w={weight}', xy=(input_t, neuron.g_i_history[idx]),
                            xytext=(input_t + 3, neuron.g_i_history[idx] + 0.1),
                            fontsize=8, color=COLORS['inhibitory'])
                ax3.axvline(input_t, color=COLORS['inhibitory'], linestyle=':', alpha=0.5)

    ax3.set_ylabel('$g_i$')
    ax3.set_xlabel('Time (ms)')
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax3.grid(True, alpha=0.5)

    # Add equation
    equation_text = (
        "Weight Update Rules: "
        "$g_e \\leftarrow g_e + w$ (if $w > 0$) | "
        "$g_i \\leftarrow g_i + |w|$ (if $w < 0$)"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig8_weight_conductance.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 9: EXCITATION, INHIBITION, AND NOISE DYNAMICS
# Demonstrates realistic neural behavior with different input patterns
# ==============================================================================
def plot_excitation_inhibition_noise():
    """
    Visualizes three realistic neural scenarios using the actual LIF neuron class:
    1. Neuron at rest → strong excitatory input → replaced by inhibitory input
    2. Chaotic mix of excitatory and inhibitory inputs
    3. Noisy membrane/synaptic potentials causing spontaneous spikes

    All logic uses the actual Layered_LIFNeuronWithReversal class methods.
    """
    print("\n--- Figure 9: Excitation, Inhibition, and Noise Dynamics ---")

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=False)

    # =========================================================================
    # Panel 1: Resting neuron → Excitatory stimulation → Inhibitory cutoff
    # Compare WITH and WITHOUT adaptation
    # =========================================================================
    ax1 = axes[0]

    dt = 0.1
    T1 = 350  # ms - longer simulation
    time1 = np.arange(0, T1, dt)

    # Define stimulation phases (spread out more)
    excitatory_start = 50   # ms - start excitatory input
    excitatory_end = 150    # ms - stop excitatory, start inhibitory
    inhibitory_end = 250    # ms - stop inhibitory

    # --- Run simulation WITH adaptation ---
    neuron1_adapt = Layered_LIFNeuronWithReversal(
        **{**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    )
    neuron1_adapt.reset()
    spike_times_adapt = []

    for t in time1:
        if excitatory_start <= t < excitatory_end:
            if int(t / dt) % 5 == 0:
                neuron1_adapt.receive_spike(0.8)
        if excitatory_end <= t < inhibitory_end:
            if int(t / dt) % 5 == 0:
                neuron1_adapt.receive_spike(-1.2)
        spiked = neuron1_adapt.update(dt)
        if spiked:
            spike_times_adapt.append(t)

    # --- Run simulation WITHOUT adaptation ---
    neuron1_no_adapt = Layered_LIFNeuronWithReversal(
        **{**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0,
           'adaptation_increment': 0.0}  # Disable adaptation
    )
    neuron1_no_adapt.reset()
    spike_times_no_adapt = []

    for t in time1:
        if excitatory_start <= t < excitatory_end:
            if int(t / dt) % 5 == 0:
                neuron1_no_adapt.receive_spike(0.8)
        if excitatory_end <= t < inhibitory_end:
            if int(t / dt) % 5 == 0:
                neuron1_no_adapt.receive_spike(-1.2)
        spiked = neuron1_no_adapt.update(dt)
        if spiked:
            spike_times_no_adapt.append(t)

    # Plot both traces
    ax1.plot(time1, neuron1_adapt.v_history, color=COLORS['membrane'], linewidth=1.5,
             label='With adaptation', alpha=0.9)
    ax1.plot(time1, neuron1_no_adapt.v_history, color=COLORS['spike'], linewidth=1.5,
             label='No adaptation', alpha=0.7, linestyle='--')

    # Mark spikes for adapted neuron only (to reduce clutter)
    for spike_t in spike_times_adapt:
        ax1.axvline(spike_t, color=COLORS['threshold'], linewidth=1, alpha=0.5)

    # Reference lines
    ax1.axhline(neuron1_adapt.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5, label=f'$V_{{thresh}}$ = {neuron1_adapt.v_threshold} mV')
    ax1.axhline(neuron1_adapt.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1,
                label=f'$V_{{rest}}$ = {neuron1_adapt.v_rest} mV')

    # Shade stimulation regions
    ax1.axvspan(excitatory_start, excitatory_end, alpha=0.15, color=COLORS['excitatory'],
                label='Excitatory input')
    ax1.axvspan(excitatory_end, inhibitory_end, alpha=0.15, color=COLORS['inhibitory'],
                label='Inhibitory input')

    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(f'Panel A: Effect of Adaptation (with: {len(spike_times_adapt)} spikes, without: {len(spike_times_no_adapt)} spikes)')
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.5)
    ax1.set_xlim(0, T1)
    ax1.set_ylim(-85, -50)

    # Add annotations (adjusted for new timing)
    ax1.annotate('Rest', xy=(25, -70), fontsize=10, ha='center', color=COLORS['annotation'])
    ax1.annotate('Excitatory\nDriving', xy=(100, -52), fontsize=9, ha='center',
                 color=COLORS['excitatory'], fontweight='bold')
    ax1.annotate('Inhibitory\nSuppression', xy=(200, -78), fontsize=9, ha='center',
                 color=COLORS['inhibitory'], fontweight='bold')
    ax1.annotate('Adaptation\nslows recovery', xy=(300, -74), fontsize=8, ha='center',
                 color=COLORS['adaptation'], fontweight='bold')

    # =========================================================================
    # Panel 2: Chaotic mix of excitatory and inhibitory inputs
    # =========================================================================
    ax2 = axes[1]

    neuron2_params = {**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0}
    neuron2 = Layered_LIFNeuronWithReversal(**neuron2_params)
    neuron2.reset()

    T2 = 500  # ms - longer simulation
    time2 = np.arange(0, T2, dt)

    # Generate random input times (Poisson-like) - fewer events
    np.random.seed(42)  # For reproducibility
    n_excitatory = 40
    n_inhibitory = 25

    excitatory_times = np.sort(np.random.uniform(10, T2 - 10, n_excitatory))
    inhibitory_times = np.sort(np.random.uniform(10, T2 - 10, n_inhibitory))

    # Random weights
    excitatory_weights = np.random.uniform(0.3, 1.2, n_excitatory)
    inhibitory_weights = np.random.uniform(-1.0, -0.3, n_inhibitory)

    spike_times_2 = []

    for t in time2:
        # Check for excitatory inputs
        for i, (input_t, weight) in enumerate(zip(excitatory_times, excitatory_weights)):
            if abs(t - input_t) < dt / 2:
                neuron2.receive_spike(weight)

        # Check for inhibitory inputs
        for i, (input_t, weight) in enumerate(zip(inhibitory_times, inhibitory_weights)):
            if abs(t - input_t) < dt / 2:
                neuron2.receive_spike(weight)

        spiked = neuron2.update(dt)
        if spiked:
            spike_times_2.append(t)

    # Plot membrane potential
    ax2.plot(time2, neuron2.v_history, color=COLORS['membrane'], linewidth=1.2,
             label='$V(t)$')

    # Mark spikes with vertical red lines
    for spike_t in spike_times_2:
        ax2.axvline(spike_t, color=COLORS['threshold'], linewidth=1.5, alpha=0.8)

    # Reference lines
    ax2.axhline(neuron2.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5)
    ax2.axhline(neuron2.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1)

    # Mark input events (small ticks at top/bottom)
    for input_t in excitatory_times:
        ax2.plot(input_t, -50, '|', color=COLORS['excitatory'], markersize=4, alpha=0.5)
    for input_t in inhibitory_times:
        ax2.plot(input_t, -85, '|', color=COLORS['inhibitory'], markersize=4, alpha=0.5)

    ax2.set_ylabel('Membrane Potential (mV)')
    ax2.set_title(f'Panel B: Chaotic Input Mix ({n_excitatory} excitatory, {n_inhibitory} inhibitory events)')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(0, T2)
    ax2.set_ylim(-88, -48)

    # Add legend for input markers
    ax2.text(T2 - 5, -51, 'E inputs', fontsize=8, ha='right', color=COLORS['excitatory'])
    ax2.text(T2 - 5, -84, 'I inputs', fontsize=8, ha='right', color=COLORS['inhibitory'])

    # =========================================================================
    # Panel 3: Noisy membrane/synaptic potentials causing spontaneous spikes
    # =========================================================================
    ax3 = axes[2]

    # Use neuron WITH noise enabled (using actual class parameters) - reduced noise
    neuron3_params = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.8,    # Membrane potential noise (mV std dev) - reduced
        'i_noise_amp': 0.08,   # Synaptic conductance noise (std dev) - reduced
    }
    neuron3 = Layered_LIFNeuronWithReversal(**neuron3_params)
    neuron3.reset()

    # Start closer to threshold to see noise-induced spikes
    neuron3.v = -60.0

    T3 = 600  # ms - longer to see noise effects
    time3 = np.arange(0, T3, dt)

    np.random.seed(123)  # Different seed for different random pattern

    spike_times_3 = []

    # Add occasional weak background excitation to keep membrane near threshold
    for t in time3:
        # Weak background excitation (sub-threshold on its own)
        if int(t / dt) % 100 == 0:  # Every 10ms
            neuron3.receive_spike(0.15)  # Small excitatory nudge

        spiked = neuron3.update(dt)
        if spiked:
            spike_times_3.append(t)

    # Plot membrane potential
    ax3.plot(time3, neuron3.v_history, color=COLORS['membrane'], linewidth=1,
             label='$V(t)$ with noise', alpha=0.9)

    # Mark spikes with vertical red lines
    for spike_t in spike_times_3:
        ax3.axvline(spike_t, color=COLORS['threshold'], linewidth=1.5, alpha=0.8)

    # Reference lines
    ax3.axhline(neuron3.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5, label=f'$V_{{thresh}}$ = {neuron3.v_threshold} mV')
    ax3.axhline(neuron3.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1)

    ax3.set_ylabel('Membrane Potential (mV)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_title(f'Panel C: Noisy Neuron (v_noise={neuron3_params["v_noise_amp"]} mV, '
                  f'i_noise={neuron3_params["i_noise_amp"]}) — {len(spike_times_3)} spontaneous spikes')
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.5)
    ax3.set_xlim(0, T3)
    ax3.set_ylim(-85, -50)

    # Add annotation about noise parameters
    noise_text = (f"Noise from LIF class:\n"
                  f"• v_noise_amp = {neuron3_params['v_noise_amp']} mV\n"
                  f"• i_noise_amp = {neuron3_params['i_noise_amp']}")
    ax3.text(T3 - 10, -82, noise_text, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['annotation'], alpha=0.9))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig9_excitation_inhibition_noise.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 10: SUBTHRESHOLD INTEGRATION TO ACTION POTENTIAL
# Shows resting neuron receiving mixed inputs, building to threshold, then spiking
# ==============================================================================
def plot_subthreshold_to_spike():
    """
    Visualizes the journey from resting potential through subthreshold integration
    to action potential generation:
    - Neuron starts at rest
    - Receives a series of excitatory and inhibitory inputs (all subthreshold)
    - Membrane potential fluctuates as inputs are integrated
    - Finally reaches threshold and fires an action potential
    - No more inputs after the spike - shows return to rest

    The action potential is marked with a prominent red vertical line.
    """
    print("\n--- Figure 10: Subthreshold Integration to Action Potential ---")

    # Create neuron with clean parameters (no noise for clear demonstration)
    neuron_params = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.0,
        'i_noise_amp': 0.0,
        'adaptation_increment': 0.0,  # Disable adaptation for cleaner visualization
    }
    neuron = Layered_LIFNeuronWithReversal(**neuron_params)
    neuron.reset()

    dt = 0.1
    T = 300  # ms total simulation time (extra time to show recovery)
    time = np.arange(0, T, dt)

    # Design input pattern: mixed E/I inputs that gradually build toward threshold
    # Input times and weights carefully chosen for pedagogical clarity
    # Weights are kept very small so individual inputs cause only small deflections
    # Spaced far enough apart that conductance mostly decays between inputs
    # All inputs occur BEFORE the spike - no inputs after
    inputs = [
        # (time_ms, weight, type)
        # Phase 1: Small isolated inputs - show individual EPSPs/IPSPs
        (20, 0.08, 'E'),   # Small excitatory - ~3mV deflection
        (45, 0.06, 'E'),   # Another small excitatory
        (70, -0.05, 'I'),  # Small inhibitory
        (95, 0.07, 'E'),   # Excitatory
        # Phase 2: Inputs start arriving closer together - temporal summation
        (115, 0.08, 'E'),  # Building up
        (130, 0.09, 'E'),  # Another push (partial summation)
        (142, -0.06, 'I'), # Inhibitory dampening
        (155, 0.10, 'E'),  # Stronger push
        (162, 0.10, 'E'),  # Close together - summation (spike ~167ms)
        # No more inputs - spike occurs from accumulated excitation
    ]

    # Separate inputs for visualization
    excitatory_inputs = [(t, w) for t, w, typ in inputs if typ == 'E']
    inhibitory_inputs = [(t, w) for t, w, typ in inputs if typ == 'I']

    spike_time = None

    # Run simulation
    for t in time:
        # Apply inputs at their scheduled times
        for input_t, weight, input_type in inputs:
            if abs(t - input_t) < dt / 2:
                if input_type == 'E':
                    neuron.receive_spike(weight)
                else:
                    neuron.receive_spike(-abs(weight))  # Ensure inhibitory is negative

        spiked = neuron.update(dt)
        if spiked and spike_time is None:
            spike_time = t

    # Create figure with three panels
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1.2, 1], hspace=0.12)

    # --- Top panel: Membrane potential ---
    ax1 = fig.add_subplot(gs[0])

    # Plot membrane potential trace
    ax1.plot(time, neuron.v_history, color=COLORS['membrane'], linewidth=2,
             label='Membrane potential $V(t)$', zorder=3)

    # Reference lines
    ax1.axhline(neuron.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Threshold $V_{{th}}$ = {neuron.v_threshold} mV')
    ax1.axhline(neuron.v_rest, color=COLORS['rest'], linestyle=':',
                linewidth=1.5, alpha=0.7, label=f'Resting $V_{{rest}}$ = {neuron.v_rest} mV')
    ax1.axhline(neuron.v_reset, color=COLORS['reset'], linestyle='-.',
                linewidth=1, alpha=0.5, label=f'Reset $V_{{reset}}$ = {neuron.v_reset} mV')

    # Mark the action potential with a prominent red dotted vertical line
    if spike_time is not None:
        ax1.axvline(spike_time, color=COLORS['threshold'], linewidth=2.5, linestyle=':',
                    alpha=0.9, label=f'Action potential (t = {spike_time:.1f} ms)', zorder=5)
        # Add text label for the spike (no arrow)
        ax1.text(spike_time + 3, -52, 'AP', fontsize=11, fontweight='bold',
                 color=COLORS['threshold'], ha='left', va='center')

    # Shade phases
    if spike_time is not None:
        # Pre-spike integration phase
        ax1.axvspan(0, spike_time, alpha=0.05, color=COLORS['membrane'])
        # Post-spike recovery phase
        ax1.axvspan(spike_time, T, alpha=0.05, color=COLORS['rest'])

        # Phase annotations
        ax1.annotate('Subthreshold\nintegration', xy=(spike_time/2, -79),
                     fontsize=10, ha='center', va='top', color=COLORS['annotation'],
                     style='italic')
        ax1.annotate('Recovery\n(no inputs)', xy=(spike_time + (T - spike_time)/2, -79),
                     fontsize=10, ha='center', va='top', color=COLORS['annotation'],
                     style='italic')

    ax1.set_ylabel('Membrane Potential (mV)', fontsize=12)
    ax1.set_title('Subthreshold Synaptic Integration Leading to Action Potential',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-85, -50)
    ax1.tick_params(axis='x', labelbottom=False)  # Hide x-axis tick labels

    # --- Middle panel: Conductances ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Get conductance histories from the neuron
    g_e_hist = np.array(neuron.g_e_history)
    g_i_hist = np.array(neuron.g_i_history)

    # Plot excitatory and inhibitory conductances
    ax2.plot(time, g_e_hist, color=COLORS['excitatory'], linewidth=1.5,
             label='$g_e(t)$ (excitatory)', alpha=0.9)
    ax2.plot(time, g_i_hist, color=COLORS['inhibitory'], linewidth=1.5,
             label='$g_i(t)$ (inhibitory)', alpha=0.9)

    # Plot net conductance (g_e - g_i for visualization, though currents are what matter)
    # Actually, let's show the net synaptic drive: g_e - g_i scaled
    ax2.fill_between(time, 0, g_e_hist, color=COLORS['excitatory'], alpha=0.2)
    ax2.fill_between(time, 0, g_i_hist, color=COLORS['inhibitory'], alpha=0.2)

    # Mark spike time
    if spike_time is not None:
        ax2.axvline(spike_time, color=COLORS['threshold'], linewidth=2, linestyle=':', alpha=0.7)

    ax2.axhline(0, color=COLORS['annotation'], linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('Conductance', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(axis='x', labelbottom=False)  # Hide x-axis tick labels

    # --- Bottom panel: Input raster ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot excitatory inputs as upward stems
    for t_input, weight in excitatory_inputs:
        ax3.vlines(t_input, 0, weight, color=COLORS['excitatory'], linewidth=2.5)
        ax3.plot(t_input, weight, 'o', color=COLORS['excitatory'], markersize=6)

    # Plot inhibitory inputs as downward stems
    for t_input, weight in inhibitory_inputs:
        ax3.vlines(t_input, 0, -abs(weight), color=COLORS['inhibitory'], linewidth=2.5)
        ax3.plot(t_input, -abs(weight), 'o', color=COLORS['inhibitory'], markersize=6)

    # Mark spike time with dotted line (matching top panel)
    if spike_time is not None:
        ax3.axvline(spike_time, color=COLORS['threshold'], linewidth=2, linestyle=':', alpha=0.7)

    ax3.axhline(0, color=COLORS['annotation'], linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Input\nWeight', fontsize=10)
    ax3.set_ylim(-0.15, 0.15)
    ax3.grid(True, alpha=0.4)

    # Add legend for input types
    ax3.plot([], [], 'o-', color=COLORS['excitatory'], linewidth=2,
             label='Excitatory input', markersize=6)
    ax3.plot([], [], 'o-', color=COLORS['inhibitory'], linewidth=2,
             label='Inhibitory input', markersize=6)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)

    # Format x-axis tick labels with "ms" suffix
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)} ms'))

    plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.1, hspace=0.15)
    save_path = os.path.join(SAVE_DIR, "fig10_subthreshold_to_spike.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 11: THREE INTERACTING NEURONS (2 Excitatory, 1 Inhibitory)
# Demonstrates synaptic interactions in a small network
# ==============================================================================
def plot_three_neuron_interaction():
    """
    Visualizes the dynamics of 3 interacting neurons:
    - Neuron E1 (Excitatory): Receives external stimulation, projects to E2 and I
    - Neuron E2 (Excitatory): Receives input from E1, projects to I
    - Neuron I (Inhibitory): Receives input from E1 and E2, projects back to both

    This demonstrates:
    - Feedforward excitation (E1 → E2)
    - Feedback inhibition (I → E1, E2)
    - The balance of excitation and inhibition in neural circuits
    """
    print("\n--- Figure 11: Three Interacting Neurons (2E, 1I) ---")

    # Create 3 neurons with slightly different parameters for visual distinction
    neuron_params_base = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.3,  # Small noise for realism
        'i_noise_amp': 0.02,
    }

    # E1: First excitatory neuron (receives external drive)
    E1 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    # E2: Second excitatory neuron (driven by E1)
    E2 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    # I: Inhibitory neuron (receives from both E, inhibits both E)
    I = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': True})

    E1.reset()
    E2.reset()
    I.reset()

    dt = 0.1
    T = 300  # ms
    time = np.arange(0, T, dt)

    # Connection weights
    w_E1_to_E2 = 0.4   # E1 excites E2
    w_E1_to_I = 0.5    # E1 excites I
    w_E2_to_I = 0.4    # E2 excites I
    w_I_to_E1 = -0.6   # I inhibits E1
    w_I_to_E2 = -0.6   # I inhibits E2

    # Synaptic delay (in timesteps)
    delay_steps = int(1.0 / dt)  # 1 ms delay

    # Record spike times for each neuron
    E1_spikes = []
    E2_spikes = []
    I_spikes = []

    # External stimulation pattern for E1
    # Pulse of stimulation from 50-150ms, then 200-250ms
    def external_stim(t):
        if 50 <= t < 150:
            return 0.5  # Weaker stim so inhibition can act
        elif 200 <= t < 250:
            return 0.6
        return 0.0

    # Simulation loop
    for i, t in enumerate(time):
        # Apply external stimulation to E1
        stim = external_stim(t)
        if stim > 0:
            E1.receive_spike(stim * dt * 0.4)  # Scale for continuous input

        # Apply delayed synaptic connections based on past spikes
        # E1 → E2 and E1 → I
        if i >= delay_steps:
            past_idx = i - delay_steps
            past_t = time[past_idx]
            if past_t in E1_spikes:
                E2.receive_spike(w_E1_to_E2)
                I.receive_spike(w_E1_to_I)
            if past_t in E2_spikes:
                I.receive_spike(w_E2_to_I)
            if past_t in I_spikes:
                E1.receive_spike(w_I_to_E1)
                E2.receive_spike(w_I_to_E2)

        # Update all neurons
        if E1.update(dt):
            E1_spikes.append(t)
        if E2.update(dt):
            E2_spikes.append(t)
        if I.update(dt):
            I_spikes.append(t)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 2, 1.5],
                  width_ratios=[3, 1], hspace=0.3, wspace=0.25)

    # --- Left column: Membrane potential traces ---

    # Panel 1: E1 membrane potential
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, E1.v_history, color=COLORS['excitatory'], linewidth=1.5,
             label='$E_1$ (receives external input)')
    ax1.axhline(E1.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5)
    ax1.axhline(E1.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.5)

    # Mark spikes
    for spike_t in E1_spikes:
        ax1.axvline(spike_t, color=COLORS['excitatory'], linewidth=0.8, alpha=0.4)

    # Shade external stimulation periods
    ax1.axvspan(50, 150, alpha=0.1, color=COLORS['spike'], label='External stim')
    ax1.axvspan(200, 250, alpha=0.1, color=COLORS['spike'])

    ax1.set_ylabel('$V_{E1}$ (mV)')
    ax1.set_title('Excitatory Neuron $E_1$ (Driven)', fontsize=11, fontweight='bold',
                  color=COLORS['excitatory'])
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-85, -50)
    ax1.tick_params(axis='x', labelbottom=False)

    # Panel 2: E2 membrane potential
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, E2.v_history, color='#f97316', linewidth=1.5,  # Darker orange
             label='$E_2$ (driven by $E_1$)')
    ax2.axhline(E2.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5)
    ax2.axhline(E2.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.5)

    # Mark spikes
    for spike_t in E2_spikes:
        ax2.axvline(spike_t, color='#f97316', linewidth=0.8, alpha=0.4)

    ax2.set_ylabel('$V_{E2}$ (mV)')
    ax2.set_title('Excitatory Neuron $E_2$ (Feedforward)', fontsize=11, fontweight='bold',
                  color='#f97316')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(-85, -50)
    ax2.tick_params(axis='x', labelbottom=False)

    # Panel 3: I membrane potential
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(time, I.v_history, color=COLORS['inhibitory'], linewidth=1.5,
             label='$I$ (inhibits $E_1$ and $E_2$)')
    ax3.axhline(I.v_threshold, color=COLORS['threshold'], linestyle='--',
                linewidth=1, alpha=0.5)
    ax3.axhline(I.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.5)

    # Mark spikes
    for spike_t in I_spikes:
        ax3.axvline(spike_t, color=COLORS['inhibitory'], linewidth=0.8, alpha=0.4)

    ax3.set_ylabel('$V_I$ (mV)')
    ax3.set_title('Inhibitory Neuron $I$ (Feedback)', fontsize=11, fontweight='bold',
                  color=COLORS['inhibitory'])
    ax3.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(-85, -50)
    ax3.tick_params(axis='x', labelbottom=False)

    # Panel 4: Combined spike raster
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)

    # Plot spike rasters for all three neurons
    if E1_spikes:
        ax4.eventplot([E1_spikes], orientation='horizontal', lineoffsets=2.5,
                      linelengths=0.8, colors=[COLORS['excitatory']])
    if E2_spikes:
        ax4.eventplot([E2_spikes], orientation='horizontal', lineoffsets=1.5,
                      linelengths=0.8, colors=['#f97316'])
    if I_spikes:
        ax4.eventplot([I_spikes], orientation='horizontal', lineoffsets=0.5,
                      linelengths=0.8, colors=[COLORS['inhibitory']])

    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['$I$', '$E_2$', '$E_1$'])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Neuron')
    ax4.set_title(f'Spike Raster ($E_1$: {len(E1_spikes)}, $E_2$: {len(E2_spikes)}, $I$: {len(I_spikes)} spikes)',
                  fontsize=10)
    ax4.grid(True, alpha=0.4, axis='x')
    ax4.set_ylim(0, 3)

    # --- Right column: Network diagram ---
    ax_diagram = fig.add_subplot(gs[0:3, 1])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect('equal')
    ax_diagram.axis('off')
    ax_diagram.set_title('Network Connectivity', fontsize=11, fontweight='bold')

    # Neuron positions (triangle arrangement)
    pos_E1 = (0, 1.0)
    pos_E2 = (-0.8, -0.5)
    pos_I = (0.8, -0.5)

    # Draw neurons as circles
    circle_E1 = plt.Circle(pos_E1, 0.25, color=COLORS['excitatory'], ec='black', linewidth=2)
    circle_E2 = plt.Circle(pos_E2, 0.25, color='#f97316', ec='black', linewidth=2)
    circle_I = plt.Circle(pos_I, 0.25, color=COLORS['inhibitory'], ec='black', linewidth=2)

    ax_diagram.add_patch(circle_E1)
    ax_diagram.add_patch(circle_E2)
    ax_diagram.add_patch(circle_I)

    # Neuron labels
    ax_diagram.text(pos_E1[0], pos_E1[1], '$E_1$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_E2[0], pos_E2[1], '$E_2$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_I[0], pos_I[1], '$I$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

    # Draw connections with arrows
    # Arrow style for excitatory (pointed) and inhibitory (ball)
    from matplotlib.patches import FancyArrowPatch

    def draw_connection(start, end, color, inhibitory=False, curve=0):
        """Draw a curved arrow between two points."""
        style = 'Simple, tail_width=2, head_width=8, head_length=6'
        if inhibitory:
            style = 'Simple, tail_width=2, head_width=0, head_length=0'

        # Calculate direction and offset for arrow start/end
        import math
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)

        # Offset by circle radius
        offset = 0.28
        start_offset = (start[0] + offset * dx/dist, start[1] + offset * dy/dist)
        end_offset = (end[0] - offset * dx/dist, end[1] - offset * dy/dist)

        arrow = FancyArrowPatch(start_offset, end_offset,
                                connectionstyle=f"arc3,rad={curve}",
                                arrowstyle=style, color=color, linewidth=2,
                                mutation_scale=1)
        ax_diagram.add_patch(arrow)

        # For inhibitory, add a circle at the end
        if inhibitory:
            ax_diagram.plot(end_offset[0], end_offset[1], 'o', color=color, markersize=8)

    # Excitatory connections (arrows)
    draw_connection(pos_E1, pos_E2, COLORS['excitatory'], curve=0.2)  # E1 → E2
    draw_connection(pos_E1, pos_I, COLORS['excitatory'], curve=-0.2)  # E1 → I
    draw_connection(pos_E2, pos_I, COLORS['excitatory'], curve=0.2)   # E2 → I

    # Inhibitory connections (with ball ends)
    draw_connection(pos_I, pos_E1, COLORS['inhibitory'], inhibitory=True, curve=0.2)  # I → E1
    draw_connection(pos_I, pos_E2, COLORS['inhibitory'], inhibitory=True, curve=-0.2) # I → E2

    # External input arrow to E1
    ax_diagram.annotate('', xy=(pos_E1[0], pos_E1[1] + 0.25),
                        xytext=(pos_E1[0], pos_E1[1] + 0.7),
                        arrowprops=dict(arrowstyle='->', color=COLORS['spike'], lw=2))
    ax_diagram.text(pos_E1[0], pos_E1[1] + 0.85, 'External\nInput', ha='center',
                    fontsize=9, color=COLORS['spike'])

    # Legend for connection types
    ax_diagram.plot([], [], '-', color=COLORS['excitatory'], linewidth=2,
                    label='Excitatory ($\\rightarrow$)')
    ax_diagram.plot([], [], 'o-', color=COLORS['inhibitory'], linewidth=2,
                    label='Inhibitory ($\\dashv$)')
    ax_diagram.legend(loc='lower center', fontsize=8, framealpha=0.9)

    # --- Bottom right: Connection weight table ---
    ax_table = fig.add_subplot(gs[3, 1])
    ax_table.axis('off')

    table_text = (
        "Connection Weights:\n\n"
        f"$E_1 \\rightarrow E_2$: {w_E1_to_E2}\n"
        f"$E_1 \\rightarrow I$: {w_E1_to_I}\n"
        f"$E_2 \\rightarrow I$: {w_E2_to_I}\n"
        f"$I \\dashv E_1$: {w_I_to_E1}\n"
        f"$I \\dashv E_2$: {w_I_to_E2}"
    )
    ax_table.text(0.5, 0.5, table_text, ha='center', va='center', fontsize=9,
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig11_three_neuron_interaction.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 12: FEEDFORWARD INHIBITION (Gating Circuit)
# E1 drives both E2 and I; I inhibits E2 creating a temporal race
# ==============================================================================
def plot_feedforward_inhibition():
    """
    Feedforward Inhibition Circuit:
    - E1 receives external input and drives both E2 and I
    - I inhibits E2
    - Creates a temporal race: E1's excitation vs I's delayed inhibition at E2
    - E2 only fires briefly before I catches up and suppresses it

    This demonstrates how inhibition can gate/control excitatory transmission.
    """
    print("\n--- Figure 12: Feedforward Inhibition (Gating) ---")

    neuron_params_base = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.2,
        'i_noise_amp': 0.01,
        'adaptation_increment': 0.05,  # Reduced adaptation
    }

    E1 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    E2 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    I = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': True})

    E1.reset()
    E2.reset()
    I.reset()

    dt = 0.1
    T = 400  # ms
    time = np.arange(0, T, dt)

    # Connection weights - strong inhibition to make effect obvious
    w_E1_to_E2 = 0.6   # E1 excites E2
    w_E1_to_I = 0.5    # E1 excites I
    w_I_to_E2 = -1.2   # I strongly inhibits E2

    # Delays - I pathway slightly slower to show the race
    delay_E1_E2 = int(1.0 / dt)   # 1 ms - fast direct path
    delay_E1_I = int(1.0 / dt)    # 1 ms
    delay_I_E2 = int(2.0 / dt)    # 2 ms - inhibition arrives later

    E1_spikes = []
    E2_spikes = []
    I_spikes = []

    # Continuous stimulation to E1 during specific windows
    def external_stim(t):
        if 50 <= t < 350:
            return 1.2
        return 0.0

    for i, t in enumerate(time):
        stim = external_stim(t)
        if stim > 0:
            E1.receive_spike(stim * dt * 0.4)

        # E1 → E2 (fast direct excitation)
        if i >= delay_E1_E2:
            past_t = time[i - delay_E1_E2]
            if past_t in E1_spikes:
                E2.receive_spike(w_E1_to_E2)

        # E1 → I
        if i >= delay_E1_I:
            past_t = time[i - delay_E1_I]
            if past_t in E1_spikes:
                I.receive_spike(w_E1_to_I)

        # I → E2 (delayed inhibition)
        if i >= delay_I_E2:
            past_t = time[i - delay_I_E2]
            if past_t in I_spikes:
                E2.receive_spike(w_I_to_E2)

        if E1.update(dt):
            E1_spikes.append(t)
        if E2.update(dt):
            E2_spikes.append(t)
        if I.update(dt):
            I_spikes.append(t)

    # Create figure
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 2, 1.2],
                  width_ratios=[3, 1], hspace=0.3, wspace=0.25)

    # Panel 1: E1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, E1.v_history, color=COLORS['excitatory'], linewidth=1.5)
    ax1.axhline(E1.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in E1_spikes:
        ax1.axvline(spike_t, color=COLORS['excitatory'], linewidth=0.5, alpha=0.3)
    ax1.axvspan(50, 350, alpha=0.1, color=COLORS['spike'], label='External input')
    ax1.set_ylabel('$V_{E1}$ (mV)')
    ax1.set_title('$E_1$: Driver Neuron (receives input)', fontsize=11, fontweight='bold',
                  color=COLORS['excitatory'])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-85, -50)
    ax1.tick_params(axis='x', labelbottom=False)

    # Panel 2: I
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, I.v_history, color=COLORS['inhibitory'], linewidth=1.5)
    ax2.axhline(I.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in I_spikes:
        ax2.axvline(spike_t, color=COLORS['inhibitory'], linewidth=0.5, alpha=0.3)
    ax2.set_ylabel('$V_I$ (mV)')
    ax2.set_title('$I$: Inhibitory Interneuron (driven by $E_1$)', fontsize=11, fontweight='bold',
                  color=COLORS['inhibitory'])
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(-85, -50)
    ax2.tick_params(axis='x', labelbottom=False)

    # Panel 3: E2 - the gated neuron
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(time, E2.v_history, color='#f97316', linewidth=1.5)
    ax3.axhline(E2.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(E2.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.5)
    for spike_t in E2_spikes:
        ax3.axvline(spike_t, color='#f97316', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('$V_{E2}$ (mV)')
    ax3.set_title('$E_2$: Gated Neuron (excitation from $E_1$, inhibition from $I$)',
                  fontsize=11, fontweight='bold', color='#f97316')
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(-85, -50)
    ax3.tick_params(axis='x', labelbottom=False)

    # Add annotation showing the gating effect
    ax3.annotate('Inhibition gates\nexcitatory drive', xy=(200, -75), fontsize=9,
                 ha='center', color=COLORS['inhibitory'], fontweight='bold')

    # Panel 4: Spike raster
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if E1_spikes:
        ax4.eventplot([E1_spikes], lineoffsets=2.5, linelengths=0.8, colors=[COLORS['excitatory']])
    if I_spikes:
        ax4.eventplot([I_spikes], lineoffsets=1.5, linelengths=0.8, colors=[COLORS['inhibitory']])
    if E2_spikes:
        ax4.eventplot([E2_spikes], lineoffsets=0.5, linelengths=0.8, colors=['#f97316'])
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['$E_2$', '$I$', '$E_1$'])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Neuron')
    ax4.set_title(f'Spike Raster: $E_1$={len(E1_spikes)}, $I$={len(I_spikes)}, $E_2$={len(E2_spikes)} spikes',
                  fontsize=10)
    ax4.grid(True, alpha=0.4, axis='x')
    ax4.set_ylim(0, 3)

    # Network diagram
    ax_diagram = fig.add_subplot(gs[0:3, 1])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect('equal')
    ax_diagram.axis('off')
    ax_diagram.set_title('Feedforward Inhibition\nCircuit', fontsize=11, fontweight='bold')

    # Positions - vertical arrangement
    pos_E1 = (0, 1.0)
    pos_I = (0.7, 0)
    pos_E2 = (0, -1.0)

    # Draw neurons
    circle_E1 = plt.Circle(pos_E1, 0.25, color=COLORS['excitatory'], ec='black', linewidth=2)
    circle_I = plt.Circle(pos_I, 0.25, color=COLORS['inhibitory'], ec='black', linewidth=2)
    circle_E2 = plt.Circle(pos_E2, 0.25, color='#f97316', ec='black', linewidth=2)
    ax_diagram.add_patch(circle_E1)
    ax_diagram.add_patch(circle_I)
    ax_diagram.add_patch(circle_E2)

    ax_diagram.text(pos_E1[0], pos_E1[1], '$E_1$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_I[0], pos_I[1], '$I$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_E2[0], pos_E2[1], '$E_2$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

    # Connections
    from matplotlib.patches import FancyArrowPatch
    import math

    def draw_arrow(start, end, color, inhibitory=False, curve=0):
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)
        offset = 0.28
        s = (start[0] + offset*dx/dist, start[1] + offset*dy/dist)
        e = (end[0] - offset*dx/dist, end[1] - offset*dy/dist)
        style = 'Simple, tail_width=2, head_width=8, head_length=6'
        if inhibitory:
            style = 'Simple, tail_width=2, head_width=0, head_length=0'
        arrow = FancyArrowPatch(s, e, connectionstyle=f"arc3,rad={curve}",
                                arrowstyle=style, color=color, linewidth=2)
        ax_diagram.add_patch(arrow)
        if inhibitory:
            ax_diagram.plot(e[0], e[1], 'o', color=color, markersize=8)

    draw_arrow(pos_E1, pos_E2, COLORS['excitatory'], curve=-0.3)  # E1 → E2 (direct)
    draw_arrow(pos_E1, pos_I, COLORS['excitatory'], curve=0)      # E1 → I
    draw_arrow(pos_I, pos_E2, COLORS['inhibitory'], inhibitory=True, curve=0)  # I → E2

    # Labels
    ax_diagram.text(-0.5, 0.2, 'Fast', fontsize=8, color=COLORS['excitatory'])
    ax_diagram.text(0.5, 0.6, 'Fast', fontsize=8, color=COLORS['excitatory'])
    ax_diagram.text(0.85, -0.5, 'Slow', fontsize=8, color=COLORS['inhibitory'])

    # External input
    ax_diagram.annotate('', xy=(pos_E1[0], pos_E1[1] + 0.25),
                        xytext=(pos_E1[0], pos_E1[1] + 0.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['spike'], lw=2))
    ax_diagram.text(pos_E1[0], pos_E1[1] + 0.75, 'Input', ha='center', fontsize=9, color=COLORS['spike'])

    # Description
    ax_table = fig.add_subplot(gs[3, 1])
    ax_table.axis('off')
    desc = ("Feedforward Inhibition:\n\n"
            "• $E_1$ drives both $E_2$ and $I$\n"
            "• $I$ inhibits $E_2$\n"
            "• Inhibition arrives later,\n"
            "  creating temporal gating")
    ax_table.text(0.5, 0.5, desc, ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig12_feedforward_inhibition.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 13: LATERAL INHIBITION (Winner-Take-All Competition)
# Two excitatory neurons compete via shared inhibitory neuron
# ==============================================================================
def plot_lateral_inhibition():
    """
    Lateral Inhibition Circuit:
    - E1 and E2 both receive external input (E1 gets stronger input)
    - Both excite shared inhibitory neuron I
    - I inhibits both E1 and E2
    - Creates winner-take-all: stronger neuron suppresses weaker one
    """
    print("\n--- Figure 13: Lateral Inhibition (Winner-Take-All) ---")

    neuron_params_base = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.2,
        'i_noise_amp': 0.01,
        'adaptation_increment': 0.05,
    }

    E1 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    E2 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    I = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': True})

    E1.reset()
    E2.reset()
    I.reset()

    dt = 0.1
    T = 400  # ms
    time = np.arange(0, T, dt)

    # Connection weights
    w_E1_to_I = 0.5
    w_E2_to_I = 0.5
    w_I_to_E1 = -0.8
    w_I_to_E2 = -0.8

    delay_steps = int(1.0 / dt)

    E1_spikes = []
    E2_spikes = []
    I_spikes = []

    # E1 gets stronger input than E2 - winner should be E1
    def stim_E1(t):
        if 50 <= t < 350:
            return 1.0  # Stronger
        return 0.0

    def stim_E2(t):
        if 50 <= t < 350:
            return 0.7  # Weaker
        return 0.0

    for i, t in enumerate(time):
        s1 = stim_E1(t)
        s2 = stim_E2(t)
        if s1 > 0:
            E1.receive_spike(s1 * dt * 0.5)
        if s2 > 0:
            E2.receive_spike(s2 * dt * 0.5)

        if i >= delay_steps:
            past_t = time[i - delay_steps]
            if past_t in E1_spikes:
                I.receive_spike(w_E1_to_I)
            if past_t in E2_spikes:
                I.receive_spike(w_E2_to_I)
            if past_t in I_spikes:
                E1.receive_spike(w_I_to_E1)
                E2.receive_spike(w_I_to_E2)

        if E1.update(dt):
            E1_spikes.append(t)
        if E2.update(dt):
            E2_spikes.append(t)
        if I.update(dt):
            I_spikes.append(t)

    # Create figure
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 2, 1.2],
                  width_ratios=[3, 1], hspace=0.3, wspace=0.25)

    # Panel 1: E1 (winner)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, E1.v_history, color=COLORS['excitatory'], linewidth=1.5)
    ax1.axhline(E1.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in E1_spikes:
        ax1.axvline(spike_t, color=COLORS['excitatory'], linewidth=0.5, alpha=0.3)
    ax1.axvspan(50, 350, alpha=0.15, color=COLORS['excitatory'], label='Strong input')
    ax1.set_ylabel('$V_{E1}$ (mV)')
    ax1.set_title('$E_1$: Winner (stronger input)', fontsize=11, fontweight='bold',
                  color=COLORS['excitatory'])
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-85, -50)
    ax1.tick_params(axis='x', labelbottom=False)

    # Panel 2: E2 (loser)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, E2.v_history, color='#f97316', linewidth=1.5)
    ax2.axhline(E2.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in E2_spikes:
        ax2.axvline(spike_t, color='#f97316', linewidth=0.5, alpha=0.3)
    ax2.axvspan(50, 350, alpha=0.1, color='#f97316', label='Weak input')
    ax2.set_ylabel('$V_{E2}$ (mV)')
    ax2.set_title('$E_2$: Suppressed (weaker input)', fontsize=11, fontweight='bold',
                  color='#f97316')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(-85, -50)
    ax2.tick_params(axis='x', labelbottom=False)

    # Panel 3: I
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(time, I.v_history, color=COLORS['inhibitory'], linewidth=1.5)
    ax3.axhline(I.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in I_spikes:
        ax3.axvline(spike_t, color=COLORS['inhibitory'], linewidth=0.5, alpha=0.3)
    ax3.set_ylabel('$V_I$ (mV)')
    ax3.set_title('$I$: Shared Inhibitory Neuron', fontsize=11, fontweight='bold',
                  color=COLORS['inhibitory'])
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(-85, -50)
    ax3.tick_params(axis='x', labelbottom=False)

    # Panel 4: Spike raster
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if E1_spikes:
        ax4.eventplot([E1_spikes], lineoffsets=2.5, linelengths=0.8, colors=[COLORS['excitatory']])
    if E2_spikes:
        ax4.eventplot([E2_spikes], lineoffsets=1.5, linelengths=0.8, colors=['#f97316'])
    if I_spikes:
        ax4.eventplot([I_spikes], lineoffsets=0.5, linelengths=0.8, colors=[COLORS['inhibitory']])
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['$I$', '$E_2$', '$E_1$'])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Neuron')
    ax4.set_title(f'Winner-Take-All: $E_1$={len(E1_spikes)} vs $E_2$={len(E2_spikes)} spikes',
                  fontsize=10)
    ax4.grid(True, alpha=0.4, axis='x')
    ax4.set_ylim(0, 3)

    # Network diagram
    ax_diagram = fig.add_subplot(gs[0:3, 1])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect('equal')
    ax_diagram.axis('off')
    ax_diagram.set_title('Lateral Inhibition\nCircuit', fontsize=11, fontweight='bold')

    # Positions - horizontal with I in middle
    pos_E1 = (-0.8, 0.5)
    pos_E2 = (0.8, 0.5)
    pos_I = (0, -0.5)

    circle_E1 = plt.Circle(pos_E1, 0.25, color=COLORS['excitatory'], ec='black', linewidth=2)
    circle_E2 = plt.Circle(pos_E2, 0.25, color='#f97316', ec='black', linewidth=2)
    circle_I = plt.Circle(pos_I, 0.25, color=COLORS['inhibitory'], ec='black', linewidth=2)
    ax_diagram.add_patch(circle_E1)
    ax_diagram.add_patch(circle_E2)
    ax_diagram.add_patch(circle_I)

    ax_diagram.text(pos_E1[0], pos_E1[1], '$E_1$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_E2[0], pos_E2[1], '$E_2$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_I[0], pos_I[1], '$I$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

    from matplotlib.patches import FancyArrowPatch
    import math

    def draw_arrow(start, end, color, inhibitory=False, curve=0):
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)
        offset = 0.28
        s = (start[0] + offset*dx/dist, start[1] + offset*dy/dist)
        e = (end[0] - offset*dx/dist, end[1] - offset*dy/dist)
        style = 'Simple, tail_width=2, head_width=8, head_length=6'
        if inhibitory:
            style = 'Simple, tail_width=2, head_width=0, head_length=0'
        arrow = FancyArrowPatch(s, e, connectionstyle=f"arc3,rad={curve}",
                                arrowstyle=style, color=color, linewidth=2)
        ax_diagram.add_patch(arrow)
        if inhibitory:
            ax_diagram.plot(e[0], e[1], 'o', color=color, markersize=8)

    # E1 and E2 both excite I
    draw_arrow(pos_E1, pos_I, COLORS['excitatory'], curve=0.2)
    draw_arrow(pos_E2, pos_I, COLORS['excitatory'], curve=-0.2)
    # I inhibits both
    draw_arrow(pos_I, pos_E1, COLORS['inhibitory'], inhibitory=True, curve=0.2)
    draw_arrow(pos_I, pos_E2, COLORS['inhibitory'], inhibitory=True, curve=-0.2)

    # External inputs
    ax_diagram.annotate('', xy=(pos_E1[0], pos_E1[1] + 0.25),
                        xytext=(pos_E1[0], pos_E1[1] + 0.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['spike'], lw=3))
    ax_diagram.text(pos_E1[0], pos_E1[1] + 0.75, 'Strong', ha='center', fontsize=9, color=COLORS['spike'])

    ax_diagram.annotate('', xy=(pos_E2[0], pos_E2[1] + 0.25),
                        xytext=(pos_E2[0], pos_E2[1] + 0.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['spike'], lw=1.5))
    ax_diagram.text(pos_E2[0], pos_E2[1] + 0.75, 'Weak', ha='center', fontsize=9, color=COLORS['spike'])

    # Description
    ax_table = fig.add_subplot(gs[3, 1])
    ax_table.axis('off')
    desc = ("Lateral Inhibition:\n\n"
            "• Both $E_1$ and $E_2$ excite $I$\n"
            "• $I$ inhibits both equally\n"
            "• Stronger neuron wins,\n"
            "  weaker is suppressed")
    ax_table.text(0.5, 0.5, desc, ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig13_lateral_inhibition.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 14: DISINHIBITION (Inhibition Silences Activity)
# Clear before/after showing inhibition stopping a firing neuron
# ==============================================================================
def plot_disinhibition():
    """
    Simple Inhibition Demo:
    - E1 continuously excites E2, causing steady firing
    - At a specific time, I receives strong input and inhibits E2
    - Clear before/after effect: E2 firing → E2 silenced
    """
    print("\n--- Figure 14: Inhibition Silences Activity ---")

    neuron_params_base = {
        **NEURON_PARAMS,
        'v_noise_amp': 0.2,
        'i_noise_amp': 0.01,
        'adaptation_increment': 0.03,
    }

    E1 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    E2 = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': False})
    I = Layered_LIFNeuronWithReversal(**{**neuron_params_base, 'is_inhibitory': True})

    E1.reset()
    E2.reset()
    I.reset()

    dt = 0.1
    T = 500  # ms
    time = np.arange(0, T, dt)

    # Weights
    w_E1_to_E2 = 0.5
    w_I_to_E2 = -1.5  # Strong inhibition

    delay_steps = int(1.0 / dt)

    E1_spikes = []
    E2_spikes = []
    I_spikes = []

    # E1 fires continuously
    def stim_E1(t):
        if 30 <= t < 450:
            return 1.0
        return 0.0

    # I receives strong input only during "inhibition ON" period
    inhibition_start = 200
    inhibition_end = 350

    def stim_I(t):
        if inhibition_start <= t < inhibition_end:
            return 1.5  # Strong drive to I
        return 0.0

    for i, t in enumerate(time):
        s1 = stim_E1(t)
        sI = stim_I(t)
        if s1 > 0:
            E1.receive_spike(s1 * dt * 0.5)
        if sI > 0:
            I.receive_spike(sI * dt * 0.5)

        if i >= delay_steps:
            past_t = time[i - delay_steps]
            if past_t in E1_spikes:
                E2.receive_spike(w_E1_to_E2)
            if past_t in I_spikes:
                E2.receive_spike(w_I_to_E2)

        if E1.update(dt):
            E1_spikes.append(t)
        if E2.update(dt):
            E2_spikes.append(t)
        if I.update(dt):
            I_spikes.append(t)

    # Create figure
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 2, 1.2],
                  width_ratios=[3, 1], hspace=0.3, wspace=0.25)

    # Panel 1: E1 (constant driver)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, E1.v_history, color=COLORS['excitatory'], linewidth=1.5)
    ax1.axhline(E1.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in E1_spikes:
        ax1.axvline(spike_t, color=COLORS['excitatory'], linewidth=0.5, alpha=0.3)
    ax1.axvspan(30, 450, alpha=0.1, color=COLORS['spike'])
    ax1.set_ylabel('$V_{E1}$ (mV)')
    ax1.set_title('$E_1$: Constant Excitatory Drive', fontsize=11, fontweight='bold',
                  color=COLORS['excitatory'])
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-85, -50)
    ax1.tick_params(axis='x', labelbottom=False)

    # Panel 2: I (inhibitor - activated in middle)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(time, I.v_history, color=COLORS['inhibitory'], linewidth=1.5)
    ax2.axhline(I.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    for spike_t in I_spikes:
        ax2.axvline(spike_t, color=COLORS['inhibitory'], linewidth=0.5, alpha=0.3)
    ax2.axvspan(inhibition_start, inhibition_end, alpha=0.2, color=COLORS['inhibitory'],
                label='Inhibition ON')
    ax2.set_ylabel('$V_I$ (mV)')
    ax2.set_title('$I$: Inhibitory Neuron (activated 200-350 ms)', fontsize=11, fontweight='bold',
                  color=COLORS['inhibitory'])
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.4)
    ax2.set_ylim(-85, -50)
    ax2.tick_params(axis='x', labelbottom=False)

    # Panel 3: E2 (target - silenced by inhibition)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(time, E2.v_history, color='#f97316', linewidth=1.5)
    ax3.axhline(E2.v_threshold, color=COLORS['threshold'], linestyle='--', linewidth=1, alpha=0.5)
    ax3.axhline(E2.v_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.5)
    for spike_t in E2_spikes:
        ax3.axvline(spike_t, color='#f97316', linewidth=0.8, alpha=0.5)
    ax3.axvspan(inhibition_start, inhibition_end, alpha=0.2, color=COLORS['inhibitory'])
    ax3.set_ylabel('$V_{E2}$ (mV)')
    ax3.set_title('$E_2$: Target Neuron (silenced by inhibition)', fontsize=11, fontweight='bold',
                  color='#f97316')
    ax3.grid(True, alpha=0.4)
    ax3.set_ylim(-85, -50)
    ax3.tick_params(axis='x', labelbottom=False)

    # Annotations showing phases
    ax3.annotate('Firing', xy=(100, -52), fontsize=10, ha='center',
                 color=COLORS['spike'], fontweight='bold')
    ax3.annotate('SILENCED', xy=(275, -52), fontsize=10, ha='center',
                 color=COLORS['inhibitory'], fontweight='bold')
    ax3.annotate('Firing\nresumes', xy=(400, -52), fontsize=10, ha='center',
                 color=COLORS['spike'], fontweight='bold')

    # Panel 4: Spike raster
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    if E1_spikes:
        ax4.eventplot([E1_spikes], lineoffsets=2.5, linelengths=0.8, colors=[COLORS['excitatory']])
    if I_spikes:
        ax4.eventplot([I_spikes], lineoffsets=1.5, linelengths=0.8, colors=[COLORS['inhibitory']])
    if E2_spikes:
        ax4.eventplot([E2_spikes], lineoffsets=0.5, linelengths=0.8, colors=['#f97316'])
    ax4.axvspan(inhibition_start, inhibition_end, alpha=0.2, color=COLORS['inhibitory'])
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['$E_2$', '$I$', '$E_1$'])
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Neuron')

    # Count spikes in each phase for E2
    e2_before = len([s for s in E2_spikes if s < inhibition_start])
    e2_during = len([s for s in E2_spikes if inhibition_start <= s < inhibition_end])
    e2_after = len([s for s in E2_spikes if s >= inhibition_end])
    ax4.set_title(f'$E_2$ spikes: Before={e2_before}, During={e2_during}, After={e2_after}',
                  fontsize=10)
    ax4.grid(True, alpha=0.4, axis='x')
    ax4.set_ylim(0, 3)

    # Network diagram
    ax_diagram = fig.add_subplot(gs[0:3, 1])
    ax_diagram.set_xlim(-1.5, 1.5)
    ax_diagram.set_ylim(-1.5, 1.5)
    ax_diagram.set_aspect('equal')
    ax_diagram.axis('off')
    ax_diagram.set_title('Inhibition Silences\nActivity', fontsize=11, fontweight='bold')

    # Simple vertical layout
    pos_E1 = (-0.5, 0.8)
    pos_I = (0.5, 0.8)
    pos_E2 = (0, -0.5)

    circle_E1 = plt.Circle(pos_E1, 0.25, color=COLORS['excitatory'], ec='black', linewidth=2)
    circle_I = plt.Circle(pos_I, 0.25, color=COLORS['inhibitory'], ec='black', linewidth=2)
    circle_E2 = plt.Circle(pos_E2, 0.25, color='#f97316', ec='black', linewidth=2)
    ax_diagram.add_patch(circle_E1)
    ax_diagram.add_patch(circle_I)
    ax_diagram.add_patch(circle_E2)

    ax_diagram.text(pos_E1[0], pos_E1[1], '$E_1$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_I[0], pos_I[1], '$I$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
    ax_diagram.text(pos_E2[0], pos_E2[1], '$E_2$', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')

    from matplotlib.patches import FancyArrowPatch
    import math

    def draw_arrow(start, end, color, inhibitory=False, curve=0):
        dx, dy = end[0] - start[0], end[1] - start[1]
        dist = math.sqrt(dx**2 + dy**2)
        offset = 0.28
        s = (start[0] + offset*dx/dist, start[1] + offset*dy/dist)
        e = (end[0] - offset*dx/dist, end[1] - offset*dy/dist)
        style = 'Simple, tail_width=2, head_width=8, head_length=6'
        if inhibitory:
            style = 'Simple, tail_width=2, head_width=0, head_length=0'
        arrow = FancyArrowPatch(s, e, connectionstyle=f"arc3,rad={curve}",
                                arrowstyle=style, color=color, linewidth=2)
        ax_diagram.add_patch(arrow)
        if inhibitory:
            ax_diagram.plot(e[0], e[1], 'o', color=color, markersize=8)

    draw_arrow(pos_E1, pos_E2, COLORS['excitatory'], curve=0.2)  # E1 → E2
    draw_arrow(pos_I, pos_E2, COLORS['inhibitory'], inhibitory=True, curve=-0.2)  # I → E2

    # External inputs
    ax_diagram.annotate('', xy=(pos_E1[0], pos_E1[1] + 0.25),
                        xytext=(pos_E1[0], pos_E1[1] + 0.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['spike'], lw=2))
    ax_diagram.text(pos_E1[0], pos_E1[1] + 0.75, 'Constant', ha='center', fontsize=8, color=COLORS['spike'])

    ax_diagram.annotate('', xy=(pos_I[0], pos_I[1] + 0.25),
                        xytext=(pos_I[0], pos_I[1] + 0.6),
                        arrowprops=dict(arrowstyle='->', color=COLORS['inhibitory'], lw=2))
    ax_diagram.text(pos_I[0], pos_I[1] + 0.75, 'Pulse', ha='center', fontsize=8, color=COLORS['inhibitory'])

    # Description
    ax_table = fig.add_subplot(gs[3, 1])
    ax_table.axis('off')
    desc = ("Inhibition Effect:\n\n"
            "• $E_1$ constantly drives $E_2$\n"
            "• $I$ activated 200-350 ms\n"
            "• $E_2$ completely silenced\n"
            "• Activity resumes after")
    ax_table.text(0.5, 0.5, desc, ha='center', va='center', fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig14_inhibition_silences.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 15: CONDUCTANCE PRODUCTION, DECAY, AND SUMMATION
# Demonstrates synaptic conductance dynamics with temporal summation
# ==============================================================================
def plot_conductance_summation():
    """
    Visualizes the production and decay of excitatory and inhibitory conductances,
    demonstrating temporal summation when two inputs arrive in sequence.

    Panel 1: Excitatory conductance g_e with two stimulations showing summation
    Panel 2: Inhibitory conductance g_i with two stimulations showing summation
    Panel 3: Net synaptic current I_syn = g_e(E_e - V) + g_i(E_i - V)
    """
    print("\n--- Figure 15: Conductance Production, Decay, and Summation ---")

    dt = 0.1
    T = 50  # ms - shorter to make waves more pronounced
    time = np.arange(0, T, dt)

    tau_e = NEURON_PARAMS['tau_e']  # 3 ms
    tau_i = NEURON_PARAMS['tau_i']  # 7 ms
    E_e = NEURON_PARAMS['e_reversal']  # 0 mV
    E_i = NEURON_PARAMS['i_reversal']  # -80 mV
    V_rest = NEURON_PARAMS['v_rest']   # -70 mV

    # Timing of the two stimulations - slight offset between exc and inh
    # Excitatory stimulations
    stim_e1_time = 5.0
    stim_e2_time = 11.0  # Second exc arrives during decay of first

    # Inhibitory stimulations (slightly offset from excitatory to see interaction)
    stim_i1_time = 7.0   # Arrives shortly after first exc
    stim_i2_time = 15.0  # Second inh arrives during decay

    # Weight magnitudes for the stimulations (same magnitude)
    w_e = 0.8   # Excitatory weight
    w_i = 0.8   # Inhibitory weight (same magnitude)

    # --- Create neurons and run simulations ---
    # Neuron receiving excitatory inputs
    neuron_e = Layered_LIFNeuronWithReversal(**{**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0})
    neuron_e.reset()

    # Neuron receiving inhibitory inputs
    neuron_i = Layered_LIFNeuronWithReversal(**{**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0})
    neuron_i.reset()

    # Neuron receiving both (for I_syn panel)
    neuron_both = Layered_LIFNeuronWithReversal(**{**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0})
    neuron_both.reset()

    # Record I_syn components
    i_e_history = []
    i_i_history = []
    i_syn_history = []
    v_history_both = []

    for t in time:
        # Apply excitatory stimulations to neuron_e
        if abs(t - stim_e1_time) < dt / 2 or abs(t - stim_e2_time) < dt / 2:
            neuron_e.receive_spike(w_e)

        # Apply inhibitory stimulations to neuron_i
        if abs(t - stim_i1_time) < dt / 2 or abs(t - stim_i2_time) < dt / 2:
            neuron_i.receive_spike(-w_i)

        # Apply both to neuron_both for I_syn calculation
        if abs(t - stim_e1_time) < dt / 2 or abs(t - stim_e2_time) < dt / 2:
            neuron_both.receive_spike(w_e)
        if abs(t - stim_i1_time) < dt / 2 or abs(t - stim_i2_time) < dt / 2:
            neuron_both.receive_spike(-w_i)

        # Calculate currents before update (using current V)
        i_e = neuron_both.g_e * (E_e - neuron_both.v)
        i_i = neuron_both.g_i * (E_i - neuron_both.v)
        i_syn = i_e + i_i

        i_e_history.append(i_e)
        i_i_history.append(i_i)
        i_syn_history.append(i_syn)
        v_history_both.append(neuron_both.v)

        # Update all neurons
        neuron_e.update(dt)
        neuron_i.update(dt)
        neuron_both.update(dt)

    # --- Create figure with 3 panels ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # =========================================================================
    # Panel 1: Excitatory Conductance
    # =========================================================================
    ax1 = axes[0]

    ax1.plot(time, neuron_e.g_e_history, color=COLORS['excitatory'], linewidth=2.5,
             label=f'$g_e(t)$ ($\\tau_e$ = {tau_e} ms)')

    # Mark excitatory stimulation times
    ax1.axvline(stim_e1_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(stim_e2_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('Excitatory Conductance $g_e$', fontsize=12)
    ax1.set_title('Excitatory Conductance: Production, Decay, and Temporal Summation', fontsize=14)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.5)
    ax1.set_ylim(0, None)
    ax1.tick_params(axis='both', labelsize=11)

    # =========================================================================
    # Panel 2: Inhibitory Conductance
    # =========================================================================
    ax2 = axes[1]

    ax2.plot(time, neuron_i.g_i_history, color=COLORS['inhibitory'], linewidth=2.5,
             label=f'$g_i(t)$ ($\\tau_i$ = {tau_i} ms)')

    # Mark inhibitory stimulation times
    ax2.axvline(stim_i1_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(stim_i2_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('Inhibitory Conductance $g_i$', fontsize=12)
    ax2.set_title('Inhibitory Conductance: Slower Decay Reflects GABA Dynamics', fontsize=14)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.5)
    ax2.set_ylim(0, None)
    ax2.tick_params(axis='both', labelsize=11)

    # =========================================================================
    # Panel 3: Net Synaptic Current
    # =========================================================================
    ax3 = axes[2]

    # Plot I_e and I_i separately (stacked visualization)
    ax3.fill_between(time, 0, i_e_history, alpha=0.3, color=COLORS['excitatory'],
                     label=f'$I_e = g_e(E_e - V)$')
    ax3.fill_between(time, 0, i_i_history, alpha=0.3, color=COLORS['inhibitory'],
                     label=f'$I_i = g_i(E_i - V)$')

    # Plot net I_syn
    ax3.plot(time, i_syn_history, color=COLORS['membrane'], linewidth=2.5,
             label='$I_{syn} = I_e + I_i$')

    ax3.axhline(0, color='black', linewidth=0.5)

    # Mark all stimulation times
    ax3.axvline(stim_e1_time, color=COLORS['excitatory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_e2_time, color=COLORS['excitatory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_i1_time, color=COLORS['inhibitory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_i2_time, color=COLORS['inhibitory'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Synaptic Current $I_{syn}$', fontsize=12)
    ax3.set_title(f'Net Synaptic Current at $V \\approx$ {V_rest:.0f} mV', fontsize=14)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax3.grid(True, alpha=0.5)
    ax3.tick_params(axis='both', labelsize=11)

    # Add equations at bottom
    equation_text = (
        "Conductance Update: $g \\leftarrow g + w$ on input | "
        "Decay: $g(t+\\Delta t) = g(t) \\cdot e^{-\\Delta t / \\tau}$ | "
        "Current: $I_{syn} = g_e(E_e - V) + g_i(E_i - V)$"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(SAVE_DIR, "fig15_conductance_summation.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 16: SYNAPTIC CURRENTS - I_e AND I_i
# Demonstrates how conductances translate to currents via driving force
# ==============================================================================
def plot_synaptic_currents():
    """
    Visualizes excitatory and inhibitory synaptic currents,
    showing how the driving force (E - V) affects current magnitude.

    Panel 1: Excitatory current I_e = g_e(E_e - V) - positive, depolarizing
    Panel 2: Inhibitory current I_i = g_i(E_i - V) - negative, hyperpolarizing
    Panel 3: Net synaptic current I_syn = I_e + I_i
    """
    print("\n--- Figure 16: Synaptic Currents (I_e and I_i) ---")

    dt = 0.1
    T = 30  # ms
    time = np.arange(0, T, dt)

    tau_e = NEURON_PARAMS['tau_e']  # 3 ms
    tau_i = NEURON_PARAMS['tau_i']  # 7 ms
    E_e = NEURON_PARAMS['e_reversal']  # 0 mV
    E_i = NEURON_PARAMS['i_reversal']  # -80 mV
    V_rest = NEURON_PARAMS['v_rest']   # -70 mV

    # Timing of the two stimulations
    stim_e1_time = 3.0
    stim_e2_time = 9.0

    stim_i1_time = 5.0
    stim_i2_time = 13.0

    # Weight magnitudes (same for both)
    w_e = 0.8
    w_i = 0.8

    # --- Create neurons and run simulations ---
    # Disable adaptation for this figure to show pure conductance/current dynamics
    no_adapt_params = {**NEURON_PARAMS, 'v_noise_amp': 0.0, 'i_noise_amp': 0.0, 'adaptation_increment': 0.0}

    # Neuron receiving only excitatory inputs
    neuron_e = Layered_LIFNeuronWithReversal(**no_adapt_params)
    neuron_e.reset()

    # Neuron receiving only inhibitory inputs
    neuron_i = Layered_LIFNeuronWithReversal(**no_adapt_params)
    neuron_i.reset()

    # Neuron receiving both (for I_syn panel)
    neuron_both = Layered_LIFNeuronWithReversal(**no_adapt_params)
    neuron_both.reset()

    # Record currents
    i_e_only_history = []  # Current from neuron receiving only exc
    i_i_only_history = []  # Current from neuron receiving only inh
    i_e_history = []       # Exc current from neuron receiving both
    i_i_history = []       # Inh current from neuron receiving both
    i_syn_history = []

    for t in time:
        # Apply excitatory stimulations to neuron_e only
        if abs(t - stim_e1_time) < dt / 2 or abs(t - stim_e2_time) < dt / 2:
            neuron_e.receive_spike(w_e)

        # Apply inhibitory stimulations to neuron_i only
        if abs(t - stim_i1_time) < dt / 2 or abs(t - stim_i2_time) < dt / 2:
            neuron_i.receive_spike(-w_i)

        # Apply both to neuron_both
        if abs(t - stim_e1_time) < dt / 2 or abs(t - stim_e2_time) < dt / 2:
            neuron_both.receive_spike(w_e)
        if abs(t - stim_i1_time) < dt / 2 or abs(t - stim_i2_time) < dt / 2:
            neuron_both.receive_spike(-w_i)

        # Calculate currents before update
        # For isolated neurons (to show individual E and I currents cleanly)
        i_e_only = neuron_e.g_e * (E_e - neuron_e.v)
        i_i_only = neuron_i.g_i * (E_i - neuron_i.v)

        # For combined neuron
        i_e = neuron_both.g_e * (E_e - neuron_both.v)
        i_i = neuron_both.g_i * (E_i - neuron_both.v)
        i_syn = i_e + i_i

        i_e_only_history.append(i_e_only)
        i_i_only_history.append(i_i_only)
        i_e_history.append(i_e)
        i_i_history.append(i_i)
        i_syn_history.append(i_syn)

        # Update all neurons
        neuron_e.update(dt)
        neuron_i.update(dt)
        neuron_both.update(dt)

    # --- Create figure with 4 panels (thin V strip + 3 main panels) ---
    fig = plt.figure(figsize=(11, 10))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[0.8, 2, 2, 2], hspace=0.3)

    # =========================================================================
    # Panel 0: Membrane Potential (thin strip at top)
    # =========================================================================
    ax0 = fig.add_subplot(gs[0])

    ax0.plot(time, neuron_e.v_history, color=COLORS['membrane'], linewidth=2,
             label='$V(t)$')
    ax0.axhline(V_rest, color=COLORS['rest'], linestyle=':', linewidth=1, alpha=0.7)

    # Mark stimulation times
    ax0.axvline(stim_e1_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax0.axvline(stim_e2_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax0.set_ylabel('$V$ (mV)', fontsize=11)
    ax0.set_title('Membrane Potential $V(t)$', fontsize=12)
    ax0.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax0.grid(True, alpha=0.5)
    ax0.tick_params(axis='both', labelsize=10)
    plt.setp(ax0.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 1: Excitatory Current
    # =========================================================================
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax1.fill_between(time, 0, i_e_only_history, alpha=0.3, color=COLORS['excitatory'])
    ax1.plot(time, i_e_only_history, color=COLORS['excitatory'], linewidth=2.5,
             label=f'$I_e = g_e(E_e - V)$')

    ax1.axhline(0, color='black', linewidth=0.5)

    # Mark stimulation times
    ax1.axvline(stim_e1_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(stim_e2_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('Excitatory Current $I_e$', fontsize=12)
    ax1.set_title('Excitatory Current: Positive (Depolarizing)', fontsize=14)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.grid(True, alpha=0.5)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 2: Inhibitory Current
    # =========================================================================
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    ax2.fill_between(time, 0, i_i_only_history, alpha=0.3, color=COLORS['inhibitory'])
    ax2.plot(time, i_i_only_history, color=COLORS['inhibitory'], linewidth=2.5,
             label=f'$I_i = g_i(E_i - V)$')

    ax2.axhline(0, color='black', linewidth=0.5)

    # Mark stimulation times
    ax2.axvline(stim_i1_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(stim_i2_time, color=COLORS['spike'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('Inhibitory Current $I_i$', fontsize=12)
    ax2.set_title('Inhibitory Current: Negative (Hyperpolarizing)', fontsize=14)
    ax2.legend(loc='lower right', framealpha=0.9, fontsize=11)
    ax2.grid(True, alpha=0.5)
    ax2.tick_params(axis='both', labelsize=11)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # =========================================================================
    # Panel 3: Net Synaptic Current
    # =========================================================================
    ax3 = fig.add_subplot(gs[3], sharex=ax0)

    # Plot I_e and I_i separately
    ax3.fill_between(time, 0, i_e_history, alpha=0.3, color=COLORS['excitatory'],
                     label=f'$I_e$')
    ax3.fill_between(time, 0, i_i_history, alpha=0.3, color=COLORS['inhibitory'],
                     label=f'$I_i$')

    # Plot net I_syn
    ax3.plot(time, i_syn_history, color=COLORS['membrane'], linewidth=2.5,
             label='$I_{syn} = I_e + I_i$')

    ax3.axhline(0, color='black', linewidth=0.5)

    # Mark all stimulation times
    ax3.axvline(stim_e1_time, color=COLORS['excitatory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_e2_time, color=COLORS['excitatory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_i1_time, color=COLORS['inhibitory'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(stim_i2_time, color=COLORS['inhibitory'], linestyle='--', linewidth=1.5, alpha=0.7)

    ax3.set_xlabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Synaptic Current $I_{syn}$', fontsize=12)
    ax3.set_title(f'Net Synaptic Current at $V \\approx$ {V_rest:.0f} mV', fontsize=14)
    ax3.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax3.grid(True, alpha=0.5)
    ax3.tick_params(axis='both', labelsize=11)

    # Add equations at bottom
    equation_text = (
        f"$I_e = g_e(E_e - V)$ : at $V$ = {V_rest} mV, driving force = {E_e - V_rest:+.0f} mV (depolarizing) | "
        f"$I_i = g_i(E_i - V)$ : driving force = {E_i - V_rest:+.0f} mV (hyperpolarizing)"
    )
    fig.text(0.5, 0.01, equation_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='#f3f4f6', edgecolor='#d1d5db'))
    save_path = os.path.join(SAVE_DIR, "fig16_synaptic_currents.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# FIGURE 17: STIMULUS-RESPONSE CURVES (F-I Curves)
# Shows firing rate as a function of input strength for excitatory and inhibitory neurons
# ==============================================================================
def plot_stimulus_response_curves():
    """
    Visualizes stimulus-response (f-I) curves for excitatory and inhibitory neurons.

    Shows how firing rate increases with input stimulus strength, demonstrating:
    - Threshold behavior (no firing below rheobase)
    - Linear rise then saturation at high stimulus
    - Effect of adaptation on firing rate
    """
    print("\n--- Figure 17: Stimulus-Response Curves (f-I Curves) ---")

    dt = 0.1
    T = 2000  # ms - longer for stable firing rate estimates and smoother curves
    T_discard = 200  # ms - discard initial transient

    # Larger stimulus range to see saturation/plateau, more points for smooth curves
    stimulus_range = np.linspace(0, 4.5, 300)

    # Excitatory neuron with adaptation
    # Regular-spiking pyramidal cells: uses class default tau_ref=4.0ms (~250 Hz max)
    excit_params = {
        **NEURON_PARAMS,
        'is_inhibitory': False,
        'v_noise_amp': 0.0,
        'i_noise_amp': 0.0,
        'adaptation_increment': 0.1,  # Standard adaptation
    }

    # Inhibitory neuron - typically faster-spiking with less adaptation
    # Fast-spiking interneurons: uses class default tau_ref=2.5ms (~400 Hz max)
    inhib_params = {
        **NEURON_PARAMS,
        'is_inhibitory': True,
        'v_noise_amp': 0.0,
        'i_noise_amp': 0.0,
        'adaptation_increment': 0.02,  # Less adaptation (fast-spiking interneuron)
        'tau_m': 8.0,  # Slightly faster membrane time constant
    }

    excit_rates = []
    inhib_rates = []

    # Effective measurement time (excluding transient)
    T_measure = T - T_discard

    # Measure firing rate for each stimulus intensity
    for stim in stimulus_range:
        time = np.arange(0, T, dt)

        # Excitatory neuron
        excit_neuron = Layered_LIFNeuronWithReversal(**excit_params)
        excit_neuron.reset()
        excit_spikes = 0

        for t in time:
            excit_neuron.apply_external_stimulus(stim)
            if excit_neuron.update(dt):
                # Only count spikes after transient period
                if t >= T_discard:
                    excit_spikes += 1

        excit_rates.append(excit_spikes / (T_measure / 1000))  # Convert to Hz

        # Inhibitory neuron
        inhib_neuron = Layered_LIFNeuronWithReversal(**inhib_params)
        inhib_neuron.reset()
        inhib_spikes = 0

        for t in time:
            inhib_neuron.apply_external_stimulus(stim)
            if inhib_neuron.update(dt):
                # Only count spikes after transient period
                if t >= T_discard:
                    inhib_spikes += 1

        inhib_rates.append(inhib_spikes / (T_measure / 1000))  # Convert to Hz

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Convert to arrays for easier manipulation
    excit_rates = np.array(excit_rates)
    inhib_rates = np.array(inhib_rates)

    # Plot the curves
    ax1.plot(stimulus_range, excit_rates, color=COLORS['excitatory'], linewidth=3,
             label='Excitatory (regular spiking)')
    ax1.plot(stimulus_range, inhib_rates, color=COLORS['inhibitory'], linewidth=3,
             label='Inhibitory (fast spiking)')

    # Find rheobase for each
    excit_rheobase_idx = np.argmax(excit_rates > 0)
    inhib_rheobase_idx = np.argmax(inhib_rates > 0)

    # Mark key regions
    if excit_rates[excit_rheobase_idx] > 0:
        excit_rheobase = stimulus_range[excit_rheobase_idx]
        ax1.axvline(excit_rheobase, color=COLORS['excitatory'], linestyle=':', alpha=0.7, linewidth=1.5)

    # Shade the subthreshold region
    ax1.axvspan(0, stimulus_range[max(excit_rheobase_idx, inhib_rheobase_idx)],
                alpha=0.1, color='gray', label='Subthreshold')

    # Mark saturation region (where curve flattens)
    max_rate = max(np.max(excit_rates), np.max(inhib_rates))
    ax1.axhline(np.max(excit_rates) * 0.95, color=COLORS['excitatory'], linestyle='--',
                alpha=0.4, linewidth=1)
    ax1.axhline(np.max(inhib_rates) * 0.95, color=COLORS['inhibitory'], linestyle='--',
                alpha=0.4, linewidth=1)

    # Annotations
    ax1.annotate('Rheobase\n(threshold)',
                 xy=(excit_rheobase, 20),
                 xytext=(excit_rheobase + 1.5, 80),
                 fontsize=10, color=COLORS['annotation'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['annotation'], lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    # Find where curves start to plateau (derivative decreases significantly)
    excit_deriv = np.gradient(excit_rates, stimulus_range)
    plateau_start_idx = np.argmax(excit_deriv < 0.3 * np.max(excit_deriv))
    if plateau_start_idx > 0 and plateau_start_idx < len(stimulus_range) - 5:
        plateau_stim = stimulus_range[plateau_start_idx]
        ax1.annotate('Saturation\n(adaptation limits rate)',
                     xy=(plateau_stim + 2, excit_rates[min(plateau_start_idx + 10, len(excit_rates)-1)]),
                     xytext=(plateau_stim + 1, excit_rates[plateau_start_idx] - 100),
                     fontsize=10, color=COLORS['annotation'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['annotation'], lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

    ax1.set_xlabel('Stimulus Intensity (a.u.)', fontsize=12)
    ax1.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax1.set_title('Stimulus-Response (f-I) Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.95, fontsize=8)
    ax1.grid(True, alpha=0.4)
    ax1.set_xlim(0, max(stimulus_range))
    ax1.set_ylim(0, max_rate * 1.1)

    # =========================================================================
    # Panel 2 (Right): Adaptation currents at a fixed stimulus
    # =========================================================================
    # Run neurons at a moderate stimulus to show adaptation current dynamics
    stim_for_adapt = 3.0
    T_adapt = 500  # ms
    time_adapt = np.arange(0, T_adapt, dt)

    # Excitatory neuron - track adaptation conductance (g_adapt), not current
    neuron_e = Layered_LIFNeuronWithReversal(**excit_params)
    neuron_e.reset()
    adapt_g_e = []

    for t in time_adapt:
        neuron_e.apply_external_stimulus(stim_for_adapt)
        neuron_e.update(dt)
        adapt_g_e.append(neuron_e.adaptation)  # Just the conductance, smoother

    # Inhibitory neuron
    neuron_i = Layered_LIFNeuronWithReversal(**inhib_params)
    neuron_i.reset()
    adapt_g_i = []

    for t in time_adapt:
        neuron_i.apply_external_stimulus(stim_for_adapt)
        neuron_i.update(dt)
        adapt_g_i.append(neuron_i.adaptation)

    ax2.plot(time_adapt, adapt_g_e, color=COLORS['excitatory'], linewidth=2,
             label=f'Excitatory (increment={excit_params["adaptation_increment"]})')
    ax2.plot(time_adapt, adapt_g_i, color=COLORS['inhibitory'], linewidth=2,
             label=f'Inhibitory (increment={inhib_params["adaptation_increment"]})')

    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Adaptation Conductance $g_{adapt}$', fontsize=12)
    ax2.set_title(f'Adaptation Buildup at Stimulus = {stim_for_adapt}', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax2.grid(True, alpha=0.4)
    ax2.set_xlim(0, T_adapt)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig17_stimulus_response_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ARTICLE MATHEMATICS VISUALIZATIONS")
    print("Generating publication-quality figures for SNN article")
    print("=" * 70)

    # Generate all figures
    plot_membrane_leak()           # Figure 1: Membrane leak term
    plot_conductance_dynamics()    # Figure 2: Conductance decay
    plot_driving_force()           # Figure 3: Synaptic current with driving force
    plot_euler_integration()       # Figure 4: Forward Euler numerical integration
    plot_action_potential()        # Figure 5: Action potential mechanism
    plot_adaptation()              # Figure 6: Spike-frequency adaptation
    plot_complete_lif_equation()   # Figure 7: Complete LIF equation
    plot_weight_conductance()      # Figure 8: Synaptic weights and conductance
    plot_excitation_inhibition_noise()  # Figure 9: Excitation, inhibition, noise dynamics
    plot_subthreshold_to_spike()         # Figure 10: Subthreshold integration to AP
    plot_three_neuron_interaction()      # Figure 11: Three interacting neurons (2E, 1I)
    plot_conductance_summation()         # Figure 15: Conductance production, decay, summation
    plot_stimulus_response_curves()      # Figure 17: Stimulus-response (f-I) curves

    print("\n" + "=" * 70)
    print(f"All figures saved to: {SAVE_DIR}/")
    print("=" * 70)
