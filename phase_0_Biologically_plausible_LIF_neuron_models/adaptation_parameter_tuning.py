# adaptation_parameter_tuning.py
# Comprehensive parameter sweep for spike-frequency adaptation tuning
# Visualizes relationship between adaptation parameters and firing behavior

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import os
import sys

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from LIF_objects.Layered_LIFNeuronWithReversal import Layered_LIFNeuronWithReversal

# Plotting style
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
    'font.size': 10,
    'figure.dpi': 150,
})

SAVE_DIR = "adaptation_tuning_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# Base neuron parameters
BASE_PARAMS = {
    'v_rest': -70.0,
    'v_threshold': -55.0,
    'v_reset': -75.0,
    'tau_m': 10.0,
    'tau_ref': None,  # Use class default based on is_inhibitory (4.0ms excit, 2.5ms inhib)
    'tau_e': 3.0,
    'tau_i': 7.0,
    'e_reversal': 0.0,
    'i_reversal': -80.0,
    'v_noise_amp': 0.0,
    'i_noise_amp': 0.0,
    'e_k_reversal': -90.0,
}


def measure_adaptation_metrics(adaptation_increment, tau_adaptation,
                                stim_strength=1.5, stim_duration=500, dt=0.1):
    """
    Measure adaptation metrics for given parameters.

    Returns:
        dict with:
        - adaptation_ratio: initial_rate / steady_state_rate
        - initial_rate: firing rate in first 50ms of stimulation (Hz)
        - steady_state_rate: firing rate in last 100ms of stimulation (Hz)
        - total_spikes: total number of spikes during stimulation
        - spike_times: list of spike times
        - isi_values: inter-spike intervals
    """
    params = {
        **BASE_PARAMS,
        'adaptation_increment': adaptation_increment,
        'tau_adaptation': tau_adaptation,
    }
    neuron = Layered_LIFNeuronWithReversal(**params)

    T = stim_duration + 100  # Extra time after stimulus
    time = np.arange(0, T, dt)
    stim_start = 50
    stim_end = stim_start + stim_duration

    neuron.reset()
    spike_times = []

    for t in time:
        if stim_start <= t < stim_end:
            neuron.apply_external_stimulus(stim_strength)
        else:
            neuron.apply_external_stimulus(0.0)

        if neuron.update(dt):
            spike_times.append(t)

    # Calculate metrics
    spike_times = np.array(spike_times)
    stim_spikes = spike_times[(spike_times >= stim_start) & (spike_times < stim_end)]

    if len(stim_spikes) < 2:
        return {
            'adaptation_ratio': 1.0,
            'initial_rate': 0.0,
            'steady_state_rate': 0.0,
            'total_spikes': len(stim_spikes),
            'spike_times': spike_times,
            'isi_values': np.array([]),
            'v_history': neuron.v_history,
            'adaptation_history': neuron.adaptation_history,
        }

    # ISI analysis
    isi_values = np.diff(stim_spikes)

    # Initial rate: first 50ms of stimulation
    initial_spikes = stim_spikes[(stim_spikes >= stim_start) & (stim_spikes < stim_start + 50)]
    initial_rate = len(initial_spikes) / 0.05 if len(initial_spikes) > 0 else 0  # Hz

    # Steady-state rate: last 100ms of stimulation
    ss_start = stim_end - 100
    ss_spikes = stim_spikes[(stim_spikes >= ss_start) & (stim_spikes < stim_end)]
    steady_state_rate = len(ss_spikes) / 0.1 if len(ss_spikes) > 0 else 0  # Hz

    # Adaptation ratio
    if steady_state_rate > 0:
        adaptation_ratio = initial_rate / steady_state_rate
    else:
        adaptation_ratio = float('inf') if initial_rate > 0 else 1.0

    return {
        'adaptation_ratio': adaptation_ratio,
        'initial_rate': initial_rate,
        'steady_state_rate': steady_state_rate,
        'total_spikes': len(stim_spikes),
        'spike_times': spike_times,
        'isi_values': isi_values,
        'v_history': neuron.v_history,
        'adaptation_history': neuron.adaptation_history,
    }


def plot_parameter_sweep():
    """
    Create heatmaps showing adaptation ratio across parameter space.
    """
    print("\n=== Parameter Sweep: Adaptation Ratio Heatmap ===")

    # Parameter ranges
    adapt_increments = np.logspace(-3, 0, 25)  # 0.001 to 1.0
    tau_adaptations = np.linspace(20, 200, 20)  # 20 to 200 ms

    # Storage for results
    adaptation_ratios = np.zeros((len(tau_adaptations), len(adapt_increments)))
    initial_rates = np.zeros_like(adaptation_ratios)
    steady_rates = np.zeros_like(adaptation_ratios)

    total = len(tau_adaptations) * len(adapt_increments)
    count = 0

    for i, tau_adapt in enumerate(tau_adaptations):
        for j, adapt_inc in enumerate(adapt_increments):
            metrics = measure_adaptation_metrics(adapt_inc, tau_adapt)
            adaptation_ratios[i, j] = min(metrics['adaptation_ratio'], 10)  # Cap at 10
            initial_rates[i, j] = metrics['initial_rate']
            steady_rates[i, j] = metrics['steady_state_rate']
            count += 1
            if count % 50 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Heatmap 1: Adaptation ratio
    ax1 = axes[0]
    im1 = ax1.imshow(adaptation_ratios, aspect='auto', origin='lower',
                     extent=[np.log10(adapt_increments[0]), np.log10(adapt_increments[-1]),
                            tau_adaptations[0], tau_adaptations[-1]],
                     cmap='viridis', vmin=1, vmax=5)
    ax1.set_xlabel('log$_{10}$(adaptation_increment)')
    ax1.set_ylabel('$\\tau_{adaptation}$ (ms)')
    ax1.set_title('Adaptation Ratio (initial/steady-state)')
    plt.colorbar(im1, ax=ax1, label='Ratio')

    # Add contour lines for target ratios
    X, Y = np.meshgrid(np.log10(adapt_increments), tau_adaptations)
    contours = ax1.contour(X, Y, adaptation_ratios, levels=[2, 3, 4], colors='white', linewidths=1)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.0f:1')

    # Heatmap 2: Initial firing rate
    ax2 = axes[1]
    im2 = ax2.imshow(initial_rates, aspect='auto', origin='lower',
                     extent=[np.log10(adapt_increments[0]), np.log10(adapt_increments[-1]),
                            tau_adaptations[0], tau_adaptations[-1]],
                     cmap='Reds')
    ax2.set_xlabel('log$_{10}$(adaptation_increment)')
    ax2.set_ylabel('$\\tau_{adaptation}$ (ms)')
    ax2.set_title('Initial Firing Rate (Hz)')
    plt.colorbar(im2, ax=ax2, label='Hz')

    # Heatmap 3: Steady-state firing rate
    ax3 = axes[2]
    im3 = ax3.imshow(steady_rates, aspect='auto', origin='lower',
                     extent=[np.log10(adapt_increments[0]), np.log10(adapt_increments[-1]),
                            tau_adaptations[0], tau_adaptations[-1]],
                     cmap='Blues')
    ax3.set_xlabel('log$_{10}$(adaptation_increment)')
    ax3.set_ylabel('$\\tau_{adaptation}$ (ms)')
    ax3.set_title('Steady-State Firing Rate (Hz)')
    plt.colorbar(im3, ax=ax3, label='Hz')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig1_parameter_sweep_heatmaps.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()

    return adapt_increments, tau_adaptations, adaptation_ratios


def plot_adaptation_increment_sweep():
    """
    Show effect of varying adaptation_increment at fixed tau_adaptation.
    """
    print("\n=== Adaptation Increment Sweep ===")

    tau_adaptation = 100.0  # Fixed
    adapt_increments = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1.5, 1.5], hspace=0.35, wspace=0.3)

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(adapt_increments)))

    # Panel 1: Membrane potential traces
    ax1 = fig.add_subplot(gs[0, :])

    dt = 0.1
    stim_start, stim_end = 50, 450

    all_metrics = []

    for adapt_inc, color in zip(adapt_increments, colors):
        metrics = measure_adaptation_metrics(adapt_inc, tau_adaptation,
                                            stim_strength=1.5, stim_duration=400)
        all_metrics.append(metrics)

        # Use actual history length for time array
        v_hist = metrics['v_history']
        time_trace = np.arange(0, len(v_hist) * dt, dt)[:len(v_hist)]

        label = f'$\\Delta g_{{adapt}}$ = {adapt_inc}'
        if adapt_inc == 0:
            label = 'No adaptation'
        ax1.plot(time_trace, v_hist, color=color, linewidth=1,
                label=label, alpha=0.8)

    ax1.axhline(-55, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvspan(stim_start, stim_end, alpha=0.1, color='orange')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(f'Effect of Adaptation Increment ($\\tau_{{adapt}}$ = {tau_adaptation} ms)')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(-85, -50)
    ax1.grid(True, alpha=0.5)

    # Panel 2: ISI evolution
    ax2 = fig.add_subplot(gs[1, 0])

    for i, (adapt_inc, metrics, color) in enumerate(zip(adapt_increments, all_metrics, colors)):
        if len(metrics['isi_values']) > 1:
            ax2.plot(range(1, len(metrics['isi_values'])+1), metrics['isi_values'],
                    'o-', color=color, markersize=3, linewidth=1,
                    label=f'{adapt_inc}')

    ax2.set_xlabel('Spike Number')
    ax2.set_ylabel('Inter-Spike Interval (ms)')
    ax2.set_title('ISI Evolution During Stimulation')
    ax2.legend(title='$\\Delta g_{adapt}$', fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.5)

    # Panel 3: Instantaneous firing rate
    ax3 = fig.add_subplot(gs[1, 1])

    for adapt_inc, metrics, color in zip(adapt_increments, all_metrics, colors):
        if len(metrics['isi_values']) > 1:
            inst_rates = 1000 / metrics['isi_values']  # Hz
            spike_times = metrics['spike_times']
            rate_times = spike_times[1:len(inst_rates)+1]
            ax3.plot(rate_times, inst_rates, 'o-', color=color, markersize=3, linewidth=1)

    ax3.axvspan(stim_start, stim_end, alpha=0.1, color='orange')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Instantaneous Rate (Hz)')
    ax3.set_title('Firing Rate Over Time')
    ax3.grid(True, alpha=0.5)
    ax3.set_xlim(0, 500)

    # Panel 4: Adaptation ratio bar chart
    ax4 = fig.add_subplot(gs[2, 0])

    ratios = [m['adaptation_ratio'] for m in all_metrics]
    ratios = [min(r, 10) for r in ratios]  # Cap for display
    bars = ax4.bar(range(len(adapt_increments)), ratios, color=colors, edgecolor='black')
    ax4.set_xticks(range(len(adapt_increments)))
    ax4.set_xticklabels([f'{a}' for a in adapt_increments], rotation=45)
    ax4.set_xlabel('adaptation_increment')
    ax4.set_ylabel('Adaptation Ratio')
    ax4.set_title('Adaptation Ratio (Initial/Steady-State Rate)')
    ax4.axhline(3, color='red', linestyle='--', alpha=0.7, label='Target: 3:1')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.5, axis='y')

    # Add value labels on bars
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax4.annotate(f'{ratio:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Panel 5: Rate comparison
    ax5 = fig.add_subplot(gs[2, 1])

    x = np.arange(len(adapt_increments))
    width = 0.35

    initial = [m['initial_rate'] for m in all_metrics]
    steady = [m['steady_state_rate'] for m in all_metrics]

    ax5.bar(x - width/2, initial, width, label='Initial (first 50ms)', color='#ef4444')
    ax5.bar(x + width/2, steady, width, label='Steady-state (last 100ms)', color='#3b82f6')

    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{a}' for a in adapt_increments], rotation=45)
    ax5.set_xlabel('adaptation_increment')
    ax5.set_ylabel('Firing Rate (Hz)')
    ax5.set_title('Initial vs Steady-State Firing Rate')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.5, axis='y')

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig2_adaptation_increment_sweep.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


def plot_tau_adaptation_sweep():
    """
    Show effect of varying tau_adaptation at fixed adaptation_increment.
    """
    print("\n=== Tau Adaptation Sweep ===")

    adaptation_increment = 0.1  # Fixed
    tau_adaptations = [30, 50, 80, 100, 150, 200]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colors = plt.cm.plasma(np.linspace(0, 0.9, len(tau_adaptations)))

    dt = 0.1
    stim_start, stim_end = 50, 450

    all_metrics = []

    # Panel 1: Membrane potential traces
    ax1 = fig.add_subplot(gs[0, :])

    for tau_adapt, color in zip(tau_adaptations, colors):
        metrics = measure_adaptation_metrics(adaptation_increment, tau_adapt,
                                            stim_strength=1.5, stim_duration=400)
        all_metrics.append(metrics)

        v_hist = metrics['v_history']
        time_trace = np.arange(0, len(v_hist) * dt, dt)[:len(v_hist)]
        ax1.plot(time_trace, v_hist, color=color, linewidth=1,
                label=f'$\\tau_{{adapt}}$ = {tau_adapt} ms', alpha=0.8)

    ax1.axhline(-55, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvspan(stim_start, stim_end, alpha=0.1, color='orange')
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_title(f'Effect of Adaptation Time Constant ($\\Delta g_{{adapt}}$ = {adaptation_increment})')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_xlim(0, 600)
    ax1.set_ylim(-85, -50)
    ax1.grid(True, alpha=0.5)

    # Panel 2: Adaptation conductance traces
    ax2 = fig.add_subplot(gs[1, 0])

    for tau_adapt, metrics, color in zip(tau_adaptations, all_metrics, colors):
        adapt_hist = metrics['adaptation_history']
        time_trace = np.arange(0, len(adapt_hist) * dt, dt)[:len(adapt_hist)]
        ax2.plot(time_trace, adapt_hist, color=color, linewidth=1.5,
                label=f'{tau_adapt} ms')

    ax2.axvspan(stim_start, stim_end, alpha=0.1, color='orange')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('$g_{adapt}$')
    ax2.set_title('Adaptation Conductance Over Time')
    ax2.legend(title='$\\tau_{adapt}$', fontsize=7, loc='upper right')
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(0, 600)

    # Panel 3: Adaptation ratio vs tau
    ax3 = fig.add_subplot(gs[1, 1])

    ratios = [min(m['adaptation_ratio'], 10) for m in all_metrics]
    ax3.plot(tau_adaptations, ratios, 'o-', color='#8b5cf6', linewidth=2, markersize=8)
    ax3.axhline(3, color='red', linestyle='--', alpha=0.7, label='Target: 3:1')

    ax3.set_xlabel('$\\tau_{adaptation}$ (ms)')
    ax3.set_ylabel('Adaptation Ratio')
    ax3.set_title('Adaptation Ratio vs Time Constant')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.5)

    # Add annotations
    for tau, ratio in zip(tau_adaptations, ratios):
        ax3.annotate(f'{ratio:.1f}', (tau, ratio), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig3_tau_adaptation_sweep.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


def find_optimal_parameters(target_ratio=3.0):
    """
    Find parameter combinations that achieve a target adaptation ratio.
    """
    print(f"\n=== Finding Parameters for Target Ratio: {target_ratio}:1 ===")

    # Fine grid search
    adapt_increments = np.logspace(-2, 0, 30)  # 0.01 to 1.0
    tau_adaptations = np.linspace(50, 150, 20)

    results = []

    for tau_adapt in tau_adaptations:
        for adapt_inc in adapt_increments:
            metrics = measure_adaptation_metrics(adapt_inc, tau_adapt)
            ratio = metrics['adaptation_ratio']

            if 0.8 * target_ratio <= ratio <= 1.2 * target_ratio:
                results.append({
                    'adaptation_increment': adapt_inc,
                    'tau_adaptation': tau_adapt,
                    'adaptation_ratio': ratio,
                    'initial_rate': metrics['initial_rate'],
                    'steady_state_rate': metrics['steady_state_rate'],
                })

    # Sort by closeness to target
    results.sort(key=lambda x: abs(x['adaptation_ratio'] - target_ratio))

    print(f"\nTop 10 parameter combinations closest to {target_ratio}:1 ratio:")
    print("-" * 80)
    print(f"{'adapt_inc':>12} {'tau_adapt':>12} {'ratio':>10} {'init_rate':>12} {'ss_rate':>12}")
    print("-" * 80)

    for r in results[:10]:
        print(f"{r['adaptation_increment']:>12.4f} {r['tau_adaptation']:>12.1f} "
              f"{r['adaptation_ratio']:>10.2f} {r['initial_rate']:>12.1f} "
              f"{r['steady_state_rate']:>12.1f}")

    if results:
        best = results[0]
        print(f"\nRecommended parameters for {target_ratio}:1 adaptation ratio:")
        print(f"  adaptation_increment = {best['adaptation_increment']:.4f}")
        print(f"  tau_adaptation = {best['tau_adaptation']:.1f} ms")
        print(f"  (Actual ratio: {best['adaptation_ratio']:.2f}:1)")

    return results


def plot_target_ratio_contours():
    """
    Create contour plot showing parameter combinations for different target ratios.
    """
    print("\n=== Target Ratio Contours ===")

    adapt_increments = np.logspace(-2.5, 0, 40)
    tau_adaptations = np.linspace(30, 180, 35)

    ratios = np.zeros((len(tau_adaptations), len(adapt_increments)))

    total = len(tau_adaptations) * len(adapt_increments)
    count = 0

    for i, tau_adapt in enumerate(tau_adaptations):
        for j, adapt_inc in enumerate(adapt_increments):
            metrics = measure_adaptation_metrics(adapt_inc, tau_adapt)
            ratios[i, j] = min(metrics['adaptation_ratio'], 8)
            count += 1
            if count % 100 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

    fig, ax = plt.subplots(figsize=(10, 7))

    X, Y = np.meshgrid(np.log10(adapt_increments), tau_adaptations)

    # Filled contours
    levels = np.arange(1, 7, 0.5)
    cf = ax.contourf(X, Y, ratios, levels=levels, cmap='viridis', extend='max')
    plt.colorbar(cf, ax=ax, label='Adaptation Ratio')

    # Line contours for integer ratios
    target_levels = [2, 3, 4, 5]
    cs = ax.contour(X, Y, ratios, levels=target_levels, colors='white', linewidths=2)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%d:1')

    # Mark common biological targets
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='$\\tau_{adapt}$ = 100 ms (typical)')

    ax.set_xlabel('log$_{10}$(adaptation_increment)')
    ax.set_ylabel('$\\tau_{adaptation}$ (ms)')
    ax.set_title('Parameter Space for Target Adaptation Ratios\n(Conductance-Based Adaptation Model)')
    ax.legend(loc='upper left')

    # Add recommended region annotation
    ax.annotate('Biologically\nplausible\nregion',
                xy=(-1.3, 100), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "fig4_target_ratio_contours.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


def main():
    print("=" * 70)
    print("ADAPTATION PARAMETER TUNING")
    print("Comprehensive sweep for conductance-based adaptation model")
    print("=" * 70)

    # Run all analyses
    plot_parameter_sweep()
    plot_adaptation_increment_sweep()
    plot_tau_adaptation_sweep()
    plot_target_ratio_contours()

    # Find optimal parameters for common targets
    for target in [2.0, 3.0, 4.0]:
        find_optimal_parameters(target_ratio=target)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
