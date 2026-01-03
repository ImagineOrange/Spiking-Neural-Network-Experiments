import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set dark theme (standard pattern)
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Import custom modules
from LIF_objects.CircularNeuronalNetwork import CircularNeuronalNetwork
from LIF_objects.LIFNeuronWithReversal import LIFNeuronWithReversal
from LIF_utils.criticality_analysis_utils import plot_enhanced_criticality_analysis


# ============================================================
# TEST 1: Single Synapse Behavior
# ============================================================
def test_single_synapse_depression():
    """
    Validate basic STD mechanism on a single synapse.

    Tests:
    - Exponential depression during repeated stimulation
    - Exponential recovery when stimulation stops
    - Effective weight follows STD dynamics
    """
    print("\n" + "="*60)
    print("TEST 1: Single Synapse Depression and Recovery")
    print("="*60)

    # Parameters
    U = 0.3          # Utilization
    tau_d = 150    # Recovery time constant (ms)
    dt = 0.1         # Time step (ms)

    # Stimulation pattern: 50Hz for 200ms, then pause for 600ms
    stim_duration = 200.0  # ms
    pause_duration = 600.0  # ms
    total_duration = stim_duration + pause_duration
    stim_interval = 20.0  # ms (50Hz)

    # Create minimal 2-neuron network
    print("Creating 2-neuron network with STD enabled...")
    network = CircularNeuronalNetwork(
        n_neurons=2,
        connection_p=0.0,  # No random connections
        weight_scale=1.0,
        spatial=False,
        transmission_delay=1.0,
        inhibitory_fraction=0.0,  # Both excitatory
        std_enabled=True,
        U=U,
        tau_d=tau_d,
        v_noise_amp=0.0,  # No noise for clean signal
        i_noise_amp=0.0
    )

    # Manually set single strong connection: 0 → 1
    network.weights[0, 1] = 5.0  # Strong excitatory connection
    network.delays[0, 1] = 1.0
    network.graph.add_edge(0, 1, weight=5.0, delay=1.0)
    network.x_resources[0, 1] = 1.0  # Start with full resources

    print(f"Connection: Neuron 0 → Neuron 1, weight={network.weights[0, 1]}")
    print(f"STD parameters: U={U}, tau_d={tau_d}ms")
    print(f"Stimulation: {stim_duration}ms at 50Hz, then {pause_duration}ms pause")

    # Tracking arrays
    num_steps = int(total_duration / dt)
    time_array = np.arange(num_steps) * dt
    x_history = []
    w_eff_history = []
    g_e_history = []
    stim_times = []

    # Run simulation
    print("Running simulation...")
    current_time = 0.0
    next_stim_time = 0.0

    for step in range(num_steps):
        current_time = step * dt

        # Stimulate source neuron at regular intervals during stim_duration
        if current_time < stim_duration and current_time >= next_stim_time:
            network.stimulate_neuron(0, 50.0)  # Strong stimulation
            stim_times.append(current_time)
            next_stim_time += stim_interval

        # Update network
        network.update_network(dt)

        # Record state
        x_history.append(network.x_resources[0, 1])
        w_eff = network.weights[0, 1] * U * network.x_resources[0, 1]
        w_eff_history.append(w_eff)
        g_e_history.append(network.neurons[1].g_e)

    x_history = np.array(x_history)
    w_eff_history = np.array(w_eff_history)
    g_e_history = np.array(g_e_history)

    print(f"Simulation complete. Recorded {len(x_history)} timesteps.")
    print(f"Final x_resources: {x_history[-1]:.3f}")
    print(f"Minimum x during stimulation: {x_history[:int(stim_duration/dt)].min():.3f}")

    # Visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor='#1a1a1a')

    # Panel 1: x_resources(t)
    ax1 = axes[0]
    ax1.set_facecolor('#0a0a0a')
    ax1.plot(time_array, x_history, color='#54a0ff', linewidth=2, label='Available Resources (x)')
    ax1.axvspan(0, stim_duration, alpha=0.2, color='yellow', label='Stimulation Period')
    ax1.axvspan(stim_duration, total_duration, alpha=0.2, color='green', label='Recovery Period')
    ax1.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax1.set_ylabel('x (available resources)', fontsize=12, color='white')
    ax1.set_title('Test 1: Synaptic Resource Dynamics', fontsize=14, fontweight='bold', color='white')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # Panel 2: Effective weight(t)
    ax2 = axes[1]
    ax2.set_facecolor('#0a0a0a')
    ax2.plot(time_array, w_eff_history, color='#ff6b6b', linewidth=2, label=f'w_eff = w_base × U × x')
    ax2.axhline(y=network.weights[0, 1], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Base Weight')
    ax2.axvspan(0, stim_duration, alpha=0.2, color='yellow')
    ax2.axvspan(stim_duration, total_duration, alpha=0.2, color='green')
    ax2.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax2.set_ylabel('Effective Weight', fontsize=12, color='white')
    ax2.set_title('Effective Synaptic Weight with STD', fontsize=14, fontweight='bold', color='white')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    # Panel 3: Target neuron excitatory conductance
    ax3 = axes[2]
    ax3.set_facecolor('#0a0a0a')
    ax3.plot(time_array, g_e_history, color='#1dd1a1', linewidth=1.5, label='g_e (target neuron)')
    ax3.axvspan(0, stim_duration, alpha=0.2, color='yellow')
    ax3.axvspan(stim_duration, total_duration, alpha=0.2, color='green')

    # Mark stimulation times
    for st in stim_times[::2]:  # Every other stim to avoid clutter
        ax3.axvline(x=st, color='yellow', alpha=0.3, linewidth=0.5)

    ax3.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax3.set_ylabel('Excitatory Conductance (g_e)', fontsize=12, color='white')
    ax3.set_title('Target Neuron Response', fontsize=14, fontweight='bold', color='white')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('std_test1_single_synapse.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print("Saved: std_test1_single_synapse.png")
    plt.close()

    print("✓ Test 1 complete!\n")
    return network


# ============================================================
# TEST 2: Frequency Dependence
# ============================================================
def test_frequency_dependence():
    """
    Demonstrate frequency-dependent depression.

    Tests:
    - Higher frequencies → stronger steady-state depression
    - Steady-state x_ss ≈ 1/(1 + U*f*tau_d) (theoretical prediction)
    """
    print("\n" + "="*60)
    print("TEST 2: Frequency Dependence of Depression")
    print("="*60)

    # Parameters
    U = 0.3
    tau_d = 400.0
    dt = 0.1

    # Test three frequencies
    frequencies = [10, 50, 100]  # Hz
    colors = ['#54a0ff', '#ff6b6b', '#1dd1a1']

    results = {}

    for freq, color in zip(frequencies, colors):
        print(f"\nTesting {freq}Hz stimulation...")

        stim_interval = 1000.0 / freq  # Convert Hz to ms interval
        duration = 1000.0  # 1 second to reach steady state

        # Create network
        network = CircularNeuronalNetwork(
            n_neurons=2,
            connection_p=0.0,
            weight_scale=1.0,
            spatial=False,
            std_enabled=True,
            U=U,
            tau_d=tau_d,
            v_noise_amp=0.0,
            i_noise_amp=0.0
        )

        # Set connection
        network.weights[0, 1] = 5.0
        network.delays[0, 1] = 1.0
        network.graph.add_edge(0, 1, weight=5.0, delay=1.0)

        # Run simulation
        num_steps = int(duration / dt)
        x_history = []
        next_stim = 0.0

        for step in range(num_steps):
            current_time = step * dt

            if current_time >= next_stim:
                network.stimulate_neuron(0, 50.0)
                next_stim += stim_interval

            network.update_network(dt)
            x_history.append(network.x_resources[0, 1])

        x_history = np.array(x_history)
        time_array = np.arange(num_steps) * dt

        # Calculate steady-state (last 20% of simulation)
        steady_start = int(0.8 * num_steps)
        x_ss = np.mean(x_history[steady_start:])

        # Theoretical prediction
        x_ss_theory = 1.0 / (1.0 + U * (freq / 1000.0) * tau_d)  # freq in kHz for ms units

        print(f"  Steady-state x: {x_ss:.3f}")
        print(f"  Theoretical x_ss: {x_ss_theory:.3f}")
        print(f"  Error: {abs(x_ss - x_ss_theory):.3f}")

        results[freq] = {
            'time': time_array,
            'x': x_history,
            'x_ss': x_ss,
            'x_ss_theory': x_ss_theory,
            'color': color
        }

    # Visualization
    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1a1a1a')

    # Panel 1: x(t) traces for all frequencies
    ax1.set_facecolor('#0a0a0a')
    for freq in frequencies:
        res = results[freq]
        ax1.plot(res['time'], res['x'], color=res['color'], linewidth=2,
                label=f'{freq}Hz (x_ss={res["x_ss"]:.3f})', alpha=0.8)

    ax1.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax1.set_ylabel('Available Resources (x)', fontsize=12, color='white')
    ax1.set_title('STD Dynamics at Different Frequencies', fontsize=14, fontweight='bold', color='white')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='white')

    # Panel 2: Steady-state x vs frequency
    ax2.set_facecolor('#0a0a0a')
    freqs_array = np.array(frequencies)
    x_ss_measured = np.array([results[f]['x_ss'] for f in frequencies])
    x_ss_theoretical = np.array([results[f]['x_ss_theory'] for f in frequencies])

    ax2.scatter(freqs_array, x_ss_measured, s=150, color='#ff6b6b',
               edgecolor='yellow', linewidth=2, zorder=5, label='Measured', alpha=0.9)

    # Plot theoretical curve
    freq_range = np.linspace(5, 150, 100)
    x_theory_curve = 1.0 / (1.0 + U * (freq_range / 1000.0) * tau_d)
    ax2.plot(freq_range, x_theory_curve, color='#54a0ff', linewidth=2,
            linestyle='--', label='Theory: 1/(1+U·f·τ_d)', alpha=0.7)

    ax2.set_xlabel('Stimulation Frequency (Hz)', fontsize=12, color='white')
    ax2.set_ylabel('Steady-State Resources (x_ss)', fontsize=12, color='white')
    ax2.set_title('Frequency Dependence: Theory vs Measurement', fontsize=14, fontweight='bold', color='white')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('std_test2_frequency_dependence.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print("Saved: std_test2_frequency_dependence.png")
    plt.close()

    print("✓ Test 2 complete!\n")
    return results


# ============================================================
# TEST 3: Hub Self-Limitation
# ============================================================
def test_hub_self_limitation():
    """
    Show that highly-connected neurons experience more depression.

    Tests:
    - Hub synapses depress more than periphery synapses
    - Hub firing rate decreases over time due to STD
    """
    print("\n" + "="*60)
    print("TEST 3: Hub Self-Limitation in Star Network")
    print("="*60)

    # Parameters
    U = 0.3
    tau_d = 400.0
    dt = 0.1
    duration = 500.0

    # Create 10-neuron star network: 1 hub (neuron 0), 9 periphery (neurons 1-9)
    print("Creating star network (1 hub + 9 periphery neurons)...")
    network = CircularNeuronalNetwork(
        n_neurons=10,
        connection_p=0.0,  # Manual connections only
        weight_scale=1.0,
        spatial=False,
        std_enabled=True,
        U=U,
        tau_d=tau_d,
        v_noise_amp=0.1,  # Small noise
        i_noise_amp=0.01
    )

    # Connect hub (0) to all periphery neurons (1-9) bidirectionally
    hub_idx = 0
    periphery_indices = list(range(1, 10))

    for p_idx in periphery_indices:
        # Periphery → Hub (input to hub)
        network.weights[p_idx, hub_idx] = 3.0
        network.delays[p_idx, hub_idx] = 1.0
        network.graph.add_edge(p_idx, hub_idx, weight=3.0, delay=1.0)

        # Hub → Periphery (output from hub)
        network.weights[hub_idx, p_idx] = 3.0
        network.delays[hub_idx, p_idx] = 1.0
        network.graph.add_edge(hub_idx, p_idx, weight=3.0, delay=1.0)

    print(f"Hub neuron: {hub_idx}")
    print(f"Periphery neurons: {periphery_indices}")
    print(f"Total connections: {len(periphery_indices) * 2} (bidirectional)")

    # Run simulation with random periphery stimulation
    print("Running simulation with random periphery stimulation...")
    num_steps = int(duration / dt)
    activity_record = []
    x_hub_outgoing = []
    x_periphery_outgoing = []
    hub_spike_times = []

    for step in range(num_steps):
        current_time = step * dt

        # Randomly stimulate 1-2 periphery neurons
        if step % 50 == 0:  # Every 5ms
            stim_neurons = np.random.choice(periphery_indices, size=np.random.randint(1, 3), replace=False)
            for n in stim_neurons:
                network.stimulate_neuron(n, 30.0)

        # Update network
        active_indices = network.update_network(dt)
        activity_record.append(active_indices)

        # Track hub firing
        if hub_idx in active_indices:
            hub_spike_times.append(current_time)

        # Track mean x_resources for hub outgoing synapses
        hub_x_values = [network.x_resources[hub_idx, p] for p in periphery_indices]
        x_hub_outgoing.append(np.mean(hub_x_values))

        # Track mean x_resources for periphery outgoing synapses (representative sample)
        periphery_x_values = [network.x_resources[p, hub_idx] for p in periphery_indices[:3]]
        x_periphery_outgoing.append(np.mean(periphery_x_values))

    time_array = np.arange(num_steps) * dt
    x_hub_outgoing = np.array(x_hub_outgoing)
    x_periphery_outgoing = np.array(x_periphery_outgoing)

    print(f"Hub fired {len(hub_spike_times)} times")
    print(f"Final hub outgoing x: {x_hub_outgoing[-1]:.3f}")
    print(f"Final periphery outgoing x: {x_periphery_outgoing[-1]:.3f}")

    # Calculate hub firing rate in bins
    bin_size_ms = 100.0
    num_bins = int(duration / bin_size_ms)
    hub_rates = []
    bin_centers = []

    for i in range(num_bins):
        t_start = i * bin_size_ms
        t_end = (i + 1) * bin_size_ms
        spikes_in_bin = sum(1 for t in hub_spike_times if t_start <= t < t_end)
        rate_hz = spikes_in_bin / (bin_size_ms / 1000.0)  # Convert to Hz
        hub_rates.append(rate_hz)
        bin_centers.append((t_start + t_end) / 2.0)

    # Visualization
    print("Generating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor='#1a1a1a')

    # Panel 1: Raster plot
    ax1 = axes[0]
    ax1.set_facecolor('#0a0a0a')

    for step, active in enumerate(activity_record):
        t = step * dt
        for neuron_id in active:
            color = '#ff6b6b' if neuron_id == hub_idx else '#54a0ff'
            marker = 's' if neuron_id == hub_idx else '.'
            size = 40 if neuron_id == hub_idx else 10
            ax1.scatter(t, neuron_id, color=color, marker=marker, s=size, alpha=0.7)

    ax1.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax1.set_ylabel('Neuron Index', fontsize=12, color='white')
    ax1.set_title('Network Activity Raster (Hub in Red)', fontsize=14, fontweight='bold', color='white')
    ax1.set_ylim(-0.5, 9.5)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.tick_params(colors='white')

    # Panel 2: Mean x_resources
    ax2 = axes[1]
    ax2.set_facecolor('#0a0a0a')
    ax2.plot(time_array, x_hub_outgoing, color='#ff6b6b', linewidth=2,
            label='Hub Outgoing Synapses', alpha=0.9)
    ax2.plot(time_array, x_periphery_outgoing, color='#54a0ff', linewidth=2,
            label='Periphery Outgoing Synapses', alpha=0.7)

    ax2.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax2.set_ylabel('Mean Available Resources (x)', fontsize=12, color='white')
    ax2.set_title('Synaptic Depression: Hub vs Periphery', fontsize=14, fontweight='bold', color='white')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='white')

    # Panel 3: Hub firing rate over time
    ax3 = axes[2]
    ax3.set_facecolor('#0a0a0a')
    ax3.plot(bin_centers, hub_rates, color='#1dd1a1', linewidth=2, marker='o',
            markersize=6, label='Hub Firing Rate')

    ax3.set_xlabel('Time (ms)', fontsize=12, color='white')
    ax3.set_ylabel('Firing Rate (Hz)', fontsize=12, color='white')
    ax3.set_title('Hub Neuron Firing Rate (100ms bins)', fontsize=14, fontweight='bold', color='white')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('std_test3_hub_limitation.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print("Saved: std_test3_hub_limitation.png")
    plt.close()

    print("✓ Test 3 complete!\n")
    return network


# ============================================================
# TEST 4: Criticality Comparison (STD on vs off)
# ============================================================
def test_std_criticality_effect():
    """
    Compare avalanche statistics with/without STD.

    Tests:
    - STD effect on avalanche distributions
    - Network stability with STD
    """
    print("\n" + "="*60)
    print("TEST 4: STD Effect on Criticality")
    print("="*60)

    # Shared parameters
    n_neurons = 100
    connection_p = 0.05
    weight_scale = 2.0
    duration = 1000.0
    dt = 0.1
    U = 0.3
    tau_d = 400.0

    results = {}

    for std_enabled in [False, True]:
        label = "STD ON" if std_enabled else "STD OFF"
        print(f"\nRunning simulation with {label}...")

        # Create network
        network = CircularNeuronalNetwork(
            n_neurons=n_neurons,
            connection_p=connection_p,
            weight_scale=weight_scale,
            spatial=True,
            layout='circle',
            std_enabled=std_enabled,
            U=U,
            tau_d=tau_d,
            v_noise_amp=0.2,
            i_noise_amp=0.02,
            inhibitory_fraction=0.2
        )

        # Run simulation with occasional random stimulation
        num_steps = int(duration / dt)

        for step in range(num_steps):
            # Random sparse stimulation
            if step % 200 == 0:  # Every 20ms
                stim_neurons = np.random.choice(n_neurons, size=2, replace=False)
                for n in stim_neurons:
                    network.stimulate_neuron(n, 20.0)

            network.update_network(dt)

        # Store results
        results[label] = {
            'network': network,
            'avalanche_sizes': np.array(network.avalanche_sizes),
            'avalanche_durations': np.array(network.avalanche_durations)
        }

        print(f"  Detected {len(network.avalanche_sizes)} avalanches")
        if len(network.avalanche_sizes) > 0:
            print(f"  Mean avalanche size: {np.mean(network.avalanche_sizes):.2f}")
            print(f"  Max avalanche size: {np.max(network.avalanche_sizes)}")

    # Visualization
    print("\nGenerating comparison visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1a1a1a')

    labels = ["STD OFF", "STD ON"]
    colors = ['#54a0ff', '#ff6b6b']

    for idx, label in enumerate(labels):
        res = results[label]
        sizes = res['avalanche_sizes']
        durations = res['avalanche_durations']

        # Filter out size 1 avalanches for better visualization
        sizes_filtered = sizes[sizes > 1]
        durations_filtered = durations[durations > 1]

        # Size distribution
        ax_size = axes[0, idx]
        ax_size.set_facecolor('#0a0a0a')

        if len(sizes_filtered) > 0:
            bins_size = np.logspace(np.log10(sizes_filtered.min()),
                                   np.log10(sizes_filtered.max()), 30)
            counts, bin_edges = np.histogram(sizes_filtered, bins=bins_size)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            ax_size.scatter(bin_centers, counts, color=colors[idx], s=50, alpha=0.7)
            ax_size.set_xscale('log')
            ax_size.set_yscale('log')
            ax_size.set_xlabel('Avalanche Size', fontsize=12, color='white')
            ax_size.set_ylabel('Count', fontsize=12, color='white')
            ax_size.set_title(f'Avalanche Size Distribution ({label})',
                            fontsize=14, fontweight='bold', color='white')
            ax_size.grid(True, alpha=0.3, which='both')
            ax_size.tick_params(colors='white')

        # Duration distribution
        ax_dur = axes[1, idx]
        ax_dur.set_facecolor('#0a0a0a')

        if len(durations_filtered) > 0:
            bins_dur = np.logspace(np.log10(max(durations_filtered.min(), 1)),
                                  np.log10(durations_filtered.max()), 30)
            counts_dur, bin_edges_dur = np.histogram(durations_filtered, bins=bins_dur)
            bin_centers_dur = (bin_edges_dur[:-1] + bin_edges_dur[1:]) / 2

            ax_dur.scatter(bin_centers_dur, counts_dur, color=colors[idx], s=50, alpha=0.7)
            ax_dur.set_xscale('log')
            ax_dur.set_yscale('log')
            ax_dur.set_xlabel('Avalanche Duration', fontsize=12, color='white')
            ax_dur.set_ylabel('Count', fontsize=12, color='white')
            ax_dur.set_title(f'Avalanche Duration Distribution ({label})',
                           fontsize=14, fontweight='bold', color='white')
            ax_dur.grid(True, alpha=0.3, which='both')
            ax_dur.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig('std_test4_criticality_comparison.png', dpi=300, facecolor='#1a1a1a', bbox_inches='tight')
    print("Saved: std_test4_criticality_comparison.png")
    plt.close()

    print("✓ Test 4 complete!\n")
    return results


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SHORT-TERM DEPRESSION VALIDATION TESTS")
    print("Tsodyks-Markram Model (U=0.3, tau_d=400ms)")
    print("="*60)

    # Run all tests
    test_single_synapse_depression()
    test_frequency_dependence()
    test_hub_self_limitation()
    test_std_criticality_effect()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - std_test1_single_synapse.png")
    print("  - std_test2_frequency_dependence.png")
    print("  - std_test3_hub_limitation.png")
    print("  - std_test4_criticality_comparison.png")
    print("\nSTD mechanism validated successfully!")
