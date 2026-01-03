import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Set dark style for all plots
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Import custom modules
from LIF_objects.LIFNeuronWithReversal import LIFNeuronWithReversal
from LIF_objects.RichClubNeuronalNetwork import RichClubNeuronalNetwork
from LIF_utils.simulation_utils import run_unified_simulation, check_criticality_during_run
from LIF_utils.activity_vis_utils import visualize_activity_grid, plot_network_activity_with_stimuli, plot_psth_and_raster
from LIF_utils.network_vis_utils import plot_network_connections_sparse, visualize_distance_weights, visualize_rich_club_distribution
from LIF_utils.neuron_vis_utils import plot_reversal_effects
from LIF_utils.criticality_analysis_utils import plot_enhanced_criticality_analysis,analyze_criticality_comprehensively
from LIF_utils.correlation_length_utils import visualize_spatial_correlations


def run_rich_club_experiment(n_neurons=100, target_degree_range=(7, 15), rich_club_boost=3.0,
                            connection_p=0.3, weight_scale=3.0,
                            duration=5000.0, dt=0.1, stim_interval=None, stim_interval_strength=10,
                            stim_fraction=0.01, transmission_delay=2.0, inhibitory_fraction=0.2,
                            stochastic_stim=True, layout='circle', no_stimulation=False,
                            enable_noise=True, v_noise_amp=0.3, i_noise_amp=0.05,
                            e_reversal=0.0, i_reversal=-80.0, random_seed=42, distance_lambda=0.1, animate=False,
                            std_enabled=False, U=0.3, tau_d=400.0):
    """
    Run experiment with rich club connectivity.

    This experiment creates a network where high-degree nodes (k=7 to k=15)
    preferentially connect to each other, exhibiting rich club organization.

    Parameters:
    -----------
    n_neurons : int
        Number of neurons in the network
    target_degree_range : tuple
        (min_degree, max_degree) for rich club nodes (default: 7-15)
    rich_club_boost : float
        Multiplier for connection probability between rich club nodes
    connection_p : float
        Base connection probability between neurons (0-1)
    weight_scale : float
        Scale factor for synaptic weights
    duration : float
        Duration of simulation in ms
    dt : float
        Time step size in ms
    stim_interval : float or None
        Interval for regular stimulation in ms (None for stochastic)
    transmission_delay : float
        Synaptic transmission delay in ms
    inhibitory_fraction : float
        Fraction of neurons that are inhibitory (0-1)
    stochastic_stim : bool
        Whether to use stochastic stimulation
    layout : str
        Spatial layout: 'grid' or 'circle'
    no_stimulation : bool
        If True, no external stimulation will be applied during the simulation
    enable_noise : bool
        If True, apply biological noise to membrane potential and synaptic input
    v_noise_amp : float
        Amplitude of membrane potential noise in mV (if enable_noise is True)
    i_noise_amp : float
        Amplitude of synaptic current noise (if enable_noise is True)
    e_reversal : float
        Excitatory reversal potential (mV)
    i_reversal : float
        Inhibitory reversal potential (mV)
    random_seed : int
        Seed for random number generation to ensure reproducibility
    distance_lambda : float
            Distance decay constant for synaptic weights (higher values mean faster decay)
    animate : bool
            If True, generate an animation of the neural activity grid
    """
    print("Creating rich club neural network with reversal potentials...")

    # Set the initial random seed for network creation and first simulation
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed} for reproducible results")

    # Set actual noise levels based on enable_noise flag
    actual_v_noise = v_noise_amp if enable_noise else 0.0
    actual_i_noise = i_noise_amp if enable_noise else 0.0

    # Create network with rich club connectivity
    network = RichClubNeuronalNetwork(
        n_neurons=n_neurons,
        target_degree_range=target_degree_range,
        rich_club_boost=rich_club_boost,
        connection_p=connection_p,
        weight_scale=weight_scale,
        spatial=True,
        transmission_delay=transmission_delay,
        inhibitory_fraction=inhibitory_fraction,
        layout=layout,
        v_noise_amp=actual_v_noise,
        i_noise_amp=actual_i_noise,
        e_reversal=e_reversal,      # Excitatory reversal potential
        i_reversal=i_reversal,       # Inhibitory reversal potential
        distance_lambda=distance_lambda,
        std_enabled=std_enabled,     # Short-Term Depression
        U=U,                         # STD utilization factor
        tau_d=tau_d                  # STD recovery time constant
    )

    # Choose a neuron to visualize
    neuron_to_highlight = 42  # This can be any valid neuron index

# Call the standalone visualization function
    print("Generating distance and weight relationship visualization...")
    network_fig, scatter_fig = visualize_distance_weights(
        network=network,
        neuron_idx=neuron_to_highlight,
        save_path="distance_weights_neuron42.png"
)



    print(f"Starting simulation with {n_neurons} neurons for {duration} ms...")
    print(f"Network parameters: connection_p={connection_p}, weight_scale={weight_scale}")
    print(f"Rich club: degree range={target_degree_range}, boost={rich_club_boost}")
    print(f"Transmission delay: {transmission_delay} ms")
    print(f"Inhibitory fraction: {inhibitory_fraction * 100:.1f}%")
    print(f"Reversal potentials: E={e_reversal}mV, I={i_reversal}mV")

    # Print noise information
    if enable_noise:
        print(f"Neural noise enabled: membrane={v_noise_amp} mV, synaptic={i_noise_amp}")
    else:
        print("Neural noise disabled")

    # Adjust stimulation parameters based on no_stimulation flag
    if no_stimulation:
        print("Stimulation: NONE (no external stimulation)")
        stochastic_stim = False
        stim_interval = None
    else:
        print(f"Stimulation: {'Stochastic' if stochastic_stim else 'Regular'}")
        if stim_interval:
            print(f"Stimulation interval: {stim_interval}ms, strength: {stim_interval_strength}")

    # FIRST RUN: Exploratory run with minimal tracking to identify neurons of interest
    print("\n=== FIRST RUN: Exploratory simulation to identify neurons of interest ===")

    # Run first simulation without tracking specific neurons
    activity_record, _, stimulation_record = run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        stim_interval=stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stim_neuron=None,
        track_neurons=None,  # Don't track any specific neurons in first run
        stochastic_stim=stochastic_stim,
        no_stimulation=no_stimulation
    )

    # Analyze avalanches from the first run
    print("\nAnalyzing avalanche statistics from first run:")
    avalanche_sizes = network.avalanche_sizes.copy()
    avalanche_durations = network.avalanche_durations.copy()
    print(f"Detected {len(avalanche_sizes)} avalanches")

    # If no stimulation was applied, select neurons based on activity
    if no_stimulation:
        # Count how many times each neuron was active
        active_count = {}
        for active_indices in activity_record:
            for idx in active_indices:
                if idx in active_count:
                    active_count[idx] += 1
                else:
                    active_count[idx] = 1

        # Get the most active neurons
        most_active = sorted(active_count.items(), key=lambda x: x[1], reverse=True)

        # Select top 3 most active neurons
        stimulated_neurons = [neuron for neuron, _ in most_active[:3]]
        print(f"Most active neurons: {stimulated_neurons}")
    else:
        # Count how many times each neuron was stimulated
        stim_count = {}
        for neurons_list in stimulation_record['neurons']:
            for neuron in neurons_list:
                if neuron in stim_count:
                    stim_count[neuron] += 1
                else:
                    stim_count[neuron] = 1

        # Get the most frequently stimulated neurons
        most_stimulated = sorted(stim_count.items(), key=lambda x: x[1], reverse=True)

        # Select top 3 most stimulated neurons
        stimulated_neurons = [neuron for neuron, _ in most_stimulated[:3]]
        print(f"Most stimulated neurons: {stimulated_neurons}")

    # Find neurons connected to these stimulated/active ones
    connected_neurons = []
    for src in stimulated_neurons:
        for tgt in range(n_neurons):
            if network.weights[src, tgt] != 0 and tgt not in stimulated_neurons and tgt not in connected_neurons:
                connected_neurons.append(tgt)
                if len(connected_neurons) >= 3:  # Limit to 3 connected neurons
                    break

    # Make sure we have something to track for connected neurons
    if not connected_neurons:
        # If no connected neurons found, just pick random ones
        potential_neurons = list(set(range(n_neurons)) - set(stimulated_neurons))
        if potential_neurons:
            connected_neurons = np.random.choice(potential_neurons, size=min(3, len(potential_neurons)), replace=False).tolist()

    print(f"Connected neurons: {connected_neurons}")

    # Create a list of all unique stimulated neurons for visualization
    if no_stimulation:
        all_stimulated_neurons = []
        stim_times = []
    else:
        # Get list of all neurons ever stimulated
        all_stimulated_neurons = list(set().union(*stimulation_record['neurons']))
        print(f"Total unique neurons stimulated during simulation: {len(all_stimulated_neurons)}")

        # Get actual stimulation times (in ms)
        stim_times = stimulation_record['times']

    # Create a combined list of neurons to track in the second run
    track_neurons = list(set(stimulated_neurons + connected_neurons))
    print(f"\n=== SECOND RUN: Detailed simulation tracking {len(track_neurons)} specific neurons ===")
    print(f"Neurons to track: {track_neurons}")

    # SECOND RUN: Reset network and random seed to get consistent behavior
    network.reset_all()

    # Important: Reset the random seed to ensure the same stimulation pattern
    np.random.seed(random_seed)

    # Run the simulation again with the same random seed, but track specific neurons
    # This will produce the same stimulation pattern as the first run
    activity_record, neuron_data, stimulation_record = run_unified_simulation(
        network,
        duration=duration,
        dt=dt,
        stim_interval=stim_interval,
        stim_interval_strength=stim_interval_strength,
        stim_fraction=stim_fraction,
        stim_neuron=None,
        track_neurons=track_neurons,  # Now track our selected neurons
        stochastic_stim=stochastic_stim,
        no_stimulation=no_stimulation
    )

    # Restore the original avalanche data from the first run
    # This ensures consistency between avalanche statistics and neuron visualizations
    network.avalanche_sizes = avalanche_sizes
    network.avalanche_durations = avalanche_durations


    if animate == True:
        # Create grid visualization animation with stimulation tracking
        print("\nGenerating visualizations...")
        visualize_activity_grid(
            network,
            activity_record,
            stim_times=stim_times,
            stim_neurons=all_stimulated_neurons,
            dt=dt,
            save_path="neural_activity_grid.gif",
            max_frames=duration*10  # Keep animation size manageable
        )

    # Plot network activity timeline
    plot_network_activity_with_stimuli(
        network,
        activity_record,
        stim_times,
        dt=dt,
        save_path="network_activity_timeline.png"
    )

    # Plot neuron traces
    if stimulated_neurons and connected_neurons:
        # Check if we have data for all required neurons
        can_plot_traces = all(n in neuron_data for n in stimulated_neurons + connected_neurons)

        # Create the special reversal potential visualization for the first stimulated neuron
        if stimulated_neurons:
            plot_reversal_effects(
                network,
                neuron_data,
                stimulated_neurons[0],
                stim_times,
                dt=dt,
                save_path="reversal_effects_stimulated.png"
            )

        # Create the special reversal potential visualization for the first connected neuron
        if connected_neurons:
            plot_reversal_effects(
                network,
                neuron_data,
                connected_neurons[0],
                stim_times,
                dt=dt,
                save_path="reversal_effects_connected.png"
            )
    else:
        missing = [n for n in stimulated_neurons + connected_neurons if n not in neuron_data]
        print(f"Warning: Cannot plot neuron traces - missing data for neurons: {missing}")

    # Replace the original avalanche statistics visualization with the new one
    print("\nGenerating enhanced avalanche criticality visualizations...")
    criticality_results = plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche")

    # Access the visualization results
    if criticality_results["success"]:
        analysis = criticality_results["analysis"]
        print(f"\nExperiment complete! Analyzed {analysis['avalanche_count']} avalanches.")
        print(f"Final assessment: {analysis['assessment']}")

        if analysis['is_critical']:
            print("✓ The network exhibits critical behavior!")
        else:
            print("The network does not exhibit critical behavior.")
    else:
        print(f"Visualization failed: {criticality_results['message']}")



    # Plot detailed network connectivity with highlighted neurons but only a fraction of edges
    plot_network_connections_sparse(
        network,
        stimulated_neurons,
        connected_neurons,
        edge_percent=.5,  # Show only 5% of edges to reduce clutter
        save_path="network_connections_sparse.png"
    )


     # Generate PSTH and raster plot
    print("\nGenerating PSTH and raster plot...")
    plot_psth_and_raster(
        activity_record,
        stim_times=stim_times if not no_stimulation else None,
        bin_size=20,  # 20 timesteps per bin = 2ms at dt=0.1
        dt=dt,
        neuron_subset=None,  # Use all neurons
        figsize=(14, 10),
        save_path="psth_raster_plot.png"
    )


   # Add spatial correlation visualization
    print("\nGenerating spatial correlation visualization...")
    spatial_correlation_results = visualize_spatial_correlations(
        network=network,
        activity_record=activity_record,
        time_bin_ms=60,  # Larger time bins to capture correlations better
        dt=dt,
        distance_bins=10,  # Fewer bins for more robust statistics
        save_prefix="spatial_correlation_map.png"
    )

    if spatial_correlation_results["success"]:
        if spatial_correlation_results["correlation_length"] is not None:
            print(f"Correlation length from visualization: {spatial_correlation_results['correlation_length']:.2f}")
        else:
            print("Could not fit correlation length, but visualization was generated successfully")
    else:
        print(f"Visualization failed: {spatial_correlation_results.get('error', 'Unknown error')}")

    # Generate rich club distribution visualization
    print("\nGenerating rich club distribution analysis...")
    rich_club_results = visualize_rich_club_distribution(
        network=network,
        save_path="rich_club_analysis.png"
    )

    if rich_club_results["success"]:
        print(f"Rich club nodes: {rich_club_results['rich_club_count']} ({100*rich_club_results['rich_club_count']/n_neurons:.1f}%)")
        if "mean_phi_in_range" in rich_club_results:
            print(f"Mean Φ(k) in target range: {rich_club_results['mean_phi_in_range']:.3f}")
            if rich_club_results.get("has_rich_club_structure", False):
                print("✓ Network exhibits rich club structure (Φ_norm > 1)!")
            else:
                print("Network does not exhibit strong rich club structure (Φ_norm ≤ 1)")
    else:
        print(f"Rich club visualization failed: {rich_club_results.get('message', 'Unknown error')}")

    return network, activity_record, neuron_data, stimulation_record, criticality_results



# Function to run experiment with biologically plausible parameters optimized for criticality
def run_rich_club_simulation(random_seed=42, std_enabled=False):
    """
    Run a simulation with rich club connectivity and biologically plausible parameters.

    This simulation creates a network where high-degree nodes (k=7 to k=15)
    preferentially connect to each other, mimicking the rich club organization
    observed in adult structural brain networks.

    Parameters:
    -----------
    random_seed : int
        Random seed for reproducibility
    std_enabled : bool
        Enable Short-Term Synaptic Depression (default: False)
    """

    return run_rich_club_experiment(
        # Network parameters
        n_neurons=3000,              # Local cortical circuit size
        target_degree_range=(200, 300), # Rich club degree range from adult studies
        rich_club_boost=40,          # 20 higher connection probability for rich club
        connection_p=0.08,           # 5% connectivity in local cortical circuits
        weight_scale=0.1,            # Further reduced weight scale to prevent super-critical activity
        inhibitory_fraction=.3,      # 20% inhibitory neurons is biologically realistic
        transmission_delay=1,        # Local connections: 0.5-2 ms delay
        distance_lambda=.1,          # higher is faster decay, check visualization

        # Simulation parameters
        duration=300,                 # 600 ms to observe many avalanches
        dt=0.1,                       # 0.1 ms for precise spike timing

        # Stimulation parameters
        stim_interval=100,           # 300ms stimulation interval
        stim_interval_strength=50,   # Reduced stimulation strength
        stim_fraction=.01,           # Regular stimulation for this experiment
        no_stimulation = True,
        stochastic_stim = False,      # No external stimulation to observe intrinsic dynamics

        # Noise parameters
        enable_noise=True,            # Biological neurons have intrinsic noise
        v_noise_amp=.08 ,              # Reduced membrane potential noise
        i_noise_amp=0.003,            # Reduced synaptic noise

        # Reversal potential parameters (biophysically realistic values)
        e_reversal=0.0,               # AMPA/NMDA reversal potential
        i_reversal=-80.0,             # GABA-A reversal potential

        # STD parameters
        std_enabled=std_enabled,      # Short-Term Depression toggle
        U=0.3,                        # 30% release probability
        tau_d=150,                    # 400ms recovery time constant

        # Other parameters
        layout='circle',              # Layout for visualization
        random_seed=random_seed,
        animate = False       # Random seed for reproducibility
    )


if __name__ == "__main__":
    # Run the rich club experiment with biologically plausible parameters
    network, activity_record, neuron_data, stimulation_record, criticality_results = run_rich_club_simulation()
