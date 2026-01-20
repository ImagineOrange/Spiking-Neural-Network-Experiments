import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec # Needed for combined plot
from matplotlib.patches import Circle
from tqdm import tqdm

# Suppress tight_layout warnings (common with GridSpec)
warnings.filterwarnings('ignore', message='.*tight_layout.*')

# Default: do NOT set dark style globally - let functions handle it based on darkstyle parameter

def visualize_activity_grid(network, activity_record, stim_times=None, stim_neurons=None, dt=0.1,
                           save_path="neural_activity_grid.gif", max_frames=5000, darkstyle=True):
    """
    Create a grid visualization showing neural activity spreading through the network.
    Each cell in the grid represents a single neuron:
    - Stimulated neurons are shown in green when they're active at stimulus times
    - Regular excitatory neurons are shown in red when active
    - Inhibitory neurons are shown in blue when active

    Trailing coloration uses frame-based decay (0.8x per frame).

    Parameters:
    -----------
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)
    """
    # Set colors based on style
    bg_color = 'white'
    text_color = 'black'

    # Get the grid dimensions from the network
    side_length = network.side_length
    
    # Sample frames if there are too many
    total_frames = len(activity_record)
    if total_frames > max_frames:
        print(f"Sampling {max_frames} frames from {total_frames} total steps for animation")
        indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        sampled_activity = [activity_record[i] for i in indices]
        sampled_times = [i * dt for i in indices]
        adjusted_dt = dt * (total_frames / max_frames)
    else:
        sampled_activity = activity_record
        sampled_times = [i * dt for i in range(total_frames)]
        adjusted_dt = dt
    
    # If stim_neurons not provided, use empty list
    if stim_neurons is None:
        stim_neurons = []
    else:
        print(f"Tracking {len(stim_neurons)} neurons for stimulation visualization")
    
    # If stim_times not provided, use empty list
    if stim_times is None:
        stim_times = []
    else:
        print(f"Using {len(stim_times)} stimulation time points")
    
    # Create lookup for fast checking of stimulation frames
    # For each frame, which neurons were stimulated
    stim_frames_map = {}
    for stim_time in stim_times:
        # Find closest frame to this stimulation time
        frame_idx = int(stim_time / dt)
        if 0 <= frame_idx < total_frames:
            stim_frames_map[frame_idx] = True
    
    # Add a small window around each stimulation frame
    stim_window = int(2 / dt)  # 5ms window (or less if dt is large)
    expanded_stim_frames = {}
    for frame in stim_frames_map:
        for i in range(max(0, frame - stim_window), min(total_frames, frame + stim_window + 1)):
            expanded_stim_frames[i] = True
    
    # Create figure with square aspect ratio for circular display
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_aspect('equal')

    # Initialize activity array for visualization
    activity_grid = np.zeros((side_length, side_length))

    # Create an RGBA array for visualization (row, col, RGBA)
    # Using green for stimulated, red for excitatory, and blue for inhibitory
    # Alpha channel used to make pixels outside the circle transparent
    activity_colors = np.zeros((side_length, side_length, 4))
    
    # Create inhibitory neuron mask
    inhibitory_mask = np.zeros((side_length, side_length), dtype=bool)
    for i in range(network.n_neurons):
        if i < len(network.neurons) and network.neurons[i].is_inhibitory:
            if i in network.neuron_grid_positions:
                row, col = network.neuron_grid_positions[i]
                if 0 <= row < side_length and 0 <= col < side_length:
                    inhibitory_mask[row, col] = True
    
    # Create stimulated neuron mask
    stimulated_mask = np.zeros((side_length, side_length), dtype=bool)
    for i in stim_neurons:
        if i in network.neuron_grid_positions:
            row, col = network.neuron_grid_positions[i]
            if 0 <= row < side_length and 0 <= col < side_length:
                stimulated_mask[row, col] = True

    # Create circular mask - pixels outside the circle will be transparent
    center = (side_length - 1) / 2.0
    radius = side_length / 2.0

    # Add a smooth anti-aliased circle behind the disk to hide pixelated edges
    # The circle is slightly larger than the disk and matches the base gray color
    border_circle = Circle(
        (center, center),  # center position in pixel coordinates
        radius + 0.5,      # slightly larger than the disk radius
        facecolor='#c8c8c8',  # lighter gray color matching the base neuron color
        edgecolor='none',
        zorder=0,          # behind the image
        antialiased=True
    )
    ax.add_patch(border_circle)

    circle_mask = np.zeros((side_length, side_length), dtype=bool)
    for r in range(side_length):
        for c in range(side_length):
            dist_from_center = np.sqrt((r - center)**2 + (c - center)**2)
            if dist_from_center <= radius:
                circle_mask[r, c] = True

    # Pre-compute subtle jitter for gray base to distinguish individual neurons
    # Base gray is 0.78 (lighter) with small variation (±0.03)
    np.random.seed(42)  # Fixed seed for consistent jitter across frames
    gray_jitter = 0.78 + (np.random.rand(side_length, side_length) - 0.5) * 0.06

    # Function to update colors based on activity and current frame
    # Returns RGBA array with transparency outside the circle
    def update_colors(activity_grid, inhibitory_mask, stimulated_mask, frame_idx):
        # RGBA: 4 channels (R, G, B, Alpha)
        colors = np.zeros((side_length, side_length, 4))

        # Check if current frame is a stimulation frame
        is_stim_time = frame_idx in expanded_stim_frames

        for r in range(side_length):
            for c in range(side_length):
                if not circle_mask[r, c]:
                    # Outside the circle - fully transparent
                    colors[r, c, 3] = 0
                    continue

                # Inside the circle - set alpha to 1 (opaque) and base color to gray with jitter
                colors[r, c, 3] = 1.0
                base_gray = gray_jitter[r, c]
                colors[r, c, 0] = base_gray  # Gray base (R) with jitter
                colors[r, c, 1] = base_gray  # Gray base (G) with jitter
                colors[r, c, 2] = base_gray  # Gray base (B) with jitter

                value = activity_grid[r, c]
                if value > 0.01:  # If there's activity
                    if inhibitory_mask[r, c]:
                        # Blue for inhibitory neurons
                        colors[r, c, 0] = base_gray * (1 - value)  # Reduce red
                        colors[r, c, 1] = base_gray * (1 - value)  # Reduce green
                        colors[r, c, 2] = base_gray + (1 - base_gray) * value  # Boost blue
                    elif stimulated_mask[r, c] and is_stim_time:
                        # Green for stimulated neurons during stimulation
                        colors[r, c, 0] = base_gray * (1 - value)  # Reduce red
                        colors[r, c, 1] = base_gray + (1 - base_gray) * value  # Boost green
                        colors[r, c, 2] = base_gray * (1 - value)  # Reduce blue
                    else:
                        # Red for excitatory neurons
                        colors[r, c, 0] = base_gray + (1 - base_gray) * value  # Boost red
                        colors[r, c, 1] = base_gray * (1 - value)  # Reduce green
                        colors[r, c, 2] = base_gray * (1 - value)  # Reduce blue
                # else: stays gray with jitter inside the circle
        return colors
    
    # Initialize activity grid
    activity_colors = update_colors(activity_grid, inhibitory_mask, stimulated_mask, 0)
    
    # Set up the imshow with initial colors (zorder=1 to be above the border circle)
    img = ax.imshow(activity_colors, interpolation='nearest', origin='upper', zorder=1)
    
    title = ax.set_title(f"Time: 0.0 ms", color=text_color, fontsize=14)

    # Remove all grid lines, ticks, and border for clean circular animation
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.axis('off')  # Remove the border/frame completely

    # Create a text element to show if we're currently in a stimulation period
    stim_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                       color=text_color, fontsize=12, verticalalignment='top')
    
    # Progress bar for animation creation
    progress_bar = tqdm(
        total=len(sampled_activity),
        desc="Generating Activity Grid Animation",
        unit="frames",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Function to update the grid for animation
    def update(frame_idx):
        nonlocal activity_grid

        progress_bar.update(1)

        # Calculate the actual time for this frame
        time = sampled_times[frame_idx] if frame_idx < len(sampled_times) else frame_idx * adjusted_dt
        title.set_text(f"Time: {time:.1f} ms")

        # Check if we're in a stimulation period
        orig_frame = int(time / dt)
        is_stim_time = orig_frame in expanded_stim_frames

        # Update the stimulation text
        if is_stim_time:
            stim_text.set_text("STIMULATION ACTIVE")
            stim_text.set_color('lime')  # Bright green on dark background
        else:
            stim_text.set_text("")

        # Update activity grid with frame-based decay for all neurons
        activity_grid *= 0.8  # Decay factor controls how quickly activity fades

        # Add new activity from spikes
        if frame_idx < len(sampled_activity):
            active_indices = sampled_activity[frame_idx]
            for idx in active_indices:
                if idx < network.n_neurons and idx in network.neuron_grid_positions:
                    row, col = network.neuron_grid_positions[idx]
                    if 0 <= row < side_length and 0 <= col < side_length:
                        activity_grid[row, col] = 1.0  # Full intensity for active neurons

        # Update colors - pass the original frame index for stimulation timing
        activity_colors = update_colors(activity_grid, inhibitory_mask, stimulated_mask, orig_frame)

        # Update the image data
        img.set_array(activity_colors)
        return [img, title, stim_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(sampled_activity),
        interval=50, blit=True
    )
    
    # Save animation
    if save_path:
        try:
            print(f"Saving animation to {save_path}...")
            # Use PillowWriter for GIF
            writer = animation.PillowWriter(fps=20)
            anim.save(save_path, writer=writer, dpi=100)
            print(f"Successfully saved animation to {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
    
    # Close progress bar
    progress_bar.close()
    plt.close(fig)
    return anim


def plot_network_activity_with_stimuli(network, activity_record, stim_times=None, dt=0.1, figsize=(12, 7),
                                      save_path="network_activity_timeline.png", darkstyle=True):
    """
    Plot time series of network activity with stimulation times marked.

    Parameters:
    -----------
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)
    """
    print("Plotting network activity timeline...")

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    fig.suptitle("Network Activity Over Time", color=text_color, fontsize=16)
    
    # Create time axis
    times = np.arange(len(activity_record)) * dt
    
    # Calculate activity (number of spikes per timestep)
    activity = [len(spikes) for spikes in activity_record]
    
    # Plot activity
    ax.plot(times, activity, color='#ff7f0e', linewidth=1.5)
    
    # Mark stimulation times if provided
    if stim_times:
        for stim_time in stim_times:
            ax.axvline(x=stim_time, color='#1dd1a1', linewidth=1.0, alpha=0.15,
                      label=f"Stim at {stim_time}ms" if stim_time == stim_times[0] else "")
    
    # Mark avalanches
    avalanche_start_times = []
    avalanche_durations = []
    
    current_avalanche_start = None
    for i, act in enumerate(activity):
        time = i * dt
        
        # Start of avalanche
        if act > 0 and (i == 0 or activity[i-1] == 0):
            current_avalanche_start = time
        
        # End of avalanche
        if act == 0 and i > 0 and activity[i-1] > 0 and current_avalanche_start is not None:
            avalanche_start_times.append(current_avalanche_start)
            avalanche_durations.append(time - current_avalanche_start)
    
    # Plot avalanche periods
    for start, duration in zip(avalanche_start_times, avalanche_durations):
        ax.axvspan(start, start + duration, alpha=0.3, color='#1f77b4')
    
    # Style and labels
    ax.set_xlabel("Time (ms)", color=text_color)
    ax.set_ylabel("Number of Spikes", color=text_color)
    ax.set_title("Network Activity with Avalanches", color=text_color)

    # Add legend for stimulations
    if stim_times:
        ax.legend(loc='upper right', framealpha=0.7)

    # Style adjustments based on darkstyle
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(spine_color)
    ax.spines['left'].set_color(spine_color)
    ax.spines['top'].set_color(spine_color)
    ax.spines['right'].set_color(spine_color)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved network activity timeline to {save_path}")
    
    return fig


def plot_psth_and_raster(activity_record, stim_times=None, stim_duration_ms=None, bin_size=10, dt=0.1,
                          neuron_subset=None, figsize=(14, 8), dpi=150,
                          save_path="psth_raster_plot.png", darkstyle=True):
    """
    Create a PSTH (Peri-Stimulus Time Histogram) plot with a raster plot below it.

    Parameters:
    -----------
    activity_record : list
        List of lists containing active neuron indices at each time step
    stim_times : list or None
        List of stimulation time points (in ms)
    stim_duration_ms : float or None
        Duration of each stimulation period in ms. If provided, draws shaded blocks
        instead of vertical lines.
    bin_size : int
        Size of bins for PSTH in time steps
    dt : float
        Time step size in ms
    neuron_subset : list or None
        List of neuron indices to include (None for all neurons)
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        raster_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        raster_color = 'black'

    # Create figure with GridSpec to have different height ratios
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)
    
    # Upper panel: PSTH
    ax_psth = fig.add_subplot(gs[0])
    
    # Lower panel: Raster plot
    ax_raster = fig.add_subplot(gs[1], sharex=ax_psth)
    
    # Calculate time axis
    n_steps = len(activity_record)
    time_ms = np.arange(n_steps) * dt
    
    # Prepare data for the raster plot
    raster_data = []
    neuron_ids = []
    
    # Extract neuron activity for raster plot
    for t, active_neurons in enumerate(activity_record):
        # Filter neurons if needed
        if neuron_subset is not None:
            active_neurons = [n for n in active_neurons if n in neuron_subset]
            
        for neuron_idx in active_neurons:
            raster_data.append(time_ms[t])  # Time of spike
            neuron_ids.append(neuron_idx)   # Neuron ID
    
    # Calculate PSTH - count spikes in bins
    bin_width_ms = bin_size * dt
    bin_edges_ms = np.arange(0, time_ms[-1] + bin_width_ms, bin_width_ms)
    
    # Count spikes in each bin
    psth_counts, _ = np.histogram(raster_data, bins=bin_edges_ms)
    
    # Convert to firing rate (Hz)
    bin_centers = (bin_edges_ms[:-1] + bin_edges_ms[1:]) / 2
    n_neurons = len(set(neuron_ids)) if neuron_ids else (
        len(neuron_subset) if neuron_subset is not None else 
        max([max(active, default=0) for active in activity_record], default=0) + 1
    )
    
    # Convert to Hz: (spikes/bin) / (seconds/bin) / (number of neurons)
    # bin_width_ms is in ms, so divide by 1000 to get seconds
    firing_rate = psth_counts / (bin_width_ms/1000) / n_neurons
    
    # Plot PSTH
    ax_psth.plot(bin_centers, firing_rate, color='#30a9de', linewidth=2)
    ax_psth.set_ylabel('Firing Rate (Hz)', color=text_color, fontsize=12)
    ax_psth.set_title('PSTH and Raster Plot', color=text_color, fontsize=14)
    
    # Remove x-axis labels from the top plot
    ax_psth.tick_params(labelbottom=False)
    
    # Plot raster
    ax_raster.scatter(raster_data, neuron_ids, s=1, color=raster_color, alpha=0.8)
    ax_raster.set_xlabel('Time (ms)', color=text_color, fontsize=12)
    ax_raster.set_ylabel('Neuron ID', color=text_color, fontsize=12)
    
    # Mark stimulation times if provided
    if stim_times:
        stim_color = '#32ff32'
        for stim_time in stim_times:
            if stim_duration_ms and stim_duration_ms > 0:
                # Draw shaded blocks showing stimulation duration
                ax_psth.axvspan(stim_time, stim_time + stim_duration_ms,
                               color=stim_color, alpha=0.15, linewidth=0)
                ax_raster.axvspan(stim_time, stim_time + stim_duration_ms,
                                 color=stim_color, alpha=0.15, linewidth=0)
            else:
                # Draw vertical lines across both plots
                ax_psth.axvline(x=stim_time, color=stim_color, linewidth=0.8, alpha=0.15)
                ax_raster.axvline(x=stim_time, color=stim_color, linewidth=0.8, alpha=0.15)
    
    # Style the plots based on darkstyle
    for ax in [ax_psth, ax_raster]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['top'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        ax.spines['right'].set_color(spine_color)
        ax.grid(True, alpha=0.2)

    # Add text for scale bar if needed
    ax_raster.text(0.02, 0.98, "Raster", transform=ax_raster.transAxes,
                 color=text_color, fontsize=12, verticalalignment='top')
    ax_psth.text(0.02, 0.98, "PSTH", transform=ax_psth.transAxes,
               color=text_color, fontsize=12, verticalalignment='top')
    
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved PSTH and raster plot to {save_path}")
    
    return fig





# Function to plot overall activity and layer-specific Peri-Stimulus Time Histograms (PSTH)
def Layered_plot_activity_and_layer_psth(network, activity_record, layer_indices, dt=0.1, stim_times=None, bin_width_ms=5.0, save_path="activity_layer_psth.png", darkstyle=False):
    """
    Generates a combined plot showing the overall network spike count over time and
    PSTHs (firing rates) for each specified layer.

    Args:
        network: The network object.
        activity_record (list): List of spiking neuron indices per time step.
        layer_indices (list of tuples): List where each tuple (start_idx, end_idx) defines a layer.
        dt (float): Simulation time step (ms).
        stim_times (list, optional): List of stimulation pulse start times (ms) to mark on plots.
        bin_width_ms (float): Width of the bins for calculating PSTH (ms).
        save_path (str): Path to save the output plot.
        darkstyle (bool): If True, use dark background style. If False, use white background (default: False)
    """
    print(f"Generating combined activity and layer PSTH plot...")

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = '#555555'
        stim_line_color = 'lime'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = '#aaaaaa'
        stim_line_color = 'green'

    num_layers = len(layer_indices)
    total_steps = len(activity_record)
    if total_steps == 0:
        print("Warning: No activity recorded.")
        return None

    # --- Data Preparation ---
    times = np.arange(total_steps) * dt # Time axis for the simulation
    overall_activity = [len(spikes) for spikes in activity_record] # Total spikes per time step

    # --- Plot Setup ---
    # Create figure and GridSpec for layout (overall activity plot taller than PSTHs)
    fig = plt.figure(figsize=(12, 7), facecolor=bg_color)
    height_ratios = [2] + [1] * num_layers # Overall plot twice as tall as each PSTH
    gs = GridSpec(num_layers + 1, 1, height_ratios=height_ratios, hspace=0.3, figure=fig)

    # --- Plot Overall Activity ---
    ax_overall = fig.add_subplot(gs[0]) # Top plot for overall activity
    ax_overall.plot(times, overall_activity, color='#ff7f0e', linewidth=1.0) # Plot total spikes vs time
    ax_overall.set_title("Overall Network Activity", color=text_color)
    ax_overall.set_ylabel("Total Spikes", color=text_color)
    ax_overall.tick_params(axis='x', labelbottom=False) # Hide x-axis labels (shared with bottom plot)
    ax_overall.grid(True, alpha=0.2) # Add subtle grid
    # Mark stimulation times if provided
    if stim_times: # Should ideally be stimulation_record['pulse_starts']
        for stim_time in stim_times:
            ax_overall.axvline(x=stim_time, color=stim_line_color, linestyle='--', linewidth=0.8, alpha=0.15)

    # --- Plot Layer PSTHs ---
    psth_axes = [] # List to store PSTH axes for shared x-axis
    # Define bins for histogram calculation
    bin_edges = np.arange(0, times[-1] + bin_width_ms, bin_width_ms)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # Center points of bins for plotting

    # Iterate through each layer defined in layer_indices
    for i in range(num_layers):
        # Add subplot for this layer's PSTH, sharing x-axis with overall plot
        ax_psth = fig.add_subplot(gs[i + 1], sharex=ax_overall)
        psth_axes.append(ax_psth)
        start_idx, end_idx = layer_indices[i] # Get neuron indices for this layer
        layer_neurons = list(range(start_idx, end_idx))
        num_neurons_in_layer = len(layer_neurons)

        # Handle case where layer has no neurons
        if num_neurons_in_layer == 0:
             ax_psth.text(0.5, 0.5, "No neurons", ha='center', va='center', color='grey', transform=ax_psth.transAxes)
             ax_psth.set_ylabel(f"Layer {i+1}\nRate (Hz)", color=text_color)
             # Set x-label only for the bottom-most plot
             if i == num_layers - 1: ax_psth.set_xlabel("Time (ms)", color=text_color)
             else: ax_psth.tick_params(axis='x', labelbottom=False)
             continue # Skip rest of loop for this empty layer

        # Collect spike times for neurons within this layer
        layer_spike_times = []
        for t_step, active_indices in enumerate(activity_record):
            time_ms = t_step * dt
            # Count spikes occurring in this layer at this time step
            spikes_in_layer = [idx for idx in active_indices if start_idx <= idx < end_idx]
            # Add the spike time for each spike in the layer
            layer_spike_times.extend([time_ms] * len(spikes_in_layer))

        # Calculate histogram (spike counts per bin)
        psth_counts, _ = np.histogram(layer_spike_times, bins=bin_edges)
        # Convert counts to firing rate (Hz): (spikes/bin) / (bin_duration_seconds) / (num_neurons)
        firing_rate = psth_counts / (bin_width_ms / 1000.0) / num_neurons_in_layer if num_neurons_in_layer > 0 else np.zeros_like(psth_counts)

        # Plot the firing rate
        ax_psth.plot(bin_centers, firing_rate, color='#30a9de', linewidth=1.5) # Blue color for PSTH
        ax_psth.set_ylabel(f"Layer {i+1}\nRate (Hz)", color=text_color) # Label y-axis
        ax_psth.grid(True, alpha=0.2) # Add subtle grid

        # Mark stimulation times
        if stim_times: # Should ideally be stimulation_record['pulse_starts']
            for stim_time in stim_times:
                ax_psth.axvline(x=stim_time, color=stim_line_color, linestyle='--', linewidth=.2, alpha=0.1)

        # Set x-label only for the bottom-most plot
        if i == num_layers - 1:
            ax_psth.set_xlabel("Time (ms)", color=text_color)
        else: # Hide x-labels for intermediate plots
            ax_psth.tick_params(axis='x', labelbottom=False)

    # --- Final Styling ---
    # Apply consistent theme styling to all axes
    for ax in [ax_overall] + psth_axes:
        ax.set_facecolor(bg_color)
        ax.tick_params(axis='y', colors=text_color, labelsize='small')
        ax.tick_params(axis='x', colors=text_color, labelsize='small')
        for spine in ax.spines.values(): # Set axis borders color
            spine.set_color(spine_color)

    fig.align_ylabels() # Align y-axis labels vertically
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout, leave space for suptitle
    fig.suptitle("Network Activity and Layer-wise PSTH", color=text_color, fontsize=16) # Overall title

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined activity and PSTH plot to {save_path}")
    return fig


# Function to create layer-specific raster plots
def Layered_plot_layer_wise_raster(network, activity_record, layer_indices, dt=0.1, stim_times=None, save_path="layer_raster.png", darkstyle=False):
    """
    Generates raster plots for each specified layer in separate subplots.

    Args:
        network: The network object.
        activity_record (list): List of spiking neuron indices per time step.
        layer_indices (list of tuples): List where each tuple (start_idx, end_idx) defines a layer.
        dt (float): Simulation time step (ms).
        stim_times (list, optional): List of stimulation pulse start times (ms) to mark on plots.
        save_path (str): Path to save the output plot.
        darkstyle (bool): If True, use dark background style. If False, use white background (default: False)
    """
    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = '#555555'
        raster_color = 'white'
        stim_line_color = 'lime'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = '#aaaaaa'
        raster_color = 'black'
        stim_line_color = 'green'

    num_layers = len(layer_indices)
    total_steps = len(activity_record)
    if total_steps == 0:
        print("Warning: No activity recorded.")
        return None

    # --- Data Preparation ---
    times_ms = np.arange(total_steps) * dt # Time axis

    # --- Plot Setup ---
    # Create figure and GridSpec, adjusting height based on number of layers
    fig = plt.figure(figsize=(12, 2 + num_layers * 1.2), facecolor=bg_color)
    gs = GridSpec(num_layers, 1, hspace=0.1, figure=fig) # Minimal vertical space between plots
    raster_axes = [] # To store axes for linking x-axis
    max_neuron_id = network.n_neurons - 1 # For setting y-limits potentially

    # --- Plot Raster for Each Layer ---
    for i in range(num_layers):
        # Add subplot, sharing x-axis with the first plot if it exists
        sharex = raster_axes[0] if i > 0 else None
        ax_raster = fig.add_subplot(gs[i], sharex=sharex)
        raster_axes.append(ax_raster)
        start_idx, end_idx = layer_indices[i] # Neuron indices for this layer
        layer_neurons = list(range(start_idx, end_idx))
        num_neurons_in_layer = len(layer_neurons)

        # Handle empty layers
        if num_neurons_in_layer == 0:
             ax_raster.text(0.5, 0.5, "No neurons", ha='center', va='center', color='grey', transform=ax_raster.transAxes)
             ax_raster.set_ylabel(f"Layer {i+1}", color=text_color)
             ax_raster.set_yticks([]) # No y-ticks for empty layer
             # Set x-label only for the bottom-most plot
             if i == num_layers - 1: ax_raster.set_xlabel("Time (ms)", color=text_color)
             else: ax_raster.tick_params(axis='x', labelbottom=False)
             continue # Skip plotting for this empty layer

        # Collect spike times and corresponding neuron IDs for this layer
        layer_spike_times = []
        layer_neuron_ids = []
        for t_step, active_indices in enumerate(activity_record):
            time_ms = t_step * dt
            for idx in active_indices:
                if start_idx <= idx < end_idx: # Check if spike belongs to this layer
                    layer_spike_times.append(time_ms)
                    layer_neuron_ids.append(idx)

        # Plot spikes if any occurred in this layer
        if layer_spike_times:
            # Use scatter plot with '|' marker for raster
            ax_raster.scatter(layer_spike_times, layer_neuron_ids, s=2, color=raster_color, alpha=0.8, marker='|')
        # Set y-axis label and limits for this layer
        ax_raster.set_ylabel(f"Layer {i+1}\nNeuron ID", color=text_color, fontsize=10)
        ax_raster.set_ylim(start_idx - 0.5, end_idx - 0.5) # Set limits based on neuron indices

        # Mark stimulation times
        if stim_times: # Should ideally be stimulation_record['pulse_starts']
            for stim_time in stim_times:
                ax_raster.axvline(x=stim_time, color=stim_line_color, linestyle='--', linewidth=0.3, alpha=0.15)

        ax_raster.grid(True, alpha=0.15, axis='x') # Subtle vertical grid lines

        # X-axis label only on the bottom plot
        if i == num_layers - 1:
            ax_raster.set_xlabel("Time (ms)", color=text_color)
        else: # Hide x-labels and ticks for intermediate plots
            ax_raster.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # --- Styling ---
        ax_raster.set_facecolor(bg_color)
        ax_raster.tick_params(axis='y', colors=text_color, labelsize='small') # Y-ticks
        ax_raster.tick_params(axis='x', colors=text_color, labelsize='small') # X-ticks
        for spine in ax_raster.spines.values(): # Axis border colors
            spine.set_color(spine_color)

        # Set y-ticks reasonably (e.g., 5 ticks or fewer)
        if num_neurons_in_layer > 1:
             y_ticks = np.linspace(start_idx, end_idx -1, min(5, num_neurons_in_layer), dtype=int)
             ax_raster.set_yticks(y_ticks)
        elif num_neurons_in_layer == 1: # Single neuron in layer
             ax_raster.set_yticks([start_idx])

    # --- Final Adjustments ---
    fig.align_ylabels(raster_axes) # Align y-labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout, leave space for title
    fig.suptitle("Layer-wise Raster Plots", color=text_color, fontsize=16, y=0.99) # Add overall title

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


# Function to visualize distance-dependent weights and delays
def Layered_visualize_distance_dependences(network, pos, neuron_idx, base_transmission_delay,
                                   network_figsize=(10, 10), # Default size for network plot (not currently generated)
                                   scatter_figsize=(10, 10), # Size for scatter plots
                                   save_path_base="distance_dependence", darkstyle=False):
    """
    Creates scatter plots showing connection weights and delays vs. distance
    for outgoing connections from a specified neuron.

    Args:
        network: The network object.
        pos (dict): Dictionary mapping neuron indices to (x, y) positions.
        neuron_idx (int): Index of the source neuron to visualize connections from.
        base_transmission_delay (float): The base delay value used in delay calculations.
        network_figsize (tuple): Size for a potential future network plot (currently unused).
        scatter_figsize (tuple): Size for the weight and delay scatter plots.
        save_path_base (str): Base filename for saving the plots (suffixes added).
        darkstyle (bool): If True, use dark background style. If False, use white background (default: False)
    """
    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        zero_line_color = 'white'
        weight_scatter_color = 'yellow'
        delay_scatter_color = 'cyan'
        theo_line_color = 'red'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        zero_line_color = 'black'
        weight_scatter_color = '#cc8800'  # Darker yellow/gold
        delay_scatter_color = '#0088aa'   # Darker cyan
        theo_line_color = 'red'

    print(f"Generating distance dependence plots for Neuron {neuron_idx}...")

    # Check if position data exists for the selected neuron
    if neuron_idx not in pos:
        print(f"Error: Position for neuron {neuron_idx} not found.")
        return None, None # Return None for both expected figures

    center_pos = pos[neuron_idx] # Position of the source neuron
    outgoing_data = [] # List to store (target_idx, distance, weight, delay)

    # --- Calculate Max Distance for Normalization ---
    # Estimate the maximum possible distance in the layout for delay scaling
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    # Calculate diagonal distance across the bounding box of all neuron positions
    max_possible_dist = np.sqrt((max(all_x) - min(all_x))**2 + (max(all_y) - min(all_y))**2) if len(pos)>1 else 1.0
    if max_possible_dist < 1e-6: max_possible_dist = 1.0 # Avoid division by zero if layout is collapsed

    # --- Collect Outgoing Connection Data ---
    # Iterate through neurons targeted by the selected neuron
    for target_idx in network.graph.successors(neuron_idx):
        if target_idx in pos: # Ensure target has position data
            target_pos = pos[target_idx]
            # Calculate Euclidean distance
            dist = np.sqrt((center_pos[0] - target_pos[0])**2 + (center_pos[1] - target_pos[1])**2)
            # Get weight and delay from network matrices
            weight = network.weights[neuron_idx, target_idx]
            delay = network.delays[neuron_idx, target_idx]
            # Store the data
            outgoing_data.append((target_idx, dist, weight, delay))

    # Note: The network visualization part from the original function seems to be missing here.
    # Only the scatter plots are generated below.

    # --- FIGURE 2: WEIGHT VS DISTANCE Scatter Plot ---
    weight_scatter_fig, ax_weight = plt.subplots(figsize=scatter_figsize, facecolor=bg_color)
    if outgoing_data:
        distances = [d for _, d, _, _ in outgoing_data]
        weights = [w for _, _, w, _ in outgoing_data]
        # Create scatter plot of weight vs distance
        ax_weight.scatter(distances, weights, color=weight_scatter_color, alpha=0.7, label='Outgoing Weights', s=30)
    # Add styling
    ax_weight.axhline(y=0, color=zero_line_color, linestyle='-', linewidth=0.5, alpha=0.5) # Zero line
    ax_weight.grid(True, alpha=0.2) # Subtle grid
    ax_weight.set_xlabel('Distance', color=text_color, fontsize=12)
    ax_weight.set_ylabel('Synaptic Weight', color=text_color, fontsize=12)
    ax_weight.set_title(f'Weight vs. Distance (Neuron {neuron_idx})', color=text_color, fontsize=14)
    ax_weight.legend(loc='best', framealpha=0.7, fontsize=10) # Add legend
    ax_weight.set_facecolor(bg_color)
    ax_weight.tick_params(colors=text_color, labelsize=10)
    for spine in ax_weight.spines.values(): spine.set_color(spine_color)
    weight_scatter_fig.tight_layout() # Adjust layout
    # Save figure
    if save_path_base:
        w_save_path = f"{save_path_base}_weight_scatter.png"
        weight_scatter_fig.savefig(w_save_path, dpi=150, facecolor=bg_color, bbox_inches='tight')
        print(f"Saved weight-distance relationship plot to {w_save_path}")

    # --- FIGURE 3: DELAY VS DISTANCE Scatter Plot ---
    delay_scatter_fig, ax_delay = plt.subplots(figsize=scatter_figsize, facecolor=bg_color)
    if outgoing_data:
        distances = [d for _, d, _, _ in outgoing_data]
        delays = [dl for _, _, _, dl in outgoing_data]
        # Create scatter plot of delay vs distance
        ax_delay.scatter(distances, delays, color=delay_scatter_color, alpha=0.7, label='Outgoing Delays', s=30)
        # Plot theoretical delay line based on the formula used during connection creation
        dist_range = np.linspace(0, max(distances) if distances else 1, 100) # Range of distances
        # Formula used in run_6_layer_experiment: base * (0.5 + 0.5 * dist / max_dist)
        theo_delays = base_transmission_delay * (0.5 + 0.5 * dist_range / max_possible_dist)
        ax_delay.plot(dist_range, theo_delays, '--', color=theo_line_color, alpha=0.7, label='Theoretical Delay (Linear)')
    # Add styling
    ax_delay.grid(True, alpha=0.2) # Subtle grid
    ax_delay.set_xlabel('Distance', color=text_color, fontsize=12)
    ax_delay.set_ylabel('Synaptic Delay (ms)', color=text_color, fontsize=12)
    ax_delay.set_title(f'Delay vs. Distance (Neuron {neuron_idx})', color=text_color, fontsize=14)
    ax_delay.legend(loc='best', framealpha=0.7, fontsize=10) # Add legend
    ax_delay.set_facecolor(bg_color)
    ax_delay.tick_params(colors=text_color, labelsize=10)
    for spine in ax_delay.spines.values(): spine.set_color(spine_color)
    delay_scatter_fig.tight_layout() # Adjust layout
    # Save figure
    if save_path_base:
        d_save_path = f"{save_path_base}_delay_scatter.png"
        delay_scatter_fig.savefig(d_save_path, dpi=150, facecolor=bg_color, bbox_inches='tight')
        print(f"Saved delay-distance relationship plot to {d_save_path}")

    # Return figure handles (Network figure handle is None as it wasn't created)
    return weight_scatter_fig, delay_scatter_fig


def plot_ei_psth_and_raster(network, activity_record, bin_size=10, dt=0.1,
                             figsize=(14, 10), dpi=150,
                             save_path="ei_psth_raster_plot.png", darkstyle=False,
                             stim_times=None, stim_duration_ms=None):
    """
    Create separate PSTH and raster plots for excitatory and inhibitory neurons.
    Excitatory neurons are shown in red, inhibitory in blue.

    Parameters:
    -----------
    network : CircularNeuronalNetwork
        The network object containing neuron information
    activity_record : list
        List of lists containing active neuron indices at each time step
    bin_size : int
        Size of bins for PSTH in time steps
    dt : float
        Time step size in ms
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style. If False, use white background
    stim_times : list or None
        List of stimulation time points (in ms) - draws green vertical lines or blocks
    stim_duration_ms : float or None
        Duration of each stimulation period in ms. If provided, draws shaded blocks
        instead of vertical lines.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Colors for E/I
    exc_color = '#e74c3c'  # Red for excitatory
    inh_color = '#3498db'  # Blue for inhibitory

    # Identify excitatory and inhibitory neurons
    # Support both object-based networks (network.neurons[i].is_inhibitory)
    # and vectorized networks (network.is_inhibitory as numpy array)
    exc_neurons = set()
    inh_neurons = set()

    if hasattr(network, 'is_inhibitory') and hasattr(network.is_inhibitory, '__iter__') and not hasattr(network, 'neurons'):
        # Vectorized network: is_inhibitory is a numpy array
        for i, is_inh in enumerate(network.is_inhibitory):
            if is_inh:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)
    else:
        # Object-based network: iterate through neuron objects
        for i, neuron in enumerate(network.neurons):
            if neuron.is_inhibitory:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)

    print(f"Network composition: {len(exc_neurons)} excitatory, {len(inh_neurons)} inhibitory neurons")

    # Create figure with GridSpec: 4 rows (exc PSTH, inh PSTH, exc raster, inh raster)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(4, 1, height_ratios=[1, 1, 2, 2], hspace=0.08)

    ax_exc_psth = fig.add_subplot(gs[0])
    ax_inh_psth = fig.add_subplot(gs[1], sharex=ax_exc_psth)
    ax_exc_raster = fig.add_subplot(gs[2], sharex=ax_exc_psth)
    ax_inh_raster = fig.add_subplot(gs[3], sharex=ax_exc_psth)

    # Calculate time axis
    n_steps = len(activity_record)
    time_ms = np.arange(n_steps) * dt

    # Separate activity by neuron type
    exc_raster_times = []
    exc_raster_ids = []
    inh_raster_times = []
    inh_raster_ids = []

    for t, active_neurons in enumerate(activity_record):
        for neuron_idx in active_neurons:
            if neuron_idx in exc_neurons:
                exc_raster_times.append(time_ms[t])
                exc_raster_ids.append(neuron_idx)
            elif neuron_idx in inh_neurons:
                inh_raster_times.append(time_ms[t])
                inh_raster_ids.append(neuron_idx)

    # Calculate PSTH for each population
    bin_width_ms = bin_size * dt
    bin_edges_ms = np.arange(0, time_ms[-1] + bin_width_ms, bin_width_ms)
    bin_centers = (bin_edges_ms[:-1] + bin_edges_ms[1:]) / 2

    # Excitatory PSTH
    exc_counts, _ = np.histogram(exc_raster_times, bins=bin_edges_ms)
    exc_firing_rate = exc_counts / (bin_width_ms / 1000) / len(exc_neurons) if len(exc_neurons) > 0 else np.zeros_like(exc_counts)

    # Inhibitory PSTH
    inh_counts, _ = np.histogram(inh_raster_times, bins=bin_edges_ms)
    inh_firing_rate = inh_counts / (bin_width_ms / 1000) / len(inh_neurons) if len(inh_neurons) > 0 else np.zeros_like(inh_counts)

    # Plot Excitatory PSTH
    ax_exc_psth.plot(bin_centers, exc_firing_rate, color=exc_color, linewidth=2)
    ax_exc_psth.set_ylabel('Firing Rate (Hz)', color=text_color, fontsize=10)
    ax_exc_psth.set_title('Excitatory and Inhibitory PSTH and Raster Plots', color=text_color, fontsize=14)
    ax_exc_psth.tick_params(labelbottom=False)
    ax_exc_psth.text(0.02, 0.85, "Excitatory PSTH", transform=ax_exc_psth.transAxes,
                     color=exc_color, fontsize=11, fontweight='bold', verticalalignment='top')

    # Plot Excitatory Raster
    if exc_raster_times:
        ax_exc_raster.scatter(exc_raster_times, exc_raster_ids, s=0.5, color=exc_color, alpha=0.6)
    ax_exc_raster.set_ylabel('Neuron ID', color=text_color, fontsize=10)
    ax_exc_raster.tick_params(labelbottom=False)
    ax_exc_raster.text(0.02, 0.95, "Excitatory Raster", transform=ax_exc_raster.transAxes,
                       color=exc_color, fontsize=11, fontweight='bold', verticalalignment='top')

    # Plot Inhibitory PSTH
    ax_inh_psth.plot(bin_centers, inh_firing_rate, color=inh_color, linewidth=2)
    ax_inh_psth.set_ylabel('Firing Rate (Hz)', color=text_color, fontsize=10)
    ax_inh_psth.tick_params(labelbottom=False)
    ax_inh_psth.text(0.02, 0.85, "Inhibitory PSTH", transform=ax_inh_psth.transAxes,
                     color=inh_color, fontsize=11, fontweight='bold', verticalalignment='top')

    # Plot Inhibitory Raster
    if inh_raster_times:
        ax_inh_raster.scatter(inh_raster_times, inh_raster_ids, s=0.5, color=inh_color, alpha=0.6)
    ax_inh_raster.set_xlabel('Time (ms)', color=text_color, fontsize=12)
    ax_inh_raster.set_ylabel('Neuron ID', color=text_color, fontsize=10)
    ax_inh_raster.text(0.02, 0.95, "Inhibitory Raster", transform=ax_inh_raster.transAxes,
                       color=inh_color, fontsize=11, fontweight='bold', verticalalignment='top')

    # Style all axes
    for ax in [ax_exc_psth, ax_exc_raster, ax_inh_psth, ax_inh_raster]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        ax.spines['bottom'].set_color(spine_color)
        ax.spines['top'].set_color(spine_color)
        ax.spines['left'].set_color(spine_color)
        ax.spines['right'].set_color(spine_color)
        ax.grid(True, alpha=0.2)

    # Draw stimulation time markers
    if stim_times:
        stim_color = '#32ff32'
        for stim_time in stim_times:
            for ax in [ax_exc_psth, ax_exc_raster, ax_inh_psth, ax_inh_raster]:
                if stim_duration_ms and stim_duration_ms > 0:
                    # Draw shaded blocks showing stimulation duration
                    ax.axvspan(stim_time, stim_time + stim_duration_ms,
                              color=stim_color, alpha=0.15, linewidth=0)
                else:
                    ax.axvline(x=stim_time, color=stim_color, linewidth=0.8, alpha=0.15)

    plt.tight_layout()

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved E/I PSTH and raster plot to {save_path}")

    return fig


def plot_oscillation_frequency_analysis(activity_record, dt=0.1, figsize=(14, 10), dpi=150,
                                         save_path="oscillation_frequency_analysis.png", darkstyle=False):
    """
    Perform frequency analysis on network activity to identify and verify oscillation frequencies.
    Uses FFT and spectrogram to decompose the signal.

    Parameters:
    -----------
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style. If False, use white background

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    freq_info : dict
        Dictionary containing frequency analysis results
    """
    from scipy import signal
    from scipy.fft import fft, fftfreq
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Calculate activity time series (total spikes per timestep)
    activity = np.array([len(spikes) for spikes in activity_record], dtype=float)
    n_samples = len(activity)
    time_ms = np.arange(n_samples) * dt

    # Sampling frequency in Hz
    fs = 1000.0 / dt  # Convert from ms to Hz

    # Remove mean (detrend) for better FFT
    activity_detrended = activity - np.mean(activity)

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # --- Panel 1: Raw activity time series ---
    ax_time = fig.add_subplot(gs[0, :])
    ax_time.plot(time_ms, activity, color='#30a9de', linewidth=1)
    ax_time.set_xlabel('Time (ms)', color=text_color)
    ax_time.set_ylabel('Spike Count', color=text_color)
    ax_time.set_title('Network Activity Time Series', color=text_color, fontsize=12)

    # --- Panel 2: Power Spectrum (FFT) ---
    ax_fft = fig.add_subplot(gs[1, 0])

    # Compute FFT
    n_fft = n_samples
    yf = fft(activity_detrended)
    xf = fftfreq(n_fft, dt / 1000.0)  # Convert dt to seconds for Hz

    # Only take positive frequencies
    positive_mask = xf > 0
    freqs = xf[positive_mask]
    power = np.abs(yf[positive_mask]) ** 2

    # Normalize power
    power = power / np.max(power)

    # Plot power spectrum (focus on 0-500 Hz for neural oscillations)
    freq_mask = freqs <= 500
    ax_fft.plot(freqs[freq_mask], power[freq_mask], color='#e74c3c', linewidth=1.5)
    ax_fft.set_xlabel('Frequency (Hz)', color=text_color)
    ax_fft.set_ylabel('Normalized Power', color=text_color)
    ax_fft.set_title('Power Spectrum (FFT)', color=text_color, fontsize=12)

    # Find peak frequency
    peak_idx = np.argmax(power[freq_mask])
    peak_freq = freqs[freq_mask][peak_idx]
    peak_power = power[freq_mask][peak_idx]
    ax_fft.axvline(x=peak_freq, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8)
    ax_fft.text(peak_freq + 10, peak_power * 0.9, f'Peak: {peak_freq:.1f} Hz',
                color='#2ecc71', fontsize=10, fontweight='bold')

    # Mark frequency bands
    bands = {
        'Delta (1-4 Hz)': (1, 4, '#9b59b6'),
        'Theta (4-8 Hz)': (4, 8, '#3498db'),
        'Alpha (8-13 Hz)': (8, 13, '#2ecc71'),
        'Beta (13-30 Hz)': (13, 30, '#f39c12'),
        'Gamma (30-100 Hz)': (30, 100, '#e74c3c'),
        'High Gamma (100-500 Hz)': (100, 500, '#e91e63')
    }

    # --- Panel 3: Power by frequency band ---
    ax_bands = fig.add_subplot(gs[1, 1])

    band_powers = {}
    band_names = []
    band_power_values = []
    band_colors = []

    for band_name, (low, high, color) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        if np.any(band_mask):
            band_power = np.sum(power[band_mask])
            band_powers[band_name] = band_power
            band_names.append(band_name.split(' ')[0])  # Just the band name
            band_power_values.append(band_power)
            band_colors.append(color)

    # Bar chart of power by band
    bars = ax_bands.bar(range(len(band_names)), band_power_values, color=band_colors, alpha=0.8)
    ax_bands.set_xticks(range(len(band_names)))
    ax_bands.set_xticklabels(band_names, rotation=45, ha='right', fontsize=9)
    ax_bands.set_ylabel('Total Power', color=text_color)
    ax_bands.set_title('Power by Frequency Band', color=text_color, fontsize=12)

    # Highlight the dominant band
    if band_power_values:
        dominant_idx = np.argmax(band_power_values)
        bars[dominant_idx].set_edgecolor('white')
        bars[dominant_idx].set_linewidth(3)

    # --- Panel 4: Spectrogram ---
    ax_spec = fig.add_subplot(gs[2, :])

    # Compute spectrogram
    nperseg = min(256, n_samples // 4)  # Window size
    noverlap = nperseg // 2
    f_spec, t_spec, Sxx = signal.spectrogram(activity_detrended, fs=fs,
                                              nperseg=nperseg, noverlap=noverlap)

    # Convert time to ms
    t_spec_ms = t_spec * 1000

    # Plot spectrogram (focus on 0-500 Hz)
    freq_limit = 500
    freq_limit_idx = np.searchsorted(f_spec, freq_limit)

    pcm = ax_spec.pcolormesh(t_spec_ms, f_spec[:freq_limit_idx],
                              10 * np.log10(Sxx[:freq_limit_idx, :] + 1e-10),
                              shading='gouraud', cmap='viridis')
    ax_spec.set_xlabel('Time (ms)', color=text_color)
    ax_spec.set_ylabel('Frequency (Hz)', color=text_color)
    ax_spec.set_title('Spectrogram (Time-Frequency Analysis)', color=text_color, fontsize=12)

    # Add colorbar
    cbar = fig.colorbar(pcm, ax=ax_spec, label='Power (dB)')
    cbar.ax.yaxis.label.set_color(text_color)
    cbar.ax.tick_params(colors=text_color)

    # Add horizontal lines for gamma band boundaries
    ax_spec.axhline(y=30, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax_spec.axhline(y=100, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax_spec.text(t_spec_ms[-1] * 0.02, 65, 'Gamma', color='white', fontsize=10, alpha=0.8)

    # Style all axes
    for ax in [ax_time, ax_fft, ax_bands, ax_spec]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, alpha=0.2)

    # Add summary text
    dominant_band = band_names[np.argmax(band_power_values)] if band_power_values else "Unknown"
    summary_text = (f"Peak Frequency: {peak_freq:.1f} Hz\n"
                    f"Dominant Band: {dominant_band}\n"
                    f"Sampling Rate: {fs:.0f} Hz")

    fig.text(0.02, 0.02, summary_text, transform=fig.transFigure,
             fontsize=10, color=text_color, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=spine_color, alpha=0.8))

    # Determine if this is gamma oscillation
    gamma_power = band_powers.get('Gamma (30-100 Hz)', 0)
    total_power = sum(band_power_values) if band_power_values else 1
    gamma_fraction = gamma_power / total_power if total_power > 0 else 0

    is_gamma = (30 <= peak_freq <= 100) or (gamma_fraction > 0.3)

    if is_gamma:
        verdict = f"GAMMA OSCILLATIONS CONFIRMED\nPeak at {peak_freq:.1f} Hz"
        verdict_color = '#2ecc71'
    elif peak_freq > 100:
        verdict = f"HIGH-FREQUENCY OSCILLATIONS\nPeak at {peak_freq:.1f} Hz (above gamma)"
        verdict_color = '#e91e63'
    else:
        verdict = f"NON-GAMMA OSCILLATIONS\nPeak at {peak_freq:.1f} Hz"
        verdict_color = '#f39c12'

    fig.text(0.98, 0.02, verdict, transform=fig.transFigure,
             fontsize=12, color=verdict_color, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=verdict_color, alpha=0.9))

    plt.tight_layout()

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved oscillation frequency analysis to {save_path}")

    # Return frequency info
    freq_info = {
        'peak_frequency_hz': peak_freq,
        'dominant_band': dominant_band,
        'band_powers': band_powers,
        'gamma_fraction': gamma_fraction,
        'is_gamma': is_gamma,
        'sampling_rate_hz': fs
    }

    return fig, freq_info


def plot_ei_frequency_analysis(network, activity_record, dt=0.1, figsize=(14, 8), dpi=150,
                                save_path="ei_frequency_analysis.png", darkstyle=False):
    """
    Perform separate frequency analysis on excitatory and inhibitory populations,
    then plot both power spectra on the same axis for comparison.

    Parameters:
    -----------
    network : CircularNeuronalNetwork or SphericalNeuronalNetwork
        The network object containing neuron information
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style. If False, use white background

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    freq_info : dict
        Dictionary containing frequency analysis results for both populations
    """
    from scipy.fft import fft, fftfreq
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Colors for E/I
    exc_color = '#e74c3c'  # Red for excitatory
    inh_color = '#3498db'  # Blue for inhibitory

    # Identify excitatory and inhibitory neurons
    # Support both object-based networks (network.neurons[i].is_inhibitory)
    # and vectorized networks (network.is_inhibitory as numpy array)
    exc_neurons = set()
    inh_neurons = set()

    if hasattr(network, 'is_inhibitory') and hasattr(network.is_inhibitory, '__iter__') and not hasattr(network, 'neurons'):
        # Vectorized network: is_inhibitory is a numpy array
        for i, is_inh in enumerate(network.is_inhibitory):
            if is_inh:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)
    else:
        # Object-based network: iterate through neuron objects
        for i, neuron in enumerate(network.neurons):
            if neuron.is_inhibitory:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)

    print(f"Frequency analysis: {len(exc_neurons)} excitatory, {len(inh_neurons)} inhibitory neurons")

    # Calculate activity time series for each population
    n_samples = len(activity_record)
    time_ms = np.arange(n_samples) * dt

    # Count spikes per timestep for each population
    exc_activity = np.zeros(n_samples, dtype=float)
    inh_activity = np.zeros(n_samples, dtype=float)

    for t, active_neurons in enumerate(activity_record):
        for neuron_idx in active_neurons:
            if neuron_idx in exc_neurons:
                exc_activity[t] += 1
            elif neuron_idx in inh_neurons:
                inh_activity[t] += 1

    # Sampling frequency in Hz
    fs = 1000.0 / dt

    # Detrend (remove mean)
    exc_detrended = exc_activity - np.mean(exc_activity)
    inh_detrended = inh_activity - np.mean(inh_activity)

    # Compute FFT for both populations
    exc_fft = fft(exc_detrended)
    inh_fft = fft(inh_detrended)
    freqs = fftfreq(n_samples, dt / 1000.0)

    # Only take positive frequencies
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]

    exc_power = np.abs(exc_fft[positive_mask]) ** 2
    inh_power = np.abs(inh_fft[positive_mask]) ** 2

    # Normalize each power spectrum by its own max
    exc_power_norm = exc_power / np.max(exc_power) if np.max(exc_power) > 0 else exc_power
    inh_power_norm = inh_power / np.max(inh_power) if np.max(inh_power) > 0 else inh_power

    # Focus on neural oscillation frequencies (0-500 Hz)
    freq_mask = freqs_pos <= 500
    freqs_plot = freqs_pos[freq_mask]
    exc_power_plot = exc_power_norm[freq_mask]
    inh_power_plot = inh_power_norm[freq_mask]

    # Find peak frequencies
    exc_peak_idx = np.argmax(exc_power_plot)
    inh_peak_idx = np.argmax(inh_power_plot)
    exc_peak_freq = freqs_plot[exc_peak_idx]
    inh_peak_freq = freqs_plot[inh_peak_idx]

    # Create figure with GridSpec: time series on top, power spectrum below
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.25)

    # --- Panel 1: Activity time series ---
    ax_time = fig.add_subplot(gs[0])
    ax_time.plot(time_ms, exc_activity, color=exc_color, linewidth=0.8, alpha=0.8, label='Excitatory')
    ax_time.plot(time_ms, inh_activity, color=inh_color, linewidth=0.8, alpha=0.8, label='Inhibitory')
    ax_time.set_xlabel('Time (ms)', color=text_color)
    ax_time.set_ylabel('Spike Count', color=text_color)
    ax_time.set_title('E/I Activity Time Series', color=text_color, fontsize=12)
    ax_time.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)

    # --- Panel 2: Power Spectrum comparison ---
    ax_fft = fig.add_subplot(gs[1])

    # Plot both power spectra
    ax_fft.plot(freqs_plot, exc_power_plot, color=exc_color, linewidth=2, alpha=0.9, label='Excitatory')
    ax_fft.plot(freqs_plot, inh_power_plot, color=inh_color, linewidth=2, alpha=0.9, label='Inhibitory')

    # Mark peak frequencies
    ax_fft.axvline(x=exc_peak_freq, color=exc_color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax_fft.axvline(x=inh_peak_freq, color=inh_color, linestyle='--', linewidth=1.5, alpha=0.7)

    # Add peak frequency annotations
    ax_fft.text(exc_peak_freq + 5, 0.95, f'E peak: {exc_peak_freq:.1f} Hz',
                color=exc_color, fontsize=10, fontweight='bold', verticalalignment='top')
    ax_fft.text(inh_peak_freq + 5, 0.85, f'I peak: {inh_peak_freq:.1f} Hz',
                color=inh_color, fontsize=10, fontweight='bold', verticalalignment='top')

    ax_fft.set_xlabel('Frequency (Hz)', color=text_color, fontsize=12)
    ax_fft.set_ylabel('Normalized Power', color=text_color, fontsize=12)
    ax_fft.set_title('E/I Power Spectrum Comparison', color=text_color, fontsize=14)
    ax_fft.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)

    # Add frequency band shading for reference
    band_colors = {
        'Gamma (30-100 Hz)': (30, 100, '#2ecc71', 0.1),
        'High Gamma (100-500 Hz)': (100, 500, '#9b59b6', 0.05)
    }
    for band_name, (low, high, color, alpha) in band_colors.items():
        ax_fft.axvspan(low, high, alpha=alpha, color=color, label=band_name)

    # Style both axes
    for ax in [ax_time, ax_fft]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, alpha=0.2)

    # Determine gamma status for each population
    def is_gamma_freq(freq):
        return 30 <= freq <= 100

    exc_is_gamma = is_gamma_freq(exc_peak_freq)
    inh_is_gamma = is_gamma_freq(inh_peak_freq)

    # Summary text
    summary_lines = [
        f"Excitatory peak: {exc_peak_freq:.1f} Hz {'(Gamma)' if exc_is_gamma else ''}",
        f"Inhibitory peak: {inh_peak_freq:.1f} Hz {'(Gamma)' if inh_is_gamma else ''}",
        f"Sampling rate: {fs:.0f} Hz"
    ]
    summary_text = '\n'.join(summary_lines)

    fig.text(0.02, 0.02, summary_text, transform=fig.transFigure,
             fontsize=10, color=text_color, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=spine_color, alpha=0.8))

    plt.tight_layout()

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved E/I frequency analysis to {save_path}")

    # Return frequency info
    freq_info = {
        'exc_peak_frequency_hz': exc_peak_freq,
        'inh_peak_frequency_hz': inh_peak_freq,
        'exc_is_gamma': exc_is_gamma,
        'inh_is_gamma': inh_is_gamma,
        'sampling_rate_hz': fs
    }

    return fig, freq_info


def plot_ei_synchrony_analysis(network, activity_record, dt=0.1,
                                figsize=(14, 12), dpi=150,
                                save_path="ei_synchrony_analysis.png", darkstyle=False):
    """
    Analyze population synchrony at multiple timescales by comparing unique neurons
    firing vs total spikes per bin.

    Key insight: With refractory periods (E: ~4ms, I: ~2.5ms), small bins will always
    show sync_index ≈ 1.0. We need larger bins to detect bursting vs population sync.

    Burst Index = total_spikes / unique_neurons
    - burst_idx ≈ 1.0 → each neuron fires once per bin (population synchrony)
    - burst_idx > 1.0 → neurons firing multiple times per bin (bursting)

    At bin_size = 10ms spanning ~2-3 oscillation cycles at 240 Hz:
    - If same neurons fire repeatedly → bursting (burst_idx >> 1)
    - If different neurons each cycle → population sync (burst_idx ≈ cycles)

    Parameters:
    -----------
    network : CircularNeuronalNetwork or SphericalNeuronalNetwork
        The network object containing neuron information
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style. If False, use white background

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    sync_info : dict
        Dictionary containing synchrony analysis results at multiple timescales
    """
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Colors for E/I
    exc_color = '#e74c3c'  # Red for excitatory
    inh_color = '#3498db'  # Blue for inhibitory
    unique_color = '#27ae60'  # Green for unique neurons

    # Identify excitatory and inhibitory neurons
    # Support both object-based networks (network.neurons[i].is_inhibitory)
    # and vectorized networks (network.is_inhibitory as numpy array)
    exc_neurons = set()
    inh_neurons = set()

    if hasattr(network, 'is_inhibitory') and hasattr(network.is_inhibitory, '__iter__') and not hasattr(network, 'neurons'):
        # Vectorized network: is_inhibitory is a numpy array
        for i, is_inh in enumerate(network.is_inhibitory):
            if is_inh:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)
    else:
        # Object-based network: iterate through neuron objects
        for i, neuron in enumerate(network.neurons):
            if neuron.is_inhibitory:
                inh_neurons.add(i)
            else:
                exc_neurons.add(i)

    print(f"Synchrony analysis: {len(exc_neurons)} excitatory, {len(inh_neurons)} inhibitory neurons")

    # Multiple bin sizes to capture different timescales
    # Small bins (< refractory) won't show bursting; larger bins will
    bin_sizes_ms = [2, 5, 10, 20, 40, 80]

    n_timesteps = len(activity_record)
    duration_ms = n_timesteps * dt

    # Create figure with 3x2 grid for different bin sizes
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25)

    # Store results for each bin size
    all_results = {}

    for idx, bin_size_ms in enumerate(bin_sizes_ms):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        steps_per_bin = max(1, int(bin_size_ms / dt))
        n_bins = n_timesteps // steps_per_bin

        # Arrays to store per-bin data
        bin_times = np.zeros(n_bins)
        exc_total_spikes = np.zeros(n_bins)
        exc_unique_neurons = np.zeros(n_bins)
        inh_total_spikes = np.zeros(n_bins)
        inh_unique_neurons = np.zeros(n_bins)

        # Process each bin
        for bin_idx in range(n_bins):
            start_step = bin_idx * steps_per_bin
            end_step = min(start_step + steps_per_bin, n_timesteps)
            bin_times[bin_idx] = (start_step + end_step) / 2 * dt  # Bin center time

            # Collect all spikes in this bin
            exc_spikes_in_bin = []
            inh_spikes_in_bin = []

            for t in range(start_step, end_step):
                for neuron_idx in activity_record[t]:
                    if neuron_idx in exc_neurons:
                        exc_spikes_in_bin.append(neuron_idx)
                    elif neuron_idx in inh_neurons:
                        inh_spikes_in_bin.append(neuron_idx)

            # Count total spikes and unique neurons
            exc_total_spikes[bin_idx] = len(exc_spikes_in_bin)
            exc_unique_neurons[bin_idx] = len(set(exc_spikes_in_bin))
            inh_total_spikes[bin_idx] = len(inh_spikes_in_bin)
            inh_unique_neurons[bin_idx] = len(set(inh_spikes_in_bin))

        # Calculate burst index: total/unique (1.0 = no bursting, >1 = bursting)
        exc_burst_idx = np.divide(exc_total_spikes, exc_unique_neurons,
                                   out=np.ones_like(exc_total_spikes),
                                   where=exc_unique_neurons > 0)
        inh_burst_idx = np.divide(inh_total_spikes, inh_unique_neurons,
                                   out=np.ones_like(inh_total_spikes),
                                   where=inh_unique_neurons > 0)

        # Mean burst index (only for bins with activity)
        exc_mean_burst = np.mean(exc_burst_idx[exc_unique_neurons > 0]) if np.any(exc_unique_neurons > 0) else 1.0
        inh_mean_burst = np.mean(inh_burst_idx[inh_unique_neurons > 0]) if np.any(inh_unique_neurons > 0) else 1.0

        # Plot total spikes (solid) and unique neurons (dashed)
        ax.plot(bin_times, exc_total_spikes, color=exc_color, linewidth=1.5, alpha=0.9, label='E total')
        ax.plot(bin_times, exc_unique_neurons, color=exc_color, linewidth=1.5, alpha=0.9, linestyle='--', label='E unique')
        ax.plot(bin_times, inh_total_spikes, color=inh_color, linewidth=1.5, alpha=0.9, label='I total')
        ax.plot(bin_times, inh_unique_neurons, color=inh_color, linewidth=1.5, alpha=0.9, linestyle='--', label='I unique')

        # Fill between to show the gap (bursting indicator)
        ax.fill_between(bin_times, exc_unique_neurons, exc_total_spikes, alpha=0.15, color=exc_color)
        ax.fill_between(bin_times, inh_unique_neurons, inh_total_spikes, alpha=0.15, color=inh_color)

        # Title with burst index
        ax.set_title(f'Bin: {bin_size_ms}ms | E burst: {exc_mean_burst:.2f}, I burst: {inh_mean_burst:.2f}',
                     color=text_color, fontsize=10)
        ax.set_xlabel('Time (ms)', color=text_color, fontsize=9)
        ax.set_ylabel('Spike count', color=text_color, fontsize=9)

        if idx == 0:
            ax.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color,
                      labelcolor=text_color, fontsize=8)

        # Style
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, alpha=0.2)

        # Store results
        all_results[bin_size_ms] = {
            'bin_times': bin_times,
            'exc_total_spikes': exc_total_spikes,
            'exc_unique_neurons': exc_unique_neurons,
            'exc_burst_idx': exc_burst_idx,
            'exc_mean_burst': exc_mean_burst,
            'inh_total_spikes': inh_total_spikes,
            'inh_unique_neurons': inh_unique_neurons,
            'inh_burst_idx': inh_burst_idx,
            'inh_mean_burst': inh_mean_burst,
        }

    # Interpretation helper
    def interpret_burst(burst_val, bin_ms, tau_ref):
        # Max possible firings per bin given refractory period
        max_possible = bin_ms / tau_ref
        if burst_val < 1.1:
            return "population sync"
        elif burst_val < max_possible * 0.5:
            return "mild bursting"
        elif burst_val < max_possible * 0.8:
            return "moderate bursting"
        else:
            return "max-rate bursting"

    # Summary text
    # Use 10ms bin as primary reference (spans ~2-3 cycles at gamma frequencies)
    ref_bin = 10 if 10 in all_results else bin_sizes_ms[2]
    ref_data = all_results[ref_bin]

    summary_lines = [
        "Burst Index = total_spikes / unique_neurons",
        "  1.0 = each neuron fires once (population sync)",
        "  >1.0 = neurons fire multiple times (bursting)",
        "",
        f"At {ref_bin}ms bins (reference):",
        f"  E burst idx: {ref_data['exc_mean_burst']:.2f} - {interpret_burst(ref_data['exc_mean_burst'], ref_bin, 4.0)}",
        f"  I burst idx: {ref_data['inh_mean_burst']:.2f} - {interpret_burst(ref_data['inh_mean_burst'], ref_bin, 2.5)}",
        "",
        f"E τ_ref=4ms → max {ref_bin/4:.1f} spikes/{ref_bin}ms",
        f"I τ_ref=2.5ms → max {ref_bin/2.5:.1f} spikes/{ref_bin}ms",
    ]
    summary_text = '\n'.join(summary_lines)

    fig.text(0.02, 0.01, summary_text, transform=fig.transFigure,
             fontsize=8, color=text_color, verticalalignment='bottom',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=spine_color, alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # Leave room for summary text

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved E/I synchrony analysis to {save_path}")

    # Return comprehensive sync info
    sync_info = {
        'bin_sizes_ms': bin_sizes_ms,
        'results_by_bin': all_results,
        'reference_bin_ms': ref_bin,
        'exc_mean_burst_ref': ref_data['exc_mean_burst'],
        'inh_mean_burst_ref': ref_data['inh_mean_burst'],
    }

    return fig, sync_info


def plot_network_activation_percentage(activity_record, n_neurons, dt=0.1, stim_times=None,
                                        stim_duration_ms=None, window_ms=5.0, save_path="network_activation.png",
                                        figsize=(14, 6), darkstyle=True):
    """
    Plot the percentage of network activated over time.

    Parameters:
    -----------
    activity_record : list of lists
        Each element contains indices of neurons that spiked at that timestep
    n_neurons : int
        Total number of neurons in the network
    dt : float
        Timestep in ms
    stim_times : list or None
        List of stimulation time points (in ms) - draws vertical lines or blocks
    stim_duration_ms : float or None
        Duration of each stimulation period in ms. If provided, draws shaded blocks
        instead of vertical lines.
    window_ms : float
        Sliding window size in ms for smoothing the activation curve
    save_path : str or None
        Path to save the figure. If None, figure is not saved.
    figsize : tuple
        Figure size
    darkstyle : bool
        If True, use dark background style

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    activation_data : dict
        Dictionary with time, raw percentage, and smoothed percentage arrays
    """
    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a2e'
        text_color = '#eee'
        spine_color = '#555'
        line_color = '#ff6b6b'
        smooth_color = '#4ecdc4'
        stim_color = '#32ff32'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = '#333'
        line_color = '#e74c3c'
        smooth_color = '#16a085'
        stim_color = 'green'

    # Calculate activation percentage for each timestep
    n_timesteps = len(activity_record)
    time_ms = np.arange(n_timesteps) * dt

    # Raw activation: number of neurons firing at each timestep / total neurons * 100
    raw_activation = np.array([len(active) / n_neurons * 100 for active in activity_record])

    # Smoothed activation using sliding window
    window_steps = max(1, int(window_ms / dt))
    smoothed_activation = np.convolve(raw_activation,
                                       np.ones(window_steps) / window_steps,
                                       mode='same')

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    # Plot raw activation (thin, semi-transparent)
    ax.plot(time_ms, raw_activation, color=line_color, alpha=0.3, linewidth=0.5,
            label='Instantaneous')

    # Plot smoothed activation (thicker, more visible)
    ax.plot(time_ms, smoothed_activation, color=smooth_color, linewidth=1.5,
            label=f'Smoothed ({window_ms:.0f}ms window)')

    # Mark stimulation times
    if stim_times:
        for stim_time in stim_times:
            if stim_duration_ms and stim_duration_ms > 0:
                # Draw shaded blocks showing stimulation duration
                ax.axvspan(stim_time, stim_time + stim_duration_ms,
                          color=stim_color, alpha=0.15, linewidth=0)
            else:
                ax.axvline(x=stim_time, color=stim_color, linewidth=0.5, alpha=0.15)

    # Labels and title
    ax.set_xlabel('Time (ms)', color=text_color, fontsize=12)
    ax.set_ylabel('Network Activation (%)', color=text_color, fontsize=12)
    ax.set_title('Percentage of Network Activated Over Time', color=text_color, fontsize=14)

    # Set y-axis limits with some padding
    max_activation = max(np.max(raw_activation), np.max(smoothed_activation))
    ax.set_ylim(0, min(100, max_activation * 1.1 + 1))
    ax.set_xlim(0, time_ms[-1])

    # Legend
    ax.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color,
              labelcolor=text_color)

    # Style
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.grid(True, alpha=0.2)

    # Add statistics text
    mean_activation = np.mean(raw_activation)
    max_activation_val = np.max(raw_activation)
    std_activation = np.std(raw_activation)

    stats_text = (f"Mean: {mean_activation:.2f}%\n"
                  f"Max: {max_activation_val:.2f}%\n"
                  f"Std: {std_activation:.2f}%")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, color=text_color, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=spine_color, alpha=0.8))

    plt.tight_layout()

    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved network activation plot to {save_path}")

    # Return data
    activation_data = {
        'time_ms': time_ms,
        'raw_activation_pct': raw_activation,
        'smoothed_activation_pct': smoothed_activation,
        'mean_pct': mean_activation,
        'max_pct': max_activation_val,
        'std_pct': std_activation
    }

    return fig, activation_data


def plot_inhibitory_effects_analysis(
    network,
    neuron_data=None,
    dt=0.1,
    figsize=(16, 14),
    dpi=150,
    save_path="inhibitory_effects_analysis.png",
    darkstyle=True
):
    """
    Analyze inhibitory effects on excitatory and inhibitory neuron populations.

    Creates a multi-panel figure showing:
    - Panel 1: Connection statistics (I->E vs I->I counts and weights)
    - Panel 2: Weight distribution histograms (I->E vs I->I)
    - Panels 3-4: Activity-integrated E/I input over time (if neuron_data provided)

    Parameters
    ----------
    network : SphericalNeuronalNetworkVectorized or SphericalNeuronalNetwork
        The network object with weights, is_inhibitory, and graph
    neuron_data : dict or None
        Dictionary of tracked neuron data with g_e_history and g_i_history.
        If None, only static panels (1-2) are generated.
    dt : float
        Time step in ms (used for time axis in panels 3-4)
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    analysis_info : dict
        Dictionary containing computed statistics
    """
    # Style setup
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'

    # Color scheme
    ie_color = '#e74c3c'    # Red for I->E (inhibition of excitatory)
    ii_color = '#3498db'    # Blue for I->I (inhibition of inhibitory)
    g_e_color = '#ff9f43'   # Orange for excitatory conductance
    g_i_color = '#0abde3'   # Cyan for inhibitory conductance

    # Get neuron type indices - support both vectorized and object-based networks
    if hasattr(network, 'is_inhibitory') and hasattr(network.is_inhibitory, '__len__'):
        # Vectorized or array-based
        exc_indices = set(np.where(~network.is_inhibitory)[0])
        inh_indices = set(np.where(network.is_inhibitory)[0])
    else:
        # Object-based
        exc_indices = set()
        inh_indices = set()
        for i, neuron in enumerate(network.neurons):
            if neuron.is_inhibitory:
                inh_indices.add(i)
            else:
                exc_indices.add(i)

    n_exc = len(exc_indices)
    n_inh = len(inh_indices)

    print(f"Analyzing inhibitory effects: {n_exc} E neurons, {n_inh} I neurons")

    # Collect inhibitory connection data
    ie_weights = []  # I->E weights
    ii_weights = []  # I->I weights

    # Count incoming inhibitory connections per neuron type
    e_incoming_inh_counts = []  # Number of I inputs per E neuron
    i_incoming_inh_counts = []  # Number of I inputs per I neuron

    for j in range(network.n_neurons):
        incoming_inh_count = 0
        for i in network.graph.predecessors(j):
            if i in inh_indices:
                weight = network.weights[i, j]
                incoming_inh_count += 1
                if j in exc_indices:
                    ie_weights.append(weight)
                else:
                    ii_weights.append(weight)

        if j in exc_indices:
            e_incoming_inh_counts.append(incoming_inh_count)
        else:
            i_incoming_inh_counts.append(incoming_inh_count)

    ie_weights = np.array(ie_weights)
    ii_weights = np.array(ii_weights)

    # Compute statistics
    ie_count = len(ie_weights)
    ii_count = len(ii_weights)
    ie_total = np.sum(np.abs(ie_weights)) if len(ie_weights) > 0 else 0
    ii_total = np.sum(np.abs(ii_weights)) if len(ii_weights) > 0 else 0
    ie_mean = np.mean(ie_weights) if len(ie_weights) > 0 else 0
    ii_mean = np.mean(ii_weights) if len(ii_weights) > 0 else 0
    ie_per_target = np.mean(e_incoming_inh_counts) if e_incoming_inh_counts else 0
    ii_per_target = np.mean(i_incoming_inh_counts) if i_incoming_inh_counts else 0

    # Determine layout based on whether we have dynamic data
    has_dynamic_data = (neuron_data is not None and len(neuron_data) > 0)

    if has_dynamic_data:
        # 4-panel layout
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.25)
        ax_stats = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_e_input = fig.add_subplot(gs[1, :])
        ax_i_input = fig.add_subplot(gs[2, :])
    else:
        # 2-panel layout (static only)
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.4), dpi=dpi, facecolor=bg_color)
        gs = GridSpec(1, 2, wspace=0.25)
        ax_stats = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])

    fig.suptitle('Inhibitory Effects Analysis', fontsize=16, color=text_color, y=0.98)

    # ========== PANEL 1: CONNECTION STATISTICS ==========
    metrics = ['Connections', 'Total |Weight|', 'Mean Weight', 'Per Target']
    ie_values = [ie_count, ie_total, abs(ie_mean), ie_per_target]
    ii_values = [ii_count, ii_total, abs(ii_mean), ii_per_target]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for better visualization (different scales)
    max_vals = [max(ie_values[i], ii_values[i], 1e-10) for i in range(len(metrics))]
    ie_norm = [ie_values[i] / max_vals[i] for i in range(len(metrics))]
    ii_norm = [ii_values[i] / max_vals[i] for i in range(len(metrics))]

    bars1 = ax_stats.bar(x - width/2, ie_norm, width, label=f'I→E (n={ie_count})',
                          color=ie_color, alpha=0.8)
    bars2 = ax_stats.bar(x + width/2, ii_norm, width, label=f'I→I (n={ii_count})',
                          color=ii_color, alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, ie_values):
        height = bar.get_height()
        ax_stats.annotate(f'{val:.1f}' if val < 1000 else f'{val:.0f}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, color=text_color)

    for bar, val in zip(bars2, ii_values):
        height = bar.get_height()
        ax_stats.annotate(f'{val:.1f}' if val < 1000 else f'{val:.0f}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, color=text_color)

    ax_stats.set_ylabel('Normalized Value', color=text_color)
    ax_stats.set_title('Inhibitory Connection Statistics', color=text_color, fontsize=12)
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(metrics, fontsize=9)
    ax_stats.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color,
                    labelcolor=text_color)
    ax_stats.set_ylim(0, 1.4)

    # ========== PANEL 2: WEIGHT DISTRIBUTIONS ==========
    if len(ie_weights) > 0:
        ax_hist.hist(ie_weights, bins=30, color=ie_color, alpha=0.6,
                     label=f'I→E (μ={ie_mean:.3f})', edgecolor='white', linewidth=0.5)
        ax_hist.axvline(ie_mean, color=ie_color, linestyle='--', linewidth=2)

    if len(ii_weights) > 0:
        ax_hist.hist(ii_weights, bins=30, color=ii_color, alpha=0.6,
                     label=f'I→I (μ={ii_mean:.3f})', edgecolor='white', linewidth=0.5)
        ax_hist.axvline(ii_mean, color=ii_color, linestyle='--', linewidth=2)

    ax_hist.axvline(0, color=spine_color, linestyle='-', linewidth=0.5, alpha=0.5)
    ax_hist.set_xlabel('Weight (negative = inhibitory)', color=text_color)
    ax_hist.set_ylabel('Count', color=text_color)
    ax_hist.set_title('Inhibitory Weight Distributions', color=text_color, fontsize=12)
    ax_hist.legend(loc='upper left', facecolor=bg_color, edgecolor=spine_color,
                   labelcolor=text_color)

    # ========== PANELS 3-4: DYNAMIC INPUT ANALYSIS ==========
    e_tracked_indices = []
    i_tracked_indices = []
    mean_e_g_e = None
    mean_e_g_i = None
    mean_i_g_e = None
    mean_i_g_i = None

    if has_dynamic_data:
        # Separate tracked neurons by type
        for idx in neuron_data.keys():
            if neuron_data[idx]['is_inhibitory']:
                i_tracked_indices.append(idx)
            else:
                e_tracked_indices.append(idx)

        print(f"Tracked neurons for dynamic analysis: {len(e_tracked_indices)} E, {len(i_tracked_indices)} I")

        # Panel 3: Excitatory neurons
        if len(e_tracked_indices) > 0:
            n_steps = len(neuron_data[e_tracked_indices[0]]['g_e_history'])
            time_ms = np.arange(n_steps) * dt

            e_g_e = np.array([neuron_data[idx]['g_e_history'] for idx in e_tracked_indices])
            e_g_i = np.array([neuron_data[idx]['g_i_history'] for idx in e_tracked_indices])

            mean_e_g_e = np.mean(e_g_e, axis=0)
            mean_e_g_i = np.mean(e_g_i, axis=0)
            std_e_g_e = np.std(e_g_e, axis=0)
            std_e_g_i = np.std(e_g_i, axis=0)

            ax_e_input.plot(time_ms, mean_e_g_e, color=g_e_color, linewidth=1.5,
                           label='Excitatory input (g_e)')
            ax_e_input.fill_between(time_ms, mean_e_g_e - std_e_g_e, mean_e_g_e + std_e_g_e,
                                   color=g_e_color, alpha=0.2)
            ax_e_input.plot(time_ms, mean_e_g_i, color=g_i_color, linewidth=1.5,
                           label='Inhibitory input (g_i)')
            ax_e_input.fill_between(time_ms, mean_e_g_i - std_e_g_i, mean_e_g_i + std_e_g_i,
                                   color=g_i_color, alpha=0.2)

            ax_e_input.set_xlabel('Time (ms)', color=text_color)
            ax_e_input.set_ylabel('Conductance', color=text_color)
            ax_e_input.set_title(f'E/I Input to Excitatory Neurons (n={len(e_tracked_indices)})',
                                color=text_color, fontsize=12)
            ax_e_input.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color,
                             labelcolor=text_color)

            # Add integrated values as text
            total_g_e = np.sum(mean_e_g_e) * dt
            total_g_i = np.sum(mean_e_g_i) * dt
            ax_e_input.text(0.02, 0.95, f'Integrated: g_e={total_g_e:.1f}, g_i={total_g_i:.1f}',
                           transform=ax_e_input.transAxes, fontsize=10, color=text_color,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8, edgecolor=spine_color))

        # Panel 4: Inhibitory neurons
        if len(i_tracked_indices) > 0:
            n_steps = len(neuron_data[i_tracked_indices[0]]['g_e_history'])
            time_ms = np.arange(n_steps) * dt

            i_g_e = np.array([neuron_data[idx]['g_e_history'] for idx in i_tracked_indices])
            i_g_i = np.array([neuron_data[idx]['g_i_history'] for idx in i_tracked_indices])

            mean_i_g_e = np.mean(i_g_e, axis=0)
            mean_i_g_i = np.mean(i_g_i, axis=0)
            std_i_g_e = np.std(i_g_e, axis=0)
            std_i_g_i = np.std(i_g_i, axis=0)

            ax_i_input.plot(time_ms, mean_i_g_e, color=g_e_color, linewidth=1.5,
                           label='Excitatory input (g_e)')
            ax_i_input.fill_between(time_ms, mean_i_g_e - std_i_g_e, mean_i_g_e + std_i_g_e,
                                   color=g_e_color, alpha=0.2)
            ax_i_input.plot(time_ms, mean_i_g_i, color=g_i_color, linewidth=1.5,
                           label='Inhibitory input (g_i)')
            ax_i_input.fill_between(time_ms, mean_i_g_i - std_i_g_i, mean_i_g_i + std_i_g_i,
                                   color=g_i_color, alpha=0.2)

            ax_i_input.set_xlabel('Time (ms)', color=text_color)
            ax_i_input.set_ylabel('Conductance', color=text_color)
            ax_i_input.set_title(f'E/I Input to Inhibitory Neurons (n={len(i_tracked_indices)})',
                                color=text_color, fontsize=12)
            ax_i_input.legend(loc='upper right', facecolor=bg_color, edgecolor=spine_color,
                             labelcolor=text_color)

            # Add integrated values as text
            total_g_e = np.sum(mean_i_g_e) * dt
            total_g_i = np.sum(mean_i_g_i) * dt
            ax_i_input.text(0.02, 0.95, f'Integrated: g_e={total_g_e:.1f}, g_i={total_g_i:.1f}',
                           transform=ax_i_input.transAxes, fontsize=10, color=text_color,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8, edgecolor=spine_color))

    # Style all axes
    all_axes = [ax_stats, ax_hist]
    if has_dynamic_data:
        all_axes.extend([ax_e_input, ax_i_input])

    for ax in all_axes:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved inhibitory effects analysis to {save_path}")

    # Build return info
    analysis_info = {
        'ie_connection_count': ie_count,
        'ii_connection_count': ii_count,
        'ie_total_weight': ie_total,
        'ii_total_weight': ii_total,
        'ie_mean_weight': ie_mean,
        'ii_mean_weight': ii_mean,
        'ie_per_target': ie_per_target,
        'ii_per_target': ii_per_target,
        'e_neurons_tracked': len(e_tracked_indices),
        'i_neurons_tracked': len(i_tracked_indices),
    }

    if has_dynamic_data and len(e_tracked_indices) > 0 and mean_e_g_e is not None:
        analysis_info['e_neuron_mean_total_g_e'] = float(np.sum(mean_e_g_e) * dt)
        analysis_info['e_neuron_mean_total_g_i'] = float(np.sum(mean_e_g_i) * dt)

    if has_dynamic_data and len(i_tracked_indices) > 0 and mean_i_g_e is not None:
        analysis_info['i_neuron_mean_total_g_e'] = float(np.sum(mean_i_g_e) * dt)
        analysis_info['i_neuron_mean_total_g_i'] = float(np.sum(mean_i_g_i) * dt)

    return fig, analysis_info


def plot_stimulation_figure(network, neuron_data, stimulation_record, activity_record,
                            dt=0.1, stim_interval=None, stim_strength=10.0,
                            figsize=(14, 10), dpi=150,
                            save_path="stimulation_figure.png", darkstyle=True):
    """
    Create a comprehensive stimulation analysis figure showing:
    1. Top panel: Net stimulatory drive to the network (# neurons × current strength)
    2. Middle panel: Network response - spikes from stimulated vs non-stimulated neurons
    3. Bottom panel: Cumulative effect - how stimulation propagates through network

    Parameters:
    -----------
    network : SphericalNeuronalNetworkVectorized
        The network object (used to identify stimulated neurons)
    neuron_data : dict
        Dictionary mapping neuron indices to their recorded data (not used in revamped version)
    stimulation_record : dict
        Dictionary with 'times' (list of stim times in ms) and 'neurons' (list of lists of stimulated neuron indices)
    activity_record : list
        List of lists containing active neuron indices at each timestep
    dt : float
        Timestep in ms
    stim_interval : float or None
        Stimulation interval in ms (for labeling)
    stim_strength : float
        Current injection strength per neuron (for calculating total drive)
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background. If False, use light background

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    stim_info : dict
        Summary statistics about stimulation effects
    """
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        stim_color = '#32ff32'      # Green for stimulation
        stim_spike_color = '#98fb98'  # Light green for stimulated neuron spikes
        network_color = '#ff6b6b'   # Red for non-stimulated spikes
        total_color = '#ffd93d'     # Yellow for total
        grid_alpha = 0.2
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        stim_color = '#2ecc71'
        stim_spike_color = '#27ae60'
        network_color = '#e74c3c'
        total_color = '#f39c12'
        grid_alpha = 0.3

    # Extract stimulation data
    stim_times_list = stimulation_record.get('times', [])
    stim_neurons_list = stimulation_record.get('neurons', [])

    if not stim_times_list:
        print("Warning: No stimulation events recorded. Cannot create stimulation figure.")
        return None, {}

    # Build set of all neurons that ever received stimulation
    all_stim_neurons = set()
    for neurons in stim_neurons_list:
        all_stim_neurons.update(neurons)

    print(f"Creating stimulation analysis figure...")
    print(f"  Total stimulation events: {len(stim_times_list)}")
    print(f"  Unique neurons stimulated: {len(all_stim_neurons)}")

    # Calculate total simulation time
    n_steps = len(activity_record)
    total_time = n_steps * dt
    time_axis = np.arange(n_steps) * dt

    # === Compute time series data ===

    # 1. Stimulation drive: number of neurons receiving stim at each timestep × strength
    stim_neuron_count = np.zeros(n_steps)
    for stim_time, stim_neurons in zip(stim_times_list, stim_neurons_list):
        timestep = int(stim_time / dt)
        if 0 <= timestep < n_steps:
            stim_neuron_count[timestep] = len(stim_neurons)

    stim_drive = stim_neuron_count * stim_strength  # Total current injected

    # 2. Spike counts: separate stimulated vs non-stimulated neurons
    stim_neuron_spikes = np.zeros(n_steps)      # Spikes from neurons that receive stimulation
    non_stim_neuron_spikes = np.zeros(n_steps)  # Spikes from neurons that don't receive stimulation

    for t_idx, active_neurons in enumerate(activity_record):
        for neuron_idx in active_neurons:
            if neuron_idx in all_stim_neurons:
                stim_neuron_spikes[t_idx] += 1
            else:
                non_stim_neuron_spikes[t_idx] += 1

    total_spikes = stim_neuron_spikes + non_stim_neuron_spikes

    # 3. Compute smoothed versions for visualization (rolling average)
    window_size = max(1, int(10 / dt))  # 10ms smoothing window
    kernel = np.ones(window_size) / window_size

    stim_drive_smooth = np.convolve(stim_drive, kernel, mode='same')
    stim_spikes_smooth = np.convolve(stim_neuron_spikes, kernel, mode='same')
    non_stim_spikes_smooth = np.convolve(non_stim_neuron_spikes, kernel, mode='same')
    total_spikes_smooth = np.convolve(total_spikes, kernel, mode='same')

    # === Create figure ===
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1.5, 1.5], hspace=0.25)

    # === Panel 1: Net Stimulatory Drive ===
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(bg_color)

    # Plot smoothed drive only (cleaner) with subtle fill
    ax1.fill_between(time_axis, stim_drive_smooth, alpha=0.15, color=stim_color)
    ax1.plot(time_axis, stim_drive_smooth, color=stim_color, linewidth=0.8,
             label=f'Stim drive (smoothed {window_size * dt:.0f}ms)')

    # Add stim interval markers - very subtle
    if stim_interval:
        interval_starts = np.arange(stim_interval, total_time, stim_interval)
        for i, t in enumerate(interval_starts):
            ax1.axvline(x=t, color=stim_color, linestyle='--', alpha=0.15, linewidth=0.5)

    ax1.set_xlim(0, total_time)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Net Stim Drive\n(# neurons × I)', color=text_color, fontsize=10)
    ax1.set_title('Stimulation Input to Network', color=text_color, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=text_color, labelsize=9)
    ax1.set_xticklabels([])
    for spine in ax1.spines.values():
        spine.set_color(spine_color)
    ax1.grid(True, alpha=grid_alpha, axis='both')
    ax1.legend(loc='upper right', fontsize=8, facecolor=bg_color,
               edgecolor=spine_color, labelcolor=text_color)

    # Add text annotation for peak drive
    peak_drive = np.max(stim_drive)
    peak_time = time_axis[np.argmax(stim_drive)]
    ax1.annotate(f'Peak: {peak_drive:.0f}', xy=(peak_time, peak_drive),
                 xytext=(peak_time + total_time * 0.05, peak_drive * 0.9),
                 color=text_color, fontsize=8,
                 arrowprops=dict(arrowstyle='->', color=text_color, lw=0.5))

    # === Panel 2: Network Response - Lines instead of stacked areas ===
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(bg_color)

    # Plot as lines with subtle fills - much cleaner than heavy stacked areas
    ax2.fill_between(time_axis, stim_spikes_smooth, alpha=0.2, color=stim_spike_color)
    ax2.plot(time_axis, stim_spikes_smooth, color=stim_spike_color, linewidth=0.8,
             label=f'Stimulated neurons (n={len(all_stim_neurons)})')

    ax2.fill_between(time_axis, non_stim_spikes_smooth, alpha=0.2, color=network_color)
    ax2.plot(time_axis, non_stim_spikes_smooth, color=network_color, linewidth=0.8,
             label=f'Non-stimulated neurons (n={network.n_neurons - len(all_stim_neurons)})')

    # Total line
    ax2.plot(time_axis, total_spikes_smooth, color=total_color, linewidth=1.0,
             linestyle='-', label='Total spikes', alpha=0.8)

    # Add stim interval markers - very subtle
    if stim_interval:
        for t in interval_starts:
            ax2.axvline(x=t, color=stim_color, linestyle='--', alpha=0.15, linewidth=0.5)

    ax2.set_xlim(0, total_time)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Spikes per\nTimestep', color=text_color, fontsize=10)
    ax2.set_title('Network Response: Stimulated vs Non-Stimulated Neurons',
                  color=text_color, fontsize=12, fontweight='bold')
    ax2.tick_params(colors=text_color, labelsize=9)
    ax2.set_xticklabels([])
    for spine in ax2.spines.values():
        spine.set_color(spine_color)
    ax2.grid(True, alpha=grid_alpha, axis='both')
    ax2.legend(loc='upper right', fontsize=8, facecolor=bg_color,
               edgecolor=spine_color, labelcolor=text_color)

    # === Panel 3: Amplification Ratio ===
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(bg_color)

    # Calculate ratio of total spikes to stimulated neuron count (amplification)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ratio: how many total spikes per neuron stimulated
        amplification = np.where(stim_neuron_count > 0,
                                 total_spikes / stim_neuron_count,
                                 0)

    # Smooth the amplification
    amplification_smooth = np.convolve(amplification, kernel, mode='same')

    # Also compute fraction of spikes from non-stimulated neurons
    with np.errstate(divide='ignore', invalid='ignore'):
        cascade_fraction = np.where(total_spikes > 0,
                                    non_stim_neuron_spikes / total_spikes,
                                    0)
    cascade_fraction_smooth = np.convolve(cascade_fraction, kernel, mode='same')

    # Plot on twin axes - subtle fills and thin lines
    ax3.fill_between(time_axis, amplification_smooth, alpha=0.15, color=total_color)
    ax3.plot(time_axis, amplification_smooth, color=total_color, linewidth=0.8,
             label='Spikes per stim neuron')

    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_axis, cascade_fraction_smooth * 100, color=network_color,
                  linewidth=0.8, linestyle='--', label='% from cascade')

    # Add stim interval markers - very subtle
    if stim_interval:
        for t in interval_starts:
            ax3.axvline(x=t, color=stim_color, linestyle='--', alpha=0.15, linewidth=0.5)

    ax3.set_xlim(0, total_time)
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel('Time (ms)', color=text_color, fontsize=10)
    ax3.set_ylabel('Spikes per\nStim Neuron', color=total_color, fontsize=10)
    ax3_twin.set_ylabel('Cascade %', color=network_color, fontsize=10)
    ax3.set_title('Stimulation Amplification & Cascade Effect',
                  color=text_color, fontsize=12, fontweight='bold')

    ax3.tick_params(colors=text_color, labelsize=9)
    ax3_twin.tick_params(colors=network_color, labelsize=9)
    ax3_twin.spines['right'].set_color(network_color)

    for spine in ['top', 'bottom', 'left']:
        ax3.spines[spine].set_color(spine_color)
    ax3.grid(True, alpha=grid_alpha, axis='both')

    # Combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8,
               facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)

    # === Add summary statistics as text ===
    total_stim_spikes = int(np.sum(stim_neuron_spikes))
    total_cascade_spikes = int(np.sum(non_stim_neuron_spikes))
    total_all_spikes = total_stim_spikes + total_cascade_spikes
    mean_amplification = total_all_spikes / len(all_stim_neurons) if len(all_stim_neurons) > 0 else 0
    cascade_pct = 100 * total_cascade_spikes / total_all_spikes if total_all_spikes > 0 else 0

    summary_text = (f"Summary:\n"
                    f"  Stim neurons: {len(all_stim_neurons):,}\n"
                    f"  Stim neuron spikes: {total_stim_spikes:,}\n"
                    f"  Cascade spikes: {total_cascade_spikes:,}\n"
                    f"  Cascade fraction: {cascade_pct:.1f}%\n"
                    f"  Amplification: {mean_amplification:.1f}x")

    fig.text(0.02, 0.02, summary_text, transform=fig.transFigure,
             fontsize=9, color=text_color, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor=spine_color, alpha=0.8))

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved stimulation analysis figure to {save_path}")

    # Build summary info
    stim_info = {
        'total_stim_events': len(stim_times_list),
        'unique_neurons_stimulated': len(all_stim_neurons),
        'stim_neuron_spikes': total_stim_spikes,
        'cascade_spikes': total_cascade_spikes,
        'total_spikes': total_all_spikes,
        'cascade_fraction': cascade_pct / 100,
        'mean_amplification': mean_amplification,
        'peak_stim_drive': float(peak_drive)
    }

    return fig, stim_info


def plot_membrane_potential_traces(network, neuron_data, activity_record, dt=0.1,
                                   n_excitatory=3, n_inhibitory=3,
                                   figsize=(16, 14), dpi=200,
                                   save_path="membrane_potential_traces.png",
                                   darkstyle=True,
                                   stim_times=None, stim_duration_ms=None):
    """
    Create high-resolution membrane potential traces for selected neurons with activity.

    Randomly selects excitatory and inhibitory neurons that have spiked during the
    simulation and plots their membrane potential over time with action potentials marked.
    Includes a final panel showing the sum of all tracked neurons' membrane potentials.

    Parameters:
    -----------
    network : SphericalNeuronalNetworkVectorized
        The network object (used to get neuron properties like threshold)
    neuron_data : dict
        Dictionary mapping neuron indices to their recorded data:
        {idx: {'v_history': [...], 'spike_times': [...], 'is_inhibitory': bool}}
    activity_record : list
        List of lists containing active neuron indices at each timestep
    dt : float
        Timestep in ms
    n_excitatory : int
        Number of excitatory neurons to plot (default 3)
    n_inhibitory : int
        Number of inhibitory neurons to plot (default 3)
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Figure resolution (default 200 for high quality)
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background. If False, use light background
    stim_times : list, optional
        List of stimulation start times in ms. If provided, translucent green shading
        will be added during stimulation periods.
    stim_duration_ms : float, optional
        Duration of each stimulation period in ms. Required if stim_times is provided.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    selected_neurons : dict
        Dictionary with 'excitatory' and 'inhibitory' lists of selected neuron indices
    """
    from matplotlib.gridspec import GridSpec

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        exc_color = '#ff6b6b'       # Red for excitatory
        inh_color = '#4dabf7'       # Blue for inhibitory
        threshold_color = '#ff9f43'  # Orange for threshold
        spike_color = '#ffd93d'      # Yellow for spike markers
        rest_color = '#888888'       # Gray for resting potential
        stim_color = '#32ff32'       # Green for stimulation periods
        sum_color = '#b19cd9'        # Light purple for sum trace
        grid_alpha = 0.15
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        exc_color = '#e74c3c'
        inh_color = '#3498db'
        threshold_color = '#e67e22'
        spike_color = '#f39c12'
        rest_color = '#7f8c8d'
        stim_color = '#2ecc71'       # Green for stimulation periods
        sum_color = '#9b59b6'        # Purple for sum trace
        grid_alpha = 0.3

    # Find neurons with activity from neuron_data
    exc_neurons_with_spikes = []
    inh_neurons_with_spikes = []

    for idx, data in neuron_data.items():
        spike_times = data.get('spike_times', [])
        v_history = data.get('v_history', [])
        is_inhibitory = data.get('is_inhibitory', False)

        # Only consider neurons with both voltage data and spikes
        if len(v_history) > 0 and len(spike_times) > 0:
            if is_inhibitory:
                inh_neurons_with_spikes.append((idx, len(spike_times)))
            else:
                exc_neurons_with_spikes.append((idx, len(spike_times)))

    if len(exc_neurons_with_spikes) == 0 and len(inh_neurons_with_spikes) == 0:
        print("Warning: No tracked neurons with spikes found. Cannot create membrane potential figure.")
        return None, {}

    # Sort by spike count and select neurons with good activity
    exc_neurons_with_spikes.sort(key=lambda x: x[1], reverse=True)
    inh_neurons_with_spikes.sort(key=lambda x: x[1], reverse=True)

    # Select neurons - prefer ones with moderate activity (not too much, not too little)
    # Take from the middle-to-high activity range for interesting traces
    n_exc_available = len(exc_neurons_with_spikes)
    n_inh_available = len(inh_neurons_with_spikes)

    selected_exc = []
    selected_inh = []

    if n_exc_available > 0:
        # Take from top half of active neurons, randomly sample
        top_half_exc = exc_neurons_with_spikes[:max(1, n_exc_available // 2 + 2)]
        np.random.shuffle(top_half_exc)
        selected_exc = [x[0] for x in top_half_exc[:min(n_excitatory, len(top_half_exc))]]

    if n_inh_available > 0:
        top_half_inh = inh_neurons_with_spikes[:max(1, n_inh_available // 2 + 2)]
        np.random.shuffle(top_half_inh)
        selected_inh = [x[0] for x in top_half_inh[:min(n_inhibitory, len(top_half_inh))]]

    total_neurons = len(selected_exc) + len(selected_inh)

    if total_neurons == 0:
        print("Warning: Could not select any neurons with activity.")
        return None, {}

    print(f"Creating membrane potential traces for {len(selected_exc)} E and {len(selected_inh)} I neurons...")

    # Calculate total simulation time
    total_time = len(activity_record) * dt

    # Create figure - extra row for the sum panel
    fig = plt.figure(figsize=figsize, facecolor=bg_color)

    # Create GridSpec - one row per neuron + 1 for sum panel at bottom
    gs = GridSpec(total_neurons + 1, 1, figure=fig, hspace=0.4)

    all_selected = selected_exc + selected_inh
    neuron_types = ['E'] * len(selected_exc) + ['I'] * len(selected_inh)
    trace_colors = [exc_color] * len(selected_exc) + [inh_color] * len(selected_inh)

    # Helper function to add stimulation shading to an axis
    def add_stim_shading(ax, stim_times, stim_duration_ms, y_min, y_max):
        """Add translucent green shading during stimulation periods."""
        if stim_times is not None and stim_duration_ms is not None:
            for stim_start in stim_times:
                stim_end = stim_start + stim_duration_ms
                ax.axvspan(stim_start, stim_end, alpha=0.12, color=stim_color, zorder=0)

    # Collect all v_histories for sum calculation
    all_v_histories = []
    common_length = None

    for panel_idx, (neuron_idx, neuron_type, trace_color) in enumerate(zip(all_selected, neuron_types, trace_colors)):
        ax = fig.add_subplot(gs[panel_idx])
        ax.set_facecolor(bg_color)

        # Get neuron data
        data = neuron_data[neuron_idx]
        v_history = np.array(data['v_history'])
        spike_times = data['spike_times']

        # Track for sum calculation
        all_v_histories.append(v_history)
        if common_length is None:
            common_length = len(v_history)
        else:
            common_length = min(common_length, len(v_history))

        # Create time axis
        time_axis = np.arange(len(v_history)) * dt

        # Get neuron-specific parameters (use actual values from network)
        if hasattr(network, 'v_threshold'):
            v_thresh = network.v_threshold[neuron_idx]
        else:
            v_thresh = -50.0

        if hasattr(network, 'v_rest'):
            v_rest = network.v_rest[neuron_idx]
        else:
            v_rest = -65.0

        if hasattr(network, 'v_reset'):
            v_reset = network.v_reset[neuron_idx]
        else:
            v_reset = -70.0

        # Set y-limits with some padding (compute before adding shading)
        v_min = min(np.min(v_history), v_reset) - 5
        v_max = v_thresh + 10

        # Add stimulation period shading (before plotting traces so it's behind)
        add_stim_shading(ax, stim_times, stim_duration_ms, v_min, v_max)

        # Plot membrane potential trace - thin line for clarity
        ax.plot(time_axis, v_history, color=trace_color, linewidth=0.5, alpha=0.85,
                label=f'V_m')

        # Add threshold line - subtle (use neuron's actual threshold)
        ax.axhline(y=v_thresh, color=threshold_color, linestyle='--', linewidth=0.8,
                   alpha=0.5, label=f'Threshold ({v_thresh:.1f} mV)')

        # Add resting potential line - very subtle
        ax.axhline(y=v_rest, color=rest_color, linestyle=':', linewidth=0.6,
                   alpha=0.3, label=f'V_rest ({v_rest:.1f} mV)')

        # Mark action potentials with small tick marks at top of plot only
        # No vertical lines through the trace - just small markers
        spike_times_in_range = [t for t in spike_times if t <= time_axis[-1]]
        if spike_times_in_range:
            # Plot small tick marks at the top
            ax.scatter(spike_times_in_range, [v_thresh + 8] * len(spike_times_in_range),
                      marker='|', s=15, color=spike_color, alpha=0.7, linewidths=0.5)

        # Styling
        ax.set_xlim(0, min(total_time, time_axis[-1]))
        ax.set_ylim(v_min, v_max)

        # Labels
        n_spikes = len(spike_times)
        firing_rate = n_spikes / (time_axis[-1] / 1000) if time_axis[-1] > 0 else 0

        ax.set_ylabel(f'Neuron {neuron_idx}\n({neuron_type})\nmV', color=text_color,
                      fontsize=10, fontweight='bold')

        # Add neuron info as text
        info_text = f'Spikes: {n_spikes} | Rate: {firing_rate:.1f} Hz'
        ax.text(0.99, 0.95, info_text, transform=ax.transAxes,
                fontsize=9, color=text_color, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
                          edgecolor=spine_color, alpha=0.8))

        # Tick styling
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, alpha=grid_alpha, axis='both')

        # Hide x-axis labels for all individual neuron panels (sum panel will have them)
        ax.set_xticklabels([])

        # Add legend only on first panel
        if panel_idx == 0:
            ax.legend(loc='upper left', fontsize=8, facecolor=bg_color,
                      edgecolor=spine_color, labelcolor=text_color, ncol=3)

    # === Create the sum panel at the bottom ===
    ax_sum = fig.add_subplot(gs[total_neurons])
    ax_sum.set_facecolor(bg_color)

    # Calculate sum of all tracked neuron potentials (truncate to common length)
    v_sum = np.zeros(common_length)
    for v_hist in all_v_histories:
        v_sum += v_hist[:common_length]

    time_axis_sum = np.arange(common_length) * dt

    # Add stimulation shading to sum panel
    sum_min = np.min(v_sum) - 20
    sum_max = np.max(v_sum) + 20
    add_stim_shading(ax_sum, stim_times, stim_duration_ms, sum_min, sum_max)

    # Plot the sum trace
    ax_sum.plot(time_axis_sum, v_sum, color=sum_color, linewidth=0.7, alpha=0.9,
                label=f'Σ V_m ({total_neurons} neurons)')

    # Styling for sum panel
    ax_sum.set_xlim(0, min(total_time, time_axis_sum[-1]))
    ax_sum.set_ylim(sum_min, sum_max)

    ax_sum.set_ylabel(f'Sum\n({total_neurons} neurons)\nmV', color=text_color,
                      fontsize=10, fontweight='bold')
    ax_sum.set_xlabel('Time (ms)', color=text_color, fontsize=11)

    # Add info text for sum panel
    mean_sum = np.mean(v_sum)
    std_sum = np.std(v_sum)
    info_text_sum = f'Mean: {mean_sum:.1f} mV | Std: {std_sum:.1f} mV'
    ax_sum.text(0.99, 0.95, info_text_sum, transform=ax_sum.transAxes,
                fontsize=9, color=text_color, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color,
                          edgecolor=spine_color, alpha=0.8))

    ax_sum.tick_params(colors=text_color, labelsize=9)
    for spine in ax_sum.spines.values():
        spine.set_color(spine_color)
    ax_sum.grid(True, alpha=grid_alpha, axis='both')

    ax_sum.legend(loc='upper left', fontsize=8, facecolor=bg_color,
                  edgecolor=spine_color, labelcolor=text_color)

    # Add title
    fig.suptitle('Membrane Potential Traces', color=text_color,
                 fontsize=14, fontweight='bold', y=0.98)

    # Add subtitle with neuron counts
    subtitle = f'Excitatory (red): {len(selected_exc)} neurons  |  Inhibitory (blue): {len(selected_inh)} neurons'
    fig.text(0.5, 0.95, subtitle, ha='center', va='top',
             fontsize=10, color=text_color, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved membrane potential traces to {save_path}")

    selected_neurons = {
        'excitatory': selected_exc,
        'inhibitory': selected_inh
    }

    return fig, selected_neurons


def plot_correlation_analysis(network, activity_record, dt=0.1,
                               time_window_ms=50, distance_bins=20,
                               figsize=(16, 12), dpi=150,
                               save_path="correlation_analysis.png", darkstyle=True):
    """
    Analyze and visualize activity correlations over time and as a function of distance.

    This function creates a 4-panel figure:
    1. Top-left: Pairwise correlation over time (sliding window)
    2. Top-right: Correlation vs inter-neuron distance
    3. Bottom-left: Correlation distance (characteristic length scale) over time
    4. Bottom-right: Correlation heatmap over distance and time

    Correlation is computed as the Pearson correlation coefficient between binary
    spike trains of neuron pairs within sliding time windows.

    Parameters:
    -----------
    network : SphericalNeuronalNetworkVectorized or similar
        The network object containing neuron positions and connectivity
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    time_window_ms : float
        Size of sliding window for correlation analysis in ms
    distance_bins : int
        Number of bins for distance-dependent correlation analysis
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    correlation_info : dict
        Dictionary containing correlation analysis results:
        - mean_correlation: Average pairwise correlation across all time
        - correlation_over_time: Array of mean correlations at each time window
        - correlation_vs_distance: Dict with 'distances' and 'correlations' arrays
        - correlation_length_over_time: Array of characteristic correlation lengths
        - time_bins: Time bin centers in ms
    """
    from matplotlib.gridspec import GridSpec
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit

    print("Computing activity correlation analysis...")

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        grid_alpha = 0.2
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        grid_alpha = 0.3

    exc_color = '#e74c3c'  # Red for excitatory
    inh_color = '#3498db'  # Blue for inhibitory
    total_color = '#9b59b6'  # Purple for total

    n_neurons = network.n_neurons
    n_timesteps = len(activity_record)
    duration_ms = n_timesteps * dt

    # Get neuron positions for distance calculations
    positions = np.array([network.neuron_3d_positions[i] for i in range(n_neurons)])

    # Identify E/I neurons
    if hasattr(network, 'is_inhibitory') and hasattr(network.is_inhibitory, '__iter__'):
        is_inhibitory = np.array(network.is_inhibitory)
    else:
        is_inhibitory = np.array([network.neurons[i].is_inhibitory for i in range(n_neurons)])

    exc_mask = ~is_inhibitory
    inh_mask = is_inhibitory

    # Convert activity_record to binary spike matrix
    print("  Building spike matrix...")
    spike_matrix = np.zeros((n_neurons, n_timesteps), dtype=np.float32)
    for t, active_neurons in enumerate(activity_record):
        for neuron_idx in active_neurons:
            if neuron_idx < n_neurons:
                spike_matrix[neuron_idx, t] = 1.0

    # Parameters for analysis
    window_steps = int(time_window_ms / dt)
    n_windows = max(1, n_timesteps // window_steps)

    # Subsample neurons for efficiency (correlation computation is O(n²))
    max_neurons_sample = min(200, n_neurons)
    if n_neurons > max_neurons_sample:
        sample_indices = np.random.choice(n_neurons, max_neurons_sample, replace=False)
        print(f"  Subsampling {max_neurons_sample} neurons for correlation analysis")
    else:
        sample_indices = np.arange(n_neurons)

    n_sample = len(sample_indices)

    # Precompute pairwise distances for sampled neurons
    print("  Computing pairwise distances...")
    distances = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            dist = np.linalg.norm(positions[sample_indices[i]] - positions[sample_indices[j]])
            distances[i, j] = dist
            distances[j, i] = dist

    # Get distance range for binning
    max_distance = np.max(distances)
    distance_bin_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_bin_centers = (distance_bin_edges[:-1] + distance_bin_edges[1:]) / 2

    # Arrays to store results
    correlation_over_time = np.zeros(n_windows)
    correlation_over_time_exc = np.zeros(n_windows)
    correlation_over_time_inh = np.zeros(n_windows)
    correlation_length_over_time = np.zeros(n_windows)
    time_bin_centers = np.zeros(n_windows)

    # Correlation vs distance for each time window (for heatmap)
    corr_vs_dist_time = np.zeros((n_windows, distance_bins))

    # Overall correlation vs distance (aggregated)
    corr_by_distance = [[] for _ in range(distance_bins)]

    # Exponential decay function for fitting correlation length
    def exp_decay(d, c0, lam):
        return c0 * np.exp(-d / lam)

    print(f"  Analyzing {n_windows} time windows...")

    for w in range(n_windows):
        start_step = w * window_steps
        end_step = min(start_step + window_steps, n_timesteps)
        time_bin_centers[w] = (start_step + end_step) / 2 * dt

        # Extract spike data for this window
        window_spikes = spike_matrix[sample_indices, start_step:end_step]

        # Skip windows with no activity
        if np.sum(window_spikes) < 2:
            correlation_over_time[w] = np.nan
            correlation_length_over_time[w] = np.nan
            continue

        # Compute pairwise correlations for this window
        window_corrs = []
        window_corrs_exc = []
        window_corrs_inh = []
        corrs_by_dist_this_window = [[] for _ in range(distance_bins)]

        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                spike_i = window_spikes[i]
                spike_j = window_spikes[j]

                # Skip if either neuron has no variance
                if np.std(spike_i) == 0 or np.std(spike_j) == 0:
                    continue

                # Compute correlation
                try:
                    corr, _ = pearsonr(spike_i, spike_j)
                    if np.isnan(corr):
                        continue
                except:
                    continue

                window_corrs.append(corr)

                # Track E/I specific correlations
                ni = sample_indices[i]
                nj = sample_indices[j]
                if exc_mask[ni] and exc_mask[nj]:
                    window_corrs_exc.append(corr)
                elif inh_mask[ni] and inh_mask[nj]:
                    window_corrs_inh.append(corr)

                # Bin by distance
                dist = distances[i, j]
                bin_idx = np.searchsorted(distance_bin_edges[1:], dist)
                bin_idx = min(bin_idx, distance_bins - 1)
                corrs_by_dist_this_window[bin_idx].append(corr)
                corr_by_distance[bin_idx].append(corr)

        # Store mean correlation for this window
        if window_corrs:
            correlation_over_time[w] = np.mean(window_corrs)
        else:
            correlation_over_time[w] = np.nan

        if window_corrs_exc:
            correlation_over_time_exc[w] = np.mean(window_corrs_exc)
        else:
            correlation_over_time_exc[w] = np.nan

        if window_corrs_inh:
            correlation_over_time_inh[w] = np.mean(window_corrs_inh)
        else:
            correlation_over_time_inh[w] = np.nan

        # Store correlation vs distance for this window
        for b in range(distance_bins):
            if corrs_by_dist_this_window[b]:
                corr_vs_dist_time[w, b] = np.mean(corrs_by_dist_this_window[b])
            else:
                corr_vs_dist_time[w, b] = np.nan

        # Fit exponential decay to get correlation length
        valid_bins = ~np.isnan(corr_vs_dist_time[w, :])
        if np.sum(valid_bins) >= 3:
            try:
                valid_dists = distance_bin_centers[valid_bins]
                valid_corrs = corr_vs_dist_time[w, valid_bins]
                # Only fit if correlations are positive and decay-like
                if valid_corrs[0] > 0 and np.any(valid_corrs > 0.01):
                    popt, _ = curve_fit(exp_decay, valid_dists, valid_corrs,
                                        p0=[valid_corrs[0], max_distance / 4],
                                        bounds=([0, 0.01], [1, max_distance * 2]),
                                        maxfev=1000)
                    correlation_length_over_time[w] = popt[1]
                else:
                    correlation_length_over_time[w] = np.nan
            except:
                correlation_length_over_time[w] = np.nan
        else:
            correlation_length_over_time[w] = np.nan

    # Compute overall correlation vs distance
    corr_vs_distance_mean = np.zeros(distance_bins)
    corr_vs_distance_std = np.zeros(distance_bins)
    for b in range(distance_bins):
        if corr_by_distance[b]:
            corr_vs_distance_mean[b] = np.mean(corr_by_distance[b])
            corr_vs_distance_std[b] = np.std(corr_by_distance[b])
        else:
            corr_vs_distance_mean[b] = np.nan
            corr_vs_distance_std[b] = np.nan

    # === Create figure ===
    print("  Creating figure...")
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.25)

    # === Panel 1: Correlation over time ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)

    # Plot all, E, and I correlations
    valid_mask = ~np.isnan(correlation_over_time)
    ax1.plot(time_bin_centers[valid_mask], correlation_over_time[valid_mask],
             color=total_color, linewidth=2, label='All pairs', alpha=0.9)

    valid_mask_exc = ~np.isnan(correlation_over_time_exc)
    if np.any(valid_mask_exc):
        ax1.plot(time_bin_centers[valid_mask_exc], correlation_over_time_exc[valid_mask_exc],
                 color=exc_color, linewidth=1.5, label='E-E pairs', alpha=0.8)

    valid_mask_inh = ~np.isnan(correlation_over_time_inh)
    if np.any(valid_mask_inh):
        ax1.plot(time_bin_centers[valid_mask_inh], correlation_over_time_inh[valid_mask_inh],
                 color=inh_color, linewidth=1.5, label='I-I pairs', alpha=0.8)

    ax1.set_xlabel('Time (ms)', color=text_color, fontsize=11)
    ax1.set_ylabel('Mean Pairwise Correlation', color=text_color, fontsize=11)
    ax1.set_title('Activity Correlation Over Time', color=text_color, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax1.axhline(y=0, color=spine_color, linewidth=0.5, linestyle='--', alpha=0.5)

    # Style
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color(spine_color)
    ax1.grid(True, alpha=grid_alpha)

    # Add mean correlation text
    mean_corr = np.nanmean(correlation_over_time)
    ax1.text(0.02, 0.98, f'Mean: {mean_corr:.3f}', transform=ax1.transAxes,
             fontsize=10, color=text_color, ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor=spine_color, alpha=0.8))

    # === Panel 2: Correlation vs distance ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)

    valid_dist_mask = ~np.isnan(corr_vs_distance_mean)
    ax2.errorbar(distance_bin_centers[valid_dist_mask], corr_vs_distance_mean[valid_dist_mask],
                 yerr=corr_vs_distance_std[valid_dist_mask], color=total_color,
                 linewidth=2, marker='o', markersize=5, capsize=3, label='Mean ± std')

    # Fit exponential decay
    if np.sum(valid_dist_mask) >= 3:
        try:
            valid_dists = distance_bin_centers[valid_dist_mask]
            valid_corrs = corr_vs_distance_mean[valid_dist_mask]
            if valid_corrs[0] > 0:
                popt, _ = curve_fit(exp_decay, valid_dists, valid_corrs,
                                    p0=[valid_corrs[0], max_distance / 4],
                                    bounds=([0, 0.01], [1, max_distance * 2]),
                                    maxfev=1000)
                fit_x = np.linspace(0, max_distance, 100)
                fit_y = exp_decay(fit_x, *popt)
                ax2.plot(fit_x, fit_y, color='#2ecc71', linewidth=2, linestyle='--',
                         label=f'Exp fit (λ={popt[1]:.2f})')
        except:
            pass

    ax2.set_xlabel('Inter-neuron Distance', color=text_color, fontsize=11)
    ax2.set_ylabel('Mean Correlation', color=text_color, fontsize=11)
    ax2.set_title('Correlation vs Distance', color=text_color, fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax2.axhline(y=0, color=spine_color, linewidth=0.5, linestyle='--', alpha=0.5)

    # Style
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color(spine_color)
    ax2.grid(True, alpha=grid_alpha)

    # === Panel 3: Correlation length over time ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg_color)

    valid_length_mask = ~np.isnan(correlation_length_over_time)
    if np.any(valid_length_mask):
        ax3.plot(time_bin_centers[valid_length_mask], correlation_length_over_time[valid_length_mask],
                 color='#f39c12', linewidth=2, marker='o', markersize=4)

        mean_length = np.nanmean(correlation_length_over_time)
        ax3.axhline(y=mean_length, color='#e67e22', linewidth=1.5, linestyle='--',
                    label=f'Mean: {mean_length:.2f}')

    ax3.set_xlabel('Time (ms)', color=text_color, fontsize=11)
    ax3.set_ylabel('Correlation Length (λ)', color=text_color, fontsize=11)
    ax3.set_title('Correlation Length Over Time', color=text_color, fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)

    # Style
    ax3.tick_params(colors=text_color)
    for spine in ax3.spines.values():
        spine.set_color(spine_color)
    ax3.grid(True, alpha=grid_alpha)

    # Add explanation text
    ax3.text(0.02, 0.02, 'λ = characteristic distance\nover which correlation decays',
             transform=ax3.transAxes, fontsize=8, color=text_color, ha='left', va='bottom',
             style='italic', alpha=0.7)

    # === Panel 4: Correlation heatmap (distance x time) ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg_color)

    # Create heatmap
    # Replace NaN with 0 for visualization
    heatmap_data = np.nan_to_num(corr_vs_dist_time.T, nan=0)

    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(corr_vs_dist_time)), abs(np.nanmax(corr_vs_dist_time)))
    vmax = max(vmax, 0.1)  # Ensure some range

    im = ax4.imshow(heatmap_data, aspect='auto', origin='lower',
                    extent=[0, duration_ms, 0, max_distance],
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax4, pad=0.02)
    cbar.set_label('Correlation', color=text_color, fontsize=10)
    cbar.ax.tick_params(colors=text_color)

    ax4.set_xlabel('Time (ms)', color=text_color, fontsize=11)
    ax4.set_ylabel('Distance', color=text_color, fontsize=11)
    ax4.set_title('Correlation by Distance Over Time', color=text_color, fontsize=12, fontweight='bold')

    # Style
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_color(spine_color)

    # Main title
    fig.suptitle('Spatial Correlation Analysis', color=text_color,
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved correlation analysis to {save_path}")

    # Compile results
    correlation_info = {
        'mean_correlation': float(np.nanmean(correlation_over_time)),
        'mean_correlation_exc': float(np.nanmean(correlation_over_time_exc)),
        'mean_correlation_inh': float(np.nanmean(correlation_over_time_inh)),
        'correlation_over_time': correlation_over_time.tolist(),
        'correlation_over_time_exc': correlation_over_time_exc.tolist(),
        'correlation_over_time_inh': correlation_over_time_inh.tolist(),
        'time_bins_ms': time_bin_centers.tolist(),
        'correlation_vs_distance_mean': corr_vs_distance_mean.tolist(),
        'correlation_vs_distance_std': corr_vs_distance_std.tolist(),
        'distance_bin_centers': distance_bin_centers.tolist(),
        'correlation_length_over_time': correlation_length_over_time.tolist(),
        'mean_correlation_length': float(np.nanmean(correlation_length_over_time)),
        'time_window_ms': time_window_ms,
        'n_neurons_sampled': n_sample
    }

    print(f"  Mean correlation: {correlation_info['mean_correlation']:.4f}")
    print(f"  Mean correlation length: {correlation_info['mean_correlation_length']:.2f}")

    return fig, correlation_info


def plot_stim_vs_unstim_correlation(network, activity_record, stim_times, stim_duration_ms,
                                     dt=0.1, buffer_ms=5.0, distance_bins=15,
                                     max_neurons_sample=200, figsize=(16, 10), dpi=150,
                                     save_path="correlation_comparison.png", darkstyle=True):
    """
    Compare spatial correlation structure between stimulated and unstimulated periods.

    This provides an alternative to avalanche-based criticality analysis that works
    better during driven activity. At criticality, correlation length diverges
    (long-range correlations). Sub-critical systems have short correlation lengths.

    Creates a 4-panel figure:
    1. Top-left: Correlation vs distance for unstimulated periods
    2. Top-right: Correlation vs distance for stimulated periods
    3. Bottom-left: Correlation length over time (both periods overlaid)
    4. Bottom-right: Direct comparison scatter plot

    Parameters:
    -----------
    network : SphericalNeuronalNetworkVectorized or similar
        The network object containing neuron positions
    activity_record : list
        List of lists containing active neuron indices at each time step
    stim_times : list
        List of stimulation onset times in ms
    stim_duration_ms : float
        Duration of each stimulation period in ms
    dt : float
        Time step size in ms
    buffer_ms : float
        Buffer time before and after stimulation window
    distance_bins : int
        Number of bins for distance-dependent correlation
    max_neurons_sample : int
        Maximum number of neurons to sample for correlation analysis.
        Higher values give better statistics but O(n²) computational cost.
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        If True, use dark background style

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    comparison_info : dict
        Dictionary containing comparison results
    """
    from matplotlib.gridspec import GridSpec
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit

    print("Computing stimulated vs unstimulated correlation comparison...")

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        spine_color = 'white'
        grid_alpha = 0.2
    else:
        bg_color = 'white'
        text_color = 'black'
        spine_color = 'black'
        grid_alpha = 0.3

    unstim_color = '#3498db'  # Blue for unstimulated
    stim_color = '#e74c3c'    # Red for stimulated

    n_neurons = network.n_neurons
    n_timesteps = len(activity_record)
    duration_ms = n_timesteps * dt

    # Validate inputs
    if not stim_times or stim_duration_ms is None or stim_duration_ms <= 0:
        print("  ERROR: No valid stimulation times or duration provided. Cannot compare stim vs unstim.")
        return None, {}

    # Build mask for stimulated periods (including buffer)
    stim_mask = np.zeros(n_timesteps, dtype=bool)
    for stim_time in stim_times:
        start_ms = stim_time - buffer_ms
        end_ms = stim_time + stim_duration_ms + buffer_ms
        start_step = max(0, int(start_ms / dt))
        end_step = min(n_timesteps, int(end_ms / dt))
        stim_mask[start_step:end_step] = True

    unstim_mask = ~stim_mask

    print(f"  Total simulation: {duration_ms:.1f} ms ({n_timesteps} steps)")
    print(f"  Stimulated steps: {np.sum(stim_mask)} ({100*np.sum(stim_mask)/n_timesteps:.1f}%)")
    print(f"  Unstimulated steps: {np.sum(unstim_mask)} ({100*np.sum(unstim_mask)/n_timesteps:.1f}%)")

    # Get neuron positions for distance calculations
    positions = np.array([network.neuron_3d_positions[i] for i in range(n_neurons)])

    # Convert activity_record to binary spike matrix
    print("  Building spike matrix...")
    spike_matrix = np.zeros((n_neurons, n_timesteps), dtype=np.float32)
    for t, active_neurons in enumerate(activity_record):
        for neuron_idx in active_neurons:
            if neuron_idx < n_neurons:
                spike_matrix[neuron_idx, t] = 1.0

    # Find neurons active in BOTH stim and unstim periods (for fair comparison)
    # First compute spikes per neuron in each period
    stim_spikes = spike_matrix[:, stim_mask].sum(axis=1)
    unstim_spikes = spike_matrix[:, unstim_mask].sum(axis=1)

    # Neurons must have at least 1 spike in BOTH periods
    active_in_both = (stim_spikes > 0) & (unstim_spikes > 0)
    active_neuron_indices = np.where(active_in_both)[0]
    n_active = len(active_neuron_indices)

    # Also count neurons active in only one period for reporting
    n_stim_only = np.sum((stim_spikes > 0) & (unstim_spikes == 0))
    n_unstim_only = np.sum((stim_spikes == 0) & (unstim_spikes > 0))
    n_either = np.sum((stim_spikes > 0) | (unstim_spikes > 0))

    print(f"  {n_either} neurons fired at least once (out of {n_neurons} total)")
    print(f"  {n_active} neurons active in BOTH periods (used for analysis)")
    print(f"  {n_stim_only} neurons active only during stim, {n_unstim_only} only during unstim")

    if n_active < 10:
        print("  ERROR: Too few neurons active in both periods for correlation analysis")
        return None, {}

    # Subsample from neurons active in both periods
    actual_sample_size = min(max_neurons_sample, n_active)
    if n_active > actual_sample_size:
        sample_indices = np.random.choice(active_neuron_indices, actual_sample_size, replace=False)
        print(f"  Subsampling {actual_sample_size} neurons for correlation analysis")
    else:
        sample_indices = active_neuron_indices
        print(f"  Using all {n_active} neurons active in both periods")

    n_sample = len(sample_indices)

    # Precompute pairwise distances
    print("  Computing pairwise distances...")
    distances = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            dist = np.linalg.norm(positions[sample_indices[i]] - positions[sample_indices[j]])
            distances[i, j] = dist
            distances[j, i] = dist

    max_distance = np.max(distances)
    distance_bin_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_bin_centers = (distance_bin_edges[:-1] + distance_bin_edges[1:]) / 2

    # Exponential decay function
    def exp_decay(d, c0, lam):
        return c0 * np.exp(-d / lam)

    def compute_correlation_by_distance(mask, label):
        """Compute correlation vs distance for timesteps where mask is True."""
        mask_indices = np.where(mask)[0]
        if len(mask_indices) < 100:
            print(f"    {label}: insufficient data ({len(mask_indices)} steps)")
            return None, None, None, None

        # Get spike data for this period
        period_spikes = spike_matrix[np.ix_(sample_indices, mask_indices)]

        print(f"    {label}: computing correlations for {len(mask_indices)} steps...")

        # Vectorized correlation computation using numpy
        # Normalize each neuron's spike train (subtract mean, divide by std)
        means = period_spikes.mean(axis=1, keepdims=True)
        stds = period_spikes.std(axis=1, keepdims=True)

        # Find neurons with non-zero variance
        valid_neurons = (stds.flatten() > 0)
        n_valid = np.sum(valid_neurons)

        if n_valid < 2:
            print(f"    {label}: insufficient variance ({n_valid} neurons with activity)")
            return None, None, None, None

        # Normalize only valid neurons
        valid_indices = np.where(valid_neurons)[0]
        normalized = np.zeros_like(period_spikes)
        normalized[valid_neurons] = (period_spikes[valid_neurons] - means[valid_neurons]) / stds[valid_neurons]

        # Compute correlation matrix using matrix multiplication (much faster than pairwise pearsonr)
        n_timepoints = period_spikes.shape[1]
        corr_matrix = np.dot(normalized, normalized.T) / n_timepoints

        # Bin correlations by distance
        corr_by_distance = [[] for _ in range(distance_bins)]

        for i in range(n_sample):
            if not valid_neurons[i]:
                continue
            for j in range(i + 1, n_sample):
                if not valid_neurons[j]:
                    continue

                corr = corr_matrix[i, j]
                if np.isnan(corr):
                    continue

                dist = distances[i, j]
                bin_idx = np.searchsorted(distance_bin_edges[1:], dist)
                bin_idx = min(bin_idx, distance_bins - 1)
                corr_by_distance[bin_idx].append(corr)

        # Compute mean and std for each distance bin
        corr_mean = np.zeros(distance_bins)
        corr_std = np.zeros(distance_bins)
        for b in range(distance_bins):
            if corr_by_distance[b]:
                corr_mean[b] = np.mean(corr_by_distance[b])
                corr_std[b] = np.std(corr_by_distance[b])
            else:
                corr_mean[b] = np.nan
                corr_std[b] = np.nan

        # Fit exponential decay to get correlation length
        corr_length = None
        valid_mask = ~np.isnan(corr_mean)
        if np.sum(valid_mask) >= 3:
            try:
                valid_dists = distance_bin_centers[valid_mask]
                valid_corrs = corr_mean[valid_mask]
                if valid_corrs[0] > 0.001:
                    popt, _ = curve_fit(exp_decay, valid_dists, valid_corrs,
                                        p0=[valid_corrs[0], max_distance / 4],
                                        bounds=([0, 0.01], [1, max_distance * 2]),
                                        maxfev=1000)
                    corr_length = popt[1]
            except:
                pass

        return corr_mean, corr_std, corr_length, corr_by_distance

    # Compute for both periods
    unstim_mean, unstim_std, unstim_length, unstim_raw = compute_correlation_by_distance(unstim_mask, "Unstimulated")
    stim_mean, stim_std, stim_length, stim_raw = compute_correlation_by_distance(stim_mask, "Stimulated")

    # Now compute correlation length over time for both periods
    print("  Computing correlation length over time...")
    time_window_ms = 50
    window_steps = int(time_window_ms / dt)
    n_windows = max(1, n_timesteps // window_steps)

    unstim_length_time = []
    stim_length_time = []
    time_centers = []

    for w in range(n_windows):
        start_step = w * window_steps
        end_step = min(start_step + window_steps, n_timesteps)
        time_centers.append((start_step + end_step) / 2 * dt)

        # Determine if this window is predominantly stim or unstim
        window_stim_frac = np.mean(stim_mask[start_step:end_step])

        # Get spikes for this window
        window_spikes = spike_matrix[sample_indices, start_step:end_step]

        # Vectorized correlation computation for this window
        means_w = window_spikes.mean(axis=1, keepdims=True)
        stds_w = window_spikes.std(axis=1, keepdims=True)
        valid_neurons_w = (stds_w.flatten() > 0)

        if np.sum(valid_neurons_w) < 2:
            # Not enough neurons with variance
            if window_stim_frac > 0.5:
                stim_length_time.append((time_centers[-1], np.nan))
            else:
                unstim_length_time.append((time_centers[-1], np.nan))
            continue

        normalized_w = np.zeros_like(window_spikes)
        normalized_w[valid_neurons_w] = (window_spikes[valid_neurons_w] - means_w[valid_neurons_w]) / stds_w[valid_neurons_w]

        n_tp = window_spikes.shape[1]
        corr_matrix_w = np.dot(normalized_w, normalized_w.T) / n_tp

        # Bin correlations by distance
        corrs_by_dist = [[] for _ in range(distance_bins)]
        for i in range(n_sample):
            if not valid_neurons_w[i]:
                continue
            for j in range(i + 1, n_sample):
                if not valid_neurons_w[j]:
                    continue
                corr = corr_matrix_w[i, j]
                if not np.isnan(corr):
                    dist = distances[i, j]
                    bin_idx = min(np.searchsorted(distance_bin_edges[1:], dist), distance_bins - 1)
                    corrs_by_dist[bin_idx].append(corr)

        # Fit to get correlation length
        corr_mean_window = np.array([np.mean(c) if c else np.nan for c in corrs_by_dist])
        valid = ~np.isnan(corr_mean_window)

        length = np.nan
        if np.sum(valid) >= 3:
            try:
                valid_dists = distance_bin_centers[valid]
                valid_corrs = corr_mean_window[valid]
                if valid_corrs[0] > 0.001:
                    popt, _ = curve_fit(exp_decay, valid_dists, valid_corrs,
                                        p0=[valid_corrs[0], max_distance / 4],
                                        bounds=([0, 0.01], [1, max_distance * 2]),
                                        maxfev=500)
                    length = popt[1]
            except:
                pass

        # Assign to appropriate list based on stim fraction
        if window_stim_frac > 0.5:
            stim_length_time.append((time_centers[-1], length))
        else:
            unstim_length_time.append((time_centers[-1], length))

    # === Compute overall mean correlations for summary ===
    # Flatten all correlations for each regime
    unstim_all_corrs = []
    stim_all_corrs = []
    if unstim_raw:
        for bin_corrs in unstim_raw:
            unstim_all_corrs.extend(bin_corrs)
    if stim_raw:
        for bin_corrs in stim_raw:
            stim_all_corrs.extend(bin_corrs)

    unstim_overall_mean = np.mean(unstim_all_corrs) if unstim_all_corrs else 0
    stim_overall_mean = np.mean(stim_all_corrs) if stim_all_corrs else 0

    # === Create figure ===
    print("  Creating figure...")
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # === Panel 1: Bar chart comparing correlation length λ ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)

    bar_positions = [0, 1]
    bar_values = [unstim_length if unstim_length else 0, stim_length if stim_length else 0]
    bar_colors = [unstim_color, stim_color]
    bar_labels = ['Unstimulated', 'Stimulated']

    bars = ax1.bar(bar_positions, bar_values, color=bar_colors, width=0.6, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, bar_values):
        height = bar.get_height()
        label_text = f'λ={val:.2f}' if val > 0 else 'N/A'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 label_text, ha='center', va='bottom', color=text_color, fontsize=12, fontweight='bold')

    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(bar_labels, color=text_color, fontsize=11)
    ax1.set_ylabel('Correlation Length (λ)', color=text_color, fontsize=11)
    ax1.set_title('Correlation Length Comparison', color=text_color, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color(spine_color)
    ax1.grid(True, alpha=grid_alpha, axis='y')

    # Add interpretation text
    if unstim_length and stim_length:
        ratio = stim_length / unstim_length
        if ratio < 0.5:
            interp = "Stimulation disrupts long-range correlations"
        elif ratio > 2:
            interp = "Stimulation enhances long-range correlations"
        else:
            interp = "Similar correlation structure"
        ax1.text(0.5, 0.95, interp, transform=ax1.transAxes,
                 ha='center', va='top', color='#f39c12', fontsize=10, fontweight='bold')

    # === Panel 2: Correlation vs distance (both overlaid) ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)

    if unstim_mean is not None:
        valid = ~np.isnan(unstim_mean)
        unstim_label = f'Unstimulated (λ={unstim_length:.2f})' if unstim_length else 'Unstimulated'
        ax2.errorbar(distance_bin_centers[valid], unstim_mean[valid],
                     yerr=unstim_std[valid], color=unstim_color,
                     linewidth=2, marker='o', markersize=6, capsize=3,
                     label=unstim_label)
        # Plot exponential fit
        if unstim_length:
            fit_x = np.linspace(0, max_distance, 100)
            fit_y = unstim_mean[valid][0] * np.exp(-fit_x / unstim_length)
            ax2.plot(fit_x, fit_y, '--', color=unstim_color, alpha=0.5, linewidth=1.5)

    if stim_mean is not None:
        valid = ~np.isnan(stim_mean)
        stim_label = f'Stimulated (λ={stim_length:.2f})' if stim_length else 'Stimulated'
        ax2.errorbar(distance_bin_centers[valid], stim_mean[valid],
                     yerr=stim_std[valid], color=stim_color,
                     linewidth=2, marker='s', markersize=6, capsize=3,
                     label=stim_label)
        # Plot exponential fit
        if stim_length:
            fit_x = np.linspace(0, max_distance, 100)
            fit_y = stim_mean[valid][0] * np.exp(-fit_x / stim_length)
            ax2.plot(fit_x, fit_y, '--', color=stim_color, alpha=0.5, linewidth=1.5)

    ax2.set_xlabel('Inter-neuron Distance', color=text_color, fontsize=11)
    ax2.set_ylabel('Correlation', color=text_color, fontsize=11)
    ax2.set_title('Correlation vs Distance', color=text_color, fontsize=12, fontweight='bold')
    ax2.axhline(y=0, color=spine_color, linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color(spine_color)
    ax2.grid(True, alpha=grid_alpha)

    # === Panel 3: Correlation length λ over time ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg_color)

    # Use the already-computed length over time data
    if unstim_length_time:
        times, lengths = zip(*unstim_length_time)
        valid = [i for i, l in enumerate(lengths) if not np.isnan(l)]
        if valid:
            valid_times = [times[i] for i in valid]
            valid_lengths = [lengths[i] for i in valid]
            ax3.scatter(valid_times, valid_lengths, color=unstim_color, s=30, alpha=0.7, label='Unstimulated')
            mean_len = np.mean(valid_lengths)
            ax3.axhline(y=mean_len, color=unstim_color, linewidth=2, linestyle='--', alpha=0.7)

    if stim_length_time:
        times, lengths = zip(*stim_length_time)
        valid = [i for i, l in enumerate(lengths) if not np.isnan(l)]
        if valid:
            valid_times = [times[i] for i in valid]
            valid_lengths = [lengths[i] for i in valid]
            ax3.scatter(valid_times, valid_lengths, color=stim_color, s=30, alpha=0.7, label='Stimulated')
            mean_len = np.mean(valid_lengths)
            ax3.axhline(y=mean_len, color=stim_color, linewidth=2, linestyle='--', alpha=0.7)

    ax3.set_xlabel('Time (ms)', color=text_color, fontsize=11)
    ax3.set_ylabel('Correlation Length (λ)', color=text_color, fontsize=11)
    ax3.set_title('Correlation Length Over Time', color=text_color, fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax3.tick_params(colors=text_color)
    for spine in ax3.spines.values():
        spine.set_color(spine_color)
    ax3.grid(True, alpha=grid_alpha)

    # === Panel 4: Histogram of pairwise correlations ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg_color)

    # Create histograms
    bins = np.linspace(-0.2, 0.4, 31)

    if unstim_all_corrs:
        ax4.hist(unstim_all_corrs, bins=bins, color=unstim_color, alpha=0.6,
                 label=f'Unstimulated (n={len(unstim_all_corrs)})', edgecolor='white', linewidth=0.5)

    if stim_all_corrs:
        ax4.hist(stim_all_corrs, bins=bins, color=stim_color, alpha=0.6,
                 label=f'Stimulated (n={len(stim_all_corrs)})', edgecolor='white', linewidth=0.5)

    # Add vertical lines for means
    if unstim_all_corrs:
        ax4.axvline(x=unstim_overall_mean, color=unstim_color, linewidth=2, linestyle='--')
    if stim_all_corrs:
        ax4.axvline(x=stim_overall_mean, color=stim_color, linewidth=2, linestyle='--')

    ax4.axvline(x=0, color=spine_color, linewidth=1, linestyle='-', alpha=0.5)

    ax4.set_xlabel('Pairwise Correlation', color=text_color, fontsize=11)
    ax4.set_ylabel('Count', color=text_color, fontsize=11)
    ax4.set_title('Distribution of Pairwise Correlations', color=text_color, fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_color(spine_color)
    ax4.grid(True, alpha=grid_alpha)

    # Main title
    fig.suptitle('Correlation Analysis: Unstimulated vs Stimulated Periods',
                 color=text_color, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved correlation comparison to {save_path}")

    # Compile results
    comparison_info = {
        'unstimulated': {
            'correlation_length': float(unstim_length) if unstim_length else None,
            'correlation_vs_distance_mean': unstim_mean.tolist() if unstim_mean is not None else None,
            'correlation_vs_distance_std': unstim_std.tolist() if unstim_std is not None else None,
        },
        'stimulated': {
            'correlation_length': float(stim_length) if stim_length else None,
            'correlation_vs_distance_mean': stim_mean.tolist() if stim_mean is not None else None,
            'correlation_vs_distance_std': stim_std.tolist() if stim_std is not None else None,
        },
        'distance_bin_centers': distance_bin_centers.tolist(),
        'length_ratio': float(stim_length / unstim_length) if (stim_length and unstim_length) else None,
    }

    print(f"  Unstimulated correlation length: {unstim_length:.2f}" if unstim_length else "  Unstimulated correlation length: N/A")
    print(f"  Stimulated correlation length: {stim_length:.2f}" if stim_length else "  Stimulated correlation length: N/A")
    if comparison_info['length_ratio']:
        print(f"  Ratio (stim/unstim): {comparison_info['length_ratio']:.2f}")

    return fig, comparison_info
