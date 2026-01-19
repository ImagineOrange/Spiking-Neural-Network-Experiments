import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec # Needed for combined plot
from matplotlib.patches import Circle
from tqdm import tqdm

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
            ax.axvline(x=stim_time, color='#1dd1a1', linewidth=1.5, alpha=0.8, 
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


def plot_psth_and_raster(activity_record, stim_times=None, bin_size=10, dt=0.1,
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
    
    # Mark stimulation times if provided (green vertical lines)
    if stim_times:
        for stim_time in stim_times:
            # Draw vertical lines across both plots
            ax_psth.axvline(x=stim_time, color='#32ff32', linewidth=1.0, alpha=0.4)
            ax_raster.axvline(x=stim_time, color='#32ff32', linewidth=1.0, alpha=0.4)
    
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
            ax_overall.axvline(x=stim_time, color=stim_line_color, linestyle='--', linewidth=1.0, alpha=0.6)

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
                ax_raster.axvline(x=stim_time, color=stim_line_color, linestyle='--', linewidth=0.3, alpha=0.3)

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
                             stim_times=None):
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
        List of stimulation time points (in ms) - draws green vertical lines

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
    exc_neurons = set()
    inh_neurons = set()
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

    # Draw stimulation time markers as green vertical lines
    if stim_times:
        for stim_time in stim_times:
            for ax in [ax_exc_psth, ax_exc_raster, ax_inh_psth, ax_inh_raster]:
                ax.axvline(x=stim_time, color='#32ff32', linewidth=1.0, alpha=0.4)

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
