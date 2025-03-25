import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Set dark style for plots
plt.style.use('dark_background')

def visualize_activity_grid(network, activity_record, stim_times=None, stim_neurons=None, dt=0.1, 
                           save_path="neural_activity_grid.gif", max_frames=5000):
    """
    Create a grid visualization showing neural activity spreading through the network.
    Each cell in the grid represents a single neuron:
    - Stimulated neurons are shown in green when they're active at stimulus times
    - Regular excitatory neurons are shown in red when active
    - Inhibitory neurons are shown in blue when active
    """
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    # Initialize activity array for visualization
    activity_grid = np.zeros((side_length, side_length))
    
    # Create an RGB array for visualization (row, col, RGB)
    # Using green for stimulated, red for excitatory, and blue for inhibitory
    activity_colors = np.zeros((side_length, side_length, 3))
    
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
    
    # Function to update colors based on activity and current frame
    def update_colors(activity_grid, inhibitory_mask, stimulated_mask, frame_idx):
        colors = np.zeros((side_length, side_length, 3))
        
        # Check if current frame is a stimulation frame
        is_stim_time = frame_idx in expanded_stim_frames
        
        for r in range(side_length):
            for c in range(side_length):
                value = activity_grid[r, c]
                if value > 0.01:  # If there's activity
                    if inhibitory_mask[r, c]:
                        # Blue for inhibitory neurons (RGB: 0,0,1)
                        colors[r, c, 2] = value  # Blue channel
                    elif stimulated_mask[r, c] and is_stim_time:
                        # Green for stimulated neurons during stimulation (RGB: 0,1,0)
                        colors[r, c, 1] = value  # Green channel
                    else:
                        # Red for excitatory neurons (RGB: 1,0,0)
                        colors[r, c, 0] = value  # Red channel
        return colors
    
    # Initialize activity grid
    activity_colors = update_colors(activity_grid, inhibitory_mask, stimulated_mask, 0)
    
    # Set up the imshow with initial colors
    img = ax.imshow(activity_colors, interpolation='nearest', origin='upper')
    
    title = ax.set_title(f"Time: 0.0 ms", color='white', fontsize=14)
    
    # Add grid lines to clearly show neuron cells
    ax.grid(which='major', color='#555555', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_xticks(np.arange(-0.5, side_length, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, side_length, 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)
    
    # Create a background image to show the network layout
    # This helps visualize the arrangement
    background = np.zeros((side_length, side_length, 4))
    for i in range(network.n_neurons):
        if i in network.neuron_grid_positions:
            row, col = network.neuron_grid_positions[i]
            if 0 <= row < side_length and 0 <= col < side_length:
                # Set very light circle for each neuron position
                # RGBA format with low alpha for subtle effect
                background[row, col, :] = [1, 1, 1, 0.1]  # white with low alpha
                
                # Highlight stimulated neuron positions slightly more
                if i in stim_neurons:
                    background[row, col, :] = [0.8, 1, 0.8, 0.2]  # light green with higher alpha
    
    # Display the background
    ax.imshow(background, interpolation='nearest', origin='upper')
    
    # Create a text element to show if we're currently in a stimulation period
    stim_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                       color='white', fontsize=12, verticalalignment='top')
    
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
            stim_text.set_color('lime')
        else:
            stim_text.set_text("")
        
        # Decay all activity values (simulate fading over time)
        activity_grid *= 0.8  # Decay factor controls how quickly activity fades
        
        # Add new activity
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
                                      save_path="network_activity_timeline.png"):
    """
    Plot time series of network activity with stimulation times marked.
    """
    print("Plotting network activity timeline...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a1a')
    fig.suptitle("Network Activity Over Time", color='white', fontsize=16)
    
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
    ax.set_xlabel("Time (ms)", color='white')
    ax.set_ylabel("Number of Spikes", color='white')
    ax.set_title("Network Activity with Avalanches", color='white')
    
    # Add legend for stimulations
    if stim_times:
        ax.legend(loc='upper right', framealpha=0.7)
    
    # Style adjustments for dark mode
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved network activity timeline to {save_path}")
    
    return fig


def plot_psth_and_raster(activity_record, stim_times=None, bin_size=10, dt=0.1, 
                          neuron_subset=None, figsize=(14, 8), dpi=150,
                          save_path="psth_raster_plot.png"):
    """
    Create a PSTH (Peri-Stimulus Time Histogram) plot with a raster plot below it,
    designed with a dark style similar to the reference image.
    
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
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    from matplotlib.gridspec import GridSpec
    
    # Create figure with GridSpec to have different height ratios
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='#1a1a1a')
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
    ax_psth.set_ylabel('Firing Rate (Hz)', color='white', fontsize=12)
    ax_psth.set_title('PSTH and Raster Plot', color='white', fontsize=14)
    
    # Remove x-axis labels from the top plot
    ax_psth.tick_params(labelbottom=False)
    
    # Plot raster
    ax_raster.scatter(raster_data, neuron_ids, s=1, color='white', alpha=0.8)
    ax_raster.set_xlabel('Time (ms)', color='white', fontsize=12)
    ax_raster.set_ylabel('Neuron ID', color='white', fontsize=12)
    
    # Mark stimulation times if provided
    if stim_times:
        for stim_time in stim_times:
            # Draw vertical lines across both plots
            ax_psth.axvline(x=stim_time, color='#1dd1a1', linewidth=1, alpha=0.7)
            ax_raster.axvline(x=stim_time, color='#1dd1a1', linewidth=1, alpha=0.7)
    
    # Style the plots for dark theme
    for ax in [ax_psth, ax_raster]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.grid(True, alpha=0.2)
    
    # Add text for scale bar if needed
    ax_raster.text(0.02, 0.98, "Raster", transform=ax_raster.transAxes, 
                 color='white', fontsize=12, verticalalignment='top')
    ax_psth.text(0.02, 0.98, "PSTH", transform=ax_psth.transAxes, 
               color='white', fontsize=12, verticalalignment='top')
    
    
    # Save the figure
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved PSTH and raster plot to {save_path}")
    
    return fig