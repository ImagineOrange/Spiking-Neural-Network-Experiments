import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Default: do NOT set dark style globally - let functions handle it based on darkstyle parameter


def compute_avalanches_from_activity(activity_record, dt=0.1, stim_times=None,
                                      stim_duration_ms=None, buffer_ms=5.0):
    """
    Compute avalanche sizes and durations from activity record, optionally filtering
    to exclude stimulated periods.

    An avalanche is defined as a contiguous period of activity (at least one neuron firing)
    preceded and followed by silence.

    Parameters:
    -----------
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    stim_times : list, optional
        List of stimulation onset times in ms. If provided along with stim_duration_ms,
        avalanches during stimulated periods will be excluded.
    stim_duration_ms : float, optional
        Duration of each stimulation period in ms
    buffer_ms : float
        Buffer time before and after stimulation window to exclude

    Returns:
    --------
    tuple
        (avalanche_sizes, avalanche_durations, network_activity) where:
        - avalanche_sizes: list of total spikes in each avalanche
        - avalanche_durations: list of duration (in ms) of each avalanche
        - network_activity: array of spike counts at each timestep (filtered)
    """
    n_timesteps = len(activity_record)

    # Build mask for unstimulated periods (True = include this timestep)
    include_mask = np.ones(n_timesteps, dtype=bool)

    if stim_times is not None and stim_duration_ms is not None and stim_duration_ms > 0:
        for stim_time in stim_times:
            start_ms = stim_time - buffer_ms
            end_ms = stim_time + stim_duration_ms + buffer_ms
            start_step = max(0, int(start_ms / dt))
            end_step = min(n_timesteps, int(end_ms / dt))
            include_mask[start_step:end_step] = False

    # Compute network activity for included timesteps
    # For excluded timesteps, we treat them as "silence" to break avalanches
    network_activity = np.zeros(n_timesteps)
    for t, active_neurons in enumerate(activity_record):
        if include_mask[t]:
            network_activity[t] = len(active_neurons)
        # else: remains 0, acting as silence

    # Detect avalanches
    avalanche_sizes = []
    avalanche_durations = []

    in_avalanche = False
    current_size = 0
    current_duration = 0

    for t in range(n_timesteps):
        activity = network_activity[t]

        if activity > 0:
            if not in_avalanche:
                # Start new avalanche
                in_avalanche = True
                current_size = activity
                current_duration = 1
            else:
                # Continue avalanche
                current_size += activity
                current_duration += 1
        else:
            if in_avalanche:
                # End avalanche
                avalanche_sizes.append(int(current_size))
                avalanche_durations.append(current_duration * dt)  # Convert to ms
                in_avalanche = False
                current_size = 0
                current_duration = 0

    # Handle avalanche that continues to end of recording
    if in_avalanche and current_size > 0:
        avalanche_sizes.append(int(current_size))
        avalanche_durations.append(current_duration * dt)

    return avalanche_sizes, avalanche_durations, network_activity


def plot_avalanche_comparison(activity_record, dt=0.1, stim_times=None, stim_duration_ms=None,
                               buffer_ms=5.0, figsize=(14, 10), dpi=150,
                               save_path="avalanche_comparison.png", darkstyle=True):
    """
    Compare avalanche statistics between stimulated and unstimulated periods.

    Creates a 4-panel figure showing:
    1. Avalanche size distributions (log-log histogram)
    2. Avalanche duration distributions (log-log histogram)
    3. Size vs duration scaling for both periods
    4. Summary statistics comparison (bar chart)

    Parameters:
    -----------
    activity_record : list
        List of lists containing active neuron indices at each time step
    dt : float
        Time step size in ms
    stim_times : list
        List of stimulation onset times in ms
    stim_duration_ms : float
        Duration of each stimulation period in ms
    buffer_ms : float
        Buffer time around stimulation to exclude
    figsize : tuple
        Figure size
    dpi : int
        Figure resolution
    save_path : str
        Path to save the figure
    darkstyle : bool
        Use dark background style

    Returns:
    --------
    fig : matplotlib.figure.Figure
    comparison_info : dict
    """
    from matplotlib.gridspec import GridSpec

    print("Computing avalanche comparison between stim and unstim periods...")

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

    unstim_color = '#3498db'  # Blue
    stim_color = '#e74c3c'    # Red

    n_timesteps = len(activity_record)
    duration_ms = n_timesteps * dt

    # Build masks for stim and unstim periods
    stim_mask = np.zeros(n_timesteps, dtype=bool)
    if stim_times and stim_duration_ms and stim_duration_ms > 0:
        for stim_time in stim_times:
            start_ms = stim_time - buffer_ms
            end_ms = stim_time + stim_duration_ms + buffer_ms
            start_step = max(0, int(start_ms / dt))
            end_step = min(n_timesteps, int(end_ms / dt))
            stim_mask[start_step:end_step] = True

    unstim_mask = ~stim_mask

    print(f"  Total: {duration_ms:.0f}ms, Stim: {100*np.mean(stim_mask):.1f}%, Unstim: {100*np.mean(unstim_mask):.1f}%")

    # Compute avalanches for unstimulated periods
    unstim_sizes, unstim_durations, _ = compute_avalanches_from_activity(
        activity_record, dt=dt, stim_times=stim_times,
        stim_duration_ms=stim_duration_ms, buffer_ms=buffer_ms
    )
    print(f"  Unstim avalanches: {len(unstim_sizes)}")

    # Compute avalanches for stimulated periods (invert the logic)
    # Create a modified activity record that zeros out unstim periods
    stim_activity = []
    for t, active in enumerate(activity_record):
        if stim_mask[t]:
            stim_activity.append(active)
        else:
            stim_activity.append([])  # Treat as silence

    # Now compute avalanches from this stim-only record
    stim_sizes = []
    stim_durations = []
    in_avalanche = False
    current_size = 0
    current_duration = 0

    for t in range(n_timesteps):
        activity = len(stim_activity[t])
        if activity > 0:
            if not in_avalanche:
                in_avalanche = True
                current_size = activity
                current_duration = 1
            else:
                current_size += activity
                current_duration += 1
        else:
            if in_avalanche:
                stim_sizes.append(int(current_size))
                stim_durations.append(current_duration * dt)
                in_avalanche = False
                current_size = 0
                current_duration = 0

    if in_avalanche and current_size > 0:
        stim_sizes.append(int(current_size))
        stim_durations.append(current_duration * dt)

    print(f"  Stim avalanches: {len(stim_sizes)}")

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=bg_color)
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # === Panel 1: Avalanche Size Distribution ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)

    if unstim_sizes:
        # Log-spaced bins for power-law visualization
        max_size = max(max(unstim_sizes), max(stim_sizes) if stim_sizes else 1)
        bins = np.logspace(0, np.log10(max_size + 1), 30)

        hist_unstim, _ = np.histogram(unstim_sizes, bins=bins)
        hist_stim, _ = np.histogram(stim_sizes, bins=bins) if stim_sizes else (np.zeros_like(hist_unstim), None)

        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Normalize to probability density
        if np.sum(hist_unstim) > 0:
            hist_unstim = hist_unstim / np.sum(hist_unstim)
        if np.sum(hist_stim) > 0:
            hist_stim = hist_stim / np.sum(hist_stim)

        # Plot
        valid_unstim = hist_unstim > 0
        valid_stim = hist_stim > 0

        if np.any(valid_unstim):
            ax1.loglog(bin_centers[valid_unstim], hist_unstim[valid_unstim], 'o-',
                      color=unstim_color, linewidth=2, markersize=6, label=f'Unstim (n={len(unstim_sizes)})')
        if np.any(valid_stim):
            ax1.loglog(bin_centers[valid_stim], hist_stim[valid_stim], 's-',
                      color=stim_color, linewidth=2, markersize=6, label=f'Stim (n={len(stim_sizes)})')

        # Add power-law reference line (slope = -1.5 is critical)
        x_ref = np.logspace(0, np.log10(max_size), 50)
        y_ref = x_ref ** (-1.5)
        y_ref = y_ref * (hist_unstim[valid_unstim][0] / y_ref[0]) if np.any(valid_unstim) else y_ref
        ax1.loglog(x_ref, y_ref, '--', color='#1dd1a1', alpha=0.7, linewidth=2, label='Critical (α=-1.5)')

    ax1.set_facecolor(bg_color)
    ax1.set_xlabel('Avalanche Size', color=text_color, fontsize=11)
    ax1.set_ylabel('P(Size)', color=text_color, fontsize=11)
    ax1.set_title('Avalanche Size Distribution', color=text_color, fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax1.tick_params(colors=text_color)
    for spine in ax1.spines.values():
        spine.set_color(spine_color)
    ax1.grid(True, alpha=grid_alpha, which='both')

    # === Panel 2: Avalanche Duration Distribution ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)

    if unstim_durations:
        max_dur = max(max(unstim_durations), max(stim_durations) if stim_durations else 1)
        bins_dur = np.logspace(np.log10(dt), np.log10(max_dur + dt), 25)

        hist_unstim_dur, _ = np.histogram(unstim_durations, bins=bins_dur)
        hist_stim_dur, _ = np.histogram(stim_durations, bins=bins_dur) if stim_durations else (np.zeros_like(hist_unstim_dur), None)

        bin_centers_dur = (bins_dur[:-1] + bins_dur[1:]) / 2

        if np.sum(hist_unstim_dur) > 0:
            hist_unstim_dur = hist_unstim_dur / np.sum(hist_unstim_dur)
        if np.sum(hist_stim_dur) > 0:
            hist_stim_dur = hist_stim_dur / np.sum(hist_stim_dur)

        valid_unstim_dur = hist_unstim_dur > 0
        valid_stim_dur = hist_stim_dur > 0

        if np.any(valid_unstim_dur):
            ax2.loglog(bin_centers_dur[valid_unstim_dur], hist_unstim_dur[valid_unstim_dur], 'o-',
                      color=unstim_color, linewidth=2, markersize=6, label='Unstim')
        if np.any(valid_stim_dur):
            ax2.loglog(bin_centers_dur[valid_stim_dur], hist_stim_dur[valid_stim_dur], 's-',
                      color=stim_color, linewidth=2, markersize=6, label='Stim')

        # Power-law reference (slope = -2.0 is critical for duration)
        x_ref_dur = np.logspace(np.log10(dt), np.log10(max_dur), 50)
        y_ref_dur = x_ref_dur ** (-2.0)
        y_ref_dur = y_ref_dur * (hist_unstim_dur[valid_unstim_dur][0] / y_ref_dur[0]) if np.any(valid_unstim_dur) else y_ref_dur
        ax2.loglog(x_ref_dur, y_ref_dur, '--', color='#1dd1a1', alpha=0.7, linewidth=2, label='Critical (α=-2.0)')

    ax2.set_xlabel('Avalanche Duration (ms)', color=text_color, fontsize=11)
    ax2.set_ylabel('P(Duration)', color=text_color, fontsize=11)
    ax2.set_title('Avalanche Duration Distribution', color=text_color, fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color(spine_color)
    ax2.grid(True, alpha=grid_alpha, which='both')

    # === Panel 3: Size vs Duration Scaling ===
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg_color)

    unstim_sigma = None
    stim_sigma = None

    if len(unstim_sizes) > 10 and len(unstim_durations) > 10:
        ax3.loglog(unstim_durations, unstim_sizes, 'o', color=unstim_color, alpha=0.5, markersize=4, label='Unstim')
        # Fit power law
        try:
            log_dur = np.log10(unstim_durations)
            log_size = np.log10(unstim_sizes)
            slope, intercept, r, _, _ = stats.linregress(log_dur, log_size)
            unstim_sigma = slope
            x_fit = np.logspace(np.log10(min(unstim_durations)), np.log10(max(unstim_durations)), 50)
            y_fit = 10**intercept * x_fit**slope
            ax3.loglog(x_fit, y_fit, '-', color=unstim_color, linewidth=2, alpha=0.8,
                      label=f'Unstim fit: σ={slope:.2f}')
        except:
            pass

    if len(stim_sizes) > 10 and len(stim_durations) > 10:
        ax3.loglog(stim_durations, stim_sizes, 's', color=stim_color, alpha=0.5, markersize=4, label='Stim')
        try:
            log_dur = np.log10(stim_durations)
            log_size = np.log10(stim_sizes)
            slope, intercept, r, _, _ = stats.linregress(log_dur, log_size)
            stim_sigma = slope
            x_fit = np.logspace(np.log10(min(stim_durations)), np.log10(max(stim_durations)), 50)
            y_fit = 10**intercept * x_fit**slope
            ax3.loglog(x_fit, y_fit, '-', color=stim_color, linewidth=2, alpha=0.8,
                      label=f'Stim fit: σ={slope:.2f}')
        except:
            pass

    # Critical reference line (σ = 1.5)
    if unstim_durations or stim_durations:
        all_dur = unstim_durations + stim_durations
        x_crit = np.logspace(np.log10(min(all_dur)), np.log10(max(all_dur)), 50)
        y_crit = x_crit ** 1.5
        # Scale to match data
        if unstim_sizes:
            y_crit = y_crit * (np.median(unstim_sizes) / np.median(y_crit))
        ax3.loglog(x_crit, y_crit, '--', color='#1dd1a1', linewidth=2, alpha=0.7, label='Critical (σ=1.5)')

    ax3.set_xlabel('Duration (ms)', color=text_color, fontsize=11)
    ax3.set_ylabel('Size', color=text_color, fontsize=11)
    ax3.set_title('Size-Duration Scaling', color=text_color, fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax3.tick_params(colors=text_color)
    for spine in ax3.spines.values():
        spine.set_color(spine_color)
    ax3.grid(True, alpha=grid_alpha, which='both')

    # === Panel 4: Summary Statistics ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg_color)

    # Compute summary stats
    unstim_mean_size = np.mean(unstim_sizes) if unstim_sizes else 0
    stim_mean_size = np.mean(stim_sizes) if stim_sizes else 0
    unstim_max_size = max(unstim_sizes) if unstim_sizes else 0
    stim_max_size = max(stim_sizes) if stim_sizes else 0

    metrics = ['Count', 'Mean Size', 'Max Size', 'σ Exponent']
    unstim_vals = [len(unstim_sizes), unstim_mean_size, unstim_max_size, unstim_sigma if unstim_sigma else 0]
    stim_vals = [len(stim_sizes), stim_mean_size, stim_max_size, stim_sigma if stim_sigma else 0]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize for display (different scales)
    unstim_display = [unstim_vals[0]/max(1, max(unstim_vals[0], stim_vals[0])),
                      unstim_vals[1]/max(1, max(unstim_vals[1], stim_vals[1])),
                      unstim_vals[2]/max(1, max(unstim_vals[2], stim_vals[2])),
                      unstim_vals[3]/2.0 if unstim_vals[3] else 0]  # Normalize σ to 2.0
    stim_display = [stim_vals[0]/max(1, max(unstim_vals[0], stim_vals[0])),
                    stim_vals[1]/max(1, max(unstim_vals[1], stim_vals[1])),
                    stim_vals[2]/max(1, max(unstim_vals[2], stim_vals[2])),
                    stim_vals[3]/2.0 if stim_vals[3] else 0]

    bars1 = ax4.bar(x - width/2, unstim_display, width, color=unstim_color, label='Unstim', edgecolor='white')
    bars2 = ax4.bar(x + width/2, stim_display, width, color=stim_color, label='Stim', edgecolor='white')

    # Add value labels
    for bar, val in zip(bars1, unstim_vals):
        if val > 0:
            label = f'{val:.0f}' if val > 10 else f'{val:.2f}'
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    label, ha='center', va='bottom', color=text_color, fontsize=8)
    for bar, val in zip(bars2, stim_vals):
        if val > 0:
            label = f'{val:.0f}' if val > 10 else f'{val:.2f}'
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    label, ha='center', va='bottom', color=text_color, fontsize=8)

    # Add critical reference line for σ
    ax4.axhline(y=1.5/2.0, color='#1dd1a1', linestyle='--', alpha=0.7, linewidth=2)
    ax4.text(3.5, 1.5/2.0 + 0.05, 'Critical σ=1.5', color='#1dd1a1', fontsize=9, ha='right')

    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, color=text_color, fontsize=10)
    ax4.set_ylabel('Normalized Value', color=text_color, fontsize=11)
    ax4.set_title('Summary Comparison', color=text_color, fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, facecolor=bg_color, edgecolor=spine_color, labelcolor=text_color)
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_color(spine_color)
    ax4.set_ylim(0, 1.3)

    # Main title
    fig.suptitle('Avalanche Statistics: Unstimulated vs Stimulated Periods',
                 color=text_color, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=bg_color)
        print(f"Saved avalanche comparison to {save_path}")

    # Compile results
    comparison_info = {
        'unstim': {
            'count': len(unstim_sizes),
            'mean_size': float(unstim_mean_size),
            'max_size': int(unstim_max_size),
            'sigma': float(unstim_sigma) if unstim_sigma else None,
            'sizes': unstim_sizes,
            'durations': unstim_durations
        },
        'stim': {
            'count': len(stim_sizes),
            'mean_size': float(stim_mean_size),
            'max_size': int(stim_max_size),
            'sigma': float(stim_sigma) if stim_sigma else None,
            'sizes': stim_sizes,
            'durations': stim_durations
        }
    }

    # Print summary
    print(f"\n  === Avalanche Comparison Summary ===")
    unstim_sigma_str = f"{unstim_sigma:.2f}" if unstim_sigma else "N/A"
    stim_sigma_str = f"{stim_sigma:.2f}" if stim_sigma else "N/A"
    print(f"  Unstim: {len(unstim_sizes)} avalanches, mean size={unstim_mean_size:.1f}, max={unstim_max_size}, σ={unstim_sigma_str}")
    print(f"  Stim:   {len(stim_sizes)} avalanches, mean size={stim_mean_size:.1f}, max={stim_max_size}, σ={stim_sigma_str}")
    if unstim_sigma and stim_sigma:
        print(f"  Critical σ=1.5: Unstim {'NEAR' if 1.3 <= unstim_sigma <= 1.7 else 'FAR'}, Stim {'NEAR' if 1.3 <= stim_sigma <= 1.7 else 'FAR'}")

    return fig, comparison_info


def plot_individual_avalanche_statistics(network, save_path_prefix="avalanche", figsize=(10, 8), darkstyle=True,
                                         activity_record=None, dt=0.1, stim_times=None,
                                         stim_duration_ms=None, buffer_ms=5.0):
    """
    Create plots showing avalanche size vs duration scaling and branching ratio visualization.

    Note: Individual size and duration distribution plots have been removed because they are
    intrinsically bounded by network size and simulation length, making power-law fits unreliable.
    The size-duration scaling relationship (σ exponent) is more robust as it measures the
    relationship between size and duration rather than fitting truncated distributions.

    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing avalanche data
    save_path_prefix : str
        Prefix for saving the output files (will append _size_vs_duration.png, _branching.png)
    figsize : tuple
        Figure size (width, height) in inches
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)
    activity_record : list, optional
        If provided along with stim_times/stim_duration_ms, avalanches will be recomputed
        from this record, filtering out stimulated periods.
    dt : float
        Time step size in ms (used when recomputing avalanches)
    stim_times : list, optional
        List of stimulation onset times in ms
    stim_duration_ms : float, optional
        Duration of each stimulation period in ms
    buffer_ms : float
        Buffer time around stimulation to exclude

    Returns:
    --------
    tuple
        (scatter_fig, branching_fig) - The figure objects
    """
    # Determine whether to use filtered avalanche data
    use_filtered = (activity_record is not None and stim_times is not None
                    and stim_duration_ms is not None and stim_duration_ms > 0)

    if use_filtered:
        print("  Recomputing avalanches from activity record (excluding stimulated periods)...")
        avalanche_sizes, avalanche_durations, network_activity = compute_avalanches_from_activity(
            activity_record, dt=dt, stim_times=stim_times,
            stim_duration_ms=stim_duration_ms, buffer_ms=buffer_ms
        )
        print(f"  Found {len(avalanche_sizes)} avalanches in unstimulated periods")
    else:
        # Use data from network object
        avalanche_sizes = network.avalanche_sizes
        avalanche_durations = network.avalanche_durations
        network_activity = np.array(network.network_activity)

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        box_color = '#222222'
        fit_line_color = 'white'
        default_info_text_color = 'white'
    else:
        bg_color = 'white'
        text_color = 'black'
        box_color = '#eeeeee'
        fit_line_color = 'black'
        default_info_text_color = 'black'

    if not avalanche_sizes:
        print("No avalanches recorded.")
        return None, None

    # ===== 1. SCATTER PLOT OF SIZE VS DURATION =====
    scatter_fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax_scatter = scatter_fig.add_subplot(111)

    # Plot scatter of size vs duration
    ax_scatter.loglog(avalanche_durations, avalanche_sizes, 'o',
                     color='#9b59b6', alpha=0.6, markersize=6, label="Individual Avalanches")
    # Force background color again after loglog (which sometimes resets it)
    ax_scatter.set_facecolor(bg_color)

    # Try to fit a power law relationship
    if len(avalanche_sizes) > 10:
        try:
            # Convert to log space for linear fitting
            log_durations = np.log10(avalanche_durations)
            log_sizes = np.log10(avalanche_sizes)

            # Linear regression in log-log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_durations, log_sizes)

            # Generate points for the fitted line
            x_fit = np.logspace(np.log10(min(avalanche_durations)),
                              np.log10(max(avalanche_durations)), 100)
            y_fit = 10**(intercept) * (x_fit ** slope)
            
            # Plot the fitted relationship
            ax_scatter.loglog(x_fit, y_fit, '--', color=fit_line_color, alpha=0.7, linewidth=2.5,
                           label=f"Fit: Size ~ Duration^{slope:.2f}, R²={r_value**2:.2f}")

            # Force background color again after loglog (which sometimes resets it)
            ax_scatter.set_facecolor(bg_color)

            # Mark the critical theoretical relationship (slope = 1.5)
            critical_y = 10**(intercept) * (x_fit ** 1.5)
            ax_scatter.loglog(x_fit, critical_y, '-.', color='#1dd1a1', alpha=0.7, linewidth=2,
                           label=f"Critical Theory: Size ~ Duration^1.5")
            # Force background color again after loglog (which sometimes resets it)
            ax_scatter.set_facecolor(bg_color)        

        except Exception as e:
            print(f"Regression error for size vs duration: {e}")
    
    # Style and label the plot
    ax_scatter.set_xlabel("Avalanche Duration (ms)", color=text_color, fontsize=14)
    ax_scatter.set_ylabel("Avalanche Size", color=text_color, fontsize=14)
    ax_scatter.set_title("Avalanche Size vs Duration", color=text_color, fontsize=16)
    ax_scatter.legend(loc='best', framealpha=0.7, fontsize=12)
    
    # Add grid and improve appearance
    ax_scatter.grid(True, which="both", ls="-", alpha=0.2)
    ax_scatter.tick_params(colors=text_color, labelsize=12)
    for spine in ax_scatter.spines.values():
        spine.set_color(text_color)
    
    # Add information text about criticality
    if len(avalanche_sizes) > 10:
        text = f"Critical theory predicts: Size ~ Duration^1.5\n"
        text += f"Current exponent: {slope:.2f}\n"
        text += f"Correlation: R²={r_value**2:.2f}"
        
        # Color-code based on how close to ideal exponent
        info_text_color = '#1dd1a1' if 1.3 <= slope <= 1.7 else default_info_text_color
        ax_scatter.text(0.03, 0.03, text, transform=ax_scatter.transAxes, fontsize=11,
                      verticalalignment='bottom', horizontalalignment='left',
                      color=info_text_color, bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    if save_path_prefix:
        scatter_path = f"{save_path_prefix}_size_vs_duration.png"
        scatter_fig.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"Saved size vs duration scatter plot to {scatter_path}")
    
    # ===== 4. BRANCHING PARAMETER VISUALIZATION (KEEPING THE ORIGINAL) =====
    branching_fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax_branch = branching_fig.add_subplot(111)
    
    # Calculate branching ratio (number of descendants per ancestor)
    activity = np.array(network_activity) if isinstance(network_activity, list) else network_activity
    non_zero_idx = np.where(activity > 0)[0]

    # We need to ensure we have enough data and no division by zero
    # Note: We filter out indices at the boundary when computing next_idx below
    if len(non_zero_idx) > 1:
        # Get indices of the next timestep after each active period
        next_idx = non_zero_idx + 1
        # Keep only valid indices (not past the end of the array)
        valid_idx = next_idx[next_idx < len(activity)]
        prev_idx = valid_idx - 1

        # Only include cases where the previous activity is non-zero (avoid div by zero)
        valid_mask = activity[prev_idx] > 0
        if np.sum(valid_mask) > 0:
            branching_ratios = activity[valid_idx[valid_mask]] / activity[prev_idx[valid_mask]]
            
            # Calculate the average branching ratio
            avg_branching = np.mean(branching_ratios)
            
            # Create histogram of branching ratios
            hist, bin_edges = np.histogram(branching_ratios, bins=30, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            # Plot histogram
            ax_branch.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                        alpha=0.7, color='#3498db', label='Distribution')
            
            # Plot average value line
            ax_branch.axvline(x=avg_branching, color='#e74c3c', linestyle='--', linewidth=2.5, 
                            label=f'Mean: {avg_branching:.3f}')
            
            # Plot critical value line (branching = 1.0)
            ax_branch.axvline(x=1.0, color='#2ecc71', linestyle='-', linewidth=2.5,
                            label='Critical: 1.0')
            
            # Style plot
            ax_branch.set_xlabel('Branching Ratio (Descendants/Ancestors)', color=text_color, fontsize=14)
            ax_branch.set_ylabel('Probability Density', color=text_color, fontsize=14)
            ax_branch.set_title('Branching Parameter Distribution', color=text_color, fontsize=16)
            
            # Add info text about criticality
            text = f"Critical branching occurs at exactly 1.0\n"
            text += f"Current average: {avg_branching:.3f}\n"
            
            # Classify the system based on branching
            if 0.95 <= avg_branching <= 1.05:
                status = "CRITICAL"
                text_color = '#2ecc71'  # Green
            elif avg_branching < 0.95:
                status = "SUB-CRITICAL"
                text_color = '#3498db'  # Blue
            else:
                status = "SUPER-CRITICAL"
                text_color = '#e74c3c'  # Red
                
            text += f"Network state: {status}"
            
            ax_branch.text(0.03, 0.97, text, transform=ax_branch.transAxes, fontsize=11,
                         verticalalignment='top', horizontalalignment='left',
                         color=text_color, bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
        else:
            ax_branch.text(0.5, 0.5, "Insufficient data for branching analysis",
                         transform=ax_branch.transAxes, fontsize=14, color=text_color,
                         ha='center', va='center')
    else:
        ax_branch.text(0.5, 0.5, "Insufficient activity data for branching analysis",
                     transform=ax_branch.transAxes, fontsize=14, color=text_color,
                     ha='center', va='center')

    # Add legend only if we have artists with labels
    handles, labels = ax_branch.get_legend_handles_labels()
    if handles:
        ax_branch.legend(loc='upper right', framealpha=0.7, fontsize=12)
    
    # Style improvements
    ax_branch.set_facecolor(bg_color)
    ax_branch.tick_params(colors=text_color, labelsize=12)
    for spine in ax_branch.spines.values():
        spine.set_color(text_color)
    ax_branch.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path_prefix:
        branch_path = f"{save_path_prefix}_branching.png"
        branching_fig.savefig(branch_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"Saved branching parameter plot to {branch_path}")
    
    return scatter_fig, branching_fig


# Function to use in the main code to replace the original visualization
def plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche", darkstyle=True,
                                        activity_record=None, dt=0.1, stim_times=None,
                                        stim_duration_ms=None, buffer_ms=5.0):
    """
    Enhanced version of criticality analysis focusing on robust metrics.

    This function creates visualizations for size-duration scaling and branching ratio.
    Individual size and duration distributions are not plotted because they are
    intrinsically bounded by network size and simulation length.

    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing avalanche data
    save_path_prefix : str
        Prefix for saving the output files
    darkstyle : bool
        If True, use dark background style. If False, use white background (default: True)
    activity_record : list, optional
        If provided along with stim_times/stim_duration_ms, avalanches will be recomputed
        from this record, filtering out stimulated periods.
    dt : float
        Time step size in ms (used when recomputing avalanches)
    stim_times : list, optional
        List of stimulation onset times in ms
    stim_duration_ms : float, optional
        Duration of each stimulation period in ms
    buffer_ms : float
        Buffer time around stimulation to exclude

    Returns:
    --------
    dict
        Dictionary with the generated figure objects and analysis results
    """
    # Determine whether to use filtered avalanche data
    use_filtered = (activity_record is not None and stim_times is not None
                    and stim_duration_ms is not None and stim_duration_ms > 0)

    if use_filtered:
        print("Computing avalanches from unstimulated periods only...")
        avalanche_sizes, avalanche_durations, network_activity = compute_avalanches_from_activity(
            activity_record, dt=dt, stim_times=stim_times,
            stim_duration_ms=stim_duration_ms, buffer_ms=buffer_ms
        )
        print(f"  Found {len(avalanche_sizes)} avalanches in unstimulated periods")
    else:
        # Use data from network object
        avalanche_sizes = network.avalanche_sizes
        avalanche_durations = network.avalanche_durations
        network_activity = np.array(network.network_activity)

    # Check if we have avalanche data
    if not avalanche_sizes or len(avalanche_sizes) < 5:
        print("Insufficient avalanche data for analysis.")
        return {"success": False, "message": "Insufficient avalanche data"}

    # Create visualizations (size vs duration scatter and branching ratio)
    scatter_fig, branching_fig = plot_individual_avalanche_statistics(
        network, save_path_prefix=save_path_prefix, darkstyle=darkstyle,
        activity_record=activity_record, dt=dt, stim_times=stim_times,
        stim_duration_ms=stim_duration_ms, buffer_ms=buffer_ms
    )

    # Run basic analysis to return metrics
    try:
        # Calculate scaling exponent (size vs duration) - this is the σ exponent
        log_avalanche_sizes = np.log10(avalanche_sizes)
        log_avalanche_durations = np.log10(avalanche_durations)
        slope, intercept, scaling_r_value, _, _ = stats.linregress(log_avalanche_durations, log_avalanche_sizes)

        # Calculate branching parameter
        activity = np.array(network_activity) if isinstance(network_activity, list) else network_activity
        non_zero_idx = np.where(activity > 0)[0]
        branching_ratio = None

        if len(non_zero_idx) > 1:
            next_idx = non_zero_idx + 1
            valid_idx = next_idx[next_idx < len(activity)]
            prev_idx = valid_idx - 1
            valid_mask = activity[prev_idx] > 0

            if np.sum(valid_mask) > 0:
                branching_ratios = activity[valid_idx[valid_mask]] / activity[prev_idx[valid_mask]]
                branching_ratio = float(np.mean(branching_ratios))

        # Compile analysis results (only robust metrics)
        analysis = {
            "avalanche_count": len(network.avalanche_sizes),
            "size_duration_scaling": float(slope),
            "size_duration_r_squared": float(scaling_r_value**2),
            "branching_ratio": branching_ratio
        }

        # Determine criticality based on robust criteria only
        is_critical_branching = branching_ratio is not None and 0.95 <= branching_ratio <= 1.05
        is_critical_scaling = 1.3 <= slope <= 1.7

        # Overall assessment based on two robust metrics
        if is_critical_branching and is_critical_scaling:
            assessment = "Strongly critical"
            is_critical = True
        elif is_critical_branching or is_critical_scaling:
            assessment = "Moderately critical"
            is_critical = True
        else:
            assessment = "Not critical"
            is_critical = False

        analysis["assessment"] = assessment
        analysis["is_critical"] = is_critical

        print(f"\n===== Enhanced Criticality Analysis =====")
        print(f"Analyzed {analysis['avalanche_count']} avalanches")
        print(f"Size-duration scaling (σ): {analysis['size_duration_scaling']:.3f} (R² = {analysis['size_duration_r_squared']:.3f}, ideal: ~1.5)")
        if branching_ratio is not None:
            print(f"Branching ratio: {analysis['branching_ratio']:.3f} (ideal: ~1.0)")
        print(f"Assessment: {analysis['assessment']}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        analysis = {"error": str(e)}

    # Return all figures and analysis in a dictionary
    return {
        "success": True,
        "figures": {
            "scatter": scatter_fig,
            "branching": branching_fig
        },
        "analysis": analysis
    }


def analyze_criticality_comprehensively(network, save_plots=True, min_avalanches=20):
    """
    Comprehensive analysis of criticality in neuronal networks using multiple methods.
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing avalanche data
    save_plots : bool
        Whether to save visualization plots
    min_avalanches : int
        Minimum number of avalanches required for analysis
    
    Returns:
    --------
    dict
        Dictionary with comprehensive criticality metrics and assessments
    """
    # Check if we have enough avalanches
    if len(network.avalanche_sizes) < min_avalanches:
        print(f"Not enough avalanches ({len(network.avalanche_sizes)}) for reliable analysis. Need at least {min_avalanches}.")
        return {"critical": False, "reliable": False, "avalanche_count": len(network.avalanche_sizes)}
    
    # Initialize results dictionary
    results = {
        "avalanche_count": len(network.avalanche_sizes),
        "reliable": True,
        "methods": {},
        "branching_ratio": None,
        "scaling_relation": None,
        "critical_score": 0.0,
        "critical": False
    }
    
    # ===== METHOD 1: Log-binning + Linear Regression =====
    try:
        # For sizes
        size_min, size_max = min(network.avalanche_sizes), max(network.avalanche_sizes)
        if size_min == size_max:
            size_min, size_max = 0.9 * size_min, 1.1 * size_max
        
        # Use logarithmic binning - crucial for power law analysis
        size_bins = np.logspace(np.log10(size_min), np.log10(size_max), 15)
        
        hist_sizes, bin_edges = np.histogram(network.avalanche_sizes, bins=size_bins)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        # Only proceed with bins that have data
        valid_indices = hist_sizes > 0
        if np.sum(valid_indices) > 3:  # Need at least 4 points for regression
            log_counts = np.log10(hist_sizes[valid_indices])
            log_bins = np.log10(bin_centers[valid_indices])
            
            # Fit a line to the log-log plot
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_bins, log_counts)
            
            size_exponent_bin = -slope  # Negative because expected relation is y ~ x^-α
            size_r_value = abs(r_value)
            
            # For durations - similar approach
            dur_min, dur_max = min(network.avalanche_durations), max(network.avalanche_durations)
            if dur_min == dur_max:
                dur_min, dur_max = 0.9 * dur_min, 1.1 * dur_max
                
            dur_bins = np.logspace(np.log10(dur_min), np.log10(dur_max), 15)
            
            hist_durs, bin_edges = np.histogram(network.avalanche_durations, bins=dur_bins)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            valid_indices = hist_durs > 0
            if np.sum(valid_indices) > 3:
                log_counts = np.log10(hist_durs[valid_indices])
                log_bins = np.log10(bin_centers[valid_indices])
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_bins, log_counts)
                
                dur_exponent_bin = -slope
                dur_r_value = abs(r_value)
                
                # Store results from method 1
                results["methods"]["binning"] = {
                    "size_exponent": size_exponent_bin,
                    "size_r_value": size_r_value,
                    "duration_exponent": dur_exponent_bin,
                    "duration_r_value": dur_r_value
                }
    
    except Exception as e:
        print(f"Error in log-binning method: {e}")
        results["methods"]["binning"] = None
    
    # ===== METHOD 2: Direct Powerlaw Fit =====
    try:
        # For sizes
        alpha_size, loc_size, scale_size = stats.powerlaw.fit(network.avalanche_sizes)
        
        # For durations
        alpha_dur, loc_dur, scale_dur = stats.powerlaw.fit(network.avalanche_durations)
        
        # Store results from method 2
        results["methods"]["powerlaw"] = {
            "size_exponent": alpha_size,
            "duration_exponent": alpha_dur
        }
        
    except Exception as e:
        print(f"Error in powerlaw fit method: {e}")
        results["methods"]["powerlaw"] = None
    
    # ===== Calculate Branching Ratio =====
    try:
        # Calculate the branching ratio (average # of neurons activated per active neuron)
        activity = np.array(network.network_activity)
        non_zero_idx = np.where(activity > 0)[0]
        
        if len(non_zero_idx) > 0 and np.max(non_zero_idx) < len(activity) - 1:
            next_idx = non_zero_idx + 1
            valid_idx = next_idx[next_idx < len(activity)]
            prev_idx = valid_idx - 1
            valid_mask = activity[prev_idx] > 0
            
            if np.sum(valid_mask) > 0:
                branching_ratios = activity[valid_idx[valid_mask]] / activity[prev_idx[valid_mask]]
                results["branching_ratio"] = float(np.mean(branching_ratios))
    
    except Exception as e:
        print(f"Error calculating branching ratio: {e}")
    
    # ===== Calculate Scaling Relation =====
    try:
        # Check for avalanche shape collapse (scaling relation)
        durations = sorted(set(network.avalanche_durations))
        avg_sizes = []
        
        for dur in durations:
            indices = [i for i, d in enumerate(network.avalanche_durations) if d == dur]
            if len(indices) > 0:
                avg_size = np.mean([network.avalanche_sizes[i] for i in indices])
                avg_sizes.append(avg_size)
        
        # In critical systems, avg_size ~ duration^(1/sigma*nu*z)
        # where sigma*nu*z is typically around 1.5
        if len(durations) > 5 and len(avg_sizes) > 5:
            # Fit power law to relationship between duration and average size
            log_durations = np.log10(durations)
            log_avg_sizes = np.log10(avg_sizes)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_durations, log_avg_sizes)
            
            results["scaling_relation"] = float(slope)
            results["scaling_relation_r"] = float(abs(r_value))
    
    except Exception as e:
        print(f"Error calculating scaling relation: {e}")
    
    # ===== Calculate Overall Criticality Score =====
    # Note: Only using branching ratio and size-duration scaling (σ exponent)
    # Individual size/duration exponents are excluded because they are intrinsically
    # bounded by network size and simulation length, making power-law fits unreliable.

    # Define ideal values and weights (only robust metrics)
    ideal_values = {
        "branching_ratio": 1.0,      # Ideal: ~1.0
        "scaling_relation": 1.5      # Ideal: ~1.5
    }

    # Equal weights for the two robust metrics
    weights = {
        "branching_ratio": 0.5,       # 50% of score
        "scaling_relation": 0.5       # 50% of score
    }

    score = 0.0
    weight_sum = 0.0

    # Branching ratio score
    if results["branching_ratio"] is not None:
        branching_ratio = results["branching_ratio"]
        # Score based on how close branching ratio is to 1.0
        # Use exponential decay for consistency with scaling score
        br_score = np.exp(-2.0 * abs(branching_ratio - ideal_values["branching_ratio"]))
        score += weights["branching_ratio"] * br_score
        weight_sum += weights["branching_ratio"]

    # Scaling relation score
    if results["scaling_relation"] is not None:
        scaling_relation = results["scaling_relation"]
        # Score based on how close scaling relation is to 1.5
        scaling_score = np.exp(-2.0 * abs(scaling_relation - ideal_values["scaling_relation"]))
        score += weights["scaling_relation"] * scaling_score
        weight_sum += weights["scaling_relation"]

    # Normalize by the sum of weights for metrics that were actually calculated
    if weight_sum > 0:
        score /= weight_sum

    # Bonus for having many avalanches (more statistical confidence)
    avalanche_count_bonus = min(0.15, results["avalanche_count"] / 1000)
    score = score * (1.0 + avalanche_count_bonus)

    # Cap at 1.0
    score = min(1.0, score)

    results["critical_score"] = float(score)

    # Determine if the system is critical
    # More nuanced assessment than a simple binary
    if score >= 0.85:
        results["critical"] = True
        results["assessment"] = "Strongly critical"
    elif score >= 0.7:
        results["critical"] = True
        results["assessment"] = "Moderately critical"
    elif score >= 0.5:
        results["critical"] = False
        results["assessment"] = "Weakly critical / near-critical"
    else:
        results["critical"] = False
        results["assessment"] = "Not critical"

    # Print summary
    print("\n===== Comprehensive Criticality Analysis =====")
    print(f"Analyzed {results['avalanche_count']} avalanches")
    print(f"Critical Score: {results['critical_score']:.4f} - {results['assessment']}")

    print(f"\nRobust metrics used for scoring:")
    if results["branching_ratio"] is not None:
        print(f"  Branching ratio: {results['branching_ratio']:.3f} (ideal: ~1.0)")

    if results["scaling_relation"] is not None:
        print(f"  Size-duration scaling (σ): {results['scaling_relation']:.3f} (ideal: ~1.5)")

    # Print warnings for reliability
    if results["avalanche_count"] < 100:
        print("\nWarning: Low avalanche count may reduce statistical reliability")
        if results["avalanche_count"] < 50:
            results["reliable"] = False

    # Create visualization plots if requested
    if save_plots:
        plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche")

    return results


# Modified function to replace plot_avalanche_statistics for backward compatibility
def plot_avalanche_statistics(network, save_path="avalanche_statistics.png", figsize=(12, 8)):
    """
    Backward compatibility function that calls the enhanced plot function.
    """
    print("Using enhanced avalanche statistics visualization...")
    result = plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche")
    if result.get("success") and result.get("figures"):
        return result["figures"].get("scatter")  # Return the scatter figure for backward compatibility
    return None


def plot_stimulated_vs_unstimulated_criticality(
    activity_record,
    stim_times,
    stim_duration_ms,
    dt=0.1,
    buffer_ms=5.0,
    save_path="criticality_comparison.png",
    darkstyle=True,
    figsize=(14, 10)
):
    """
    Compare criticality metrics between stimulated and unstimulated periods.

    Creates a figure with branching ratio distributions and size-duration scaling
    curves for both periods (no stimulation vs during/around stimulation).

    Parameters:
    -----------
    activity_record : list of lists
        Each element contains indices of neurons that spiked at that timestep
    stim_times : list
        List of stimulation onset times in ms (the interval start times)
    stim_duration_ms : float
        Duration of each stimulation period in ms
    dt : float
        Simulation timestep in ms
    buffer_ms : float
        Buffer time before and after stimulation window (default 5ms)
    save_path : str
        Path to save the figure
    darkstyle : bool
        Use dark background style
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    tuple
        (fig, analysis_dict) - The figure and a dictionary with analysis results
    """
    # Set style colors
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        box_color = '#222222'
        grid_alpha = 0.2
    else:
        bg_color = 'white'
        text_color = 'black'
        box_color = '#eeeeee'
        grid_alpha = 0.3

    n_steps = len(activity_record)
    total_duration = n_steps * dt

    # Build mask for stimulated periods (including buffer)
    stim_mask = np.zeros(n_steps, dtype=bool)

    for stim_time in stim_times:
        # Stimulated window with buffer on both sides
        start_ms = stim_time - buffer_ms
        end_ms = stim_time + stim_duration_ms + buffer_ms

        start_step = max(0, int(start_ms / dt))
        end_step = min(n_steps, int(end_ms / dt))

        stim_mask[start_step:end_step] = True

    unstim_mask = ~stim_mask

    print(f"\nCriticality comparison analysis:")
    print(f"  Total simulation: {total_duration:.1f} ms ({n_steps} steps)")
    print(f"  Stimulated period steps: {np.sum(stim_mask)} ({100*np.sum(stim_mask)/n_steps:.1f}%)")
    print(f"  Unstimulated period steps: {np.sum(unstim_mask)} ({100*np.sum(unstim_mask)/n_steps:.1f}%)")

    def compute_avalanches_from_activity(mask_indices):
        """
        Detect avalanches from a subset of activity.
        An avalanche starts when activity appears after silence and ends when activity ceases.
        """
        avalanche_sizes = []
        avalanche_durations = []

        # Convert mask indices to a contiguous activity array for this period
        activity_counts = []
        for idx in mask_indices:
            activity_counts.append(len(activity_record[idx]))

        activity_counts = np.array(activity_counts)

        in_avalanche = False
        current_size = 0
        current_duration = 0

        for count in activity_counts:
            if count > 0:
                if not in_avalanche:
                    # Start new avalanche
                    in_avalanche = True
                    current_size = count
                    current_duration = 1
                else:
                    # Continue avalanche
                    current_size += count
                    current_duration += 1
            else:
                if in_avalanche:
                    # End avalanche
                    avalanche_sizes.append(current_size)
                    avalanche_durations.append(current_duration * dt)  # Convert to ms
                    in_avalanche = False
                    current_size = 0
                    current_duration = 0

        # Handle avalanche that extends to end
        if in_avalanche and current_size > 0:
            avalanche_sizes.append(current_size)
            avalanche_durations.append(current_duration * dt)

        return np.array(avalanche_sizes), np.array(avalanche_durations)

    def compute_branching_ratios(mask_indices):
        """Compute branching ratios for activity in the given period."""
        activity_counts = np.array([len(activity_record[idx]) for idx in mask_indices])

        non_zero_idx = np.where(activity_counts > 0)[0]
        if len(non_zero_idx) < 2:
            return np.array([])

        next_idx = non_zero_idx + 1
        valid_idx = next_idx[next_idx < len(activity_counts)]
        prev_idx = valid_idx - 1
        valid_mask = activity_counts[prev_idx] > 0

        if np.sum(valid_mask) == 0:
            return np.array([])

        branching_ratios = activity_counts[valid_idx[valid_mask]] / activity_counts[prev_idx[valid_mask]]
        return branching_ratios

    # Get indices for each period
    stim_indices = np.where(stim_mask)[0]
    unstim_indices = np.where(unstim_mask)[0]

    # Compute avalanches for each period
    unstim_sizes, unstim_durations = compute_avalanches_from_activity(unstim_indices)
    stim_sizes, stim_durations = compute_avalanches_from_activity(stim_indices)

    # Compute branching ratios for each period
    unstim_branching = compute_branching_ratios(unstim_indices)
    stim_branching = compute_branching_ratios(stim_indices)

    print(f"  Unstimulated avalanches: {len(unstim_sizes)}")
    print(f"  Stimulated avalanches: {len(stim_sizes)}")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=bg_color)

    # Colors for each period
    unstim_color = '#3498db'  # Blue for unstimulated
    stim_color = '#e74c3c'    # Red for stimulated

    # ===== Top Left: Branching Ratio Distribution (Unstimulated) =====
    ax_br_unstim = axes[0, 0]
    ax_br_unstim.set_facecolor(bg_color)

    if len(unstim_branching) > 5:
        hist, bin_edges = np.histogram(unstim_branching, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax_br_unstim.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] if len(bin_centers) > 1 else 0.1,
                         alpha=0.7, color=unstim_color, label='Distribution')

        avg_br = np.mean(unstim_branching)
        ax_br_unstim.axvline(x=avg_br, color='#f39c12', linestyle='--', linewidth=2.5,
                             label=f'Mean: {avg_br:.3f}')
        ax_br_unstim.axvline(x=1.0, color='#2ecc71', linestyle='-', linewidth=2.5,
                             label='Critical: 1.0')
        ax_br_unstim.legend(loc='upper right', fontsize=10)

        # Assessment text
        if 0.95 <= avg_br <= 1.05:
            status = "CRITICAL"
            status_color = '#2ecc71'
        elif avg_br < 0.95:
            status = "SUB-CRITICAL"
            status_color = '#3498db'
        else:
            status = "SUPER-CRITICAL"
            status_color = '#e74c3c'

        ax_br_unstim.text(0.03, 0.97, f"Mean: {avg_br:.3f}\nState: {status}",
                          transform=ax_br_unstim.transAxes, fontsize=11,
                          verticalalignment='top', color=status_color,
                          bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
    else:
        ax_br_unstim.text(0.5, 0.5, "Insufficient data", transform=ax_br_unstim.transAxes,
                          fontsize=14, color=text_color, ha='center', va='center')

    ax_br_unstim.set_xlabel('Branching Ratio', color=text_color, fontsize=12)
    ax_br_unstim.set_ylabel('Probability Density', color=text_color, fontsize=12)
    ax_br_unstim.set_title('Branching Ratio (Unstimulated)', color=unstim_color, fontsize=14, fontweight='bold')
    ax_br_unstim.tick_params(colors=text_color)
    ax_br_unstim.grid(True, alpha=grid_alpha)
    for spine in ax_br_unstim.spines.values():
        spine.set_color(text_color)

    # ===== Top Right: Branching Ratio Distribution (Stimulated) =====
    ax_br_stim = axes[0, 1]
    ax_br_stim.set_facecolor(bg_color)

    if len(stim_branching) > 5:
        hist, bin_edges = np.histogram(stim_branching, bins=30, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax_br_stim.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0] if len(bin_centers) > 1 else 0.1,
                       alpha=0.7, color=stim_color, label='Distribution')

        avg_br = np.mean(stim_branching)
        ax_br_stim.axvline(x=avg_br, color='#f39c12', linestyle='--', linewidth=2.5,
                           label=f'Mean: {avg_br:.3f}')
        ax_br_stim.axvline(x=1.0, color='#2ecc71', linestyle='-', linewidth=2.5,
                           label='Critical: 1.0')
        ax_br_stim.legend(loc='upper right', fontsize=10)

        # Assessment text
        if 0.95 <= avg_br <= 1.05:
            status = "CRITICAL"
            status_color = '#2ecc71'
        elif avg_br < 0.95:
            status = "SUB-CRITICAL"
            status_color = '#3498db'
        else:
            status = "SUPER-CRITICAL"
            status_color = '#e74c3c'

        ax_br_stim.text(0.03, 0.97, f"Mean: {avg_br:.3f}\nState: {status}",
                        transform=ax_br_stim.transAxes, fontsize=11,
                        verticalalignment='top', color=status_color,
                        bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
    else:
        ax_br_stim.text(0.5, 0.5, "Insufficient data", transform=ax_br_stim.transAxes,
                        fontsize=14, color=text_color, ha='center', va='center')

    ax_br_stim.set_xlabel('Branching Ratio', color=text_color, fontsize=12)
    ax_br_stim.set_ylabel('Probability Density', color=text_color, fontsize=12)
    ax_br_stim.set_title('Branching Ratio (During Stimulation)', color=stim_color, fontsize=14, fontweight='bold')
    ax_br_stim.tick_params(colors=text_color)
    ax_br_stim.grid(True, alpha=grid_alpha)
    for spine in ax_br_stim.spines.values():
        spine.set_color(text_color)

    # ===== Bottom Left: Size vs Duration (Unstimulated) =====
    ax_sd_unstim = axes[1, 0]
    ax_sd_unstim.set_facecolor(bg_color)

    unstim_slope = None
    unstim_r2 = None
    if len(unstim_sizes) > 10:
        ax_sd_unstim.loglog(unstim_durations, unstim_sizes, 'o', color=unstim_color,
                            alpha=0.6, markersize=5, label='Avalanches')
        ax_sd_unstim.set_facecolor(bg_color)

        try:
            log_dur = np.log10(unstim_durations)
            log_size = np.log10(unstim_sizes)
            slope, intercept, r_value, _, _ = stats.linregress(log_dur, log_size)
            unstim_slope = slope
            unstim_r2 = r_value**2

            x_fit = np.logspace(np.log10(min(unstim_durations)), np.log10(max(unstim_durations)), 50)
            y_fit = 10**intercept * (x_fit ** slope)
            ax_sd_unstim.loglog(x_fit, y_fit, '--', color='white', linewidth=2,
                                label=f'Fit: σ={slope:.2f}, R²={r_value**2:.2f}')
            ax_sd_unstim.set_facecolor(bg_color)

            # Critical reference line
            y_crit = 10**intercept * (x_fit ** 1.5)
            ax_sd_unstim.loglog(x_fit, y_crit, '-.', color='#2ecc71', linewidth=1.5,
                                alpha=0.7, label='Critical (σ=1.5)')
            ax_sd_unstim.set_facecolor(bg_color)

            # Assessment
            if 1.3 <= slope <= 1.7:
                status = "Near-critical"
                status_color = '#2ecc71'
            elif slope < 1.3:
                status = "Sub-critical"
                status_color = '#3498db'
            else:
                status = "Super-critical"
                status_color = '#e74c3c'

            ax_sd_unstim.text(0.03, 0.03, f"σ = {slope:.2f}\nR² = {r_value**2:.2f}\n{status}",
                              transform=ax_sd_unstim.transAxes, fontsize=11,
                              verticalalignment='bottom', color=status_color,
                              bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
        except Exception as e:
            print(f"Fitting error (unstim): {e}")

        ax_sd_unstim.legend(loc='upper left', fontsize=9)
    else:
        ax_sd_unstim.text(0.5, 0.5, "Insufficient avalanches", transform=ax_sd_unstim.transAxes,
                          fontsize=14, color=text_color, ha='center', va='center')

    ax_sd_unstim.set_xlabel('Avalanche Duration (ms)', color=text_color, fontsize=12)
    ax_sd_unstim.set_ylabel('Avalanche Size', color=text_color, fontsize=12)
    ax_sd_unstim.set_title('Size-Duration Scaling (Unstimulated)', color=unstim_color, fontsize=14, fontweight='bold')
    ax_sd_unstim.tick_params(colors=text_color)
    ax_sd_unstim.grid(True, which='both', alpha=grid_alpha)
    for spine in ax_sd_unstim.spines.values():
        spine.set_color(text_color)

    # ===== Bottom Right: Size vs Duration (Stimulated) =====
    ax_sd_stim = axes[1, 1]
    ax_sd_stim.set_facecolor(bg_color)

    stim_slope = None
    stim_r2 = None
    if len(stim_sizes) > 10:
        ax_sd_stim.loglog(stim_durations, stim_sizes, 'o', color=stim_color,
                          alpha=0.6, markersize=5, label='Avalanches')
        ax_sd_stim.set_facecolor(bg_color)

        try:
            log_dur = np.log10(stim_durations)
            log_size = np.log10(stim_sizes)
            slope, intercept, r_value, _, _ = stats.linregress(log_dur, log_size)
            stim_slope = slope
            stim_r2 = r_value**2

            x_fit = np.logspace(np.log10(min(stim_durations)), np.log10(max(stim_durations)), 50)
            y_fit = 10**intercept * (x_fit ** slope)
            ax_sd_stim.loglog(x_fit, y_fit, '--', color='white', linewidth=2,
                              label=f'Fit: σ={slope:.2f}, R²={r_value**2:.2f}')
            ax_sd_stim.set_facecolor(bg_color)

            # Critical reference line
            y_crit = 10**intercept * (x_fit ** 1.5)
            ax_sd_stim.loglog(x_fit, y_crit, '-.', color='#2ecc71', linewidth=1.5,
                              alpha=0.7, label='Critical (σ=1.5)')
            ax_sd_stim.set_facecolor(bg_color)

            # Assessment
            if 1.3 <= slope <= 1.7:
                status = "Near-critical"
                status_color = '#2ecc71'
            elif slope < 1.3:
                status = "Sub-critical"
                status_color = '#3498db'
            else:
                status = "Super-critical"
                status_color = '#e74c3c'

            ax_sd_stim.text(0.03, 0.03, f"σ = {slope:.2f}\nR² = {r_value**2:.2f}\n{status}",
                            transform=ax_sd_stim.transAxes, fontsize=11,
                            verticalalignment='bottom', color=status_color,
                            bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round'))
        except Exception as e:
            print(f"Fitting error (stim): {e}")

        ax_sd_stim.legend(loc='upper left', fontsize=9)
    else:
        ax_sd_stim.text(0.5, 0.5, "Insufficient avalanches", transform=ax_sd_stim.transAxes,
                        fontsize=14, color=text_color, ha='center', va='center')

    ax_sd_stim.set_xlabel('Avalanche Duration (ms)', color=text_color, fontsize=12)
    ax_sd_stim.set_ylabel('Avalanche Size', color=text_color, fontsize=12)
    ax_sd_stim.set_title('Size-Duration Scaling (During Stimulation)', color=stim_color, fontsize=14, fontweight='bold')
    ax_sd_stim.tick_params(colors=text_color)
    ax_sd_stim.grid(True, which='both', alpha=grid_alpha)
    for spine in ax_sd_stim.spines.values():
        spine.set_color(text_color)

    # Add overall title
    fig.suptitle('Criticality Analysis: Unstimulated vs Stimulated Periods',
                 color=text_color, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
        print(f"Saved criticality comparison to {save_path}")

    # Build analysis results dictionary
    analysis = {
        'unstimulated': {
            'n_avalanches': len(unstim_sizes),
            'branching_ratio_mean': float(np.mean(unstim_branching)) if len(unstim_branching) > 0 else None,
            'branching_ratio_std': float(np.std(unstim_branching)) if len(unstim_branching) > 0 else None,
            'size_duration_exponent': float(unstim_slope) if unstim_slope is not None else None,
            'size_duration_r_squared': float(unstim_r2) if unstim_r2 is not None else None,
            'period_fraction': float(np.sum(unstim_mask) / n_steps)
        },
        'stimulated': {
            'n_avalanches': len(stim_sizes),
            'branching_ratio_mean': float(np.mean(stim_branching)) if len(stim_branching) > 0 else None,
            'branching_ratio_std': float(np.std(stim_branching)) if len(stim_branching) > 0 else None,
            'size_duration_exponent': float(stim_slope) if stim_slope is not None else None,
            'size_duration_r_squared': float(stim_r2) if stim_r2 is not None else None,
            'period_fraction': float(np.sum(stim_mask) / n_steps)
        },
        'buffer_ms': buffer_ms,
        'stim_duration_ms': stim_duration_ms
    }

    # Print summary
    print(f"\n===== Criticality Comparison Summary =====")
    print(f"Unstimulated period ({100*analysis['unstimulated']['period_fraction']:.1f}% of simulation):")
    if analysis['unstimulated']['branching_ratio_mean'] is not None:
        print(f"  Branching ratio: {analysis['unstimulated']['branching_ratio_mean']:.3f} ± {analysis['unstimulated']['branching_ratio_std']:.3f}")
    if analysis['unstimulated']['size_duration_exponent'] is not None:
        print(f"  Size-duration σ: {analysis['unstimulated']['size_duration_exponent']:.3f} (R²={analysis['unstimulated']['size_duration_r_squared']:.3f})")

    print(f"Stimulated period ({100*analysis['stimulated']['period_fraction']:.1f}% of simulation):")
    if analysis['stimulated']['branching_ratio_mean'] is not None:
        print(f"  Branching ratio: {analysis['stimulated']['branching_ratio_mean']:.3f} ± {analysis['stimulated']['branching_ratio_std']:.3f}")
    if analysis['stimulated']['size_duration_exponent'] is not None:
        print(f"  Size-duration σ: {analysis['stimulated']['size_duration_exponent']:.3f} (R²={analysis['stimulated']['size_duration_r_squared']:.3f})")

    return fig, analysis