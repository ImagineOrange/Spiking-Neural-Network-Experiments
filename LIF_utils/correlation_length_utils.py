import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

# Set dark style for plots to match existing visualization style
plt.style.use('dark_background')

def calculate_correlation_length(network, activity_record, dt=0.1, distance_bins=5, plot=True, 
                           time_window=None, use_covariance=False, save_path=None):
    """
    Calculate the correlation length in a neuronal network based on activity records.
    
    Correlation length (ξ) measures how far correlations extend spatially in the system.
    In critical systems, correlation length tends to increase significantly.
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing spatial positions
    activity_record : list
        List of active neuron indices at each time step
    dt : float
        Time step size in ms
    distance_bins : int
        Number of distance bins for averaging correlations
    plot : bool
        Whether to generate visualizations
    save_path : str or None
        Path to save the visualization
        
    Returns:
    --------
    dict
        Dictionary containing correlation length and related metrics
    """
    # 1. Create activity time series for each neuron
    n_steps = len(activity_record)
    n_neurons = network.n_neurons
    
    # Apply time window if specified
    if time_window is not None:
        start_idx, end_idx = time_window
        if start_idx < 0 or end_idx > n_steps or start_idx >= end_idx:
            raise ValueError("Invalid time window")
        activity_record = activity_record[start_idx:end_idx]
        n_steps = len(activity_record)
        print(f"Using time window: {start_idx}-{end_idx} ({n_steps} steps)")
    
    # Initialize activity matrix: rows=time, cols=neurons
    activity_matrix = np.zeros((n_steps, n_neurons))
    
    # Fill the activity matrix
    for t, active_indices in enumerate(activity_record):
        for idx in active_indices:
            activity_matrix[t, idx] = 1.0
    
    # 2. Calculate pairwise correlations between neurons
    if use_covariance:
        # Use covariance instead of correlation for systems with varying activity levels
        correlation_matrix = np.cov(activity_matrix.T)
        # Normalize diagonal for easier comparison
        diag_vals = np.diag(correlation_matrix)
        diag_mean = np.mean(diag_vals[diag_vals > 0])
        if diag_mean > 0:
            correlation_matrix = correlation_matrix / diag_mean
    else:
        # Use Pearson correlation coefficient (normalized covariance)
        correlation_matrix = np.corrcoef(activity_matrix.T)
    
    # 3. Get distances between all neuron pairs
    distances = np.zeros((n_neurons, n_neurons))
    
    # Check if we have spatial positions
    if not hasattr(network, 'neuron_grid_positions') or not network.neuron_grid_positions:
        raise ValueError("Network does not have spatial positions")
    
    # Calculate distances
    for i in range(n_neurons):
        if i not in network.neuron_grid_positions:
            continue
            
        pos_i = network.neuron_grid_positions[i]
        
        for j in range(n_neurons):
            if j not in network.neuron_grid_positions:
                continue
                
            pos_j = network.neuron_grid_positions[j]
            distances[i, j] = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
    
    # 4. Group correlations by distance
    # Find the maximum distance for binning
    max_distance = np.max(distances)
    distance_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_centers = 0.5 * (distance_edges[1:] + distance_edges[:-1])
    
    # Initialize arrays for binned correlations
    mean_correlations = np.zeros(distance_bins)
    std_correlations = np.zeros(distance_bins)
    
    # Fill the bins
    for b in range(distance_bins):
        lower = distance_edges[b]
        upper = distance_edges[b+1]
        
        # Find pairs with distances in this bin
        mask = (distances > lower) & (distances <= upper)
        
        # Don't include self-correlations
        np.fill_diagonal(mask, False)
        
        # Extract correlations for these pairs
        bin_correlations = correlation_matrix[mask]
        
        if len(bin_correlations) > 0:
            mean_correlations[b] = np.mean(bin_correlations)
            std_correlations[b] = np.std(bin_correlations)
        else:
            mean_correlations[b] = np.nan
            std_correlations[b] = np.nan
    
    # 5. Fit exponential decay function to extract correlation length
    # Define the exponential decay function: C(r) = A * exp(-r/ξ) + B
    def exp_decay(r, A, xi, B):
        return A * np.exp(-r/xi) + B
    
    # Filter out NaN values before fitting
    valid_mask = ~np.isnan(mean_correlations)
    valid_distances = distance_centers[valid_mask]
    valid_correlations = mean_correlations[valid_mask]
    
    # Only proceed with fitting if we have enough valid data points
    if sum(valid_mask) >= 3:
        # Initial parameter guesses
        p0 = [0.5, max_distance/4, 0.0]
        
        try:
            # Fit the function to data
            popt, pcov = curve_fit(exp_decay, valid_distances, valid_correlations, p0=p0)
            A_fit, xi_fit, B_fit = popt
            
            # Calculate fitted curve for plotting
            fit_distances = np.linspace(0, max_distance, 100)
            fit_correlations = exp_decay(fit_distances, *popt)
            
            # Calculate R-squared to assess fit quality
            residuals = valid_correlations - exp_decay(valid_distances, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((valid_correlations - np.mean(valid_correlations))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Create visualization if requested
            if plot:
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a1a')
                
                # Plot data points with error bars
                ax.errorbar(distance_centers, mean_correlations, yerr=std_correlations,
                           fmt='o', color='#1dd1a1', ecolor='#a5b1c2', capsize=5,
                           label='Data', zorder=5)
                
                # Plot fit
                ax.plot(fit_distances, fit_correlations, '--', color='#ff6b6b', linewidth=2,
                       label=f'Fit: ξ = {xi_fit:.2f}', zorder=10)
                
                # Add regions
                # Mark correlation length on the plot
                if xi_fit > 0:
                    # Mark the correlation length
                    ax.axvline(x=xi_fit, color='#ff9f43', linestyle='--', alpha=0.7,
                              label=f'Correlation Length: {xi_fit:.2f}')
                    
                    # Highlight the region within one correlation length
                    ax.axvspan(0, xi_fit, alpha=0.2, color='#ff9f43')
                
                # Style the plot
                ax.set_xlabel('Distance (grid units)', color='white', fontsize=14)
                ax.set_ylabel('Correlation', color='white', fontsize=14)
                ax.set_title('Spatial Correlation Function and Correlation Length', 
                            color='white', fontsize=16)
                
                # Add text with fit parameters
                text = f"Correlation Length (ξ) = {xi_fit:.2f}\n"
                text += f"Amplitude (A) = {A_fit:.3f}\n"
                text += f"Offset (B) = {B_fit:.3f}\n"
                text += f"R² = {r_squared:.3f}"
                
                ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=12,
                       va='top', ha='right', color='white',
                       bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
                
                # Style improvements
                ax.set_facecolor('#1a1a1a')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                ax.legend(loc='upper right', framealpha=0.7)
                
                # Save if path provided
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Saved correlation length visualization to {save_path}")
                
                plt.tight_layout()
                
            return {
                "correlation_length": xi_fit,
                "amplitude": A_fit,
                "offset": B_fit,
                "r_squared": r_squared,
                "correlation_matrix": correlation_matrix,
                "distances": distances,
                "mean_correlations": mean_correlations,
                "distance_centers": distance_centers
            }
            
        except Exception as e:
            print(f"Error fitting correlation function: {e}")
            # Return partial results without fit
            return {
                "correlation_length": None,
                "error": str(e),
                "correlation_matrix": correlation_matrix,
                "distances": distances,
                "mean_correlations": mean_correlations,
                "distance_centers": distance_centers
            }
    else:
        print("Insufficient valid data points for fitting")
        return {
            "correlation_length": None,
            "error": "Insufficient valid data points for fitting",
            "correlation_matrix": correlation_matrix,
            "distances": distances,
            "mean_correlations": mean_correlations,
            "distance_centers": distance_centers
        }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def visualize_spatial_correlations(network, activity_record, time_bin_ms=10, dt=0.1, 
                                  distance_bins=10, save_prefix="spatial_correlation_map.png"):
    
    print("Computing spatial correlation data...")
    
    # Convert activity record to time-binned matrix
    n_steps = len(activity_record)
    n_neurons = network.n_neurons
    n_bins = int(n_steps * dt / time_bin_ms)
    bin_size_steps = int(time_bin_ms / dt)
    
    print(f"Time bins: {n_bins}, bin size: {bin_size_steps} steps")
    
    # Create binned activity matrix
    binned_activity = np.zeros((n_bins, n_neurons))
    
    # Fill the binned activity matrix
    for b in range(n_bins):
        start_idx = b * bin_size_steps
        end_idx = min((b + 1) * bin_size_steps, n_steps)
        
        # Count spikes in this bin
        for t in range(start_idx, end_idx):
            if t < len(activity_record):
                for idx in activity_record[t]:
                    binned_activity[b, idx] += 1
    
    print(f"Activity matrix shape: {binned_activity.shape}, non-zero entries: {np.count_nonzero(binned_activity)}")
    
    # Calculate correlation matrix
    # First, identify neurons that were active at least once
    active_mask = np.sum(binned_activity, axis=0) > 0
    active_neurons = np.where(active_mask)[0]
    
    print(f"Active neurons for correlation: {len(active_neurons)} out of {n_neurons}")
    
    if len(active_neurons) < 10:
        print("Too few active neurons for correlation analysis")
        return {"success": False, "error": "Too few active neurons"}

    # Filter out neurons with zero variance (constant activity)
    # These produce NaN in correlation calculations
    activity_subset = binned_activity[:, active_neurons].T
    neuron_std = np.std(activity_subset, axis=1)
    varying_mask = neuron_std > 0

    # Keep only neurons with varying activity
    varying_neurons = active_neurons[varying_mask]

    print(f"Neurons with varying activity: {len(varying_neurons)} out of {len(active_neurons)}")

    if len(varying_neurons) < 10:
        print("Too few varying neurons for correlation analysis")
        return {"success": False, "error": "Too few varying neurons"}

    # Update active neurons to only include those with variance
    active_neurons = varying_neurons

    # Calculate correlation matrix for neurons with varying activity
    with np.errstate(invalid='ignore'):  # Suppress warnings, we've already filtered
        correlation_matrix = np.corrcoef(binned_activity[:, active_neurons].T)

    # Replace any remaining NaNs with 0
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Correlation range: {np.nanmin(correlation_matrix):.4f} to {np.nanmax(correlation_matrix):.4f}")
    
    # Calculate distance matrix for active neurons
    distances = np.zeros((len(active_neurons), len(active_neurons)))
    
    for i, neuron_i in enumerate(active_neurons):
        if neuron_i not in network.neuron_grid_positions:
            continue
        pos_i = network.neuron_grid_positions[neuron_i]
        
        for j, neuron_j in enumerate(active_neurons):
            if j <= i:  # Only calculate upper triangle
                continue
                
            if neuron_j not in network.neuron_grid_positions:
                continue
                
            pos_j = network.neuron_grid_positions[neuron_j]
            dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
            distances[i, j] = distances[j, i] = dist
    
    print(f"Distance range: {np.min(distances[distances > 0]):.4f} to {np.max(distances):.4f}")
    
    # Group correlations by distance
    max_distance = np.max(distances)
    distance_edges = np.linspace(0, max_distance, distance_bins + 1)
    distance_centers = 0.5 * (distance_edges[1:] + distance_edges[:-1])
    
    # Initialize arrays for binned correlations
    mean_correlations = np.zeros(distance_bins)
    std_correlations = np.zeros(distance_bins)
    counts = np.zeros(distance_bins)
    
    # Fill the bins
    for b in range(distance_bins):
        lower = distance_edges[b]
        upper = distance_edges[b+1]
        
        # Find pairs with distances in this bin
        mask = (distances > lower) & (distances <= upper)
        
        # Don't include self-correlations
        np.fill_diagonal(mask, False)
        
        # Extract correlations for these pairs
        bin_correlations = correlation_matrix[mask]
        counts[b] = len(bin_correlations)
        
        if len(bin_correlations) > 0:
            mean_correlations[b] = np.mean(bin_correlations)
            std_correlations[b] = np.std(bin_correlations)
        else:
            mean_correlations[b] = np.nan
            std_correlations[b] = np.nan
    
    correlation_length = None
    
    # 1. FIGURE 1: Correlation vs Distance scatter plot
    fig_scatter = plt.figure(figsize=(12, 7), facecolor='#1a1a1a')
    ax_scatter = fig_scatter.add_subplot(111)
    
    # Create a scatter plot with density coloring
    correlation_values = []
    distance_values = []
    
    for i in range(len(active_neurons)):
        for j in range(i+1, len(active_neurons)):
            if not np.isnan(correlation_matrix[i, j]) and distances[i, j] > 0:
                correlation_values.append(correlation_matrix[i, j])
                distance_values.append(distances[i, j])
    
    # Create a 2D histogram for the scatter density
    h, xedges, yedges = np.histogram2d(distance_values, correlation_values, bins=[20, 20], 
                                      range=[[0, max_distance], [-1.0, 1.0]])
    
    # Plot the scatter with density coloring
    h = h.T  # Transpose for correct orientation
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    
    # Use a viridis-like colormap
    cmap = plt.cm.viridis
    
    # Plot density heatmap
    im = ax_scatter.pcolormesh(X, Y, h, cmap=cmap, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_scatter)
    cbar.set_label('Pair Count', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Add scatter plot with low alpha
    ax_scatter.scatter(distance_values, correlation_values, s=3, color='white', alpha=0.05)
    
    # Style the plot
    ax_scatter.set_xlabel('Distance (grid units)', color='white', fontsize=14)
    ax_scatter.set_ylabel('Correlation', color='white', fontsize=14)
    ax_scatter.set_title('Neuron Pair Correlations vs Distance', color='white', fontsize=16)
    ax_scatter.set_facecolor('#1a1a1a')
    ax_scatter.tick_params(colors='white')
    ax_scatter.grid(True, alpha=0.3)
    
    for spine in ax_scatter.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    
    # Save Figure 1
    scatter_path = f"{save_prefix}_scatter.png"
    fig_scatter.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"Saved correlation scatter plot to {scatter_path}")
    plt.close(fig_scatter)
    
    # 2. FIGURE 2: Mean correlation vs distance with error bars
    fig_mean = plt.figure(figsize=(12, 7), facecolor='#1a1a1a')
    ax_mean = fig_mean.add_subplot(111)
    
    ax_mean.errorbar(distance_centers, mean_correlations, yerr=std_correlations,
                    fmt='o-', color='#1dd1a1', ecolor='#a5b1c2', capsize=5,
                    linewidth=2, markersize=8)
    
    # Fit exponential decay if enough valid points
    valid_mask = ~np.isnan(mean_correlations)
    valid_distances = distance_centers[valid_mask]
    valid_means = mean_correlations[valid_mask]
    
    if np.sum(valid_mask) >= 3:
        # Function to fit: C(r) = A * exp(-r/ξ) + B
        def exp_decay(r, A, xi, B):
            return A * np.exp(-r/xi) + B
        
        try:
            from scipy.optimize import curve_fit
            
            # Initial parameter guesses
            p0 = [0.5, max_distance/4, 0.0]
            
            # Fit the function
            popt, pcov = curve_fit(exp_decay, valid_distances, valid_means, p0=p0)
            A_fit, xi_fit, B_fit = popt
            
            # Generate fitted curve
            fit_distances = np.linspace(0, max_distance, 100)
            fit_correlations = exp_decay(fit_distances, *popt)
            
            # Plot the fit
            ax_mean.plot(fit_distances, fit_correlations, '--', color='#ff6b6b', 
                        linewidth=2, label=f'Fit: ξ = {xi_fit:.2f}')
            
            # Add correlation length marker
            ax_mean.axvline(x=xi_fit, color='#ff9f43', linestyle='--', alpha=0.7)
            ax_mean.axhspan(0, exp_decay(xi_fit, *popt), xmax=xi_fit/max_distance, 
                           alpha=0.2, color='#ff9f43')
            
            # Add text with fit parameters
            text = f"Correlation Length (ξ) = {xi_fit:.2f}\n"
            text += f"Amplitude (A) = {A_fit:.3f}\n"
            text += f"Offset (B) = {B_fit:.3f}"
            
            ax_mean.text(0.97, 0.97, text, transform=ax_mean.transAxes, fontsize=12,
                        va='top', ha='right', color='white',
                        bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
            
            correlation_length = xi_fit
            
            # Add legend
            ax_mean.legend(loc='upper right', framealpha=0.7, fontsize=12)
        except Exception as e:
            print(f"Error fitting correlation function: {e}")
    
    # Style the plot
    ax_mean.set_xlabel('Distance (grid units)', color='white', fontsize=14)
    ax_mean.set_ylabel('Mean Correlation', color='white', fontsize=14)
    ax_mean.set_title('Mean Correlation vs Distance', color='white', fontsize=16)
    ax_mean.set_facecolor('#1a1a1a')
    ax_mean.tick_params(colors='white')
    ax_mean.grid(True, alpha=0.3)
    
    for spine in ax_mean.spines.values():
        spine.set_color('white')
    
    # Add counts info
    bin_info = f"Bin counts: {', '.join([f'{int(c)}' for c in counts])}"
    ax_mean.text(0.5, -0.1, bin_info, transform=ax_mean.transAxes, fontsize=10,
                ha='center', va='center', color='white')
    
    plt.tight_layout()
    
    # Save Figure 2
    mean_path = f"{save_prefix}_mean.png"
    fig_mean.savefig(mean_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"Saved mean correlation plot to {mean_path}")
    plt.close(fig_mean)
    
    # 3. FIGURE 3: Correlation matrix heatmap
    fig_heatmap = plt.figure(figsize=(12, 7), facecolor='#1a1a1a')
    ax_heatmap = fig_heatmap.add_subplot(111)
    
    # Sample a manageable number of neurons for visualization
    max_display = min(500, len(active_neurons))
    if len(active_neurons) > max_display:
        sample_indices = np.random.choice(len(active_neurons), max_display, replace=False)
        sample_indices.sort()  # Keep them in order
        sample_corr = correlation_matrix[np.ix_(sample_indices, sample_indices)]
    else:
        sample_corr = correlation_matrix
    
    # Sort neurons by their average correlation
    if sample_corr.shape[0] > 1:
        mean_corrs = np.mean(sample_corr, axis=1)
        sort_indices = np.argsort(mean_corrs)[::-1]  # Descending order
        sample_corr = sample_corr[np.ix_(sort_indices, sort_indices)]
    
    # Create a custom colormap from blue to white to red
    colors = [(0, 0, 0.8), (1, 1, 1), (0.8, 0, 0)]  # Blue -> White -> Red
    cmap_name = 'correlation_map'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    # Plot the heatmap
    im = ax_heatmap.imshow(sample_corr, cmap=cm, aspect='auto', 
                          vmin=-0.2, vmax=1.0, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Correlation', color='white', fontsize=12)
    cbar.ax.tick_params(colors='white')
    
    # Style the plot
    ax_heatmap.set_title('Correlation Matrix Heatmap (Sample of Neurons)', 
                        color='white', fontsize=16)
    ax_heatmap.set_facecolor('#1a1a1a')
    ax_heatmap.tick_params(colors='white', labelsize=8)
    
    # Optional: create a cleaner look by removing tick labels
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    
    for spine in ax_heatmap.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    
    # Save Figure 3
    heatmap_path = f"{save_prefix}_heatmap.png"
    fig_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"Saved correlation matrix heatmap to {heatmap_path}")
    plt.close(fig_heatmap)
    
    # Return analysis results
    return {
        "success": True,
        "correlation_length": correlation_length,
        "mean_correlations": mean_correlations,
        "distance_centers": distance_centers,
        "active_neuron_count": len(active_neurons),
        "correlation_range": (np.min(correlation_matrix), np.max(correlation_matrix))
    }