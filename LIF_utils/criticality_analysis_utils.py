import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set dark style for plots
plt.style.use('dark_background')
sns.set_style("darkgrid")
# Disable gridlines globally
plt.rc('grid', alpha=0)  # Makes grids fully transparent
# Alternative approach to completely disable them:
plt.rc('axes', grid=False)



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set dark style for plots
plt.style.use('dark_background')
sns.set_style("darkgrid")

def plot_individual_avalanche_statistics(network, save_path_prefix="avalanche", figsize=(10, 8)):
    """
    Create plots showing individual avalanche data points for size and duration distributions
    on log-log scale, along with the branching ratio visualization.
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing avalanche data
    save_path_prefix : str
        Prefix for saving the output files (will append _size.png, _duration.png, etc.)
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    tuple
        (size_fig, duration_fig, branching_fig) - The three figure objects
    """
    if not network.avalanche_sizes:
        print("No avalanches recorded.")
        return None, None, None
    
    # ===== 1. AVALANCHE SIZE DISTRIBUTION WITH INDIVIDUAL POINTS =====
    size_fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax_size = size_fig.add_subplot(111)
    
    # Get unique sizes and count occurrences
    unique_sizes, size_counts = np.unique(network.avalanche_sizes, return_counts=True)
    
    # Normalize counts to get probability
    total_avalanches = len(network.avalanche_sizes)
    size_probabilities = size_counts / total_avalanches
    
    # Plot individual avalanche size data points
    ax_size.loglog(unique_sizes, size_probabilities, 'o', color='#ff7f0e', 
                  markersize=6, alpha=0.7, label="Individual Avalanches")
    # Force background color again after loglog (which sometimes resets it)
    ax_size.set_facecolor('#1a1a1a')
    
    # Fit power law using linear regression in log-log space
    if len(network.avalanche_sizes) > 10:
        try:
            # Convert to log scale for linear regression
            log_sizes = np.log10(unique_sizes)
            log_probs = np.log10(size_probabilities)
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_probs)
            alpha_size = -slope  # Negative slope is the power law exponent
            
            # Generate points for the fitted line
            x_fit = np.logspace(np.log10(min(unique_sizes)), np.log10(max(unique_sizes)), 100)
            log_y_fit = intercept + slope * np.log10(x_fit)
            y_fit = 10**log_y_fit
            
            # Plot the fitted power law
            ax_size.loglog(x_fit, y_fit, '--', color='white', alpha=0.7, linewidth=2.5,
                         label=f"Power Law Fit: α = {alpha_size:.2f}, R² = {r_value**2:.2f}")
            
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')
            # Add reference line with ideal slope
            ideal_alpha = 1.5
            # Keep same intercept as our fitted line
            ideal_log_y = intercept - ideal_alpha * np.log10(x_fit)
            ideal_y = 10**ideal_log_y
            
            ax_size.loglog(x_fit, ideal_y, '-.', color='#1dd1a1', alpha=0.7, linewidth=2,
                         label=f"Ideal Power Law: α = {ideal_alpha}")
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')    
            
        except Exception as e:
            print(f"Power law fitting error for sizes: {e}")
    
    # Style and label the plot
    ax_size.set_xlabel("Avalanche Size", color='white', fontsize=14)
    ax_size.set_ylabel("Probability", color='white', fontsize=14)
    ax_size.set_title("Avalanche Size Distribution (Individual Data Points)", color='white', fontsize=16)
    ax_size.legend(loc='best', framealpha=0.7, fontsize=12)
    
    # Add grid and improve appearance
    ax_size.grid(True, which="both", ls="-", alpha=0.2)
    ax_size.tick_params(colors='white', labelsize=12)
    for spine in ax_size.spines.values():
        spine.set_color('white')
    
    # Add information text about criticality
    if len(network.avalanche_sizes) > 10:
        text = f"Critical systems show power law exponent α ≈ 1.5\n"
        text += f"Current exponent: α = {alpha_size:.2f} (R² = {r_value**2:.2f})\n"
        text += f"Number of avalanches: {len(network.avalanche_sizes)}"
        
        # Color-code based on how close to ideal exponent
        text_color = '#1dd1a1' if 1.3 <= alpha_size <= 1.7 else 'white'
        ax_size.text(0.03, 0.03, text, transform=ax_size.transAxes, fontsize=11,
                    verticalalignment='bottom', horizontalalignment='left',
                    color=text_color, bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    if save_path_prefix:
        size_path = f"{save_path_prefix}_size_individual.png"
        size_fig.savefig(size_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved individual avalanche size plot to {size_path}")
    
    # ===== 2. AVALANCHE DURATION DISTRIBUTION WITH INDIVIDUAL POINTS =====
    dur_fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax_dur = dur_fig.add_subplot(111)
    
    # Get unique durations and count occurrences
    unique_durations, dur_counts = np.unique(network.avalanche_durations, return_counts=True)
    
    # Normalize counts to get probability
    dur_probabilities = dur_counts / total_avalanches
    
    # Plot individual avalanche duration data points
    ax_dur.loglog(unique_durations, dur_probabilities, 'o', color='#1f77b4', 
                markersize=6, alpha=0.7, label="Individual Avalanches")
    # Force background color again after loglog (which sometimes resets it)
    ax_size.set_facecolor('#1a1a1a')
    
    # Fit power law using linear regression in log-log space
    if len(network.avalanche_durations) > 10:
        try:
            # Convert to log scale for linear regression
            log_durations = np.log10(unique_durations)
            log_probs = np.log10(dur_probabilities)
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_durations, log_probs)
            alpha_dur = -slope  # Negative slope is the power law exponent
            
            # Generate points for the fitted line
            x_fit = np.logspace(np.log10(min(unique_durations)), np.log10(max(unique_durations)), 100)
            log_y_fit = intercept + slope * np.log10(x_fit)
            y_fit = 10**log_y_fit
            
            # Plot the fitted power law
            ax_dur.loglog(x_fit, y_fit, '--', color='white', alpha=0.7, linewidth=2.5,
                        label=f"Power Law Fit: α = {alpha_dur:.2f}, R² = {r_value**2:.2f}")
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')
            
            # Add reference line with ideal slope
            ideal_alpha = 2.0
            # Keep same intercept as our fitted line
            ideal_log_y = intercept - ideal_alpha * np.log10(x_fit)
            ideal_y = 10**ideal_log_y
            
            ax_dur.loglog(x_fit, ideal_y, '-.', color='#1dd1a1', alpha=0.7, linewidth=2,
                        label=f"Ideal Power Law: α = {ideal_alpha}")
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')
            
        except Exception as e:
            print(f"Power law fitting error for durations: {e}")
    
    # Style and label the plot
    ax_dur.set_xlabel("Avalanche Duration (ms)", color='white', fontsize=14)
    ax_dur.set_ylabel("Probability", color='white', fontsize=14)
    ax_dur.set_title("Avalanche Duration Distribution (Individual Data Points)", color='white', fontsize=16)
    ax_dur.legend(loc='best', framealpha=0.7, fontsize=12)
    
    # Add grid and improve appearance
    ax_dur.grid(True, which="both", ls="-", alpha=0.2)
    ax_dur.tick_params(colors='white', labelsize=12)
    for spine in ax_dur.spines.values():
        spine.set_color('white')
    
    
    
    # Add information text about criticality
    if len(network.avalanche_durations) > 10:
        text = f"Critical systems show power law exponent α ≈ 2.0\n"
        text += f"Current exponent: α = {alpha_dur:.2f} (R² = {r_value**2:.2f})\n"
        text += f"Number of avalanches: {len(network.avalanche_durations)}"
        
        # Color-code based on how close to ideal exponent
        text_color = '#1dd1a1' if 1.8 <= alpha_dur <= 2.2 else 'white'
        ax_dur.text(0.03, 0.03, text, transform=ax_dur.transAxes, fontsize=11,
                   verticalalignment='bottom', horizontalalignment='left',
                   color=text_color, bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    if save_path_prefix:
        dur_path = f"{save_path_prefix}_duration_individual.png"
        dur_fig.savefig(dur_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved individual avalanche duration plot to {dur_path}")
    
    # ===== 3. SCATTER PLOT OF SIZE VS DURATION =====
    scatter_fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax_scatter = scatter_fig.add_subplot(111)
    
    # Plot scatter of size vs duration
    ax_scatter.loglog(network.avalanche_durations, network.avalanche_sizes, 'o', 
                     color='#9b59b6', alpha=0.6, markersize=6, label="Individual Avalanches")
    # Force background color again after loglog (which sometimes resets it)
    ax_size.set_facecolor('#1a1a1a')
    
    # Force background color again after loglog (which sometimes resets it)
    ax_size.set_facecolor('#1a1a1a')
    
    # Try to fit a power law relationship
    if len(network.avalanche_sizes) > 10:
        try:
            # Convert to log space for linear fitting
            log_durations = np.log10(network.avalanche_durations)
            log_sizes = np.log10(network.avalanche_sizes)
            
            # Linear regression in log-log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_durations, log_sizes)
            
            # Generate points for the fitted line
            x_fit = np.logspace(np.log10(min(network.avalanche_durations)), 
                              np.log10(max(network.avalanche_durations)), 100)
            y_fit = 10**(intercept) * (x_fit ** slope)
            
            # Plot the fitted relationship
            ax_scatter.loglog(x_fit, y_fit, '--', color='white', alpha=0.7, linewidth=2.5,
                           label=f"Fit: Size ~ Duration^{slope:.2f}, R²={r_value**2:.2f}")
        
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')
            
            # Mark the critical theoretical relationship (slope = 1.5)
            critical_y = 10**(intercept) * (x_fit ** 1.5)
            ax_scatter.loglog(x_fit, critical_y, '-.', color='#1dd1a1', alpha=0.7, linewidth=2,
                           label=f"Critical Theory: Size ~ Duration^1.5")
            # Force background color again after loglog (which sometimes resets it)
            ax_size.set_facecolor('#1a1a1a')        

        except Exception as e:
            print(f"Regression error for size vs duration: {e}")
    
    # Style and label the plot
    ax_scatter.set_xlabel("Avalanche Duration (ms)", color='white', fontsize=14)
    ax_scatter.set_ylabel("Avalanche Size", color='white', fontsize=14)
    ax_scatter.set_title("Avalanche Size vs Duration", color='white', fontsize=16)
    ax_scatter.legend(loc='best', framealpha=0.7, fontsize=12)
    
    # Add grid and improve appearance
    ax_scatter.grid(True, which="both", ls="-", alpha=0.2)
    ax_scatter.tick_params(colors='white', labelsize=12)
    for spine in ax_scatter.spines.values():
        spine.set_color('white')
    
    # Add information text about criticality
    if len(network.avalanche_sizes) > 10:
        text = f"Critical theory predicts: Size ~ Duration^1.5\n"
        text += f"Current exponent: {slope:.2f}\n"
        text += f"Correlation: R²={r_value**2:.2f}"
        
        # Color-code based on how close to ideal exponent
        text_color = '#1dd1a1' if 1.3 <= slope <= 1.7 else 'white'
        ax_scatter.text(0.03, 0.03, text, transform=ax_scatter.transAxes, fontsize=11,
                      verticalalignment='bottom', horizontalalignment='left',
                      color=text_color, bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
    
    plt.tight_layout()
    if save_path_prefix:
        scatter_path = f"{save_path_prefix}_size_vs_duration.png"
        scatter_fig.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved size vs duration scatter plot to {scatter_path}")
    
    # ===== 4. BRANCHING PARAMETER VISUALIZATION (KEEPING THE ORIGINAL) =====
    branching_fig = plt.figure(figsize=figsize, facecolor='#1a1a1a')
    ax_branch = branching_fig.add_subplot(111)
    
    # Calculate branching ratio (number of descendants per ancestor)
    activity = np.array(network.network_activity)
    non_zero_idx = np.where(activity > 0)[0]
    
    # We need to ensure we have enough data and no division by zero
    if len(non_zero_idx) > 1 and np.max(non_zero_idx) < len(activity) - 1:
        next_idx = non_zero_idx + 1
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
            ax_branch.set_xlabel('Branching Ratio (Descendants/Ancestors)', color='white', fontsize=14)
            ax_branch.set_ylabel('Probability Density', color='white', fontsize=14)
            ax_branch.set_title('Branching Parameter Distribution', color='white', fontsize=16)
            
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
                         color=text_color, bbox=dict(facecolor='#222222', alpha=0.7, boxstyle='round'))
        else:
            ax_branch.text(0.5, 0.5, "Insufficient data for branching analysis", 
                         transform=ax_branch.transAxes, fontsize=14, color='white',
                         ha='center', va='center')
    else:
        ax_branch.text(0.5, 0.5, "Insufficient activity data for branching analysis", 
                     transform=ax_branch.transAxes, fontsize=14, color='white',
                     ha='center', va='center')
    
    # Add legend
    ax_branch.legend(loc='upper right', framealpha=0.7, fontsize=12)
    
    # Style improvements
    ax_branch.set_facecolor('#1a1a1a')
    ax_branch.tick_params(colors='white', labelsize=12)
    for spine in ax_branch.spines.values():
        spine.set_color('white')
    ax_branch.grid(True, which='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if save_path_prefix:
        branch_path = f"{save_path_prefix}_branching.png"
        branching_fig.savefig(branch_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved branching parameter plot to {branch_path}")
    
    return size_fig, dur_fig, scatter_fig, branching_fig


# Function to use in the main code to replace the original visualization
def plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche"):
    """
    Enhanced version of criticality analysis that shows individual data points
    and additional visualizations.
    
    This function can be used as a replacement for plot_simplified_avalanche_statistics.
    
    Parameters:
    -----------
    network : ExtendedNeuronalNetworkWithReversal
        The neural network object containing avalanche data
    save_path_prefix : str
        Prefix for saving the output files
        
    Returns:
    --------
    dict
        Dictionary with the generated figure objects and analysis results
    """
    # Check if we have avalanche data
    if not network.avalanche_sizes or len(network.avalanche_sizes) < 5:
        print("Insufficient avalanche data for analysis.")
        return {"success": False, "message": "Insufficient avalanche data"}
    
    # Create all the visualizations
    size_fig, dur_fig, scatter_fig, branching_fig = plot_individual_avalanche_statistics(
        network, save_path_prefix=save_path_prefix
    )
    
    # Run basic analysis to return metrics
    try:
        # Get unique sizes and their probabilities
        unique_sizes, size_counts = np.unique(network.avalanche_sizes, return_counts=True)
        size_probabilities = size_counts / len(network.avalanche_sizes)
        
        # Get unique durations and their probabilities
        unique_durations, dur_counts = np.unique(network.avalanche_durations, return_counts=True)
        dur_probabilities = dur_counts / len(network.avalanche_durations)
        
        # Fit power law using linear regression in log-log space for sizes
        log_sizes = np.log10(unique_sizes)
        log_size_probs = np.log10(size_probabilities)
        size_slope, size_intercept, size_r_value, _, _ = stats.linregress(log_sizes, log_size_probs)
        alpha_size = -size_slope  # Negative slope is the power law exponent
        
        # Fit power law using linear regression in log-log space for durations
        log_durations = np.log10(unique_durations)
        log_dur_probs = np.log10(dur_probabilities)
        dur_slope, dur_intercept, dur_r_value, _, _ = stats.linregress(log_durations, log_dur_probs)
        alpha_duration = -dur_slope  # Negative slope is the power law exponent
        
        # Calculate correlation between log size and log duration for individual avalanches
        log_avalanche_sizes = np.log10(network.avalanche_sizes)
        log_avalanche_durations = np.log10(network.avalanche_durations)
        r_value = np.corrcoef(log_avalanche_sizes, log_avalanche_durations)[0, 1]
        
        # Calculate scaling exponent (size vs duration)
        slope, intercept, scaling_r_value, _, _ = stats.linregress(log_avalanche_durations, log_avalanche_sizes)
        
        # Calculate branching parameter
        activity = np.array(network.network_activity)
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
        
        # Compile analysis results
        analysis = {
            "avalanche_count": len(network.avalanche_sizes),
            "size_exponent": float(alpha_size),
            "size_r_squared": float(size_r_value**2),
            "duration_exponent": float(alpha_duration),
            "duration_r_squared": float(dur_r_value**2),
            "size_duration_correlation": float(r_value),
            "size_duration_scaling": float(slope),
            "size_duration_r_squared": float(scaling_r_value**2),
            "branching_ratio": branching_ratio
        }
        
        # Determine criticality based on standard criteria
        is_critical_size = 1.3 <= alpha_size <= 1.7
        is_critical_duration = 1.8 <= alpha_duration <= 2.2
        is_critical_branching = branching_ratio is not None and 0.95 <= branching_ratio <= 1.05
        is_critical_scaling = 1.3 <= slope <= 1.7
        
        # Overall assessment
        critical_count = sum([is_critical_size, is_critical_duration, 
                             is_critical_branching, is_critical_scaling])
        
        if critical_count >= 3:
            assessment = "Strongly critical"
            is_critical = True
        elif critical_count >= 2:
            assessment = "Moderately critical"
            is_critical = True
        elif critical_count >= 1:
            assessment = "Weakly critical / near-critical"
            is_critical = False
        else:
            assessment = "Not critical"
            is_critical = False
            
        analysis["assessment"] = assessment
        analysis["is_critical"] = is_critical
        
        print(f"\n===== Enhanced Criticality Analysis =====")
        print(f"Analyzed {analysis['avalanche_count']} avalanches")
        print(f"Size exponent: α = {analysis['size_exponent']:.3f} (R² = {analysis['size_r_squared']:.3f}, ideal: ~1.5)")
        print(f"Duration exponent: α = {analysis['duration_exponent']:.3f} (R² = {analysis['duration_r_squared']:.3f}, ideal: ~2.0)")
        print(f"Size-duration scaling: {analysis['size_duration_scaling']:.3f} (R² = {analysis['size_duration_r_squared']:.3f}, ideal: ~1.5)")
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
            "size": size_fig,
            "duration": dur_fig,
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
    
    # Define ideal values and weights
    ideal_values = {
        "size_exponent": 1.5,       # Ideal: ~1.5
        "duration_exponent": 2.0,    # Ideal: ~2.0
        "branching_ratio": 1.0,      # Ideal: ~1.0
        "scaling_relation": 1.5      # Ideal: ~1.5
    }
    
    weights = {
        "size_exponent": 0.3,        # 30% of score
        "duration_exponent": 0.3,     # 30% of score
        "branching_ratio": 0.2,       # 20% of score
        "scaling_relation": 0.2       # 20% of score
    }
    
    score = 0.0
    weight_sum = 0.0
    
    # Prioritize binning method but use powerlaw fit if binning not available
    method_to_use = "binning" if results["methods"].get("binning") else "powerlaw"
    
    if results["methods"].get(method_to_use):
        # Size exponent score
        if "size_exponent" in results["methods"][method_to_use]:
            size_exponent = results["methods"][method_to_use]["size_exponent"]
            # Score decreases with distance from ideal, using inverse exponential
            size_score = np.exp(-2.0 * abs(size_exponent - ideal_values["size_exponent"]))
            score += weights["size_exponent"] * size_score
            weight_sum += weights["size_exponent"]
            
            results["size_exponent_used"] = size_exponent
        
        # Duration exponent score
        if "duration_exponent" in results["methods"][method_to_use]:
            dur_exponent = results["methods"][method_to_use]["duration_exponent"]
            dur_score = np.exp(-2.0 * abs(dur_exponent - ideal_values["duration_exponent"]))
            score += weights["duration_exponent"] * dur_score
            weight_sum += weights["duration_exponent"]
            
            results["duration_exponent_used"] = dur_exponent
    
    # Branching ratio score
    if results["branching_ratio"] is not None:
        branching_ratio = results["branching_ratio"]
        # Score based on how close branching ratio is to 1.0
        br_score = 1.0 - min(1.0, abs(branching_ratio - ideal_values["branching_ratio"]))
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
    
    if method_to_use == "binning":
        method_name = "log-binning + regression"
    else:
        method_name = "direct powerlaw fit"
        
    print(f"\nExponents (using {method_name}):")
    if "size_exponent_used" in results:
        print(f"  Size exponent: α = {results['size_exponent_used']:.3f} (ideal: ~1.5)")
    if "duration_exponent_used" in results:
        print(f"  Duration exponent: α = {results['duration_exponent_used']:.3f} (ideal: ~2.0)")
    
    if results["branching_ratio"] is not None:
        print(f"Branching ratio: {results['branching_ratio']:.3f} (ideal: ~1.0)")
    
    if results["scaling_relation"] is not None:
        print(f"Size-duration scaling: {results['scaling_relation']:.3f} (ideal: ~1.5)")
    
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
    Backward compatibility function that calls the simplified plot function.
    """
    print("Using simplified avalanche statistics visualization...")
    size_fig, dur_fig, branching_fig = plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche")
    return size_fig  # Return the size figure for backward compatibility