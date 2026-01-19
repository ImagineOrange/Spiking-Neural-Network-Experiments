import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Default: do NOT set dark style globally - let functions handle it based on darkstyle parameter

def plot_individual_avalanche_statistics(network, save_path_prefix="avalanche", figsize=(10, 8), darkstyle=True):
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

    Returns:
    --------
    tuple
        (scatter_fig, branching_fig) - The figure objects
    """
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

    if not network.avalanche_sizes:
        print("No avalanches recorded.")
        return None, None

    # ===== 1. SCATTER PLOT OF SIZE VS DURATION =====
    scatter_fig = plt.figure(figsize=figsize, facecolor=bg_color)
    ax_scatter = scatter_fig.add_subplot(111)
    
    # Plot scatter of size vs duration
    ax_scatter.loglog(network.avalanche_durations, network.avalanche_sizes, 'o',
                     color='#9b59b6', alpha=0.6, markersize=6, label="Individual Avalanches")
    # Force background color again after loglog (which sometimes resets it)
    ax_scatter.set_facecolor(bg_color)
    
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
    if len(network.avalanche_sizes) > 10:
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
    activity = np.array(network.network_activity)
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
def plot_enhanced_criticality_analysis(network, save_path_prefix="avalanche", darkstyle=True):
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

    Returns:
    --------
    dict
        Dictionary with the generated figure objects and analysis results
    """
    # Check if we have avalanche data
    if not network.avalanche_sizes or len(network.avalanche_sizes) < 5:
        print("Insufficient avalanche data for analysis.")
        return {"success": False, "message": "Insufficient avalanche data"}

    # Create visualizations (size vs duration scatter and branching ratio)
    scatter_fig, branching_fig = plot_individual_avalanche_statistics(
        network, save_path_prefix=save_path_prefix, darkstyle=darkstyle
    )

    # Run basic analysis to return metrics
    try:
        # Calculate scaling exponent (size vs duration) - this is the σ exponent
        log_avalanche_sizes = np.log10(network.avalanche_sizes)
        log_avalanche_durations = np.log10(network.avalanche_durations)
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