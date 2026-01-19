#!/usr/bin/env python3
"""
Criticality Grid Search for Circular Network Experiment

This script performs a grid search over network-level parameters to find
configurations that maximize criticality, measured by 5 metrics:
1. Branching Ratio (ideal: 1.0)
2. Size-Duration Scaling Exponent (ideal: 1.5)
3. Correlation Length (ideal: large relative to system size)
4. Dynamic Range (ideal: maximized)
5. Mutual Information (ideal: maximized)

Usage:
    python criticality_grid_search.py [options]

Options:
    --workers, -w     Number of parallel workers (default: cpu_count - 1)
    --samples, -n     Number of LHS samples (default: 500)
    --mode, -m        Search mode: 'lhs' or 'full' (default: lhs)
    --output, -o      Output directory (default: grid_search_results)
    --resume, -r      Resume from previous run
    --n-neurons       Number of neurons (default: 6000)
    --duration        Simulation duration in ms (default: 1500)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import warnings

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import custom modules
from LIF_objects.CircularNeuronalNetwork import CircularNeuronalNetwork
from LIF_utils.simulation_utils import run_unified_simulation
from LIF_utils.criticality_analysis_utils import analyze_criticality_comprehensively
from LIF_utils.correlation_length_utils import calculate_correlation_length

# Suppress warnings during parallel execution
warnings.filterwarnings('ignore')


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def compute_dynamic_range(network, activity_record=None, dt=0.1):
    """
    Estimate dynamic range from activity variance.

    Simplified approach for grid search:
    - Use coefficient of variation (CV) of network activity as a proxy
    - Critical networks have high CV (variable responses to similar inputs)
    - Also consider the range of responses

    Args:
        network: CircularNeuronalNetwork with network_activity populated
        activity_record: Not used directly, kept for API consistency
        dt: Timestep in ms

    Returns:
        float: Dynamic range proxy metric (higher = better)
    """
    activity = np.array(network.network_activity)

    if len(activity) == 0:
        return 0.0

    # Compute response characteristics
    mean_activity = np.mean(activity)
    std_activity = np.std(activity)
    max_activity = np.max(activity)

    if mean_activity == 0:
        return 0.0

    # Coefficient of variation as proxy for dynamic range
    # Critical networks have high CV (variable responses)
    cv = std_activity / mean_activity

    # Also consider the range of responses
    response_range = max_activity / max(1, mean_activity)

    # Combined metric (will be normalized across trials)
    dynamic_range = cv * np.log10(1 + response_range)

    return float(dynamic_range)


def _pairwise_mutual_info(x, y):
    """
    Compute mutual information between two binary sequences.

    Args:
        x, y: Binary arrays (0 or 1)

    Returns:
        float: Mutual information in bits
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Joint distribution
    p00 = np.sum((x == 0) & (y == 0)) / n
    p01 = np.sum((x == 0) & (y == 1)) / n
    p10 = np.sum((x == 1) & (y == 0)) / n
    p11 = np.sum((x == 1) & (y == 1)) / n

    # Marginals
    px0 = np.sum(x == 0) / n
    px1 = np.sum(x == 1) / n
    py0 = np.sum(y == 0) / n
    py1 = np.sum(y == 1) / n

    # MI calculation with safe log
    mi = 0.0
    for pxy, px, py in [(p00, px0, py0), (p01, px0, py1),
                         (p10, px1, py0), (p11, px1, py1)]:
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * np.log2(pxy / (px * py))

    return mi


def compute_mutual_information(activity_record, n_neurons, n_samples=100, bin_size_steps=100):
    """
    Compute average mutual information between sampled neuron pairs.

    Args:
        activity_record: List of active neuron indices per timestep
        n_neurons: Total number of neurons in the network
        n_samples: Number of neuron pairs to sample
        bin_size_steps: Number of timesteps per bin for discretization

    Returns:
        float or None: Average MI in bits, or None if insufficient data
    """
    n_steps = len(activity_record)
    n_bins = n_steps // bin_size_steps

    if n_bins < 5:
        return None  # Not enough data for reliable MI estimation

    # Create binned activity matrix (binary: did neuron spike in this bin?)
    activity_matrix = np.zeros((n_bins, n_neurons), dtype=np.int8)
    for b in range(n_bins):
        start = b * bin_size_steps
        end = (b + 1) * bin_size_steps
        for t in range(start, min(end, n_steps)):
            if t < len(activity_record):
                for neuron_idx in activity_record[t]:
                    if neuron_idx < n_neurons:
                        activity_matrix[b, neuron_idx] = 1

    # Find neurons that were active at least once
    active_neurons = np.where(activity_matrix.sum(axis=0) > 0)[0]
    if len(active_neurons) < 10:
        return None  # Need enough active neurons

    # Sample random neuron pairs
    pairs = []
    for _ in range(n_samples):
        i, j = np.random.choice(active_neurons, 2, replace=False)
        pairs.append((i, j))

    # Compute MI for each pair
    mi_values = []
    for i, j in pairs:
        mi = _pairwise_mutual_info(activity_matrix[:, i], activity_matrix[:, j])
        mi_values.append(mi)

    return float(np.mean(mi_values))


def compute_criticality_score(metrics, max_dynamic_range=None, max_mutual_info=None):
    """
    Compute combined criticality score from all 5 metrics.

    Args:
        metrics: dict with raw metric values
        max_dynamic_range: Maximum dynamic range observed (for normalization)
        max_mutual_info: Maximum mutual info observed (for normalization)

    Returns:
        tuple: (total_score, individual_scores_dict)
    """
    scores = {}
    weights = {
        'branching': 0.20,
        'scaling': 0.20,
        'correlation_length': 0.20,
        'dynamic_range': 0.20,
        'mutual_info': 0.20
    }

    # 1. Branching ratio score (ideal = 1.0)
    if metrics.get('branching_ratio') is not None:
        scores['branching'] = np.exp(-2.0 * abs(metrics['branching_ratio'] - 1.0))
    else:
        scores['branching'] = 0.0

    # 2. Scaling exponent score (ideal = 1.5)
    if metrics.get('scaling_exponent') is not None:
        scores['scaling'] = np.exp(-2.0 * abs(metrics['scaling_exponent'] - 1.5))
    else:
        scores['scaling'] = 0.0

    # 3. Correlation length score (higher = better, normalized by system size)
    if metrics.get('correlation_length') is not None and metrics.get('max_distance') is not None:
        max_dist = metrics['max_distance']
        if max_dist > 0:
            scores['correlation_length'] = min(1.0, metrics['correlation_length'] / max_dist)
        else:
            scores['correlation_length'] = 0.0
    else:
        scores['correlation_length'] = 0.0

    # 4. Dynamic range score (normalized to max observed)
    if metrics.get('dynamic_range') is not None and max_dynamic_range is not None and max_dynamic_range > 0:
        scores['dynamic_range'] = min(1.0, metrics['dynamic_range'] / max_dynamic_range)
    else:
        scores['dynamic_range'] = 0.0

    # 5. Mutual information score (normalized to max observed)
    if metrics.get('mutual_info') is not None and max_mutual_info is not None and max_mutual_info > 0:
        scores['mutual_info'] = min(1.0, metrics['mutual_info'] / max_mutual_info)
    else:
        scores['mutual_info'] = 0.0

    # Weighted sum
    total_score = sum(weights[k] * scores[k] for k in weights)

    # Bonus for having many avalanches (statistical confidence)
    avalanche_count = metrics.get('avalanche_count', 0)
    avalanche_bonus = min(0.1, avalanche_count / 1000)
    total_score = min(1.0, total_score * (1.0 + avalanche_bonus))

    return float(total_score), scores


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_single_config(args):
    """
    Evaluate one parameter configuration, computing all 5 metrics.

    Args:
        args: tuple of (config, fixed_params, trial_idx)

    Returns:
        dict: Results including all metrics and config
    """
    config, fixed_params, trial_idx = args

    # Set unique random seed for this trial
    seed = fixed_params.get('base_seed', 42) + trial_idx
    np.random.seed(seed)

    try:
        # Merge parameters
        all_params = {**fixed_params, **config}

        # Create network
        network = CircularNeuronalNetwork(
            n_neurons=all_params['n_neurons'],
            connection_p=config['connection_p'],
            weight_scale=config['weight_scale'],
            spatial=True,
            transmission_delay=config['transmission_delay'],
            inhibitory_fraction=config['inhibitory_fraction'],
            layout=all_params['layout'],
            v_noise_amp=config['v_noise_amp'],
            i_noise_amp=config['i_noise_amp'],
            distance_lambda=config['distance_lambda']
            # e_reversal and i_reversal use LIFNeuron defaults
        )

        # Run single simulation
        activity_record, _, _ = run_unified_simulation(
            network,
            duration=all_params['duration'],
            dt=all_params['dt'],
            stim_interval=all_params.get('stim_interval'),
            stim_interval_strength=all_params.get('stim_interval_strength', 50),
            stim_fraction=all_params.get('stim_fraction', 0.01),
            track_neurons=None,  # Don't track for efficiency
            stochastic_stim=all_params.get('stochastic_stim', False),
            no_stimulation=all_params.get('no_stimulation', True)
        )

        # 1. Branching ratio + Scaling exponent (existing functions)
        crit_results = analyze_criticality_comprehensively(network, save_plots=False, min_avalanches=10)
        branching_ratio = crit_results.get('branching_ratio')
        scaling_exponent = crit_results.get('scaling_relation')
        avalanche_count = crit_results.get('avalanche_count', 0)

        # 2. Correlation length (existing function)
        try:
            corr_results = calculate_correlation_length(
                network,
                activity_record,
                dt=all_params['dt'],
                distance_bins=10,
                plot=False
            )
            correlation_length = corr_results.get('correlation_length')
        except Exception:
            correlation_length = None

        # Calculate max distance for normalization
        max_distance = np.sqrt(2) * network.side_length

        # 3. Dynamic range (new function)
        dynamic_range = compute_dynamic_range(network, activity_record)

        # 4. Mutual information (new function)
        mutual_info = compute_mutual_information(
            activity_record,
            network.n_neurons,
            n_samples=100,
            bin_size_steps=100
        )

        return {
            'config': config,
            'trial_idx': trial_idx,
            'seed': seed,
            'success': True,
            'branching_ratio': branching_ratio,
            'scaling_exponent': scaling_exponent,
            'correlation_length': correlation_length,
            'max_distance': max_distance,
            'dynamic_range': dynamic_range,
            'mutual_info': mutual_info,
            'avalanche_count': avalanche_count,
            # Flatten config for CSV
            **{f'param_{k}': v for k, v in config.items()}
        }

    except Exception as e:
        return {
            'config': config,
            'trial_idx': trial_idx,
            'seed': seed,
            'success': False,
            'error': str(e),
            'branching_ratio': None,
            'scaling_exponent': None,
            'correlation_length': None,
            'max_distance': None,
            'dynamic_range': None,
            'mutual_info': None,
            'avalanche_count': 0,
            **{f'param_{k}': v for k, v in config.items()}
        }


# =============================================================================
# SEARCH CONFIGURATION GENERATION
# =============================================================================

def generate_full_grid(param_grid):
    """
    Generate all combinations of parameters (full grid search).

    Args:
        param_grid: dict mapping parameter names to lists of values

    Returns:
        list of dicts, each representing one configuration
    """
    import itertools

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configurations = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        configurations.append(config)

    return configurations


def generate_latin_hypercube(param_grid, n_samples=500, seed=42):
    """
    Generate parameter combinations using Latin Hypercube Sampling.

    Provides better coverage of parameter space than random sampling.

    Args:
        param_grid: dict mapping parameter names to lists of values
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        list of dicts, each representing one configuration
    """
    try:
        from scipy.stats import qmc
        use_scipy = True
    except ImportError:
        use_scipy = False

    np.random.seed(seed)
    param_names = list(param_grid.keys())
    n_params = len(param_names)

    if use_scipy:
        # Use scipy's LHS implementation
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        samples = sampler.random(n=n_samples)
    else:
        # Simple LHS fallback
        samples = np.zeros((n_samples, n_params))
        for i in range(n_params):
            perm = np.random.permutation(n_samples)
            samples[:, i] = (perm + np.random.random(n_samples)) / n_samples

    # Map [0,1] samples to actual parameter values
    configurations = []
    for sample in samples:
        config = {}
        for i, name in enumerate(param_names):
            values = param_grid[name]
            # Map [0,1] to index in values list
            idx = int(sample[i] * len(values))
            idx = min(idx, len(values) - 1)  # Ensure valid index
            config[name] = values[idx]
        configurations.append(config)

    return configurations


def generate_search_configurations(param_grid, mode='lhs', n_samples=500, seed=42):
    """
    Generate parameter configurations for grid search.

    Args:
        param_grid: dict mapping parameter names to lists of values
        mode: 'full' for full grid, 'lhs' for Latin Hypercube Sampling
        n_samples: Number of samples for LHS mode
        seed: Random seed

    Returns:
        list of configuration dicts
    """
    if mode == 'full':
        configs = generate_full_grid(param_grid)
        print(f"Generated {len(configs)} configurations (full grid)")
    elif mode == 'lhs':
        configs = generate_latin_hypercube(param_grid, n_samples, seed)
        print(f"Generated {len(configs)} configurations (Latin Hypercube Sampling)")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return configs


# =============================================================================
# RESULTS MANAGEMENT
# =============================================================================

class ResultsManager:
    """
    Manages saving/loading results with resume capability.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, 'grid_search_results.csv')
        self.scored_csv_path = os.path.join(output_dir, 'grid_search_results_scored.csv')
        self.best_json_path = os.path.join(output_dir, 'best_params.json')

        os.makedirs(output_dir, exist_ok=True)

        self.results = []
        self.best_score = 0.0
        self.best_params = None

    def load_existing(self):
        """Load existing results for resume capability."""
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.results = df.to_dict('records')
                print(f"Loaded {len(self.results)} existing results")

                if self.results and 'critical_score' in df.columns:
                    best_idx = df['critical_score'].idxmax()
                    self.best_score = df.loc[best_idx, 'critical_score']

                return self._get_completed_configs()
            except Exception as e:
                print(f"Error loading existing results: {e}")
                return set()
        return set()

    def _get_completed_configs(self):
        """Get set of completed configuration hashes."""
        completed = set()
        for result in self.results:
            config = result.get('config', {})
            if isinstance(config, str):
                try:
                    config = json.loads(config.replace("'", '"'))
                except:
                    continue
            config_hash = self._config_hash(config)
            completed.add(config_hash)
        return completed

    def _config_hash(self, config):
        """Create hashable key for configuration."""
        return tuple(sorted((k, v) for k, v in config.items()))

    def save_result(self, result):
        """Incrementally save a single result."""
        self.results.append(result)

        # Save to CSV
        df = pd.DataFrame(self.results)
        df.to_csv(self.csv_path, index=False)

    def save_scored_results(self, max_dynamic_range, max_mutual_info):
        """
        Compute final scores and save scored results.
        """
        scored_results = []
        best_score = 0
        best_result = None

        for result in self.results:
            if not result.get('success', False):
                continue

            metrics = {
                'branching_ratio': result.get('branching_ratio'),
                'scaling_exponent': result.get('scaling_exponent'),
                'correlation_length': result.get('correlation_length'),
                'max_distance': result.get('max_distance'),
                'dynamic_range': result.get('dynamic_range'),
                'mutual_info': result.get('mutual_info'),
                'avalanche_count': result.get('avalanche_count', 0)
            }

            total_score, individual_scores = compute_criticality_score(
                metrics, max_dynamic_range, max_mutual_info
            )

            scored_result = {
                **result,
                'critical_score': total_score,
                'score_branching': individual_scores['branching'],
                'score_scaling': individual_scores['scaling'],
                'score_correlation_length': individual_scores['correlation_length'],
                'score_dynamic_range': individual_scores['dynamic_range'],
                'score_mutual_info': individual_scores['mutual_info']
            }
            scored_results.append(scored_result)

            if total_score > best_score:
                best_score = total_score
                best_result = scored_result

        # Save scored results
        df = pd.DataFrame(scored_results)
        df.to_csv(self.scored_csv_path, index=False)
        print(f"Saved scored results to {self.scored_csv_path}")

        # Save best params
        if best_result:
            self.best_score = best_score
            self.best_params = best_result.get('config', {})
            self._save_best_params(best_result)

        return scored_results

    def _save_best_params(self, best_result):
        """Save best parameters to JSON."""
        config = best_result.get('config', {})
        if isinstance(config, str):
            try:
                config = json.loads(config.replace("'", '"'))
            except:
                config = {}

        with open(self.best_json_path, 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_params': config,
                'metrics': {
                    'branching_ratio': best_result.get('branching_ratio'),
                    'scaling_exponent': best_result.get('scaling_exponent'),
                    'correlation_length': best_result.get('correlation_length'),
                    'dynamic_range': best_result.get('dynamic_range'),
                    'mutual_info': best_result.get('mutual_info'),
                    'avalanche_count': best_result.get('avalanche_count')
                },
                'individual_scores': {
                    'branching': best_result.get('score_branching'),
                    'scaling': best_result.get('score_scaling'),
                    'correlation_length': best_result.get('score_correlation_length'),
                    'dynamic_range': best_result.get('score_dynamic_range'),
                    'mutual_info': best_result.get('score_mutual_info')
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        print(f"Saved best params to {self.best_json_path}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def update_progress_plots(results, output_dir, darkstyle=True):
    """
    Create progress visualization plots.

    Args:
        results: List of scored result dicts
        output_dir: Directory to save plots
        darkstyle: Use dark background style
    """
    if len(results) < 2:
        return

    # Filter successful results
    successful = [r for r in results if r.get('success', False) and r.get('critical_score') is not None]
    if len(successful) < 2:
        return

    # Set colors based on style
    if darkstyle:
        bg_color = '#1a1a1a'
        text_color = 'white'
        plt.style.use('dark_background')
    else:
        bg_color = 'white'
        text_color = 'black'

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), facecolor=bg_color)

    # 1. Combined score over iterations
    ax1 = axes[0, 0]
    scores = [r['critical_score'] for r in successful]
    ax1.plot(range(len(scores)), scores, 'o-', alpha=0.6, color='#ff7f0e', markersize=4)
    ax1.set_xlabel('Iteration', color=text_color)
    ax1.set_ylabel('Criticality Score', color=text_color)
    ax1.set_title('Combined Score Over Iterations', color=text_color)

    # Highlight best
    best_idx = np.argmax(scores)
    ax1.scatter([best_idx], [scores[best_idx]], color='#1dd1a1', s=100, zorder=5,
                label=f'Best: {scores[best_idx]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(bg_color)

    # 2. Individual metric scores distribution
    ax2 = axes[0, 1]
    metric_names = ['branching', 'scaling', 'correlation_length', 'dynamic_range', 'mutual_info']
    metric_scores = {name: [] for name in metric_names}
    for r in successful:
        for name in metric_names:
            score = r.get(f'score_{name}')
            if score is not None:
                metric_scores[name].append(score)

    positions = range(len(metric_names))
    bp = ax2.boxplot([metric_scores[name] for name in metric_names], positions=positions, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Branch', 'Scale', 'Corr.Len', 'Dyn.Rng', 'MI'], color=text_color)
    ax2.set_ylabel('Score', color=text_color)
    ax2.set_title('Individual Metric Scores', color=text_color)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(bg_color)

    # 3. Branching ratio distribution
    ax3 = axes[0, 2]
    branching_ratios = [r['branching_ratio'] for r in successful if r.get('branching_ratio') is not None]
    if branching_ratios:
        ax3.hist(branching_ratios, bins=20, alpha=0.7, color='#9b59b6')
        ax3.axvline(x=1.0, color='#1dd1a1', linestyle='--', linewidth=2, label='Critical (1.0)')
        ax3.set_xlabel('Branching Ratio', color=text_color)
        ax3.set_ylabel('Count', color=text_color)
        ax3.set_title('Branching Ratio Distribution', color=text_color)
        ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor(bg_color)

    # 4. Scaling exponent distribution
    ax4 = axes[1, 0]
    scaling_exps = [r['scaling_exponent'] for r in successful if r.get('scaling_exponent') is not None]
    if scaling_exps:
        ax4.hist(scaling_exps, bins=20, alpha=0.7, color='#e74c3c')
        ax4.axvline(x=1.5, color='#1dd1a1', linestyle='--', linewidth=2, label='Critical (1.5)')
        ax4.set_xlabel('Scaling Exponent', color=text_color)
        ax4.set_ylabel('Count', color=text_color)
        ax4.set_title('Size-Duration Scaling Distribution', color=text_color)
        ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor(bg_color)

    # 5. Correlation length distribution
    ax5 = axes[1, 1]
    corr_lengths = [r['correlation_length'] for r in successful if r.get('correlation_length') is not None]
    if corr_lengths:
        ax5.hist(corr_lengths, bins=20, alpha=0.7, color='#2ecc71')
        ax5.set_xlabel('Correlation Length', color=text_color)
        ax5.set_ylabel('Count', color=text_color)
        ax5.set_title('Correlation Length Distribution', color=text_color)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor(bg_color)

    # 6. Parameter importance (correlation with score)
    ax6 = axes[1, 2]
    param_cols = [col for col in successful[0].keys() if col.startswith('param_')]
    correlations = []

    df = pd.DataFrame(successful)
    for col in param_cols:
        if col in df.columns:
            try:
                corr = df[col].astype(float).corr(df['critical_score'])
                if not pd.isna(corr):
                    param_name = col.replace('param_', '')
                    correlations.append((param_name, corr))
            except:
                pass

    if correlations:
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        params = [c[0] for c in correlations]
        corrs = [c[1] for c in correlations]

        colors = ['#1dd1a1' if c > 0 else '#ff6b6b' for c in corrs]
        ax6.barh(params, corrs, color=colors)
        ax6.axvline(x=0, color=text_color, linestyle='-', alpha=0.5)
        ax6.set_xlabel('Correlation with Score', color=text_color)
        ax6.set_title('Parameter Importance', color=text_color)
    ax6.set_facecolor(bg_color)

    # Style all axes
    for ax in axes.flat:
        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(text_color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'progress_plot.png'), dpi=150, facecolor=bg_color,
                bbox_inches='tight')
    plt.close()
    print(f"Saved progress plot to {os.path.join(output_dir, 'progress_plot.png')}")


# =============================================================================
# PARALLEL SEARCH
# =============================================================================

def run_parallel_search(configurations, fixed_params, n_workers=None, results_manager=None,
                        completed_configs=None, output_dir=None, checkpoint_interval=20):
    """
    Execute grid search in parallel with progress tracking.

    Args:
        configurations: List of parameter configuration dicts
        fixed_params: Dict of fixed parameters
        n_workers: Number of parallel workers (default: cpu_count - 1)
        results_manager: ResultsManager instance for saving
        completed_configs: Set of already completed config hashes to skip
        output_dir: Directory for saving plots (uses results_manager.output_dir if None)
        checkpoint_interval: Number of trials between renormalization checkpoints (default: 20)

    Returns:
        list of result dicts
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if completed_configs is None:
        completed_configs = set()

    if output_dir is None and results_manager is not None:
        output_dir = results_manager.output_dir

    # Filter out already completed configurations
    remaining_configs = []
    for idx, config in enumerate(configurations):
        config_hash = tuple(sorted((k, v) for k, v in config.items()))
        if config_hash not in completed_configs:
            remaining_configs.append((config, fixed_params, idx))

    print(f"Running {len(remaining_configs)} configurations with {n_workers} workers")
    print(f"Skipping {len(configurations) - len(remaining_configs)} already completed")
    print(f"Checkpoint interval: every {checkpoint_interval} trials")

    if not remaining_configs:
        return results_manager.results if results_manager else []

    # Use spawn context for macOS compatibility
    ctx = mp.get_context('spawn')

    results = []
    trials_since_checkpoint = 0
    last_best_score = 0.0

    with ctx.Pool(processes=n_workers) as pool:
        # Use imap_unordered for better load balancing with progress bar
        for result in tqdm(pool.imap_unordered(evaluate_single_config, remaining_configs),
                          total=len(remaining_configs), desc="Grid Search"):
            results.append(result)

            # Incremental save
            if results_manager:
                results_manager.save_result(result)

            trials_since_checkpoint += 1

            # Print progress
            if result.get('success'):
                br = result.get('branching_ratio')
                se = result.get('scaling_exponent')
                br_str = f"{br:.3f}" if br is not None else "N/A"
                se_str = f"{se:.3f}" if se is not None else "N/A"
                tqdm.write(f"  Trial {result['trial_idx']}: BR={br_str}, Scale={se_str}")
            else:
                tqdm.write(f"  Trial {result['trial_idx']}: FAILED - {result.get('error', 'Unknown error')}")

            # Checkpoint: renormalize and update best every N trials
            if trials_since_checkpoint >= checkpoint_interval and results_manager:
                trials_since_checkpoint = 0

                # Compute current normalization factors from ALL results
                successful = [r for r in results_manager.results if r.get('success', False)]
                if successful:
                    max_dr = max((r.get('dynamic_range') or 0) for r in successful)
                    max_mi = max((r.get('mutual_info') or 0) for r in successful)

                    # Rescore all results and save
                    scored = results_manager.save_scored_results(max_dr, max_mi)

                    # Check if best changed (could be older result with new norms)
                    new_best = results_manager.best_score
                    if new_best != last_best_score:
                        tqdm.write(f"\n  [CHECKPOINT] New best score: {new_best:.4f} (was {last_best_score:.4f})")
                        last_best_score = new_best

                    # Update progress plots
                    if output_dir:
                        update_progress_plots(scored, output_dir)
                        tqdm.write(f"  [CHECKPOINT] Updated plots and best_params.json")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Grid Search for Criticality in Circular Networks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    parser.add_argument('--samples', '-n', type=int, default=500,
                        help='Number of samples for LHS mode (default: 500)')
    parser.add_argument('--mode', '-m', choices=['full', 'lhs'], default='lhs',
                        help='Search mode: lhs or full (default: lhs)')
    parser.add_argument('--output', '-o', type=str, default='grid_search_results',
                        help='Output directory (default: grid_search_results)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from previous run')
    parser.add_argument('--n-neurons', type=int, default=6000,
                        help='Number of neurons (default: 6000)')
    parser.add_argument('--duration', type=float, default=1500.0,
                        help='Simulation duration in ms (default: 1500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--checkpoint', '-c', type=int, default=20,
                        help='Checkpoint interval for renormalization (default: 20)')

    args = parser.parse_args()

    # Define parameter grid
    param_grid = {
        'connection_p': [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        'weight_scale': [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
        'inhibitory_fraction': [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        'transmission_delay': [0.5, 1.0, 2.0, 3.0],
        'distance_lambda': [0.05, 0.10, 0.20, 0.50],
        'v_noise_amp': [0.05, 0.1, 0.2, 0.3],
        'i_noise_amp': [0.01, 0.02, 0.05]
    }

    # Fixed parameters (use LIFNeuron defaults except noise)
    fixed_params = {
        'n_neurons': args.n_neurons,
        'duration': args.duration,
        'dt': 0.1,
        'layout': 'circle',
        'no_stimulation': True,
        'stochastic_stim': False,
        'base_seed': args.seed
    }

    # Calculate total combinations
    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    print(f"Parameter space: {total_combos} total combinations")

    # Initialize results manager
    results_manager = ResultsManager(args.output)

    # Load existing results if resuming
    completed_configs = set()
    if args.resume:
        completed_configs = results_manager.load_existing()

    # Generate configurations
    configurations = generate_search_configurations(
        param_grid,
        mode=args.mode,
        n_samples=args.samples,
        seed=args.seed
    )

    # Run parallel search
    print(f"\nStarting grid search...")
    print(f"  Mode: {args.mode}")
    print(f"  Samples: {len(configurations)}")
    print(f"  Workers: {args.workers or (os.cpu_count() - 1)}")
    print(f"  Neurons: {args.n_neurons}")
    print(f"  Duration: {args.duration} ms")
    print(f"  Checkpoint interval: {args.checkpoint}")
    print()

    results = run_parallel_search(
        configurations,
        fixed_params,
        n_workers=args.workers,
        results_manager=results_manager,
        completed_configs=completed_configs,
        output_dir=args.output,
        checkpoint_interval=args.checkpoint
    )

    # Compute normalization factors
    successful_results = [r for r in results_manager.results if r.get('success', False)]
    if successful_results:
        max_dynamic_range = max(
            (r.get('dynamic_range') or 0) for r in successful_results
        )
        max_mutual_info = max(
            (r.get('mutual_info') or 0) for r in successful_results
        )

        # Compute final scores and save
        scored_results = results_manager.save_scored_results(max_dynamic_range, max_mutual_info)

        # Generate visualizations
        update_progress_plots(scored_results, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("GRID SEARCH COMPLETE")
        print("=" * 60)
        print(f"Total evaluations: {len(results_manager.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Best criticality score: {results_manager.best_score:.4f}")
        print(f"\nBest parameters:")
        for k, v in (results_manager.best_params or {}).items():
            print(f"  {k}: {v}")
        print(f"\nResults saved to: {args.output}/")
    else:
        print("\nNo successful evaluations!")

    return results_manager.results


if __name__ == "__main__":
    main()
