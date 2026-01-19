#!/usr/bin/env python
"""
Progress Tracker for Grid Search Experiments

Analyzes completed experiments and generates visualizations showing:
- Accuracy per configuration
- Average spikes per example
- Active neurons per example
- Dead neuron counts
- Neuron assignment distribution across classes

Usage:
    python progress_tracker.py                          # Auto-detect latest grid search
    python progress_tracker.py --dir grid_search_XXX    # Specify directory
    python progress_tracker.py --watch                  # Continuous monitoring mode
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob
import time
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mnist_intensity_stats():
    """Load MNIST data and compute per-class intensity statistics."""
    mnist_path = os.path.join(SCRIPT_DIR, 'mnist_data', 'training.pickle')

    if not os.path.exists(mnist_path):
        return None

    with open(mnist_path, 'rb') as f:
        data = pickle.load(f)

    images = data['x'] / 255.0  # Normalize to 0-1
    labels = data['y'].flatten()

    stats = {}
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            class_images = images[mask]
            stats[c] = {
                'mean_intensity': float(class_images.mean()),
                'total_intensity': float(class_images.sum(axis=(1, 2)).mean()),
                'active_pixels': float((class_images > 0.1).sum(axis=(1, 2)).mean()),
                'count': int(mask.sum())
            }

    return stats


def find_latest_grid_search():
    """Find the most recent grid search directory."""
    pattern = os.path.join(SCRIPT_DIR, 'grid_search_*')
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def analyze_experiment(exp_dir):
    """
    Analyze a single experiment directory.

    Returns dict with metrics or None if not complete/analyzable.
    """
    activity_path = os.path.join(exp_dir, 'activity')
    weights_path = os.path.join(exp_dir, 'weights')
    params_path = os.path.join(exp_dir, 'params.json')

    # Check if experiment has completed training
    weight_files = glob.glob(os.path.join(weights_path, '*.npy')) if os.path.exists(weights_path) else []
    is_complete = len(weight_files) > 2

    # Load parameters
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        params = {}

    # Try to load activity data
    pop_vecs_file = os.path.join(activity_path, 'resultPopVecs2500_clean.npy')
    labels_file = os.path.join(activity_path, 'inputNumbers2500_clean.npy')

    result = {
        'exp_name': os.path.basename(exp_dir),
        'exp_dir': exp_dir,
        'is_complete': is_complete,
        'params': params,
        'pConn_ei_input': params.get('pConn_ei_input'),
        'pConn_ei': params.get('pConn_ei'),
        'pConn_ie': params.get('pConn_ie'),
        'weight_ie': params.get('weight_ie'),
    }

    if not os.path.exists(pop_vecs_file) or not os.path.exists(labels_file):
        # Check training progress from log
        log_path = os.path.join(exp_dir, 'training.log')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                # Find last progress percentage
                import re
                matches = re.findall(r'(\d+)%', content)
                if matches:
                    result['training_progress'] = int(matches[-1])
            except:
                pass
        return result

    # Load and analyze activity data
    pop_vecs = np.load(pop_vecs_file)  # Shape: (n_examples, n_neurons)
    labels = np.load(labels_file)

    n_examples, n_neurons = pop_vecs.shape
    n_classes = len(np.unique(labels))

    # Basic spike statistics
    total_spikes = pop_vecs.sum(axis=1)
    active_per_example = (pop_vecs > 0).sum(axis=1)

    result['avg_spikes_per_example'] = float(total_spikes.mean())
    result['std_spikes_per_example'] = float(total_spikes.std())
    result['avg_active_neurons'] = float(active_per_example.mean())
    result['std_active_neurons'] = float(active_per_example.std())

    # Dead neuron analysis
    neuron_total_spikes = pop_vecs.sum(axis=0)
    result['dead_neurons'] = int((neuron_total_spikes == 0).sum())
    result['total_neurons'] = n_neurons

    # Compute neuron assignments
    class_responses = np.zeros((n_neurons, n_classes))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_responses[:, c] = pop_vecs[mask].mean(axis=0)

    assignments = np.argmax(class_responses, axis=1)
    # Handle neurons that never fire - assign to -1
    never_fired = neuron_total_spikes == 0
    assignments[never_fired] = -1

    result['assignment_distribution'] = [int((assignments == c).sum()) for c in range(n_classes)]
    result['unassigned_neurons'] = int((assignments == -1).sum())

    # Compute accuracy using population voting
    predictions = []
    for i in range(len(labels)):
        class_votes = np.zeros(n_classes)
        for c in range(n_classes):
            class_mask = assignments == c
            if class_mask.sum() > 0:
                class_votes[c] = pop_vecs[i, class_mask].sum()
        predictions.append(np.argmax(class_votes))

    predictions = np.array(predictions)
    result['accuracy'] = float((predictions == labels).mean() * 100)

    # Per-class accuracy
    result['per_class_accuracy'] = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            result['per_class_accuracy'][int(c)] = float((predictions[mask] == c).mean() * 100)

    # Per-class spike statistics
    result['spikes_per_class'] = {}
    result['avg_spikes_per_class'] = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            class_spikes = pop_vecs[mask].sum(axis=1)
            result['spikes_per_class'][int(c)] = float(class_spikes.sum())
            result['avg_spikes_per_class'][int(c)] = float(class_spikes.mean())

    return result


def analyze_all_experiments(grid_dir):
    """Analyze all experiments in a grid search directory."""
    exp_dirs = sorted(glob.glob(os.path.join(grid_dir, 'exp_*')))

    results = []
    for exp_dir in exp_dirs:
        result = analyze_experiment(exp_dir)
        if result:
            results.append(result)

    return results


def print_summary(results):
    """Print a text summary of results."""
    complete = [r for r in results if r.get('accuracy') is not None]
    in_progress = [r for r in results if r.get('accuracy') is None and r.get('training_progress')]
    pending = [r for r in results if r.get('accuracy') is None and not r.get('training_progress')]

    print("\n" + "=" * 80)
    print("GRID SEARCH PROGRESS SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(f"  Complete:    {len(complete)}")
    print(f"  In progress: {len(in_progress)}")
    print(f"  Pending:     {len(pending)}")
    print()

    if complete:
        print("-" * 80)
        print("COMPLETED EXPERIMENTS (sorted by accuracy)")
        print("-" * 80)
        print(f"{'Experiment':<35} {'Acc%':<8} {'Spikes':<8} {'Active':<8} {'Dead':<6} {'Assignment Dist'}")
        print("-" * 80)

        for r in sorted(complete, key=lambda x: x['accuracy'], reverse=True):
            dist_str = str(r.get('assignment_distribution', []))
            print(f"{r['exp_name']:<35} "
                  f"{r['accuracy']:<8.1f} "
                  f"{r['avg_spikes_per_example']:<8.1f} "
                  f"{r['avg_active_neurons']:<8.1f} "
                  f"{r['dead_neurons']:<6} "
                  f"{dist_str}")

    if in_progress:
        print("\n" + "-" * 80)
        print("IN PROGRESS")
        print("-" * 80)
        for r in sorted(in_progress, key=lambda x: x.get('training_progress', 0), reverse=True):
            print(f"  {r['exp_name']}: {r.get('training_progress', '?')}%")

    print("=" * 80)


def plot_results(results, output_dir=None, show=True):
    """Generate visualization plots for completed experiments."""
    complete = [r for r in results if r.get('accuracy') is not None]

    if not complete:
        print("No completed experiments to plot.")
        return

    # Sort by experiment number for consistent ordering
    complete = sorted(complete, key=lambda x: x['exp_name'])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Grid Search Progress - {len(complete)} Completed Experiments', fontsize=14, fontweight='bold')

    exp_names = [r['exp_name'].replace('exp_', '').split('_')[0] for r in complete]
    x_pos = np.arange(len(complete))

    # 1. Accuracy bar chart
    ax1 = axes[0, 0]
    accuracies = [r['accuracy'] for r in complete]
    colors = ['green' if a > 25 else 'orange' if a > 21 else 'red' for a in accuracies]
    ax1.bar(x_pos, accuracies, color=colors, alpha=0.7)
    ax1.axhline(y=20, color='red', linestyle='--', label='Chance (20%)')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Classification Accuracy')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, max(accuracies) * 1.1 if max(accuracies) > 25 else 30)

    # 2. Average spikes per example
    ax2 = axes[0, 1]
    spikes = [r['avg_spikes_per_example'] for r in complete]
    ax2.bar(x_pos, spikes, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Avg Spikes')
    ax2.set_title('Average Spikes per Example')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)

    # 3. Active neurons per example
    ax3 = axes[0, 2]
    active = [r['avg_active_neurons'] for r in complete]
    ax3.bar(x_pos, active, color='darkorange', alpha=0.7)
    ax3.axhline(y=complete[0]['total_neurons'], color='gray', linestyle='--',
                label=f'Total neurons ({complete[0]["total_neurons"]})')
    ax3.set_xlabel('Experiment')
    ax3.set_ylabel('Avg Active Neurons')
    ax3.set_title('Average Active Neurons per Example')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax3.legend()

    # 4. Dead neurons
    ax4 = axes[1, 0]
    dead = [r['dead_neurons'] for r in complete]
    ax4.bar(x_pos, dead, color='darkred', alpha=0.7)
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('Dead Neurons')
    ax4.set_title('Dead Neurons (never fired)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)

    # 5. Assignment distribution (stacked bar)
    ax5 = axes[1, 1]
    n_classes = len(complete[0]['assignment_distribution'])
    class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    bottom = np.zeros(len(complete))
    for c in range(n_classes):
        counts = [r['assignment_distribution'][c] for r in complete]
        ax5.bar(x_pos, counts, bottom=bottom, label=f'Class {c}', color=class_colors[c], alpha=0.8)
        bottom += counts

    ax5.set_xlabel('Experiment')
    ax5.set_ylabel('Neuron Count')
    ax5.set_title('Neuron Assignment Distribution')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=8)
    ax5.legend(loc='upper right', fontsize=8)

    # 6. Accuracy vs Parameters scatter
    ax6 = axes[1, 2]
    pConn_ei = [r['pConn_ei'] * 100 for r in complete]
    pConn_ie = [r['pConn_ie'] * 100 for r in complete]

    scatter = ax6.scatter(pConn_ei, pConn_ie, c=accuracies, cmap='RdYlGn',
                          s=100, alpha=0.7, edgecolors='black')
    ax6.set_xlabel('E→I Connectivity (%)')
    ax6.set_ylabel('I→E Connectivity (%)')
    ax6.set_title('Accuracy vs Connectivity Parameters')
    plt.colorbar(scatter, ax=ax6, label='Accuracy (%)')

    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'progress_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_assignment_heatmap(results, output_dir=None, show=True):
    """Plot a heatmap showing assignment distribution across experiments."""
    complete = [r for r in results if r.get('accuracy') is not None]

    if not complete:
        return

    complete = sorted(complete, key=lambda x: x['exp_name'])

    n_experiments = len(complete)
    n_classes = len(complete[0]['assignment_distribution'])

    # Create assignment matrix
    assignment_matrix = np.array([r['assignment_distribution'] for r in complete])

    fig, ax = plt.subplots(figsize=(10, max(6, n_experiments * 0.4)))

    im = ax.imshow(assignment_matrix, aspect='auto', cmap='Blues')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax.set_yticks(range(n_experiments))
    ax.set_yticklabels([r['exp_name'] for r in complete], fontsize=8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Experiment')
    ax.set_title('Neuron Assignment Distribution Heatmap')

    # Add text annotations
    for i in range(n_experiments):
        for j in range(n_classes):
            val = assignment_matrix[i, j]
            color = 'white' if val > assignment_matrix.max() / 2 else 'black'
            ax.text(j, i, str(int(val)), ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Neuron Count')
    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'assignments_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Assignment heatmap saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_intensity_spike_analysis(results, output_dir=None, show=True):
    """
    Plot analysis of input intensity vs output spikes per digit class.

    Generates 3 figures:
    1. Spikes per digit class (bar chart)
    2. MNIST input intensity per digit class (bar chart)
    3. Correlation between input intensity and output spikes
    """
    complete = [r for r in results if r.get('avg_spikes_per_class') is not None]

    if not complete:
        print("No completed experiments with per-class spike data.")
        return

    # Load MNIST intensity stats
    mnist_stats = load_mnist_intensity_stats()
    if mnist_stats is None:
        print("Could not load MNIST data for intensity analysis.")
        return

    # Get the classes used (assume first experiment is representative)
    classes = sorted([int(c) for c in complete[0]['avg_spikes_per_class'].keys()])
    n_classes = len(classes)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Input Intensity vs Output Spikes Analysis', fontsize=14, fontweight='bold')

    class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    # =========================================================================
    # Figure 1: Average spikes per digit class (averaged across experiments)
    # =========================================================================
    ax1 = axes[0]

    # Compute mean spikes per class across all experiments
    mean_spikes_per_class = []
    std_spikes_per_class = []
    for c in classes:
        spikes = [r['avg_spikes_per_class'].get(c, 0) for r in complete]
        mean_spikes_per_class.append(np.mean(spikes))
        std_spikes_per_class.append(np.std(spikes))

    x_pos = np.arange(n_classes)
    bars1 = ax1.bar(x_pos, mean_spikes_per_class, yerr=std_spikes_per_class,
                    color=class_colors, alpha=0.8, capsize=3, edgecolor='black')

    ax1.set_xlabel('Digit Class', fontsize=11)
    ax1.set_ylabel('Avg Spikes per Example', fontsize=11)
    ax1.set_title('Output Spikes by Digit Class\n(averaged across experiments)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(c) for c in classes])

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, mean_spikes_per_class)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_spikes_per_class[i] + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # Figure 2: MNIST input intensity per digit class
    # =========================================================================
    ax2 = axes[1]

    intensities = [mnist_stats[c]['total_intensity'] for c in classes]

    bars2 = ax2.bar(x_pos, intensities, color=class_colors, alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Digit Class', fontsize=11)
    ax2.set_ylabel('Total Pixel Intensity (normalized)', fontsize=11)
    ax2.set_title('MNIST Input Intensity by Digit Class\n(determines Poisson spike rate)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(c) for c in classes])

    # Add value labels on bars
    for bar, val in zip(bars2, intensities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # Figure 3: Correlation between input intensity and output spikes
    # =========================================================================
    ax3 = axes[2]

    # Scatter plot with regression line
    ax3.scatter(intensities, mean_spikes_per_class, c=class_colors, s=150,
                edgecolors='black', linewidths=1.5, zorder=5)

    # Add digit labels to each point
    for i, c in enumerate(classes):
        ax3.annotate(str(c), (intensities[i], mean_spikes_per_class[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=12, fontweight='bold')

    # Compute and plot regression line
    z = np.polyfit(intensities, mean_spikes_per_class, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(intensities) * 0.9, max(intensities) * 1.1, 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Linear fit')

    # Compute correlation coefficient
    corr = np.corrcoef(intensities, mean_spikes_per_class)[0, 1]

    ax3.set_xlabel('MNIST Input Intensity', fontsize=11)
    ax3.set_ylabel('Avg Output Spikes', fontsize=11)
    ax3.set_title(f'Input Intensity vs Output Spikes\nCorrelation: r = {corr:.3f}', fontsize=12)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Add annotation about the correlation
    if corr > 0.7:
        interpretation = "Strong positive correlation:\nHigh-intensity digits dominate network response"
    elif corr > 0.4:
        interpretation = "Moderate correlation:\nInput intensity partially drives output"
    else:
        interpretation = "Weak correlation:\nNetwork has learned input-independent representations"

    ax3.text(0.95, 0.05, interpretation, transform=ax3.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'intensity_spike_analysis_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Intensity-spike analysis saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_experiment_spike_breakdown(results, output_dir=None, show=True):
    """
    Plot spikes per class for each individual experiment as a grouped bar chart.
    """
    complete = [r for r in results if r.get('avg_spikes_per_class') is not None]

    if not complete:
        return

    complete = sorted(complete, key=lambda x: x['exp_name'])

    classes = sorted([int(c) for c in complete[0]['avg_spikes_per_class'].keys()])
    n_classes = len(classes)
    n_experiments = len(complete)

    fig, ax = plt.subplots(figsize=(max(12, n_experiments * 0.8), 6))

    x = np.arange(n_experiments)
    width = 0.8 / n_classes
    class_colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, c in enumerate(classes):
        spikes = [r['avg_spikes_per_class'].get(c, 0) for r in complete]
        offset = (i - n_classes / 2 + 0.5) * width
        ax.bar(x + offset, spikes, width, label=f'Digit {c}', color=class_colors[i], alpha=0.8)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Avg Spikes per Example')
    ax.set_title('Spikes per Digit Class by Experiment')
    ax.set_xticks(x)
    exp_labels = [r['exp_name'].replace('exp_', '').replace('_', '\n', 1).split('\n')[0]
                  for r in complete]
    ax.set_xticklabels(exp_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(title='Digit', loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'spikes_by_experiment_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Per-experiment spike breakdown saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_class_response_heatmap(results, output_dir=None, show=True):
    """
    Plot a heatmap showing average spikes per class across all experiments.
    """
    complete = [r for r in results if r.get('avg_spikes_per_class') is not None]

    if not complete:
        return

    complete = sorted(complete, key=lambda x: x['exp_name'])

    classes = sorted([int(c) for c in complete[0]['avg_spikes_per_class'].keys()])
    n_classes = len(classes)
    n_experiments = len(complete)

    # Create spike matrix: experiments x classes
    spike_matrix = np.zeros((n_experiments, n_classes))
    for i, r in enumerate(complete):
        for j, c in enumerate(classes):
            spike_matrix[i, j] = r['avg_spikes_per_class'].get(c, 0)

    fig, ax = plt.subplots(figsize=(10, max(6, n_experiments * 0.4)))

    im = ax.imshow(spike_matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f'Digit {c}' for c in classes])
    ax.set_yticks(range(n_experiments))
    ax.set_yticklabels([r['exp_name'] for r in complete], fontsize=8)

    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Experiment')
    ax.set_title('Average Spikes per Digit Class Heatmap\n(Yellow=Low, Red=High)')

    # Add text annotations
    for i in range(n_experiments):
        for j in range(n_classes):
            val = spike_matrix[i, j]
            color = 'white' if val > spike_matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Avg Spikes per Example')
    plt.tight_layout()

    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'class_response_heatmap_{timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Class response heatmap saved to: {plot_path}")

    if show:
        plt.show()
    else:
        plt.close()


def watch_mode(grid_dir, interval=60):
    """Continuously monitor and update results."""
    print(f"Watching {grid_dir} for updates (Ctrl+C to stop)")
    print(f"Update interval: {interval} seconds")

    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            results = analyze_all_experiments(grid_dir)
            print_summary(results)

            complete = [r for r in results if r.get('accuracy') is not None]
            if complete:
                plot_results(results, output_dir=grid_dir, show=False)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description='Track and visualize grid search progress',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dir', '-d', type=str, default=None,
                        help='Grid search directory (default: auto-detect latest)')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Continuous monitoring mode')
    parser.add_argument('--interval', '-i', type=int, default=60,
                        help='Update interval in seconds for watch mode')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save plots to grid search directory')

    args = parser.parse_args()

    # Find grid search directory
    if args.dir:
        grid_dir = args.dir
        if not os.path.isabs(grid_dir):
            grid_dir = os.path.join(SCRIPT_DIR, grid_dir)
    else:
        grid_dir = find_latest_grid_search()

    if not grid_dir or not os.path.exists(grid_dir):
        print("Error: No grid search directory found.")
        print("Use --dir to specify a directory.")
        return 1

    print(f"Analyzing: {grid_dir}")

    if args.watch:
        watch_mode(grid_dir, args.interval)
    else:
        results = analyze_all_experiments(grid_dir)
        print_summary(results)

        if not args.no_plot:
            output_dir = grid_dir if args.save else None
            plot_results(results, output_dir=output_dir, show=True)
            plot_assignment_heatmap(results, output_dir=output_dir, show=True)
            # New intensity/spike analysis plots
            plot_intensity_spike_analysis(results, output_dir=output_dir, show=True)
            plot_per_experiment_spike_breakdown(results, output_dir=output_dir, show=True)
            plot_class_response_heatmap(results, output_dir=output_dir, show=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
