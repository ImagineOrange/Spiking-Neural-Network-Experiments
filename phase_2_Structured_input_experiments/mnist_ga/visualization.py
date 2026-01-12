"""
Visualization utilities for GA progress and results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import os


def plot_ga_progress(generation: int,
                    best_fitness_history: List[float],
                    avg_fitness_history: List[float],
                    save_path: str,
                    test_accuracy: Optional[float] = None,
                    eval_examples: int = 1000):
    """
    Plot and save GA fitness evolution.

    Args:
        generation: Current generation number
        best_fitness_history: List of best fitness per generation
        avg_fitness_history: List of average fitness per generation
        save_path: Path to save the plot
        test_accuracy: Optional test accuracy to display
        eval_examples: Number of examples used for fitness evaluation
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')

    gen_axis = range(1, generation + 1)

    # Filter out -inf values before plotting
    best_plot = [f if np.isfinite(f) else np.nan for f in best_fitness_history]
    avg_plot = [f if np.isfinite(f) else np.nan for f in avg_fitness_history]

    # Plot fitness curves
    ax.plot(gen_axis, best_plot, marker='x', linestyle='--',
            color='cyan', markersize=4, linewidth=2, label='Best Fitness')
    ax.plot(gen_axis, avg_plot, marker='o', linestyle='-',
            color='orange', markersize=3, linewidth=1.5, alpha=0.7, label='Average Fitness')

    # Styling
    ax.set_title(f'GA Fitness Evolution - Generation {generation}',
                 color='white', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', color='white', fontsize=12)
    ax.set_ylabel(f'Fitness (Balanced Accuracy on {eval_examples} examples)',
                  color='white', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.7, fontsize=10)
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white')

    # Set spine colors
    for spine in ax.spines.values():
        spine.set_color('white')

    # Add test accuracy if provided
    if test_accuracy is not None:
        ax.text(0.95, 0.1, f'Test Acc: {test_accuracy:.1%}',
                transform=ax.transAxes,
                ha='right', va='bottom',
                color='lime', fontsize=11, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))

    plt.tight_layout()

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor='#1a1a1a')
    except Exception as e:
        print(f"Error saving plot to {save_path}: {e}")

    plt.close(fig)


if __name__ == "__main__":
    # Demo: Create sample GA progress plot
    print("Creating demo GA progress plot...")

    # Simulate GA evolution
    generations = 50
    best_fitness = []
    avg_fitness = []

    for gen in range(generations):
        # Simulate improving fitness
        best = min(0.95, 0.3 + 0.65 * (gen / generations) + np.random.rand() * 0.05)
        avg = best - 0.1 - np.random.rand() * 0.1
        best_fitness.append(best)
        avg_fitness.append(avg)

    plot_ga_progress(
        generation=generations,
        best_fitness_history=best_fitness,
        avg_fitness_history=avg_fitness,
        save_path="demo_ga_progress.png",
        test_accuracy=0.87,
        eval_examples=1000
    )

    print(f"Saved demo plot to: demo_ga_progress.png")
