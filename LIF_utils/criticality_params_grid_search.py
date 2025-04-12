import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

# Import custom modules
from LIF_objects.CircularNeuronalNetwork import CircularNeuronalNetwork
from LIF_utils.simulation_utils import run_unified_simulation
from LIF_utils.criticality_analysis_utils import analyze_criticality_comprehensively

def evaluate_criticality_params(params, fixed_params):
    """
    Evaluate a set of parameters for criticality by running a simulation and analyzing the results.
    
    Parameters:
    -----------
    params : dict
        Dictionary of parameters to vary in the grid search
    fixed_params : dict
        Dictionary of parameters to keep fixed
    
    Returns:
    --------
    dict
        Results including criticality score and other metrics
    """
    # Set random seed for reproducibility
    random_seed = fixed_params.get('random_seed', 42)
    np.random.seed(random_seed)
    
    # Merge parameters
    all_params = {**fixed_params, **params}
    
    # Create network with the parameters
    network = CircularNeuronalNetwork(
        n_neurons=all_params['n_neurons'], 
        connection_p=params['connection_p'],
        weight_scale=params['weight_scale'],
        spatial=True,
        transmission_delay=params['transmission_delay'],
        inhibitory_fraction=params['inhibitory_fraction'],
        layout=all_params['layout'],
        v_noise_amp=params['v_noise_amp'],
        i_noise_amp=params['i_noise_amp'],
        e_reversal=params['e_reversal'],
        i_reversal=params['i_reversal'],
        distance_lambda=all_params.get('distance_lambda', 0.1)
    )
    
    # Run simulation
    activity_record, _, stimulation_record = run_unified_simulation(
        network, 
        duration=all_params['duration'], 
        dt=all_params['dt'], 
        stim_interval=params['stim_interval'],
        stim_interval_strength=all_params['stim_interval_strength'],
        stim_fraction=all_params['stim_fraction'],
        stim_neuron=None,
        track_neurons=None,  # Don't track neurons for efficiency
        stochastic_stim=all_params['stochastic_stim'],
        no_stimulation=all_params['no_stimulation']
    )
    
    # Analyze criticality
    criticality_results = analyze_criticality_comprehensively(network, save_plots=False)
    
    # Collect metrics
    results = {
        'params': params.copy(),
        'avalanche_count': criticality_results['avalanche_count'],
        'critical_score': criticality_results.get('critical_score', 0),
        'is_critical': criticality_results.get('critical', False),
        'assessment': criticality_results.get('assessment', 'Not critical'),
        'branching_ratio': criticality_results.get('branching_ratio', None),
        'size_exponent': criticality_results.get('size_exponent_used', None),
        'duration_exponent': criticality_results.get('duration_exponent_used', None),
        'scaling_relation': criticality_results.get('scaling_relation', None),
    }
    
    return results


def grid_search_criticality(param_grid, fixed_params, results_dir='grid_search_results', max_simulations=None):
    """
    Perform grid search to find parameters that bring the network closest to criticality.
    
    Parameters:
    -----------
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of values to try
    fixed_params : dict
        Dictionary of parameters to keep fixed
    results_dir : str
        Directory to save results
    max_simulations : int or None
        Maximum number of simulations to run (None for all combinations)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing all results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Calculate total combinations
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)
    
    print(f"Total parameter combinations: {total_combinations}")
    
    if max_simulations is not None and max_simulations < total_combinations:
        print(f"Limiting to {max_simulations} random combinations")
        # Sample random combinations
        combinations_indices = np.random.choice(total_combinations, max_simulations, replace=False)
    else:
        combinations_indices = range(total_combinations)
    
    # Initialize results list
    all_results = []
    best_score = 0
    best_params = None
    
    # Set up progress tracking
    start_time = time.time()
    results_file = os.path.join(results_dir, 'grid_search_results.csv')
    params_file = os.path.join(results_dir, 'best_params.json')
    
    '''
    # Load previous results if they exist
    if os.path.exists(results_file):
        print("Loading previous results...")
        previous_results = pd.read_csv(results_file)
        all_results = previous_results.to_dict('records')
        if not all_results:
            all_results = []
        else:
            # Find best previous result
            best_idx = previous_results['critical_score'].idxmax()
            best_score = previous_results.loc[best_idx, 'critical_score']
            best_params = {param: previous_results.loc[best_idx, param] for param in param_names}
            print(f"Best previous score: {best_score} with params: {best_params}")
    '''
    
    # Helper function to convert combination index to parameter values
    def index_to_params(idx):
        params = {}
        temp_idx = idx
        for i in range(len(param_names)-1, -1, -1):
            divisor = 1
            for j in range(i):
                divisor *= len(param_values[j])
            param_idx = temp_idx // divisor
            temp_idx = temp_idx % divisor
            params[param_names[i]] = param_values[i][param_idx]
        return params
    
    
    # Run grid search
    for combination_idx in tqdm(combinations_indices, desc="Grid Search Progress"):
        # Convert index to parameter combination
        params = index_to_params(combination_idx)
        
        # Check if this combination has already been evaluated
        already_evaluated = False
        for result in all_results:
            if all(result['params'].get(key) == value for key, value in params.items()):
                already_evaluated = True
                break
        
        if already_evaluated:
            continue
        
        # Print current parameter combination
        print(f"\nEvaluating parameters: {params}")
        
        # Evaluate this parameter combination
        try:
            results = evaluate_criticality_params(params, fixed_params)
            all_results.append(results)
            
            # Update best parameters if better score found
            if results['critical_score'] > best_score:
                best_score = results['critical_score']
                best_params = params.copy()
                print(f"\n=== NEW BEST SCORE: {best_score:.4f} ===")
                print(f"Parameters: {best_params}")
                print(f"Assessment: {results['assessment']}")
                
                # Save best parameters
                with open(params_file, 'w') as f:
                    json.dump({
                        'best_score': best_score,
                        'best_params': best_params,
                        'assessment': results['assessment'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'avalanche_count': results['avalanche_count']
                    }, f, indent=4)
            
            # Save results after each evaluation
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            
            # Update progress plot
            update_progress_plot(all_results, results_dir)
            
        except Exception as e:
            print(f"Error evaluating parameters {params}: {e}")
    
    # Final results
    results_df = pd.DataFrame(all_results)
    
    print("\n=== Grid Search Complete ===")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Best criticality score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Print in format suitable for run_improved_experiment
    print("\nBest parameters for run_improved_experiment:")
    print_experiment_params(best_params, fixed_params)
    
    return results_df


def update_progress_plot(all_results, results_dir):
    """Create a progress plot showing the criticality scores for all parameter combinations."""
    if not all_results:
        return
    
    plt.figure(figsize=(12, 8), facecolor='#1a1a1a')
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_results)
    
    # Create a new column for iteration number
    df['iteration'] = range(1, len(df) + 1)
    
    # Plot criticality score over iterations
    ax1 = plt.subplot(211)
    ax1.plot(df['iteration'], df['critical_score'], 'o-', color='#ff7f0e', markersize=8, alpha=0.7)
    ax1.set_ylabel('Criticality Score', color='white', fontsize=12)
    ax1.set_title('Grid Search Progress', color='white', fontsize=14)
    
    # Plot a horizontal line at the ideal score
    ax1.axhline(y=1.0, color='#1dd1a1', linestyle='--', alpha=0.7, label='Ideal Score')
    
    # Find best score and highlight it
    best_idx = df['critical_score'].idxmax()
    best_score = df.loc[best_idx, 'critical_score']
    ax1.plot(df.loc[best_idx, 'iteration'], best_score, 'o', color='#1dd1a1', 
             markersize=12, label=f'Best Score: {best_score:.4f}')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.7)
    
    # Plot avalanche count in a separate subplot
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(df['iteration'], df['avalanche_count'], 'o-', color='#2980b9', markersize=8, alpha=0.7)
    ax2.set_xlabel('Iteration', color='white', fontsize=12)
    ax2.set_ylabel('Avalanche Count', color='white', fontsize=12)
    ax2.set_title('Number of Avalanches Detected', color='white', fontsize=14)
    
    # Highlight avalanche count for best score
    best_count = df.loc[best_idx, 'avalanche_count']
    ax2.plot(df.loc[best_idx, 'iteration'], best_count, 'o', color='#1dd1a1', 
             markersize=12, label=f'Best Count: {best_count}')
    
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.7)
    
    # Style the plot for dark theme
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'progress_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create parameter importance plot
    if len(df) >= 5:  # Need at least a few data points
        plt.figure(figsize=(14, 10), facecolor='#1a1a1a')
        
        # Extract parameter columns
        param_cols = [col for col in df.columns if col.startswith('params.')]
        
        # Calculate correlation between each parameter and criticality score
        correlations = []
        for col in param_cols:
            param_name = col.replace('params.', '')
            try:
                corr = df[col].corr(df['critical_score'])
                if not pd.isna(corr):
                    correlations.append((param_name, corr))
            except:
                pass
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Plot correlations
        if correlations:
            params = [c[0] for c in correlations]
            corrs = [c[1] for c in correlations]
            
            plt.barh(params, corrs, color=['#1dd1a1' if c > 0 else '#ff6b6b' for c in corrs])
            plt.xlabel('Correlation with Criticality Score', color='white', fontsize=12)
            plt.ylabel('Parameter', color='white', fontsize=12)
            plt.title('Parameter Importance for Criticality', color='white', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axvline(x=0, color='white', linestyle='-', alpha=0.5)
            
            plt.gca().set_facecolor('#1a1a1a')
            plt.gca().tick_params(colors='white')
            for spine in plt.gca().spines.values():
                spine.set_color('white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()


def print_experiment_params(best_params, fixed_params):
    """Print parameters in format suitable for run_improved_experiment."""
    all_params = {**fixed_params, **best_params}
    
    param_template = """
return run_improved_experiment(
    # Network parameters
    n_neurons={n_neurons},              # Local cortical circuit size
    connection_p={connection_p},            # Connection probability
    weight_scale={weight_scale},           # Weight scale
    inhibitory_fraction={inhibitory_fraction},      # Inhibitory neuron fraction
    transmission_delay={transmission_delay},       # Transmission delay
    distance_lambda={distance_lambda},         # Distance decay parameter
    
    # Simulation parameters
    duration={duration},                 # Simulation duration
    dt={dt},                       # Timestep size
    
    # Stimulation parameters
    stim_interval={stim_interval},              # Stimulation interval
    stim_interval_strength={stim_interval_strength},     # Stimulation strength
    stim_fraction={stim_fraction},                # Fraction of neurons to stimulate
    no_stimulation={no_stimulation},
    stochastic_stim={stochastic_stim},            # Stochastic stimulation
    
    # Noise parameters
    enable_noise={enable_noise},            # Enable noise
    v_noise_amp={v_noise_amp},              # Membrane potential noise
    i_noise_amp={i_noise_amp},             # Synaptic noise
    
    # Reversal potential parameters
    e_reversal={e_reversal},               # Excitatory reversal potential
    i_reversal={i_reversal},             # Inhibitory reversal potential
    
    # Other parameters
    layout='{layout}',              # Network layout
    random_seed=random_seed       # Random seed for reproducibility
)""".format(**all_params)
    
    print(param_template)


def main():
    # Define fixed parameters
    fixed_params = {
        'n_neurons': 3000,              # Smaller network for faster search
        'duration': 1000,               # Shorter duration for faster search
        'dt': 0.1,                      # Timestep size
        'stim_interval_strength': 50,   # Stimulation strength
        'stim_fraction': 0.01,          # Fraction of neurons to stimulate
        'no_stimulation': False,        # Apply external stimulation
        'stochastic_stim': False,       # Use regular stimulation
        'enable_noise': True,           # Enable neural noise
        'layout': 'circle',             # Network layout
        'random_seed': 42,              # Random seed
        'distance_lambda': 0.1          # Distance decay parameter
    }
    
    # Define parameter ranges to search
    param_grid = {
        'connection_p': [0.05, 0.1, 0.2, 0.3, 0.4],           # Connection probability
        'weight_scale': [0.01, 0.05, 0.1, 0.5, 1.0],          # Weight scale
        'inhibitory_fraction': [0.2, 0.3, 0.4, 0.5, 0.6],     # Inhibitory fraction
        'transmission_delay': [0.2, 0.5, 1.0, 2.0],           # Transmission delay
        'v_noise_amp': [0.05, 0.1, 0.2, 0.3],                 # Membrane potential noise
        'i_noise_amp': [0.005, 0.01, 0.02, 0.05],             # Synaptic noise
        'e_reversal': [0.0],                                  # Excitatory reversal potential (fixed)
        'i_reversal': [-80.0, -70.0],                         # Inhibitory reversal potential
        'stim_interval': [50, 100, 200, 300]                  # Stimulation interval
    }
    
    # Run grid search
    results = grid_search_criticality(
        param_grid=param_grid, 
        fixed_params=fixed_params,
        results_dir='grid_search_results',
        max_simulations=2500  # Limit to 2500 randomly sampled combinations for faster search
    )
    
    return results

if __name__ == "__main__":
    main()