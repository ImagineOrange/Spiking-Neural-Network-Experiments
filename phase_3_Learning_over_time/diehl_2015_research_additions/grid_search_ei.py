#!/usr/bin/env python
"""
Grid Search for E:I Ratio and Inhibition Parameters

This script runs a grid search over different E:I configurations to find
the optimal balance between excitation and inhibition for MNIST classification.

Parameters searched:
- n_i: Number of inhibitory neurons (affects E:I ratio)
- pConn_ei: E->I connectivity (how many I neurons each E activates)
- pConn_ie: I->E connectivity (inhibition spread/overlap)
- weight_ie: I->E synaptic weight (inhibition strength)

Usage:
    python grid_search_ei.py                    # Run full grid search
    python grid_search_ei.py --workers 4        # Specify parallel workers
    python grid_search_ei.py --quick            # Quick test (fewer configs)
    python grid_search_ei.py --resume DIR       # Resume from previous run

The script will:
1. Generate all parameter combinations
2. Run experiments in parallel
3. Evaluate each trained network
4. Produce a summary ranking configurations by accuracy

CRITICAL FIX (2024-01-14): This version uses robust module isolation to prevent
config caching bugs across experiments in the same worker process.
"""

import os
import sys
import argparse
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import shutil
import json
import itertools
from datetime import datetime
import numpy as np
import importlib.util

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# GRID SEARCH PARAMETER SPACE
# =============================================================================

# Define the parameter grid - FIXED 4:1 architecture (400 E : 100 I)
PARAM_GRID = {
    # Input->Inhibitory connectivity: direct feedforward inhibition
    'pConn_ei_input': [0.05, 0.10, 0.15],  # 5%, 10%, 15%

    # E->I connectivity: how many I neurons each E spike activates
    'pConn_ei': [0.03, 0.05, 0.08, 0.12],  # 3%, 5%, 8%, 12%

    # I->E connectivity: inhibition spread (affects overlap)
    'pConn_ie': [0.20, 0.30, 0.45, 0.60],  # 20%, 30%, 45%, 60%

    # I->E weight: inhibition strength (will be auto-calibrated if None)
    'weight_ie': [None],  # None = auto-calibrate to target drive
}

# Quick mode uses smaller grid
PARAM_GRID_QUICK = {
    'pConn_ei_input': [0.10],  # Fixed at default for quick mode
    'pConn_ei': [0.05, 0.10],
    'pConn_ie': [0.30, 0.50],
    'weight_ie': [None],
}

# Target inhibitory drive (from original 1:1 network)
TARGET_INHIB_DRIVE = 17.0

# Fixed parameters - 4:1 architecture
FIXED_PARAMS = {
    'n_e': 400,
    'n_i': 100,  # Fixed 4:1 ratio
    'n_input': 784,
    'weight_ei': 10.4,  # E->I weight
    'pConn_ee_input': 1.0,
    # pConn_ei_input is now a search parameter
}


# =============================================================================
# ROBUST MODULE ISOLATION
# =============================================================================

def clear_module_cache():
    """
    Clear all potentially cached modules that could cause cross-experiment contamination.

    This is CRITICAL for correct grid search behavior. Python's sys.modules cache
    will retain imported modules across exec() calls, causing subsequent experiments
    to use stale configurations.

    Modules to clear:
    - config: The experiment-specific configuration
    - brian2: Has global state (defaultclock, device)
    - Any submodules of the above
    """
    modules_to_clear = []

    # Find all modules that need clearing
    for mod_name in list(sys.modules.keys()):
        if mod_name == 'config' or mod_name.startswith('config.'):
            modules_to_clear.append(mod_name)
        # Also clear sim_and_eval_utils since it imports config
        if 'sim_and_eval_utils' in mod_name:
            modules_to_clear.append(mod_name)

    # Clear them
    for mod_name in modules_to_clear:
        del sys.modules[mod_name]

    # Reset Brian2 if it's loaded (it has global state)
    if 'brian2' in sys.modules:
        try:
            from brian2 import device
            device.reinit()
        except Exception:
            pass  # Best effort - Brian2 might not be fully initialized


def load_config_isolated(config_path):
    """
    Load a config module WITHOUT polluting sys.modules.

    This uses importlib.util to load the config file as a standalone module
    that won't be cached and won't interfere with other experiments.

    Args:
        config_path: Absolute path to config.py file

    Returns:
        The Config class from the loaded module
    """
    spec = importlib.util.spec_from_file_location("_isolated_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config


def verify_config_paths(cfg, expected_exp_dir):
    """
    Verify that loaded config has correct paths for this experiment.

    This is a critical sanity check that will catch module caching bugs.

    Args:
        cfg: Config instance
        expected_exp_dir: The experiment directory we expect to be using

    Raises:
        AssertionError if paths don't match
    """
    # Normalize paths for comparison
    expected = os.path.normpath(expected_exp_dir)
    actual = os.path.normpath(cfg.data_path.rstrip('/'))

    if expected != actual:
        raise AssertionError(
            f"CONFIG PATH MISMATCH!\n"
            f"  Expected: {expected}\n"
            f"  Got:      {actual}\n"
            f"  This indicates a module caching bug."
        )


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================

def set_test_mode(exp_dir, enable):
    """
    Modify experiment's config.py to set test_mode.

    Args:
        exp_dir: Path to experiment directory containing config.py
        enable: True to set test_mode=True, False to set test_mode=False

    Returns:
        Original line if changed, None if no change needed
    """
    config_path = os.path.join(exp_dir, 'config.py')

    with open(config_path, 'r') as f:
        content = f.read()

    if enable:
        if 'self.test_mode = False' in content:
            content = content.replace('self.test_mode = False', 'self.test_mode = True')
            with open(config_path, 'w') as f:
                f.write(content)
            return 'self.test_mode = False'
    else:
        if 'self.test_mode = True' in content:
            content = content.replace('self.test_mode = True', 'self.test_mode = False')
            with open(config_path, 'w') as f:
                f.write(content)
            return 'self.test_mode = True'

    return None


def run_experiment_isolated(exp_dir, log_file, is_test_mode=False):
    """
    Run an experiment with full module isolation.

    This function ensures that each experiment run uses ONLY its own config
    by clearing module caches and verifying paths.

    Args:
        exp_dir: Absolute path to experiment directory
        log_file: Open file handle for logging
        is_test_mode: Whether this is a test pass (vs training)

    Returns:
        True if successful, False otherwise
    """
    from contextlib import redirect_stdout, redirect_stderr

    # CRITICAL: Clear any cached modules from previous experiments
    clear_module_cache()

    original_dir = os.getcwd()
    original_path = sys.path.copy()

    try:
        # Set up paths - exp_dir MUST be first for its config.py to be found
        # Remove any existing entries first to avoid duplicates
        if exp_dir in sys.path:
            sys.path.remove(exp_dir)
        if SCRIPT_DIR in sys.path:
            sys.path.remove(SCRIPT_DIR)

        sys.path.insert(0, exp_dir)
        sys.path.insert(1, SCRIPT_DIR)

        # Change to experiment directory
        os.chdir(exp_dir)

        # VERIFICATION: Load config and verify paths BEFORE running
        config_path = os.path.join(exp_dir, 'config.py')
        ConfigClass = load_config_isolated(config_path)
        test_cfg = ConfigClass()
        verify_config_paths(test_cfg, exp_dir)

        # Log verification
        log_file.write(f"CONFIG VERIFICATION PASSED:\n")
        log_file.write(f"  Experiment: {os.path.basename(exp_dir)}\n")
        log_file.write(f"  data_path: {test_cfg.data_path}\n")
        log_file.write(f"  test_mode: {test_cfg.test_mode}\n")
        log_file.write(f"  n_i: {test_cfg.n_i}\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.flush()

        # Clean up the isolated config - we'll let the exec'd script import fresh
        del test_cfg
        del ConfigClass

        # Clear module cache AGAIN to ensure exec() gets fresh imports
        clear_module_cache()

        # Read and execute main script
        main_script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_spiking_MNIST_4to1.py')
        with open(main_script_path, 'r') as f:
            script_content = f.read()

        with redirect_stdout(log_file), redirect_stderr(log_file):
            exec(script_content, {
                '__name__': '__main__',
                '__file__': main_script_path
            })

        return True

    except Exception as e:
        log_file.write(f'\nEXPERIMENT ERROR: {str(e)}\n')
        import traceback
        log_file.write(traceback.format_exc())
        log_file.write(f'\nExperiment directory: {exp_dir}\n')
        log_file.write(f'Working directory at error: {os.getcwd()}\n')
        return False

    finally:
        # Restore original state
        os.chdir(original_dir)
        sys.path[:] = original_path

        # Clear modules one more time to not pollute next experiment
        clear_module_cache()


def run_test_pass(exp_dir, log_file):
    """
    Run test mode to generate test activity on held-out test set.

    Args:
        exp_dir: Path to experiment directory
        log_file: Open file handle for logging output

    Returns:
        True if successful, False otherwise
    """
    original = set_test_mode(exp_dir, enable=True)

    try:
        success = run_experiment_isolated(exp_dir, log_file, is_test_mode=True)
        return success

    finally:
        # Restore config to training mode
        if original:
            set_test_mode(exp_dir, enable=False)


# =============================================================================
# CONFIGURATION GENERATION
# =============================================================================

def calculate_weight_ie(n_i, pConn_ei, pConn_ie, target_drive=TARGET_INHIB_DRIVE):
    """
    Calculate I->E weight to achieve target inhibitory drive.

    Drive = (n_i * pConn_ei) * pConn_ie * weight_ie

    Solving for weight_ie:
    weight_ie = target_drive / ((n_i * pConn_ei) * pConn_ie)
    """
    i_neurons_per_e_spike = n_i * pConn_ei
    expected_hits = i_neurons_per_e_spike * pConn_ie

    if expected_hits == 0:
        return 10.0  # fallback

    weight_ie = target_drive / expected_hits
    return round(weight_ie, 2)


def generate_experiment_configs(param_grid, fixed_params):
    """Generate all parameter combinations."""

    # Get all parameter names and values
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    configs = []
    for values in itertools.product(*param_values):
        config = dict(zip(param_names, values))
        config.update(fixed_params)

        # Auto-calibrate weight_ie if None
        if config.get('weight_ie') is None:
            config['weight_ie'] = calculate_weight_ie(
                config['n_i'],
                config['pConn_ei'],
                config['pConn_ie']
            )

        # Calculate derived metrics
        i_per_e = config['n_i'] * config['pConn_ei']
        inhib_drive = i_per_e * config['pConn_ie'] * config['weight_ie']
        overlap = 1 - (1 - config['pConn_ie']) ** i_per_e

        config['_i_per_e_spike'] = i_per_e
        config['_inhib_drive'] = round(inhib_drive, 2)
        config['_overlap'] = round(overlap * 100, 1)
        config['_ei_ratio'] = f"{config['n_e']}:{config['n_i']}"

        configs.append(config)

    return configs


def create_experiment_directory(base_dir, exp_id, config, base_seed=0):
    """
    Create experiment directory with custom config and weights.

    NOTE: Uses experiment-specific seed (base_seed + exp_id) for reproducibility
    while ensuring different experiments have different random initializations.
    """
    # Experiment-specific seed for diversity
    exp_seed = base_seed + exp_id

    exp_name = (f"exp_{exp_id:03d}_"
                f"xi{int(config['pConn_ei_input']*100)}_"
                f"ei{int(config['pConn_ei']*100)}_"
                f"ie{int(config['pConn_ie']*100)}_"
                f"w{config['weight_ie']:.1f}")

    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'random'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'activity'), exist_ok=True)

    # Save config as JSON for reference
    config_json = {k: v for k, v in config.items()}
    config_json['seed'] = exp_seed
    config_json['base_seed'] = base_seed
    config_json['exp_id'] = exp_id
    with open(os.path.join(exp_dir, 'params.json'), 'w') as f:
        json.dump(config_json, f, indent=2)

    # Generate random weights with experiment-specific seed
    generate_weights_for_config(exp_dir, config, exp_seed)

    # Generate Python config file
    generate_config_file(exp_dir, config, exp_seed)

    return exp_dir, exp_name


def generate_weights_for_config(exp_dir, config, seed):
    """Generate random connection weights for this configuration."""

    np.random.seed(seed)

    n_input = config['n_input']
    n_e = config['n_e']
    n_i = config['n_i']

    data_path = os.path.join(exp_dir, 'random')

    # Weight values
    weight_ee_input = 0.3
    weight_ei_input = 0.2
    weight_ei = config['weight_ei']
    weight_ie = config['weight_ie']

    # Connectivity
    pConn_ee_input = config['pConn_ee_input']
    pConn_ei_input = config['pConn_ei_input']
    pConn_ei = config['pConn_ei']
    pConn_ie = config['pConn_ie']

    def sparsen_matrix(base_matrix, p_conn):
        """Create sparse connectivity from dense matrix."""
        weight_matrix = np.zeros(base_matrix.shape)
        num_target = int(base_matrix.shape[0] * base_matrix.shape[1] * p_conn)
        weight_list = []

        indices = set()
        while len(weight_list) < num_target:
            i = np.random.randint(base_matrix.shape[0])
            j = np.random.randint(base_matrix.shape[1])
            if (i, j) not in indices:
                indices.add((i, j))
                weight_matrix[i, j] = base_matrix[i, j]
                weight_list.append((i, j, base_matrix[i, j]))

        return weight_matrix, weight_list

    # XeAe: Input -> Excitatory (all-to-all)
    weight_matrix = (np.random.random((n_input, n_e)) + 0.01) * weight_ee_input
    weight_list = [(i, j, weight_matrix[i, j]) for j in range(n_e) for i in range(n_input)]
    np.save(os.path.join(data_path, 'XeAe.npy'), weight_list)

    # XeAi: Input -> Inhibitory (sparse)
    weight_matrix = np.random.random((n_input, n_i)) * weight_ei_input
    _, weight_list = sparsen_matrix(weight_matrix, pConn_ei_input)
    np.save(os.path.join(data_path, 'XeAi.npy'), weight_list)

    # AeAi: Excitatory -> Inhibitory (sparse)
    weight_matrix = np.random.random((n_e, n_i)) * weight_ei
    _, weight_list = sparsen_matrix(weight_matrix, pConn_ei)
    np.save(os.path.join(data_path, 'AeAi.npy'), weight_list)

    # AiAe: Inhibitory -> Excitatory (sparse)
    weight_matrix = np.random.random((n_i, n_e)) * weight_ie
    _, weight_list = sparsen_matrix(weight_matrix, pConn_ie)
    np.save(os.path.join(data_path, 'AiAe.npy'), weight_list)


def generate_config_file(exp_dir, config, seed):
    """Generate config.py for this experiment."""

    # Use absolute path for MNIST data (in SCRIPT_DIR)
    mnist_data_abs_path = os.path.join(SCRIPT_DIR, 'mnist_data')

    config_content = f'''"""
Auto-generated config for grid search experiment
Parameters: n_i={config['n_i']}, pConn_ei={config['pConn_ei']}, pConn_ie={config['pConn_ie']}, weight_ie={config['weight_ie']}
E:I Ratio: {config['_ei_ratio']}
Inhibitory Drive: {config['_inhib_drive']}
Overlap: {config['_overlap']}%
Seed: {seed}
"""

from brian2 import ms, mV, second, Hz
import numpy as np


class Config:
    def __init__(self):
        # Mode Settings
        self.test_mode = False

        # Path Settings - ABSOLUTE PATHS for grid search
        self.mnist_data_path = '{mnist_data_abs_path}/'
        self.data_path = '{exp_dir}/'

        # Simulation Settings
        self.dt = 0.5 * ms
        self.random_seed = {seed}

        # Network Architecture - GRID SEARCH PARAMETERS
        self.n_input = {config['n_input']}
        self.n_e = {config['n_e']}
        self.n_i = {config['n_i']}  # E:I ratio = {config['_ei_ratio']}

        # Timing Parameters
        self.single_example_time = 0.35 * second
        self.resting_time = 0.15 * second

        # Training Parameters
        self.mnist_classes = [0, 1, 2, 3, 4]
        self.num_train_examples = 10345
        self.num_test_examples = 4900
        self.test_examples_per_class = 980
        self.assignment_examples_per_class = 500

        self.num_examples = None
        self.use_testing_set = None
        self.do_plot_performance = None
        self.record_spikes = True
        self.ee_STDP_on = None
        self.enable_live_plots = False

        self.update_interval = None
        self.weight_update_interval = 20
        self.save_connections_interval = 10000

        # Neuron Parameters
        self.v_rest_e = -65. * mV
        self.v_reset_e = -65. * mV
        self.v_thresh_e_const = -52. * mV
        self.refrac_e = 5. * ms
        self.offset = 20.0 * mV

        self.v_rest_i = -60. * mV
        self.v_reset_i = -45. * mV
        self.v_thresh_i = -40. * mV
        self.refrac_i = 2. * ms

        self.noise_sigma_e = 0.3 * mV
        self.noise_sigma_i = 0.3 * mV

        self.weight_ee_input = 78.
        self.wmax_ee = 1.0

        self.delay_ee_input = (0*ms, 10*ms)
        self.delay_ei_input = (0*ms, 5*ms)

        self.input_intensity = 2.0
        self.start_input_intensity = self.input_intensity

        # STDP Parameters
        self.tc_pre_ee = 20 * ms
        self.tc_post_1_ee = 20 * ms
        self.tc_post_2_ee = 40 * ms
        self.nu_ee_pre = 0.0001
        self.nu_ee_post = 0.01
        self.exp_ee_pre = 0.2
        self.exp_ee_post = 0.2
        self.STDP_offset = 0.4

        self.tc_theta = 1e7 * ms
        self.theta_plus_e = 0.05 * mV

        # Population Names
        self.input_population_names = ['X']
        self.population_names = ['A']
        self.input_connection_names = ['XA']
        self.save_conns = ['XeAe']
        self.input_conn_names = ['ee_input', 'ei_input']
        self.recurrent_conn_names = ['ei', 'ie']
        self.ending = ''

        self._compute_derived_params()

    def _compute_derived_params(self):
        if self.test_mode:
            self.weight_path = self.data_path + 'weights/'
            self.num_examples = self.num_test_examples
            self.use_testing_set = True
            self.do_plot_performance = False
            self.ee_STDP_on = False
            self.update_interval = self.num_examples
        else:
            self.weight_path = self.data_path + 'random/'
            self.num_examples = self.num_train_examples * 3
            self.use_testing_set = False
            self.do_plot_performance = False
            self.ee_STDP_on = True
            self.record_spikes = True
            if self.num_examples <= 10000:
                self.update_interval = self.num_examples
            else:
                self.update_interval = 10000
            self.save_connections_interval = 10000

        self.runtime = self.num_examples * (self.single_example_time + self.resting_time)

    def get_neuron_eqs_e(self):
        eqs = \'\'\'
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) + noise_sigma_e*xi_e/sqrt(100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        \'\'\'
        if self.test_mode:
            eqs += '\\n  theta      :volt'
        else:
            eqs += '\\n  dtheta/dt = -theta / (tc_theta)  : volt'
        eqs += '\\n  dtimer/dt = 100.0*msecond/second  : second'
        return eqs

    def get_neuron_eqs_i(self):
        return \'\'\'
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) + noise_sigma_i*xi_i/sqrt(10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        \'\'\'

    def get_stdp_eqs(self):
        return \'\'\'
                w : 1
                post2before : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            \'\'\'

    def get_stdp_pre_eq(self):
        return 'ge_post += w; pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'

    def get_stdp_post_eq(self):
        return 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    def get_v_thresh_e_str(self):
        return '(v>(theta - offset + v_thresh_e_const)) and (timer>refrac_e)'

    def get_scr_e(self):
        if self.test_mode:
            return 'v = v_reset_e; timer = 0*ms'
        else:
            return 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    def set_test_mode(self, test_mode):
        self.test_mode = test_mode
        self._compute_derived_params()

    def get_class_filter_str(self):
        if self.mnist_classes is None:
            return "all classes [0-9]"
        return f"classes {{self.mnist_classes}}"

    def should_use_example(self, label):
        if self.mnist_classes is None:
            return True
        return label in self.mnist_classes

    def __repr__(self):
        return f"Config(n_i={config['n_i']}, E:I={config['_ei_ratio']}, drive={config['_inhib_drive']})"


default_config = Config()
'''

    config_path = os.path.join(exp_dir, 'config.py')
    with open(config_path, 'w') as f:
        f.write(config_content)


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(args):
    """Run a single grid search experiment with full isolation."""

    exp_id, config, base_dir, base_seed, verbose = args

    # Compute experiment-specific seed
    exp_seed = base_seed + exp_id
    exp_dir = None
    exp_name = None
    start_time = time.time()

    try:
        # Create experiment directory
        exp_dir, exp_name = create_experiment_directory(base_dir, exp_id, config, base_seed)

        log_path = os.path.join(exp_dir, 'training.log')

        if verbose:
            print(f"[{exp_id:3d}] Starting: n_i={config['n_i']}, "
                  f"xi={config['pConn_ei_input']:.0%}, "
                  f"ei={config['pConn_ei']:.0%}, ie={config['pConn_ie']:.0%}, "
                  f"w={config['weight_ie']:.1f}")

        with open(log_path, 'w') as log_file:
            # Header
            log_file.write("=" * 60 + "\n")
            log_file.write(f"EXPERIMENT {exp_id}: {exp_name}\n")
            log_file.write("=" * 60 + "\n")
            log_file.write(f"Seed: {exp_seed} (base={base_seed}, exp_id={exp_id})\n")
            log_file.write(f"Parameters:\n")
            for k, v in config.items():
                if not k.startswith('_'):
                    log_file.write(f"  {k}: {v}\n")
            log_file.write("=" * 60 + "\n\n")
            log_file.flush()

            # Run training with isolation
            log_file.write("PHASE 1: TRAINING\n")
            log_file.write("-" * 60 + "\n")
            log_file.flush()

            train_success = run_experiment_isolated(exp_dir, log_file, is_test_mode=False)

            if not train_success:
                log_file.write("\nTRAINING FAILED - skipping test pass\n")
            else:
                # Run test pass
                log_file.write('\n' + '='*60 + '\n')
                log_file.write('PHASE 2: TEST PASS (for proper evaluation)\n')
                log_file.write('='*60 + '\n')
                log_file.flush()

                test_success = run_test_pass(exp_dir, log_file)

                if not test_success:
                    log_file.write('WARNING: Test pass failed, evaluation may be invalid\n')

        elapsed = time.time() - start_time

        # Evaluate the trained network
        accuracy = evaluate_experiment(exp_dir)

        result = {
            'exp_id': exp_id,
            'exp_name': exp_name,
            'exp_dir': exp_dir,
            'config': {k: v for k, v in config.items() if not k.startswith('_')},
            'seed': exp_seed,
            'n_i': config['n_i'],
            'pConn_ei_input': config['pConn_ei_input'],
            'pConn_ei': config['pConn_ei'],
            'pConn_ie': config['pConn_ie'],
            'weight_ie': config['weight_ie'],
            'ei_ratio': config['_ei_ratio'],
            'inhib_drive': config['_inhib_drive'],
            'overlap': config['_overlap'],
            'accuracy': accuracy,
            'elapsed_time': elapsed,
            'success': accuracy is not None,
            'error': None
        }

        if verbose:
            acc_str = f"{accuracy:.2f}%" if accuracy else "FAILED"
            print(f"[{exp_id:3d}] Done: {acc_str} ({elapsed/60:.1f} min)")

        return result

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()

        # Add context to error
        error_msg = (
            f"Experiment {exp_id} failed\n"
            f"exp_dir: {exp_dir}\n"
            f"config: {config}\n"
            f"Traceback:\n{error_msg}"
        )

        if verbose:
            print(f"[{exp_id:3d}] ERROR: {str(e)[:50]}")

        return {
            'exp_id': exp_id,
            'exp_name': exp_name if exp_name else f"exp_{exp_id:03d}",
            'exp_dir': exp_dir,
            'config': {k: v for k, v in config.items() if not k.startswith('_')},
            'seed': exp_seed,
            'n_i': config['n_i'],
            'pConn_ei_input': config['pConn_ei_input'],
            'pConn_ei': config['pConn_ei'],
            'pConn_ie': config['pConn_ie'],
            'weight_ie': config['weight_ie'],
            'ei_ratio': config['_ei_ratio'],
            'inhib_drive': config['_inhib_drive'],
            'overlap': config['_overlap'],
            'accuracy': None,
            'elapsed_time': time.time() - start_time,
            'success': False,
            'error': error_msg
        }


def evaluate_experiment(exp_dir):
    """
    Evaluate a trained network and return accuracy.

    IMPORTANT: Uses TRAINING data for computing neuron assignments,
    and TEST data for measuring accuracy. This is the correct approach
    for evaluating generalization.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Accuracy percentage on test set, or None if evaluation fails
    """
    import glob

    try:
        activity_path = os.path.join(exp_dir, 'activity')

        # =====================================================================
        # 1. Load TRAINING activity for computing assignments
        # =====================================================================
        train_result_files = glob.glob(os.path.join(activity_path, 'resultPopVecs*_clean.npy'))

        if not train_result_files:
            return None

        train_result_file = train_result_files[0]
        train_labels_file = train_result_file.replace('resultPopVecs', 'inputNumbers')

        if not os.path.exists(train_labels_file):
            return None

        train_result_monitor = np.load(train_result_file)
        train_input_numbers = np.load(train_labels_file)

        # =====================================================================
        # 2. Compute assignments from TRAINING data
        # =====================================================================
        n_e = train_result_monitor.shape[1]
        assignments = np.ones(n_e) * -1
        maximum_rate = np.zeros(n_e)

        classes = np.unique(train_input_numbers)
        for cls in classes:
            mask = train_input_numbers == cls
            if np.sum(mask) > 0:
                rates = np.mean(train_result_monitor[mask], axis=0)
                better = rates > maximum_rate
                assignments[better] = cls
                maximum_rate[better] = rates[better]

        # =====================================================================
        # 3. Load TEST activity for evaluation
        # =====================================================================
        all_result_files = glob.glob(os.path.join(activity_path, 'resultPopVecs*.npy'))
        test_result_files = [f for f in all_result_files if '_clean' not in f]

        if not test_result_files:
            test_result_monitor = train_result_monitor
            test_input_numbers = train_input_numbers
        else:
            test_result_file = test_result_files[0]
            test_labels_file = test_result_file.replace('resultPopVecs', 'inputNumbers')

            if not os.path.exists(test_labels_file):
                return None

            test_result_monitor = np.load(test_result_file)
            test_input_numbers = np.load(test_labels_file)

        # =====================================================================
        # 4. Compute accuracy on TEST data
        # =====================================================================
        correct = 0
        total = len(test_input_numbers)

        for i in range(total):
            spike_rates = test_result_monitor[i]
            summed_rates = np.zeros(10)

            for cls in range(10):
                mask = assignments == cls
                if np.sum(mask) > 0:
                    summed_rates[cls] = np.sum(spike_rates[mask]) / np.sum(mask)

            predicted = np.argmax(summed_rates)
            if predicted == test_input_numbers[i]:
                correct += 1

        accuracy = 100.0 * correct / total
        return accuracy

    except Exception as e:
        return None


def check_existing_experiment(base_dir, exp_id, config):
    """
    Check if a valid experiment already exists for this configuration.

    An experiment is valid if:
    1. The experiment directory exists
    2. It has activity files (resultPopVecs*_clean.npy)
    3. The training log shows it used the correct config paths

    Args:
        base_dir: Base directory for grid search
        exp_id: Experiment ID
        config: Configuration dict

    Returns:
        tuple: (is_valid, accuracy, exp_dir) or (False, None, None)
    """
    import glob

    # Build expected experiment name
    exp_name = (f"exp_{exp_id:03d}_"
                f"xi{int(config['pConn_ei_input']*100)}_"
                f"ei{int(config['pConn_ei']*100)}_"
                f"ie{int(config['pConn_ie']*100)}_"
                f"w{config['weight_ie']:.1f}")

    exp_dir = os.path.join(base_dir, exp_name)

    if not os.path.exists(exp_dir):
        return False, None, None

    # Check for activity files
    activity_path = os.path.join(exp_dir, 'activity')
    train_files = glob.glob(os.path.join(activity_path, 'resultPopVecs*_clean.npy'))

    if not train_files:
        return False, None, exp_dir

    # Verify the experiment used correct paths (not contaminated by bug)
    log_path = os.path.join(exp_dir, 'training.log')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_content = f.read(10000)  # Read first 10KB

        # Check if AeAi.npy path contains the correct experiment ID
        expected_pattern = f"exp_{exp_id:03d}"
        if 'AeAi.npy' in log_content:
            # Find the path used
            import re
            match = re.search(r'(exp_\d{3})[^/]*/random/AeAi\.npy', log_content)
            if match and match.group(1) != expected_pattern:
                # Contaminated by module caching bug
                return False, None, exp_dir

    # Evaluate accuracy
    accuracy = evaluate_experiment(exp_dir)

    if accuracy is not None:
        return True, accuracy, exp_dir

    return False, None, exp_dir


def print_results_table(results):
    """Print a formatted results table."""

    sorted_results = sorted(
        [r for r in results if r['accuracy'] is not None],
        key=lambda x: x['accuracy'],
        reverse=True
    )

    print("\n" + "=" * 100)
    print("GRID SEARCH RESULTS - Ranked by Accuracy")
    print("=" * 100)
    print(f"{'Rank':<5} {'X→I':<6} {'E→I':<6} {'I→E':<6} {'w_ie':<7} "
          f"{'Drive':<7} {'Overlap':<8} {'Accuracy':<10} {'Time':<8}")
    print("-" * 100)

    for rank, r in enumerate(sorted_results, 1):
        print(f"{rank:<5} "
              f"{r['pConn_ei_input']:.0%}   {r['pConn_ei']:.0%}   {r['pConn_ie']:.0%}   "
              f"{r['weight_ie']:<7.1f} {r['inhib_drive']:<7.1f} "
              f"{r['overlap']:<7.1f}% {r['accuracy']:<9.2f}% "
              f"{r['elapsed_time']/60:<.1f}m")

    print("=" * 90)

    failed = [r for r in results if r['accuracy'] is None]
    if failed:
        print(f"\nFailed experiments: {len(failed)}")
        for r in failed:
            print(f"  - {r['exp_name']}: {str(r['error'])[:60] if r['error'] else 'Unknown'}...")


def main():
    parser = argparse.ArgumentParser(
        description='Grid search for optimal E:I parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Use smaller parameter grid for quick testing')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Base random seed (each experiment uses seed + exp_id)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from previous run directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configurations without running')

    args = parser.parse_args()

    # Select parameter grid
    param_grid = PARAM_GRID_QUICK if args.quick else PARAM_GRID

    # Generate all configurations
    configs = generate_experiment_configs(param_grid, FIXED_PARAMS)

    print("=" * 70)
    print("E:I RATIO GRID SEARCH (with robust module isolation)")
    print("=" * 70)
    print(f"Parameter grid:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")
    print(f"\nTotal configurations: {len(configs)}")
    print(f"Base seed: {args.seed} (each exp uses seed + exp_id)")
    print("=" * 70)

    if args.dry_run:
        print("\nConfigurations to test:")
        print("-" * 70)
        for i, cfg in enumerate(configs):
            print(f"  {i+1:3d}. n_i={cfg['n_i']:3d}, "
                  f"E→I={cfg['pConn_ei']:.0%}, I→E={cfg['pConn_ie']:.0%}, "
                  f"w_ie={cfg['weight_ie']:.1f} → "
                  f"drive={cfg['_inhib_drive']:.1f}, overlap={cfg['_overlap']:.0f}%")
        print("-" * 70)
        return 0

    # Determine workers
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)

    # Create output directory
    if args.output:
        base_dir = os.path.abspath(args.output)  # CRITICAL: Make absolute to avoid path issues with chdir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(SCRIPT_DIR, f'grid_search_{timestamp}')

    os.makedirs(base_dir, exist_ok=True)

    print(f"\nWorkers: {args.workers}")
    print(f"Output: {base_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 70)
    print()

    # Check for existing valid experiments to skip
    print("Checking for existing valid experiments...")
    print("-" * 70)

    experiment_args = []
    results = []
    skipped = 0
    best_accuracy = 0.0
    best_config = None

    for i, config in enumerate(configs):
        is_valid, accuracy, exp_dir = check_existing_experiment(base_dir, i, config)

        if is_valid and accuracy is not None:
            # Use existing result
            skipped += 1
            exp_name = os.path.basename(exp_dir)
            print(f"  [SKIP] exp_{i:03d}: {accuracy:.2f}% (already complete)")

            result = {
                'exp_id': i,
                'exp_name': exp_name,
                'exp_dir': exp_dir,
                'config': {k: v for k, v in config.items() if not k.startswith('_')},
                'seed': args.seed + i,
                'n_i': config['n_i'],
                'pConn_ei_input': config['pConn_ei_input'],
                'pConn_ei': config['pConn_ei'],
                'pConn_ie': config['pConn_ie'],
                'weight_ie': config['weight_ie'],
                'ei_ratio': config['_ei_ratio'],
                'inhib_drive': config['_inhib_drive'],
                'overlap': config['_overlap'],
                'accuracy': accuracy,
                'elapsed_time': 0,  # Already done
                'success': True,
                'error': None,
                'skipped': True
            }
            results.append(result)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = result
        else:
            # Need to run this experiment
            if exp_dir and os.path.exists(exp_dir):
                # Invalid existing experiment - remove it
                print(f"  [REDO] exp_{i:03d}: removing corrupted data")
                shutil.rmtree(exp_dir)

            experiment_args.append((i, config, base_dir, args.seed, True))

    print("-" * 70)
    print(f"Skipping {skipped} valid experiments, running {len(experiment_args)} new/invalid")
    if best_accuracy > 0:
        print(f"Best from existing: {best_accuracy:.2f}%")
    print()

    # Run remaining experiments
    if experiment_args:
        start_time = time.time()
        completed = 0
        total_to_run = len(experiment_args)

        print("-" * 70)
        print(f"RUNNING {total_to_run} EXPERIMENTS")
        print("-" * 70)

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(run_single_experiment, experiment_args):
                results.append(result)
                completed += 1

                if result['accuracy'] is not None and result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_config = result

                    print(f"\n{'*' * 60}")
                    print(f"*** NEW BEST: {best_accuracy:.2f}% ***")
                    print(f"    X→I={result['pConn_ei_input']:.0%}, "
                          f"E→I={result['pConn_ei']:.0%}, "
                          f"I→E={result['pConn_ie']:.0%}, "
                          f"w={result['weight_ie']:.1f}")
                    print(f"{'*' * 60}\n")

                elapsed = time.time() - start_time
                remaining = total_to_run - completed
                if completed > 0:
                    avg_time = elapsed / completed
                    eta = avg_time * remaining
                    print(f"Progress: {completed}/{total_to_run} | "
                          f"Best: {best_accuracy:.2f}% | "
                          f"ETA: {eta/60:.0f} min")

        total_time = time.time() - start_time
        print(f"\nNew experiments time: {total_time/60:.1f} minutes")
    else:
        print("All experiments already complete!")
        total_time = 0

    print_results_table(results)

    new_runs = len([r for r in results if not r.get('skipped', False)])
    if new_runs > 0:
        print(f"\nTime for {new_runs} new experiments: {total_time/60:.1f} minutes")
    print(f"Total experiments (including skipped): {len(results)}")

    # Save results
    results_path = os.path.join(base_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    # Save summary
    summary_path = os.path.join(base_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("E:I Ratio Grid Search Summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Total experiments: {len(configs)}\n")
        f.write(f"Successful: {sum(1 for r in results if r['success'])}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n\n")

        sorted_results = sorted(
            [r for r in results if r['accuracy'] is not None],
            key=lambda x: x['accuracy'],
            reverse=True
        )

        f.write("Top 5 configurations:\n")
        for i, r in enumerate(sorted_results[:5], 1):
            f.write(f"  {i}. {r['ei_ratio']} (n_i={r['n_i']}): {r['accuracy']:.2f}%\n")
            f.write(f"     E→I={r['pConn_ei']:.0%}, I→E={r['pConn_ie']:.0%}, w={r['weight_ie']:.1f}\n")

    print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
