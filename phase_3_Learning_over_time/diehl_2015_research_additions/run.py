#!/usr/bin/env python
"""
Unified run script for 4:1 E:I ratio experiment

Usage:
    python run.py train     # Train the network (STDP on)
    python run.py test      # Test with trained weights (STDP off)
    python run.py evaluate  # Compute accuracy from saved activity
    python run.py eval      # Alias for evaluate
    python run.py validate  # Validate config without running simulation

This script uses the local Diehl&Cook_spiking_MNIST_4to1.py which has been
modified to properly handle Input→Inhibitory connections (ei_input) with
fixed weights while applying STDP only to Input→Excitatory (ee_input).
"""

import sys
import os

# Get the directory where this script lives (diehl_2015_research_additions/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Migration directory for evaluation script
MIGRATION_DIR = os.path.join(SCRIPT_DIR, '..', 'diehl_2015_migration')

# Ensure experiment directory is first in path for config import
sys.path.insert(0, SCRIPT_DIR)


def validate_config():
    """Validate the config is correct for 4:1 architecture"""
    from config import Config
    cfg = Config()

    print("\n" + "="*60)
    print("CONFIG VALIDATION")
    print("="*60)

    errors = []
    warnings = []

    # Check architecture
    print(f"\nArchitecture:")
    print(f"  n_input = {cfg.n_input}")
    print(f"  n_e = {cfg.n_e}")
    print(f"  n_i = {cfg.n_i}")
    print(f"  E:I ratio = {cfg.n_e}:{cfg.n_i} = {cfg.n_e/cfg.n_i:.1f}:1")

    if cfg.n_e != 400:
        errors.append(f"n_e should be 400, got {cfg.n_e}")
    if cfg.n_i != 100:
        errors.append(f"n_i should be 100 for 4:1 ratio, got {cfg.n_i}")

    # Check input connections
    print(f"\nInput connections:")
    print(f"  input_conn_names = {cfg.input_conn_names}")

    if 'ee_input' not in cfg.input_conn_names:
        errors.append("'ee_input' missing from input_conn_names!")
    if 'ei_input' not in cfg.input_conn_names:
        errors.append("'ei_input' missing from input_conn_names! (Input→Inhibitory won't be loaded)")

    # Check mode settings
    print(f"\nMode settings:")
    print(f"  test_mode = {cfg.test_mode}")
    print(f"  ee_STDP_on = {cfg.ee_STDP_on}")

    # Check paths
    print(f"\nPaths:")
    print(f"  data_path = {cfg.data_path}")
    print(f"  weight_path = {cfg.weight_path}")

    # Check weight files exist
    print(f"\nWeight files:")
    random_path = os.path.join(SCRIPT_DIR, cfg.data_path, 'random')
    weights_path = os.path.join(SCRIPT_DIR, cfg.data_path, 'weights')

    required_random = ['XeAe.npy', 'XeAi.npy', 'AeAi.npy', 'AiAe.npy']
    for fname in required_random:
        fpath = os.path.join(random_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"  [OK] random/{fname} ({size/1024:.1f} KB)")
        else:
            errors.append(f"Missing random weight file: {fname}")
            print(f"  [MISSING] random/{fname}")

    # Check for trained weights (only required for test mode)
    if cfg.test_mode:
        trained_files = ['XeAe.npy', 'theta_A.npy']
        for fname in trained_files:
            fpath = os.path.join(weights_path, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                print(f"  [OK] weights/{fname} ({size/1024:.1f} KB)")
            else:
                errors.append(f"Missing trained weight file: {fname} (required for test mode)")
                print(f"  [MISSING] weights/{fname}")

    # Summary
    print("\n" + "="*60)
    if errors:
        print("VALIDATION FAILED")
        print("="*60)
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print("VALIDATION PASSED")
        print("="*60)
        print("  All checks passed. Ready to run.")
        return True


def set_test_mode(enable):
    """Temporarily modify config.py to set test_mode"""
    config_path = os.path.join(SCRIPT_DIR, 'config.py')

    with open(config_path, 'r') as f:
        content = f.read()

    if enable:
        # Set test_mode = True
        if 'self.test_mode = False' in content:
            content = content.replace('self.test_mode = False', 'self.test_mode = True')
            with open(config_path, 'w') as f:
                f.write(content)
            return 'self.test_mode = False'  # Return original for restoration
    else:
        # Set test_mode = False
        if 'self.test_mode = True' in content:
            content = content.replace('self.test_mode = True', 'self.test_mode = False')
            with open(config_path, 'w') as f:
                f.write(content)
            return 'self.test_mode = True'  # Return original for restoration

    return None  # No change needed


def restore_config(original_line):
    """Restore config.py to original state"""
    if original_line is None:
        return

    config_path = os.path.join(SCRIPT_DIR, 'config.py')

    with open(config_path, 'r') as f:
        content = f.read()

    if original_line == 'self.test_mode = False':
        content = content.replace('self.test_mode = True', 'self.test_mode = False')
    else:
        content = content.replace('self.test_mode = False', 'self.test_mode = True')

    with open(config_path, 'w') as f:
        f.write(content)


def run_train():
    """Run training mode"""
    print("\n" + "="*60)
    print("4:1 E:I RATIO EXPERIMENT - TRAINING MODE")
    print("="*60)

    # Ensure test_mode = False
    original = set_test_mode(False)

    try:
        # Validate before running
        if not validate_config():
            print("\nFix the errors above before training.")
            return False

        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")

        # Run the LOCAL 4:1 simulation script (handles ei_input correctly)
        script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_spiking_MNIST_4to1.py')
        exec(open(script_path).read(), {'__name__': '__main__'})
        return True

    finally:
        restore_config(original)


def run_test():
    """Run test mode"""
    print("\n" + "="*60)
    print("4:1 E:I RATIO EXPERIMENT - TEST MODE")
    print("="*60)

    # Ensure test_mode = True
    original = set_test_mode(True)

    try:
        # Validate before running
        if not validate_config():
            print("\nFix the errors above before testing.")
            return False

        print("\n" + "="*60)
        print("Starting testing...")
        print("="*60 + "\n")

        # Run the LOCAL 4:1 simulation script (handles ei_input correctly)
        script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_spiking_MNIST_4to1.py')
        exec(open(script_path).read(), {'__name__': '__main__'})
        return True

    finally:
        restore_config(original)


def run_evaluate():
    """Run evaluation"""
    print("\n" + "="*60)
    print("4:1 E:I RATIO EXPERIMENT - EVALUATION")
    print("="*60)

    # Check for activity files
    activity_path = os.path.join(SCRIPT_DIR, 'mnist_data', 'activity')
    if not os.path.exists(activity_path):
        print(f"\nERROR: Activity directory not found: {activity_path}")
        print("Please run training first.")
        return False

    import glob
    activity_files = glob.glob(os.path.join(activity_path, 'resultPopVecs*.npy'))
    if not activity_files:
        print(f"\nERROR: No activity files found in {activity_path}")
        print("Please run training first.")
        return False

    print(f"\nFound {len(activity_files)} activity file(s)")

    # Run the LOCAL evaluation script (with specificity analysis)
    script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_MNIST_evaluation.py')
    exec(open(script_path).read(), {'__name__': '__main__'})
    return True


def print_usage():
    """Print usage information"""
    print(__doc__)
    print("\nExamples:")
    print("  python run.py train      # Train for 3 epochs (~1.5-2 hours)")
    print("  python run.py test       # Test with trained weights (~15-20 min)")
    print("  python run.py evaluate   # Compute and print accuracy")
    print("  python run.py validate   # Check config without running")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'train':
        run_train()
    elif command == 'test':
        run_test()
    elif command in ['evaluate', 'eval']:
        run_evaluate()
    elif command == 'validate':
        validate_config()
    elif command in ['help', '-h', '--help']:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == '__main__':
    main()
