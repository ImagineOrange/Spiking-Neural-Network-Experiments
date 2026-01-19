#!/usr/bin/env python
"""
Test script wrapper for 4:1 E:I ratio experiment
Uses the local Diehl&Cook_spiking_MNIST_4to1.py which properly handles
Input→Inhibitory connections with fixed weights (no STDP).
"""

import sys
import os

# Get the directory where this script lives (sim_and_eval_utils/)
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
# Parent is diehl_2015_research_additions/
SCRIPT_DIR = os.path.dirname(UTILS_DIR)

# Ensure experiment directory is first in path for config import
sys.path.insert(0, SCRIPT_DIR)

# Temporarily modify config.py to force test mode
config_path = os.path.join(SCRIPT_DIR, 'config.py')
with open(config_path, 'r') as f:
    config_content = f.read()

original_test_mode_line = None
if 'self.test_mode = False' in config_content:
    original_test_mode_line = 'self.test_mode = False'
    config_content_modified = config_content.replace('self.test_mode = False', 'self.test_mode = True')

    # Write modified config temporarily
    with open(config_path, 'w') as f:
        f.write(config_content_modified)

    print("\n" + "="*60)
    print("Temporarily setting test_mode = True for this run")
    print("="*60 + "\n")

try:
    # Now import and run the LOCAL 4:1 simulation script
    script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_spiking_MNIST_4to1.py')
    exec(open(script_path).read())
finally:
    # Restore original config
    if original_test_mode_line:
        with open(config_path, 'w') as f:
            f.write(config_content)
