#!/usr/bin/env python
"""
Training script wrapper for 4:1 E:I ratio experiment
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

# Now import and run the LOCAL 4:1 simulation script
if __name__ == '__main__':
    script_path = os.path.join(SCRIPT_DIR, 'Diehl&Cook_spiking_MNIST_4to1.py')
    exec(open(script_path).read())
