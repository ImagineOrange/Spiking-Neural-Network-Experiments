# 4:1 Excitatory:Inhibitory Ratio Experiment

This directory implements the **biologically plausible 4:1 E:I ratio** experiment suggested in Diehl & Cook (2015). It extends the original implementation (found in `../diehl_2015_migration/`) with calibrated inhibitory drive based on cortical connectivity data.

> **Relationship to Original**: The `diehl_2015_migration/` directory contains a faithful Python3/Brian2 port of the original Diehl & Cook (2015) paper with a 1:1 E:I ratio. This directory modifies that implementation to explore the biologically realistic 4:1 ratio mentioned in Section 2.1 of the paper.

---

## Core Differences from Original Implementation

The table below summarizes the key architectural and parametric differences:

| Aspect | Original (`diehl_2015_migration/`) | This Implementation (v2) |
|--------|-----------------------------------|---------------------|
| **E:I Neuron Ratio** | 1:1 (400:400) | **4:1 (400:100)** |
| **Ae→Ai Connectivity** | One-to-one diagonal (400 connections) | **5% sparse** (local competition) |
| **Ai→Ae Connectivity** | 99.75% (all-to-all except self) | **30% sparse** (local inhibition) |
| **Ai→Ae Weight** | 17.0 | **10.0** (calibrated for ~15 drive) |
| **Inhibition Pattern** | Homogeneous (global, symmetric) | **Local** (sparse, ~83% overlap) |
| **Interface** | Edit `config.py` manually | **Unified `run.py` CLI** |
| **Membrane Noise** | None | **0.3 mV** (stochastic) |
| **Default Mode** | Test (pretrained weights) | **Train (from scratch)** |
| **Training Epochs** | 3 | **5** |

### Why These Changes Matter

1. **4:1 E:I Ratio**: Cortical E:I ratios range from 4:1 to 6:1 in mammalian cortex (Tremblay et al., 2016). The original 1:1 ratio was a computational simplification.

2. **Sparse E→I Connectivity (5%)**: Creates local inhibitory neighborhoods. Each E neuron activates only ~5 I neurons, limiting the spread of inhibition.

3. **Sparse I→E Connectivity (30%)**: Each I neuron inhibits only ~120 E neurons (30% of 400), creating local competition rather than global suppression. This is critical for allowing weak stimuli to compete.

4. **Local Inhibition**: The combination of 5% E→I and 30% I→E creates ~83% inhibition overlap between neuron pairs, compared to ~100% in both the original 1:1 and v1 4:1 implementation. See "Experimental Notes" at the end for details.

5. **Membrane Noise (0.3 mV)**: Real cortical neurons experience constant membrane potential fluctuations (σ = 2-6 mV in vivo) from synaptic bombardment and ion channel stochasticity (Destexhe et al., 2003). This noise enables stochastic resonance—subthreshold inputs can occasionally trigger spikes—and prevents deterministic "deadlock" states where the same neurons always win.

### Inhibitory Drive Calibration

A critical challenge when implementing 4:1 ratio is maintaining appropriate inhibitory drive while avoiding global suppression:

| Parameter | Original 1:1 | v1 4:1 (Failed) | v2 4:1 (Current) |
|-----------|-------------|-----------------|------------------|
| E→I connectivity | One-to-one | 12% | **5%** |
| I neurons per E spike | 1 | ~12 | **~5** |
| I→E connectivity | 99.75% | 90% | **30%** |
| I→E weight | 17.0 | 3.0 | **10.0** |
| Expected I hits per E | 1.0 | 10.8 | **1.5** |
| **Drive per target E** | 17.0 | 32.4 | **15.0** |
| **Pairwise overlap** | ~100% | ~100% | **~83%** |

The v1 4:1 configuration achieved 82.78% accuracy but completely failed on digit "1" (only 6 neurons assigned). Analysis revealed the problem wasn't drive strength (v1 had 32.4, well above original's 17.0) but rather **100% inhibition overlap**—every spike inhibited all other neurons equally.

v2 reduces overlap from 100% to 83% while keeping drive close to original (15.0 vs 17.0), allowing weak stimuli to compete.

---

## Quick Start

```bash
# 1. Generate random weights for 4:1 architecture
python Diehl&Cook_MNIST_random_conn_generator.py

# 2. Train the network (3 epochs, ~1.5-2 hours)
python run.py train

# 3. Test with trained weights (~15-20 min)
python run.py test

# 4. Evaluate accuracy
python run.py evaluate

# Or validate config without running
python run.py validate
```

---

## Grid Search

For systematic exploration of E:I connectivity parameters, see [GRID_SEARCH_README.md](GRID_SEARCH_README.md).

```bash
# Preview configurations
python grid_search_ei.py --dry-run

# Run full search
python grid_search_ei.py --workers 6
```

---

## File Structure

```
diehl_2015_research_additions/
├── README.md                          # This file
├── GRID_SEARCH_README.md              # Grid search documentation
├── run.py                             # Unified CLI for train/test/evaluate
├── config.py                          # Network configuration
├── grid_search_ei.py                  # Parallel grid search script
├── Diehl&Cook_spiking_MNIST_4to1.py   # Main simulation (modified for 4:1)
├── Diehl&Cook_MNIST_random_conn_generator.py  # Weight initialization
├── Diehl&Cook_MNIST_evaluation.py     # Evaluation script (symlink)
├── sim_and_eval_utils/                # Shared utilities
│   ├── data_loader.py                 # MNIST data loading
│   └── ...
└── mnist_data/                        # Data and experiment outputs
    ├── raw/                           # MNIST binary files
    ├── random/                        # Initial random weights
    ├── weights/                       # Trained weights
    └── activity/                      # Spike recordings
```

---

## References

- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience*, 9, 99.
- Tremblay, R., Lee, S., & Bhattacharjee, A. (2016). GABAergic interneurons in the neocortex: from cellular properties to circuits. *Neuron*, 91(3), 521-536.
- Packer, A. M., & Bhattacharjee, A. (2011). Dense, unspecific connectivity of neocortical parvalbumin-positive interneurons. *Nature Neuroscience*, 14(4), 471-478.
- Destexhe, A., Rudolph, M., & Bhattacharjee, A. (2003). The high-conductance state of neocortical neurons in vivo. *Nature Reviews Neuroscience*, 4(9), 739-751.
