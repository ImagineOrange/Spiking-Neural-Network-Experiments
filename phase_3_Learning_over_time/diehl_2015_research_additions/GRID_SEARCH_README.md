# E:I Connectivity Grid Search

Systematic exploration of excitatory-inhibitory connectivity parameters for the 4:1 architecture.

## Quick Start

```bash
# Preview all configurations
python grid_search_ei.py --dry-run

# Run full grid search (recommended for 24GB RAM)
python grid_search_ei.py --workers 6

# Quick test (4 configs instead of 48)
python grid_search_ei.py --workers 6 --quick
```

## Parameter Space

**Fixed Architecture:** 400 E : 100 I neurons (4:1 ratio)

| Parameter | Values | Description |
|-----------|--------|-------------|
| X→I (`pConn_ei_input`) | 5%, 10%, 15% | Input→Inhibitory connectivity (feedforward inhibition) |
| E→I (`pConn_ei`) | 3%, 5%, 8%, 12% | Excitatory→Inhibitory connectivity |
| I→E (`pConn_ie`) | 20%, 30%, 45%, 60% | Inhibitory→Excitatory connectivity |
| w_ie | Auto-calibrated | I→E weight adjusted to maintain ~17.0 inhibitory drive |

**Total Configurations:** 3 × 4 × 4 = 48

## Stability Predictions

### Likely Stable Regimes (60-90% overlap)

| E→I | I→E | Overlap | Rationale |
|-----|-----|---------|-----------|
| 3% | 30% | ~59% | Very local competition |
| 5% | 30% | ~83% | Current baseline (~83% accuracy) |
| 5% | 45% | ~92% | Stronger competition, may help selectivity |
| 8% | 20% | ~67% | More I activation, narrow spread |

### Likely Unstable Regimes (>95% overlap)

| E→I | I→E | Overlap | Problem |
|-----|-----|---------|---------|
| 12% | 45% | ~99% | Near-global inhibition |
| 12% | 60% | ~99.9% | Winner-take-all too extreme |
| 8% | 60% | ~99% | Weak stimuli get crushed |

### Potentially Weak Regimes (<50% overlap)

| E→I | I→E | Overlap | Problem |
|-----|-----|---------|---------|
| 3% | 20% | ~27% | Insufficient competition |

## Key Metrics

**Inhibitory Drive:** Total inhibition received by an E neuron per spike
- Formula: `(n_i × pConn_ei) × pConn_ie × weight_ie`
- Target: ~17.0 (matched to original 1:1 network)
- All configs auto-calibrated to this target

**Inhibition Overlap:** Probability two E neurons share inhibitory input
- Formula: `1 - (1 - pConn_ie)^(n_i × pConn_ei)`
- Sweet spot: 60-90%
- Too high (>95%): global suppression, weak digits fail
- Too low (<40%): poor selectivity, neurons don't specialize

## Output Structure

```
grid_search_YYYYMMDD_HHMMSS/
├── results.json          # Full results with all metrics
├── summary.txt           # Human-readable top performers
├── exp_000_xi10_ei5_ie30_w11.3/
│   ├── params.json       # Experiment parameters
│   ├── config.py         # Auto-generated config
│   ├── training.log      # stdout/stderr from training
│   ├── random/           # Initial random weights
│   ├── weights/          # Trained weights (XeAe.npy, theta_A.npy)
│   └── activity/         # Spike recordings for evaluation
├── exp_001_xi10_ei5_ie45_w7.6/
│   └── ...
└── ...
```

## Expected Runtime

- **Per experiment:** ~70-90 minutes (3 epochs, 31,035 examples)
- **Full grid search (6 workers):** ~9-12 hours
- **Quick mode (6 workers):** ~30-45 minutes

## Interpreting Results

The grid search ranks configurations by classification accuracy. Key columns:

| Column | Meaning |
|--------|---------|
| X→I | Input→Inhibitory connectivity |
| E→I | Excitatory→Inhibitory connectivity |
| I→E | Inhibitory→Excitatory connectivity |
| w_ie | Auto-calibrated I→E weight |
| Drive | Total inhibitory drive (~17.0 target) |
| Overlap | Pairwise inhibition overlap (%) |
| Accuracy | Classification accuracy (%) |

## Files in This Directory

| File | Purpose |
|------|---------|
| `grid_search_ei.py` | Main grid search script |
| `run.py` | Single-experiment runner (train/test/evaluate) |
| `config.py` | Configuration for single experiments |
| `Diehl&Cook_spiking_MNIST_4to1.py` | Core SNN simulation |
| `Diehl&Cook_MNIST_random_conn_generator.py` | Weight initialization |
| `Diehl&Cook_MNIST_evaluation.py` | Accuracy evaluation (symlink) |

## References

- Diehl & Cook (2015). Unsupervised learning of digit recognition using STDP.
- Tremblay et al. (2016). GABAergic Interneurons in the Neocortex: E:I ratios.
- Packer & Yuste (2011). Dense connectivity of neocortical PV+ interneurons.
