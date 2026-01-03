# Quick Start Guide

## Installation

Ensure you have all dependencies:
```bash
pip install numpy matplotlib scikit-learn torch torchvision tqdm networkx scikit-image seaborn pandas
```

## Basic Workflow

### 1. Train a Network
```bash
cd MNIST_GA_experiment

# Default 5-class network (original configuration)
python run_experiment.py

# Quick 2-class test
python run_experiment.py --arch tiny_2class

# Custom parameters
python run_experiment.py --arch small_2class --generations 200
```

### 2. Evaluate the Network
```bash
# Basic evaluation
python evaluate_network.py outputs/5class_mnist_ga/final_network

# Thorough evaluation with animation
python evaluate_network.py outputs/5class_mnist_ga/final_network --samples 5000 --animate
```

## Command Reference

### Training (`run_experiment.py`)

```bash
# List available architectures
python run_experiment.py --list-archs

# Use pre-defined architecture
python run_experiment.py --arch ARCH_NAME

# Override parameters
python run_experiment.py --name my_experiment
python run_experiment.py --generations 200
python run_experiment.py --population 150

# Combine options
python run_experiment.py --arch medium_3class --generations 300 --name long_run
```

### Evaluation (`evaluate_network.py`)

```bash
# Required: path to final_network directory
python evaluate_network.py PATH_TO_FINAL_NETWORK

# Options
--samples N          # Number of test samples (default: 3000)
--animate            # Generate activity animation GIF
--skip-eval          # Only generate visualizations, skip metrics
--output DIR         # Custom output directory
```

## Output Structure

After training:
```
outputs/
└── [experiment_name]/
    ├── plots/                              # GA evolution
    │   ├── gen_001.png ... gen_150.png
    │   └── final_fitness_evolution.png
    └── final_network/                      # Trained weights
        ├── best_snn_5class_weights.npy
        ├── best_snn_5class_connection_map.npy
        ├── best_snn_5class_delays.npy
        ├── best_snn_5class_inhibitory.npy
        ├── best_snn_5class_positions.npy
        └── best_snn_5class_config.json
```

After evaluation:
```
outputs/
└── [experiment_name]/
    └── evaluation/                         # Test results
        ├── evaluation_summary.png          # Accuracy + confusion matrix
        ├── weight_distribution_by_source.png
        ├── vis_input_digit_X_example.png
        ├── vis_digit_X_pred_Y_*_structure.png
        ├── vis_digit_X_pred_Y_*_raster.png
        ├── vis_digit_X_pred_Y_*_activity_psth.png
        ├── vis_digit_X_pred_Y_*_activity_heatmap.png
        └── vis_digit_X_pred_Y_*_animation.gif (if --animate)
```

## Common Use Cases

### Quick Debug Run
```bash
python run_experiment.py --arch debug
# Runs in ~5 minutes with 2 classes, tiny network
```

### Production 5-Class Run
```bash
python run_experiment.py --arch default
# Default configuration, ~6-8 hours on modern CPU
```

### Full 10-Class Experiment
```bash
python run_experiment.py --arch large_10class --name mnist_10class
# Challenging task, ~12-15 hours
```

### Custom Experiment
```python
from config.experiment_config import get_config

cfg = get_config(
    name="custom_3class",
    target_classes=[3, 7, 9],    # Classify 3, 7, 9
    hidden_layers=[60, 45, 30],   # Three hidden layers
    num_generations=200,
    population_size=120
)

# Then use cfg in your script
```

## Monitoring Progress

During training, watch:
- Console output for generation progress
- `outputs/[experiment]/plots/` for fitness curves
- Best fitness should increase over time
- If plateauing, consider more generations

## Expected Performance

| Classes | Architecture | Test Accuracy | Training Time* |
|---------|--------------|--------------|----------------|
| 2 | tiny_2class | ~85-90% | 30-60 min |
| 2 | small_2class | ~92-96% | 2-3 hours |
| 3 | medium_3class | ~85-92% | 4-5 hours |
| 5 | default | ~85-90% | 6-8 hours |
| 10 | large_10class | ~70-75% | 12-15 hours |

*On modern multi-core CPU (8+ cores)

## Troubleshooting

**Slow training?**
- Use fewer cores: Check `n_cores` in config
- Reduce `fitness_eval_examples`
- Use smaller architecture

**Low accuracy?**
- Increase `num_generations`
- Increase `population_size`
- Try different `hidden_layers` sizes
- Check if weights are in good range (`weight_min`, `weight_max`)

**Out of memory?**
- Reduce `population_size`
- Reduce network size (`hidden_layers`)
- Reduce `n_cores`

**CNN weights not found?**
- Switch to `encoding_mode='intensity_to_neuron'`
- Or train the CNN first (see MNIST_utils/)

## Tips

1. Start with `--arch debug` to verify setup
2. Use `--arch tiny_2class` for quick experiments
3. Monitor `plots/` directory during training
4. Save config JSON for reproducibility
5. Use `--animate` only when needed (slow)
6. Check evaluation accuracy before long runs

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- See [REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md) for architecture details
- Explore [config/network_architectures.py](config/network_architectures.py) for all architectures
- Check [config/experiment_config.py](config/experiment_config.py) for all parameters
