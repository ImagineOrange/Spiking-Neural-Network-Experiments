# MNIST GA SNN Experiment

Train Spiking Neural Networks (SNNs) to classify MNIST digits using Genetic Algorithms (GA).

This experiment uses evolutionary optimization instead of backpropagation to find optimal synaptic weights for a biologically-inspired SNN classifier.

## What This Does

1. **Creates a fixed SNN structure** with realistic neuron dynamics (LIF model)
2. **Encodes MNIST digits** into spike trains (temporal patterns)
3. **Evolves synaptic weights** using a genetic algorithm
4. **Evaluates fitness** via balanced accuracy on random subsets
5. **Saves the best network** for later evaluation

## Quick Start

### Basic Usage

```bash
cd MNIST_GA_experiment
python run_experiment.py
```

This runs the default 5-class MNIST classifier with the original pre-refactor configuration (100 population, 150 generations, [40, 30] hidden layers).

### Using Pre-defined Architectures

```bash
# List available architectures
python run_experiment.py --list-archs

# Run with a specific architecture
python run_experiment.py --arch small_2class

# Quick debug run
python run_experiment.py --arch debug
```

### Custom Configuration

```bash
# Override specific parameters
python run_experiment.py --name my_experiment --generations 200 --population 150
```

### Programmatic Usage

```python
from config.experiment_config import get_config
from config.network_architectures import get_architecture

# Use pre-defined architecture
arch = get_architecture('standard_5class')
cfg = get_config(**arch)

# Or create custom config
cfg = get_config(
    name="custom_experiment",
    target_classes=[0, 1, 2],
    hidden_layers=[30, 20],
    num_generations=100
)
```

## Project Structure

```
MNIST_GA_experiment/
├── README.md                          # This file
├── config/
│   ├── experiment_config.py           # Configuration dataclass
│   └── network_architectures.py       # Pre-defined architectures
├── mnist_ga/
│   ├── network_builder.py             # SNN structure creation
│   ├── encoding.py                    # MNIST → spike trains
│   ├── simulation.py                  # SNN simulation
│   ├── evaluation.py                  # Fitness & prediction
│   └── visualization.py               # Progress plotting
├── run_experiment.py                  # Main entry point (training)
├── evaluate_network.py                # Evaluation & visualization
└── outputs/                           # Generated during runs
    └── [experiment_name]/
        ├── plots/                     # GA evolution plots
        ├── final_network/             # Best weights & config
        └── evaluation/                # Test metrics & visualizations
```

## How It Works

### 1. Network Creation

Creates a **fixed topology** with:
- **Layered architecture**: Input → Hidden(s) → Output
- **Distance-dependent delays**: Realistic spike propagation
- **Hardcoded output inhibition**: Winner-take-all dynamics
- **Mixed excitatory/inhibitory**: ~20% inhibitory neurons

### 2. Spike Encoding

Converts MNIST images to spike trains:

**Intensity-to-Neuron Mode** (default):
- Downsample 28×28 → 7×7 (49 pixels)
- Pixel intensity → firing rate (0-200Hz)
- Poisson spike generation

**Conv-Feature-to-Neuron Mode**:
- Pre-trained CNN extracts 49 features
- Feature activation → firing rate
- More abstract representations

### 3. Genetic Algorithm Evolution

```
Generation 1:
  ├─ Create 100 random weight vectors
  ├─ Simulate each on 1000 MNIST examples
  ├─ Calculate fitness (balanced accuracy)
  ├─ Select best (tournament selection)
  ├─ Crossover & mutate → new generation
  └─ Repeat...

Generation 150:
  └─ Return best weights found
```

### 4. Classification

Winner-take-all: Output neuron with MOST spikes = prediction

```python
# Example output neuron spike counts:
{
    neuron_0: 3,    # Represents digit "0"
    neuron_1: 1,    # Represents digit "1"
    neuron_2: 15,   # Represents digit "2" ← WINNER!
    neuron_3: 0,    # Represents digit "3"
    neuron_4: 2     # Represents digit "4"
}
# Predicted class: 2
```

## Available Architectures

| Name | Classes | Hidden Layers | Pop | Gens | Use Case |
|------|---------|---------------|-----|------|----------|
| **`default`** | [0-4] | [40, 30] | 100 | 150 | **Default** original config |
| `tiny_2class` | [0,1] | [15, 10] | 30 | 50 | Quick testing |
| `small_2class` | [0,1] | [20, 15] | 50 | 100 | 2-digit classification |
| `medium_3class` | [0,1,2] | [30, 25] | 75 | 120 | 3-digit classification |
| `standard_5class` | [0-4] | [40, 30] | 100 | 150 | Standard 5-digit |
| `deep_5class` | [0-4] | [50,40,30,20] | 120 | 180 | Deeper network |
| `large_10class` | [0-9] | [80,60,40] | 150 | 200 | Full 10-digit |
| `conv_features_5class` | [0-4] | [40, 30] | 100 | 150 | Uses CNN features |
| `debug` | [0,1] | [10, 5] | 10 | 5 | Ultra-fast debugging |


## Configuration Options

### Key Parameters

**Network Architecture:**
- `target_classes`: Which digits to classify (e.g., `[0,1,2,3,4]`)
- `hidden_layers`: Hidden layer sizes (e.g., `[40, 30]`)
- `inhibitory_fraction`: Fraction of inhibitory neurons (default: 0.2)

**Encoding:**
- `encoding_mode`: `'intensity_to_neuron'` or `'conv_feature_to_neuron'`
- `downsample_factor`: Downsampling for intensity mode (default: 4)
- `max_freq_hz`: Maximum firing rate (default: 200Hz)

**GA Parameters:**
- `population_size`: Number of individuals (default: 100)
- `num_generations`: Generations to evolve (default: 150)
- `mutation_rate`: Probability of mutation (default: 0.05)
- `crossover_rate`: Probability of crossover (default: 0.7)
- `fitness_eval_examples`: Examples per fitness evaluation (default: 1000)

**Simulation:**
- `sim_duration_ms`: Total simulation time per example (default: 70ms)
- `mnist_stim_duration_ms`: Stimulus presentation time (default: 50ms)
- `sim_dt_ms`: Simulation timestep (default: 0.1ms)

## Monitoring Progress

During evolution, plots are saved to `outputs/[experiment]/plots/`:
- `gen_001.png`, `gen_002.png`, ... - Progress per generation
- `final_fitness_evolution.png` - Complete evolution curve

Watch for:
- **Best fitness** (cyan line) - Should increase over time
- **Average fitness** (orange line) - Population-wide performance
- **Plateaus** - May indicate convergence or need for more generations

## Saved Outputs

After evolution, find results in `outputs/[experiment]/final_network/`:

- `best_snn_Xclass_weights.npy` - Evolved synaptic weights
- `best_snn_Xclass_connection_map.npy` - Network topology
- `best_snn_Xclass_delays.npy` - Transmission delays
- `best_snn_Xclass_inhibitory.npy` - Inhibitory neuron flags
- `best_snn_Xclass_positions.npy` - Spatial positions
- `best_snn_Xclass_config.json` - Full configuration

## Evaluating Trained Networks

After training, evaluate your network on the test set with comprehensive visualizations:

```bash
# Basic evaluation (3000 test samples)
python evaluate_network.py outputs/5class_mnist_ga/final_network

# More thorough evaluation with animation
python evaluate_network.py outputs/5class_mnist_ga/final_network --samples 5000 --animate

# Skip evaluation, only generate visualizations
python evaluate_network.py outputs/5class_mnist_ga/final_network --skip-eval
```

This generates in `outputs/[experiment]/evaluation/`:
- Confusion matrix and accuracy metrics
- Network structure visualization
- Weight distribution analysis
- Activity rasters, PSTHs, and heatmaps
- Distance-dependence plots
- Single example deep dive
- Optional: Activity animation GIF

## Advanced Usage

### Custom Network Architecture

```python
from config.experiment_config import get_config

cfg = get_config()
cfg.target_classes = [3, 7, 9]  # Classify just 3, 7, and 9
cfg.hidden_layers = [60, 45, 30]  # Three hidden layers
cfg.num_generations = 200
cfg.population_size = 120
```

### Different Encoding Modes

```python
# Use CNN feature encoding
cfg.encoding_mode = 'conv_feature_to_neuron'
cfg.conv_weights_path = '../MNIST_utils/conv_model_weights/conv_model_weights.pth'
```

### Adjust GA Behavior

```python
# More exploration
cfg.mutation_rate = 0.1  # Higher mutation
cfg.mutation_strength = 0.02  # Larger mutations

# More exploitation
cfg.elitism_count = 5  # Keep more top individuals
cfg.tournament_size = 5  # Stronger selection pressure
```

## Troubleshooting

**"CNN weights not found" error:**
- Switch to `encoding_mode = 'intensity_to_neuron'`, or
- Train the CNN first using the appropriate script

**Slow evolution:**
- Use a smaller architecture (e.g., `--arch small_2class`)
- Reduce `fitness_eval_examples` (trades speed for accuracy)
- Ensure multi-core is working (check `n_cores` in config)

**Low accuracy:**
- Increase `num_generations` (more evolution time)
- Increase `population_size` (more diversity)
- Try different `hidden_layers` architectures
- Check `weight_min` and `weight_max` bounds

**Memory issues:**
- Reduce `population_size`
- Use smaller network (`hidden_layers`)
- Reduce `n_cores` (less parallelization)

## References

This implementation is based on:
- **LIF Neurons**: Leaky Integrate-and-Fire model with adaptation
- **Genetic Algorithms**: Tournament selection, elitism, crossover
- **Winner-Take-All**: Mutual inhibition between output neurons
- **Balanced Accuracy**: Fair metric for potentially imbalanced classes

## Related Files

- `../MNIST_GA_experiment.py` - Original monolithic implementation (kept for reference)
- `../LIF_objects/LayeredNeuronalNetworkVectorized.py` - Vectorized SNN implementation
- `../LIF_objects/GeneticAlgorithm.py` - GA implementation
- `../MNIST_utils/` - MNIST loading and encoding utilities

