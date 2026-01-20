# Spherical Neural Network Simulation

A biologically plausible 3D spiking neural network simulation featuring conductance-based Leaky Integrate-and-Fire (LIF) neurons arranged in a spherical volume with distance-dependent connectivity.

---

## Table of Contents

1. [Overview](#overview)
2. [Network Topology](#network-topology)
   - [3D Spatial Arrangement](#3d-spatial-arrangement)
   - [Sphere Radius Calculation](#sphere-radius-calculation)
   - [Layout Types](#layout-types)
3. [Connectivity Architecture](#connectivity-architecture)
   - [Per-Type Connection Probabilities](#per-type-connection-probabilities)
   - [Distance-Dependent Weight Decay](#distance-dependent-weight-decay)
   - [Dual Lambda System](#dual-lambda-system)
   - [Transmission Delays](#transmission-delays)
   - [Weight Assignment](#weight-assignment)
   - [Zero-Weight Pruning](#zero-weight-pruning)
4. [Neuron Model](#neuron-model)
   - [Membrane Dynamics](#membrane-dynamics)
   - [Conductance-Based Synapses](#conductance-based-synapses)
   - [Spike-Frequency Adaptation](#spike-frequency-adaptation)
5. [Parameter Heterogeneity (Jitter)](#parameter-heterogeneity-jitter)
   - [Why Heterogeneity Matters](#why-heterogeneity-matters)
   - [Gaussian Jitter (Voltage Parameters)](#gaussian-jitter-voltage-parameters)
   - [Log-Normal Jitter (Time Constants)](#log-normal-jitter-time-constants)
   - [Biologically Plausible Clipping](#biologically-plausible-clipping)
6. [Network-Level Noise](#network-level-noise)
7. [E/I Balance Mechanisms](#ei-balance-mechanisms)
8. [Stimulation Modes](#stimulation-modes)
9. [Current Experiment Configuration](#current-experiment-configuration)
10. [Output Files](#output-files)
11. [Usage Examples](#usage-examples)
12. [References](#references)

---

## Overview

This simulation models a network of excitatory and inhibitory neurons positioned within a 3D sphere. The architecture implements several biologically motivated design principles:

- **Spatial embedding**: Neurons exist in 3D space with realistic distance-dependent connectivity
- **Cell-type specificity**: Different connection probabilities for E→E, E→I, I→E, and I→I pathways
- **Conductance-based synapses**: Reversal potentials provide automatic gain control and shunting inhibition
- **Spike-frequency adaptation**: Prevents runaway excitation via calcium-activated potassium channels
- **Parameter heterogeneity**: Realistic variability across neurons prevents artificial synchronization
- **Transmission delays**: Distance-dependent axonal conduction delays

---

## Network Topology

### 3D Spatial Arrangement

Neurons are positioned within a 3D spherical volume. This spatial embedding serves multiple purposes:

1. **Biological plausibility**: Cortical tissue is inherently 3D
2. **Distance-dependent connectivity**: Nearby neurons connect more strongly than distant ones
3. **Realistic propagation**: Transmission delays scale with physical distance
4. **Visualization**: 3D animations reveal spatial patterns of activity propagation

The network uses a **NetworkX DiGraph** to represent connectivity, with nodes storing 3D coordinates and edges storing weights, delays, and distance information.

### Sphere Radius Calculation

The sphere radius is calculated to maintain consistent neuron density regardless of network size:

```
radius = (3n / 4π)^(1/3) × 1.5
```

| n_neurons | Calculated Radius |
|-----------|-------------------|
| 1,000     | 9.3 units         |
| 3,000     | 13.4 units        |
| 6,000     | 16.9 units        |
| 10,000    | 20.0 units        |

The 1.5 scaling factor provides comfortable spacing between neurons. For the current 6000-neuron configuration, the sphere has a radius of approximately **16.9 units** and diameter of **33.8 units**.

### Layout Types

| Layout | Method | Use Case |
|--------|--------|----------|
| `'sphere'` (default) | Rejection sampling in sphere volume | Uniform 3D distribution, models cortical column |
| `'sphere-surface'` | Fibonacci spiral on sphere surface | 2D sheet wrapped on sphere, models cortical surface |

**Rejection sampling algorithm** (for volume distribution):
```python
while len(positions) < n_neurons:
    x, y, z = uniform(-radius, radius)
    if x² + y² + z² ≤ radius²:
        positions.append((x, y, z))
```

---

## Connectivity Architecture

### Per-Type Connection Probabilities

The network implements **cell-type-specific connection probabilities**, reflecting the distinct connectivity patterns observed in cortical microcircuits:

| Connection Type | Code Key | Current Value | Biological Range | Description |
|-----------------|----------|---------------|------------------|-------------|
| **E→E** | `ee` | 0.10 | 0.05–0.15 | Excitatory recurrence; drives persistent activity |
| **E→I** | `ei` | 0.15 | 0.15–0.25 | Feedforward inhibition; recruits interneurons |
| **I→E** | `ie` | 0.25 | 0.25–0.60 | Blanket inhibition; controls excitatory gain |
| **I→I** | `ii` | 0.15 | 0.10–0.20 | Disinhibition circuits; rhythmogenesis |

**Biological rationale**:
- **E→E** is relatively sparse to prevent epileptiform activity
- **E→I** ensures rapid interneuron recruitment for feedforward inhibition
- **I→E** is the highest probability to maintain E/I balance (inhibitory neurons contact many pyramidal cells)
- **I→I** enables disinhibition and contributes to gamma oscillations

### Distance-Dependent Weight Decay

Synaptic weights decay exponentially with Euclidean distance:

```
weight = base_weight × exp(-λ × distance)
```

Where:
- `base_weight` is drawn from a uniform distribution (see [Weight Assignment](#weight-assignment))
- `λ` (lambda) is the distance decay constant
- `distance` is the 3D Euclidean distance between pre- and post-synaptic neurons

**Effect of λ on connectivity**:

| λ Value | Characteristic Length (1/λ) | Effect |
|---------|----------------------------|--------|
| 0.05    | 20 units                   | Very long-range; global connectivity |
| 0.10    | 10 units                   | Long-range; ~30% of sphere diameter |
| 0.15    | 6.7 units                  | Medium-range (current E→E, E→I, I→I) |
| 0.20    | 5 units                    | Short-range; local clusters |
| 0.30    | 3.3 units                  | Very local; nearest-neighbor dominated |

With `λ = 0.15` and sphere diameter of 33.8 units:
- Neurons at distance 6.7 units: weights decay to ~37% (1/e)
- Neurons at distance 13.4 units: weights decay to ~13% (1/e²)
- Neurons across the sphere (33.8 units): weights decay to ~0.6%

### Dual Lambda System

The network implements **separate distance decay constants** for different connection types:

| Parameter | Current Value | Applies To | Biological Rationale |
|-----------|---------------|------------|----------------------|
| `distance_lambda` | 0.15 | E→E, E→I, I→I | Standard excitatory and local inhibitory range |
| `lambda_decay_ie` | 0.08 | I→E only | **Longer-range inhibition** for gain control |

**Why I→E connections have lower λ (longer range)**:

1. **Basket cells** (PV+ interneurons) have extensive axonal arbors that contact hundreds of pyramidal neurons
2. **Lateral inhibition** extends beyond local excitatory connections
3. **Gain normalization** requires inhibition to "see" the broader activity level
4. **Surround suppression** in sensory cortex depends on long-range inhibition

With `λ_ie = 0.08` (characteristic length = 12.5 units):
- I→E connections reach ~37% of sphere diameter before decaying to 1/e
- This is nearly **twice the range** of other connection types

### Transmission Delays

Synaptic transmission delays scale with inter-neuron distance, modeling axonal conduction time:

```
delay = base_delay × (0.5 + 0.5 × distance / sphere_diameter)
```

| Distance | Delay (base = 1.0 ms) |
|----------|----------------------|
| 0 (adjacent) | 0.5 ms |
| Half diameter | 0.75 ms |
| Full diameter | 1.0 ms |

**Effects of transmission delays**:
- Prevents artificial synchrony from instantaneous propagation
- Creates traveling waves of activity
- Contributes to gamma-band oscillations (~30-80 Hz)
- Enables temporal coding through spike timing differences

### Weight Assignment

Weights are assigned based on presynaptic neuron type:

| Presynaptic Type | Weight Range | Distribution |
|------------------|--------------|--------------|
| **Excitatory** | `[0, weight_scale]` | Uniform; positive values |
| **Inhibitory** | `[-4×weight_scale, 0]` | Uniform; negative values, 4× stronger |

With `weight_scale = 0.3`:
- Excitatory weights: `[0, 0.3]`
- Inhibitory weights: `[-1.2, 0]`

The **4× inhibitory multiplier** reflects:
- Biological data showing IPSPs are typically 3-5× larger than EPSPs
- Compensates for the ~80:20 E:I ratio to maintain balance
- Ensures that ~20% inhibitory neurons can effectively control ~80% excitatory neurons

### Zero-Weight Pruning

Connections with weights in the range `[-0.01, 0.01]` are automatically pruned during network construction. This:
- Reduces memory usage by not storing negligible connections
- Speeds up simulation by eliminating spike propagation to pruned targets
- Results in the `nonzero_connections` metric being lower than `attempted_connections`

Current statistics (6000 neurons):
- Attempted connections: ~4.8 million
- Non-zero connections after pruning: ~3.0 million
- **Effective connection density**: ~8.4%

---

## Neuron Model

### Membrane Dynamics

Each neuron follows the **Leaky Integrate-and-Fire (LIF)** dynamics with conductance-based synapses:

```
τ_m × dV/dt = -(V - V_rest) + I_syn + I_adaptation + noise
```

**Membrane parameters**:

| Parameter | Symbol | Base Value | Unit | Description |
|-----------|--------|------------|------|-------------|
| Resting potential | V_rest | -65.0 | mV | Equilibrium potential without input |
| Spike threshold | V_threshold | -55.0 | mV | Voltage at which spike is emitted |
| Reset potential | V_reset | -75.0 | mV | Voltage after spike (hyperpolarized) |
| Membrane time constant | τ_m | 10.0 | ms | Integration timescale |
| Refractory period (E) | τ_ref | 4.0 | ms | Absolute refractory period (~250 Hz max) |
| Refractory period (I) | τ_ref | 2.5 | ms | Faster for interneurons (~400 Hz max) |

### Conductance-Based Synapses

Unlike current-based models, conductance-based synapses model the actual biophysical process:

```
I_syn = g_e × (E_e - V) + g_i × (E_i - V)
```

Where:
- `g_e`, `g_i` are time-varying excitatory/inhibitory conductances
- `E_e`, `E_i` are reversal potentials
- `(E - V)` is the driving force

**Synaptic parameters**:

| Parameter | Symbol | Value | Unit | Receptor Type |
|-----------|--------|-------|------|---------------|
| Excitatory τ | τ_e | 3.0 | ms | AMPA-like |
| Inhibitory τ | τ_i | 7.0 | ms | GABA_A-like |
| Excitatory reversal | E_e | 0.0 | mV | Glutamatergic |
| Inhibitory reversal | E_i | -80.0 | mV | GABAergic |

**Conductance decay**:
```
dg_e/dt = -g_e / τ_e
dg_i/dt = -g_i / τ_i
```

**Advantages of conductance-based synapses**:

1. **Automatic gain control**: Driving force `(E - V)` decreases as V approaches reversal potential
2. **Shunting inhibition**: Inhibition is most effective when neuron is depolarized (larger driving force)
3. **Realistic saturation**: Currents naturally saturate near reversal potentials
4. **Voltage-dependent EPSP/IPSP amplitude**: Same conductance produces different effects depending on membrane potential

### Spike-Frequency Adaptation

Models calcium-activated potassium channels (SK, BK types):

```
I_adaptation = g_adaptation × (E_K - V)
```

**Adaptation parameters**:

| Parameter | Base Value | Unit | Description |
|-----------|------------|------|-------------|
| Adaptation increment | 0.2 | - | Conductance increase per spike |
| Adaptation τ | 100.0 | ms | Decay time constant |
| K+ reversal (E_K) | -90.0 | mV | Potassium equilibrium potential |

**Mechanism**:
1. Each spike increases `g_adaptation` by `adaptation_increment`
2. Between spikes, `g_adaptation` decays exponentially with τ = 100 ms
3. The adaptation current hyperpolarizes the neuron, reducing firing rate
4. Current automatically bounds near E_K = -90 mV (no unrealistic hyperpolarization)

**Effects**:
- Prevents tonic high-frequency firing
- Creates spike-rate adaptation (first ISI shorter than later ISIs)
- Contributes to bursting dynamics
- Helps maintain network stability

---

## Parameter Heterogeneity (Jitter)

### Why Heterogeneity Matters

Real neurons exhibit substantial variability in their intrinsic properties. Parameter heterogeneity serves several critical functions:

1. **Prevents artificial synchronization**: Identical neurons with identical inputs fire simultaneously, creating unrealistic network-wide synchrony
2. **Broadens frequency response**: Population can respond to a wider range of input frequencies
3. **Increases robustness**: Network function doesn't depend on precise parameter values
4. **Enables population coding**: Different neurons respond optimally to different inputs
5. **Matches biological data**: Cortical neurons show 20-40% CV in most parameters

### Gaussian Jitter (Voltage Parameters)

Applied to parameters that can vary symmetrically around a baseline:

```
jittered_value = base_value + Normal(0, σ)
```

| Parameter | Config Key | Recommended σ | Effect |
|-----------|------------|---------------|--------|
| V_rest | `jitter_v_rest` | **3.0 mV** | Different "resting points" → varied excitability |
| V_threshold | `jitter_v_threshold` | **2.5 mV** | Different spike thresholds → varied sensitivity |

**Current configuration**: `jitter_v_rest=3.0`, `jitter_v_threshold=2.5`

With these settings:
- ~68% of neurons have V_rest within ±3 mV of -65 mV
- ~95% of neurons have V_rest within ±6 mV of -65 mV
- Combined with V_threshold jitter, the rheobase (threshold - rest) varies by ~4-5 mV

### Log-Normal Jitter (Time Constants)

Applied to strictly positive parameters using multiplicative (coefficient of variation) specification:

```
σ_log = sqrt(ln(1 + CV²))
μ_log = ln(base_value) - σ_log²/2
jittered_value = LogNormal(μ_log, σ_log)
```

This ensures:
- Values are always positive (no negative time constants)
- Mean equals base value (not median)
- Right-skewed distribution matches biological observations

| Parameter | Config Key | Recommended CV | Effect |
|-----------|------------|----------------|--------|
| τ_m | `jitter_tau_m` | **0.30** | Different integration speeds |
| τ_ref | `jitter_tau_ref` | **0.25** | Different maximum firing rates |
| τ_e | `jitter_tau_e` | **0.30** | Different EPSP kinetics |
| τ_i | `jitter_tau_i` | **0.30** | Different IPSP kinetics |
| Adaptation increment | `jitter_adaptation_increment` | **0.40** | Different adaptation strengths |
| τ_adaptation | `jitter_tau_adaptation` | **0.35** | Different adaptation timescales |

**Example**: With `jitter_tau_m = 0.30` (30% CV) and base τ_m = 10 ms:
- Mean: 10 ms
- Standard deviation: ~3 ms
- ~68% of neurons have τ_m between ~7-13 ms
- Right tail extends to ~15-20 ms (slow integrators)

### Biologically Plausible Clipping

After jitter application, values are clipped to biologically realistic ranges:

| Parameter | Minimum | Maximum | Rationale |
|-----------|---------|---------|-----------|
| V_rest | -80 mV | -55 mV | Can't exceed typical threshold range |
| V_threshold | V_rest + 5 mV | -40 mV | Must exceed rest; can't be too high |
| τ_m | 3 ms | 30 ms | Biological range for cortical neurons |
| τ_ref | 1 ms | 10 ms | Physical limits of Na+ channel recovery |
| τ_e | 0.5 ms | 10 ms | AMPA: fast; some NMDA contribution possible |
| τ_i | 2 ms | 20 ms | GABA_A typical range |
| Adaptation increment | 0 | 1 | Must be non-negative |
| τ_adaptation | 20 ms | 300 ms | AHP timescale range |

---

## Network-Level Noise

Two types of noise operate continuously during simulation:

| Noise Type | Parameter | Current Value | Target | Description |
|------------|-----------|---------------|--------|-------------|
| **Membrane noise** | `v_noise_amp` | 0.6 mV | V | Gaussian noise added to voltage per timestep |
| **Synaptic noise** | `i_noise_amp` | 0.003 | g_e, g_i | Positive-only noise to conductances |

**Membrane noise** models:
- Ion channel stochasticity
- Background synaptic bombardment
- Thermal fluctuations

**Synaptic noise** models:
- Spontaneous vesicle release
- Ongoing background network activity
- Neuromodulatory fluctuations

**Key distinction**: Noise is **dynamic** (changes every timestep) while jitter is **static** (set once at network creation).

---

## E/I Balance Mechanisms

The network maintains excitation-inhibition balance through multiple mechanisms:

| Mechanism | Implementation | Effect |
|-----------|----------------|--------|
| **Higher I→E probability** | 25% vs 10-15% for others | Inhibitory neurons contact more targets |
| **Stronger inhibitory weights** | 4× excitatory | Each inhibitory spike has larger impact |
| **Longer-range inhibition** | λ_ie = 0.08 vs λ = 0.15 | Inhibition "sees" broader activity |
| **Faster inhibitory firing** | τ_ref = 2.5 ms vs 4.0 ms | Interneurons can fire at ~400 Hz vs ~250 Hz |
| **Shunting inhibition** | Conductance-based model | Inhibition more effective when neuron is excited |

**E/I ratio**: The network uses ~80% excitatory and ~20% inhibitory neurons (`inhibitory_fraction = 0.195`), matching cortical proportions.

---

## Stimulation Modes

### Current Injection Mode
```python
current_injection_stimulation=True
current_injection_duration=200  # timesteps (20 ms at dt=0.1)
```
- Selected neurons receive sustained conductance increase
- Applied for N consecutive timesteps
- Creates reliable, synchronized activation

### Poisson Process Mode (Current Default)
```python
poisson_process_stimulation=True
poisson_process_probability=0.005  # 0.5% per timestep
poisson_process_duration=50        # timesteps (5 ms window)
```
- At each stim_interval, a stimulation window opens
- During window, each selected neuron has independent probability of receiving input per timestep
- Creates more naturalistic, asynchronous activation
- Better models ongoing cortical input

### Stimulation Parameters

| Parameter | Current Value | Description |
|-----------|---------------|-------------|
| `stim_interval` | 100 ms | Time between stimulation events |
| `stim_fraction` | 0.20 | Fraction of network receiving stimulation |
| `stim_interval_strength` | 10 | Conductance increase per stimulation event |

---

## Current Experiment Configuration

From `experiment_config.json`:

### Network Structure
| Parameter | Value |
|-----------|-------|
| Total neurons | 6,000 |
| Excitatory neurons | 4,817 (80.3%) |
| Inhibitory neurons | 1,183 (19.7%) |
| Sphere radius | 16.91 units |
| Connection density | 8.38% |
| Non-zero connections | 3,017,699 |

### Connectivity
| Parameter | Value |
|-----------|-------|
| P(E→E) | 0.10 |
| P(E→I) | 0.15 |
| P(I→E) | 0.25 |
| P(I→I) | 0.15 |
| Weight scale | 0.30 |
| Distance λ | 0.15 |
| I→E λ | 0.08 |
| Base delay | 1.0 ms |

### Neuron Parameters (with jitter)
| Parameter | Base | Jitter | Type |
|-----------|------|--------|------|
| V_rest | -65 mV | σ = 3.0 mV | Gaussian |
| V_threshold | -55 mV | σ = 2.5 mV | Gaussian |
| τ_m | 10 ms | CV = 0.30 | Log-normal |
| τ_ref (E) | 4.0 ms | CV = 0.25 | Log-normal |
| τ_ref (I) | 2.5 ms | CV = 0.25 | Log-normal |
| τ_e | 3.0 ms | CV = 0.30 | Log-normal |
| τ_i | 7.0 ms | CV = 0.30 | Log-normal |
| Adaptation inc. | 0.2 | CV = 0.40 | Log-normal |
| τ_adaptation | 100 ms | CV = 0.35 | Log-normal |

### Simulation
| Parameter | Value |
|-----------|-------|
| Duration | 500 ms |
| Timestep (dt) | 0.1 ms |
| V noise | 0.6 mV |
| I noise | 0.003 |

---

## Output Files

| File | Description |
|------|-------------|
| `experiment_config.json` | Complete parameter dump with actual neuron samples |
| `spherical_activity.html` | Interactive 3D Three.js animation of network activity |
| `neuron_parameter_distributions.png` | Histograms showing jittered parameter distributions |
| `connection_type_distribution.png` | Bar charts of configured vs actual connection probabilities |
| `psth_raster_plot.png` | Population spike histogram + raster (all neurons) |
| `ei_psth_raster_plot.png` | E/I separated PSTH and raster plots |
| `network_activation.png` | Percentage of network active over time |
| `avalanche_branching.png` | Avalanche size distribution for criticality analysis |
| `avalanche_size_vs_duration.png` | Size-duration scaling relationship |
| `oscillation_frequency_analysis.png` | Power spectrum and gamma band detection |
| `ei_frequency_analysis.png` | E/I separated frequency analysis |
| `ei_synchrony_analysis.png` | Synchrony metrics at multiple timescales |
| `distance_weights_3d.html` | Interactive 3D visualization of distance-weight relationships |
| `ie_distance_weights_3d.html` | I→E specific distance-weight visualization |

---

## Usage Examples

### Basic Usage
```python
from spherical_network_experiment import run_biologically_plausible_simulation

network, activity_record, neuron_data, stim_record = run_biologically_plausible_simulation()
```

### Custom Configuration
```python
from spherical_network_experiment import run_spherical_experiment

network, activity, neurons, stim = run_spherical_experiment(
    # Network topology
    n_neurons=6000,
    inhibitory_fraction=0.20,
    layout='sphere',

    # Connectivity
    connection_probabilities={
        'ee': 0.10,  # Sparse E→E recurrence
        'ei': 0.15,  # Feedforward to interneurons
        'ie': 0.25,  # Strong blanket inhibition
        'ii': 0.15,  # Moderate disinhibition
    },
    weight_scale=0.3,
    distance_lambda=0.15,      # Standard decay
    lambda_decay_ie=0.08,      # Longer-range I→E
    transmission_delay=1.0,

    # Parameter heterogeneity
    jitter_v_rest=3.0,         # mV (Gaussian)
    jitter_v_threshold=2.5,    # mV (Gaussian)
    jitter_tau_m=0.30,         # CV (Log-normal)
    jitter_tau_ref=0.25,       # CV (Log-normal)
    jitter_tau_e=0.30,         # CV (Log-normal)
    jitter_tau_i=0.30,         # CV (Log-normal)
    jitter_adaptation_increment=0.40,  # CV (Log-normal)
    jitter_tau_adaptation=0.35,        # CV (Log-normal)

    # Network noise
    enable_noise=True,
    v_noise_amp=0.6,           # mV per timestep
    i_noise_amp=0.003,         # Conductance per timestep

    # Stimulation
    stim_interval=100,         # ms between stim events
    stim_fraction=0.20,        # 20% of network
    poisson_process_stimulation=True,
    poisson_process_probability=0.005,
    poisson_process_duration=50,

    # Simulation
    duration=500,              # ms
    dt=0.1,                    # ms
    random_seed=42,
)
```

### Accessing Network Data
```python
# Neuron parameter distributions
v_rest_distribution = network.neuron_params['v_rest']
tau_m_distribution = network.neuron_params['tau_m']
inhibitory_mask = network.neuron_params['is_inhibitory']

# Connectivity matrices
weights = network.weights      # (n_neurons, n_neurons) dense matrix
delays = network.delays        # Transmission delays

# Connection statistics
print(f"E→E connections: {network.connection_counts_by_type['ee']}")
print(f"I→E connections: {network.connection_counts_by_type['ie']}")

# Avalanche statistics (for criticality analysis)
sizes = network.avalanche_sizes
durations = network.avalanche_durations

# 3D positions
positions = network.neuron_3d_positions  # Dict[int, (x, y, z)]
```

---

## References

1. **Brunel, N. (2000)**. Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *Journal of Computational Neuroscience*, 8(3), 183-208.
   - Foundational work on E/I network dynamics and balance

2. **Beggs, J. M., & Plenz, D. (2003)**. Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.
   - Criticality and avalanche dynamics in neural networks

3. **Markram, H., et al. (2015)**. Reconstruction and simulation of neocortical microcircuitry. *Cell*, 163(2), 456-492.
   - Blue Brain Project data on cortical connectivity and cell types

4. **Destexhe, A., et al. (2003)**. Fluctuating synaptic conductances recreate in vivo-like activity in neocortical neurons. *Neuroscience*, 107(1), 13-24.
   - Conductance-based synapses and in vivo-like dynamics

5. **Compte, A., et al. (2003)**. Cellular and network mechanisms of slow oscillatory activity. *Journal of Neurophysiology*, 89(5), 2707-2725.
   - Adaptation and slow oscillations in cortical networks
