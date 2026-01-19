# Spherical Neural Network Simulation

A biologically plausible 3D spiking neural network simulation featuring conductance-based Leaky Integrate-and-Fire (LIF) neurons arranged in a spherical volume with distance-dependent connectivity.

## Overview

This simulation models a network of excitatory and inhibitory neurons positioned within a 3D sphere. Key features include:

- **Conductance-based synapses** with reversal potentials
- **Distance-dependent connectivity** with exponential decay
- **Spike-frequency adaptation** to prevent runaway excitation
- **Heterogeneous neuron parameters** via Gaussian/log-normal jitter
- **Transmission delays** based on inter-neuron distance
- **Multiple stimulation modes** (current injection, Poisson process)
- **Avalanche detection** for criticality analysis

---

## Architecture

### Network Structure

```
SphericalNeuronalNetwork
├── neurons: List[LIFNeuronWithReversal]  # Individual neuron objects
├── neuron_3d_positions: Dict[int, (x,y,z)]  # 3D coordinates
├── weights: np.array (n×n)  # Synaptic weight matrix
├── delays: np.array (n×n)   # Transmission delay matrix
├── graph: nx.DiGraph        # Network connectivity graph
└── neuron_params: Dict      # Stored jittered parameters
```

### Neuron Model: LIFNeuronWithReversal

Conductance-based LIF neuron with the following dynamics:

```
Membrane equation:
τ_m * dV/dt = -(V - V_rest) + I_syn + I_adaptation + noise

Synaptic current:
I_syn = g_e * (E_e - V) + g_i * (E_i - V)

Adaptation current:
I_adaptation = g_adaptation * (E_K - V)

Conductance decay:
dg_e/dt = -g_e / τ_e
dg_i/dt = -g_i / τ_i
```

---

## Parameters Reference

### Network Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neurons` | 6000 | Total number of neurons in the network |
| `inhibitory_fraction` | 0.20 | Fraction of neurons that are inhibitory (20% is biologically typical) |
| `sphere_radius` | auto | Calculated from n_neurons: `(3n/4π)^(1/3) * 1.5` |
| `layout` | 'sphere' | Spatial layout: 'sphere' (volume) or 'sphere-surface' |

### Connection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `connection_p` | 0.1 | Default connection probability (fallback) |
| `connection_probabilities` | dict | Per-type probabilities (see below) |
| `weight_scale` | 0.3 | Base scale factor for synaptic weights |
| `distance_lambda` | 0.15 | Distance decay constant for E→E, E→I, I→I connections |
| `lambda_decay_ie` | 0.08 | Distance decay for I→E connections (longer range inhibition) |
| `transmission_delay` | 1.0 ms | Base synaptic transmission delay |

#### Connection Probabilities by Type

| Type | Key | Biological Range | Description |
|------|-----|------------------|-------------|
| E→E | `ee` | 0.05-0.15 | Excitatory to excitatory (local recurrence) |
| E→I | `ei` | 0.15-0.25 | Excitatory to inhibitory (feedforward) |
| I→E | `ie` | 0.40-0.60 | Inhibitory to excitatory (blanket inhibition) |
| I→I | `ii` | 0.10-0.20 | Inhibitory to inhibitory (disinhibition) |

Example configuration:
```python
connection_probabilities={
    'ee': 0.00,   # No E→E recurrence
    'ei': 0.15,   # Moderate E→I
    'ie': 0.40,   # Strong blanket inhibition
    'ii': 0.15,   # Moderate I→I
}
```

### Neuron Parameters (Base Values)

| Parameter | Base Value | Unit | Description |
|-----------|------------|------|-------------|
| `v_rest` | -65.0 | mV | Resting membrane potential |
| `v_threshold` | -55.0 | mV | Spike threshold |
| `v_reset` | -75.0 | mV | Post-spike reset potential |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `tau_ref` (E) | 4.0 | ms | Excitatory refractory period (~250 Hz max) |
| `tau_ref` (I) | 2.5 | ms | Inhibitory refractory period (~400 Hz max) |
| `tau_e` | 3.0 | ms | Excitatory synaptic time constant (AMPA-like) |
| `tau_i` | 7.0 | ms | Inhibitory synaptic time constant (GABA_A-like) |
| `e_reversal` | 0.0 | mV | Excitatory reversal potential |
| `i_reversal` | -80.0 | mV | Inhibitory reversal potential |
| `k_reversal` | -90.0 | mV | Potassium reversal (for adaptation) |
| `adaptation_increment` | 0.2 | - | Adaptation increase per spike |
| `tau_adaptation` | 100.0 | ms | Adaptation decay time constant |

### Parameter Jitter (Heterogeneity)

Two types of jitter are used to create biologically realistic heterogeneity:

#### Gaussian Jitter (Voltage Parameters)
Applied to parameters that can vary symmetrically around a baseline.

| Parameter | Jitter Type | Recommended σ | Description |
|-----------|-------------|---------------|-------------|
| `jitter_v_rest` | Gaussian | 3.0 mV | Resting potential variation |
| `jitter_v_threshold` | Gaussian | 2.5 mV | Threshold variation |

#### Log-Normal Jitter (Time Constants)
Applied to strictly positive parameters. Uses coefficient of variation (CV = σ/μ).

| Parameter | Jitter Type | Recommended CV | Description |
|-----------|-------------|----------------|-------------|
| `jitter_tau_m` | Log-normal | 0.30 | 30% variation in membrane τ |
| `jitter_tau_ref` | Log-normal | 0.25 | 25% variation in refractory period |
| `jitter_tau_e` | Log-normal | 0.30 | 30% variation in excitatory τ |
| `jitter_tau_i` | Log-normal | 0.30 | 30% variation in inhibitory τ |
| `jitter_adaptation_increment` | Log-normal | 0.40 | 40% variation in adaptation |
| `jitter_tau_adaptation` | Log-normal | 0.35 | 35% variation in adaptation τ |

**Why log-normal?**
- Naturally bounded above zero (no negative time constants)
- Right-skewed distribution matches biological observations
- Multiplicative variation (CV) is biologically meaningful

### Noise Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_noise` | True | Master switch for noise |
| `v_noise_amp` | 0.6 mV | Membrane potential noise (per timestep) |
| `i_noise_amp` | 0.003 | Synaptic conductance noise (per timestep) |

**Note:** Noise is dynamic (changes every timestep), while jitter is static (set once at network creation).

### Stimulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `no_stimulation` | False | If True, no external stimulation |
| `stochastic_stim` | False | Random vs regular stimulation timing |
| `stim_interval` | 20 ms | Interval between stimulation events |
| `stim_interval_strength` | 10 | Current injection strength |
| `stim_fraction` | 0.20 | Fraction of neurons to stimulate |

#### Stimulation Modes

**1. Current Injection Mode** (`current_injection_stimulation=True`)
```python
current_injection_stimulation=True
current_injection_duration=200  # timesteps
```
Applies sustained current to selected neurons for N consecutive timesteps.

**2. Poisson Process Mode** (`poisson_process_stimulation=True`)
```python
poisson_process_stimulation=True
poisson_process_probability=0.005  # per-timestep probability
poisson_process_duration=3000      # window in timesteps
```
Each timestep within the window has independent probability of triggering stimulation.

### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration` | 100 ms | Total simulation duration |
| `dt` | 0.1 ms | Integration timestep |
| `random_seed` | 42 | Seed for reproducibility |

### Visualization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `animate` | True | Generate 3D animation |
| `max_animation_frames` | 1000 | Maximum frames (downsamples if exceeded) |

---

## Output Files

The simulation generates the following outputs:

| File | Description |
|------|-------------|
| `experiment_config.json` | Complete parameter dump |
| `spherical_activity.html` | Interactive 3D Three.js animation |
| `neuron_parameter_distributions.png` | Histograms of jittered parameters |
| `connection_type_distribution.png` | Connection probability validation |
| `psth_raster_plot.png` | Population spike histogram + raster |
| `ei_psth_raster_plot.png` | E/I separated PSTH and raster |
| `network_activation.png` | Percentage of network active over time |
| `avalanche_branching.png` | Avalanche size distribution (criticality) |
| `avalanche_size_vs_duration.png` | Size-duration relationship |
| `oscillation_frequency_analysis.png` | Power spectrum / gamma detection |
| `ei_frequency_analysis.png` | E/I separated frequency analysis |
| `ei_synchrony_analysis.png` | Synchrony metrics at multiple timescales |
| `distance_weights_3d.html` | 3D visualization of distance-weight decay |
| `ie_distance_weights_3d.html` | I→E specific distance-weight visualization |

---

## Usage

### Basic Usage

```python
from spherical_network_experiment import run_biologically_plausible_simulation

network, activity_record, neuron_data, stim_record = run_biologically_plausible_simulation()
```

### Custom Configuration

```python
from spherical_network_experiment import run_spherical_experiment

network, activity, neurons, stim = run_spherical_experiment(
    # Network size
    n_neurons=3000,
    inhibitory_fraction=0.20,

    # Connectivity
    connection_probabilities={
        'ee': 0.10,
        'ei': 0.15,
        'ie': 0.40,
        'ii': 0.15,
    },
    weight_scale=0.5,
    distance_lambda=0.12,

    # Simulation
    duration=200,  # ms
    dt=0.1,

    # Stimulation
    stim_interval=50,
    stim_fraction=0.10,
    poisson_process_stimulation=True,
    poisson_process_probability=0.01,

    # Heterogeneity
    jitter_v_rest=3.0,
    jitter_v_threshold=2.5,
    jitter_tau_m=0.3,
    jitter_tau_ref=0.25,

    # Noise
    enable_noise=True,
    v_noise_amp=0.5,
    i_noise_amp=0.002,
)
```

### Accessing Network Data

```python
# Get neuron parameters
v_rest_distribution = network.neuron_params['v_rest']
tau_m_distribution = network.neuron_params['tau_m']
inhibitory_mask = network.neuron_params['is_inhibitory']

# Get connectivity
weights = network.weights  # (n_neurons, n_neurons) matrix
delays = network.delays    # Transmission delays

# Get avalanche statistics
sizes = network.avalanche_sizes
durations = network.avalanche_durations

# Get 3D positions
positions = network.neuron_3d_positions  # Dict[int, (x, y, z)]
```

---

## Biophysical Basis

### Conductance-Based Synapses

Unlike current-based synapses, conductance-based synapses model the actual biophysical process:

```
I_syn = g * (E_rev - V)
```

This provides:
- **Automatic gain control**: Driving force decreases as V approaches E_rev
- **Shunting inhibition**: Inhibition is most effective when the neuron is depolarized
- **Realistic saturation**: Currents naturally saturate near reversal potentials

### Spike-Frequency Adaptation

Models calcium-activated potassium channels (SK, BK):

```
I_adapt = g_adapt * (E_K - V)
```

- Increases with each spike
- Decays exponentially (τ ~ 100ms)
- Prevents runaway excitation
- Bounded by E_K reversal potential (prevents unrealistic hyperpolarization)

### Distance-Dependent Connectivity

Synaptic weight decays exponentially with distance:

```
weight = base_weight * exp(-λ * distance)
```

- `distance_lambda` controls decay rate for most connections
- `lambda_decay_ie` controls I→E connections (typically longer range)
- Transmission delay scales with distance

### E/I Balance

The network maintains excitation-inhibition balance through:
- Higher I→E connection probability (~40%)
- Stronger inhibitory weights (4× excitatory)
- Longer-range inhibition (lower λ for I→E)

---

## Criticality Analysis

The simulation includes avalanche detection for criticality analysis:

- **Avalanche**: Continuous cascade of activity (no silent timesteps)
- **Size**: Total number of spikes in avalanche
- **Duration**: Number of timesteps

Critical networks exhibit:
- Power-law size distribution: P(s) ~ s^(-τ) with τ ≈ 1.5
- Power-law duration distribution: P(d) ~ d^(-α) with α ≈ 2.0
- Size-duration scaling: ⟨s⟩ ~ d^(1/σνz) with exponent ≈ 2.0

---

## Performance Notes

- Network creation: O(n²) for connectivity matrix
- Simulation: O(n × connections × timesteps)
- Memory: ~8 bytes × n² for weight matrix
- Recommended: n ≤ 10,000 neurons for interactive use

For larger networks, consider:
- Sparse matrix representations
- GPU acceleration
- Reducing connection probability

---

## Dependencies

- numpy
- networkx
- matplotlib
- plotly
- scipy (for frequency analysis)

---

## References

1. Brunel, N. (2000). Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons. *Journal of Computational Neuroscience*, 8(3), 183-208.

2. Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

3. Markram, H., et al. (2015). Reconstruction and simulation of neocortical microcircuitry. *Cell*, 163(2), 456-492.
