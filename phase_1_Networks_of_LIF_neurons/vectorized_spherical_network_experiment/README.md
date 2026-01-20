# Vectorized Spherical Network Experiment

Vectorized spiking neural network simulations exploring criticality across three distinct activity regimes.

## Network Topology

Neurons are distributed uniformly within a **3D spherical volume** with radius calculated from neuron count: `r = (3n/4π)^(1/3) × 1.5`.

### Distance-Dependent Connectivity

Synaptic connections follow an **exponential distance decay**:

```
P(connection) = P_base × exp(-λ × distance)
weight = w_base × exp(-λ × distance)
```

- `distance_lambda = 0.2` — Decay constant for E→E, E→I, I→I connections
- `lambda_decay_ie = 0.15` — Slower decay for I→E (longer-range inhibition)

### Connection Type Probabilities

| Type | Probability | Description |
|------|-------------|-------------|
| E→E | 12% | Local recurrent excitation |
| E→I | 15% | Feedforward to interneurons |
| I→E | 25% | Blanket inhibition (wide-reaching) |
| I→I | 15% | Interneuron networks / disinhibition |

The asymmetric I→E connectivity (higher probability, longer range) implements biologically realistic "blanket inhibition" where interneurons provide broad suppression across the excitatory population.

## Activity Regimes

This project contains three parameter configurations demonstrating sub-critical, critical, and super-critical network dynamics:

### 1. Sub-Critical (`sub_critical_activity_regime/`)

Activity dies out quickly after stimulation. The network cannot sustain propagating activity.

**Characteristics:**
- Branching ratio: **0.44** (< 1.0)
- Total spikes: ~26k (sparse)
- Mean activation: 0.017%
- Assessment: "Not critical"

### 2. Critical (`critical_activity_regime/`)

Balanced dynamics at the "edge of chaos." Activity propagates without dying out or exploding.

**Characteristics:**
- Branching ratio: **0.91** (≈ 1.0)
- Total spikes: ~65k (moderate)
- Mean activation: 0.043%
- Assessment: "Moderately critical"

### 3. Super-Critical (`super_critical_activity_regime/`)

Runaway excitation with sustained high activity and explosive avalanches.

**Characteristics:**
- Branching ratio: **1.003** (> 1.0)
- Total spikes: ~3.5M (explosive)
- Mean activation: 2.37%
- Assessment: "Strongly critical"

## Criticality Measurement

Criticality is assessed using **neural avalanche analysis**:

### Branching Ratio (σ)
The average number of neurons activated by each previously active neuron:
- **σ < 1.0**: Sub-critical — activity dies out
- **σ = 1.0**: Critical — balanced propagation
- **σ > 1.0**: Super-critical — activity explodes

### Avalanche Statistics
- **Avalanche**: Continuous cascade of activity (no silent timesteps)
- **Size**: Total spikes in an avalanche
- **Duration**: Number of timesteps

Critical networks exhibit power-law distributions:
- Size distribution: P(s) ~ s^(-τ), τ ≈ 1.5
- Duration distribution: P(d) ~ d^(-α), α ≈ 2.0
- Size-duration scaling: ⟨s⟩ ~ d^(1/σνz)

### R² Goodness of Fit
Measures how well the size-duration relationship follows a power law (closer to 1.0 = better fit).

## Parameter Differences

| Parameter | Sub-Critical | Critical | Super-Critical |
|-----------|--------------|----------|----------------|
| `inhibitory_fraction` | **0.55** | **0.15** | **0.20** |
| `weight_scale` | 0.35 | 0.35 | **0.60** |

### Key Insights

**Inhibitory fraction** is the primary control parameter:
- High inhibition (55%) → Sub-critical (activity suppressed)
- Low inhibition (15%) → Critical (balanced E/I)
- Low inhibition + strong weights → Super-critical (runaway excitation)

**Weight scale** amplifies excitability:
- Higher weights (0.6 vs 0.35) push the network toward super-criticality

All other parameters remain constant across regimes:
- 6000 neurons
- Connection probabilities: E→E 12%, E→I 15%, I→E 25%, I→I 15%
- Distance decay λ = 0.2
- Poisson stimulation with p = 0.005

## Files

Each regime folder contains:
- `spherical_network_experiment_vectorized.py` — Simulation script
- `experiment_config_vectorized.json` — Full parameter configuration
- `analysis_results_vectorized.json` — Criticality metrics and statistics
- Various `.png` and `.html` visualization outputs
