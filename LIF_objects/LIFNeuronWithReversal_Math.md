<head>
<style>
body {
    background-color: white;
    color: #333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
}
h1, h2, h3 {
    color: #2c3e50;
}
code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: 'Consolas', monospace;
}
pre {
    background-color: #f8f8f8;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #3498db;
    overflow-x: auto;
}
.code-link {
    color: #3498db;
    font-size: 0.9em;
}
</style>
</head>

# Leaky Integrate-and-Fire Neuron with Reversal Potentials

## Mathematical Foundations and Code Implementation

This document provides an in-depth explanation of each mathematical equation in the LIF neuron model and shows exactly how it maps to the Python implementation.

---

## 1. The Membrane Potential Equation

### The Differential Equation

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + I_{syn} + I_{adapt} + \eta_V$$

where $\eta_V \sim \mathcal{N}(0, \sigma_V)$ is Gaussian membrane noise.

### What This Equation Means

This is the **heart of the neuron model**. Let's break down each term:

**Left side: $\tau_m \frac{dV}{dt}$**

- $\frac{dV}{dt}$ is the rate of change of voltage over time
- $\tau_m$ (membrane time constant) controls how quickly the neuron responds
- A larger $\tau_m$ means the neuron integrates inputs more slowly (sluggish response)
- A smaller $\tau_m$ means rapid voltage changes (quick response)

**First term: $-(V - V_{rest})$**

This is the **leak term**. It represents the passive return to resting potential:
- When $V > V_{rest}$: the term is negative, pulling voltage DOWN
- When $V < V_{rest}$: the term is positive, pulling voltage UP
- The neuron always "wants" to return to $V_{rest}$
- This models real neurons where ion channels maintain resting potential

**Second term: $I_{syn}$**

The total synaptic input current (detailed in Section 2).

**Third term: $I_{adapt}$**

The adaptation current that reduces excitability after spiking (detailed in Section 5). Note: this is a conductance-based adaptation current with its own reversal potential.

**Fourth term: $\eta_V$**

Gaussian noise representing channel stochasticity and background activity. This adds biological realism by preventing identical neurons from behaving identically.

### Rearranging for Implementation

Dividing both sides by $\tau_m$:

$$\frac{dV}{dt} = \frac{-(V - V_{rest})}{\tau_m} + I_{syn} + I_{adapt} + \eta_V$$

### Code Implementation

```python
# Line 128 in LIFNeuronWithReversal.py
dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_total + i_adaptation) + v_noise
self.v += dv
```

**How the code maps to math:**

| Math Term | Code Variable | Description |
|-----------|---------------|-------------|
| $\Delta t$ | `dt` | Time step size |
| $V$ | `self.v` | Current membrane potential |
| $V_{rest}$ | `self.v_rest` | Resting potential (-65 mV) |
| $\tau_m$ | `self.tau_m` | Membrane time constant (10 ms) |
| $I_{syn}$ | `i_total` | Sum of excitatory and inhibitory currents |
| $I_{adapt}$ | `self.adaptation` | Adaptation current |
| $\eta_V$ | `v_noise` | Gaussian noise term |

### Numerical Method: Forward Euler

The code uses **Forward Euler integration** with additive noise:

$$V(t + \Delta t) = V(t) + \Delta t \cdot \frac{dV}{dt} + \eta_V$$

where the noise term $\eta_V$ is added after the deterministic integration step.

This is the simplest numerical integration method:
1. Calculate the derivative at the current time
2. Multiply by the time step
3. Add to the current value
4. Add the stochastic noise term

**Why Forward Euler?** It's computationally cheap and stable for small $\Delta t$. For typical simulations with $\Delta t = 0.1$ ms and $\tau_m = 10$ ms, the ratio $\Delta t / \tau_m = 0.01$ is small enough for stability.

---

## 2. Conductance-Based Synaptic Currents

### Summing Multiple Synaptic Inputs

A neuron receives input from many presynaptic neurons. The total synaptic current is the sum of all individual contributions:

$$I_{syn} = \sum_{j=1}^{N} I_j$$

where $N$ is the total number of synapses onto the neuron.

For a **conductance-based model**, we separate excitatory and inhibitory inputs:

$$I_{syn} = \underbrace{\sum_{j \in \mathcal{E}} g_j(t) \cdot (E_e - V)}_{\text{excitatory inputs}} + \underbrace{\sum_{k \in \mathcal{I}} g_k(t) \cdot (E_i - V)}_{\text{inhibitory inputs}}$$

where $\mathcal{E}$ is the set of excitatory synapses and $\mathcal{I}$ is the set of inhibitory synapses.

Since the driving force $(E_{rev} - V)$ is the same for all synapses of the same type, we can factor it out:

$$I_{syn} = (E_e - V) \sum_{j \in \mathcal{E}} g_j(t) + (E_i - V) \sum_{k \in \mathcal{I}} g_k(t)$$

Defining the **total conductances** as:

$$g_e(t) = \sum_{j \in \mathcal{E}} g_j(t) \qquad g_i(t) = \sum_{k \in \mathcal{I}} g_k(t)$$

We arrive at the compact form:

### The Equations

$$I_e = g_e \cdot (E_e - V)$$

$$I_i = g_i \cdot (E_i - V)$$

$$I_{syn} = I_e + I_i$$

### Deep Dive: Why Reversal Potentials?

**The biological basis:**

Real synaptic currents arise from ions flowing through channels. The direction and magnitude of flow depends on:
1. **Conductance** ($g$): How many channels are open
2. **Driving force** ($E_{rev} - V$): The electrochemical gradient

**Excitatory synapses (AMPA/NMDA receptors):**
- Allow Na+ and K+ to flow
- Net reversal potential $E_e \approx 0$ mV
- Since typical $V \approx -65$ mV, the driving force $(0 - (-65)) = +65$ mV is positive
- Current flows INTO the cell, depolarizing it

**Inhibitory synapses (GABA receptors):**
- Allow Cl- to flow
- Reversal potential $E_i \approx -70$ mV
- When $V > -70$ mV, driving force is negative
- Current flows OUT (or Cl- flows in), hyperpolarizing the cell

### The Self-Limiting Property

This is crucial: **conductance-based synapses are self-limiting**.

Consider excitatory current as $V$ approaches $E_e$:

| $V$ (mV) | $E_e - V$ (mV) | $I_e$ (relative) |
|----------|----------------|------------------|
| -65 | 65 | Large positive |
| -40 | 40 | Medium positive |
| -10 | 10 | Small positive |
| 0 | 0 | **Zero** |

The neuron can **never** exceed $E_e$ through excitatory input alone! This prevents unrealistic runaway excitation.

Similarly, inhibitory input can never push $V$ below $E_i$.

### Code Implementation

```python
# Lines 116-118 in LIFNeuronWithReversal.py
i_e = self.g_e * (self.e_reversal - self.v)  # Excitatory current
i_i = self.g_i * (self.i_reversal - self.v)  # Inhibitory current
i_total = i_e + i_i
```

**Variable mapping:**

| Math | Code | Default Value |
|------|------|---------------|
| $g_e$ | `self.g_e` | 0.0 (state variable) |
| $g_i$ | `self.g_i` | 0.0 (state variable) |
| $E_e$ | `self.e_reversal` | 0 mV |
| $E_i$ | `self.i_reversal` | -70 mV |
| $V$ | `self.v` | Current membrane potential |

---

## 3. Conductance Dynamics

### The Differential Equations

$$\frac{dg_e}{dt} = -\frac{g_e}{\tau_e} + \max(0, \eta_{g_e})$$

$$\frac{dg_i}{dt} = -\frac{g_i}{\tau_i} + \max(0, \eta_{g_i})$$

where $\eta_{g_e}, \eta_{g_i} \sim \mathcal{N}(0, \sigma_g)$ are independent Gaussian noise terms, rectified to ensure conductances remain non-negative (you can't have negative channel openings).

### Understanding Exponential Decay

This is a **first-order linear ODE** with the well-known solution:

$$g(t) = g_0 \cdot e^{-t/\tau}$$

**What does this mean physically?**

After a synaptic input arrives:
1. Conductance jumps up instantaneously
2. Then decays back to zero exponentially
3. The time constant $\tau$ determines decay speed

**Time constant interpretation:**
- After time $\tau$: conductance drops to $\approx 37\%$ of initial
- After time $2\tau$: drops to $\approx 13.5\%$
- After time $3\tau$: drops to $\approx 5\%$
- After time $5\tau$: essentially zero ($< 1\%$)

### Why Different Time Constants for E and I?

The code uses $\tau_e = 3$ ms and $\tau_i = 7$ ms because:

**Excitatory (AMPA) synapses** have fast kinetics:
- Channels open and close quickly
- Sharp, brief excitatory responses
- Allows precise temporal coding

**Inhibitory (GABA-A) synapses** are slower:
- Longer-lasting inhibition
- Provides sustained suppression
- Important for rhythm generation

### The Analytical Solution for Discrete Updates

For a discrete time step, we solve the deterministic part exactly and add noise:

$$g_e(t + \Delta t) = g_e(t) \cdot e^{-\Delta t / \tau_e} + \max(0, \eta_{g_e})$$

$$g_i(t + \Delta t) = g_i(t) \cdot e^{-\Delta t / \tau_i} + \max(0, \eta_{g_i})$$

**Why use the analytical solution instead of Euler?**

For exponential decay, the analytical solution is:
1. **Exact** - no numerical error accumulation
2. **Unconditionally stable** - works for any $\Delta t$
3. **Computationally efficient** - single multiplication

The noise terms are added separately after the decay step.

### Code Implementation

```python
# Lines 131-140 in LIFNeuronWithReversal.py
# Add synaptic conductance noise
if self.i_noise_amp > 0:
    e_noise = np.random.normal(0, self.i_noise_amp)
    i_noise = np.random.normal(0, self.i_noise_amp)
    self.g_e += e_noise if e_noise > 0 else 0  # Rectified noise
    self.g_i += i_noise if i_noise > 0 else 0  # Rectified noise

# Decay synaptic conductances
self.g_e *= np.exp(-dt/self.tau_e)
self.g_i *= np.exp(-dt/self.tau_i)
```

This is elegant: add rectified noise, then multiply current conductance by the decay factor.

**Example calculation:**

If $g_e = 1.0$, $\tau_e = 3$ ms, $\Delta t = 0.1$ ms:

$$g_e(t + 0.1) = 1.0 \times e^{-0.1/3} = 1.0 \times e^{-0.0333} \approx 0.967$$

So conductance drops by about 3.3% per 0.1 ms time step (before noise is added).

---

## 4. Receiving Synaptic Input

### The Update Rule

When a presynaptic spike arrives with weight $w$:

$$g_e \leftarrow g_e + w \quad \text{(if } w > 0 \text{)}$$

$$g_i \leftarrow g_i + |w| \quad \text{(if } w < 0 \text{)}$$

### What This Models

**Instantaneous conductance increase:**

When a presynaptic neuron fires:
1. Neurotransmitter is released into the synaptic cleft
2. Binds to postsynaptic receptors
3. Ion channels open almost instantaneously (< 1 ms)

The model simplifies this to an instantaneous jump in conductance.

**Weight sign determines synapse type:**
- Positive weight → excitatory synapse (glutamatergic)
- Negative weight → inhibitory synapse (GABAergic)

### Code Implementation

```python
# Lines 153-159 in LIFNeuronWithReversal.py
def receive_spike(self, weight):
    """Receive spike from presynaptic neuron with given weight."""
    if weight > 0:
        self.g_e += weight  # Increase excitatory conductance
    else:
        self.g_i += -weight  # Increase inhibitory conductance (weight is negative)
```

Note the `-weight` for inhibitory: if weight is -0.5, we add 0.5 to $g_i$.

---

## 5. Spike-Frequency Adaptation

### The Differential Equations

**Adaptation conductance dynamics:**

$$\frac{dg_{adapt}}{dt} = -\frac{g_{adapt}}{\tau_{adapt}}$$

**Adaptation current (conductance-based with potassium reversal):**

$$I_{adapt} = g_{adapt} \cdot (E_K - V)$$

where $E_K \approx -90$ mV is the potassium reversal potential.

### On Spike:

$$g_{adapt} \leftarrow g_{adapt} + \Delta_{adapt}$$

### What Is Adaptation?

**Biological basis:**

Real neurons don't fire at constant rates under constant input. After firing, they become temporarily less excitable due to:

1. **Calcium-activated potassium channels**: After a spike, Ca2+ enters the cell and opens K+ channels, hyperpolarizing the membrane
2. **Sodium channel inactivation**: Na+ channels need time to recover after opening
3. **Slow potassium currents**: M-type and other slow K+ currents

### Why Conductance-Based Adaptation?

Using a reversal potential for adaptation provides key benefits:

1. **Self-limiting hyperpolarization**: The current $I_{adapt} = g_{adapt} \cdot (E_K - V)$ goes to zero as $V \to E_K$
2. **Prevents unrealistic voltages**: The membrane cannot hyperpolarize below $E_K$ through adaptation alone
3. **Biological accuracy**: Real K+ channels produce currents that depend on the K+ driving force

**Driving force analysis:**

| $V$ (mV) | $E_K - V$ (mV) | Effect |
|----------|----------------|--------|
| -65 (rest) | -25 | Strong hyperpolarizing current |
| -75 (reset) | -15 | Moderate hyperpolarizing current |
| -85 | -5 | Weak hyperpolarizing current |
| -90 ($E_K$) | 0 | **No current** (equilibrium) |

### How Adaptation Works in the Model

1. Neuron fires a spike
2. $g_{adapt}$ increases by $\Delta_{adapt}$
3. $I_{adapt} = g_{adapt} \cdot (E_K - V)$ produces a **hyperpolarizing current**
4. Makes it harder to reach threshold
5. $g_{adapt}$ slowly decays with time constant $\tau_{adapt}$

**Effect on firing rate:**

| Scenario | $g_{adapt}$ | Firing Rate |
|----------|-------------|-------------|
| First spike | Low | High |
| After several spikes | Accumulated | Reduced |
| Long pause | Decayed | Recovered |

### Code Implementation

**Adaptation current calculation:**
```python
# Lines 122-125 in LIFNeuronWithReversal.py
# Calculate adaptation current with potassium reversal potential
# Models calcium-activated potassium channels (e.g., SK, BK channels)
# Current goes to zero as V approaches E_K (~-90 mV), preventing unrealistic hyperpolarization
i_adaptation = self.adaptation * (self.k_reversal - self.v)
```

**Decay (every time step):**
```python
# Line 143 in LIFNeuronWithReversal.py
self.adaptation *= np.exp(-dt/self.tau_adaptation)
```

**Increment (on spike):**
```python
# Lines 150-151 in LIFNeuronWithReversal.py
# Increase adaptation current when spike occurs
self.adaptation += self.adaptation_increment
```

**Added in membrane equation:**
```python
# Line 128 - note "+ i_adaptation" (current is already negative due to E_K - V)
dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_total + i_adaptation) + v_noise
```

### Time Constant Significance

With $\tau_{adapt} = 100$ ms:
- Adaptation builds up over multiple spikes
- Takes ~300-500 ms to fully decay
- Creates **spike-frequency adaptation**: initial burst followed by steady lower rate

---

## 6. Spike Generation and Reset

### The Threshold Condition

$$\text{Spike occurs when: } V \geq V_{thresh}$$

### The Reset Mechanism

Upon spiking:

$$V \leftarrow V_{reset}$$

$$t_{spike} \leftarrow 0$$

### Why These Values?

**Threshold ($V_{thresh} = -55$ mV):**
- Real neurons have voltage-gated Na+ channels
- These channels open rapidly when $V$ crosses ~-55 mV
- Triggers the action potential upstroke

**Reset ($V_{reset} = -75$ mV):**
- After an action potential, the membrane hyperpolarizes
- Due to K+ channels remaining open
- Called the "afterhyperpolarization" (AHP)
- Reset below rest (-75 < -65) models this AHP

### Code Implementation

```python
# Lines 145-155 in LIFNeuronWithReversal.py
if self.v >= self.v_threshold:
    self.v = self.v_reset
    self.t_since_spike = 0.0

    # Increase adaptation current when spike occurs
    self.adaptation += self.adaptation_increment

    if hasattr(self, 'spike_times'):
        self.spike_times.append(len(self.v_history) * dt)
    return True
```

---

## 7. Refractory Period

### The Logic

$$\text{If } t_{spike} < \tau_{ref}: \text{ voltage integration is blocked}$$

### Biological Basis

After an action potential, neurons enter an **absolute refractory period**:
- Na+ channels are inactivated
- Cannot fire regardless of input strength
- Lasts ~1-2 ms in real neurons

This is followed by a **relative refractory period**:
- Harder but not impossible to fire
- Modeled here by the adaptation current

### Code Implementation

```python
# Lines 91-110 in LIFNeuronWithReversal.py
if self.t_since_spike < self.tau_ref:
    # During refractory period, voltage stays at reset value
    self.v = self.v_reset

    # But synaptic conductances still decay
    self.g_e *= np.exp(-dt/self.tau_e)
    self.g_i *= np.exp(-dt/self.tau_i)

    # And adaptation also decays
    self.adaptation *= np.exp(-dt/self.tau_adaptation)

    return False  # Cannot spike during refractory period
```

**Key insight:** During refractory period:
- Voltage is clamped to $V_{reset}$
- But conductances and adaptation still evolve
- Synaptic inputs are "remembered" even though they can't cause a spike yet

---

## 8. Noise Terms

### Membrane Noise

$$V \leftarrow V + \eta_V, \quad \eta_V \sim \mathcal{N}(0, \sigma_V)$$

### Conductance Noise

$$g_e \leftarrow g_e + \max(0, \eta_g), \quad \eta_g \sim \mathcal{N}(0, \sigma_g)$$

$$g_i \leftarrow g_i + \max(0, \eta_g), \quad \eta_g \sim \mathcal{N}(0, \sigma_g)$$

### Why Add Noise?

**Biological realism:**
1. **Channel noise**: Ion channels open/close stochastically
2. **Background synaptic activity**: Thousands of synapses receiving random input
3. **Thermal fluctuations**: Molecular-level randomness

**Computational benefits:**
1. **Prevents synchronization artifacts**: Without noise, identical neurons behave identically
2. **Exploration**: Noise helps escape local minima in learning
3. **Regularization**: Adds robustness to the network

### Why Rectify Conductance Noise?

```python
self.g_e += e_noise if e_noise > 0 else 0
```

Conductances are **physically non-negative** - you can't have negative channel openings. The rectification ensures:
- Only positive noise is added
- Conductances never go negative
- Models spontaneous channel openings (not closings below zero)

### Code Implementation

**Membrane noise:**
```python
# Lines 112-115 in LIFNeuronWithReversal.py
v_noise = 0
if self.v_noise_amp > 0:
    v_noise = np.random.normal(0, self.v_noise_amp)
```

**Conductance noise:**
```python
# Lines 131-136 in LIFNeuronWithReversal.py
if self.i_noise_amp > 0:
    e_noise = np.random.normal(0, self.i_noise_amp)
    i_noise = np.random.normal(0, self.i_noise_amp)
    self.g_e += e_noise if e_noise > 0 else 0
    self.g_i += i_noise if i_noise > 0 else 0
```

---

## 9. Complete Update Algorithm

Here's the full algorithm executed each time step:

```
FUNCTION update(dt):

    1. Record current state to history

    2. Increment time since last spike: t_spike += dt

    3. IF in refractory period (t_spike < τ_ref):
        a. Hold voltage at V_reset
        b. Decay conductances: g_e *= exp(-dt/τ_e), g_i *= exp(-dt/τ_i)
        c. Decay adaptation: I_adapt *= exp(-dt/τ_adapt)
        d. Add rectified conductance noise
        e. RETURN False (no spike)

    4. Generate membrane noise: η_V ~ N(0, σ_V)

    5. Calculate synaptic currents:
        I_e = g_e × (E_e - V)
        I_i = g_i × (E_i - V)
        I_syn = I_e + I_i

    6. Calculate adaptation current:
        I_adapt = g_adapt × (E_K - V)

    7. Update membrane potential (Forward Euler):
        dV = dt × [-(V - V_rest)/τ_m + I_syn + I_adapt] + η_V
        V += dV

    8. Add rectified conductance noise

    9. Decay conductances and adaptation

    10. IF V ≥ V_thresh:
        a. Reset: V = V_reset
        b. Reset timer: t_spike = 0
        c. Increase adaptation: g_adapt += Δ_adapt
        d. Record spike time
        e. RETURN True (spike occurred)

    11. RETURN False (no spike)
```

---

## 10. Parameter Reference

| Parameter | Symbol | Code Variable | Default | Unit | Biological Interpretation |
|-----------|--------|---------------|---------|------|---------------------------|
| Resting potential | $V_{rest}$ | `v_rest` | -65 | mV | K+ equilibrium potential dominates |
| Threshold | $V_{thresh}$ | `v_threshold` | -55 | mV | Na+ channel activation threshold |
| Reset potential | $V_{reset}$ | `v_reset` | -75 | mV | Afterhyperpolarization |
| Membrane time constant | $\tau_m$ | `tau_m` | 10 | ms | RC time constant of membrane |
| Refractory period | $\tau_{ref}$ | `tau_ref` | 2 | ms | Na+ channel inactivation time |
| Excitatory time constant | $\tau_e$ | `tau_e` | 3 | ms | AMPA receptor kinetics |
| Inhibitory time constant | $\tau_i$ | `tau_i` | 7 | ms | GABA-A receptor kinetics |
| Excitatory reversal | $E_e$ | `e_reversal` | 0 | mV | Mixed Na+/K+ permeability |
| Inhibitory reversal | $E_i$ | `i_reversal` | -70 | mV | Cl- equilibrium potential |
| Potassium reversal | $E_K$ | `k_reversal` | -90 | mV | K+ equilibrium potential (adaptation) |
| Membrane noise | $\sigma_V$ | `v_noise_amp` | 0.5 | mV | Channel stochasticity |
| Conductance noise | $\sigma_g$ | `i_noise_amp` | 0.05 | - | Background synaptic activity |
| Adaptation increment | $\Delta_{adapt}$ | `adaptation_increment` | 0.5 | - | Ca2+-activated K+ current |
| Adaptation time constant | $\tau_{adapt}$ | `tau_adaptation` | 100 | ms | Slow K+ channel kinetics |

---

## 11. Key Insights

### Why Conductance-Based Over Current-Based?

**Current-based model:** $I_{syn} = w$ (constant current per spike)

**Conductance-based model:** $I_{syn} = g \cdot (E_{rev} - V)$

Advantages of conductance-based:
1. **Biological realism**: Matches how real synapses work
2. **Voltage-dependent**: Current depends on membrane state
3. **Self-limiting**: Cannot exceed reversal potentials
4. **Shunting inhibition**: Inhibition can reduce excitatory drive even without hyperpolarization

### The Interplay of Time Constants

The model has four time constants that create rich dynamics:

| Time Constant | Value | Effect |
|---------------|-------|--------|
| $\tau_m$ | 10 ms | Integration window for inputs |
| $\tau_e$ | 3 ms | Sharp, fast excitation |
| $\tau_i$ | 7 ms | Broader, sustained inhibition |
| $\tau_{adapt}$ | 100 ms | Slow firing rate modulation |

### Stability Considerations

For stable simulation:
- Use $\Delta t \ll \min(\tau_e, \tau_m)$
- Typical choice: $\Delta t = 0.1$ ms
- Conductance decay uses exact solution (always stable)
- Membrane equation uses Euler (stable for small $\Delta t$)

---

## 12. Mathematical Summary

A complete visual reference of the LIF neuron equations, presented in order of the model's operation.

---

### Membrane Potential Dynamics

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + I_{syn} + I_{adapt} + \eta_V$$

---

### Synaptic Currents

$$I_e = g_e \cdot (E_e - V)$$

$$I_i = g_i \cdot (E_i - V)$$

$$I_{syn} = I_e + I_i$$

---

### Conductance Dynamics

$$\frac{dg_e}{dt} = -\frac{g_e}{\tau_e}$$

$$\frac{dg_i}{dt} = -\frac{g_i}{\tau_i}$$

---

### Synaptic Input

$$g_e \leftarrow g_e + w \quad \text{(if } w > 0 \text{)}$$

$$g_i \leftarrow g_i + |w| \quad \text{(if } w < 0 \text{)}$$

---

### Total Conductances from Network

$$g_e(t) = \sum_{j \in \mathcal{E}} g_j(t)$$

$$g_i(t) = \sum_{k \in \mathcal{I}} g_k(t)$$

---

### Spike-Frequency Adaptation

$$\frac{dg_{adapt}}{dt} = -\frac{g_{adapt}}{\tau_{adapt}}$$

$$I_{adapt} = g_{adapt} \cdot (E_K - V)$$

$$g_{adapt} \leftarrow g_{adapt} + \Delta_{adapt} \quad \text{(on spike)}$$

---

### Spike Generation and Reset

$$\text{If } V \geq V_{thresh}: \quad V \leftarrow V_{reset}, \quad t_{spike} \leftarrow 0$$

---

### Refractory Period

$$\text{If } t_{spike} < \tau_{ref}: \quad \text{voltage integration blocked}$$

---

### Noise Terms

$$V \leftarrow V + \eta_V, \quad \eta_V \sim \mathcal{N}(0, \sigma_V)$$

$$g_e \leftarrow g_e + \max(0, \eta_g), \quad \eta_g \sim \mathcal{N}(0, \sigma_g)$$

$$g_i \leftarrow g_i + \max(0, \eta_g), \quad \eta_g \sim \mathcal{N}(0, \sigma_g)$$

---

### Forward Euler Discretization

$$V(t + \Delta t) = V(t) + \Delta t \cdot \frac{dV}{dt} + \eta_V$$

$$g_e(t + \Delta t) = g_e(t) \cdot e^{-\Delta t / \tau_e}$$

$$g_i(t + \Delta t) = g_i(t) \cdot e^{-\Delta t / \tau_i}$$
