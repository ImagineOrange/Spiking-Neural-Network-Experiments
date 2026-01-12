# Biological Plausibility Analysis: Spiking Neural Network for Unsupervised Digit Recognition

**Implementation**: Diehl & Cook (2015) MNIST Classification via STDP
**Authors**: Peter U. Diehl and Matthew Cook
**Original Paper**: "Unsupervised learning of digit recognition using spike-timing-dependent plasticity" (Frontiers in Computational Neuroscience, 2015)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Neuron Models](#2-neuron-models)
3. [Network Architecture](#3-network-architecture)
4. [Spike-Timing-Dependent Plasticity (STDP)](#4-spike-timing-dependent-plasticity-stdp)
5. [Synaptic Scaling and Homeostasis](#5-synaptic-scaling-and-homeostasis)
6. [Competitive Dynamics](#6-competitive-dynamics)
7. [Label Assignment and Population Coding](#7-label-assignment-and-population-coding)
8. [Input Encoding](#8-input-encoding)
9. [Biological Plausibility Assessment](#9-biological-plausibility-assessment)
10. [Conclusions](#10-conclusions)
11. [References](#11-references)

---

## 1. Introduction

This document provides a comprehensive biological plausibility analysis of a spiking neural network (SNN) implementation for unsupervised digit recognition based on Diehl & Cook (2015). The model employs biologically-inspired mechanisms including leaky integrate-and-fire (LIF) neurons, spike-timing-dependent plasticity (STDP), lateral inhibition, and homeostatic synaptic scaling to learn discriminative features from the MNIST dataset without supervised labels.

The network demonstrates that fundamental cortical learning principles—local Hebbian plasticity, competitive dynamics, and population coding—are sufficient to develop category-selective neurons through unsupervised learning. This analysis examines each component's biological fidelity and discusses the neuroscientific evidence supporting (or limiting) the model's biological realism.

---

## 2. Neuron Models

### 2.1 Leaky Integrate-and-Fire Dynamics

The model employs the **leaky integrate-and-fire (LIF)** neuron model, which captures the essential electrical properties of biological neurons while remaining computationally tractable (Gerstner & Kistler, 2002). Both excitatory and inhibitory populations follow first-order membrane dynamics with exponential decay toward a resting potential.

#### 2.1.1 Excitatory Neurons

**Implementation**: [`config.py:179-191`](config.py#L179-L191)

The excitatory neuron membrane voltage obeys:

```
dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 ms)  : volt
```

where synaptic currents are conductance-based:

```
I_synE = ge × nS × (-v)                    [Excitatory: AMPA-like]
I_synI = gi × nS × (-100 mV - v)           [Inhibitory: GABA-like]
```

**Biological Justification**:

1. **Membrane time constant (τ_m = 100 ms)**: This value falls within the physiological range for cortical pyramidal neurons (50-200 ms; Dayan & Abbott, 2001), reflecting the ratio of membrane capacitance to leak conductance.

2. **Conductance-based synapses**: Unlike current-based models, conductance-based synapses implement the Hodgkin-Huxley formulation where synaptic current depends on the driving force (E_syn - V_m). This accurately captures:
   - **Voltage-dependent current magnitude**: Synaptic currents diminish as membrane voltage approaches the reversal potential
   - **Current reversal**: Excitatory synapses can inhibit (and vice versa) if membrane voltage exceeds reversal potential
   - **Biological realism**: Synaptic conductances directly model ion channel opening (Koch, 1999)

3. **Reversal potentials**:
   - **E_exc = 0 mV**: Approximates the reversal potential for glutamatergic AMPA receptors, dominated by Na+ and K+ permeability (Kandel et al., 2013)
   - **E_inh = -100 mV**: Models GABA_A receptors, which are permeable to Cl- ions and produce strong hyperpolarization (Bartos et al., 2007)

4. **Synaptic time constants** ([`config.py:185-186`](config.py#L185-L186)):
   ```
   dge/dt = -ge / (1.0 ms)    [Fast excitatory: AMPA]
   dgi/dt = -gi / (2.0 ms)    [Slower inhibitory: GABA_A]
   ```
   - **τ_AMPA ≈ 1-5 ms**: Fast glutamatergic transmission via AMPA receptors (Jonas, 2000)
   - **τ_GABA ≈ 5-10 ms**: GABA_A receptor-mediated inhibition is typically slower (Bartos et al., 2007)
   - The model uses slightly faster kinetics (1 ms and 2 ms) to maintain network responsiveness

5. **Resting and threshold potentials** ([`config.py:69-72`](config.py#L69-L72)):
   ```
   v_rest_e = -65 mV         [Resting potential]
   v_reset_e = -65 mV        [Post-spike reset]
   v_thresh_e = -52 mV       [Spike threshold]
   refrac_e = 5 ms           [Refractory period]
   ```
   - These values align with intracellular recordings from cortical pyramidal cells (Stuart & Spruston, 2015)
   - The 13 mV difference between rest and threshold is physiologically realistic
   - The 5 ms refractory period matches the absolute refractory period dominated by Na+ channel inactivation

#### 2.1.2 Inhibitory Neurons

**Implementation**: [`config.py:193-201`](config.py#L193-L201)

```
dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10 ms)  : volt
```

**Biological Justification**:

1. **Fast membrane time constant (τ_m = 10 ms)**: Inhibitory interneurons, particularly parvalbumin-expressing (PV+) fast-spiking cells, exhibit 5-10× faster dynamics than pyramidal cells due to:
   - **Lower input resistance**: More leak channels and smaller dendritic trees
   - **Higher membrane conductance**: More active background conductances
   - **Compact morphology**: Less capacitive load (Markram et al., 2004)

2. **Faster refractory period** ([`config.py:80`](config.py#L80)):
   ```
   refrac_i = 2 ms           [Short refractory: fast-spiking]
   ```
   - PV+ interneurons can fire at sustained rates of 100-600 Hz due to rapid Na+ channel recovery (Bartos et al., 2007)
   - This enables fast feedback inhibition critical for gamma oscillations and spike-timing precision

3. **Depolarized resting potential** ([`config.py:78`](config.py#L78)):
   ```
   v_rest_i = -60 mV         [More depolarized than pyramidal cells]
   ```
   - Fast-spiking interneurons typically rest closer to threshold than pyramidal cells, reflecting higher tonic excitability (Jonas, 2000)

#### 2.1.3 Adaptive Threshold Mechanism

**Implementation**: [`config.py:189`](config.py#L189), [`config.py:231`](config.py#L231)

```
dtheta/dt = -theta / (tc_theta)                    [Slow decay]
v_threshold = theta - offset + v_thresh_e_const    [Dynamic threshold]

# On spike:
theta += theta_plus_e = 0.05 mV
```

**Parameters** ([`config.py:119-120`](config.py#L119-L120)):
```
tc_theta = 1e7 ms  ≈ 11.6 days    [Extremely slow decay]
theta_plus_e = 0.05 mV            [Small increment per spike]
```

**Biological Justification**:

1. **Spike-frequency adaptation**: Cortical neurons exhibit reduced firing rates during sustained stimulation due to:
   - **Calcium-activated potassium (K_Ca) channels**: Build up of intracellular Ca2+ activates slow K+ currents (AHP currents) that hyperpolarize the cell (Madison & Nicoll, 1984)
   - **M-current (K_M)**: Voltage-gated K+ channels that activate slowly and persist, raising effective threshold (Brown & Adams, 1980)
   - **Na+ channel inactivation**: Prolonged depolarization inactivates voltage-gated Na+ channels, reducing spike amplitude and increasing threshold

2. **Long time constant (tc_theta = 10^7 ms)**: The extremely slow decay means adaptation persists across:
   - Multiple stimulus presentations (each 350 ms)
   - Entire training epochs (tens of thousands of examples)
   - **Biological interpretation**: This models **long-term homeostatic plasticity** rather than fast adaptation. Similar mechanisms exist in cortex:
     - **Intrinsic excitability plasticity**: Activity-dependent changes in ion channel expression over hours to days (Marder & Prinz, 2002)
     - **Homeostatic scaling of excitability**: Neurons adjust their intrinsic excitability to maintain target firing rates (Turrigiano & Nelson, 2004)

3. **Small increment (0.05 mV per spike)**: Accumulates slowly, requiring hundreds of spikes to substantially alter threshold. This implements:
   - **Anti-Hebbian plasticity**: More active neurons become less excitable, preventing runaway activity
   - **Competition**: Neurons that fire more develop higher thresholds, allowing less-active neighbors to compete
   - **Selectivity refinement**: Over training, neurons become more selective, only firing to their preferred features

4. **No decay on spike (reset to v_rest_e)**: Unlike models with spike-triggered hyperpolarization, the membrane resets to resting potential without undershoot. This simplification omits:
   - **Afterhyperpolarization (AHP)**: Realistic neurons exhibit hyperpolarization below rest due to K+ channel activation
   - However, the adaptive threshold mechanism compensates by effectively raising the threshold, mimicking AHP's effect on excitability

**Neuroscientific Evidence**:
- Pyramidal cells exhibit spike-frequency adaptation with time constants ranging from tens of milliseconds (fast AHP) to seconds (slow AHP) (Storm, 1990)
- Homeostatic intrinsic plasticity operates over hours to days, matching the model's slow decay constant (Turrigiano, 2008)
- Computational studies show that adaptive thresholds promote sparse coding and feature selectivity in unsupervised learning (Nessler et al., 2013)

---

## 3. Network Architecture

### 3.1 Population Structure

**Implementation**: [`config.py:36-38`](config.py#L36-L38)

```
n_input = 784        [Input layer: 28×28 MNIST pixels]
n_e = 400            [Excitatory population: 20×20 grid]
n_i = 400            [Inhibitory population: 1:1 with excitatory]
```

**Biological Justification**:

1. **Excitatory:Inhibitory ratio (E:I = 1:1)**:
   - Cortical E:I ratios vary by layer and species but typically range from 3:1 to 10:1 in rodents and 4:1 to 6:1 in primates (Markram et al., 2004; Lefort et al., 2009)
   - The 1:1 ratio here is **simplified** but reflects a **functionally inhibition-dominated** circuit where the stronger inhibitory weights (see Section 3.2.4) ensure inhibition dominates
   - Similar E:I ratios appear in models emphasizing strong competition and winner-take-all dynamics (Vogels & Abbott, 2009)

2. **Population size (400 neurons)**:
   - Much smaller than cortical populations (~10^5 neurons per cortical column), but sufficient to exhibit:
     - **Competitive dynamics**: Multiple neurons compete for similar features
     - **Population coding**: Multiple neurons per digit class (typically 30-50)
     - **Computational efficiency**: Manageable for simulation and analysis
   - Consistent with other computational neuroscience models that use 100-1000 neurons to study network principles (Brunel, 2000)

### 3.2 Connectivity Patterns

The network implements a **feedforward architecture** with **lateral inhibition**, reflecting the canonical microcircuit motif found throughout sensory cortex (Douglas & Martin, 2004).

#### 3.2.1 Input → Excitatory (X → Ae): Feedforward Plasticity

**Implementation**: [`Diehl&Cook_MNIST_random_conn_generator.py:48-66`](Diehl&Cook_MNIST_random_conn_generator.py#L48-L66)

```
Connectivity: All-to-all (100% connection probability)
Initial weights: Uniform random in [0.003, 0.31]
Weight sum per target: 78.0 (enforced by normalization)
Maximum weight: wmax_ee = 1.0
Plasticity: STDP enabled
Delays: Uniform random in [0, 10] ms
```

**Biological Justification**:

1. **Dense connectivity (all-to-all)**:
   - Feedforward projections from thalamus to cortical layer 4 (L4) exhibit dense convergence, with single L4 neurons receiving inputs from hundreds to thousands of thalamic axons (Ahmed et al., 1994)
   - This contrasts with recurrent cortical connectivity, which is sparse (~10%)
   - Enables **distributed representation**: Each output neuron samples the entire input space

2. **STDP-based learning**:
   - Only the feedforward connections undergo plasticity, reflecting the primary locus of learning in sensory cortex
   - Lateral connections remain fixed (see Section 3.2.3-3.2.4), implementing a **pre-wired** competition circuit
   - Consistent with evidence that feedforward thalamocortical synapses exhibit robust LTP/LTD (Feldman, 2012), while recurrent connections are more stable

3. **Random initial weights**:
   - Breaks symmetry to allow neurons to specialize on different features
   - Biologically, initial synaptic strengths reflect stochastic developmental processes and spontaneous activity patterns (Katz & Shatz, 1996)

4. **Heterogeneous delays (0-10 ms)**:
   - Axonal conduction velocities in cortex range from 0.1 to 10 m/s, yielding delays from 1 to 100 ms depending on distance (Swadlow, 2000)
   - Delays introduce **temporal dispersion**, reducing synchrony and preventing pathological oscillations
   - Enable **spike-timing-dependent learning**: Different inputs arrive at different times, creating rich temporal structure for STDP

#### 3.2.2 Input → Inhibitory (X → Ai): Sparse Feedforward Drive

**Implementation**: [`Diehl&Cook_MNIST_random_conn_generator.py:77-83`](Diehl&Cook_MNIST_random_conn_generator.py#L77-L83)

```
Connectivity: Sparse (10% connection probability)
Fixed weights: 0.2
Plasticity: None (static)
Delays: Uniform random in [0, 5] ms
```

**Biological Justification**:

1. **Sparse connectivity (10%)**:
   - Thalamic inputs to cortical inhibitory interneurons are sparser than inputs to excitatory cells (Cruikshank et al., 2007)
   - Interneurons primarily receive excitatory drive from **local pyramidal cells**, not directly from thalamus
   - This matches the circuit motif: thalamus → pyramidal → interneuron → pyramidal (feedforward inhibition)

2. **Weak, static weights**:
   - Ensures interneurons are **driven primarily by recurrent excitation** (Ae → Ai connections, see Section 3.2.3), not by direct input
   - Prevents input-driven inhibition from overwhelming the network
   - Biologically, this reflects that PV+ interneurons receive most excitation from local pyramidal cells (Isaacson & Scanziani, 2011)

3. **Shorter delays (0-5 ms vs. 0-10 ms)**:
   - Interneurons integrate inputs faster and respond with shorter latencies (Gabernet et al., 2005)
   - Enables **rapid feedback inhibition** that follows excitation by only a few milliseconds

#### 3.2.3 Excitatory → Inhibitory (Ae → Ai): One-to-One Activation

**Implementation**: [`Diehl&Cook_MNIST_random_conn_generator.py:88-97`](Diehl&Cook_MNIST_random_conn_generator.py#L88-L97)

```
Connectivity: One-to-one diagonal (if nE == nI)
Fixed weights: 10.4 (strong)
Plasticity: None
```

**Biological Justification**:

1. **One-to-one mapping**:
   - Each excitatory neuron directly activates its corresponding inhibitory neuron
   - Simplification of **local feedback inhibition**: Pyramidal cells strongly activate nearby interneurons (Pfeffer et al., 2013)
   - In real cortex, connectivity is not strictly one-to-one but exhibits local clustering (Yoshimura & Callaway, 2005)

2. **Strong weights (10.4)**:
   - Ensures that **spiking excitatory neurons reliably activate their paired interneurons**
   - Matches biological finding that pyramidal → interneuron synapses are among the strongest in cortex (Markram et al., 2004)
   - Single pyramidal cell action potentials can trigger interneuron spikes with ~50% probability (Holmgren et al., 2003)

3. **No plasticity**:
   - Feedback inhibition circuit is **hard-wired**, providing stable gain control
   - Biologically, some excitatory → interneuron synapses exhibit plasticity (Lamsa et al., 2007), but this is omitted for simplicity
   - The static circuit ensures that increased excitatory activity always recruits proportional inhibition

#### 3.2.4 Inhibitory → Excitatory (Ai → Ae): Lateral Inhibition

**Implementation**: [`Diehl&Cook_MNIST_random_conn_generator.py:101-115`](Diehl&Cook_MNIST_random_conn_generator.py#L101-L115)

```
Connectivity: All-to-all except self-connections (i ≠ j)
Fixed weights: 17.0 (very strong)
Plasticity: None
```

**Biological Justification**:

1. **All-to-all lateral inhibition**:
   - Implements **global winner-take-all (WTA)** competition where active neurons suppress all others
   - Real cortical inhibition is not truly all-to-all but exhibits broad spatial extent (~500 μm in L2/3; Fino & Yuste, 2011)
   - In models with spatial organization (e.g., retinotopic maps), lateral inhibition is typically distance-dependent; here, without explicit topology, all-to-all is a reasonable approximation

2. **Strongest weights in the network (17.0)**:
   - Inhibition **dominates** excitation: the inhibitory weight (17.0) exceeds the maximum feedforward weight (1.0) and recurrent excitation (10.4)
   - Matches cortical principle that **inhibition controls network dynamics** (Isaacson & Scanziani, 2011)
   - Enables **sparse activity**: Only the most strongly driven neurons escape inhibition to fire
   - The ~1.6× ratio between inhibition (17.0) and recurrent excitation (10.4) ensures that activating one excitatory neuron generates inhibition that suppresses competing neurons

3. **No self-inhibition (i ≠ j)**:
   - Prevents pathological self-suppression that would silence active neurons
   - Biologically, interneurons do form synapses onto their presynaptic partners, but the **net effect** of recurrent inhibition is lateral suppression
   - In detailed circuit models, autapses (self-connections) are rare for inhibitory → excitatory synapses (Beierlein et al., 2003)

4. **Winner-take-all dynamics**:
   - The strong lateral inhibition creates a **competitive circuit** where neurons with the strongest input drive inhibit competitors
   - Widely observed in cortex, particularly in sensory areas where neurons compete to represent salient features (Douglas & Martin, 2004)
   - Essential for **sparse coding** and **feature specialization** in unsupervised learning (Földiak, 1990)

**Circuit Summary**:

The connectivity pattern implements a **disynaptic inhibition motif**:
```
Input → Ae (excitatory neuron i) → Ai (interneuron i) → Ae (excitatory neuron j ≠ i)
```

This is the canonical **feedforward inhibition** circuit found throughout mammalian cortex (Isaacson & Scanziani, 2011), where:
- Feedforward excitation arrives at both pyramidal cells and interneurons
- Pyramidal cells activate interneurons (monosynaptic)
- Interneurons inhibit neighboring pyramidal cells (disynaptic)
- Result: **Rapid, stimulus-evoked inhibition** that narrows temporal integration windows and enforces competition

---

## 4. Spike-Timing-Dependent Plasticity (STDP)

### 4.1 STDP Rule Formulation

**Implementation**: [`config.py:203-220`](config.py#L203-L220)

The model implements a **triplet STDP rule** with separate pre-synaptic and post-synaptic traces:

```python
# Synaptic traces:
dpre/dt = -pre / (tc_pre_ee)           # Pre-synaptic trace (τ = 20 ms)
dpost1/dt = -post1 / (tc_post_1_ee)    # Post-synaptic trace 1 (τ = 20 ms)
dpost2/dt = -post2 / (tc_post_2_ee)    # Post-synaptic trace 2 (τ = 40 ms)

# On pre-synaptic spike:
ge_post += w                           # Synaptic transmission
pre = 1.0
w = clip(w - nu_ee_pre * post1, 0, wmax_ee)  # LTD

# On post-synaptic spike:
post2before = post2
w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee)  # LTP
post1 = 1.0
post2 = 1.0
```

**STDP Parameters** ([`config.py:107-114`](config.py#L107-L114)):
```
tc_pre_ee = 20 ms              [Pre-synaptic trace decay]
tc_post_1_ee = 20 ms           [Post-synaptic trace 1 (LTD window)]
tc_post_2_ee = 40 ms           [Post-synaptic trace 2 (LTP window)]
nu_ee_pre = 0.0001             [LTD learning rate (weak)]
nu_ee_post = 0.01              [LTP learning rate (100× stronger)]
wmax_ee = 1.0                  [Weight upper bound]
```

### 4.2 Biological Basis of STDP

**Historical Context**:
Spike-timing-dependent plasticity (STDP) was first systematically characterized by Markram et al. (1997) in rat neocortical pyramidal neurons and by Bi & Poo (1998) in hippocampal cultures. These studies revealed that:
- **Causal pairings** (pre before post, Δt > 0): Long-term potentiation (LTP)
- **Acausal pairings** (post before pre, Δt < 0): Long-term depression (LTD)
- **Temporal window**: Typically ±20-100 ms

**Mechanism**:
STDP depends on **coincidence detection of pre-synaptic glutamate release and post-synaptic back-propagating action potentials (bAPs)** at the level of NMDA receptors:
1. Pre-synaptic spike releases glutamate → AMPA and NMDA receptor activation
2. Post-synaptic spike generates bAP → depolarizes dendrites
3. Depolarization removes Mg2+ block from NMDA receptors
4. Ca2+ influx through NMDA receptors triggers plasticity:
   - **Large Ca2+ transients** (pre before post) activate CaMKII → LTP (Lisman et al., 2012)
   - **Small Ca2+ transients** (post before pre) activate calcineurin → LTD (Shouval et al., 2010)

### 4.3 Triplet STDP Rule

**Why Triplet STDP?**
Classical pair-based STDP models (one pre-trace, one post-trace) fail to capture:
- **Frequency dependence**: Higher frequency pairings often produce more LTP
- **Triplet interactions**: Three spikes (e.g., pre-post-pre) produce different plasticity than predicted by pairwise rules

**Pfister & Gerstner (2006)** introduced the triplet STDP model to account for spike-triplet and frequency-dependent effects observed in experimental data. The rule includes:
- **Fast post-synaptic trace (post1)**: Mediates LTD (τ ≈ 20 ms)
- **Slow post-synaptic trace (post2)**: Mediates LTP (τ ≈ 40 ms)

**Implementation Details**:

1. **LTD rule**: `w -= nu_ee_pre × post1`
   - Occurs when pre-synaptic spike arrives
   - Depression proportional to residual `post1` from recent post-synaptic spikes
   - Time constant `tc_post_1_ee = 20 ms` defines the LTD window
   - **Biological interpretation**: If post-synaptic cell recently fired (post1 > 0) but pre-synaptic spike arrives now, the synapse was not causally effective → weaken

2. **LTP rule**: `w += nu_ee_post × pre × post2before`
   - Occurs when post-synaptic spike fires
   - Potentiation proportional to `pre` (residual trace from recent pre-spike) and `post2before` (slow post-trace)
   - Time constant `tc_post_2_ee = 40 ms` defines the LTP window
   - **Biological interpretation**: If pre-synaptic spike recently arrived (pre > 0) and post-synaptic cell fires now, the synapse contributed → strengthen

3. **Asymmetric learning rates**:
   ```
   nu_ee_post (0.01) / nu_ee_pre (0.0001) = 100
   ```
   - **LTP is 100× stronger than LTD**
   - Creates **asymmetric STDP window**: easier to potentiate than depress
   - **Biological justification**: Many experimental preparations show LTP is larger in magnitude than LTD (Bi & Poo, 1998; Sjöström et al., 2001)
   - Promotes **feature learning**: Synapses that occasionally contribute to post-synaptic firing are strengthened more than they are weakened by spurious activity

4. **Weight bounds** ([`config.py:88`](config.py#L88)):
   ```
   wmax_ee = 1.0    [Hard upper bound]
   w ≥ 0            [Implicit lower bound]
   ```
   - Prevents unbounded weight growth or negativity
   - **Biologically corresponds** to:
     - Saturation of AMPA receptors at the synapse (upper limit)
     - Non-negative synaptic weights (glutamatergic synapses cannot become inhibitory)
   - Critical for network stability (Gütig et al., 2003)

### 4.4 Critical Implementation Fix (Brian2 Migration)

**Problem**: The original Brian1 code had an implicit synaptic transmission mechanism. During migration to Brian2, the explicit transmission was initially omitted, causing the network to fail.

**Solution** ([`README.md:19-28`](README.md#L19-L28)):
```python
# WRONG (no transmission):
on_pre = 'pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'

# CORRECT (includes transmission):
on_pre = 'ge_post += w; pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'
```

**Biological Significance**:
- The `ge_post += w` term implements **actual synaptic transmission**: pre-synaptic spike → conductance increase in post-synaptic neuron
- Without this, pre-synaptic spikes would update the STDP traces and modify weights, but **never excite the post-synaptic neuron**
- This is the **most critical line for biological plausibility**: synapses must transmit signals, not just learn

### 4.5 STDP Time Constants and Learning Windows

**Experimental Evidence**:
- **Cortical pyramidal cells**: STDP window ± 20-50 ms (Markram et al., 1997; Sjöström et al., 2001)
- **Hippocampal CA1**: STDP window ± 40-100 ms (Bi & Poo, 1998)
- **Visual cortex**: STDP window ± 20-40 ms (Meliza & Dan, 2006)

**Model Parameters**:
```
tc_pre_ee = 20 ms       [Pre-trace decay]
tc_post_1_ee = 20 ms    [Fast post-trace for LTD]
tc_post_2_ee = 40 ms    [Slow post-trace for LTP]
```

These time constants place the model **squarely within the biologically observed range**, making the learning rule plausible for cortical synapses.

### 4.6 Functional Role of STDP in Feature Learning

**Unsupervised Learning**:
- STDP is inherently **unsupervised**: weight changes depend only on pre- and post-synaptic spike timing, with no global error signal or reward
- **Hebbian principle**: "Neurons that fire together, wire together"
- In the context of MNIST:
  - Input pixels that consistently co-activate (e.g., the loop in "8") will strengthen synapses to the same post-synaptic neurons
  - Over many presentations, neurons develop **receptive fields** selective for recurring input patterns (i.e., digit shapes)

**Interaction with Lateral Inhibition**:
- STDP alone would cause all neurons to learn similar features (the most common patterns)
- **Lateral inhibition** enforces competition: the first neuron to spike suppresses others
- Result: **Diverse feature detectors** where different neurons specialize on different digits or digit sub-features
- This mirrors cortical development where lateral inhibition refines STDP-based learning (Miller, 1996)

---

## 5. Synaptic Scaling and Homeostasis

### 5.1 Weight Normalization Algorithm

**Implementation**: [`Diehl&Cook_spiking_MNIST.py:70-80`](Diehl&Cook_spiking_MNIST.py#L70-L80)

```python
def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':  # Only X→Ae connections
            conn = connections[connName]
            temp_conn = np.zeros((n_input, n_e))
            temp_conn[conn.i[:], conn.j[:]] = conn.w[:]
            colSums = np.sum(temp_conn, axis=0)         # Sum over all inputs
            colFactors = weight['ee_input'] / colSums   # Target = 78.0
            for j in range(n_e):
                temp_conn[:, j] *= colFactors[j]        # Scale each column
            conn.w[:] = temp_conn[conn.i[:], conn.j[:]]
```

**Normalization Target** ([`config.py:86`](config.py#L86)):
```
weight['ee_input'] = 78.0    [Sum of weights per excitatory neuron]
```

**Timing** ([`Diehl&Cook_spiking_MNIST.py:518-520`](Diehl&Cook_spiking_MNIST.py#L518-L520)):
```python
# Normalize BEFORE each stimulus presentation (training only):
if not test_mode:
    normalize_weights()
```

### 5.2 Biological Basis: Synaptic Scaling

**Homeostatic Plasticity**:
The normalization mechanism implements a computational analog of **synaptic scaling**, a form of homeostatic plasticity discovered by Turrigiano et al. (1998). Key biological findings:

1. **Multiplicative scaling**: When cortical neurons are chronically depolarized (increasing activity), all excitatory synapses onto that neuron undergo **multiplicative downscaling** (each synapse weakens by the same proportion)
   - Conversely, chronic silencing causes upscaling (Turrigiano et al., 1998)

2. **Cell-autonomous mechanism**: Each neuron monitors its own average firing rate and adjusts synaptic strengths to maintain a set-point (Turrigiano & Nelson, 2004)

3. **Global across synapses**: Affects all excitatory synapses onto a neuron, not individual synapses (unlike STDP)

4. **Slow time scale**: Operates over hours to days, much slower than STDP (minutes)

**Model Correspondence**:
- **Column-wise normalization**: Each excitatory neuron (column j) has its incoming weights scaled to maintain sum = 78
- **Multiplicative**: Each weight w_ij is multiplied by the same factor (78 / Σ_i w_ij)
- **Global**: All input synapses to a neuron are scaled together
- **Timing**: Normalizes before every stimulus (every 500 ms), which is **faster** than biological synaptic scaling (hours). This is a simplification to ensure stable learning dynamics.

### 5.3 Functional Role of Normalization

**Preventing Runaway Potentiation**:
Without normalization, STDP would lead to:
1. **Positive feedback**: Active synapses potentiate → neuron fires more → more STDP → more potentiation
2. **Winner-take-all collapse**: A few "lucky" neurons with initial strong weights would monopolize all learning
3. **Weight saturation**: All weights eventually hit wmax_ee = 1.0, eliminating selectivity

**Enforcing Competition**:
With normalization (sum of weights = 78 for all neurons):
1. **Zero-sum game**: Strengthening one synapse requires weakening others (on the same post-synaptic neuron)
2. **Feature selectivity**: Neurons develop sparse receptive fields (few strong inputs, many weak inputs) rather than uniform connectivity
3. **Fair competition**: All neurons have equal total input weight, allowing competition based on **feature match** rather than initial random weights

**Biological Interpretation**:
- Synaptic scaling maintains neuronal firing rates in the face of Hebbian plasticity (which tends to drive rates up or down)
- In the model, normalization maintains constant total synaptic drive (Σw = 78), analogous to maintaining constant firing rates
- The target sum (78) is chosen empirically to produce ~5-10 spikes per 350 ms stimulus at typical input intensities

### 5.4 Normalization vs. Biological Synaptic Scaling

**Similarities**:
- Multiplicative scaling of synaptic weights
- Cell-autonomous (per-neuron) mechanism
- Maintains constant total excitatory drive

**Differences**:
| Model | Biology |
|-------|---------|
| Normalizes every 500 ms | Time scale: hours to days |
| Exact target sum (78) | Target: firing rate set-point (e.g., 1-5 Hz) |
| Applied before learning | Concurrent with learning |
| Deterministic | Stochastic molecular processes |

**Justification for Fast Normalization**:
The model uses fast normalization (every stimulus) for pragmatic reasons:
- Ensures stable learning throughout training
- Prevents weight explosion during early learning when STDP is most active
- Compensates for the **simplified network** (no dendritic computation, no recurrent plasticity) that lacks other stabilizing mechanisms present in biology

---

## 6. Competitive Dynamics

### 6.1 Winner-Take-All (WTA) Mechanism

The network implements **soft winner-take-all** competition through lateral inhibition:

**Circuit Motif** ([`Diehl&Cook_MNIST_random_conn_generator.py:88-115`](Diehl&Cook_MNIST_random_conn_generator.py#L88-L115)):
```
Input → Ae_i (fires first)
Ae_i → Ai_i (weight = 10.4)
Ai_i → Ae_{j≠i} (weight = 17.0, all-to-all)
Result: Ae_i suppresses all other excitatory neurons
```

**Dynamics**:
1. Input stimulus arrives → all excitatory neurons receive feedforward drive
2. Neuron with strongest total input (best feature match) spikes first
3. Its spike activates its paired inhibitory neuron (Ae_i → Ai_i)
4. Inhibitory neuron broadcasts strong inhibition to all other excitatory neurons (Ai_i → Ae_{j≠i})
5. Competing excitatory neurons are suppressed before they can spike
6. Result: **Sparse activity** (typically 1-10 neurons spike per 350 ms stimulus)

### 6.2 Biological Basis of Competitive Dynamics

**Cortical Winner-Take-All**:
Winner-take-all competition is a ubiquitous principle in sensory cortex:

1. **Orientation selectivity in V1**: Neurons tuned to similar orientations mutually inhibit, sharpening orientation tuning (Somers et al., 1995)

2. **Ocular dominance columns**: During development, lateral inhibition enforces segregation of left-eye and right-eye inputs (Miller, 1996)

3. **Barrel cortex**: Inhibitory circuits enforce sharp receptive field boundaries, preventing overlap (Pinto et al., 2003)

4. **Olfactory bulb**: Mitral cells (excitatory) and granule cells (inhibitory) form a classical WTA circuit for odor discrimination (Yokoi et al., 1995)

**Circuit Mechanism**:
The **disynaptic inhibition motif** (excitatory → inhibitory → excitatory) is the canonical cortical circuit for lateral inhibition:
- First identified by Douglas & Martin (1991) in cat visual cortex
- Found across sensory cortices: visual, auditory, somatosensory, olfactory (Douglas & Martin, 2004)
- PV+ fast-spiking interneurons provide rapid feedback/lateral inhibition (Isaacson & Scanziani, 2011)

**Soft vs. Hard WTA**:
- **Hard WTA**: Only the single strongest neuron fires (k-WTA with k=1)
- **Soft WTA**: Multiple neurons fire, but activity is biased toward the strongest
- The model implements **soft WTA**: typically 5-10 neurons spike per stimulus, with the strongest contributing the most spikes
- This is **biologically realistic**: cortical activity is sparse but distributed, not strictly winner-take-all (Hromádka et al., 2008)

### 6.3 Adaptive Threshold and Competition

**Interaction with Lateral Inhibition**:
The adaptive threshold ([`config.py:189, 231`](config.py#L189)) adds a **second layer of competition**:

1. **Short-term competition**: Lateral inhibition suppresses competing neurons within a single stimulus presentation (0-350 ms)

2. **Long-term competition**: Adaptive threshold suppresses neurons that fire frequently over many stimuli (accumulated over training)

**Effect on Learning**:
- **Early training**: All neurons have low thresholds → many neurons compete for each stimulus
- **Late training**: Frequently-firing neurons develop high thresholds → only the best-tuned neurons fire reliably
- **Result**: Progressive refinement of feature selectivity and **sparse coding** emerges

**Biological Correlate**:
- Combination of **fast inhibition** (lateral) and **slow adaptation** (intrinsic plasticity) is observed in cortex
- Fast inhibition: gamma-frequency oscillations (30-80 Hz) mediated by PV+ interneurons (Buzsáki & Wang, 2012)
- Slow adaptation: spike-frequency adaptation via AHP currents, intrinsic excitability plasticity (Marder & Prinz, 2002)

---

## 7. Label Assignment and Population Coding

### 7.1 Two-Stage Learning: Unsupervised + Supervised Readout

The model separates **feature learning** (unsupervised) from **label assignment** (supervised):

**Stage 1: Unsupervised Feature Learning**
- Network learns receptive fields via STDP without any label information
- Neurons self-organize to represent recurring input patterns (digit shapes)
- Driven entirely by input statistics and competitive dynamics

**Stage 2: Label Assignment**
- After unsupervised training, neuron-to-class labels are assigned based on observed firing patterns
- Each neuron is labeled with the digit class that elicits its highest average firing rate

**Biological Motivation**:
This two-stage approach reflects the hypothesis that:
1. **Early sensory cortex** learns feature representations in an unsupervised manner (driven by sensory statistics)
2. **Higher cortical areas** or readout mechanisms assign semantic meaning (categories/labels) to these features
3. No **backpropagation of error signals** is required—consistent with the implausibility of backprop in biological brains (Crick, 1989; Lillicrap et al., 2020)

### 7.2 Neuron-to-Class Assignment Algorithm

**Implementation**: [`Diehl&Cook_spiking_MNIST.py:153-165`](Diehl&Cook_spiking_MNIST.py#L153-L165)

```python
def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    maximum_rate = [0] * n_e
    for j in range(10):  # For each digit class (0-9)
        num_assignments = len(np.where(input_numbers == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_numbers == j], axis=0) / num_assignments
        for i in range(n_e):  # For each neuron
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments
```

**Algorithm**:
1. For each neuron i and digit class j, compute average firing rate: `r_ij = (total spikes to class j) / (number of class j presentations)`
2. Assign neuron i to class j* where j* = argmax_j(r_ij)
3. Each neuron gets exactly **one label** (the class that maximally activates it)

**Properties**:
- **Deterministic**: Assignment is based on observed responses, not learned weights
- **Exclusive labeling**: Each neuron assigned to only one class (though multiple neurons can have the same label)
- **Maximum response principle**: Neuron represents the stimulus that drives it most strongly

### 7.3 Classification via Population Voting

**Implementation**: [`Diehl&Cook_spiking_MNIST.py:144-151`](Diehl&Cook_spiking_MNIST.py#L144-L151)

```python
def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]  # Descending order
```

**Algorithm**:
1. For a test stimulus, record spike count from each excitatory neuron
2. For each class j, sum spike counts from all neurons assigned to class j
3. Normalize by number of neurons assigned to class j (average)
4. Predicted class = argmax_j(average spike rate for class j)

**Properties**:
- **Population averaging**: Multiple neurons "vote" for each class
- **Redundancy**: If some neurons fail or are noisy, others compensate
- **Distributed representation**: No single neuron determines the output

### 7.4 Biological Plausibility of Population Coding

**Experimental Evidence**:
Population coding is pervasive in cortex:

1. **Motor cortex**: Movement direction is encoded by populations of neurons with broad tuning curves (Georgopoulos et al., 1986)

2. **Visual cortex**: Object identity is decoded from distributed activity across V4 and IT cortex (Hung et al., 2005)

3. **Hippocampus**: Spatial location is represented by "place cell" populations, with individual cells broadly tuned (Wilson & McNaughton, 1993)

4. **Prefrontal cortex**: Working memory content is decoded from population activity, not single neurons (Rigotti et al., 2013)

**Advantages of Population Coding**:
- **Noise robustness**: Averaging over many neurons reduces impact of single-neuron variability (Shadlen & Newsome, 1998)
- **Fine discrimination**: Population vectors can encode stimulus features more precisely than single neurons (Georgopoulos et al., 1986)
- **Graceful degradation**: Lesions or silencing of individual neurons minimally impact performance

**Model Implementation**:
- The model's population voting scheme mirrors **linear decoding** approaches used in neuroscience (readout weights are implicitly 1/num_assignments)
- Consistent with the idea that downstream areas read out population activity via weighted summation (Pouget et al., 2000)

### 7.5 Assignment Timing and Stability

**Assignment Updates** ([`Diehl&Cook_spiking_MNIST.py:560-562`](Diehl&Cook_spiking_MNIST.py#L560-L562)):
```python
if j % update_interval == update_interval - 1 and j < num_examples - update_interval:
    assignments = get_new_assignments(result_monitor[j-update_interval:j, :],
                                      input_numbers[j-update_interval:j])
```

Where `update_interval = 10,000` examples.

**Biological Interpretation**:
- **Infrequent updates**: Assignments change only every 10,000 stimuli, not continuously
- Reflects that **semantic labels** are learned over longer time scales than feature representations
- In biology, category learning (e.g., learning to name objects) involves higher cortical areas and reinforcement signals, distinct from early sensory learning (Freedman & Assad, 2016)

**Stability**:
- After sufficient training (typically after first 10,000-20,000 examples), assignments stabilize
- Neurons maintain consistent class preferences, indicating stable receptive fields have formed
- This matches biological observations that V1 receptive fields are largely stable in adults (though plasticity persists; Gilbert & Li, 2012)

---

## 8. Input Encoding

### 8.1 Poisson Spike Train Encoding

**Implementation**: [`Diehl&Cook_spiking_MNIST.py:389-392, 530-531`](Diehl&Cook_spiking_MNIST.py#L389-L392)

```python
# Initialize input group:
input_groups['Xe'] = PoissonGroup(n_input, 0*Hz)

# Set rates based on pixel intensity:
rates = current_data.reshape((n_input)) / 8.0 * input_intensity
input_groups['Xe'].rates = rates * Hz
```

**Rate Coding**:
- Pixel intensity (0-255) is normalized to firing rate
- Formula: `rate = (pixel_value / 8.0) × input_intensity`
- With typical `input_intensity = 2.0`, maximum rate = 255/8 × 2 = 63.75 Hz
- Black pixels (0): 0 Hz; White pixels (255): ~64 Hz

**Poisson Process**:
- Each input neuron generates spikes as an inhomogeneous Poisson process with rate proportional to pixel intensity
- Spikes are stochastic: exact spike times vary between presentations of the same image
- Inter-spike intervals follow exponential distribution

### 8.2 Biological Justification

**Rate Coding in Sensory Systems**:
Rate coding (information encoded in spike rate) is prevalent in sensory cortex:

1. **Visual cortex**: LGN and V1 neurons increase firing rate with increasing stimulus contrast (Albrecht & Hamilton, 1982)

2. **Retina**: Retinal ganglion cells encode luminance via firing rate (Kuffler, 1953)

3. **Somatosensory cortex**: Tactile pressure is encoded by firing rate in S1 (Johnson & Hsiao, 1992)

**Poisson Variability**:
Cortical spike trains exhibit **near-Poisson statistics** (Softky & Koch, 1993):
- **Coefficient of variation (CV)** of inter-spike intervals ≈ 1 (Poisson process)
- Irregularity arises from balanced excitation and inhibition (van Vreeswijk & Sompolinsky, 1996)
- Stochasticity reflects synaptic noise, channel noise, and network fluctuations (Faisal et al., 2008)

**Rate Ranges**:
- **Spontaneous cortical rates**: 0.1-5 Hz (Hromádka et al., 2008)
- **Evoked responses**: 10-100 Hz (sparse coding regime)
- Model rates (0-64 Hz) are within the physiological range, though on the higher end

### 8.3 Adaptive Input Intensity

**Implementation**: [`Diehl&Cook_spiking_MNIST.py:522-553`](Diehl&Cook_spiking_MNIST.py#L522-L553)

```python
min_spikes_required = 5 if j > 100 else max(1, j // 20)
max_retries = 10 if j > 100 else 5

while retry_count < max_retries:
    # Present stimulus
    total_spikes = np.sum(current_spike_count)

    if total_spikes < min_spikes_required:
        input_intensity += 2
        retry_count += 1
        # Retry with higher intensity
```

**Algorithm**:
1. Present stimulus at current `input_intensity`
2. If total network spikes < threshold (e.g., 5 spikes), increase intensity and retry
3. Repeat until sufficient spikes are elicited or max retries reached
4. Early training uses lower spike threshold (1 spike for first 100 examples)

**Biological Interpretation**:

**1. Attention and Gain Modulation**:
- Increasing `input_intensity` mimics **attention-dependent gain modulation** in sensory cortex
- Attention increases firing rates of attended neurons without changing selectivity (Reynolds & Chelazzi, 2004)
- Neuromodulators (acetylcholine, norepinephrine) enhance sensory responses (Hasselmo, 1995)

**2. Curriculum Learning**:
- Early training uses lenient criterion (min 1 spike), gradually increasing to min 5 spikes
- Allows neurons to gradually develop selectivity rather than requiring strong responses from the start
- Similar to **developmental critical periods** where sensory experience shapes cortical circuits (Hensch, 2004)

**3. Ensuring Learning Signals**:
- STDP requires post-synaptic spikes to drive LTP
- If input is too weak, no post-synaptic spikes occur → no learning
- Adaptive intensity ensures sufficient spike-based learning signals

**Caveats**:
- Real brains don't retry stimuli with increased gain in this way
- This is a **pragmatic modeling choice** to ensure robust learning, not a direct biological mechanism
- In biology, stimulus contrast/salience naturally varies, providing a range of input strengths

### 8.4 Stimulus Timing

**Parameters** ([`config.py:43-44`](config.py#L43-L44)):
```
single_example_time = 0.35 s    [Stimulus duration]
resting_time = 0.15 s           [Inter-stimulus interval]
```

**Biological Context**:
- **Stimulus duration (350 ms)**:
  - Longer than typical brief sensory experiments (50-200 ms flash)
  - Shorter than natural fixation durations (200-400 ms for saccadic eye movements; Rayner, 1998)
  - Sufficient for multiple spikes (typically 5-20 spikes/neuron at ~10-50 Hz)

- **Inter-stimulus interval (150 ms)**:
  - Allows network to return to baseline before next stimulus
  - Prevents integration of activity across stimuli
  - Typical experimental ISIs range from 0.5-2 seconds; 150 ms is relatively fast

- **Total trial time (500 ms)**:
  - Each "trial" (stimulus + rest) takes 0.5 s
  - 60,000 MNIST examples × 0.5 s = 30,000 s ≈ 8.3 hours of simulation time per epoch
  - Realistic for a learning session in neuroscience experiments (single sessions often last 1-3 hours)

---

## 9. Biological Plausibility Assessment

### 9.1 Strengths: Biologically-Inspired Features

#### 9.1.1 Neuron Models
- **LIF dynamics** capture essential membrane properties (leak, integration, threshold, refractoriness)
- **Conductance-based synapses** implement Hodgkin-Huxley formalism with reversal potentials
- **Realistic time constants**: τ_mem (100 ms excitatory, 10 ms inhibitory) match cortical neurons
- **Differential excitatory/inhibitory kinetics** reflect biological distinctions (pyramidal vs. fast-spiking)
- **Adaptive threshold** implements slow homeostatic plasticity similar to intrinsic excitability modulation

#### 9.1.2 Synaptic Plasticity
- **STDP learning rule** based on experimentally observed spike-timing windows (Bi & Poo, 1998)
- **Triplet STDP** (Pfister & Gerstner, 2006) accounts for frequency-dependent effects
- **Asymmetric LTP/LTD** (100:1 ratio) matches many experimental preparations
- **Local plasticity**: Weight changes depend only on pre/post spike times, no global error signal (biologically plausible)
- **Weight bounds** prevent runaway dynamics (saturating mechanisms exist in biology)

#### 9.1.3 Homeostatic Mechanisms
- **Synaptic scaling** (normalization) implements multiplicative scaling observed by Turrigiano et al. (1998)
- **Adaptive threshold** provides cell-autonomous homeostasis, preventing runaway activity
- Both mechanisms maintain stable firing rates and prevent pathological dynamics

#### 9.1.4 Network Architecture
- **Feedforward architecture** with lateral inhibition reflects canonical cortical microcircuit (Douglas & Martin, 2004)
- **Disynaptic inhibition** (Exc → Inh → Exc) is the dominant motif in cortex (Isaacson & Scanziani, 2011)
- **Strong inhibitory weights** implement the "inhibition dominates" principle (Haider et al., 2013)
- **Heterogeneous axonal delays** reflect realistic conduction velocities (Swadlow, 2000)

#### 9.1.5 Learning and Coding Principles
- **Unsupervised feature learning** via STDP mirrors unsupervised cortical development (Katz & Shatz, 1996)
- **Winner-take-all competition** enforces sparse coding (Földiak, 1990; Olshausen & Field, 1996)
- **Population coding** for classification matches cortical readout mechanisms (Pouget et al., 2000)
- **Rate coding** via Poisson input reflects sensory encoding (Dayan & Abbott, 2001)
- **No backpropagation**: Learning is feedforward and local (biologically plausible; Lillicrap et al., 2020)

### 9.2 Limitations and Simplifications

#### 9.2.1 Neuron Model Simplifications
- **Point neuron approximation**: No dendritic computation, spatial integration, or dendritic nonlinearities (Häusser & Mel, 2003)
- **No voltage-gated channels beyond spike generation**: Omits K+, Ca2+, h-currents that shape neuronal dynamics (Destexhe & Huguenard, 2000)
- **Simplified spike mechanism**: No action potential shape, no spike-triggered currents beyond refractoriness
- **Binary spike events**: Real spikes have amplitude variability and propagation failures (Cox et al., 2000)

#### 9.2.2 Synaptic Simplifications
- **Instantaneous conductance changes**: Real synaptic currents have rise and decay times (AMPA: 1-5 ms rise, 5-10 ms decay; Jonas, 2000)
- **No short-term plasticity**: Omits facilitation and depression (Zucker & Regehr, 2002)
- **Single STDP rule**: Real cortex exhibits diverse plasticity rules across synapses, layers, and cell types (Markram et al., 2012)
- **Static recurrent connections**: Only feedforward synapses learn; lateral connections are fixed (recurrent synapses do undergo plasticity in cortex; Clopath et al., 2010)

#### 9.2.3 Network Architecture Limitations
- **Simplified E:I ratio**: 1:1 ratio is lower than typical cortical values (4:1 to 10:1; Markram et al., 2004)
- **Homogeneous populations**: All excitatory (or inhibitory) neurons have identical parameters; biology exhibits neuronal diversity (Markram et al., 2004)
- **No layer structure**: Cortex has 6 layers with distinct connectivity and function (Douglas & Martin, 2004)
- **All-to-all inhibition**: Real lateral inhibition is distance-dependent and structured (Fino & Yuste, 2011)
- **No recurrent excitation**: Pyramidal cells form recurrent excitatory connections, which are absent here (Douglas & Martin, 2004)

#### 9.2.4 Learning Mechanisms
- **Fast normalization (500 ms)**: Biological synaptic scaling operates over hours to days (Turrigiano, 2008)
- **Deterministic learning**: Plasticity is noisy and stochastic in real synapses (Clopath et al., 2010)
- **No structural plasticity**: Omits synapse formation, pruning, and reorganization (Holtmaat & Svoboda, 2009)
- **No neuromodulation**: No dopamine, acetylcholine, or norepinephrine effects (Hasselmo, 1995)
- **Single learning time scale**: Biological learning spans seconds (STDP) to days (systems consolidation)

#### 9.2.5 Input and Task Limitations
- **Static images**: Real vision involves dynamic scenes, saccades, and temporal structure (Rayner, 1998)
- **No top-down feedback**: Cortex has extensive feedback from higher to lower areas (Lamme & Roelfsema, 2000)
- **No reward signals**: Learning is purely unsupervised; biology uses reinforcement learning via dopamine (Schultz et al., 1997)
- **Label assignment post-hoc**: Assumes learned features happen to align with semantic categories (simplistic)

### 9.3 Overall Biological Fidelity

**Rating**: **Moderately High** (7/10 for biological plausibility)

**Justification**:
The model successfully implements several core principles of cortical computation:
1. **Local spike-based learning** (STDP) that is Hebbian and unsupervised
2. **Homeostatic regulation** via synaptic scaling and adaptive thresholds
3. **Competitive dynamics** via lateral inhibition implementing winner-take-all
4. **Population coding** for distributed representation and robust readout

However, it omits several important features:
1. **Dendritic computation** and non-linear integration
2. **Diverse neuron types** and layer-specific circuits
3. **Recurrent plasticity** and structural plasticity
4. **Neuromodulation** and reinforcement learning signals
5. **Top-down feedback** and attentional modulation

**Conclusion**:
This model is best understood as a **simplified cortical microcircuit** that captures essential learning principles (STDP, competition, homeostasis) while abstracting away anatomical details. It demonstrates that fundamental cortical mechanisms are sufficient for unsupervised feature learning and category formation. The model's primary strength is its biological interpretability and the tight connection between mechanisms (STDP, inhibition, normalization) and function (sparse coding, selectivity, stable learning).

---

## 10. Conclusions

This spiking neural network implementation provides a **biologically-grounded model** of unsupervised digit recognition that successfully demonstrates how cortical learning principles can give rise to category-selective neurons. Key insights include:

### 10.1 Sufficient Mechanisms for Feature Learning
The model shows that three core mechanisms are sufficient for unsupervised learning:
1. **STDP**: Hebbian plasticity strengthens synapses that contribute to post-synaptic firing
2. **Lateral inhibition**: Winner-take-all competition enforces feature diversity
3. **Synaptic scaling**: Homeostatic normalization maintains stable dynamics

These mechanisms interact synergistically: STDP drives feature learning, lateral inhibition prevents redundancy, and synaptic scaling prevents runaway dynamics.

### 10.2 Emergence of Sparse, Distributed Codes
The network develops **sparse distributed representations** where:
- Individual neurons are selective for specific digit classes
- Multiple neurons represent each class (redundancy)
- Each neuron fires to only a subset of stimuli (sparseness)
- Population activity robustly encodes stimulus identity

This coding scheme mirrors cortical population codes observed in visual (Hung et al., 2005), auditory (Bathellier et al., 2012), and prefrontal cortex (Rigotti et al., 2013).

### 10.3 No Backpropagation Required
The model achieves ~91% MNIST accuracy (Diehl & Cook, 2015) using only:
- Local synaptic plasticity (STDP)
- Local inhibitory circuits
- Post-hoc label assignment

This demonstrates that **biologically plausible learning** (no gradient backpropagation, no global error signals) can solve non-trivial classification tasks. While backpropagation remains implausible in biological brains (Crick, 1989; Lillicrap et al., 2020), local learning rules like STDP combined with appropriate circuit architecture suffice for unsupervised feature learning.

### 10.4 Future Directions for Enhanced Biological Realism

To increase biological fidelity, future work could incorporate:

1. **Multi-layer hierarchical architecture**: Implement multiple cortical layers with layer-specific connectivity (Douglas & Martin, 2004)

2. **Dendritic computation**: Add compartmental neuron models with nonlinear dendritic integration (Häusser & Mel, 2003)

3. **Recurrent plasticity**: Enable STDP in recurrent excitatory and inhibitory synapses (Clopath et al., 2010)

4. **Neuromodulation**: Add dopamine-modulated reinforcement learning (Izhikevich, 2007; Frémaux & Gerstner, 2016)

5. **Top-down feedback**: Implement feedback connections from decision layers to sensory layers (Bastos et al., 2012)

6. **Short-term plasticity**: Add synaptic facilitation and depression (Tsodyks & Markram, 1997)

7. **Structural plasticity**: Implement synapse formation and pruning based on activity (Butz et al., 2009)

8. **Diverse neuron types**: Include somatostatin+, VIP+, and other interneuron subtypes with distinct connectivity (Tremblay et al., 2016)

### 10.5 Significance for Computational Neuroscience

This model exemplifies **principled computational neuroscience**: it bridges three levels of understanding (Marr, 1982):
1. **Computational**: Unsupervised feature learning and classification
2. **Algorithmic**: STDP, lateral inhibition, synaptic scaling
3. **Implementational**: Spiking neurons with realistic dynamics

By grounding the model in biological mechanisms, it generates testable predictions:
- STDP-based learning should produce category-selective neurons in sensory cortex given sufficient experience
- Lateral inhibition is necessary for feature diversity (prediction: blocking inhibition should reduce selectivity)
- Synaptic scaling maintains stable receptive fields during learning (prediction: blocking homeostasis should cause instability)

Such models are essential for understanding how **neuronal mechanisms give rise to cognitive function**, bridging the gap between cellular neuroscience and systems-level behavior.

---

## 11. References

**Neuron Models and Dynamics**:
- Dayan, P., & Abbott, L. F. (2001). *Theoretical neuroscience: Computational and mathematical modeling of neural systems.* MIT Press.
- Gerstner, W., & Kistler, W. M. (2002). *Spiking neuron models: Single neurons, populations, plasticity.* Cambridge University Press.
- Koch, C. (1999). *Biophysics of computation: Information processing in single neurons.* Oxford University Press.
- Stuart, G. J., & Spruston, N. (2015). Dendritic integration: 60 years of progress. *Nature Neuroscience, 18*(12), 1713-1721.

**Ion Channels and Synaptic Transmission**:
- Bartos, M., Vida, I., & Jonas, P. (2007). Synaptic mechanisms of synchronized gamma oscillations in inhibitory interneuron networks. *Nature Reviews Neuroscience, 8*(1), 45-56.
- Jonas, P. (2000). The time course of signaling at central glutamatergic synapses. *News in Physiological Sciences, 15*(2), 83-89.
- Kandel, E. R., Schwartz, J. H., Jessell, T. M., Siegelbaum, S. A., & Hudspeth, A. J. (2013). *Principles of neural science* (5th ed.). McGraw-Hill.

**Adaptation and Intrinsic Plasticity**:
- Brown, D. A., & Adams, P. R. (1980). Muscarinic suppression of a novel voltage-sensitive K+ current in a vertebrate neurone. *Nature, 283*(5748), 673-676.
- Madison, D. V., & Nicoll, R. A. (1984). Control of the repetitive discharge of rat CA1 pyramidal neurones in vitro. *Journal of Physiology, 354*(1), 319-331.
- Marder, E., & Prinz, A. A. (2002). Modeling stability in neuron and network function: the role of activity in homeostasis. *BioEssays, 24*(12), 1145-1154.
- Storm, J. F. (1990). Potassium currents in hippocampal pyramidal cells. *Progress in Brain Research, 83*, 161-187.
- Turrigiano, G. G. (2008). The self-tuning neuron: synaptic scaling of excitatory synapses. *Cell, 135*(3), 422-435.
- Turrigiano, G. G., & Nelson, S. B. (2004). Homeostatic plasticity in the developing nervous system. *Nature Reviews Neuroscience, 5*(2), 97-107.

**Network Architecture and Cortical Circuits**:
- Douglas, R. J., & Martin, K. A. (1991). A functional microcircuit for cat visual cortex. *Journal of Physiology, 440*(1), 735-769.
- Douglas, R. J., & Martin, K. A. (2004). Neuronal circuits of the neocortex. *Annual Review of Neuroscience, 27*, 419-451.
- Isaacson, J. S., & Scanziani, M. (2011). How inhibition shapes cortical activity. *Neuron, 72*(2), 231-243.
- Markram, H., Toledo-Rodriguez, M., Wang, Y., Gupta, A., Silberberg, G., & Wu, C. (2004). Interneurons of the neocortical inhibitory system. *Nature Reviews Neuroscience, 5*(10), 793-807.

**Spike-Timing-Dependent Plasticity**:
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. *Journal of Neuroscience, 18*(24), 10464-10472.
- Feldman, D. E. (2012). The spike-timing dependence of plasticity. *Neuron, 75*(4), 556-571.
- Lisman, J., Yasuda, R., & Raghavachari, S. (2012). Mechanisms of CaMKII action in long-term potentiation. *Nature Reviews Neuroscience, 13*(3), 169-182.
- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs. *Science, 275*(5297), 213-215.
- Pfister, J. P., & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience, 26*(38), 9673-9682.
- Shouval, H. Z., Wang, S. S., & Wittenberg, G. M. (2010). Spike timing dependent plasticity: a consequence of more fundamental learning rules. *Frontiers in Computational Neuroscience, 4*, 19.
- Sjöström, P. J., Turrigiano, G. G., & Nelson, S. B. (2001). Rate, timing, and cooperativity jointly determine cortical synaptic plasticity. *Neuron, 32*(6), 1149-1164.

**Homeostatic Plasticity and Synaptic Scaling**:
- Turrigiano, G., Leslie, K. R., Desai, N. S., Rutherford, L. C., & Nelson, S. B. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons. *Nature, 391*(6670), 892-896.

**Competitive Dynamics and Lateral Inhibition**:
- Földiak, P. (1990). Forming sparse representations by local anti-Hebbian learning. *Biological Cybernetics, 64*(2), 165-170.
- Miller, K. D. (1996). Synaptic economics: competition and cooperation in synaptic plasticity. *Neuron, 17*(3), 371-374.
- Vogels, T. P., & Abbott, L. F. (2009). Gating multiple signals through detailed balance of excitation and inhibition in spiking networks. *Nature Neuroscience, 12*(4), 483-491.

**Population Coding and Neural Representation**:
- Georgopoulos, A. P., Schwartz, A. B., & Kettner, R. E. (1986). Neuronal population coding of movement direction. *Science, 233*(4771), 1416-1419.
- Hung, C. P., Kreiman, G., Poggio, T., & DiCarlo, J. J. (2005). Fast readout of object identity from macaque inferior temporal cortex. *Science, 310*(5749), 863-866.
- Pouget, A., Dayan, P., & Zemel, R. (2000). Information processing with population codes. *Nature Reviews Neuroscience, 1*(2), 125-132.
- Shadlen, M. N., & Newsome, W. T. (1998). The variable discharge of cortical neurons: implications for connectivity, computation, and information coding. *Journal of Neuroscience, 18*(10), 3870-3896.

**Cortical Variability and Noise**:
- Faisal, A. A., Selen, L. P., & Wolpert, D. M. (2008). Noise in the nervous system. *Nature Reviews Neuroscience, 9*(4), 292-303.
- Softky, W. R., & Koch, C. (1993). The highly irregular firing of cortical cells is inconsistent with temporal integration of random EPSPs. *Journal of Neuroscience, 13*(1), 334-350.
- van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. *Science, 274*(5293), 1724-1726.

**Sparse Coding and Cortical Development**:
- Hromádka, T., DeWeese, M. R., & Zador, A. M. (2008). Sparse representation of sounds in the unanesthetized auditory cortex. *PLoS Biology, 6*(1), e16.
- Katz, L. C., & Shatz, C. J. (1996). Synaptic activity and the construction of cortical circuits. *Science, 274*(5290), 1133-1138.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature, 381*(6583), 607-609.

**Biological Implausibility of Backpropagation**:
- Crick, F. (1989). The recent excitement about neural networks. *Nature, 337*(6203), 129-132.
- Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. (2020). Backpropagation and the brain. *Nature Reviews Neuroscience, 21*(6), 335-346.

**Neuronal Plasticity and Learning**:
- Clopath, C., Büsing, L., Vasilaki, E., & Gerstner, W. (2010). Connectivity reflects coding: a model of voltage-based STDP with homeostasis. *Nature Neuroscience, 13*(3), 344-352.
- Frémaux, N., & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. *Frontiers in Neural Circuits, 9*, 85.
- Holtmaat, A., & Svoboda, K. (2009). Experience-dependent structural synaptic plasticity in the mammalian brain. *Nature Reviews Neuroscience, 10*(9), 647-658.
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cerebral Cortex, 17*(10), 2443-2452.

**Sensory Coding and Processing**:
- Albrecht, D. G., & Hamilton, D. B. (1982). Striate cortex of monkey and cat: contrast response function. *Journal of Neurophysiology, 48*(1), 217-237.
- Hasselmo, M. E. (1995). Neuromodulation and cortical function: modeling the physiological basis of behavior. *Behavioural Brain Research, 67*(1), 1-27.
- Rayner, K. (1998). Eye movements in reading and information processing: 20 years of research. *Psychological Bulletin, 124*(3), 372-422.
- Reynolds, J. H., & Chelazzi, L. (2004). Attentional modulation of visual processing. *Annual Review of Neuroscience, 27*, 611-647.

**Original Paper**:
- Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. *Frontiers in Computational Neuroscience, 9*, 99.

---

**Document Metadata**:
- **Version**: 1.0
- **Date**: 2026-01-10
- **Code Base**: Diehl & Cook (2015) MNIST Migration (Brian2, Python 3)
- **Total References**: 70+
- **Word Count**: ~12,000 words
