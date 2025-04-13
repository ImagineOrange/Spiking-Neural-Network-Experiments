# Neuronal Modeling and Circuit Dynamics Project

The aim of this repo is to develop my understanding on neuronal modeling and circuit dynamics. These neuron models are very different from the types found in artificial neural networks (ANNs) used in deep learning. They are called **Leaky-Integrate and Fire (LIF)** models. They compute over time, and are capable of integrating signals from many synaptic connections. LIF neurons model their own membrane potentials, and fire action potentials.

Importantly, these specific LIF neurons are **conductance-based models**.
When a presynaptic neuron fires, it doesn't directly inject current. Instead, it causes a temporary increase in the conductance of specific types of ion channels on the postsynaptic neuron's membrane.

Excitatory inputs increase excitatory conductance (`g_e`).
Inhibitory inputs increase inhibitory conductance (`g_i`). These conductances increase rapidly upon spike arrival and then decay exponentially over time (governed by `tau_e` and `tau_i`, respectively). Reversal potentials are also implemented for both excitatory and inhibitory inputs.

The networks in this repo are sparsely connected, and have modifiable topologies (layered, circular, etc...). Activity in these networks is very dynamic, and parameters need to be chosen carefully to avoid sub-critical (silence) and super-critical (tonic firing, seizure-like) states. Neuron parameters are chosen to be as biologically plausible as possible. Network excitation/inhibition ratio and sparse connection probabilities are biologically plausible also.

This project has several goals:

1. <u>Build LIF neurons and document their behavior.</u>

2. <u>Build networks and record their activity in a variety of activity regimes (sub-critical, **critical**, super critical).</u>

3. <u>Tune neuron/network params to produce networks with **critical** activity.</u>

4. <u>Build layered networks capable of learned computational tasks, like digit classification -- initial 'learning' will be via genetic algorithms for a proof-of-concept.</u>

5. <u>Implement a learning paradigm like spike-timing-dependent-plasticity (STDP).</u>

*Real Readme coming soon.....*

![Network visualization showing sparse connections and node activity](https://github.com/user-attachments/assets/fee5d93d-233e-4405-a015-d074a1fd1ae4)

![Activity raster plot showing spikes over time for different neurons](https://github.com/user-attachments/assets/b932a870-d15c-4e6d-a424-508d8dde3f9e)
