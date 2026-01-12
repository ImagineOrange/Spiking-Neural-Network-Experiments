# Spiking Neural Networks as Statistical Machines

![vis_digit_0_example_structure copy](https://github.com/user-attachments/assets/13f5652c-879f-44b9-b498-9c60fdffd87c)

The aim of this repo is to develop my understanding on biologically plausible distributed networks. These neuron models are very different from the types found in artificial neural networks (ANNs) used in deep learning. They are called **Leaky-Integrate and Fire (LIF)** models. They compute over time, and are capable of integrating signals from many synaptic connections. LIF neurons model their own membrane potentials, and fire action potentials. Also --- I am very excited to be able to visualize these networks. I have not seen activity in spiking networks visualized in this way before, and it's quite interesting to watch!

Importantly, these specific LIF neurons are **conductance-based models**.
When a presynaptic neuron fires, it doesn't directly inject current. Instead, it causes a temporary increase in the conductance of specific class of ion channels (abstracted) on the postsynaptic neuron's membrane.

Excitatory inputs increase excitatory conductance (`g_e`).
Inhibitory inputs increase inhibitory conductance (`g_i`). These conductances increase rapidly upon spike arrival and then decay exponentially over time (governed by `tau_e` and `tau_i`, respectively). Reversal potentials are also implemented for both excitatory and inhibitory inputs.

The networks in this repo are sparsely connected, and have modifiable topologies (layered, circular, etc...). Activity in these networks is very dynamic and sometimes sparse. Parameters need to be chosen carefully to avoid sub-critical (silence) and super-critical (tonic firing, seizure-like) states. Neuron parameters are chosen to be as biologically plausible as possible. Network excitation/inhibition ratio and sparse connection probabilities are biologically plausible also.

**This project is organized into four phases:**

## Phase 0: Biologically Plausible LIF Neuron Models
Development and validation of conductance-based Leaky-Integrate-and-Fire (LIF) neurons. Includes functional diagnostics demonstrating subthreshold dynamics, action potential generation, membrane decay, spike-frequency adaptation, and conductance-based synaptic transmission. Contains scripts for characterizing single neuron behavior and short-term synaptic depression.

## Phase 1: Networks of LIF Neurons
Construction of sparse, recurrently connected networks with modifiable topologies (layered, circular). Exploration of network dynamics across activity regimes (sub-critical, critical, super-critical) with parameter tuning to achieve critical dynamics where activity propagates reliably through layers without dying off or becoming runaway excitation.

## Phase 2: Structured Input Experiments
Implementation of computational tasks using spiking networks, specifically MNIST digit classification. Uses genetic algorithms to evolve synaptic weights for networks with fixed topology. Includes multiple encoding strategies (intensity-to-rate, convolutional features), network architectures, and evaluation pipelines. Contains successful trained networks achieving 76-94% accuracy on 2-5 class subsets of MNIST.

## Phase 3: Learning Over Time
Migration toward biologically-plausible learning rules, specifically implementing Spike-Timing-Dependent Plasticity (STDP) based on the Diehl & Cook 2015 approach. Aims to replace genetic algorithm weight optimization with online learning through local synaptic plasticity rules. This phase represents a transition from our home-made neuron and network objects, and implements those objects directly from a legagacy codebase provided by the original paper in 2015. Online learning proved to be computationally intractable with our objects, so moving to a c-compiled simulation base (Brian2) was necessary. The codebase has been modernized for ease of use. 

![Network visualization showing sparse connections and node activity](https://github.com/user-attachments/assets/fee5d93d-233e-4405-a015-d074a1fd1ae4)





