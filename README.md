# Spiking Neural Networks as Statistical Machines

![vis_digit_0_example_structure copy](https://github.com/user-attachments/assets/13f5652c-879f-44b9-b498-9c60fdffd87c)

The aim of this repo is to develop my understanding on biologically plausible distributed networks. These neuron models are very different from the types found in artificial neural networks (ANNs) used in deep learning. They are called **Leaky-Integrate and Fire (LIF)** models. They compute over time, and are capable of integrating signals from many synaptic connections. LIF neurons model their own membrane potentials, and fire action potentials. Also --- I am very excited to be able to visualize these networks. I have not seen activity in spiking networks visualized in this way before, and it's quite interesting to watch!

Importantly, these specific LIF neurons are **conductance-based models**.
When a presynaptic neuron fires, it doesn't directly inject current. Instead, it causes a temporary increase in the conductance of specific class of ion channels (abstracted) on the postsynaptic neuron's membrane.

Excitatory inputs increase excitatory conductance (`g_e`).
Inhibitory inputs increase inhibitory conductance (`g_i`). These conductances increase rapidly upon spike arrival and then decay exponentially over time (governed by `tau_e` and `tau_i`, respectively). Reversal potentials are also implemented for both excitatory and inhibitory inputs.

The networks in this repo are sparsely connected, and have modifiable topologies (layered, circular, etc...). Activity in these networks is very dynamic and sometimes sparse. Parameters need to be chosen carefully to avoid sub-critical (silence) and super-critical (tonic firing, seizure-like) states. Neuron parameters are chosen to be as biologically plausible as possible. Network excitation/inhibition ratio and sparse connection probabilities are biologically plausible also.

**This project has several goals:**

1. **Build LIF neurons** and document their behavior (Achieved -- figures found in functional_diagnostic directory)

2. **Build networks** and record their activity in a variety of activity regimes (sub-critical, **critical**, super critical), with activity that can propagate through layers without dying off (Achieved -- figures can be found in experiment_layered directory)

3. Tune neuron/network params to produce networks with **critical** activity (Achieved -- using grid search on many hyper params, figures in experiment_circular directory)

4. Build layered networks **capable of learned computational tasks**, like digit classification -- initially weights will be selected for using genetic algorithms as a proof-of-concept -- IN PROGRESS

5. Implement a **learning paradigm** like spike-timing-dependent-plasticity (STDP) -- IN PROGRESS

*Real Readme coming soon.....*

![Network visualization showing sparse connections and node activity](https://github.com/user-attachments/assets/fee5d93d-233e-4405-a015-d074a1fd1ae4)

![Activity raster plot showing spikes over time for different neurons](https://github.com/user-attachments/assets/b932a870-d15c-4e6d-a424-508d8dde3f9e)



