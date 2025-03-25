import numpy as np

class LIFNeuronWithReversal:
    """
    Enhanced Leaky Integrate-and-Fire neuron model with reversal potentials and spike-frequency adaptation.
    
    This implementation includes:
    - Separate excitatory and inhibitory synaptic conductances
    - Reversal potentials for excitatory and inhibitory inputs
    - Membrane potential noise and synaptic input noise
    - Spike-frequency adaptation to prevent tonic firing
    """
    def __init__(self, v_rest=-65.0, v_threshold=-55.0, v_reset=-75.0, 
                 tau_m=10.0, tau_ref=2.0, tau_e=3.0, tau_i=7.0, is_inhibitory=False,
                 e_reversal=0.0, i_reversal=-70.0, v_noise_amp=0.5, i_noise_amp=0.05,
                 adaptation_increment=0.5, tau_adaptation=100.0):
        # Membrane parameters
        self.v_rest = v_rest          # Resting potential (mV)
        self.v_threshold = v_threshold  # Threshold potential (mV)
        self.v_reset = v_reset        # Reset potential after spike (mV)
        self.tau_m = tau_m            # Membrane time constant (ms)
        self.tau_ref = tau_ref        # Refractory period (ms)
        
        # Synaptic time constants
        self.tau_e = tau_e            # Excitatory synaptic time constant (ms)
        self.tau_i = tau_i            # Inhibitory synaptic time constant (ms)
        
        # Reversal potentials
        self.e_reversal = e_reversal  # Excitatory reversal potential (mV)
        self.i_reversal = i_reversal  # Inhibitory reversal potential (mV)
        
        # Noise parameters
        self.v_noise_amp = v_noise_amp  # Membrane potential noise amplitude (mV)
        self.i_noise_amp = i_noise_amp  # Synaptic current noise amplitude
        
        # Adaptation parameters
        self.adaptation_increment = adaptation_increment  # How much adaptation increases per spike
        self.tau_adaptation = tau_adaptation  # Adaptation time constant (ms)
        
        # State variables
        self.v = v_rest               # Current membrane potential (mV)
        self.g_e = 0.0                # Excitatory conductance
        self.g_i = 0.0                # Inhibitory conductance
        self.adaptation = 0.0         # Adaptation current (initially zero)
        self.t_since_spike = tau_ref  # Time since last spike (ms)
        self.is_inhibitory = is_inhibitory
        
        # Recording variables
        self.v_history = []
        self.g_e_history = []
        self.g_i_history = []
        self.adaptation_history = []
        self.i_syn_history = []       # Total synaptic current for backward compatibility
        self.spike_times = []
    
    def reset(self):
        """Reset neuron to initial state."""
        self.v = self.v_rest
        self.g_e = 0.0
        self.g_i = 0.0
        self.adaptation = 0.0
        self.t_since_spike = self.tau_ref
        self.v_history = []
        self.g_e_history = []
        self.g_i_history = []
        self.adaptation_history = []
        self.i_syn_history = []
        self.spike_times = []
    
    def update(self, dt):
        """
        Update neuron state by one time step.
        Returns True if a spike occurred, False otherwise.
        """
        # Record current state if tracking is enabled
        if hasattr(self, 'v_history'):
            self.v_history.append(self.v)
            self.g_e_history.append(self.g_e)
            self.g_i_history.append(self.g_i)
            self.adaptation_history.append(self.adaptation)
            
            # Calculate total synaptic current for backward compatibility
            i_syn = self.g_e * (self.e_reversal - self.v) + self.g_i * (self.i_reversal - self.v)
            self.i_syn_history.append(i_syn)
        
        # Update time since last spike
        self.t_since_spike += dt
        
        # If in refractory period, no voltage integration
        if self.t_since_spike < self.tau_ref:
            # During refractory period, voltage stays at reset value
            self.v = self.v_reset
            
            # But synaptic conductances still decay
            self.g_e *= np.exp(-dt/self.tau_e)
            self.g_i *= np.exp(-dt/self.tau_i)
            
            # And adaptation also decays
            self.adaptation *= np.exp(-dt/self.tau_adaptation)
            
            # Add synaptic noise even during refractory period
            if self.i_noise_amp > 0:
                e_noise = np.random.normal(0, self.i_noise_amp)
                i_noise = np.random.normal(0, self.i_noise_amp)
                self.g_e += e_noise if e_noise > 0 else 0  # Only add positive noise to excitatory
                self.g_i += i_noise if i_noise > 0 else 0  # Only add positive noise to inhibitory
                
            return False
        
        # Add membrane potential noise (if enabled)
        v_noise = 0
        if self.v_noise_amp > 0:
            v_noise = np.random.normal(0, self.v_noise_amp)
        
        # Calculate synaptic currents based on conductances and reversal potentials
        i_e = self.g_e * (self.e_reversal - self.v)  # Excitatory current
        i_i = self.g_i * (self.i_reversal - self.v)  # Inhibitory current
        i_total = i_e + i_i
        
        # Integrate membrane potential (leaky integration with conductance-based inputs)
        # The adaptation current is subtracted here, acting as an additional hyperpolarizing current
        dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_total - self.adaptation) + v_noise
        self.v += dv
        
        # Add synaptic conductance noise
        if self.i_noise_amp > 0:
            e_noise = np.random.normal(0, self.i_noise_amp)
            i_noise = np.random.normal(0, self.i_noise_amp)
            self.g_e += e_noise if e_noise > 0 else 0  # Only add positive noise to excitatory
            self.g_i += i_noise if i_noise > 0 else 0  # Only add positive noise to inhibitory
        
        # Decay synaptic conductances
        self.g_e *= np.exp(-dt/self.tau_e)
        self.g_i *= np.exp(-dt/self.tau_i)
        
        # Decay adaptation current
        self.adaptation *= np.exp(-dt/self.tau_adaptation)
        
        # Check for spike
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self.t_since_spike = 0.0
            
            # Increase adaptation current when spike occurs
            self.adaptation += self.adaptation_increment
            
            if hasattr(self, 'spike_times'):
                self.spike_times.append(len(self.v_history) * dt)
            return True
        
        return False
    
    def receive_spike(self, weight):
        """Receive spike from presynaptic neuron with given weight."""
        # Positive weight -> excitatory, Negative weight -> inhibitory
        if weight > 0:
            self.g_e += weight  # Increase excitatory conductance
        else:
            self.g_i += -weight  # Increase inhibitory conductance (weight is negative)
        
    def stimulate(self, current):
        """
        Directly stimulate neuron with given current.
        Positive current is treated as excitatory.
        """
        if current > 0:
            self.g_e += current
        else:
            self.g_i += -current