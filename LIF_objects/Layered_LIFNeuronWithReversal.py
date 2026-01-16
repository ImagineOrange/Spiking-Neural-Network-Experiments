import numpy as np


class Layered_LIFNeuronWithReversal:
    """
    Represents a Leaky Integrate-and-Fire (LIF) neuron model incorporating:
    - Separate excitatory (g_e) and inhibitory (g_i) conductances.
    - Reversal potentials (e_reversal, i_reversal) for synaptic currents.
    - Membrane potential noise (v_noise_amp).
    - Synaptic input noise (i_noise_amp).
    - Spike-frequency adaptation via conductance-based K+ current (adaptation_increment, tau_adaptation, e_k_reversal).
    - Refractory period (tau_ref).
    - External stimulation via conductance change (external_stim_g).
    """
    def __init__(self, v_rest=-65.0, v_threshold=-55.0, v_reset=-75.0,
                 tau_m=10.0, tau_ref=2.0, tau_e=3.0, tau_i=7.0, is_inhibitory=False,
                 e_reversal=0.0, i_reversal=-70.0, v_noise_amp=0.5, i_noise_amp=0.05,
                 adaptation_increment=0.3, tau_adaptation=100.0, e_k_reversal=-90.0):
        """
        Initializes the LIF neuron with specified parameters.

        Args:
            v_rest (float): Resting membrane potential (mV).
            v_threshold (float): Spike threshold potential (mV).
            v_reset (float): Reset potential after a spike (mV).
            tau_m (float): Membrane time constant (ms).
            tau_ref (float): Absolute refractory period (ms).
            tau_e (float): Excitatory synapse time constant (ms).
            tau_i (float): Inhibitory synapse time constant (ms).
            is_inhibitory (bool): True if the neuron is inhibitory, False otherwise.
            e_reversal (float): Reversal potential for excitatory synapses (mV).
            i_reversal (float): Reversal potential for inhibitory synapses (mV).
            v_noise_amp (float): Amplitude (std dev) of Gaussian noise added to membrane potential (mV).
            i_noise_amp (float): Amplitude (std dev) of Gaussian noise added to conductances.
            adaptation_increment (float): Amount added to adaptation conductance (g_adapt) after each spike.
            tau_adaptation (float): Time constant for the decay of the adaptation conductance (ms).
            e_k_reversal (float): Reversal potential for potassium (adaptation) current (mV).
        """
        # Membrane properties
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.tau_ref = tau_ref

        # Synaptic properties
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.e_reversal = e_reversal
        self.i_reversal = i_reversal

        # Noise properties
        self.v_noise_amp = v_noise_amp
        self.i_noise_amp = i_noise_amp

        # Adaptation properties
        self.adaptation_increment = adaptation_increment
        self.tau_adaptation = tau_adaptation
        self.e_k_reversal = e_k_reversal  # K+ reversal potential for conductance-based adaptation

        # State variables initialized
        self.v = v_rest               # Current membrane potential
        self.g_e = 0.0                # Current excitatory conductance
        self.g_i = 0.0                # Current inhibitory conductance
        self.adaptation = 0.0         # Current adaptation conductance (g_adapt, not a current)
        self.t_since_spike = tau_ref  # Time elapsed since the last spike (initialized outside refractory period)
        self.is_inhibitory = is_inhibitory # Neuron type
        self.layer = None             # Placeholder for layer assignment (if used in a layered network)
        self.external_stim_g = 0.0    # Conductance change from direct external stimulation

        # History tracking lists (for analysis/visualization)
        self.v_history = []
        self.g_e_history = []
        self.g_i_history = []
        self.adaptation_history = []
        self.i_syn_history = []       # Total synaptic current history
        self.spike_times = []         # List of times when the neuron spiked

    def reset(self):
        """Resets the neuron's state variables and history to initial conditions."""
        self.v = self.v_rest
        self.g_e = 0.0
        self.g_i = 0.0
        self.adaptation = 0.0
        self.t_since_spike = self.tau_ref # Start outside refractory period
        # Clear history lists
        self.v_history = []
        self.g_e_history = []
        self.g_i_history = []
        self.adaptation_history = []
        self.i_syn_history = []
        self.spike_times = []
        self.external_stim_g = 0.0 # Reset external stimulus

    def update(self, dt):
        """
        Updates the neuron's state for a single time step 'dt'.

        Args:
            dt (float): The simulation time step (ms).

        Returns:
            bool: True if the neuron spiked during this time step, False otherwise.
        """
        # Record current state before update
        self.v_history.append(self.v)
        self.g_e_history.append(self.g_e)
        self.g_i_history.append(self.g_i)
        self.adaptation_history.append(self.adaptation)

        # Calculate synaptic currents based on conductances and driving forces (V_reversal - V_membrane)
        i_syn_internal = self.g_e * (self.e_reversal - self.v) + self.g_i * (self.i_reversal - self.v)
        # Add current from direct external stimulation conductance
        i_syn_external = self.external_stim_g * (self.e_reversal - self.v) # Assuming external stim is excitatory
        i_syn = i_syn_internal + i_syn_external # Total synaptic current
        self.i_syn_history.append(i_syn) # Record total synaptic current

        # Increment time since last spike
        self.t_since_spike += dt

        # --- Refractory Period ---
        if self.t_since_spike < self.tau_ref:
            self.v = self.v_reset # Clamp voltage at reset potential
            # Conductances and adaptation still decay during refractory period
            self.g_e *= np.exp(-dt / self.tau_e)
            self.g_i *= np.exp(-dt / self.tau_i)
            self.adaptation *= np.exp(-dt / self.tau_adaptation)
            # Apply synaptic noise even during refractory period (scaled by sqrt(dt) for proper variance)
            if self.i_noise_amp > 0:
                e_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
                i_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
                # Ensure noise doesn't make conductance negative
                self.g_e = max(0, self.g_e + e_noise)
                self.g_i = max(0, self.g_i + i_noise)
            return False # No spike can occur during refractory period

        # --- Membrane Potential Update (Outside Refractory Period) ---
        # Add membrane potential noise (scaled by sqrt(dt))
        v_noise = np.random.normal(0, self.v_noise_amp * np.sqrt(dt)) if self.v_noise_amp > 0 else 0

        # Calculate adaptation current using conductance-based formulation
        # I_adapt = g_adapt * (E_K - V), where E_K ~ -90mV
        # This is biophysically accurate: adaptation current cannot pull V below E_K
        i_adapt = self.adaptation * (self.e_k_reversal - self.v)

        # Calculate change in membrane potential using the LIF equation with conductance-based adaptation
        # dv/dt = (-(V - V_rest) / tau_m) + I_syn + I_adapt + noise
        # Note: I_adapt is negative (hyperpolarizing) when V > E_K, which is always true in normal operation
        dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_syn + i_adapt) + v_noise

        # OLD SUBTRACTIVE ADAPTATION (commented out):
        # dv/dt = (-(V - V_rest) / tau_m) + I_syn - I_adaptation + noise
        # This could drive V below any reversal potential, which is not biophysically realistic
        # dv = dt * ((-(self.v - self.v_rest) / self.tau_m) + i_syn - self.adaptation) + v_noise
        # Update membrane potential
        self.v += dv

        # --- Conductance Decay and Noise ---
        # Decay conductances exponentially
        self.g_e *= np.exp(-dt / self.tau_e)
        self.g_i *= np.exp(-dt / self.tau_i)
        # Apply synaptic noise (scaled by sqrt(dt))
        if self.i_noise_amp > 0:
            e_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
            i_noise = np.random.normal(0, self.i_noise_amp * np.sqrt(dt))
            # Ensure noise doesn't make conductance negative
            self.g_e = max(0, self.g_e + e_noise)
            self.g_i = max(0, self.g_i + i_noise)

        # --- Adaptation Conductance Decay ---
        # Decay adaptation conductance exponentially (models Ca2+-activated K+ channel dynamics)
        self.adaptation *= np.exp(-dt / self.tau_adaptation)

        # --- Spike Detection ---
        if self.v >= self.v_threshold:
            self.v = self.v_reset # Reset potential
            self.t_since_spike = 0.0 # Reset time since spike (enter refractory period)
            self.adaptation += self.adaptation_increment # Increase adaptation conductance (g_adapt)
            self.spike_times.append(len(self.v_history) * dt) # Record spike time
            return True # Neuron spiked

        return False # Neuron did not spike

    def receive_spike(self, weight):
        """
        Updates the neuron's conductances upon receiving a spike from another neuron.

        Args:
            weight (float): The synaptic weight of the incoming connection.
                           Positive for excitatory, negative for inhibitory.
        """
        if weight > 0:
            # Increase excitatory conductance for positive weight
            self.g_e += weight
        else:
            # Increase inhibitory conductance for negative weight (add absolute value)
            self.g_i += -weight

    def apply_external_stimulus(self, conductance_change):
        """
        Applies a direct external stimulus by setting the external conductance.

        Args:
            conductance_change (float): The value to set the external excitatory conductance to.
                                        Typically non-negative.
        """
        # Set the external conductance, ensuring it's not negative
        self.external_stim_g = max(0, conductance_change)