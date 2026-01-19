"""
Auto-generated config for grid search experiment
Parameters: n_i=100, pConn_ei=0.03, pConn_ie=0.6, weight_ie=9.44
E:I Ratio: 400:100
Inhibitory Drive: 16.99
Overlap: 93.6%
Seed: 42
"""

from brian2 import ms, mV, second, Hz
import numpy as np


class Config:
    def __init__(self):
        # Mode Settings
        self.test_mode = False

        # Path Settings - ABSOLUTE PATHS for grid search
        self.mnist_data_path = '/Users/ethancrouse/Desktop/Spiking-Neural-Network-Experiments/phase_3_Learning_over_time/diehl_2015_research_additions/mnist_data/'
        self.data_path = '/Users/ethancrouse/Desktop/Spiking-Neural-Network-Experiments/phase_3_Learning_over_time/diehl_2015_research_additions/grid_search_20260114_022210/exp_003_xi5_ei3_ie60_w9.4/'

        # Simulation Settings
        self.dt = 0.5 * ms
        self.random_seed = 42

        # Network Architecture - GRID SEARCH PARAMETERS
        self.n_input = 784
        self.n_e = 400
        self.n_i = 100  # E:I ratio = 400:100

        # Timing Parameters
        self.single_example_time = 0.35 * second
        self.resting_time = 0.15 * second

        # Training Parameters
        self.mnist_classes = [0, 1, 2, 3, 4]
        self.num_train_examples = 10345
        self.num_test_examples = 4900
        self.test_examples_per_class = 980
        self.assignment_examples_per_class = 500

        self.num_examples = None
        self.use_testing_set = None
        self.do_plot_performance = None
        self.record_spikes = True
        self.ee_STDP_on = None
        self.enable_live_plots = False

        self.update_interval = None
        self.weight_update_interval = 20
        self.save_connections_interval = 10000

        # Neuron Parameters
        self.v_rest_e = -65. * mV
        self.v_reset_e = -65. * mV
        self.v_thresh_e_const = -52. * mV
        self.refrac_e = 5. * ms
        self.offset = 20.0 * mV

        self.v_rest_i = -60. * mV
        self.v_reset_i = -45. * mV
        self.v_thresh_i = -40. * mV
        self.refrac_i = 2. * ms

        self.noise_sigma_e = 0.3 * mV
        self.noise_sigma_i = 0.3 * mV

        self.weight_ee_input = 78.
        self.wmax_ee = 1.0

        self.delay_ee_input = (0*ms, 10*ms)
        self.delay_ei_input = (0*ms, 5*ms)

        self.input_intensity = 2.0
        self.start_input_intensity = self.input_intensity

        # STDP Parameters
        self.tc_pre_ee = 20 * ms
        self.tc_post_1_ee = 20 * ms
        self.tc_post_2_ee = 40 * ms
        self.nu_ee_pre = 0.0001
        self.nu_ee_post = 0.01
        self.exp_ee_pre = 0.2
        self.exp_ee_post = 0.2
        self.STDP_offset = 0.4

        self.tc_theta = 1e7 * ms
        self.theta_plus_e = 0.05 * mV

        # Population Names
        self.input_population_names = ['X']
        self.population_names = ['A']
        self.input_connection_names = ['XA']
        self.save_conns = ['XeAe']
        self.input_conn_names = ['ee_input', 'ei_input']
        self.recurrent_conn_names = ['ei', 'ie']
        self.ending = ''

        self._compute_derived_params()

    def _compute_derived_params(self):
        if self.test_mode:
            self.weight_path = self.data_path + 'weights/'
            self.num_examples = self.num_test_examples
            self.use_testing_set = True
            self.do_plot_performance = False
            self.ee_STDP_on = False
            self.update_interval = self.num_examples
        else:
            self.weight_path = self.data_path + 'random/'
            self.num_examples = self.num_train_examples * 3
            self.use_testing_set = False
            self.do_plot_performance = False
            self.ee_STDP_on = True
            self.record_spikes = True
            if self.num_examples <= 10000:
                self.update_interval = self.num_examples
            else:
                self.update_interval = 10000
            self.save_connections_interval = 10000

        self.runtime = self.num_examples * (self.single_example_time + self.resting_time)

    def get_neuron_eqs_e(self):
        eqs = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms) + noise_sigma_e*xi_e/sqrt(100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
        if self.test_mode:
            eqs += '\n  theta      :volt'
        else:
            eqs += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
        eqs += '\n  dtimer/dt = 100.0*msecond/second  : second'
        return eqs

    def get_neuron_eqs_i(self):
        return '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms) + noise_sigma_i*xi_i/sqrt(10*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-85.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''

    def get_stdp_eqs(self):
        return '''
                w : 1
                post2before : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''

    def get_stdp_pre_eq(self):
        return 'ge_post += w; pre = 1.; w = clip(w - nu_ee_pre * post1, 0, wmax_ee)'

    def get_stdp_post_eq(self):
        return 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'

    def get_v_thresh_e_str(self):
        return '(v>(theta - offset + v_thresh_e_const)) and (timer>refrac_e)'

    def get_scr_e(self):
        if self.test_mode:
            return 'v = v_reset_e; timer = 0*ms'
        else:
            return 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    def set_test_mode(self, test_mode):
        self.test_mode = test_mode
        self._compute_derived_params()

    def get_class_filter_str(self):
        if self.mnist_classes is None:
            return "all classes [0-9]"
        return f"classes {self.mnist_classes}"

    def should_use_example(self, label):
        if self.mnist_classes is None:
            return True
        return label in self.mnist_classes

    def __repr__(self):
        return f"Config(n_i=100, E:I=400:100, drive=16.99)"


default_config = Config()
