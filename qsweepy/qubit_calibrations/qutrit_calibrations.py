from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse


def simulation_angles(r, t):
    h = np.sqrt(1 - r ** 2 + 0j) + 0.00000001
    a = np.sqrt(np.abs(1 - r ** 2 * np.cos(h * t) ** 2))
    b = np.abs(r * np.sin(h * t))
    sigma = (a - b) / (a + b + 0.00000001)

    phi = np.arctan(np.tan(h * t) / h)
    theta = -2 * np.arccos(sigma)

    return phi.real, theta.real
#
# def simulation_angles(r, t):
#     h = np.sqrt(1 - r ** 2 + 0j)
#     a = np.sqrt(np.abs(1 - r ** 2 * np.cos(h * t) ** 2))
#     b = np.abs(r * np.sin(h * t))
#     sigma = (a - b) / (a + b )
#
#     phi = np.arctan(np.tan(h * t) / h)
#     theta = -2 * np.arccos(sigma)
#
#     return phi.real, theta.real


# def pt_symmetric_non_herm_ham(device, qubit_transitions_id, channel_amplitudes=None, phis=None,
#                               thetas=None, additional_references={}, additional_metadata={}, gauss=True, sort='best',
#                               measurement_type='pt_symmetric_non_herm_ham', readout_delay=None
#                               ):
def pt_symmetric_non_herm_ham(device, qubit_transitions_id, channel_amplitudes=None, r_s=None,t_s=None,
                              r=None, additional_references={}, additional_metadata={}, gauss=True,
                              sort='best',
                              measurement_type='pt_symmetric_non_herm_ham', readout_delay=None
                              ):
    """
    Run algorithm for simulation of evolution of PT-symmetric non-hermitian Hamiltonian according to the idea from
    https://www.nature.com/articles/s42005-021-00534-2
    :param device:
    :param qubit_transitions_id:
    :param channel_amplitudes:
    :param channel_amplitudes:
    :param gauss:
    :param sort:
    """
    post_selection_flag = False

    qubit_id = qubit_transitions_id['01']  # for transition 01
    auxiliary_qubit_id = qubit_transitions_id['12']  # for transition 12
    from .readout_pulse2 import get_uncalibrated_measurer
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition='01', qutrit_readout=True)

    qubit_excitation_pulses = {}
    for t in list(qubit_transitions_id.keys()):
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_transitions_id[t],
                                                                       rotation_angle=np.pi / 2, gauss=gauss,
                                                                       sort=sort)
        qubit_excitation_pulses[t] = qubit_excitation_pulse

    exitation_channels = {}
    for t in list(qubit_transitions_id.keys()):
        exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_transitions_id[t]).keys()][0]
        exitation_channels[t] = exitation_channel

    ex_channels = {}
    for t in list(qubit_transitions_id.keys()):
        exitation_channel = exitation_channels[t]
        ex_channel = device.awg_channels[exitation_channel]
        ex_channels[t] = ex_channel

    awg_and_seq_id = {}  # [awg, seq_id]
    for t in list(qubit_transitions_id.keys()):
        ex_channel = ex_channels[t]
        if ex_channel.is_iq():
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.parent.sequencer_id
        else:
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.channel // 2
        awg_and_seq_id[t] = [control_qubit_awg, control_qubit_seq_id]

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    control_qubit_sequence = {}
    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[],
                                               post_selection_flag=post_selection_flag)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
                                               post_selection_flag=post_selection_flag)
            control_sequence = ex_seq
        for t in list(qubit_transitions_id.keys()):
            control_qubit_awg, control_qubit_seq_id = awg_and_seq_id[t]
            if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
                control_qubit_sequence[t] = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)

    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.qubit_excitation_pulses = qubit_excitation_pulses

            self.prepare_seq = []

            self.r = r

            self.rx_01()
            self.rx_12()
            self.rx_01()

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()


        def rx_01(self):
            """
            Prepare sequence for Rx(01) rotation for transition 01 of qutrit system
            """
            # define phase (in grad)
            #
            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=(2 * np.pi - np.pi / 3) * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=(2 * np.pi - 2 * np.pi / 3) * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=np.pi / 2 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=np.pi / 2 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))


        def rx_12(self):
            """
            Prepare sequence for Rx(12) rotation for transition 12 of qutrit system
            """

            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=np.pi / 3 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=2 * np.pi / 3 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=np.pi / 2 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=np.pi / 2 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

        def set_r_param(self, r):
            self.r = r

        def set_t_param(self, t):
            phi, theta = simulation_angles(self.r, t)
            # in case of quasi-binary fast control with resolution 8
            phase_phi = (phi + np.pi) % (2 * np.pi)
            phase_theta = (theta + np.pi) % (2 * np.pi)

            # phase_phi = (phi + np.pi)
            # phase_theta = (theta + np.pi)
            print('phi', phi, 'theta', theta)
            # print('phase_phi', phase_phi, 'phase_theta', phase_theta,
            #       'phi_register', int(phase_phi / (2 * np.pi) * (2 ** 16)), 'theta_register',
            #       int(phase_theta / (2 * np.pi) * (2 ** 16)))
            #
            # if phase_phi >= 0:
            #     self.control_qubit_sequence['01'].set_phase(int(phase_phi / (2 * np.pi) * (2 ** 16)))
            # else:
            #     self.control_qubit_sequence['01'].set_phase(int((phase_phi + 2 * np.pi) / (2 * np.pi) * (2 ** 16)))
            #
            # if phase_theta >= 0:
            #     self.control_qubit_sequence['12'].set_phase(int(phase_theta / (2 * np.pi) * (2 ** 16)))
            # else:
            #     self.control_qubit_sequence['12'].set_phase(int((phase_theta + 2 * np.pi) / (2 * np.pi) * (2 ** 16)))
            print('phase_phi', phase_phi, 'phase_theta', phase_theta,
                  'phi_register', int(phase_phi / (2 * np.pi) * (2 ** 8)), 'theta_register', int(phase_theta / (2 * np.pi) * (2 ** 8)))

            if phase_phi >= 0:
                self.control_qubit_sequence['01'].set_phase(int(phase_phi / (2 * np.pi) * (2 ** 8)))
            else:
                self.control_qubit_sequence['01'].set_phase(int((phase_phi + 2 * np.pi) / (2 * np.pi) * (2 ** 8)))


            if phase_theta >= 0:
                self.control_qubit_sequence['12'].set_phase(int(phase_theta / (2 * np.pi) * (2 ** 8)))
            else:
                self.control_qubit_sequence['12'].set_phase(int((phase_theta + 2 * np.pi) / (2 * np.pi) * (2 ** 8)))



        # def set_phi_angle(self, phi):
        #     # in case of quasi-binary fast control
        #     phase = phi % (2 * np.pi)
        #     # print (phase: ', phase, ', phase register: ', int(phase/360*(2**6)))
        #
        #     if phase >= 0:
        #         self.control_qubit_sequence['01'].set_phase(int(phase / 360 * (2 ** 6)))
        #     else:
        #         self.control_qubit_sequence['01'].set_phase(int((360 + phase) / 360 * (2 ** 6)))
        #
        # def set_theta_angle(self, theta):
        #     # in case of quasi-binary fast control
        #     phase = theta % (2 * np.pi)
        #     # print (phase: ', phase, ', phase register: ', int(phase/360*(2**6)))
        #
        #     if phase >= 0:
        #         self.control_qubit_sequence['12'].set_phase(int(phase / 360 * (2 ** 6)))
        #     else:
        #         self.control_qubit_sequence['12'].set_phase(int((360 + phase) / 360 * (2 ** 6)))

    setter = ParameterSetter()

    references = {'ex_pulse01':qubit_excitation_pulses['01'].id,
                  'ex_pulse12': qubit_excitation_pulses['12'].id,
                  'readout_pulse': readout_pulse.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    metadata = {'qubit_id': qubit_id,
                'auxiliary_qubit_id': auxiliary_qubit_id,
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    # measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
    #                                                         (phis, setter.set_phi_angle, 'phi', 'rad'),
    #                                                         (thetas, setter.set_theta_angle, 'theta', 'rad'),
    #                                                         measurement_type=measurement_type,
    #                                                         metadata=metadata,
    #                                                         references=references)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (r_s, setter.set_r_param, 'r', ''),
                                                            (t_s, setter.set_t_param, 't', ''),
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references)


    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def qutrit_ramsey(device, qubit_transitions_id, target_freq_offset=None, lengths=None, additional_references={},
                  additional_metadata={}, gauss=True, sort='best',
                  measurement_type='qutrit_Ramsey', readout_delay=None, ramsey_transition='01'
                  ):
    """
    Run qutrit Ramsey oscillations
    """
    post_selection_flag = False

    qubit_id = qubit_transitions_id['01']  # for transition 01
    auxiliary_qubit_id = qubit_transitions_id['12']  # for transition 12
    from .readout_pulse2 import get_uncalibrated_measurer
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition='01', qutrit_readout=True)

    qubit_excitation_pulses = {}
    for t in list(qubit_transitions_id.keys()):
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_transitions_id[t],
                                                                       rotation_angle=np.pi / 2, gauss=gauss,
                                                                       sort=sort)
        qubit_excitation_pulses[t] = qubit_excitation_pulse

    exitation_channels = {}
    for t in list(qubit_transitions_id.keys()):
        exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_transitions_id[t]).keys()][0]
        exitation_channels[t] = exitation_channel

    ex_channels = {}
    for t in list(qubit_transitions_id.keys()):
        exitation_channel = exitation_channels[t]
        ex_channel = device.awg_channels[exitation_channel]
        ex_channels[t] = ex_channel

    awg_and_seq_id = {}  # [awg, seq_id]
    for t in list(qubit_transitions_id.keys()):
        ex_channel = ex_channels[t]
        if ex_channel.is_iq():
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.parent.sequencer_id
        else:
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.channel // 2
        awg_and_seq_id[t] = [control_qubit_awg, control_qubit_seq_id]

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    control_qubit_sequence = {}
    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[],
                                               post_selection_flag=post_selection_flag)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
                                               post_selection_flag=post_selection_flag)
            control_sequence = ex_seq
        for t in list(qubit_transitions_id.keys()):
            control_qubit_awg, control_qubit_seq_id = awg_and_seq_id[t]
            if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
                control_qubit_sequence[t] = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)

        if ex_seq.params['is_iq']:
            ex_seq.start()
        else:
            ex_seq.start(holder=1)

        # ex_seq.start()
        ex_sequencers.append(ex_seq)

    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.lengths = lengths
            self.readout_sequencer = readout_sequencer
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.qubit_excitation_pulses = qubit_excitation_pulses

            self.prepare_seq = []

            if ramsey_transition == '01':
                self.prepare_seq01()
            else:
                self.prepare_seq12()

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()


        def prepare_seq01(self):
            """
            Prepare sequence for Ramsey oscillations between 0 and 1 states
            """
            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
            self.prepare_seq.extend([device.pg.pmulti(device, 0)])

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=(64/self.control_sequence.clock)*target_freq_offset*360 % 360,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))


        def prepare_seq12(self):
            """
            Prepare sequence for Ramsey oscillations between 1 and 2 states
            """
            # Rx01(pi)
            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))
            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            # Rx12(pi / 2)
            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            # delay
            self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
            self.prepare_seq.extend([device.pg.pmulti(device, 0)])

            # Rz12(phase)
            self.prepare_seq.extend(excitation_pulse.get_s(device, auxiliary_qubit_id,
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            # Rx12(pi / 2)
            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            # Rx01(pi)
            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))
            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))


        def set_delay(self, length):
            phase = int(np.round((length + 140e-9) * self.control_sequence.clock) + 64) / self.control_sequence.clock * target_freq_offset * 360 % 360
            # phase = int(np.round((length) * self.control_sequence.clock) + 64) / self.control_sequence.clock * target_freq_offset * 360 % 360
            # print ('length: ', length, ', phase: ', phase, ', phase register: ', int(phase/360*(2**6)))
            for ex_seq in self.ex_sequencers:
                ex_seq.set_length(length)
                    #ex_seq.set_phase(int(phase / 360 * (2 ** 8)))
            if phase >= 0:
                self.control_sequence.set_phase(int(phase/360*(2**8)))
            else:
                self.control_sequence.set_phase(int((360+phase) / 360 * (2 ** 8)))




    setter = ParameterSetter()

    references = {'ex_pulse01':qubit_excitation_pulses['01'].id,
                  'ex_pulse12': qubit_excitation_pulses['12'].id,
                  'readout_pulse': readout_pulse.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    metadata = {'qubit_id': qubit_id,
                'auxiliary_qubit_id': auxiliary_qubit_id,
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (lengths, setter.set_delay, 'Delay', 's'),
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references)


    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement