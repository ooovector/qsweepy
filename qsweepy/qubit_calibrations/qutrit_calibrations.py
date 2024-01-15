from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
import textwrap
import time

#
# def simulation_angles(r, t):
#     h = np.sqrt(1 - r ** 2 + 0j) + 0.00000001
#     a = np.sqrt(np.abs(1 - r ** 2 * np.cos(h * t) ** 2))
#     b = np.abs(r * np.sin(h * t))
#
#     sigma = (a - b) / (a + b + 0.00000001)
#
#     phi = np.arctan(np.tan(h * t) / h)
#     theta = -2 * np.arccos(sigma)
#
#     return phi.real, theta.real

# def simulation_angles(r, t):
#     h = np.sqrt(1 - r ** 2 + 0j) + 0.00000001
#     a = np.sqrt(np.abs(1 - r ** 2 * np.cos(h * t) ** 2))
#     b = np.abs(r * np.sin(h * t))
#     sigma = (a - b) / (a + b + 0.00000001)
#
#     d = 0
#     if (np.real(h * t) // (np.pi / 2)) % 2 == 1:
#         d = np.pi
#
#     phi = np.arctan(np.tan(h * t) / h)
#     theta = -2 * np.arccos(sigma)
#
#     #     return phi.real, theta.real
#     if h * t < np.pi / 2:
#         phi = phi
#     elif h * t < np.pi / 2:
#         phi = phi
#     elif h * t < np.pi:
#         phi = -phi
#     elif h * t < 3 * np.pi / 2:
#         phi = phi
#     elif h * t < 4 * np.pi / 2:
#         phi = -phi
#
#     return phi.real % (2 * np.pi), theta.real

def simulation_angles(r, t):
    h = np.sqrt(1 - r ** 2 + 0j) + 0.00000001
    a = np.sqrt(np.abs(1 - r ** 2 * np.cos(h * t) ** 2))
    b = np.abs(r * np.sin(h * t))
    sigma = (a - b) / (a + b + 0.00000001)

    phi = np.arctan(np.tan(h * t) / h)
    if phi < 0:
        phi += np.pi
    theta = -2 * np.arccos(sigma)

    return phi.real, theta.real


def pt_symmetric_non_herm_ham(device, qubit_id, r_s=None, t_s=None, additional_references={}, additional_metadata={}, gauss=True,
                              sort='best', measurement_type='pt_symmetric_non_herm_ham', readout_delay=None
                              ):
    """
    Run algorithm for simulation of evolution of PT-symmetric non-hermitian Hamiltonian according to the idea from
    https://www.nature.com/articles/s42005-021-00534-2
    :param device:
    :param qubit_id:
    :param r_s:
    :param t_s:
    :param gauss:
    :param sort:
    :param measurement_type:
    :param readout_delay:
    """
    post_selection_flag = False
    transitions_list = ['01', '12']

    from .readout_pulse2 import get_uncalibrated_measurer
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition='01', qutrit_readout=True)

    qubit_excitation_pulses = {}
    for t in transitions_list:
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, transition=t,
                                                                       rotation_angle=np.pi / 2, gauss=gauss,
                                                                       sort=sort)
        qubit_excitation_pulses[t] = qubit_excitation_pulse

    exitation_channels = {}
    for t in transitions_list:
        exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=t).keys()][0]
        exitation_channels[t] = exitation_channel

    ex_channels = {}
    for t in transitions_list:
        exitation_channel = exitation_channels[t]
        ex_channel = device.awg_channels[exitation_channel]
        ex_channels[t] = ex_channel

    awg_and_seq_id = {}  # [awg, seq_id]
    for t in transitions_list:
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
        for t in transitions_list:
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
            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='01',
                                                           phase=np.pi / 2 * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='01',
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['01'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='01',
                                                           phase=(np.pi / 2) * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))


        def rx_12(self):
            """
            Prepare sequence for Rx(12) rotation for transition 12 of qutrit system
            """
            # Add global qutrit phase
            self.prepare_seq.extend(excitation_pulse.get_s_(device, qubit_id,  transition='01',
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['01']))

            self.prepare_seq.extend(excitation_pulse.get_s_(device, qubit_id, transition='12',
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='12',
                                                           phase=(np.pi / 2) * 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='12',
                                                           phase=0,
                                                           fast_control='quasi-binary', gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

            self.prepare_seq.extend(self.qubit_excitation_pulses['12'].get_pulse_sequence(0))

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id, transition='12',
                                                           phase=(np.pi / 2)* 180 / np.pi,
                                                           fast_control=False, gauss=gauss, sort=sort,
                                                           reference_pulse=self.qubit_excitation_pulses['12']))

        def set_r_param(self, r):
            self.r = r

        def set_t_param(self, t):
            phi, theta = simulation_angles(self.r, t)
            # phi, theta = self.r, t
            # in case of quasi-binary fast control with resolution 8
            # phase_phi = (phi + np.pi + self.correction) % (2 * np.pi)
            # phase_theta = (theta + np.pi + self.correction) % (2 * np.pi)

            # phase_phi = (np.abs(phi+ self.correction))  % (2 * np.pi) * np.sign(phi)
            # phase_theta = (np.abs(theta+ self.correction) ) % (2 * np.pi) * np.sign(theta)

            phase_phi = (phi + np.pi) % (2 * np.pi)
            phase_theta = (theta + np.pi) % (2 * np.pi)
            global_phase = (- theta / 2) % (2 * np.pi)


            print('phi', phi, 'theta', theta)
            print('phi', phase_phi, 'theta', phase_theta)

            resolution = 8

            phi_reg = int(phase_phi / (2 * np.pi) * (2 ** resolution))
            theta_reg = int(phase_theta / (2 * np.pi) * (2 ** resolution))
            global_phase_reg = int(global_phase / (2 * np.pi) * (2 ** resolution))

            print('phase_phi', phase_phi, 'phase_theta', phase_theta,
                  'phi_register', phi_reg,
                  'theta_register', theta_reg,
                  'global_phase_reg', global_phase_reg
                   )

            self.control_qubit_sequence['01'].set_phase(int(phase_phi / (2 * np.pi) * (2 ** resolution)))
            self.control_qubit_sequence['12'].set_phase(int(phase_theta / (2 * np.pi) * (2 ** resolution)))

            self.control_qubit_sequence['01'].set_phase_(global_phase_reg)
            self.control_qubit_sequence['12'].set_phase_(global_phase_reg)

    setter = ParameterSetter()

    references = {'ex_pulse01':qubit_excitation_pulses['01'].id,
                  'ex_pulse12': qubit_excitation_pulses['12'].id,
                  'readout_pulse': readout_pulse.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    metadata = {'qubit_id': qubit_id,
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

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

def pt_symmetric_non_herm_ham_ct(device, qubit_id, r_s=None, t_s=None, additional_references={}, additional_metadata={}, gauss=True,
                              sort='best', measurement_type='pt_symmetric_non_herm_ham', readout_delay=None
                              ):
    """
    Run algorithm for simulation of evolution of PT-symmetric non-hermitian Hamiltonian according to the idea from
    https://www.nature.com/articles/s42005-021-00534-2 with command table implementation
    :param device: qubit device object
    :param qubit_id: qubit id
    :param r_s:
    :param t_s:
    :param gauss:
    :param sort:
    :param measurement_type:
    :param readout_delay:
    """
    post_selection_flag = False
    transitions_list = ['01', '12']

    from .readout_pulse2 import get_uncalibrated_measurer
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition='01', qutrit_readout=True)

    exitation_channels = {}
    for t in transitions_list:
        exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=t).keys()][0]
        exitation_channels[t] = exitation_channel

    ex_channels = {}
    for t in transitions_list:
        exitation_channel = exitation_channels[t]
        ex_channel = device.awg_channels[exitation_channel]
        ex_channels[t] = ex_channel

    awg_and_seq_id = {}  # [awg, seq_id]
    for t in transitions_list:
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

    qutrit_awgs, qutrit_seq_ids = {}, {}
    for t in transitions_list:
        ex_channel = ex_channels[t]
        if ex_channel.is_iq():
            seq_id = ex_channel.parent.sequencer_id
        else:
            seq_id = ex_channel.channel // 2
        awg = ex_channel.parent.awg
        qutrit_seq_ids.update({t: seq_id})
        qutrit_awgs.update({t: awg})

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
        for t in transitions_list:
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
            self.device = device
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.t_s = t_s
            self.r_s = r_s
            self.instructions = []
            # all angles for fast phase incrementation
            self.angles = np.linspace(0, 359, 360)
            # Define pulses
            self.rx01 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id, transition='01',
                                                              rotation_angle=np.pi / 2, sort=sort, gauss=gauss)

            self.rx12 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id, transition='12',
                                                              rotation_angle=np.pi / 2, sort=sort, gauss=gauss)

            self.vs01 = excitation_pulse.get_s(device=device, qubit_id=qubit_id, transition='01',
                                               phase=np.pi / 2 * 180 / np.pi,
                                               sort=sort,
                                               gauss=gauss)
            self.vs12 = excitation_pulse.get_s(device=device, qubit_id=qubit_id, transition='12',
                                               phase=np.pi / 2 * 180 / np.pi,
                                               sort=sort,
                                               gauss=gauss)

            self.prepare_seq = []
            self.prepare_virtual_seq = []
            # dictionary of instructions with indexes
            self.instructions_dict = {}

            self.instructions_dict = {'rf_calibration': 0, 'ref_incrementation': 1}

            # Add gates
            self.prepare_seq.append(self.rx01.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx01': 2})

            self.prepare_seq.append(self.rx12.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx12': 3})

            # Add virtual gates
            self.prepare_virtual_seq.append(self.vs01[0])
            self.instructions_dict.update({'s01': 4})

            self.prepare_virtual_seq.append(self.vs12[0])
            self.instructions_dict.update({'s12': 5})

        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '1.2'},
                             'table': []}

            # get information about I and Q phases from calibrations in case of iq excitation channel
            phase0 = ex_seq.phaseI
            phase1 = ex_seq.phaseQ

            # Add two auxiliary table entries, one for phase calibrations and one for reference phase incrementation
            table_entry = {'index': 0}
            table_entry['phase0'] = {'value': phase0, 'increment': False}
            table_entry['phase1'] = {'value': phase1, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': 1}
            table_entry['phase0'] = {'value': 0, 'increment': True}
            table_entry['phase1'] = {'value': 0, 'increment': True}
            command_table['table'].append(table_entry)

            assign_waveform_indexes = {}
            random_command_id = 2
            waveform_id = -1

            # cycle for all prepare seqs
            for prep_seq in self.prepare_seq:
                # cycle for all used hdawg
                for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        # add waveform definition part
                        if single_sequence[0] not in definition_part:
                            definition_part += single_sequence[0]

                        assign_fragments = single_sequence[3].split(';')[:-1]

                        for id_, entry_table_index_constant in enumerate(single_sequence[2]):
                            table_entry = {'index': random_command_id}  # define command table id index
                            random_command_id += 1
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                definition_part += '\nconst ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                definition_part += assign_fragments[id_] + ';'
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {
                                    'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = single_sequence[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)

            # cycle for all prepare virtual seqs
            for prep_seq in self.prepare_virtual_seq:
                # cycle for all used hdawg
                for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        random_pulse = single_sequence[4]
                        table_entry = {'index': random_command_id}  # define command table id index
                        random_command_id += 1
                        if 'phase0' in random_pulse:
                            table_entry['phase0'] = random_pulse['phase0']
                        if 'phase1' in random_pulse:
                            table_entry['phase1'] = random_pulse['phase1']
                        command_table['table'].append(table_entry)

            # create instructions for fast phase incrementation
            for phase in self.angles:
                if (ex_seq.params['sequencer_id'] == qutrit_seq_ids['01']) and (
                        ex_seq.awg.device_id == qutrit_awgs['01'].device_id):
                    table_entry = {'index': random_command_id}
                    table_entry['phase0'] = {'value': phase, 'increment': True}
                    table_entry['phase1'] = {'value': phase, 'increment': True}
                    command_table['table'].append(table_entry)
                    random_command_id += 1
                else:
                    table_entry = {'index': random_command_id}
                    table_entry['phase0'] = {'value': 0, 'increment': True}
                    table_entry['phase1'] = {'value': 0, 'increment': True}
                    command_table['table'].append(table_entry)
                    random_command_id += 1

            for phase in self.angles:
                if (ex_seq.params['sequencer_id'] == qutrit_seq_ids['12']) and (
                        ex_seq.awg.device_id == qutrit_awgs['12'].device_id):
                    table_entry = {'index': random_command_id}
                    table_entry['phase0'] = {'value': phase, 'increment': True}
                    table_entry['phase1'] = {'value': phase, 'increment': True}
                    command_table['table'].append(table_entry)
                    random_command_id += 1
                else:
                    table_entry = {'index': random_command_id}
                    table_entry['phase0'] = {'value': 0, 'increment': True}
                    table_entry['phase1'] = {'value': 0, 'increment': True}
                    command_table['table'].append(table_entry)
                    random_command_id += 1

            play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry({rf_calibration});
    executeTableEntry({ref_incrementation});

    // Rx01(phi)
    executeTableEntry({s01});
    executeTableEntry({rx01});
    executeTableEntry(6 + variable_register2);
    executeTableEntry({rx01});
    executeTableEntry({s01});
    
    // Rx12(theta)
    executeTableEntry(6 + variable_register14);
    executeTableEntry({angles_num} + 6 + variable_register14);
    executeTableEntry({s12});
    executeTableEntry({rx12});
    executeTableEntry({angles_num} + 6 + variable_register2);
    executeTableEntry({rx12});
    executeTableEntry({s12});
    
    // Rx01(phi)
    executeTableEntry({s01});
    executeTableEntry({rx01});
    executeTableEntry(6 + variable_register2);
    executeTableEntry({rx01});
    executeTableEntry({s01});

    executeTableEntry({ref_incrementation});
    executeTableEntry({rf_calibration});
    resetOscPhase();''').format(**self.instructions_dict, angles_num=len(self.angles))

            self.instructions.append(command_table)
            print('Command table for sequencer id {}'.format(ex_seq.params['sequencer_id']), command_table)
            return definition_part, play_part

        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            control_awg_ids = []
            self.instructions = []

            for device_id in self.prepare_seq[0][0].keys():
                pulses.update({device_id: {}})

            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.awg.device_id][ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])
                control_awg_ids.append(ex_seq.awg.device_id)
            return [[pulses, control_seq_ids, control_awg_ids]]

        def set_r_param(self, r):
            self.r = r

        def set_t_param(self, t):
            phi, theta = simulation_angles(self.r, t)
            phase_phi = (phi + np.pi) % (2 * np.pi)
            phase_theta = (theta + np.pi) % (2 * np.pi)
            global_phase = (- theta / 2) % (2 * np.pi)

            print('phi', phase_phi * 180 / np.pi, 'theta', phase_theta * 180 / np.pi,
                  'global phase', global_phase * 180 / np.pi)

            index_phi = self.find_closest_phase(self.angles, phase_phi * 180 / np.pi)
            index_theta = self.find_closest_phase(self.angles, phase_theta * 180 / np.pi)
            index_global_phase = self.find_closest_phase(self.angles, global_phase * 180 / np.pi)

            if (t == self.t_s[0]) and (self.r == self.r_s[0]):
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                if (seq.params['sequencer_id'] == qutrit_seq_ids['01']) and (
                        seq.awg.device_id == qutrit_awgs['01'].device_id):
                    seq.set_phase_index(index_phi)
                    seq.set_phase_index_(index_global_phase)

                elif (seq.params['sequencer_id'] == qutrit_seq_ids['12']) and (
                        seq.awg.device_id == qutrit_awgs['12'].device_id):
                    seq.set_phase_index(index_theta)
                    seq.set_phase_index_(index_global_phase)

        def find_closest_phase(self, array, val):
            idx = np.abs(array - val).argmin()
            return idx

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    metadata = {'qubit_id': qubit_id,
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

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