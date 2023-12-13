from qsweepy.libraries import pulses2 as pulses
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
from qsweepy.qubit_calibrations.readout_pulse2 import *
import textwrap
import time




def error_detection(device, qubit_ids, gate, phis=None, *extra_sweep_args, additional_metadata={}, error='RX', gate2=None,
                    multi_qubit_readout_flag=False):
    """
    Quantum circuit for Rx error detection
    data    --H--Rz(phi)--H--CZ(01)-----
    ancilla --------------H--CZ(01)--H--Meas
    :param qubit_ids: [data qubit, ancilla qubit]
    """
    assert (error == 'RX' or error == 'RZ')
    data_qubit_id, ancilla_qubit_id = qubit_ids

    if multi_qubit_readout_flag:
        readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)
    else:
        # in this case only ancilla is readout
        readout_pulse, measurer = get_uncalibrated_measurer(device, ancilla_qubit_id)

    excitation_channel_data =  list(device.get_qubit_excitation_channel_list(data_qubit_id).keys())[0]
    excitation_channel_ancilla = list(device.get_qubit_excitation_channel_list(ancilla_qubit_id).keys())[0]


    ex_channel = device.awg_channels[excitation_channel_data]

    ex_channel_gate = device.awg_channels[gate.metadata['carrier_name']]
    control_gate_channel_id = ex_channel_gate.channel % 2
    if control_gate_channel_id == 0:
        ampl = float(gate.metadata['amplitude'])
    elif control_gate_channel_id == 1:
        ampl = 1j * float(gate.metadata['amplitude'])
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: ampl})

    if gate2:
        ex_channel_gate2 = device.awg_channels[gate2.metadata['carrier_name']]
        control_gate2_channel_id = ex_channel_gate2.channel % 2
        if control_gate2_channel_id == 0:
            ampl2 = complex(gate2.metadata['amplitude'])
        elif control_gate2_channel_id == 1:
            ampl2 = 1j * float(gate2.metadata['amplitude'])
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{gate2.metadata['carrier_name']: ampl2})

    data_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        data_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        data_qubit_seq_id = ex_channel.channel // 2

    ex_channel = device.awg_channels[excitation_channel_ancilla]
    ancilla_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        ancilla_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        ancilla_qubit_seq_id = ex_channel.channel // 2


    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []


    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        if [awg, seq_id] == [data_qubit_awg, data_qubit_seq_id]:
            data_qubit_sequence = ex_seq
        if [awg, seq_id] == [ancilla_qubit_awg, ancilla_qubit_seq_id]:
            ancilla_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # ex_seq.start(holder=1)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()


    # ampl = complex(gate.metadata['amplitude'])
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.error_type = error

            # self.phis = phis / np.pi * 180
            self.phis = phis % (2 * np.pi) * 180 / np.pi

            # self.gate_freq = float(gate.metadata['frequency'])

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer


            self.rx1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=data_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)

            self.rx2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=ancilla_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)

            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                         channel_amplitudes=channel_amplitudes_,
                                                                         tail_length=float(gate.metadata['tail_length']),
                                                                         length=float(gate.metadata['length']),
                                                                         phase=0.0,
                                                                         fast_control=False)
            if gate2:
                gate2_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                         channel_amplitudes=channel_amplitudes2_,
                                                                         tail_length=float(
                                                                             gate2.metadata['tail_length']),
                                                                         length=float(gate2.metadata['length']),
                                                                         phase=0.0,
                                                                         fast_control=False)

            self.vs1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id, phase=np.pi / 2 * 180 / np.pi, sort='newest',
                                              gauss=True)
            self.vs2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id, phase=np.pi / 2 * 180 / np.pi, sort='newest',
                                               gauss=True)

            self.vrz_phi1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                  phase=float(gate.metadata['phi1']) * 180 / np.pi, sort='newest',
                                                  gauss=True)
            self.vrz_phi2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id,
                                                  phase=float(gate.metadata['phi2']) * 180 / np.pi, sort='newest',
                                                  gauss=True)

            self.vrz1_pi = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                   phase=np.pi * 180 / np.pi, sort='newest',
                                                   gauss=True)


            self.control_sequence = control_sequence
            self.data_qubit_sequence = data_qubit_sequence
            self.data_qubit_awg = data_qubit_awg

            self.ancilla_qubit_sequence = ancilla_qubit_sequence
            self.ancilla_qubit_awg = ancilla_qubit_awg

            self.instructions_dict = {'calib_rf_inst': 0, 'ref_increment_inst': 1}

            # Prepare sequence with waveforms
            self.prepare_seq = []
            self.prepare_seq.append(self.rx1.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx1_inst': 2})

            self.prepare_seq.append(self.rx2.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx2_inst': 3})

            self.prepare_seq.append(
                device.pg.parallel(self.rx1.get_pulse_sequence(0)[0], self.rx2.get_pulse_sequence(0)[0]))
            self.instructions_dict.update({'rx1_rx2_inst': 4})

            self.prepare_seq.extend(gate_sequence)
            self.instructions_dict.update({'cz_gate_inst': 5})

            # Prepare virtual sequence (without waveforms)
            self.prepare_virtual_seq = []
            self.prepare_virtual_seq.append(self.vs1[0])
            self.instructions_dict.update({'s1_inst': 6})

            self.prepare_virtual_seq.append(self.vs2[0])
            self.instructions_dict.update({'s2_inst': 7})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vs1[0], self.vs2[0]))
            self.instructions_dict.update({'s1_s2_inst': 8})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vrz_phi1[0], self.vrz_phi2[0]))
            self.instructions_dict.update({'phi1_phi2_inst': 9})

            self.prepare_virtual_seq.append(self.vrz1_pi[0])
            self.instructions_dict.update({'rz1_phi_inst': 10})



        def phi_setter(self, phi):
            phi = phi % (2 * np.pi) * 180 / np.pi
            index = np.where(self.phis == phi)[0][0]
            if phi == self.phis[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index(index)

        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '0.2'},
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
            for phase in self.phis:
                if (ex_seq.params['sequencer_id'] == self.data_qubit_sequence.params['sequencer_id']) and (
                        ex_seq.awg.device_id == self.data_qubit_awg.device_id):
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

#             play_part = textwrap.dedent('''
# //
#     // Table entry play part
#     executeTableEntry({calib_rf_inst});
#     executeTableEntry({ref_increment_inst});
#             ''').format(**self.instructions_dict)
#
#             if self.error_type == 'RX':
#                 play_part += textwrap.dedent('''
# //
#     // Hadamard data qubit
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({rx1_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_inst});
#
#     // Rz(phi) gate data qubit for error
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry(11 + variable_register2);
#             ''').format(**self.instructions_dict)
#             elif self.error_type == 'RZ':
#                 play_part += textwrap.dedent('''
# //
#     // Rz(phi) gate data qubit for error
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry(11 + variable_register2);
#             ''').format(**self.instructions_dict)
#             else:
#                 raise ValueError("Can detect only RX or RZ errors!")
#
#             play_part += textwrap.dedent('''
# //
#     // Hadamard gate for both qubits
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_s2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({rx1_rx2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_s2_inst});
#
#     // CZ gate
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({cz_gate_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({phi1_phi2_inst});
#             ''').format(**self.instructions_dict)
#
#             if self.error_type == 'RX':
#                 play_part += textwrap.dedent('''
# //
#     // Hadamard ancilla qubit
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({rx2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s2_inst});
#             ''').format(**self.instructions_dict)
#             elif self.error_type == 'RZ':
#                 play_part += textwrap.dedent('''
# //
#     // Hadamard gate for both qubits
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_s2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({rx1_rx2_inst});
#     executeTableEntry({ref_increment_inst});
#     executeTableEntry({s1_s2_inst});
#             ''').format(**self.instructions_dict)
#             else:
#                 raise ValueError("Can detect only RX or RZ errors!")

            play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry({calib_rf_inst});
    executeTableEntry({ref_increment_inst});

    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});

    // Rz(phi) gate data qubit for error
    executeTableEntry({ref_increment_inst});
    executeTableEntry({num_inst} + variable_register2);

    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});

    // CZ gate
    executeTableEntry({ref_increment_inst});
    executeTableEntry({cz_gate_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({phi1_phi2_inst});
    
    // Hadamard ancilla qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});

    // Return data qubit to initial state
    //executeTableEntry({rz1_phi_inst});

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry(12 + variable_register2);

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});

    executeTableEntry({calib_rf_inst});
    resetOscPhase();''').format(**self.instructions_dict, num_inst=len(self.instructions_dict.keys()))

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

    measurement_type = 'error_detection'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + ancilla_qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=ancilla_qubit_id),
                  'readout_pulse': readout_pulse.id}

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    fitter_arguments = ('iq' + ancilla_qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))
    if multi_qubit_readout_flag:
        fitter_arguments = ('resultnumbers', exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(ancilla_qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'error': error}
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (phis, setter.phi_setter, 'Z phase', 'rad'),
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references
                                                            )

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def error_detection_sa(device, qubit_ids, gate, phis=None, *extra_sweep_args, additional_metadata={}, error='RX',
                    gate2=None,
                    multi_qubit_readout_flag=False):
    """
    Quantum circuit for Rx error detection
    data    --H--Rz(phi)--H--H--CZ(01)--H--CZ(01)-----Meas
    ancilla --------------H--H--CZ(01)--H--CZ(01)--H--
    :param qubit_ids: [data qubit, ancilla qubit]
    """
    assert (error == 'RX' or error == 'RZ')
    data_qubit_id, ancilla_qubit_id = qubit_ids

    if multi_qubit_readout_flag:
        readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)
    else:
        # in this case only data is readout
        readout_pulse, measurer = get_uncalibrated_measurer(device, data_qubit_id)

    excitation_channel_data = list(device.get_qubit_excitation_channel_list(data_qubit_id).keys())[0]
    excitation_channel_ancilla = list(device.get_qubit_excitation_channel_list(ancilla_qubit_id).keys())[0]

    ex_channel = device.awg_channels[excitation_channel_data]

    ex_channel_gate = device.awg_channels[gate.metadata['carrier_name']]
    control_gate_channel_id = ex_channel_gate.channel % 2
    if control_gate_channel_id == 0:
        ampl = float(gate.metadata['amplitude'])
    elif control_gate_channel_id == 1:
        ampl = 1j * float(gate.metadata['amplitude'])
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: ampl})

    # if gate2:
    #     ex_channel_gate2 = device.awg_channels[gate2.metadata['carrier_name']]
    #     control_gate2_channel_id = ex_channel_gate2.channel % 2
    #     if control_gate2_channel_id == 0:
    #         ampl2 = complex(gate2.metadata['amplitude'])
    #     elif control_gate2_channel_id == 1:
    #         ampl2 = 1j * float(gate2.metadata['amplitude'])
    #     channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{gate2.metadata['carrier_name']: ampl2})

    data_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        data_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        data_qubit_seq_id = ex_channel.channel // 2

    ex_channel = device.awg_channels[excitation_channel_ancilla]
    ancilla_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        ancilla_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        ancilla_qubit_seq_id = ex_channel.channel // 2

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        if [awg, seq_id] == [data_qubit_awg, data_qubit_seq_id]:
            data_qubit_sequence = ex_seq
        if [awg, seq_id] == [ancilla_qubit_awg, ancilla_qubit_seq_id]:
            ancilla_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # ex_seq.start(holder=1)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    # ampl = complex(gate.metadata['amplitude'])
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.error_type = error

            # self.phis = phis / np.pi * 180
            self.phis = phis % (2 * np.pi) * 180 / np.pi

            # self.gate_freq = float(gate.metadata['frequency'])

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            self.rx1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=data_qubit_id,
                                                             rotation_angle=np.pi / 2, sort='newest', gauss=True)

            self.rx2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=ancilla_qubit_id,
                                                             rotation_angle=np.pi / 2, sort='newest', gauss=True)

            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                         channel_amplitudes=channel_amplitudes_,
                                                                         tail_length=float(
                                                                             gate.metadata['tail_length']),
                                                                         length=float(gate.metadata['length']),
                                                                         phase=0.0,
                                                                         fast_control=False)
            # if gate2:
            #     gate2_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
            #                                                                   channel_amplitudes=channel_amplitudes2_,
            #                                                                   tail_length=float(
            #                                                                       gate2.metadata['tail_length']),
            #                                                                   length=float(gate2.metadata['length']),
            #                                                                   phase=0.0,
            #                                                                   fast_control=False)

            self.vs1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id, phase=np.pi / 2 * 180 / np.pi,
                                              sort='newest',
                                              gauss=True)
            self.vs2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id, phase=np.pi / 2 * 180 / np.pi,
                                              sort='newest',
                                              gauss=True)

            self.vrz_phi1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                   phase=float(gate.metadata['phi1']) * 180 / np.pi, sort='newest',
                                                   gauss=True)
            self.vrz_phi2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id,
                                                   phase=float(gate.metadata['phi2']) * 180 / np.pi, sort='newest',
                                                   gauss=True)

            self.vrz1_pi = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                  phase=np.pi * 180 / np.pi, sort='newest',
                                                  gauss=True)

            self.control_sequence = control_sequence
            self.data_qubit_sequence = data_qubit_sequence
            self.data_qubit_awg = data_qubit_awg

            self.ancilla_qubit_sequence = ancilla_qubit_sequence
            self.ancilla_qubit_awg = ancilla_qubit_awg

            self.instructions_dict = {'calib_rf_inst': 0, 'ref_increment_inst': 1}

            # Prepare sequence with waveforms
            self.prepare_seq = []
            self.prepare_seq.append(self.rx1.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx1_inst': 2})

            self.prepare_seq.append(self.rx2.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx2_inst': 3})

            self.prepare_seq.append(
                device.pg.parallel(self.rx1.get_pulse_sequence(0)[0], self.rx2.get_pulse_sequence(0)[0]))
            self.instructions_dict.update({'rx1_rx2_inst': 4})

            self.prepare_seq.extend(gate_sequence)
            self.instructions_dict.update({'cz_gate_inst': 5})

            # Prepare virtual sequence (without waveforms)
            self.prepare_virtual_seq = []
            self.prepare_virtual_seq.append(self.vs1[0])
            self.instructions_dict.update({'s1_inst': 6})

            self.prepare_virtual_seq.append(self.vs2[0])
            self.instructions_dict.update({'s2_inst': 7})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vs1[0], self.vs2[0]))
            self.instructions_dict.update({'s1_s2_inst': 8})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vrz_phi1[0], self.vrz_phi2[0]))
            self.instructions_dict.update({'phi1_phi2_inst': 9})

            self.prepare_virtual_seq.append(self.vrz1_pi[0])
            self.instructions_dict.update({'rz1_phi_inst': 10})

        def phi_setter(self, phi):
            phi = phi % (2 * np.pi) * 180 / np.pi
            index = np.where(self.phis == phi)[0][0]
            if phi == self.phis[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index(index)

        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '0.2'},
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
            for phase in self.phis:
                if (ex_seq.params['sequencer_id'] == self.data_qubit_sequence.params['sequencer_id']) and (
                        ex_seq.awg.device_id == self.data_qubit_awg.device_id):
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

            #             play_part = textwrap.dedent('''
            # //
            #     // Table entry play part
            #     executeTableEntry({calib_rf_inst});
            #     executeTableEntry({ref_increment_inst});
            #             ''').format(**self.instructions_dict)
            #
            #             if self.error_type == 'RX':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard data qubit
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_inst});
            #
            #     // Rz(phi) gate data qubit for error
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry(11 + variable_register2);
            #             ''').format(**self.instructions_dict)
            #             elif self.error_type == 'RZ':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Rz(phi) gate data qubit for error
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry(11 + variable_register2);
            #             ''').format(**self.instructions_dict)
            #             else:
            #                 raise ValueError("Can detect only RX or RZ errors!")
            #
            #             play_part += textwrap.dedent('''
            # //
            #     // Hadamard gate for both qubits
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #
            #     // CZ gate
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({cz_gate_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({phi1_phi2_inst});
            #             ''').format(**self.instructions_dict)
            #
            #             if self.error_type == 'RX':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard ancilla qubit
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s2_inst});
            #             ''').format(**self.instructions_dict)
            #             elif self.error_type == 'RZ':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard gate for both qubits
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #             ''').format(**self.instructions_dict)
            #             else:
            #                 raise ValueError("Can detect only RX or RZ errors!")

            play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry({calib_rf_inst});
    executeTableEntry({ref_increment_inst});

    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});

    // Rz(phi) gate data qubit for error
    executeTableEntry({ref_increment_inst});
    executeTableEntry({num_inst} + variable_register2);

    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    
    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    

    // CZ gate
    executeTableEntry({ref_increment_inst});
    executeTableEntry({cz_gate_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({phi1_phi2_inst});
    
    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});

    // CZ gate
    executeTableEntry({ref_increment_inst});
    executeTableEntry({cz_gate_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({phi1_phi2_inst});
    
    
    // Hadamard ancilla qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});




    // Return data qubit to initial state
    //executeTableEntry({rz1_phi_inst});

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry(12 + variable_register2);

    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});

    executeTableEntry({calib_rf_inst});
    resetOscPhase();''').format(**self.instructions_dict, num_inst=len(self.instructions_dict.keys()))

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

    measurement_type = 'error_detection_sa'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + data_qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=data_qubit_id),
                  'readout_pulse': readout_pulse.id}

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    fitter_arguments = ('iq' + data_qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))
    if multi_qubit_readout_flag:
        fitter_arguments = ('resultnumbers', exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(data_qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'error': error}
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (phis, setter.phi_setter, 'Z phase', 'rad'),
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references
                                                            )

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def frequency_drift(device, qubit_ids, gate, phis=None, *extra_sweep_args, additional_metadata={}, error='RX',
                       gate2=None,
                       multi_qubit_readout_flag=False):
    """
    Quantum circuit for Rx error detection
    data    -----H--Rz(phi)--H--H--H--H--H-----Meas
    ancilla --H--H-----------------------H--H--
    :param qubit_ids: [data qubit, ancilla qubit]
    """
    assert (error == 'RX' or error == 'RZ')
    data_qubit_id, ancilla_qubit_id = qubit_ids

    if multi_qubit_readout_flag:
        readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)
    else:
        # in this case only data is readout
        readout_pulse, measurer = get_uncalibrated_measurer(device, data_qubit_id)

    excitation_channel_data = list(device.get_qubit_excitation_channel_list(data_qubit_id).keys())[0]
    excitation_channel_ancilla = list(device.get_qubit_excitation_channel_list(ancilla_qubit_id).keys())[0]

    ex_channel = device.awg_channels[excitation_channel_data]

    ex_channel_gate = device.awg_channels[gate.metadata['carrier_name']]
    control_gate_channel_id = ex_channel_gate.channel % 2
    if control_gate_channel_id == 0:
        ampl = float(gate.metadata['amplitude'])
    elif control_gate_channel_id == 1:
        ampl = 1j * float(gate.metadata['amplitude'])
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: ampl})

    # if gate2:
    #     ex_channel_gate2 = device.awg_channels[gate2.metadata['carrier_name']]
    #     control_gate2_channel_id = ex_channel_gate2.channel % 2
    #     if control_gate2_channel_id == 0:
    #         ampl2 = complex(gate2.metadata['amplitude'])
    #     elif control_gate2_channel_id == 1:
    #         ampl2 = 1j * float(gate2.metadata['amplitude'])
    #     channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{gate2.metadata['carrier_name']: ampl2})

    data_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        data_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        data_qubit_seq_id = ex_channel.channel // 2

    ex_channel = device.awg_channels[excitation_channel_ancilla]
    ancilla_qubit_awg = ex_channel.parent.awg
    if ex_channel.is_iq():
        ancilla_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        ancilla_qubit_seq_id = ex_channel.channel // 2

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        if [awg, seq_id] == [data_qubit_awg, data_qubit_seq_id]:
            data_qubit_sequence = ex_seq
        if [awg, seq_id] == [ancilla_qubit_awg, ancilla_qubit_seq_id]:
            ancilla_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # ex_seq.start(holder=1)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    # ampl = complex(gate.metadata['amplitude'])
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.error_type = error

            # self.phis = phis / np.pi * 180
            self.phis = phis % (2 * np.pi) * 180 / np.pi

            # self.gate_freq = float(gate.metadata['frequency'])

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            self.rx1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=data_qubit_id,
                                                             rotation_angle=np.pi / 2, sort='newest', gauss=True)

            self.rx2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=ancilla_qubit_id,
                                                             rotation_angle=np.pi / 2, sort='newest', gauss=True)

            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                         channel_amplitudes=channel_amplitudes_,
                                                                         tail_length=float(
                                                                             gate.metadata['tail_length']),
                                                                         length=float(gate.metadata['length']),
                                                                         phase=0.0,
                                                                         fast_control=False)
            # if gate2:
            #     gate2_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
            #                                                                   channel_amplitudes=channel_amplitudes2_,
            #                                                                   tail_length=float(
            #                                                                       gate2.metadata['tail_length']),
            #                                                                   length=float(gate2.metadata['length']),
            #                                                                   phase=0.0,
            #                                                                   fast_control=False)

            self.vs1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id, phase=np.pi / 2 * 180 / np.pi,
                                              sort='newest',
                                              gauss=True)
            self.vs2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id, phase=np.pi / 2 * 180 / np.pi,
                                              sort='newest',
                                              gauss=True)

            self.vrz_phi1 = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                   phase=float(gate.metadata['phi1']) * 180 / np.pi, sort='newest',
                                                   gauss=True)
            self.vrz_phi2 = excitation_pulse.get_s(device=device, qubit_id=ancilla_qubit_id,
                                                   phase=float(gate.metadata['phi2']) * 180 / np.pi, sort='newest',
                                                   gauss=True)

            self.vrz1_pi = excitation_pulse.get_s(device=device, qubit_id=data_qubit_id,
                                                  phase=np.pi * 180 / np.pi, sort='newest',
                                                  gauss=True)

            self.control_sequence = control_sequence
            self.data_qubit_sequence = data_qubit_sequence
            self.data_qubit_awg = data_qubit_awg

            self.ancilla_qubit_sequence = ancilla_qubit_sequence
            self.ancilla_qubit_awg = ancilla_qubit_awg

            self.instructions_dict = {'calib_rf_inst': 0, 'ref_increment_inst': 1}

            # Prepare sequence with waveforms
            self.prepare_seq = []
            self.prepare_seq.append(self.rx1.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx1_inst': 2})

            self.prepare_seq.append(self.rx2.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx2_inst': 3})

            self.prepare_seq.append(
                device.pg.parallel(self.rx1.get_pulse_sequence(0)[0], self.rx2.get_pulse_sequence(0)[0]))
            self.instructions_dict.update({'rx1_rx2_inst': 4})

            self.prepare_seq.extend(gate_sequence)
            self.instructions_dict.update({'cz_gate_inst': 5})

            # Prepare virtual sequence (without waveforms)
            self.prepare_virtual_seq = []
            self.prepare_virtual_seq.append(self.vs1[0])
            self.instructions_dict.update({'s1_inst': 6})

            self.prepare_virtual_seq.append(self.vs2[0])
            self.instructions_dict.update({'s2_inst': 7})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vs1[0], self.vs2[0]))
            self.instructions_dict.update({'s1_s2_inst': 8})

            self.prepare_virtual_seq.append(device.pg.parallel(self.vrz_phi1[0], self.vrz_phi2[0]))
            self.instructions_dict.update({'phi1_phi2_inst': 9})

            self.prepare_virtual_seq.append(self.vrz1_pi[0])
            self.instructions_dict.update({'rz1_phi_inst': 10})

        def phi_setter(self, phi):
            phi = phi % (2 * np.pi) * 180 / np.pi
            index = np.where(self.phis == phi)[0][0]
            if phi == self.phis[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index(index)

        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '0.2'},
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
            for phase in self.phis:
                if (ex_seq.params['sequencer_id'] == self.data_qubit_sequence.params['sequencer_id']) and (
                        ex_seq.awg.device_id == self.data_qubit_awg.device_id):
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

            #             play_part = textwrap.dedent('''
            # //
            #     // Table entry play part
            #     executeTableEntry({calib_rf_inst});
            #     executeTableEntry({ref_increment_inst});
            #             ''').format(**self.instructions_dict)
            #
            #             if self.error_type == 'RX':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard data qubit
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_inst});
            #
            #     // Rz(phi) gate data qubit for error
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry(11 + variable_register2);
            #             ''').format(**self.instructions_dict)
            #             elif self.error_type == 'RZ':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Rz(phi) gate data qubit for error
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry(11 + variable_register2);
            #             ''').format(**self.instructions_dict)
            #             else:
            #                 raise ValueError("Can detect only RX or RZ errors!")
            #
            #             play_part += textwrap.dedent('''
            # //
            #     // Hadamard gate for both qubits
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #
            #     // CZ gate
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({cz_gate_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({phi1_phi2_inst});
            #             ''').format(**self.instructions_dict)
            #
            #             if self.error_type == 'RX':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard ancilla qubit
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s2_inst});
            #             ''').format(**self.instructions_dict)
            #             elif self.error_type == 'RZ':
            #                 play_part += textwrap.dedent('''
            # //
            #     // Hadamard gate for both qubits
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({rx1_rx2_inst});
            #     executeTableEntry({ref_increment_inst});
            #     executeTableEntry({s1_s2_inst});
            #             ''').format(**self.instructions_dict)
            #             else:
            #                 raise ValueError("Can detect only RX or RZ errors!")

            play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry({calib_rf_inst});
    executeTableEntry({ref_increment_inst});

    // Hadamard ancilla qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    
    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    
    // Hadamard data qubit
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    
    
    
    // Rz(phi) gate data qubit for error
    executeTableEntry({ref_increment_inst});
    executeTableEntry({num_inst} + variable_register2);
    
    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    
    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    
    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    
    // Hadamard data qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_inst});
    
    
    
    
    // Hadamard data qubit
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({rx1_inst});
    //executeTableEntry({ref_increment_inst});
    //executeTableEntry({s1_inst});
    
    // Hadamard gate for both qubits
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx1_rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s1_s2_inst});
    
    // Hadamard ancilla qubit
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({rx2_inst});
    executeTableEntry({ref_increment_inst});
    executeTableEntry({s2_inst});
    

    executeTableEntry({calib_rf_inst});
    resetOscPhase();''').format(**self.instructions_dict, num_inst=len(self.instructions_dict.keys()))

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

    measurement_type = 'error_detection_sa'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + data_qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=data_qubit_id),
                  'readout_pulse': readout_pulse.id}

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    fitter_arguments = ('iq' + data_qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))

    metadata = {'qubit_id': ','.join(data_qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'error': error}
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (phis, setter.phi_setter, 'Z phase', 'rad'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement





