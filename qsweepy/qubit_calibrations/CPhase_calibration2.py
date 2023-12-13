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

def CPhase_calibration(device, qubit_ids, gate, amplitudes, length, transition='01', phis=None, tail_length=0e-9,
                       readout_delay=0e-9, *extra_sweep_args, repeats=1, pulse_U=None, sign=False,
                       additional_metadata={}, gate_freq=None, bipolar_pulse=False):
    """
    Provide CPhase calibration
    :param device:
    :param qubit_ids: [control qubit id, target qubit]
    :param pulse_U: additional prepulse on a control qubit identity  or X gate
    """

    control_qubit_id, qubit_id = qubit_ids
    assert (pulse_U == 'I' or pulse_U == 'X')

    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    excitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[excitation_channel]
    ex_channel_gate = device.awg_channels[gate.metadata['carrier_name']]
    control_gate_channel_id = ex_channel_gate.channel % 2
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            # ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            # print('control_sequence=',control_sequence)
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()
    if control_gate_channel_id == 0:
        ampl = 1 + 0j
    elif control_gate_channel_id == 1:
        ampl = 0 + 1j
    # channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(amplitudes[0])})
    # channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,**{gate.metadata['carrier_name']: complex(1)})
    # channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(-1)})
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: ampl})
    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: -ampl})
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.length = length - 2 * tail_length
            # self.phis = phis / np.pi * 180
            self.phis = phis  % (2 * np.pi) * 180 / np.pi
            self.tail_length = tail_length
            # self.amplitude = amplitude
            self.amplitudes = amplitudes
            self.amplitude = amplitudes[0]
            self.gate_freq = gate_freq
            if self.gate_freq is None:
                self.gate_freq = float(gate.metadata['frequency'])
            self.frequency = 0
            self.sign = sign

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            # pi / 2 pulse for control qubit (первый в списке кубит)
            self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=control_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # pi / 2 pulse for target qubit (второй в списке кубит)
            self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # gate pulse
            fast_control = False
            # channel_pulses = [(c, device.pg.rect_cos, complex(a), self.tail_length, fast_control) for c, a in channel_amplitudes_.metadata.items()]
            # gate_sequence = device.pg.pmulti(device, self.length, *tuple(channel_pulses))

            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_,
                                                                       tail_length=self.tail_length,
                                                                       length=self.length,
                                                                       phase=0.0,
                                                                       fast_control=False)

            gate_sequence1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                         channel_amplitudes=channel_amplitudes1_,
                                                                         tail_length=self.tail_length,
                                                                         length=self.length,
                                                                         phase=0.0,
                                                                         fast_control=False)

            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.control_qubit_awg = control_qubit_awg

            self.prepare_seq = []

            # if pulse_U == 'I':
            #     self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            # elif pulse_U == 'X':
            #     # add prepulse for target qubit
            #     self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])
            #     self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))
            #     self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            # else:
            #     raise ValueError('Prepulse can be only I or X!')

            self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])
            self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))


            self.prepare_seq.extend(gate_sequence)
            if bipolar_pulse:
                self.prepare_seq.extend(gate_sequence1)
            # self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])

        def phi_setter(self, phi):
            # phi = phi / np.pi * 180
            phi = phi  % (2 * np.pi) * 180 / np.pi
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

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

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
            print('Command table phases {} and {}'.format(phase0, phase1))

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

                            if not ex_seq.is_iq:
                                if control_gate_channel_id == 0:
                                    table_entry['amplitude0'] = {'value': self.amplitude}
                                    table_entry['amplitude1'] = {'value': 1.0}
                                elif control_gate_channel_id == 1:
                                    table_entry['amplitude0'] = {'value': 1.0}
                                    table_entry['amplitude1'] = {'value': self.amplitude}

                                # # TODO: this is kostyl for amplitude changing via command table, make it better
                                # table_entry['amplitude0'] = {'value': self.amplitude}
                                # table_entry['amplitude1'] = {'value': 1.0}

                            command_table['table'].append(table_entry)



            for phase in self.phis:
                if (ex_seq.params['sequencer_id'] == self.control_qubit_sequence.params['sequencer_id']) and (
                        ex_seq.awg.device_id == self.control_qubit_awg.device_id):

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
            if pulse_U == 'I':
                play_part = textwrap.dedent('''
                                    // Table entry play part
                                        executeTableEntry(0);
                                        
                                        executeTableEntry(1);
                                        executeTableEntry(3);
                                        
                                        cvar j;
                                        wait(5);
                                        for (j=0;j<{repeats}; j++){{
                                            executeTableEntry(1);
                                            executeTableEntry(5);
                                            wait(5);
                                        }}
                                        executeTableEntry(1);
                                        executeTableEntry(6+variable_register2);
                                        executeTableEntry(1);
                                        executeTableEntry(3);

                                        executeTableEntry(0);
                                        resetOscPhase();
                                    ''').format(repeats=repeats)
                # play_part = textwrap.dedent('''
                #                     // Table entry play part
                #                         executeTableEntry(0);
                #                         executeTableEntry(1);
                #
                #                         executeTableEntry(3);
                #
                #                         executeTableEntry(6+variable_register2);
                #                         executeTableEntry(3);
                #                     ''').format(repeats=repeats)
            elif pulse_U == 'X':
                play_part = textwrap.dedent('''
                                    // Table entry play part
                                        executeTableEntry(0);
                                        
                                        executeTableEntry(1);
                                        executeTableEntry(2);
                                        executeTableEntry(1);
                                        executeTableEntry(4);
                                        cvar j;
                                        wait(5);
                                        //for (j=0;j<{repeats}; j++){{
                                        //    executeTableEntry(1);
                                        //    executeTableEntry(5);
                                        //    wait(5);
                                        //}}
                                        executeTableEntry(1);
                                        executeTableEntry(6+variable_register2);
                                        executeTableEntry(1);
                                        executeTableEntry(4);
                                        executeTableEntry(1);
                                        executeTableEntry(2);

                                        executeTableEntry(0);
                                        resetOscPhase();
                                    ''').format(repeats=repeats)
            else:
                raise ValueError('Prepulse can be only I or X!')

            if bipolar_pulse:
                if pulse_U == 'I':
                    play_part = textwrap.dedent('''
                                        // Table entry play part
                                            executeTableEntry(0);

                                            executeTableEntry(1);
                                            executeTableEntry(3);

                                            cvar j;
                                            wait(5);
                                            for (j=0;j<{repeats}; j++){{
                                                executeTableEntry(1);
                                                executeTableEntry(5);
                                                executeTableEntry(1);
                                                executeTableEntry(6);
                                                wait(5);
                                            }}
                                            executeTableEntry(1);
                                            executeTableEntry(7+variable_register2);
                                            executeTableEntry(1);
                                            executeTableEntry(3);

                                            executeTableEntry(0);
                                            resetOscPhase();
                                        ''').format(repeats=repeats)
                elif pulse_U == 'X':
                    play_part = textwrap.dedent('''
                                        // Table entry play part
                                            executeTableEntry(0);

                                            executeTableEntry(1);
                                            executeTableEntry(2);
                                            executeTableEntry(1);
                                            executeTableEntry(4);
                                            cvar j;
                                            wait(5);
                                            for (j=0;j<{repeats}; j++){{
                                                executeTableEntry(1);
                                                executeTableEntry(5);
                                                executeTableEntry(1);
                                                executeTableEntry(6);
                                                wait(5);
                                            }}
                                            executeTableEntry(1);
                                            executeTableEntry(7+variable_register2);
                                            executeTableEntry(1);
                                            executeTableEntry(4);
                                            executeTableEntry(1);
                                            executeTableEntry(2);

                                            executeTableEntry(0);
                                            resetOscPhase();
                                        ''').format(repeats=repeats)
                else:
                    raise ValueError('Prepulse can be only I or X!')
            # if pulse_U == 'I':
            #     play_part = textwrap.dedent('''
            #                         // Table entry play part
            #                             executeTableEntry(0);
            #                             executeTableEntry(1);
            #
            #                             executeTableEntry(3);
            #                             cvar j;
            #                             wait(5);
            #                             for (j=0;j<{repeats}; j++){{
            #                                 executeTableEntry(5);
            #                                 wait(5);
            #                             }}
            #                             executeTableEntry(6+variable_register2);
            #                             executeTableEntry(3);
            #
            #                             //executeTableEntry(0);
            #                             //resetOscPhase();
            #                         ''').format(repeats=repeats)
            #     # play_part = textwrap.dedent('''
            #     #                     // Table entry play part
            #     #                         executeTableEntry(0);
            #     #                         executeTableEntry(1);
            #     #
            #     #                         executeTableEntry(3);
            #     #
            #     #                         executeTableEntry(6+variable_register2);
            #     #                         executeTableEntry(3);
            #     #                     ''').format(repeats=repeats)
            # elif pulse_U == 'X':
            #     play_part = textwrap.dedent('''
            #                         // Table entry play part
            #                             executeTableEntry(0);
            #                             executeTableEntry(1);
            #
            #                             executeTableEntry(2);
            #                             executeTableEntry(4);
            #                             cvar j;
            #                             wait(5);
            #                             for (j=0;j<{repeats}; j++){{
            #                                 executeTableEntry(5);
            #                                 wait(5);
            #                             }}
            #                             executeTableEntry(6+variable_register2);
            #                             executeTableEntry(4);
            #                             executeTableEntry(2);
            #
            #                             //executeTableEntry(0);
            #                             //resetOscPhase();
            #                         ''').format(repeats=repeats)
            #     # play_part = textwrap.dedent('''
            #     #                     // Table entry play part
            #     #                         executeTableEntry(0);
            #     #                         executeTableEntry(1);
            #     #
            #     #                         executeTableEntry(2);
            #     #                         executeTableEntry(4);
            #     #
            #     #                         executeTableEntry(6+variable_register2);
            #     #                         executeTableEntry(4);
            #     #                         executeTableEntry(2);
            #     #                     ''').format(repeats=repeats)
            # else:
            #     raise ValueError('Prepulse can be only I or X!')
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


    measurement_type = 'CPhase_calibration'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id),
                  'channel_amplitudes': channel_amplitudes_.id,
                  'readout_pulse': readout_pulse.id}


    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    # fitter_arguments = (measurement_name, SinglePeriodSinFitter(), arg_id, np.arange(len(extra_sweep_args)))
    fitter_arguments = ('iq' + qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)+1))

    metadata = {'qubit_id': ','.join(qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition': transition}
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (amplitudes, setter.amplitude_setter, 'Voltage', 'V'),
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


def CPhase_calibration_leakage(device, qubit_ids, gate, amplitudes, lengths, transition='01', tail_length=0e-9,
                               readout_delay=0e-9, *extra_sweep_args, repeats=1, sign=False,
                               additional_metadata={}, gate_freq=None, initstate=None):
    """
    Provide CPhase calibration
    :param device:
    :param qubit_ids: [control qubit id, target qubit]
    :param pulse_U: additional prepulse on a control qubit identity  or X gate
    """

    control_qubit_id, qubit_id = qubit_ids

    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    excitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[excitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            # ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            # print('control_sequence=',control_sequence)
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    # redefine lengths as integer val
    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    # channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(amplitude)})
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(1)})
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.lengths = lengths
            self.tail_length = tail_length
            # self.amplitude = amplitude
            self.amplitudes = amplitudes
            self.amplitude = amplitudes[0]
            self.gate_freq = gate_freq
            if self.gate_freq is None:
                self.gate_freq = float(gate.metadata['frequency'])
            self.frequency = 0
            self.sign = sign

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer


            # pi / 2 pulse for control qubit (первый в списке кубит)
            self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=control_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # pi / 2 pulse for target qubit (второй в списке кубит)
            self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # fast control is true
            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_,
                                                                       tail_length=self.tail_length,
                                                                       length=self.lengths,
                                                                       phase=0.0,
                                                                       fast_control=True)

            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.control_qubit_awg = control_qubit_awg

            self.prepare_seq = []
            # self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))
            # self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            # self.prepare_seq.extend(gate_sequence)

            self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])
            self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))
            self.prepare_seq.extend(gate_sequence)



        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def set_length(self, length):
            if length == self.lengths[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_length(length)

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

            waveforms  = {}

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

                            if not ex_seq.is_iq:
                                control_channel_id = ex_channel.channel % 2
                                # TODO: this is kostyl for amplitude changing via command table, make it better
                                table_entry['amplitude0'] = {'value': self.amplitude}
                                table_entry['amplitude1'] = {'value': 1.0}

                            command_table['table'].append(table_entry)

                        if initstate == '00':
                            pre_pulse_command = 0
                        elif initstate == '10':
                            pre_pulse_command = 2
                        elif initstate == '01':
                            pre_pulse_command = 3
                        elif initstate == '11':
                            pre_pulse_command = 4
                        else:
                            raise ValueError('wrong initstate')

                        if pre_pulse_command == 0:
                            play_part = textwrap.dedent('''
                            // Table entry play part
                              executeTableEntry(0);
                              executeTableEntry(1);

                              executeTableEntry(variable_register1+5);
                              wait(variable_register0);
                              executeTableEntry(14);

                            ''')
                        else:
                            play_part = textwrap.dedent('''
                            // Table entry play part
                              executeTableEntry(0);
                              executeTableEntry(1);

                              executeTableEntry({0});
                              executeTableEntry({0});

                              executeTableEntry(variable_register1+5);
                              wait(variable_register0);
                              executeTableEntry(14);
                              wait(5);

                              executeTableEntry({0});
                              executeTableEntry({0});
                            ''').format(pre_pulse_command)


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


    measurement_type = 'CPhase_calibration_leakage'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id),
                  'channel_amplitudes': channel_amplitudes_.id,
                  'readout_pulse': readout_pulse.id}


    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    # fitter_arguments = (measurement_name, SinglePeriodSinFitter(), arg_id, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition': transition,
                # 'amplitude': str(amplitude)
                }
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (lengths, setter.set_length, 'Excitation length', 's'),
                                                            (amplitudes, setter.amplitude_setter, 'Voltage', 'V'),
                                                            # fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement




def CPhase_calibration_fast(device, qubit_ids, gate, amplitudes, lengths, transition='01', tail_length=0e-9,
                               readout_delay=0e-9, *extra_sweep_args, repeats=1, sign=False,
                               additional_metadata={}, gate_freq=None, control_q_initstate=None):
    """
    Provide CPhase calibration
    :param device:
    :param qubit_ids: [control qubit id, target qubit]
    """

    control_qubit_id, qubit_id = qubit_ids

    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    excitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[excitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            # ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            # print('control_sequence=',control_sequence)
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    # redefine lengths as integer val
    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round((lengths[_i]) * control_sequence.clock))
    lengths = times / control_sequence.clock
    # lengths = lengths - 2 * tail_length

    # channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(amplitude)})
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(1)})
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.tail_length = tail_length
            self.lengths = lengths # - 2 * tail_length
            # self.length = self.lengths[0]
            # print('LENGTH', lengths, self.lengths)
            # self.amplitude = amplitude
            self.amplitudes = amplitudes
            self.amplitude = amplitudes[0]
            self.gate_freq = gate_freq
            if self.gate_freq is None:
                self.gate_freq = float(gate.metadata['frequency'])
            self.frequency = 0
            self.sign = sign

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer


            # pi / 2 pulse for control qubit (первый в списке кубит)
            self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=control_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # pi / 2 pulse for target qubit (второй в списке кубит)
            self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # fast control is true
            gate_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_,
                                                                       tail_length=self.tail_length,
                                                                       length=self.lengths,
                                                                       phase=0.0,
                                                                       fast_control=True)

            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.control_qubit_awg = control_qubit_awg

            self.prepare_seq = []
            # self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))
            # self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            # self.prepare_seq.extend(gate_sequence)

            self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])
            self.prepare_seq.append(device.pg.parallel(self.pre_pulse2.get_pulse_sequence(0)[0], self.pre_pulse1.get_pulse_sequence(0)[0]))
            self.prepare_seq.extend(gate_sequence)

            self.control_q_initstate = control_q_initstate



        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def set_length(self, length):
            if length == self.lengths[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_length(length)

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

            waveforms  = {}

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

                            if not ex_seq.is_iq:
                                # TODO: this is kostyl for amplitude changing via command table, make it better
                                table_entry['amplitude0'] = {'value': self.amplitude}
                                table_entry['amplitude1'] = {'value': 1.0}

                            command_table['table'].append(table_entry)

            if self.control_q_initstate == '0':
                play_part = textwrap.dedent('''
// Table entry play part
  executeTableEntry(0);
  executeTableEntry(1);

  executeTableEntry(3);

  executeTableEntry(variable_register1+5);
  wait(variable_register0);
  executeTableEntry(14);
  executeTableEntry(14);
  wait(5);

  executeTableEntry(3);
    ''')
            elif self.control_q_initstate == '1':
                play_part = textwrap.dedent('''
// Table entry play part
  executeTableEntry(0);
  executeTableEntry(1);

  executeTableEntry(2);
  executeTableEntry(4);

  executeTableEntry(variable_register1+5);
  wait(variable_register0);
  executeTableEntry(14);
  wait(5);

  executeTableEntry(4);
  executeTableEntry(2);
  // executeTableEntry(3);
    ''')


            else:
                raise ValueError('wrong initstate')



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


    measurement_type = 'CPhase_calibration_fast'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id

    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id),
                  'channel_amplitudes': channel_amplitudes_.id,
                  'readout_pulse': readout_pulse.id}


    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # arg_id = -1
    # fitter_arguments = (measurement_name, SinglePeriodSinFitter(), arg_id, np.arange(len(extra_sweep_args)))
    # fitter_arguments = ('iq' + qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)+1))
    fitter_arguments = ('iq' + qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))


    metadata = {'qubit_id': ','.join(qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition': transition,
                # 'amplitude': str(amplitude)
                }
    metadata.update(additional_metadata)

    references['long_process'] = gate.id
    references['readout_pulse'] = readout_pulse.id

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (amplitudes, setter.amplitude_setter, 'Voltage', 'V'),
                                                            (lengths, setter.set_length, 'Excitation length', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement