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

def CPhase_calibration(device, qubit_ids, gate, amplitude, length, transition='01', phis=None, tail_length=0e-9,
                       readout_delay=0e-9, *extra_sweep_args, repeats=1, pulse_U=None, sign=False,
                       additional_metadata={}, gate_freq=None):
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

    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']: complex(amplitude)})
    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.length = length - 2 * tail_length
            self.phis = phis / np.pi * 180
            self.tail_length = tail_length
            self.amplitude = amplitude
            self.gate_freq = gate_freq
            if self.gate_freq is None:
                self.gate_freq = float(gate.metadata['frequency'])
            self.frequency = 0
            self.sign = sign

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            # pi / 2 pulse for target qubit
            self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # pi / 2 pulse for control qubit
            self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=control_qubit_id,
                                                                    rotation_angle=np.pi / 2, sort='newest', gauss=True)
            # gate pulse
            fast_control = False
            channel_pulses = [(c, device.pg.rect_cos, complex(a), self.tail_length, fast_control) for c, a in channel_amplitudes_.metadata.items()]
            gate_sequence = device.pg.pmulti(device, self.length, *tuple(channel_pulses))

            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.control_qubit_awg = control_qubit_awg

            self.prepare_seq = []

            if pulse_U == 'I':
                self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            elif pulse_U == 'X':
                # add prepulse for target qubit
                self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])
                self.prepare_seq.append(self.pre_pulse2.get_pulse_sequence(0)[0])

                self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])
            else:
                raise ValueError('Prepulse can be only I or X!')

            self.prepare_seq.append(gate_sequence)
            self.prepare_seq.append(self.pre_pulse1.get_pulse_sequence(0)[0])

        def phi_setter(self, phi):
            phi = phi / np.pi * 180
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

    executeTableEntry(2);
    executeTableEntry(3);

    executeTableEntry(5+variable_register2);
    executeTableEntry(4);
''')
            elif pulse_U == 'X':
                play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry(0);
    executeTableEntry(1);

    executeTableEntry(2);
    executeTableEntry(3);

    executeTableEntry(4);
    executeTableEntry(5);
    executeTableEntry(7+variable_register2);
    executeTableEntry(6);
                ''')
            else:
                raise ValueError('Prepulse can be only I or X!')
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

    arg_id = -1
    fitter_arguments = (measurement_name, SinglePeriodSinFitter(), arg_id, np.arange(len(extra_sweep_args)))

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


