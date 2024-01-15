from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
import textwrap
import time


def Ramsey(device, qubit_id, transition='01', *extra_sweep_args, channel_amplitudes=None, lengths=None,
           target_freq_offset=None, readout_delay=None, pre_pulse_gate=None, measurement_type='Ramsey',
           additional_references = {}, additional_metadata = {}, gauss = True , sort = 'best', qutrit_readout=False):

    from .readout_pulse2 import get_uncalibrated_measurer
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition, qutrit_readout=qutrit_readout)

    ex_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., transition=transition,
                                                      channel_amplitudes_override=channel_amplitudes, gauss = gauss,
                                                      sort = sort)

    excitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=transition).keys()][0]
    ex_channel = device.awg_channels[excitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel//2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            #ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        if ex_seq.params['is_iq']:
            ex_seq.start()
        else:
            ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    class ParameterSetter:
        def __init__(self):
            self.device = device
            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            self.control_qubit_awg = control_qubit_awg
            self.instructions = []

            self.lengths = lengths
            self.phases = np.asarray([self.virtual_phase(length) for length in  self.lengths])

            if len(self.phases) < 2**8:
               self.incrementation_type = 'phase'
            else:
                self.incrementation_type = 'quasi-binary'

            self.prepare_seq = []
            self.instructions_dict = {'rf_calibration': 0, 'ref_incrementation': 1}

            self.prepare_seq.append(ex_pulse.get_pulse_sequence(0)[0])
            self.instructions_dict.update({'rx2': 2})

            self.prepare_seq.append(device.pg.pmulti(device, self.lengths))

            for i in range(9):
                self.instructions_dict.update({'w' + str(i): 3 + i})
            self.instructions_dict.update({'tail_fall': list(self.instructions_dict.values())[-1] + 1})

        def virtual_phase(self, length):
            """
            Calculate virtual Z phase from length
            """
            phase = int(np.round(length * self.control_sequence.clock)) / self.control_sequence.clock * target_freq_offset * 360 % 360
            return  phase

        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
            #                  'header': {'version': '0.2'},
            #                  'table': []}

            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '1.0'},
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
                                # definition_part += single_sequence[3][id_]
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

            # prepare command table for phase incrementation
            if self.incrementation_type == 'phase':
                for phase in self.phases:
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
            else:
                for binary in range(2**8):
                    if (ex_seq.params['sequencer_id'] == self.control_qubit_sequence.params['sequencer_id']) and (
                            ex_seq.awg.device_id == self.control_qubit_awg.device_id):

                        table_entry = {'index': random_command_id}
                        table_entry['phase0'] = {'value': binary * 360 / 2**8, 'increment': True}
                        table_entry['phase1'] = {'value': binary * 360 / 2**8, 'increment': True}
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
    
    executeTableEntry({rx2});
    
    executeTableEntry(variable_register1+3);
    wait(variable_register0);
    executeTableEntry({tail_fall});
    
    executeTableEntry({tail_fall} + 1 + variable_register2);
    executeTableEntry({rx2});
    
    executeTableEntry({rf_calibration});
    resetOscPhase();
''').format(**self.instructions_dict)
            self.instructions.append(command_table)
            print('Command table for sequencer id {}'.format(ex_seq.params['sequencer_id']), command_table)
            return definition_part, play_part


        def set_delay(self, length):

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
                if self.incrementation_type == 'phase':
                    index = np.where(self.lengths == length)[0][0]
                else:
                    index = np.where(self.lengths == length)[0][0]
                    phase = self.phases[index]
                    index = int(np.round(phase / (360 / 2 ** 8)))
                seq.set_phase_index(index)

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

    setter = ParameterSetter()

    references = {'ex_pulse1': ex_pulse.id,
                  'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id),
                  'readout_pulse': readout_pulse.id}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))
    if qutrit_readout:
        fitter_arguments = ('resultnumbers_states', exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': qubit_id,
                'transition': transition,
                'extra_sweep_args':str(len(extra_sweep_args)),
                'target_offset_freq': str(target_freq_offset),
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (lengths, setter.set_delay, 'Delay', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=10)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement
