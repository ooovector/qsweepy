# from qsweepy.fitters.exp_sin import exp_sin_fitter
# from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
# import numpy as np
# from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
# from qsweepy.qubit_calibrations import sequence_control
# from qsweepy import zi_scripts
# import traceback
# import textwrap
# import time
#
#
# def Rabi_Mark(device, qubit_id, channel_amplitudes, transition='01',
#               lengths=None, *extra_sweep_args, phi1 = 0, phi2 = 0, phi3 = 0, basis = 'Z', tail_length=0,
#               readout_delay=0, pre_pulses=tuple(), repeats=1, measurement_type='Rabi_Mark', samples=False,
#               additional_metadata={}, comment='', shots=False,
#               dot_products=False, post_selection_flag=False, qutrit_readout=False):
#     """
#     Rect rect sweep measurement via a command table
#     """
#     from qsweepy.qubit_calibrations.readout_pulse2 import get_uncalibrated_measurer
#     from qsweepy.qubit_calibrations.calibrated_readout2 import get_calibrated_measurer
#
#     if post_selection_flag:
#         readouts_per_repetition = 2
#     else:
#         readouts_per_repetition = 1
#
#     if type(qubit_id) is not list and type(
#             qubit_id) is not tuple:  # if we are working with a single qubit, use uncalibrated measurer
#         readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition=transition,
#                                                             samples=samples,
#                                                             shots=shots, dot_products=dot_products,
#                                                             readouts_per_repetition=readouts_per_repetition,
#                                                             qutrit_readout=qutrit_readout)
#         measurement_name = 'iq' + qubit_id
#         if qutrit_readout:
#             measurement_name = 'resultnumbers_states'
#
#         qubit_id = [qubit_id]
#         exp_sin_fitter_mode = 'unsync'
#         excitation_channel = \
#         [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]
#     else:  # otherwise use calibrated measurer
#         readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)
#         measurement_name = 'resultnumbers'
#         exp_sin_fitter_mode = 'unsync'
#         excitation_channel = \
#         [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]
#
#     ex_channel = device.awg_channels[excitation_channel]
#     if ex_channel.is_iq():
#         control_qubit_awg = ex_channel.parent.awg
#         control_qubit_seq_id = ex_channel.parent.sequencer_id
#     else:
#         control_qubit_awg = ex_channel.parent.awg
#         control_qubit_seq_id = ex_channel.channel // 2
#     control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
#     ex_sequencers = []
#
#     # define excitation pre pulses seqs
#     ex_pre_pulses_seq = []
#     for ex_pre_pulse in pre_pulses:
#         if ex_pre_pulse.metadata['pulse_type'] == 'rect':
#             ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0)
#         else:
#             # in case of gauss pulse we use two pulses with pi/2 rotation
#             ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0) + ex_pre_pulse.get_pulse_sequence(0)
#         ex_pre_pulses_seq.append(ex_pre_pulse_seq)
#
#     for awg, seq_id in device.pre_pulses.seq_in_use:
#         if [awg, seq_id] != [control_awg, control_seq_id]:
#             ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
#                                                awg_amp=1, use_modulation=True, pre_pulses=[],
#                                                post_selection_flag=post_selection_flag)
#         else:
#             ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
#                                                awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
#                                                post_selection_flag=post_selection_flag)
#             control_sequence = ex_seq
#         if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
#             control_qubit_sequence = ex_seq
#         device.pre_pulses.set_seq_offsets(ex_seq)
#         device.pre_pulses.set_seq_prepulses(ex_seq)
#         ex_seq.start(holder=1)
#         ex_sequencers.append(ex_seq)
#
#     # TODO: how to make excitation pre pulses setting better
#     if ex_pre_pulses_seq:
#         for _id, ex_sequence in enumerate(ex_sequencers):
#             ex_sequence.awg.stop_seq(ex_sequence.params['sequencer_id'])
#             ex_sequence.clear_pulse_sequence()
#             for prep_seq in ex_pre_pulse_seq:
#                 for seq_id, single_sequence in prep_seq[0][ex_sequence.awg.device_id].items():
#                     if seq_id == ex_sequence.params['sequencer_id']:
#                         ex_sequence.add_exc_pre_pulse(single_sequence[0], single_sequence[1])
#             ex_sequence.awg.set_sequence(ex_sequence.params['sequencer_id'], ex_sequence)
#
#     readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
#     readout_sequencer.start()
#
#     # redefine lengths as integer val
#     times = np.zeros(len(lengths))
#     for _i in range(len(lengths)):
#         times[_i] = int(round(lengths[_i] * control_sequence.clock))
#     lengths = times / control_sequence.clock
#
#     class ParameterSetter:
#         def __init__(self):
#             self.device = device
#             self.ex_sequencers = ex_sequencers
#             self.readout_sequencer = readout_sequencer
#             self.control_sequence = control_sequence
#             self.control_qubit_sequence = control_qubit_sequence
#             self.length_tail = tail_length
#             self.instructions = []
#
#             self.lengths = lengths
#             fast_control = True
#
#             self.prepare_seq = []
#
#             self.rx = excitation_pulse.get_excitation_pulse(device=device, qubit_id = qubit_id[0],
#                                                                     rotation_angle=np.pi / 2, sort='newest', gauss=True)
#
#             self.prepare_seq.append(self.rx.get_pulse_sequence(0)[0])
#
#             channel_pulses = [(c, device.pg.rect_cos, complex(a), self.length_tail, fast_control) for c, a in
#                               channel_amplitudes.metadata.items()]
#
#             self.prepare_seq.append(device.pg.pmulti(device, self.lengths, *tuple(channel_pulses)))
#
#         def set_length(self, length):
#             """
#             Set length into reg0 and reg1 values and create hdawg program
#             """
#             if length == self.lengths[0]:
#                 # for the first time you need to create hdawg generator
#                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
#                 prepare_seq = self.create_hdawg_generator()
#                 sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
#                                                           instructions=self.instructions)
#                 time.sleep(0.1)
#                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
#
#             for seq in ex_sequencers:
#                 seq.set_length(length)
#
#         def return_hdawg_program(self, ex_seq):
#             """
#             Return hdawg program for defined ex_seq object
#             :param ex_seq: excitation sequencer object
#             """
#             definition_part = ''''''
#
#             # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#             #                  'header': {'version': '0.2'},
#             #                  'table': []}
#             # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#             #                  'header': {'version': '1.0'},
#             #                  'table': []}
#             command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#                              'header': {'version': '1.2'},
#                              'table': []}
#             #
#             # command_table = {'$schema': 'https://json-schema.org/draft-04/schema#',
#             #                  'header': {'version': '1.2.0'},
#             #                  'table': []}
#
#             assign_waveform_indexes = {}
#             random_command_id = 7
#             waveform_id = -1
#
#             # get information about I and Q phases from calibrations in case of iq excitation channel
#             phase0 = ex_seq.phaseI
#             phase1 = ex_seq.phaseQ
#
#             # cycle for all prepare seqs
#             for prep_seq in self.prepare_seq:
#                 # cycle for all used hdawg
#                 for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
#                     if seq_id == ex_seq.params['sequencer_id']:
#
#                         # add waveform definition part
#                         if single_sequence[0] not in definition_part:
#                             definition_part += single_sequence[0]
#
#                         assign_fragments = single_sequence[3].split(';')[:-1]
#
#                         for id_, entry_table_index_constant in enumerate(single_sequence[2]):
#                             table_entry = {'index': random_command_id}  # define command table id index
#                             random_command_id += 1
#                             if entry_table_index_constant not in assign_waveform_indexes.keys():
#                                 waveform_id += 1
#                                 definition_part += '\nconst ' + entry_table_index_constant + '={_id};'.format(
#                                     _id=waveform_id)
#                                 assign_waveform_indexes[entry_table_index_constant] = waveform_id
#                                 # definition_part += single_sequence[3][id_]
#                                 definition_part += assign_fragments[id_] + ';'
#                                 table_entry['waveform'] = {'index': waveform_id}
#                             else:
#                                 table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}
#
#                             random_pulse = single_sequence[4]
#                             if 'phase0' in random_pulse:
#                                 table_entry['phase0'] = random_pulse['phase0']
#                             if 'phase1' in random_pulse:
#                                 table_entry['phase1'] = random_pulse['phase1']
#                             command_table['table'].append(table_entry)
#
#             # Add two auxiliary table entries, one for phase calibrations and one for reference phase
#             table_entry = {'index': 0}
#             table_entry['phase0'] = {'value': phase0, 'increment': False}
#             table_entry['phase1'] = {'value': phase1, 'increment': False}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 1}
#             table_entry['phase0'] = {'value': 0, 'increment': True}
#             table_entry['phase1'] = {'value': 0, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 2}
#             table_entry['phase0'] = {'value': 90, 'increment': True}
#             table_entry['phase1'] = {'value': 90, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 3}
#             table_entry['phase0'] = {'value': 180, 'increment': True}
#             table_entry['phase1'] = {'value': 180, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 4}
#             table_entry['phase0'] = {'value': phi1, 'increment': True}
#             table_entry['phase1'] = {'value': phi1, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 5}
#             table_entry['phase0'] = {'value': phi2, 'increment': True}
#             table_entry['phase1'] = {'value': phi2, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 6}
#             table_entry['phase0'] = {'value': phi3, 'increment': True}
#             table_entry['phase1'] = {'value': phi3, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             play_part = textwrap.dedent('''
# // Table entry play part
#     executeTableEntry(0);
#     executeTableEntry(1);
#
#     // Generation of random initial point
#     executeTableEntry(4);
#     executeTableEntry(7);
#     executeTableEntry(5);
#     executeTableEntry(7);
#     executeTableEntry(6);
#
#     // Rabi sequence
#     executeTableEntry(variable_register1+8);
#     wait(variable_register0);
#     executeTableEntry(17);
#
#
#     ''')
#             if basis == 'Z':
#                 play_part+=textwrap.dedent('''
#     // Z-basis
#     resetOscPhase();
#                 ''')
#
#             if basis == 'X':
#                 play_part += textwrap.dedent('''
#     // X-basis
#     executeTableEntry(2);
#     executeTableEntry(7);
#     resetOscPhase();
#                 ''')
#
#             if basis == 'Y':
#                 play_part += textwrap.dedent('''
#     // Y-basis
#     executeTableEntry(7);
#     resetOscPhase();
#                 ''')
#
#
#
#             self.instructions.append(command_table)
#             print('Command table for sequencer id {}'.format(ex_seq.params['sequencer_id']), command_table)
#             return definition_part, play_part
#
#         def create_hdawg_generator(self):
#             pulses = {}
#             control_seq_ids = []
#             control_awg_ids = []
#             self.instructions = []
#
#             for device_id in self.prepare_seq[0][0].keys():
#                 pulses.update({device_id: {}})
#
#             for ex_seq in self.ex_sequencers:
#                 pulses[ex_seq.awg.device_id][ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
#                 control_seq_ids.append(ex_seq.params['sequencer_id'])
#                 control_awg_ids.append(ex_seq.awg.device_id)
#
#             return [[pulses, control_seq_ids, control_awg_ids]]
#
#     # Add metadata
#     metadata = {'qubit_id': ','.join(qubit_id),
#                 'transition': transition,
#                 'extra_sweep_args': str(len(extra_sweep_args)),
#                 'tail_length': str(tail_length),
#                 'readout_delay': str(readout_delay),
#                 'repeats': str(repeats),
#                 'amplitude': channel_amplitudes.metadata[excitation_channel],
#                 'command_table': str(True),
#                 'phi1': str(phi1),
#                 'phi2': str(phi2),
#                 'phi3': str(phi3),
#                 'basis': basis
#                 }
#     metadata.update(additional_metadata)
#
#     # Add references
#     references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for
#                   qubit_id_ in qubit_id}
#     references['channel_amplitudes'] = channel_amplitudes.id,
#     references['readout_pulse'] = readout_pulse.id
#     references['readout_pulse'] = readout_pulse.id
#     for pre_pulse_id, pre_pulse in enumerate(pre_pulses):
#         if hasattr(pre_pulse, 'id'):
#             references.update({('pre_pulse', str(pre_pulse_id)): pre_pulse.id})
#
#     # Add fitter arguments
#     if len(qubit_id) > 1:
#         arg_id = -2  # the last parameter is resultnumbers, so the time-like argument is -2
#     else:
#         arg_id = -1
#     fitter_arguments = (
#     measurement_name, exp_sin_fitter(mode=exp_sin_fitter_mode), arg_id, np.arange(len(extra_sweep_args)))
#
#     setter = ParameterSetter()
#     measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
#                                                             *extra_sweep_args,
#                                                             (lengths, setter.set_length, 'Excitation length', 's'),
#                                                             fitter_arguments=fitter_arguments,
#                                                             measurement_type=measurement_type,
#                                                             metadata=metadata,
#                                                             references=references,
#                                                             on_update_divider=5, shuffle=False, comment=comment)
#
#     return measurement
#
#
#
#


# from qsweepy.fitters.exp_sin import exp_sin_fitter
# from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
# import numpy as np
# from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
# from qsweepy.qubit_calibrations import sequence_control
# from qsweepy import zi_scripts
# import traceback
# import textwrap
# import time
#
#
# def Rabi_Mark(device, qubit_id, channel_amplitudes, transition='01',
#               lengths=None, *extra_sweep_args, circuits=None, basis='Z', tail_length=0,
#               readout_delay=0, pre_pulses=tuple(), repeats=1, measurement_type='Rabi_Mark', samples=False,
#               additional_metadata={}, comment='', shots=False,
#               dot_products=False, post_selection_flag=False, qutrit_readout=False):
#     """
#     Rect rect sweep measurement via a command table
#     """
#     from qsweepy.qubit_calibrations.readout_pulse2 import get_uncalibrated_measurer
#     from qsweepy.qubit_calibrations.calibrated_readout2 import get_calibrated_measurer
#
#     if post_selection_flag:
#         readouts_per_repetition = 2
#     else:
#         readouts_per_repetition = 1
#
#     if type(qubit_id) is not list and type(
#             qubit_id) is not tuple:  # if we are working with a single qubit, use uncalibrated measurer
#         readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition=transition,
#                                                             samples=samples,
#                                                             shots=shots, dot_products=dot_products,
#                                                             readouts_per_repetition=readouts_per_repetition,
#                                                             qutrit_readout=qutrit_readout)
#         measurement_name = 'iq' + qubit_id
#         if qutrit_readout:
#             measurement_name = 'resultnumbers_states'
#
#         qubit_id = [qubit_id]
#         exp_sin_fitter_mode = 'unsync'
#         excitation_channel = \
#             [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]
#     else:  # otherwise use calibrated measurer
#         readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)
#         measurement_name = 'resultnumbers'
#         exp_sin_fitter_mode = 'unsync'
#         excitation_channel = \
#             [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]
#
#     ex_channel = device.awg_channels[excitation_channel]
#     if ex_channel.is_iq():
#         control_qubit_awg = ex_channel.parent.awg
#         control_qubit_seq_id = ex_channel.parent.sequencer_id
#     else:
#         control_qubit_awg = ex_channel.parent.awg
#         control_qubit_seq_id = ex_channel.channel // 2
#     control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
#     ex_sequencers = []
#
#     # define excitation pre pulses seqs
#     ex_pre_pulses_seq = []
#     for ex_pre_pulse in pre_pulses:
#         if ex_pre_pulse.metadata['pulse_type'] == 'rect':
#             ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0)
#         else:
#             # in case of gauss pulse we use two pulses with pi/2 rotation
#             ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0) + ex_pre_pulse.get_pulse_sequence(0)
#         ex_pre_pulses_seq.append(ex_pre_pulse_seq)
#
#     for awg, seq_id in device.pre_pulses.seq_in_use:
#         if [awg, seq_id] != [control_awg, control_seq_id]:
#             ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
#                                                awg_amp=1, use_modulation=True, pre_pulses=[],
#                                                post_selection_flag=post_selection_flag)
#         else:
#             ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
#                                                awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
#                                                post_selection_flag=post_selection_flag)
#             control_sequence = ex_seq
#         if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
#             control_qubit_sequence = ex_seq
#         device.pre_pulses.set_seq_offsets(ex_seq)
#         device.pre_pulses.set_seq_prepulses(ex_seq)
#         ex_seq.start(holder=1)
#         ex_sequencers.append(ex_seq)
#
#     # TODO: how to make excitation pre pulses setting better
#     if ex_pre_pulses_seq:
#         for _id, ex_sequence in enumerate(ex_sequencers):
#             ex_sequence.awg.stop_seq(ex_sequence.params['sequencer_id'])
#             ex_sequence.clear_pulse_sequence()
#             for prep_seq in ex_pre_pulse_seq:
#                 for seq_id, single_sequence in prep_seq[0][ex_sequence.awg.device_id].items():
#                     if seq_id == ex_sequence.params['sequencer_id']:
#                         ex_sequence.add_exc_pre_pulse(single_sequence[0], single_sequence[1])
#             ex_sequence.awg.set_sequence(ex_sequence.params['sequencer_id'], ex_sequence)
#
#     readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
#     readout_sequencer.start()
#
#     # redefine lengths as integer val
#     times = np.zeros(len(lengths))
#     for _i in range(len(lengths)):
#         times[_i] = int(round(lengths[_i] * control_sequence.clock))
#     lengths = times / control_sequence.clock
#
#     class ParameterSetter:
#         def __init__(self):
#             self.device = device
#             self.ex_sequencers = ex_sequencers
#             self.readout_sequencer = readout_sequencer
#             self.control_sequence = control_sequence
#             self.control_qubit_sequence = control_qubit_sequence
#             self.length_tail = tail_length
#             self.instructions = []
#
#             self.l = int(len(circuits) / 3)
#             self.phis1 = circuits[0:self.l]
#             self.phis2 = circuits[self.l:self.l * 2]
#             self.phis3 = circuits[self.l * 2:]
#
#             self.lengths = lengths
#             fast_control = True
#
#             self.prepare_seq = []
#
#             self.rx = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id[0],
#                                                             rotation_angle=np.pi / 2, sort='newest', gauss=True)
#
#             self.prepare_seq.append(self.rx.get_pulse_sequence(0)[0])
#
#             channel_pulses = [(c, device.pg.rect_cos, complex(a), self.length_tail, fast_control) for c, a in
#                               channel_amplitudes.metadata.items()]
#
#             self.prepare_seq.append(device.pg.pmulti(device, self.lengths, *tuple(channel_pulses)))
#
#         def set_length(self, length):
#             """
#             Set length into reg0 and reg1 values and create hdawg program
#             """
#             if length == self.lengths[0]:
#                 # for the first time you need to create hdawg generator
#                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
#                 prepare_seq = self.create_hdawg_generator()
#                 sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
#                                                           instructions=self.instructions)
#                 time.sleep(0.1)
#                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
#
#             for seq in ex_sequencers:
#                 seq.set_length(length)
#
#         def phi1_setter(self, phi1):
#             index = np.where(self.phis1 == phi1)[0][0]
#             if phi1 == self.phis1[0]:
#                 # for the first time you need to create hdawg generator
#                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
#                 prepare_seq = self.create_hdawg_generator()
#                 sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
#                                                           instructions=self.instructions)
#                 time.sleep(0.1)
#                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
#
#             for seq in ex_sequencers:
#                 seq.set_phase_index(index)
#
#         def phi2_setter(self, phi2):
#             index = np.where(self.phis2 == phi2)[0][0]
#             if phi2 == self.phis2[0]:
#                 # for the first time you need to create hdawg generator
#                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
#                 prepare_seq = self.create_hdawg_generator()
#                 sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
#                                                           instructions=self.instructions)
#                 time.sleep(0.1)
#                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
#
#             for seq in ex_sequencers:
#                 seq.set_phase_index_(index)
#
#         def phi3_setter(self, phi3):
#             index = np.where(self.phis3 == phi3)[0][0]
#             if phi3 == self.phis3[0]:
#                 # for the first time you need to create hdawg generator
#                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
#                 prepare_seq = self.create_hdawg_generator()
#                 sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
#                                                           instructions=self.instructions)
#                 time.sleep(0.1)
#                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
#
#             for seq in ex_sequencers:
#                 seq.set_phase_index__(index)
#
#         def main_phi_setter(self, circuit_index):
#             l = int(len(circuits) / 3)
#             self.phi1_setter(circuits[circuit_index])
#             self.phi2_setter(circuits[circuit_index + l])
#             self.phi3_setter(circuits[circuit_index + 2 * l])
#
#
#         def return_hdawg_program(self, ex_seq):
#             """
#             Return hdawg program for defined ex_seq object
#             :param ex_seq: excitation sequencer object
#             """
#             definition_part = ''''''
#
#             # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#             #                  'header': {'version': '0.2'},
#             #                  'table': []}
#             # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#             #                  'header': {'version': '1.0'},
#             #                  'table': []}
#             command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
#                              'header': {'version': '1.2'},
#                              'table': []}
#             #
#             # command_table = {'$schema': 'https://json-schema.org/draft-04/schema#',
#             #                  'header': {'version': '1.2.0'},
#             #                  'table': []}
#
#             assign_waveform_indexes = {}
#             random_command_id = 4
#             waveform_id = -1
#
#             # get information about I and Q phases from calibrations in case of iq excitation channel
#             phase0 = ex_seq.phaseI
#             phase1 = ex_seq.phaseQ
#
#             # cycle for all prepare seqs
#             for prep_seq in self.prepare_seq:
#                 # cycle for all used hdawg
#                 for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
#                     if seq_id == ex_seq.params['sequencer_id']:
#
#                         # add waveform definition part
#                         if single_sequence[0] not in definition_part:
#                             definition_part += single_sequence[0]
#
#                         assign_fragments = single_sequence[3].split(';')[:-1]
#
#                         for id_, entry_table_index_constant in enumerate(single_sequence[2]):
#                             table_entry = {'index': random_command_id}  # define command table id index
#                             random_command_id += 1
#                             if entry_table_index_constant not in assign_waveform_indexes.keys():
#                                 waveform_id += 1
#                                 definition_part += '\nconst ' + entry_table_index_constant + '={_id};'.format(
#                                     _id=waveform_id)
#                                 assign_waveform_indexes[entry_table_index_constant] = waveform_id
#                                 # definition_part += single_sequence[3][id_]
#                                 definition_part += assign_fragments[id_] + ';'
#                                 table_entry['waveform'] = {'index': waveform_id}
#                             else:
#                                 table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}
#
#                             random_pulse = single_sequence[4]
#                             if 'phase0' in random_pulse:
#                                 table_entry['phase0'] = random_pulse['phase0']
#                             if 'phase1' in random_pulse:
#                                 table_entry['phase1'] = random_pulse['phase1']
#                             command_table['table'].append(table_entry)
#
#             # Add two auxiliary table entries, one for phase calibrations and one for reference phase
#             table_entry = {'index': 0}
#             table_entry['phase0'] = {'value': phase0, 'increment': False}
#             table_entry['phase1'] = {'value': phase1, 'increment': False}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 1}
#             table_entry['phase0'] = {'value': 0, 'increment': True}
#             table_entry['phase1'] = {'value': 0, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 2}
#             table_entry['phase0'] = {'value': 90, 'increment': True}
#             table_entry['phase1'] = {'value': 90, 'increment': True}
#             command_table['table'].append(table_entry)
#
#             table_entry = {'index': 3}
#             table_entry['phase0'] = {'value': 180, 'increment': True}
#             table_entry['phase1'] = {'value': 180, 'increment': True}
#             command_table['table'].append(table_entry)
#
#
#             for i in range(self.l):
#                 table_entry = {'index': random_command_id}
#                 table_entry['phase0'] = {'value': circuits[i], 'increment': True}
#                 table_entry['phase1'] = {'value': circuits[i], 'increment': True}
#                 command_table['table'].append(table_entry)
#                 random_command_id += 1
#
#                 table_entry = {'index': random_command_id}
#                 table_entry['phase0'] = {'value': circuits[i + self.l], 'increment': True}
#                 table_entry['phase1'] = {'value': circuits[i + self.l], 'increment': True}
#                 command_table['table'].append(table_entry)
#                 random_command_id += 1
#
#                 table_entry = {'index': random_command_id}
#                 table_entry['phase0'] = {'value': circuits[i + 2 * self.l], 'increment': True}
#                 table_entry['phase1'] = {'value': circuits[i + 2 * self.l], 'increment': True}
#                 command_table['table'].append(table_entry)
#                 random_command_id += 1
#
#
#
#             play_part = textwrap.dedent('''
# // Table entry play part
#     executeTableEntry(0);
#     executeTableEntry(1);
#
#     // Generation of random initial point
#     executeTableEntry(15+3*variable_register2);
#     executeTableEntry(4);
#     executeTableEntry(16+3*variable_register14);
#     executeTableEntry(4);
#     executeTableEntry(17+3*variable_register13);
#
#     // Rabi sequence
#     executeTableEntry(variable_register1+5);
#     wait(variable_register0);
#     executeTableEntry(14);
#
#
#     ''')
#             if basis == 'Z':
#                 play_part += textwrap.dedent('''
#     // Z-basis
#     resetOscPhase();
#                 ''')
#
#             if basis == 'X':
#                 play_part += textwrap.dedent('''
#     // X-basis
#     executeTableEntry(2);
#     executeTableEntry(4);
#     resetOscPhase();
#                 ''')
#
#             if basis == 'Y':
#                 play_part += textwrap.dedent('''
#     // Y-basis
#     executeTableEntry(4);
#     resetOscPhase();
#                 ''')
#
#             self.instructions.append(command_table)
#             print('Command table for sequencer id {}'.format(ex_seq.params['sequencer_id']), command_table)
#             return definition_part, play_part
#
#         def create_hdawg_generator(self):
#             pulses = {}
#             control_seq_ids = []
#             control_awg_ids = []
#             self.instructions = []
#
#             for device_id in self.prepare_seq[0][0].keys():
#                 pulses.update({device_id: {}})
#
#             for ex_seq in self.ex_sequencers:
#                 pulses[ex_seq.awg.device_id][ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
#                 control_seq_ids.append(ex_seq.params['sequencer_id'])
#                 control_awg_ids.append(ex_seq.awg.device_id)
#
#             return [[pulses, control_seq_ids, control_awg_ids]]
#
#     # Add metadata
#     metadata = {'qubit_id': ','.join(qubit_id),
#                 'transition': transition,
#                 'extra_sweep_args': str(len(extra_sweep_args)),
#                 'tail_length': str(tail_length),
#                 'readout_delay': str(readout_delay),
#                 'repeats': str(repeats),
#                 'amplitude': channel_amplitudes.metadata[excitation_channel],
#                 'command_table': str(True),
#                 'circuits': str(circuits),
#                 'basis': basis
#                 }
#     metadata.update(additional_metadata)
#
#     # Add references
#     references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for
#                   qubit_id_ in qubit_id}
#     references['channel_amplitudes'] = channel_amplitudes.id,
#     references['readout_pulse'] = readout_pulse.id
#     references['readout_pulse'] = readout_pulse.id
#     for pre_pulse_id, pre_pulse in enumerate(pre_pulses):
#         if hasattr(pre_pulse, 'id'):
#             references.update({('pre_pulse', str(pre_pulse_id)): pre_pulse.id})
#
#
#     fitter_arguments = ('iq' + qubit_id[0], exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))
#
#     setter = ParameterSetter()
#     measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
#                                                             *extra_sweep_args,
#                                                             (np.arange(int(len(circuits)/3)), setter.main_phi_setter, 'circuit', 'index'),
#                                                             (lengths, setter.set_length, 'Excitation length', 's'),
#                                                             fitter_arguments=fitter_arguments,
#                                                             measurement_type=measurement_type,
#                                                             metadata=metadata,
#                                                             references=references,
#                                                             on_update_divider=5, shuffle=False, comment=comment)
#
#     return measurement



from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.qubit_calibrations import channel_amplitudes
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import traceback
import textwrap
import time


def Rabi_Mark_gate(device, qubit_id, channel_amplitudes_, coupler_name, gate_amplitude, transition='01',
              lengths=None, *extra_sweep_args, circuits=None, basis='Z', tail_length=0,
              readout_delay=0, pre_pulses=tuple(), repeats=1, measurement_type='Rabi_Mark', samples=False,
              additional_metadata={}, comment='', shots=False,
              dot_products=False, post_selection_flag=False, qutrit_readout=False):
    """
    Rect rect sweep measurement via a command table
    """
    from qsweepy.qubit_calibrations.readout_pulse2 import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations.calibrated_readout2 import get_calibrated_measurer

    if post_selection_flag:
        readouts_per_repetition = 2
    else:
        readouts_per_repetition = 1

    if type(qubit_id) is not list and type(
            qubit_id) is not tuple:  # if we are working with a single qubit, use uncalibrated measurer
        readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition=transition,
                                                            samples=samples,
                                                            shots=shots, dot_products=dot_products,
                                                            readouts_per_repetition=readouts_per_repetition,
                                                            qutrit_readout=qutrit_readout)
        measurement_name = 'iq' + qubit_id
        if qutrit_readout:
            measurement_name = 'resultnumbers_states'

        qubit_id = [qubit_id]
        exp_sin_fitter_mode = 'unsync'
        excitation_channel = \
            [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]
    else:  # otherwise use calibrated measurer
        readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)
        measurement_name = 'resultnumbers'
        exp_sin_fitter_mode = 'unsync'
        excitation_channel = \
            [i for i in device.get_qubit_excitation_channel_list(qubit_id[0], transition=transition).keys()][0]

    ex_channel = device.awg_channels[excitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    # define excitation pre pulses seqs
    ex_pre_pulses_seq = []
    for ex_pre_pulse in pre_pulses:
        if ex_pre_pulse.metadata['pulse_type'] == 'rect':
            ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0)
        else:
            # in case of gauss pulse we use two pulses with pi/2 rotation
            ex_pre_pulse_seq = ex_pre_pulse.get_pulse_sequence(0) + ex_pre_pulse.get_pulse_sequence(0)
        ex_pre_pulses_seq.append(ex_pre_pulse_seq)

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
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)

    # TODO: how to make excitation pre pulses setting better
    if ex_pre_pulses_seq:
        for _id, ex_sequence in enumerate(ex_sequencers):
            ex_sequence.awg.stop_seq(ex_sequence.params['sequencer_id'])
            ex_sequence.clear_pulse_sequence()
            for prep_seq in ex_pre_pulse_seq:
                for seq_id, single_sequence in prep_seq[0][ex_sequence.awg.device_id].items():
                    if seq_id == ex_sequence.params['sequencer_id']:
                        ex_sequence.add_exc_pre_pulse(single_sequence[0], single_sequence[1])
            ex_sequence.awg.set_sequence(ex_sequence.params['sequencer_id'], ex_sequence)

    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    # redefine lengths as integer val
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
            self.length_tail = tail_length
            self.instructions = []

            self.l = int(len(circuits) / 3)
            self.phis1 = circuits[0:self.l]
            self.phis2 = circuits[self.l:self.l * 2]
            self.phis3 = circuits[self.l * 2:]

            self.lengths = lengths
            fast_control = True

            self.prepare_seq = []

            self.rx = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id[0],
                                                            rotation_angle=np.pi / 2, sort='newest', gauss=True)

            self.prepare_seq.append(self.rx.get_pulse_sequence(0)[0])


            qubit_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_,
                                                                       tail_length=self.length_tail,
                                                                       length=self.lengths,
                                                                       phase=0.0,
                                                                       fast_control=True)

            coupler_channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                        **{coupler_name: complex(gate_amplitude)})

            coupler_sequence = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=coupler_channel_amplitudes_,
                                                                       tail_length=self.length_tail,
                                                                       length=self.lengths,
                                                                       phase=0.0,
                                                                       fast_control=True)

            self.prepare_seq.append(device.pg.parallel(qubit_sequence[0], coupler_sequence[0]))

        def set_length(self, length):
            """
            Set length into reg0 and reg1 values and create hdawg program
            """
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

        def phi1_setter(self, phi1):
            index = np.where(self.phis1 == phi1)[0][0]
            if phi1 == self.phis1[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index(index)

        def phi2_setter(self, phi2):
            index = np.where(self.phis2 == phi2)[0][0]
            if phi2 == self.phis2[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index_(index)

        def phi3_setter(self, phi3):
            index = np.where(self.phis3 == phi3)[0][0]
            if phi3 == self.phis3[0]:
                # for the first time you need to create hdawg generator
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                prepare_seq = self.create_hdawg_generator()
                sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                          instructions=self.instructions)
                time.sleep(0.1)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

            for seq in ex_sequencers:
                seq.set_phase_index__(index)

        def main_phi_setter(self, circuit_index):
            l = int(len(circuits) / 3)
            self.phi1_setter(circuits[circuit_index])
            self.phi2_setter(circuits[circuit_index + l])
            self.phi3_setter(circuits[circuit_index + 2 * l])


        def return_hdawg_program(self, ex_seq):
            """
            Return hdawg program for defined ex_seq object
            :param ex_seq: excitation sequencer object
            """
            definition_part = ''''''

            # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
            #                  'header': {'version': '0.2'},
            #                  'table': []}
            # command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
            #                  'header': {'version': '1.0'},
            #                  'table': []}
            command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                             'header': {'version': '1.2'},
                             'table': []}
            #
            # command_table = {'$schema': 'https://json-schema.org/draft-04/schema#',
            #                  'header': {'version': '1.2.0'},
            #                  'table': []}

            assign_waveform_indexes = {}
            random_command_id = 4
            waveform_id = -1

            # get information about I and Q phases from calibrations in case of iq excitation channel
            phase0 = ex_seq.phaseI
            phase1 = ex_seq.phaseQ

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
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = single_sequence[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)

            # Add two auxiliary table entries, one for phase calibrations and one for reference phase
            table_entry = {'index': 0}
            table_entry['phase0'] = {'value': phase0, 'increment': False}
            table_entry['phase1'] = {'value': phase1, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': 1}
            table_entry['phase0'] = {'value': 0, 'increment': True}
            table_entry['phase1'] = {'value': 0, 'increment': True}
            command_table['table'].append(table_entry)

            table_entry = {'index': 2}
            table_entry['phase0'] = {'value': 90, 'increment': True}
            table_entry['phase1'] = {'value': 90, 'increment': True}
            command_table['table'].append(table_entry)

            table_entry = {'index': 3}
            table_entry['phase0'] = {'value': 180, 'increment': True}
            table_entry['phase1'] = {'value': 180, 'increment': True}
            command_table['table'].append(table_entry)


            for i in range(self.l):
                table_entry = {'index': random_command_id}
                table_entry['phase0'] = {'value': circuits[i], 'increment': True}
                table_entry['phase1'] = {'value': circuits[i], 'increment': True}
                command_table['table'].append(table_entry)
                random_command_id += 1

                table_entry = {'index': random_command_id}
                table_entry['phase0'] = {'value': circuits[i + self.l], 'increment': True}
                table_entry['phase1'] = {'value': circuits[i + self.l], 'increment': True}
                command_table['table'].append(table_entry)
                random_command_id += 1

                table_entry = {'index': random_command_id}
                table_entry['phase0'] = {'value': circuits[i + 2 * self.l], 'increment': True}
                table_entry['phase1'] = {'value': circuits[i + 2 * self.l], 'increment': True}
                command_table['table'].append(table_entry)
                random_command_id += 1



            play_part = textwrap.dedent('''
// Table entry play part
    executeTableEntry(0);
    executeTableEntry(1);

    // Generation of random initial point
    executeTableEntry(15+3*variable_register2);
    executeTableEntry(4);
    executeTableEntry(16+3*variable_register14);
    executeTableEntry(4);
    executeTableEntry(17+3*variable_register13);

    // Rabi sequence
    executeTableEntry(variable_register1+5);
    wait(variable_register0);
    executeTableEntry(14);


    ''')
            if basis == 'Z':
                play_part += textwrap.dedent('''
    // Z-basis
    resetOscPhase();
                ''')

            if basis == 'X':
                play_part += textwrap.dedent('''
    // X-basis
    executeTableEntry(2);
    executeTableEntry(4);
    resetOscPhase();
                ''')

            if basis == 'Y':
                play_part += textwrap.dedent('''
    // Y-basis
    executeTableEntry(4);
    resetOscPhase();
                ''')

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

    # Add metadata
    metadata = {'qubit_id': ','.join(qubit_id),
                'transition': transition,
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'qubit amplitude': channel_amplitudes_.metadata[excitation_channel],
                'command_table': str(True),
                'circuits': str(circuits),
                'basis': basis,
                'coupler name':coupler_name,
                'coupler gate amplitude':gate_amplitude
                }
    metadata.update(additional_metadata)

    # Add references
    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for
                  qubit_id_ in qubit_id}
    references['channel_amplitudes'] = channel_amplitudes_.id,
    references['readout_pulse'] = readout_pulse.id
    references['readout_pulse'] = readout_pulse.id
    for pre_pulse_id, pre_pulse in enumerate(pre_pulses):
        if hasattr(pre_pulse, 'id'):
            references.update({('pre_pulse', str(pre_pulse_id)): pre_pulse.id})


    fitter_arguments = ('iq' + qubit_id[0], exp_sin_fitter(), -1, np.arange(len(extra_sweep_args) + 1))

    setter = ParameterSetter()
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (np.arange(int(len(circuits)/3)), setter.main_phi_setter, 'circuit', 'index'),
                                                            (lengths, setter.set_length, 'Excitation length', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5, shuffle=False, comment=comment)

    return measurement



