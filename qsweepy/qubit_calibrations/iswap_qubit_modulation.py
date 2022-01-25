from qsweepy.ponyfiles.data_structures import *
import traceback
#from .import
from qsweepy.libraries import pulses2 as pulses
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import Rabi2 as Rabi
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
import textwrap


def iswap_calibration_confuse_matrix(device, qubit_ids, correspondence, amplitudes_1=[0], amplitudes_2=[0],
                                     amplitudes_c=[0], lengths=[0], number_of_circles=1):

    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    # sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_seq_id = ex_channel.parent.sequencer_id
    else:
        control_seq_id = ex_channel.channel // 2
    ex_sequencers = []

    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            print('control_sequence=', control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            self.device=device
            self.control = 3
            if len(amplitudes_c) > 1:
                self.control = 2
            elif len(amplitudes_2) > 1:
                self.control = 1
            elif len(amplitudes_1) > 1:
                self.control = 0

            self.amplitude_1 = amplitudes_1[0]
            self.amplitude_2 = amplitudes_2[0]
            self.amplitude_c = amplitudes_c[0]
            self.length = lengths[0]

            self.number_of_circles = number_of_circles

            self.interleavers = {}
            self.instructions = []
            self.qubit_ids = qubit_ids
            self.correspondence = correspondence

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

        def set_amplitude_1(self, amplitude):
            self.amplitude_1 = amplitude
            if self.control == 0:
                self.create_program()

        def set_amplitude_2(self, amplitude):
            self.amplitude_2 = amplitude
            if self.control == 1:
                self.create_program()

        def set_amplitude_c(self, phase):
            self.amplitude_c = phase
            if self.control == 2:
                self.create_program()

        def set_length(self, length):
            self.length = length
            if self.control == 3:
                self.create_program()


        def create_program(self):
            self.interleavers = self.create_interleavers()
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            prepare_seq = self.create_hdawg_generator()
            for ex_seq in self.ex_sequencers:
                for register in range(7):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, prepare_seq,
                                                      instructions=self.instructions)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def set_target_state(self, state):
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            for _id, qubit_id in enumerate(self.qubit_ids):
                state_register = 0
                qubit_state = (1 << _id) & state
                if qubit_state:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 1)
                else:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 0)
            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def return_hdawg_program(self, ex_seq):
            random_gate_num = len(self.interleavers)
            assign_waveform_indexes = {}
            definition_part = ''''''
            command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                             "header": {"version": "0.2"},
                             "table": []}

            random_command_id = 0
            waveform_id = -1
            for name, gate in self.interleavers.items():
                for j in range(len(gate['pulses'])):
                    for seq_id, part in gate['pulses'][j][0].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            # if part[0] not in definition_part:
                            # definition_part += part[0]
                            # for entry_table_index_constant in part[2]:
                            table_entry = {'index': random_command_id}
                            random_command_id += 1

                            entry_table_index_constant = part[2][0]
                            # if entry_table_index_constant not in definition_part:
                            # if entry_table_index_constant not in definition_part:
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                if part[0] not in definition_part:
                                    definition_part += part[0]
                                definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                definition_part += part[3]
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = part[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)
            two_qubit_gate_index = 2

            table_entry = {'index': random_gate_num}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': False}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': random_gate_num + 1}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': True}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            play_part = textwrap.dedent('''
//  Confuse play part
    executeTableEntry({random_gate_num});
    wait(5);

//Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1

    repeat({repeat}){{
// iSWAP gate group 
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});}}

//Post pulses - Not necessary here
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register1); // variable_register1 = 0 or 1
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register2); // variable_register1 = 0 or 1

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num,
                               repeat=self.number_of_circles))

            self.instructions.append(command_table)
            print(command_table)

            return definition_part, play_part

        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            self.instructions = []
            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])

            return [[pulses, control_seq_ids]]

        def create_interleavers(self):
            interleavers = {}
            # Preparation gates
            # Exitation
            ex1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            ex2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            preparation1 = {
                'X/2': {'pulses': [device.pg.parallel(ex1.get_pulse_sequence(0)[0], ex2.get_pulse_sequence(0)[0])]}}
            # Identical
            channel_pulses_I1 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            channel_pulses_I2 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            I1_pulse = [device.pg.pmulti(device, float(ex1.metadata['length']), *tuple(channel_pulses_I1))]
            I2_pulse = [device.pg.pmulti(device, float(ex2.metadata['length']), *tuple(channel_pulses_I2))]
            preparation0 = {'I': {'pulses': [device.pg.parallel(I1_pulse[0], I2_pulse[0])]}}

            # Two qubit gate definition

            gate_c = device.get_two_qubit_gates()['iSWAP(1,2)_CZ']
            full_length = self.length # float(gate_c.metadata['length'])
            tail_length = float(gate_c.metadata['tail_length'])
            length = full_length - 2 * tail_length
            channel_amplitudes_c = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate_c.metadata['carrier_name']: float(
                                                                             gate_c.metadata['amplitude'])})
            pulse_seq_c = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_c,
                                                                       tail_length=tail_length,
                                                                       length=length,
                                                                       phase=0.0,
                                                                       fast_control=False)

            ex_pulse1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2, preferred_length=13.3e-9)
            ex_pulse2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2, preferred_length=13.3e-9)

            channel_amplitudes_1 = channel_amplitudes.channel_amplitudes(device,
                                                                         **{ex_pulse1.metadata['carrier_name']: self.amplitude_1})
            pulse_seq_1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes_1,
                                                                      tail_length=0,
                                                                      length=self.length,
                                                                      phase=0.0)

            channel_amplitudes_2 = channel_amplitudes.channel_amplitudes(device,
                                                                         **{ex_pulse2.metadata[
                                                                                'carrier_name']: self.amplitude_2})
            pulse_seq_2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes_2,
                                                                       tail_length=0,
                                                                       length=self.length,
                                                                       phase=0.0)

            two_qubit_gate_group = {'iSWAP': {'pulses': [device.pg.parallel(pulse_seq_1[0], pulse_seq_2[0], pulse_seq_c[0])]}, }

            interleavers.update(preparation0)  # 0
            interleavers.update(preparation1)  # 1
            interleavers.update(two_qubit_gate_group)  # 6s
            return interleavers

    metadata = {'qubit_id': qubit_ids,
                'number_of_circles': number_of_circles}

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id}
    measurement_type = 'iswap_rabi_z_confuse_scan'

    measurement = device.sweeper.sweep(measurer,
                                       (amplitudes_1, setter.set_amplitude_1, 'Amplitude 1', ''),
                                       (amplitudes_2, setter.set_amplitude_2, 'Amplitude 2', ''),
                                       (amplitudes_c, setter.set_amplitude_c, 'Amplitude C', ''),
                                       (lengths, setter.set_length, 'lengths', ''),
                                       (np.arange(2 ** len(qubit_ids)), setter.set_target_state, 'Target state', ''),
                                       measurement_type=measurement_type,
                                       references=references,
                                       metadata=metadata)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement
