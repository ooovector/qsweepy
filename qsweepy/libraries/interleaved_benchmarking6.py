from qsweepy.libraries import sweep
import numpy as np
import textwrap
from qsweepy.qubit_calibrations import sequence_control
import json
import copy
import matplotlib.pyplot as plt
import time



class HDAWG_PRNG:
    def __init__(self, seed=0xcafe, lower=0, upper=2 ** 16 - 1):
        self.lsfr = seed
        self.lower = lower
        self.upper = upper

    def next(self):
        lsb = self.lsfr & 1
        self.lsfr = self.lsfr >> 1
        if (lsb):
            self.lsfr = 0xb400 ^ self.lsfr
        rand = ((self.lsfr * (self.upper - self.lower + 1) >> 16) + self.lower) & 0xffff
        return rand


class interleaved_benchmarking:
    # def __init__(self, measurer, set_seq, interleavers=None, random_sequence_num=8):
    def __init__(self, device, measurer, ex_sequencers, seeds, seq_lengths, interleavers=None, random_sequence_num=8,
                 two_qubit_num=1, random_gate_num=1, readout_sequencer=None, two_qubit = None):

        self.seeds = seeds
        self.seq_lengths = np.asarray(seq_lengths)
        assert seeds.shape[2] == random_sequence_num, 'seeds.shape[2] != random_sequence_num'
        assert seeds.shape[1] == len(ex_sequencers), 'seeds.shape[1] != len(ex_sequencers)'
        assert seeds.shape[0] == len(self.seq_lengths), 'seeds.shape[0] != len(self.seq_lengths)'

        self.seed_register = 1
        self.sequence_length_register = 0
        self.two_qubit_gate_register = 2
        self.random_gate_register = 3
        self.readout_sequencer = readout_sequencer

        self.device = device
        self.measurer = measurer
        self.ex_sequencers = ex_sequencers
        self.two_qubit_num = two_qubit_num
        self.two_qubit = two_qubit
        self.random_gate_num = random_gate_num
        self.interleavers = {}
        self.instructions = []

        if interleavers is not None:
            for name, gate in interleavers.items():
                self.add_interleaver(name, gate['pulses'], gate['unitary'])

        # Set registers for two_qubit_num - number of two qubit gates in sequence
        # Set registers for random_gate_num - number of random gate in sequence
        # Sequence = (-(random_gate)^random_gate_num - (two_qubit_gate)^two_qubit_num)^sequence_length
        # sequence_length is the sweep parameter
        # random_sequence_num is the number of sequnces for each sequence length
        for seq in self.ex_sequencers:
            seq.awg.set_register(seq.params['sequencer_id'], self.two_qubit_gate_register, self.two_qubit_num)
            seq.awg.set_register(seq.params['sequencer_id'], self.random_gate_register, self.random_gate_num)

        # self.initial_state_vector = np.asarray([1, 0]).T

        self.random_sequence_num = random_sequence_num
        self.sequence_length = self.seq_lengths[0]

        self.target_gate = []
        # self.target_gate_unitary = np.asarray([[1,0],[0,1]], dtype=np.complex)
        self.target_gate_name = 'Identity (benchmarking)'

        self.reference_benchmark_result = None
        self.interleaving_sequences = None

        self.final_ground_state_rotation = True
        self.prepare_random_sequence_before_measure = True
        self.seq_id = 0

    def command_table_entry_creation(self, part, definition_part, random_command_id, assign_waveform_indexes, waveform_id, phase0=0, phase1=0):
        table_entry = {'index': random_command_id}
        random_command_id += 1

        entry_table_index_constant = part[2][0]

        if entry_table_index_constant not in assign_waveform_indexes.keys():
            waveform_id += 1
            assign_waveform_indexes[entry_table_index_constant] = waveform_id
            if part[0] not in definition_part:
                definition_part += part[0]
            definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(_id=waveform_id)
            definition_part += part[3]
            table_entry['waveform'] = {'index': waveform_id}
        else:
            table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

        random_pulse = part[4]
        # if 'amplitude0' in random_pulse:
        # table_entry['amplitude0'] = random_pulse['amplitude0']
        # VOT phase bleat
        table_entry['phase0'] = {'value': random_pulse['phase0']['value']+phase0, 'increment': random_pulse['phase0']['increment']}
        table_entry['phase1'] = {'value': random_pulse['phase1']['value']+phase1, 'increment': random_pulse['phase1']['increment']}
        #raise ValueError('FallowError')
        # if 'amplitude1' in random_pulse:
        # table_entry['amplitude1'] = random_pulse['amplitude1']
        return table_entry, definition_part, assign_waveform_indexes, random_command_id, waveform_id


    def return_hdawg_program(self, ex_seq, seq_len=0):
        random_gate_num = len(self.interleavers)

        if self.two_qubit is not None:
            definition_part = textwrap.dedent('''
const random_gate_num = {random_gate_num};
setPRNGRange(0, random_gate_num-1);
const sequence_len = {sequence_len};'''.format(random_gate_num=random_gate_num-1, sequence_len=seq_len))
        else:
            definition_part = textwrap.dedent('''
const random_gate_num = {random_gate_num};
setPRNGRange(0, random_gate_num-1);
const sequence_len = {sequence_len};'''.format(random_gate_num=random_gate_num, sequence_len=seq_len))

        command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                         "header": { "version": "0.2" },
                         "table": [] }
        assign_waveform_indexes = {}
        random_command_id = 0
        waveform_id = -1
        # First gate

        # for _i in range(12):
        #     phase0 = 0
        #     phase0 = (((_i + 2) % 4) - 2) * 90
        #     if _i in [0,1,2,3,4,6]:
        #         gate = self.interleavers['X/2']
        #     elif _i in [5,7,9,10,11]:
        #         gate = self.interleavers['-X/2']
        #     else:
        #         gate = self.interleavers['I']
        #
        #     for j in range(len(gate['pulses'])):
        #         for seq_id, part in gate['pulses'][j][0].items():
        #             if seq_id == ex_seq.params['sequencer_id']:
        #                 table_entry, definition_part, assign_waveform_indexes, random_command_id, waveform_id = self.command_table_entry_creation(part,
        #                                             definition_part, random_command_id, assign_waveform_indexes, waveform_id, phase0=phase0, phase1=0)
        #                 if _i in [0,1,2,3,4,6]:
        #                     table_entry['waveform'] = {'index': 0}
        #                 elif _i in [5,7,9,10,11]:
        #                     table_entry['waveform'] = {'index': 1}
        #                 else:
        #                     table_entry['waveform'] = {'index': 2}
        #                 command_table['table'].append(table_entry)

        for _i in range(6):
            phase0 = 0
            phase0 = ((_i % 4) ) * 90
            if _i==3:
                phase0=-90
            if _i in [0, 1]:
                gate = self.interleavers['X/2']
            elif _i in [2, 3]:
                gate = self.interleavers['-X/2']
            else:
                gate = self.interleavers['I']

            for j in range(len(gate['pulses'])):
                for seq_id, part in gate['pulses'][j][0].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        table_entry, definition_part, assign_waveform_indexes, random_command_id, waveform_id = self.command_table_entry_creation(part,
                                                    definition_part, random_command_id, assign_waveform_indexes, waveform_id, phase0=phase0, phase1=0)
                        # if _i in [0, 1]:
                        #     table_entry['waveform'] = {'index': 0}
                        # elif _i in [2, 3]:
                        #     table_entry['waveform'] = {'index': 1}
                        # else:
                        #     table_entry['waveform'] = {'index': 2}
                        command_table['table'].append(table_entry)

        if self.two_qubit is not None:
            gate = self.two_qubit['fSIM']
            for j in range(len(gate['pulses'])):
                for seq_id, part in gate['pulses'][j][0].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        table_entry, definition_part, assign_waveform_indexes, random_command_id, waveform_id = self.command_table_entry_creation(part,
                                                    definition_part, random_command_id,assign_waveform_indexes, waveform_id, phase0=0, phase1=0)
                        command_table['table'].append(table_entry)
            two_qubit_gate_index = random_command_id-1


        table_entry = {'index': random_command_id}
        #table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': 0.0, 'increment': False}
        #table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': 90.0, 'increment': False}
        command_table['table'].append(table_entry)
        random_command_id += 1
        table_entry = {'index': random_command_id}
        # table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': 0.0, 'increment': True}
        # table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': 90.0, 'increment': False}
        command_table['table'].append(table_entry)
        if self.two_qubit is not None:
            play_part = textwrap.dedent('''
// Clifford play part
//setPRNGSeed(variable_register1);
    //resetOscPhase();
    executeTableEntry({random_gate_num});
    wait(5);
    repeat (variable_register0) {{
        repeat(5){{
            var rand_value1 = getPRNGValue();
            executeTableEntry({random_gate_num}+1);
            executeTableEntry(rand_value1);
        }}  
        repeat ({two_qubit_num}){{
            executeTableEntry({random_gate_num}+1);
            executeTableEntry({two_qubit_gate_index});
        }}
    }} 
    executeTableEntry({random_gate_num});
    resetOscPhase();
    '''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num = random_command_id-1,
                  two_qubit_num=self.two_qubit_num))
        else:
            play_part = textwrap.dedent('''
// Clifford play part
//setPRNGSeed(variable_register1);
    executeTableEntry({random_gate_num});
    //resetOscPhase();
    wait(5);
    //executeTableEntry({random_gate_num}+1);
    repeat (variable_register0) {{
        repeat(5){{
            var rand_value1 = getPRNGValue();
            executeTableEntry({random_gate_num}+1);
            executeTableEntry(rand_value1); 
        }}
    }} 
    executeTableEntry({random_gate_num});
    resetOscPhase();
    '''.format(random_gate_num=random_command_id-1))
        self.instructions.append(command_table)
        print(command_table)
        return definition_part, play_part

    def create_hdawg_generator(self):
        pulses = {}
        control_seq_ids = []
        self.instructions=[]
        for ex_seq in self.ex_sequencers:
            pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq, self.sequence_length)
            control_seq_ids.append(ex_seq.params['sequencer_id'])

        return [[pulses, control_seq_ids]]

    # for _i in group

    # used to transformed any of the |0>, |1>, |+>, |->, |i+>, |i-> states into the |0> state
    # low-budget function only appropriate for clifford benchmarking
    # higher budget functions require arbitrary rotation pulse generator
    # which we unfortunately don't have (yet)
    # maybe Chernogolovka control experiment will do this
    def state_to_zero_transformer(self, psi):
        good_interleavers_length = {}
        # try each of the interleavers available
        for name, interleaver in self.interleavers.items():
            result = np.dot(interleaver['unitary'], psi)
            if (1 - np.abs(np.dot(result, self.initial_state_vector))) < 1e-6:
                # if the gate does what we want than we append it to our pulse list
                good_interleavers_length[name] = len(interleaver['pulses'])
        # check our pulse list for the shortest pulse
        # return the name of the best interleaver
        # the whole 'pulse name' logic is stupid
        # if gates are better parameterized by numbers rather than names (which is true for non-Cliffords)

        name = min(good_interleavers_length, key=good_interleavers_length.get)
        return name, self.interleavers[name]

    def set_sequence_length(self, sequence_length):
        # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        self.sequence_length = sequence_length
        for i, ex_seq in enumerate(self.ex_sequencers):
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg0'], self.sequence_length)
        # #if sequence_length
        # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        # #if self.sequence_length==self.seq_lengths[0]:
        # prepare_seq = self.create_hdawg_generator()
        # #raise ValueError('fallos')
        # sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq, instructions=self.instructions)
        # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])


    def set_sequence_length_and_regenerate(self, sequence_length):
        print('Bull happens')
        self.sequence_length = sequence_length
        self.set_interleaved_sequence()
        self.prepare_random_interleaving_sequences()

    def set_target_pulse(self, x):
        self.target_gate = x['pulses']
        self.target_gate_unitary = x['unitary']

    def prepare_random_interleaving_sequences(self):
        self.interleaving_sequences = [self.generate_random_interleaver_sequence(self.sequence_length) for i in
                                       range(self.random_sequence_num)]

    def add_interleaver(self, name, pulse_seq, unitary):
        self.d = unitary.shape[0]
        self.initial_state_vector = np.zeros(self.d)
        self.initial_state_vector[0] = 1.
        self.target_gate_unitary = np.identity(self.d, dtype=np.complex)
        self.interleavers[name] = {'pulses': pulse_seq, 'unitary': unitary}

    def generate_interleaver_sequence_from_names(self, names):
        sequence_pulses = [self.interleavers[k]['pulses'] for k in names]
        sequence_unitaries = [self.interleavers[k]['unitary'] for k in names]

        psi = self.initial_state_vector.copy()
        for U in sequence_unitaries:
            psi = np.dot(U, psi)

        rho = np.einsum('i,j->ij', np.conj(psi), psi)

        return {'Gate names': names,
                'Gate unitaries': sequence_unitaries,
                'Pulse sequence': sequence_pulses,
                'Final state vector': psi,
                'Final state matrix': rho}

    def generate_random_interleaver_sequence(self, n):
        ilk = [k for k in self.interleavers.keys()]
        ilv = [self.interleavers[k] for k in ilk]
        sequence = np.random.randint(len(ilv), size=n).tolist()
        sequence_pulses = [j for i in [ilv[i]['pulses'] for i in sequence] for j in i]
        sequence_unitaries = [ilv[i]['unitary'] for i in sequence]
        sequence_gate_names = [ilk[i] for i in sequence]

        psi = self.initial_state_vector.copy()
        for U in sequence_unitaries:
            psi = np.dot(U, psi)

        rho = np.einsum('i,j->ij', np.conj(psi), psi)

        return {'Gate names': sequence_gate_names,
                'Gate unitaries': sequence_unitaries,
                'Pulse sequence': sequence_pulses,
                'Final state vector': psi,
                'Final state matrix': rho}

    # TODO: density matrix evolution operator for non-hamiltonian mechnics
    def interleave(self, sequence_gate_names, pulse, unitary, gate_name):

        sequence_unitaries = [self.interleavers[i]['unitary'] for i in sequence_gate_names]

        sequence_pulses = []
        interleaved_sequence_gate_names = []
        psi = self.initial_state_vector.copy()
        for i in sequence_gate_names:
            psi = np.dot(unitary, np.dot(self.interleavers[i]['unitary'], psi))
            sequence_pulses.extend(self.interleavers[i]['pulses'])
            sequence_pulses.extend(pulse)
            interleaved_sequence_gate_names.append(i)
            interleaved_sequence_gate_names.append(gate_name)

        if self.final_ground_state_rotation:
            final_rotation_name, final_rotation = self.state_to_zero_transformer(psi)

        # we need something separate from the interleavers for the final gate
        # the logic is in principle OK but the final gate should be a function, n
        psi = np.dot(final_rotation['unitary'], psi)
        sequence_pulses.extend(final_rotation['pulses'])
        interleaved_sequence_gate_names.append(final_rotation_name)

        rho = np.einsum('i,j->ij', np.conj(psi), psi)

        return {'Gate names': interleaved_sequence_gate_names,
                'Gate unitaries': sequence_unitaries,
                'Pulse sequence': sequence_pulses,
                'Final state vector': psi,
                'Final state matrix': rho}

    def reference_benchmark(self):
        old_target_gate = self.target_gate
        old_target_gate_unitary = self.target_gate_unitary
        old_target_gate_name = self.target_gate_name
        self.target_gate = []
        self.target_gate_unitary = np.asarray([[1, 0], [0, 1]], dtype=np.complex)
        self.target_gate_name = 'Identity (benchmarking)'

        self.reference_benchmark_result = self.measure()

        self.target_gate = old_target_gate
        self.target_gate_unitary = old_target_gate_unitary
        self.target_gate_name = old_target_gate_name

        return self.reference_benchmark_result

    def get_dtype(self):
        dtypes = self.measurer.get_dtype()
        dtypes['seed'] = int
        return dtypes

    def get_opts(self):
        opts = self.measurer.get_opts()
        opts['seed'] = {}
        return opts

    def get_points(self):
        points = (('sequencer_id', np.arange(len(self.ex_sequencers)), ''),)
        _points = {keys: values for keys, values in self.measurer.get_points().items()}
        _points['seed'] = points
        return _points

    def set_interleaved_sequence(self, seq_id):

        # seq = self.interleave(self.interleaving_sequences[seq_id]['Gate names'], self.target_gate,
        #                       self.target_gate_unitary, self.target_gate_name)
        # self.set_seq(seq['Pulse sequence'])
        # self.current_seq = seq
        # TODO set seed for PRNG
        self.seq_id = seq_id
        # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        # for i, ex_seq in enumerate(self.ex_sequencers):
        #     #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
        #     ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg0'], self.sequence_length)
        #     ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg1'],
        #                         self.seeds[np.where(self.seq_lengths==self.sequence_length)[0], i, seq_id])
        # ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg2'], self.two_qubit_num)
        # ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
        # time.sleep(0.001)

        # time.sleep(0.01)
        # for i, ex_seq in enumerate(self.ex_sequencers):
        # ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
        # ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
        # time.sleep(0.01)
        # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
    def pre_sweep(self):
        self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        prepare_seq = self.create_hdawg_generator()
        sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq, instructions=self.instructions)
        self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

    def measure(self):
        self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        for i, ex_seq in enumerate(self.ex_sequencers):
            #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg0'], self.sequence_length)
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg1'],
                                    self.seeds[np.where(self.seq_lengths == self.sequence_length)[0], i, self.seq_id])
            #ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
        time.sleep(0.1)
        self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
        measurement = self.measurer.measure()
        measurement['seed'] = self.seeds[np.where(self.seq_lengths == self.sequence_length)[0], :, self.seq_id]

        return measurement