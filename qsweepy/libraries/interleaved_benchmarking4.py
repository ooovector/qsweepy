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

    def return_hdawg_program(self, ex_seq, seq_len=0):
        random_gate_num = len(self.interleavers)
        assign_waveform_indexes={}
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

        random_command_id = 0
        waveform_id = -1

        for name, gate in self.interleavers.items():
            for j in range(len(gate['pulses'])):
                print('j', j)
                print('####################################')
                # print(gate)
                print(gate['pulses'][j][0].keys())
                for seq_id, part in gate['pulses'][j][0]['hdawg-dev8108'].items():
                    print('seq_id', seq_id)
                    print('sequencer_id', ex_seq.params['sequencer_id'])
                    if seq_id == ex_seq.params['sequencer_id']:
                        # if part[0] not in definition_part:
                        #     definition_part += part[0]
                        #for entry_table_index_constant in part[2]:

                        table_entry = {'index': random_command_id}
                        random_command_id += 1

                        entry_table_index_constant = part[2][0]
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

                        #table_entry['waveform'] = {'index': waveform_id}

                        random_pulse = part[4]
                        #if 'amplitude0' in random_pulse:
                            #table_entry['amplitude0'] = random_pulse['amplitude0']
                        if 'phase0' in random_pulse:
                            table_entry['phase0'] = random_pulse['phase0']
                        #if 'amplitude1' in random_pulse:
                            #table_entry['amplitude1'] = random_pulse['amplitude1']
                        if 'phase1' in random_pulse:
                            table_entry['phase1'] = random_pulse['phase1']

                        command_table['table'].append(table_entry)
                        #command_table['table'].append(table_entry)

        phase0 = ex_seq.phaseI
        phase1 = ex_seq.phaseQ
        table_entry = {'index': random_gate_num}
        #table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': phase0, 'increment': False}
        #table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': phase1, 'increment': False}
        command_table['table'].append(table_entry)

        table_entry = {'index': random_gate_num+1}
        # table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': 0.0, 'increment': True}
        # table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': 0.0, 'increment': True}
        command_table['table'].append(table_entry)

        if self.two_qubit is not None:
            print('two_qubit_gate_index', random_command_id-1)
            two_qubit_gate_index = random_command_id-1

            play_part = textwrap.dedent('''
// Clifford play part
//setPRNGSeed(variable_register1);
    executeTableEntry({random_gate_num});
    //resetOscPhase();
    wait(5);
    //repeat (sequence_len) {{
    repeat (variable_register0) {{
        repeat(1){{
            var rand_value1 = getPRNGValue();
            executeTableEntry({random_gate_num}+1);
            executeTableEntry(rand_value1);
        }}
        repeat ({two_qubit_num}){{'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num = random_gate_num,
                  two_qubit_num=self.two_qubit_num))
            for i in range(self.two_qubit_num):
                play_part += textwrap.dedent('''
//repeat ({two_qubit_num}){{
            wait(5);
            executeTableEntry({random_gate_num}+1);
            executeTableEntry({two_qubit_gate_index});
            wait(5);'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num = random_gate_num,
                  two_qubit_num=self.two_qubit_num))
            play_part += textwrap.dedent('''
//            
        }}        
    }} 
    executeTableEntry({random_gate_num});
    resetOscPhase();
    '''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num = random_gate_num,
                  two_qubit_num=self.two_qubit_num))
        else:
            play_part = textwrap.dedent('''
// Clifford play part
//setPRNGSeed(variable_register1);
    executeTableEntry({random_gate_num});
    //resetOscPhase();
    wait(5);
    //repeat (sequence_len) {{
    repeat (variable_register0) {{
        repeat(1){{
            var rand_value1 = getPRNGValue();
            executeTableEntry({random_gate_num}+1);
            executeTableEntry(rand_value1);
        }}
    }} 
    executeTableEntry({random_gate_num});
    resetOscPhase();
    '''.format(random_gate_num=random_gate_num))
        self.instructions.append(command_table)
        print('command_table')
        print(command_table)

        return definition_part, play_part

    # def create_hdawg_generator(self):
    #     pulses = {}
    #     control_seq_ids = []
    #     self.instructions=[]
    #     for ex_seq in self.ex_sequencers:
    #         pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq, self.sequence_length)
    #         control_seq_ids.append(ex_seq.params['sequencer_id'])
    #
    #     return [[pulses, control_seq_ids]]

    def create_hdawg_generator(self):
        pulses = {}
        control_seq_ids = []
        control_awg_ids = []
        self.instructions = []

        # for device_id in self.prepare_seq[0][0].keys():
        #     pulses.update({device_id: {}})

        for ex_seq in self.ex_sequencers:
            pulses.update({ex_seq.awg.device_id: {}})
        for ex_seq in self.ex_sequencers:
            pulses[ex_seq.awg.device_id][ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq, self.sequence_length)
            control_seq_ids.append(ex_seq.params['sequencer_id'])
            control_awg_ids.append(ex_seq.awg.device_id)
        return [[pulses, control_seq_ids, control_awg_ids]]

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
        # if sequence_length==0:
        # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        # prepare_seq = self.create_hdawg_generator()
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