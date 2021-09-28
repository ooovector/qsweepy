from qsweepy.libraries import sweep
import numpy as np
import textwrap
import copy
import matplotlib.pyplot as plt
import time

class HDAWG_PRNG:
    def __init__(self, seed=0xcafe, lower=0, upper=2**16-1):
        self.lsfr = seed
        self.lower = lower
        self.upper = upper

    def next(self):
        lsb = self.lsfr & 1
        self.lsfr = self.lsfr >> 1
        if (lsb):
            self.lsfr = 0xb400 ^ self.lsfr
        rand = ((self.lsfr * (self.upper-self.lower+1) >> 16) + self.lower) & 0xffff
        return rand


class interleaved_benchmarking:
    #def __init__(self, measurer, set_seq, interleavers=None, random_sequence_num=8):
    def __init__(self, measurer, ex_sequencers, seeds, seq_lengths, interleavers=None, random_sequence_num=8,
                 two_qubit_num=1, random_gate_num=1, readout_sequencer=None, two_qubit_name='fSIM'):

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

        self.measurer = measurer
        self.ex_sequencers = ex_sequencers
        self.two_qubit_num = two_qubit_num
        self.two_qubit_name = two_qubit_name
        self.random_gate_num = random_gate_num
        self.interleavers = {}

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


    def return_hdawg_program(self, ex_seq):
        random_gate_num = len(self.interleavers)
        definition_part =  textwrap.dedent('''
const random_gate_num = {random_gate_num};
//setPRNGSeed(variable_register1);
setPRNGRange(0, random_gate_num-1);
var i;
var rand_value;'''.format(random_gate_num=random_gate_num))

        play_part = textwrap.dedent('''
// Clifford play part
    //setPRNGSeed(variable_register1);
    
    wait(5);
    for (i = 0; i < variable_register0; i = i + 1) {{
        
        repeat(variable_register3){{
            rand_value = getPRNGValue(); 
            switch(rand_value) {{'''.format(range=random_gate_num))
        i = 0;
        for name, gate in self.interleavers.items():
            #if name != self.two_qubit_name:
            play_part += textwrap.dedent('''
//
            case {index}://'''.format(index=i))
            i += 1
            for j in range(len(gate['pulses'])):
                for seq_id, part in gate['pulses'][j][0].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        if part[0] not in definition_part:
                            definition_part += part[0]
                        play_part += textwrap.indent(part[1], '                 ')
                        #if i==random_gate_num:
        play_part += '''
//
            }
        }'''
#             else:
#                 for seq_id, part in gate['pulses'][0][0].items():
#                     if seq_id == ex_seq.params['sequencer_id']:
#                         definition_part += part[0]
#                         play_part += textwrap.dedent('''
# //
#         repeat (variable_register2){:''')
#                         play_part += textwrap.indent(part[1],'  ')
#                         play_part += '''
# //
#         }'''
        play_part += '''
//
    }'''

        return definition_part, play_part
    #def create_program_command_table(self, ):


    def create_hdawg_generator(self):
        # variable_register0 corresponds to the length of random sequence
        # variable_register1 corresponds to the seed of random generator PRNG
        # variable_register2 corresponds to the repetition number of the two qubit gate
        prepare_seq = []
        pulses = {}
        control_seq_ids=[]
        for ex_seq in self.ex_sequencers:
            pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
            control_seq_ids.append(ex_seq.params['sequencer_id'])

        return [[pulses, control_seq_ids]]

    #for _i in group

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
        #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        self.sequence_length = sequence_length
        for i, ex_seq in enumerate(self.ex_sequencers):
            #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg0'], self.sequence_length)
            #ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])

        #self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

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
        #TODO set seed for PRNG
        self.seq_id = seq_id
        #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        for i, ex_seq in enumerate(self.ex_sequencers):
            #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg0'], self.sequence_length)
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg1'],
                                    self.seeds[np.where(self.seq_lengths==self.sequence_length)[0], i, seq_id])
            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], ex_seq.params['var_reg2'], self.two_qubit_num)
            #ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
        #time.sleep(0.001)

        #time.sleep(0.01)
        #for i, ex_seq in enumerate(self.ex_sequencers):
            #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
            #ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
        #time.sleep(0.01)
        #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        #self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])


    def measure(self):
        # if self.prepare_random_sequence_before_measure:
        #     print('Some bullshit happens(')
            # self.prepare_random_interleaving_sequences()
            # self.set_interleaved_sequence(0)
        #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        #time.sleep(0.01)
        #self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
        measurement = self.measurer.measure()

        # measurement['Pulse sequence'] = np.array([object()])

        #for i, ex_seq in enumerate(self.ex_sequencers):
        measurement['seed'] = self.seeds[np.where(self.seq_lengths==self.sequence_length)[0], :, self.seq_id]

        #print(measurement['seed'])


        return measurement