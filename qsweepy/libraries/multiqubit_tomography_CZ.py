from . import data_reduce
import numpy as np
import textwrap
from qsweepy.qubit_calibrations import sequence_control
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import channel_amplitudes
import time
from . import readout_classifier


class multiqubit_tomography:
    def __init__(self, device, measurer, ex_sequencers, readout_sequencer, pulse_generator, proj_seq,
                 qubit_ids, correspondence, reconstruction_basis={}, interleavers=None, phase_1=0, phase_2=0, phase_x=0,
                 phase_3=0, phase_4=0, number_of_circles=1):
        self.number_of_circles = number_of_circles
        # self.sz_measurer = sz_measurer
        # self.adc = adc
        self.pulse_generator = pulse_generator
        self.proj_seq = proj_seq
        self.reconstruction_basis = reconstruction_basis

        self.device = device
        self.measurer = measurer
        self.ex_sequencers = ex_sequencers
        self.readout_sequencer = readout_sequencer
        self.correspondence = correspondence
        self.qubit_ids = qubit_ids

        self.phase_1 = phase_1
        self.phase_2 = phase_2
        self.phase_x = phase_x
        self.phase_3 = phase_3
        self.phase_4 = phase_4
        # self.adc_reducer = data_reduce.data_reduce(self.sz_measurer.adc)
        # self.adc_reducer.filters['SZ'] = {k:v for k,v in self.sz_measurer.filter_binary.items()}
        # self.adc_reducer.filters['SZ']['filter'] = lambda x: 1-2*self.sz_measurer.filter_binary_func(x)

        ## create a list of all readout names
        readout_names = []
        for readout_operators in self.proj_seq.values():
            for readout_name in readout_operators['operators'].keys():
                readout_names.append(readout_name)
        readout_names = list(set(readout_names))

        self.readout_names = readout_names

        ## initialize the confusion matrix to identity so that it doesn't produce any problems
        self.confusion_matrix = np.identity(len(readout_names))

        self.output_array = []
        self.output_mode = 'array'
        self.reconstruction_output_mode = 'array'
        self.reconstruction_type = 'cvxopt'

        self.prepare_seq = None
        self.middle_pulse = None
        self.interleavers = {}
        self.instructions = []
        if interleavers is not None:
            for name, gate in interleavers.items():
                self.add_interleaver(name, gate['pulses'], gate['unitary'])
        self.interleavers.update(self.create_interleavers())


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
                        if part[0] not in definition_part:
                            definition_part += part[0]
                            # for entry_table_index_constant in part[2]:

                        table_entry = {'index': random_command_id}
                        random_command_id += 1

                        entry_table_index_constant = part[2][0]
                        #if entry_table_index_constant not in definition_part:
                        if entry_table_index_constant not in assign_waveform_indexes.keys():
                            waveform_id += 1
                            assign_waveform_indexes[entry_table_index_constant] = waveform_id
                            definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(_id=waveform_id)
                            definition_part += part[3]
                            table_entry['waveform'] = {'index': waveform_id}
                        else:

                            table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}


                        random_pulse = part[4]
                        # if 'amplitude0' in random_pulse:
                        # table_entry['amplitude0'] = random_pulse['amplitude0']
                        if 'phase0' in random_pulse:
                            table_entry['phase0'] = random_pulse['phase0']
                        # if 'amplitude1' in random_pulse:
                        # table_entry['amplitude1'] = random_pulse['amplitude1']
                        if 'phase1' in random_pulse:
                            table_entry['phase1'] = random_pulse['phase1']

                        command_table['table'].append(table_entry)
                        # command_table['table'].append(table_entry)

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

        two_qubit_gate_index = 9
        play_part = textwrap.dedent('''
//  Tomography play part
    executeTableEntry({random_gate_num});
    wait(5);

    //Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0);
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register1);
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register2);
    
    repeat({repeat}){{
//  First Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(5);
    
//First two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});
    
//Middle X gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(6);
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(7);

//Second two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});

// Second Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(8);}}
    
    //Pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register3);
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register4);
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register5);

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num, repeat = self.number_of_circles))


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

    def add_interleaver(self, name, pulse_seq, unitary):
        self.d = unitary.shape[0]
        self.initial_state_vector = np.zeros(self.d)
        self.initial_state_vector[0] = 1.
        self.target_gate_unitary = np.identity(self.d, dtype=np.complex)
        self.interleavers[name] = {'pulses': pulse_seq, 'unitary': unitary}

    def create_interleavers(self):
        interleavers = {}


        # First Hadamars definition
        h1 = excitation_pulse.get_excitation_pulse(self.device, '1', np.pi / 2)
        channel_pulses_h1 = [
            (c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(h1.metadata['amplitude']) * np.exp(1j * 0),
             float(h1.metadata['sigma']), float(h1.metadata['alpha']), self.phase_1)
            for c, a in h1.channel_amplitudes.metadata.items()]
        h1_pulse = [self.device.pg.pmulti(self.device, float(h1.metadata['length']), *tuple(channel_pulses_h1))]

        h2 = excitation_pulse.get_excitation_pulse(self.device, '2', np.pi / 2)
        channel_pulses_h2 = [
            (c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(h2.metadata['amplitude']) * np.exp(1j * 0),
             float(h2.metadata['sigma']), float(h2.metadata['alpha']), self.phase_2)
            for c, a in h2.channel_amplitudes.metadata.items()]
        h2_pulse = [self.device.pg.pmulti(self.device, float(h2.metadata['length']), *tuple(channel_pulses_h2))]

        hadamars_1 = {'H1': {'pulses': [self.device.pg.parallel(h1_pulse[0], h2_pulse[0])]}, }

        # Two qubit gate definition
        # Gate 1
        gate1 = self.device.get_two_qubit_gates()['iSWAP(1,2)2']
        gate2 = self.device.get_zgates()['z2p_sin']
        full_length = float(gate1.metadata['length'])
        tail_length = float(gate1.metadata['tail_length'])
        length = full_length - 2 * tail_length
        channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(self.device,
                                                                     **{gate1.metadata['carrier_name']: float(
                                                                         gate1.metadata['amplitude'])})
        gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=self.device,
                                                                   channel_amplitudes=channel_amplitudes1_,
                                                                   tail_length=tail_length,
                                                                   length=length,
                                                                   phase=0.0,
                                                                   fast_control=False)
        # Gate 2
        amplitude2 = 1j * float(gate2.metadata['amplitude'])
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(self.device,
                                                                     **{gate2.metadata['carrier_name']: amplitude2})
        frequency2 = float(gate2.metadata['frequency'])
        phase = 0.0
        initial_phase = 0
        fast_control = False
        channel_pulses = [(c, self.device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                          c, a in channel_amplitudes2_.items()]
        gate2_pulse = [self.device.pg.pmulti(self.device, full_length, *tuple(channel_pulses))]

        two_qubit_gate = {'fSIM': {'pulses': [self.device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]}, }

        # Middle_pulse qubit 1
        # Middle_pulse X/2 first
        ex_pulse = excitation_pulse.get_excitation_pulse(self.device, '1', np.pi / 2)
        channel_pulses = [(c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(ex_pulse.metadata['amplitude']) * np.exp(1j * 0),
        float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']), self.phase_x)
        for c, a in ex_pulse.channel_amplitudes.metadata.items()]
        pulse1 = [self.device.pg.pmulti(self.device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]
        # Middle_pulse X/2 second
        channel_pulses = [(c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(ex_pulse.metadata['amplitude']) * np.exp(1j * 0),
        float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']),
        self.phase_x + float(ex_pulse.metadata['phase']) + 2 * np.pi * (round(float(ex_pulse.metadata['length'])* 2.4e9/16)*16/2.4e9 * self.device.pg.channels[c].get_frequency() % 1))
        for c, a in ex_pulse.channel_amplitudes.metadata.items()]
        pulse2 = [self.device.pg.pmulti(self.device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]

        middle_pulse = {'X1': {'pulses': pulse1}, 'X2': {'pulses': pulse2}}

        # Second Hadamars definition
        h3 = excitation_pulse.get_excitation_pulse(self.device, '1', np.pi / 2)
        channel_pulses_h3 = [(c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(h3.metadata['amplitude']) * np.exp(1j * 0),
        float(h3.metadata['sigma']), float(h3.metadata['alpha']), self.phase_3)
        for c, a in h3.channel_amplitudes.metadata.items()]
        h3_pulse = [self.device.pg.pmulti(self.device, float(h3.metadata['length']), *tuple(channel_pulses_h3))]

        h4 = excitation_pulse.get_excitation_pulse(self.device, '2', np.pi / 2)
        channel_pulses_h4 = [(c, self.device.pg.gauss_hd_modulation, float(a) * 1j * float(h4.metadata['amplitude']) * np.exp(1j * 0),
        float(h4.metadata['sigma']), float(h4.metadata['alpha']), self.phase_4)
        for c, a in h4.channel_amplitudes.metadata.items()]
        h4_pulse = [self.device.pg.pmulti(self.device, float(h4.metadata['length']), *tuple(channel_pulses_h4))]

        hadamars_2 = {'H2': {'pulses': [self.device.pg.parallel(h3_pulse[0], h4_pulse[0])]}}

        interleavers.update(hadamars_1)  # 5
        interleavers.update(middle_pulse)  # 6,7
        interleavers.update(hadamars_2)  # 8
        interleavers.update(two_qubit_gate)  # 9
        return interleavers

    def get_points(self):
        if self.output_mode == 'single':
            points = {p: {} for p in self.proj_seq.keys()}
        elif self.output_mode == 'array':
            points = {'measurement': [('ax{}'.format(ax_id), np.arange(self.output_array.shape[ax_id]), 'projection') \
                                      for ax_id in range(len(self.output_array.shape))]}
        else:
            raise Exception('unknown output_mode')
        if self.reconstruction_output_mode == 'single':
            points.update({p: {} for p in self.reconstruction_basis.keys()})
        elif self.reconstruction_output_mode == 'array':
            points['reconstruction'] = [
                ('ax{}'.format(ax_id), np.arange(self.reconstruction_output_array.shape[ax_id]), 'projection') \
                for ax_id in range(len(self.reconstruction_output_array.shape))]
        return points

    def get_dtype(self):
        if self.output_mode == 'single':
            dtypes = {p: float for p in self.proj_seq.keys()}
        elif self.output_mode == 'array':
            dtypes = {'measurement': float}
        else:
            raise Exception('unknown output_mode')
        if self.reconstruction_output_mode == 'single':
            dtypes.update({p: complex for p in self.reconstruction_basis.keys()})
        elif self.reconstruction_output_mode == 'array':
            dtypes['reconstruction'] = complex
        return dtypes

    def reconstruct(self, measurement_results):
        from cvxpy import Variable, atoms, abs, reshape, Minimize, Problem, CVXOPT
        from traceback import print_exc
        reconstruction_operator_names = []
        reconstruction_operators = []
        basis_axes_names = self.reconstruction_basis.keys()
        basis_vector_norms = np.asarray(
            [np.linalg.norm(self.reconstruction_basis[r]['operator']) for r in basis_axes_names])

        for reconstruction_operator_name, reconstruction_operator in self.reconstruction_basis.items():
            reconstruction_operator_names.append(reconstruction_operator_name)
            reconstruction_operators.append(reconstruction_operator['operator'])

        reconstruction_matrix = []
        for rot, projection in self.proj_seq.items():
            for measurement_name, projection_operator in projection['operators'].items():
                reconstruction_matrix.append([np.sum(projection_operator * np.conj(reconstruction_operator)) / np.sum(
                    np.abs(reconstruction_operator) ** 2) for reconstruction_operator in reconstruction_operators])

        reconstruction_matrix_pinv = np.linalg.pinv(reconstruction_matrix)
        reconstruction_matrix = np.asarray(reconstruction_matrix)
        self.reconstruction_matrix = reconstruction_matrix
        self.reconstruction_matrix_pinv = reconstruction_matrix_pinv

        projections = np.dot(reconstruction_matrix_pinv, measurement_results)
        reconstruction = {str(k): v for k, v in zip(basis_axes_names, projections)}

        if self.reconstruction_type == 'cvxopt':
            # x = cvxpy.Variable(len(projections), complex=True)
            x = Variable(len(projections), complex=True)
            rmat_normalized = np.asarray(reconstruction_matrix / np.mean(np.abs(measurement_results)), dtype=complex)
            meas_normalized = np.asarray(measurement_results).ravel() / np.mean(np.abs(measurement_results))
            # lstsq_objective = cvxpy.atoms.sum_squares(cvxpy.abs(rmat_normalized @ x - meas_normalized))
            lstsq_objective = atoms.sum_squares(abs(rmat_normalized @ x - meas_normalized))
            matrix_size = int(np.round(np.sqrt(len(projections))))
            # x_reshaped = cvxpy.reshape(x, (matrix_size, matrix_size))
            x_reshaped = reshape(x, (matrix_size, matrix_size))
            psd_constraint = x_reshaped >> 0
            hermitian_constraint = x_reshaped.H == x_reshaped
            # Create two constraints.
            constraints = [psd_constraint, hermitian_constraint]
            # Form objective.
            # obj = cvxpy.Minimize(lstsq_objective)
            obj = Minimize(lstsq_objective)
            # Form and solve problem.
            # prob = cvxpy.Problem(obj, constraints)
            prob = Problem(obj, constraints)
            try:
                prob.solve(solver=CVXOPT, verbose=True)
                reconstruction = {str(k): v for k, v in zip(basis_axes_names, np.asarray(x.value))}
            except ValueError as e:
                print_exc()

        if self.reconstruction_output_mode == 'array':
            it = np.nditer([self.reconstruction_output_array, None], flags=['refs_ok'], op_dtypes=(object, complex))
            with it:
                for x, z in it:
                    z[...] = reconstruction[str(x)]
                reconstruction = it.operands[1]

        return reconstruction

    def pre_sweep(self):
        self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
        prepare_seq = self.create_hdawg_generator()
        for ex_seq in self.ex_sequencers:
            for register in range(7):
                ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)

        sequence_control.set_preparation_sequence(self.device, self.ex_sequencers, prepare_seq,
                                                  instructions=self.instructions)
        self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

    def function_recognition(self, qubit_id, registers, name):
        values = self.correspondence[qubit_id][name]
        for ex_seq in self.ex_sequencers:
            if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                for _i, register in enumerate(registers):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, values[_i])

    def measure(self):
        meas = {}
        measurement_results = []

        for rot, projection in self.proj_seq.items():
            # if 'pre_pulses' in self.proj_seq[rot]:
            #     self.pulse_generator.set_seq(
            #         self.proj_seq[rot]['pre_pulses'] + self.prepare_seq + self.proj_seq[rot]['pulses'])
            # else:
            #     self.pulse_generator.set_seq(self.prepare_seq + self.proj_seq[rot]['pulses'])
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            if 'pre_pulses' in self.proj_seq[rot]:
                for _id, qubit_id in enumerate(self.qubit_ids):
                    self.function_recognition(qubit_id, registers=[0, 1, 2], name=self.proj_seq[rot]['pre_pulses'][_id])
                    self.function_recognition(qubit_id, registers=[3, 4, 5], name=self.proj_seq[rot]['pulses'][_id])
            else:
                for _id, qubit_id in enumerate(self.qubit_ids):
                    self.function_recognition(qubit_id, registers=[3, 4, 5], name=self.proj_seq[rot]['pulses'][_id])

            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            # time.sleep(0.1)
            measurement = self.measurer.measure()
            # print (projection.keys())
            measurement_ordered = [measurement[readout_name] for readout_name in self.readout_names]
            # print (rot)
            # print ('uncorreted:', measurement)
            # measurement_corrected = np.linalg.lstsq(self.confusion_matrix.T, measurement_ordered)[0]
            # measurement = {readout_name:measurement for readout_name, measurement in zip(self.readout_names, measurement_corrected)}
            # print ('corrected:', measurement)

            for measurement_name, projection_operator in projection['operators'].items():
                measurement_basis_coefficients = []
                projection_operator_name = str(rot) + '-P' + str(measurement_name)

                meas[projection_operator_name] = measurement[measurement_name]
                measurement_results.append(measurement[measurement_name])

        self.measurement_results = measurement_results
        reconstruction = self.reconstruct(measurement_results)

        # reconstruction_matrix_pinv = np.linalg.pinv(reconstruction_matrix)
        # self.reconstruction_matrix = reconstruction_matrix
        # self.reconstruction_matrix_pinv = reconstruction_matrix_pinv

        # print ('reconstruction_matrix (real): ', np.real(reconstruction_matrix).astype(int))
        # print('reconstruction_matrix (imag): ', np.imag(reconstruction_matrix).astype(int))
        # print ('measured projections: ', measurement_results)

        # projections = np.dot(reconstruction_matrix_pinv, measurement_results)
        # print('reconstruction_results (real): ', np.real(projections))
        # print('reconstruction_results (imag): ', np.imag(projections))

        if self.output_mode == 'array':
            it = np.nditer([self.output_array, None], flags=['refs_ok'], op_dtypes=(object, float))
            with it:
                for x, z in it:
                    print(x)
                    z[...] = meas[str(x)]
                meas = {'measurement': it.operands[1]}  # same as z

        if self.reconstruction_output_mode == 'array':
            meas['reconstruction'] = reconstruction
        else:
            meas.update(reconstruction)

        return meas

    def get_opts(self):
        if self.output_mode == 'single':
            opts = {p: {} for p in self.proj_seq.keys()}
        elif self.output_mode == 'array':
            opts = {'measurement': {}}
        opts.update({p: {} for p in self.reconstruction_basis.keys()})
        return opts
