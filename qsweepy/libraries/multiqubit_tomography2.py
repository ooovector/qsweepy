from . import data_reduce
import numpy as np
import textwrap
from qsweepy.qubit_calibrations import sequence_control
import time
from . import readout_classifier


class multiqubit_tomography:
    def __init__(self, device, measurer, ex_sequencers, readout_sequencer, pulse_generator, proj_seq,
                 qubit_ids, correspondence, reconstruction_basis={}, interleavers=None, virtual_phase1=0,
                 virtual_phase2=0, sign=False):
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

        self.virtual_phase_1 = virtual_phase1
        self.virtual_phase_2 = virtual_phase2
        self.sign = sign
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
        self.interleavers = {}
        self.instructions = []
        if interleavers is not None:
            for name, gate in interleavers.items():
                self.add_interleaver(name, gate['pulses'], gate['unitary'])

    def return_hdawg_program(self, ex_seq):
        random_gate_num = len(self.interleavers)
        definition_part = ''''''

        command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                             "header": {"version": "0.2"},
                             "table": []}
        # command_table = {'$schema': 'https://json-schema.org/draft-04/schema#',
        #                  'header': {'version': '1.0.0'},
        #                  'table': []}

        random_command_id = 0
        waveform_id = -1

        for name, gate in self.interleavers.items():
            for j in range(len(gate['pulses'])):
                for seq_id, part in gate['pulses'][j][0][ex_seq.awg.device_id].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        phase0 = ex_seq.phaseI
                        phase1 = ex_seq.phaseQ
                        if part[0] not in definition_part:
                            definition_part += part[0]
                            # for entry_table_index_constant in part[2]:

                        table_entry = {'index': random_command_id}
                        random_command_id += 1

                        entry_table_index_constant = part[2][0]
                        if entry_table_index_constant not in definition_part:
                            waveform_id += 1
                            definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                _id=waveform_id)
                            definition_part += part[3]

                        table_entry['waveform'] = {'index': waveform_id}

                        random_pulse = part[4]
                        # if 'amplitude0' in random_pulse:
                        # table_entry['amplitude0'] = random_pulse['amplitude0']
                        # if ex_seq.device.get_sample_global('is_fluxonium')=='True':
                        if ex_seq.params['is_iq']:
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            # if 'amplitude1' in random_pulse:
                            # table_entry['amplitude1'] = random_pulse['amplitude1']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                        else:
                            if ex_seq.device.get_sample_global('is_fluxonium') == 'True':
                                if 'phase0' in random_pulse:
                                    table_entry['phase0'] = random_pulse['phase0']
                                # if 'amplitude1' in random_pulse:
                                # table_entry['amplitude1'] = random_pulse['amplitude1']
                                if 'phase1' in random_pulse:
                                    table_entry['phase1'] = random_pulse['phase1']
                            else:
                                if 'phase0' in random_pulse:
                                    table_entry['phase0'] = random_pulse['phase0']
                                # if 'amplitude1' in random_pulse:
                                # table_entry['amplitude1'] = random_pulse['amplitude1']
                                if 'phase1' in random_pulse:
                                    table_entry['phase1'] = random_pulse['phase0']

                        command_table['table'].append(table_entry)
                        # command_table['table'].append(table_entry)

        table_entry = {'index': random_gate_num}
        # table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': phase0, 'increment': False}
        # table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': phase1, 'increment': False}
        command_table['table'].append(table_entry)

        table_entry = {'index': random_gate_num + 1}
        # table_entry['amplitude0'] = {'value': 1}
        table_entry['phase0'] = {'value': 0, 'increment': True}
        # table_entry['amplitude1'] = {'value': 1}
        table_entry['phase1'] = {'value': 0, 'increment': True}
        command_table['table'].append(table_entry)

        if self.sign:
            two_qubit_gate_index = random_command_id - 2
        else:
            two_qubit_gate_index = random_command_id - 1

        if self.prepare_seq is not None:
            play_part = textwrap.dedent('''
//  Tomography play part
    variable_register0 = getUserReg(var_reg0);
    executeTableEntry({random_gate_num});
    wait(5);

    //Pre pulses
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register1);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register2);
    
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry({two_qubit_gate_index});
    //Pulses
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register3);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register4);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register5);

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num))
        else:
            play_part = textwrap.dedent('''
// Tomography play part
    executeTableEntry({random_gate_num});
    wait(5);
    
    //Pre pulses
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register1);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register2);

    //Pulses
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register3);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register4);
    //executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register5);

    executeTableEntry({random_gate_num});
resetOscPhase();'''.format(random_gate_num=random_gate_num))

        self.instructions.append(command_table)
        print(command_table)

        return definition_part, play_part

    def create_hdawg_generator(self):
        pulses = {}
        control_seq_ids = []
        control_awg_ids = []
        self.instructions = []

        for name, gate in self.interleavers.items():
            for j in range(len(gate['pulses'])):
                for device_id in gate['pulses'][j][0].keys():
                    pulses.update({device_id: {}})

        for ex_seq in self.ex_sequencers:
            pulses[ex_seq.awg.device_id][ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
            control_seq_ids.append(ex_seq.params['sequencer_id'])
            control_awg_ids.append(ex_seq.awg.device_id)
        return [[pulses, control_seq_ids, control_awg_ids]]

    def add_interleaver(self, name, pulse_seq, unitary):
        self.d = unitary.shape[0]
        self.initial_state_vector = np.zeros(self.d)
        self.initial_state_vector[0] = 1.
        self.target_gate_unitary = np.identity(self.d, dtype=np.complex)
        self.interleavers[name] = {'pulses': pulse_seq, 'unitary': unitary}

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

    def set_prepare_seq(self, seq):
        self.prepare_seq = seq
        for name, gate in self.prepare_seq.items():
            self.add_interleaver(name, gate['pulses'], gate['unitary'])

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
        values=self.correspondence[qubit_id][name]
        for ex_seq in self.ex_sequencers:
            if self.correspondence[qubit_id]['awg_id'] == ex_seq.awg.device_id:
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
            #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            if 'pre_pulses' in self.proj_seq[rot]:
                for _id, qubit_id in enumerate(self.qubit_ids):
                    self.function_recognition(qubit_id, registers=[0,1,2], name=self.proj_seq[rot]['pre_pulses'][_id])
                    self.function_recognition(qubit_id, registers=[3,4,5], name=self.proj_seq[rot]['pulses'][_id])
            else:
                for _id, qubit_id in enumerate(self.qubit_ids):
                    self.function_recognition(qubit_id, registers=[3,4,5], name=self.proj_seq[rot]['pulses'][_id])

            #self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            #time.sleep(0.1)
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
