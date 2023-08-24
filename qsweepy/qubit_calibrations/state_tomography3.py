from qsweepy.libraries.multiqubit_tomography3 import *
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.libraries import clifford
from qsweepy.qubit_calibrations.calibrated_readout2 import *
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.qubit_calibrations import gauss_hd3 as gauss_hd
import itertools


class ProcessTomography(multiqubit_tomography):
    def __init__(self, device, qubit_ids, correspondence, pause_length=0):
        qubit_readout_pulse, readout_device, confusion_matrix = \
            get_confusion_matrix(device, qubit_ids, pause_length, recalibrate=True, force_recalibration=False)

        #ro_seq = [device.pg.pmulti(pause_length)] + device.trigger_readout_seq + qubit_readout_pulse.get_pulse_sequence()

        # TODO pulses definition
        # part from benchmarking because I am a lazy ass
        channel_amplitudes_ = {}
        pi2_pulses = {}
        pi_pulses = {}
        generators = {}
        for qubit_id in qubit_ids:
            pi2_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi / 2.)
            pi_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi)
            channel_amplitudes_[qubit_id] = channel_amplitudes.channel_amplitudes(
                device.exdir_db.select_measurement_by_id(pi2_pulses[qubit_id].references['channel_amplitudes']))

        def get_pulse_seq_z(z_phase, qubit_id):
            fast_control = False
            length = float(pi2_pulses[qubit_id].metadata['length'])
            z_pulse = [(c, device.pg.virtual_z, z_phase * 360 / 2 / np.pi, fast_control) for c, a in
                       channel_amplitudes_[qubit_id].items()]
            sequence_z = [device.pg.pmulti(device, length, *tuple(z_pulse))]
            return sequence_z

        def tensor_product(unitary, qubit_id):
            U = [[1]]
            for i in qubit_ids:
                U = np.kron(U, np.identity(2) if i != qubit_id else unitary)
            return U

        generators = {}
        for qubit_id in qubit_ids:
            HZ = {
                'X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(0),
                        'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, -1j], [-1j, 1]]), qubit_id),
                        'price': 1.0},
                '-X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(np.pi),
                         'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, 1j], [1j, 1]]), qubit_id),
                         'price': 1.0},
                # 'Z': {'pulses': get_pulse_seq_z(np.pi, 0, qubit_id),
                #       'unitary': tensor_product([[1, 0], [0, -1]], qubit_id), 'price': 0.1},
                'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2, qubit_id),
                        'unitary': tensor_product([[1, 0], [0, 1j]], qubit_id), 'price': 0.1},
                '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2., qubit_id),
                          'unitary': tensor_product([[1, 0], [0, -1j]], qubit_id), 'price': 0.1},
                'I': {'pulses': get_pulse_seq_z(0, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1]], qubit_id),
                      'price': 0.1}
            }
            generators[qubit_id] = HZ
        if len(qubit_ids) == 2:
            # TODO
            HZ_group = clifford.two_qubit_clifford(*tuple([g for g in generators.values()]),
                                                   plus_op_parallel=device.pg.parallel)
        elif len(qubit_ids) == 1:
            HZ_group = clifford.generate_group(generators=generators[qubit_ids[0]], Clifford_group=False)
        else:
            raise ValueError('More than two qubits are unsupported')

        # TODO qubit sequencer
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
            device.pre_pulses.set_seq_offsets(ex_seq)
            device.pre_pulses.set_seq_prepulses(ex_seq)
            ex_seq.start()
            ex_sequencers.append(ex_seq)

            # Now it should be like this
            exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
            ex_channel = device.awg_channels[exitation_channel]
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
                ex_seq.start()
                ex_sequencers.append(ex_seq)

        # TODO readout sequence
        # ro_seq = [device.pg.pmulti(pause_length)]+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence()
        readout_sequencer = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)

        reconstruction_basis = {}
        reducer = data_reduce.data_reduce(source=readout_device)
        for state in range(2**len(qubit_ids)):
            reducer.filters[state] = data_reduce.cross_section_reducer(source=readout_device, src_meas='resultnumbers', index=state, axis=0)
        for state in range(2**(4*len(qubit_ids))):
            reconstruction_basis[state] = {'operator': np.reshape(np.identity(2**(4*len(qubit_ids)))[state, :],
                                                                 (2**(len(qubit_ids)*2), 2**(len(qubit_ids)*2)))}

        sigma_x = np.asarray([[0,1],  [1,0]])
        sigma_y = np.asarray([[0,-1j],[1j,0]])
        sigma_z = np.asarray([[1,0],  [0,-1]])
        sigma_i = np.asarray([[1,0],  [0,1]])

        cube_faces = {'+Z': (0.0, 0.0),
                      '+X': (np.pi/2., -np.pi/2.,),
                      '+Y': (np.pi/2., 0.0),
                      '-Z': (np.pi, 0.0),
                      '-X': (np.pi/2., np.pi/2.),
                      '-Y': (np.pi/2., np.pi)}

        cube_faces_unitaries = {k:    np.cos(v[0]/2)*sigma_i -
                                   1j*np.sin(v[0]/2)*np.cos(v[1])*sigma_x -
                                   1j*np.sin(v[0]/2)*np.sin(v[1])*sigma_y for k, v in cube_faces.items()}
        pulses = {}
        for qubit_id in qubit_ids:
            pulses[qubit_id] = {}
            for cube_face_name, angles in cube_faces.items():
                pulses[qubit_id][cube_face_name] = gauss_hd.get_excitation_pulse_from_gauss_hd_Rabi_amplitude(
                    device, qubit_id, angles[0], recalibrate=True).get_pulse_sequence(angles[1]) if angles[0] > 0 else []

        multi_qubit_observables = {}
        output_array = np.zeros([len(cube_faces)]*len(qubit_ids)*2+[2**len(qubit_ids)], dtype=object)
        reconstruction_output_array = np.zeros((2**(len(qubit_ids)*2), 2**(len(qubit_ids)*2)), dtype=object)
        for state_id1 in range(2**(len(qubit_ids)*2)):
            for state_id2 in range(2**(len(qubit_ids)*2)):
                reconstruction_output_array[state_id1, state_id2] = str(state_id1*(2**(len(qubit_ids)*2))+state_id2)

        cube_faces_list = [i for i in cube_faces.keys()]
        for multi_observable in itertools.product(*tuple([cube_faces_list]*len(qubit_ids))):
            unitary = np.asarray([1.+0j])
            for _qubit_id, qubit_id in enumerate(qubit_ids):
                #unitary = np.kron(unitary, cube_faces_unitaries[multi_observable[_qubit_id]])
                unitary = np.kron(cube_faces_unitaries[multi_observable[_qubit_id]], unitary)
            measurement_operators = {}
            for state in range(2**len(qubit_ids)):
                O = np.diag(confusion_matrix.datasets['resultnumbers'].data[:, state])
                measurement_operators[state] = (np.conj(unitary.T) @ O @ unitary).T

            for multi_initial in itertools.product(*tuple([cube_faces_list]*len(qubit_ids))):
                unitary = np.asarray([1. + 0j])
                for _qubit_id, qubit_id in enumerate(qubit_ids):
                    #unitary = np.kron(unitary, cube_faces_unitaries[multi_initial[_qubit_id]])
                    unitary = np.kron(cube_faces_unitaries[multi_initial[_qubit_id]], unitary)
                I = np.zeros((2**len(qubit_ids), 2**len(qubit_ids)), dtype=np.complex)
                I[0, 0] = 1.0
                initialization_operator = (unitary @ I @ np.conj(unitary.T))

                for state in range(2 ** len(qubit_ids)):
                    output_array[tuple([cube_faces_list.index(o) for o in list(multi_initial)+list(multi_observable)] + [state])] = ''.join(
                        list(multi_initial)+list(multi_observable)) + '-P' + str(state)

                superoperators = {state:np.kron(observable_operator, initialization_operator)
                                  #state:np.einsum('i,j->ij', observable_operator.ravel(), initialization_operator.ravel()) \
                                        for state, observable_operator in measurement_operators.items()}

                multi_qubit_observables[''.join(list(multi_initial) + list(multi_observable))] = {
                    'pulses': multi_observable,
                    'pre_pulses': multi_initial,
                    'operators': superoperators}

        super().__init__(device, reducer, ex_sequencers, readout_sequencer, device.pg, multi_qubit_observables,
                         qubit_ids=qubit_ids, correspondence=correspondence, reconstruction_basis=reconstruction_basis,
                         interleavers=HZ_group)
        self.output_array = output_array
        self.reconstruction_output_array = reconstruction_output_array

        self.confusion_matrix_id = confusion_matrix.id


class StateTomography(multiqubit_tomography):
    def __init__(self, device, qubit_ids, pause_length=0):
        qubit_readout_pulse, readout_device, confusion_matrix = \
            get_confusion_matrix(device, qubit_ids, pause_length, recalibrate=True, force_recalibration=False)
        ro_seq = [device.pg.pmulti(pause_length)] + device.trigger_readout_seq + qubit_readout_pulse.get_pulse_sequence()

        reconstruction_basis = {}
        reducer = data_reduce.data_reduce(source=readout_device)
        for state in range(2**len(qubit_ids)):
            reducer.filters[state] = data_reduce.cross_section_reducer(source=readout_device, src_meas='resultnumbers', index=state, axis=0)
        for state in range(2**(2*len(qubit_ids))):
            reconstruction_basis[state] = {'operator': np.reshape(np.identity(2**(2*len(qubit_ids)))[state, :],
                                                                 (2**len(qubit_ids), 2**len(qubit_ids)))}

        sigma_x = np.asarray([[0,1],  [1,0]])
        sigma_y = np.asarray([[0,-1j],[1j,0]])
        sigma_z = np.asarray([[1,0],  [0,-1]])
        sigma_i = np.asarray([[1,0],  [0,1]])

        cube_faces = {'+Z': (0.0, 0.0),
                      '+X': (np.pi/2., -np.pi/2.,),
                      '+Y': (np.pi/2., 0.0),
                      '-Z': (np.pi, 0.0),
                      '-X': (np.pi/2., np.pi/2.),
                      '-Y': (np.pi/2., np.pi)}

        cube_faces_unitaries = {k:    np.cos(v[0]/2)*sigma_i -
                                   1j*np.sin(v[0]/2)*np.cos(v[1])*sigma_x -
                                   1j*np.sin(v[0]/2)*np.sin(v[1])*sigma_y for k, v in cube_faces.items()}
        pulses = {}
        for qubit_id in qubit_ids:
            pulses[qubit_id] = {}
            for cube_face_name, angles in cube_faces.items():
                pulses[qubit_id][cube_face_name] = gauss_hd.get_excitation_pulse_from_gauss_hd_Rabi_amplitude(
                    device, qubit_id, angles[0], recalibrate=True).get_pulse_sequence(angles[1]) if angles[0] > 0 else []

        multi_qubit_observables = {}
        output_array = np.zeros([len(cube_faces)]*len(qubit_ids)+[2**len(qubit_ids)], dtype=object)
        reconstruction_output_array = np.zeros((2**len(qubit_ids), 2**len(qubit_ids)), dtype=object)
        for state_id1 in range(2**len(qubit_ids)):
            for state_id2 in range(2**len(qubit_ids)):
                reconstruction_output_array[state_id1, state_id2] = str(state_id1*(2**len(qubit_ids))+state_id2)

        cube_faces_list = [i for i in cube_faces.keys()]
        for multi_observable in itertools.product(*tuple([cube_faces_list]*len(qubit_ids))):
            unitary = np.asarray([1.+0j])
            for _qubit_id, qubit_id in enumerate(qubit_ids):
                unitary = np.kron(unitary, cube_faces_unitaries[multi_observable[_qubit_id]])
            measurement_operators = {}
            for state in range(2**len(qubit_ids)):
                output_array[tuple([cube_faces_list.index(o) for o in multi_observable] + [state])] = ''.join(
                    multi_observable) + '-P' + str(state)

                O = np.diag(confusion_matrix.datasets['resultnumbers'].data[:, state])
                measurement_operators[state] = np.conj(unitary.T) @ O @ unitary
            multi_qubit_observables[''.join(multi_observable)] = {
                    #'pulses':[i for i in itertools.chain(*tuple([pulses[qubit_ids[len(qubit_ids)-qubit_id_-1]][cube_face_name] \
                    #              for qubit_id_, cube_face_name in enumerate(multi_observable)]))]+ro_seq,

                    'pulses':[i for i in itertools.chain(*tuple([pulses[qubit_ids[len(qubit_ids)-qubit_id_-1]][cube_face_name] \
                              for qubit_id_, cube_face_name in enumerate(multi_observable)]))]+ro_seq,
                    'operators':measurement_operators }

        super().__init__(reducer, device.pg, multi_qubit_observables, reconstruction_basis)
        self.output_array = output_array
        self.reconstruction_output_array = reconstruction_output_array

        self.confusion_matrix_id = confusion_matrix.id