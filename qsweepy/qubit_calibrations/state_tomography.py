from ..multiqubit_tomography import *
from .calibrated_readout import *
from . import gauss_hd
import itertools


class ProcessTomography(multiqubit_tomography):
    def __init__(self, device, qubit_ids, pause_length=0):
        qubit_readout_pulse, readout_device, confusion_matrix = \
            get_confusion_matrix(device, qubit_ids, pause_length, recalibrate=True, force_recalibration=False)
        ro_seq = [device.pg.pmulti(pause_length)] + device.trigger_readout_seq + qubit_readout_pulse.get_pulse_sequence()

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
                unitary = np.kron(unitary, cube_faces_unitaries[multi_observable[_qubit_id]])
            measurement_operators = {}
            for state in range(2**len(qubit_ids)):
                O = np.diag(confusion_matrix.datasets['resultnumbers'].data[:, state])
                measurement_operators[state] = (np.conj(unitary.T) @ O @ unitary).T

            for multi_initial in itertools.product(*tuple([cube_faces_list]*len(qubit_ids))):
                unitary = np.asarray([1. + 0j])
                for _qubit_id, qubit_id in enumerate(qubit_ids):
                    unitary = np.kron(unitary, cube_faces_unitaries[multi_initial[_qubit_id]])
                I = np.zeros((2**len(qubit_ids), 2**len(qubit_ids)), dtype=np.complex)
                I[0, 0] = 1.0
                initialization_operator = (unitary @ I @ np.conj(unitary.T))

                for state in range(2 ** len(qubit_ids)):
                    output_array[tuple([cube_faces_list.index(o) for o in list(multi_initial)+list(multi_observable)] + [state])] = ''.join(
                        list(multi_initial)+list(multi_observable)) + '-P' + str(state)

                superoperators = {state:np.kron(observable_operator, initialization_operator)
                                  #state:np.einsum('i,j->ij', observable_operator.ravel(), initialization_operator.ravel()) \
                                        for state, observable_operator in measurement_operators.items()}

                multi_qubit_observables[''.join(list(multi_initial)+list(multi_observable))] = {
                    'pulses': [i for i in itertools.chain(*tuple([pulses[qubit_ids[len(qubit_ids)-qubit_id_-1]][cube_face_name] \
                                  for qubit_id_, cube_face_name in enumerate(multi_observable)]))]+ro_seq,
                    'pre_pulses': [i for i in itertools.chain(*tuple([pulses[qubit_ids[len(qubit_ids)-qubit_id_-1]][cube_face_name] \
                                  for qubit_id_, cube_face_name in enumerate(multi_initial)]))],
                    'operators': superoperators }

        super().__init__(reducer, device.pg, multi_qubit_observables, reconstruction_basis)
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