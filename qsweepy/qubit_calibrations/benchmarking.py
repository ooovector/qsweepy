from . import calibrated_readout
from .. import clifford
from .. import interleaved_benchmarking
from . import Ramsey
from . import excitation_pulse
from . import channel_amplitudes
from ..fitters import fit_dataset
from ..ponyfiles.data_structures import *
from ..fitters import exp
import numpy as np
from ..pulses import *


def create_flat_dataset(measurement, dataset_name):
    dataset = measurement.datasets[dataset_name]
    averaging_parameter_name = 'Random sequence id'
    time_parameter_name = 'Gate number'
    for parameter_id, parameter in enumerate(dataset.parameters):
        if parameter.name == time_parameter_name:
            time_parameter_id = parameter_id
        if parameter.name == averaging_parameter_name:
            averaging_parameter_id = parameter_id
    flat_dataset = np.reshape(dataset.data, list(dataset.data.shape[:(time_parameter_id)])+
                [dataset.data.shape[time_parameter_id]*dataset.data.shape[time_parameter_id+1]]+
                list(dataset.data.shape[time_parameter_id+2:]))
    flat_dataset_parameters = [MeasurementParameter(**p.__dict__) for p in dataset.parameters if p.name != 'Random sequence id']
    flat_dataset_parameters[time_parameter_id].values = np.repeat(
        flat_dataset_parameters[time_parameter_id].values, len(dataset.parameters[averaging_parameter_id].values))
    measurement.datasets[dataset_name+'_flat'] = MeasurementDataset(flat_dataset_parameters, flat_dataset)


def benchmarking_pi2_multi(device, qubit_ids, *params, interleaver=None, two_qubit_gate=None, pause_length=0, random_sequence_num=1, seq_lengths_num=400):
    max_pulses = []
    channel_amplitudes_ = {}
    pi2_pulses = {}
    generators = {}
    for qubit_id in qubit_ids:
        coherence_measurement = Ramsey.get_Ramsey_coherence_measurement(device, qubit_id)
        T2 = float(coherence_measurement.metadata['T'])
        pi2_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi / 2.)
        pi2_pulse_length = float(pi2_pulses[qubit_id].metadata['length'])
        channel_amplitudes_[qubit_id] = channel_amplitudes.channel_amplitudes(
            device.exdir_db.select_measurement_by_id(pi2_pulses[qubit_id].references['channel_amplitudes']))
        max_pulses.append(T2 / pi2_pulse_length)

    if two_qubit_gate is not None:
        max_pulses = np.asarray(max_pulses)/3.

    seq_lengths = np.asarray(np.round(np.linspace(0, min(max_pulses), seq_lengths_num)), int)

    def get_pulse_seq_z(z_phase, qubit_id):
        pg = device.pg
        z_pulse = [(c, vz, z_phase) for c, a in channel_amplitudes_[qubit_id].items()]
        sequence_z = [pg.pmulti(0, *tuple(z_pulse))]
        return sequence_z

    def tensor_product(unitary, qubit_id):
        U = [[1]]
        for i in qubit_ids:
            U = np.kron(U, np.identity(2) if i != qubit_id else unitary)
        return U

    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    generators = {}
    for qubit_id in qubit_ids:
        HZ = {'H_'+qubit_id: {
            'pulses': get_pulse_seq_z(np.pi / 2, qubit_id) + pi2_pulses[qubit_id].get_pulse_sequence(np.pi) + get_pulse_seq_z(np.pi / 2, qubit_id),
            'unitary': np.sqrt(0.5) * tensor_product([[1, 1], [1, -1]], qubit_id),
            'price': 1.0},
            'Z_'+qubit_id: {'pulses': get_pulse_seq_z(np.pi, qubit_id), 'unitary': tensor_product([[1, 0], [0, -1]], qubit_id), 'price':0.1},
            'Z/2_'+qubit_id: {'pulses': get_pulse_seq_z(np.pi / 2, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1j]], qubit_id), 'price':0.1},
            '-Z/2_'+qubit_id: {'pulses': get_pulse_seq_z(-np.pi / 2., qubit_id), 'unitary': tensor_product([[1, 0], [0, -1j]], qubit_id), 'price':0.1},
            'I_'+qubit_id: {'pulses': [], 'unitary': tensor_product([[1, 0], [0, 1]], qubit_id), 'price':0.1}
        }
        generators[qubit_id] = HZ

    if len(qubit_ids) == 2:
        HZ_group = clifford.two_qubit_clifford(*tuple([g for g in generators.values()]),  plus_op_parallel=device.pg.parallel, cphase = two_qubit_gate)
    elif len(qubit_ids) == 1:
        HZ_group = clifford.generate_group(generators[qubit_ids[0]])
    else:
        raise ValueError ('More than two qubits are unsupported')

    print ('group length:', len(HZ_group))

    ro_seq = [device.pg.pmulti(pause_length)]+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence()
    pi2_bench = interleaved_benchmarking.interleaved_benchmarking(readout_device,
            set_seq = lambda x: device.pg.set_seq(x+ro_seq), interleavers = HZ_group)

    pi2_bench.random_sequence_num = random_sequence_num
    random_sequence_ids = np.arange(random_sequence_num)

    references = {('pi2_pulse', qubit_id): pi2_pulses[qubit_id].id for qubit_id in qubit_ids}

    pi2_bench.prepare_random_interleaving_sequences()

    ### search db for previous version of the interleaver measurement
    found = False
    try:
        clifford_bench = device.exdir_db.select_measurement(measurement_type='clifford_bench',
                                    metadata={'qubit_ids': ','.join(qubit_ids)},
                                    references_that=references)
        found = True
    except IndexError:
        pass
    if (not found) or (interleaver is None):
        measurement_name = [m for m in pi2_bench.get_points().keys()][0]
        fitter_arguments = (measurement_name, exp.exp_fitter(), 0, np.arange(len(params)).tolist())

        clifford_bench = device.sweeper.sweep_fit_dataset_1d_onfly(pi2_bench,
                                    (seq_lengths, pi2_bench.set_sequence_length_and_regenerate, 'Gate number', ''),
                                    *params,
                                    fitter_arguments=fitter_arguments,
                                    measurement_type='clifford_bench',
                                    metadata={'qubit_ids': ','.join(qubit_ids)},
                                    shuffle=True,
                                    references=references)

    ## interleaver measurement is found, bench "interleaver" gate
    references['Clifford-bench'] = clifford_bench.id
    if interleaver is not None:
        if 'references' in interleaver:
            references.update(interleaver['references'])

        pi2_bench.set_target_pulse(interleaver)

        measurement_name = [m for m in pi2_bench.get_points().keys()][0]
        fitter_arguments = (measurement_name, exp.exp_fitter(), 0, np.arange(len(params)).tolist())

        interleaved_bench = device.sweeper.sweep_fit_dataset_1d_onfly(pi2_bench,
                                    (seq_lengths, pi2_bench.set_sequence_length_and_regenerate, 'Gate number', ''),
                                    *params,
                                    fitter_arguments=fitter_arguments,
                                    measurement_type='interleaved_bench',
                                    metadata={'qubit_ids': ','.join(qubit_ids)},
                                    shuffle=True,
                                    references=references)

        return interleaved_bench

    return clifford_bench


def benchmarking_pi2(device, qubit_id, *params, pause_length=0, random_sequence_num=1, seq_lengths_num=400):
    coherence_measurement = Ramsey.get_Ramsey_coherence_measurement(device, qubit_id)
    T2 = float(coherence_measurement.metadata['T'])
    pi2_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2.)
    pi2_pulse_length = float(pi2_pulse.metadata['length'])
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(
        device.exdir_db.select_measurement_by_id(pi2_pulse.references['channel_amplitudes']))
    max_pulses = T2/pi2_pulse_length
    seq_lengths = np.asarray(np.round(np.linspace(0, max_pulses, seq_lengths_num)), int)

    def get_pulse_seq_z(z_phase):
        pg = device.pg
        z_pulse = [(c, vz, z_phase) for c, a in channel_amplitudes_.items()]
        sequence_z = [pg.pmulti(0, *tuple(z_pulse))]
        return sequence_z

    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, [qubit_id])
    HZ = {'H': {
        'pulses': get_pulse_seq_z(np.pi / 2) + pi2_pulse.get_pulse_sequence(np.pi) +get_pulse_seq_z(np.pi / 2),
        'unitary': np.sqrt(0.5) * np.asarray([[1, 1], [1, -1]]), 'price':1.0},
          'Z': {'pulses': get_pulse_seq_z(np.pi), 'unitary': np.asarray([[1, 0], [0, -1]]), 'price':0.1},
          'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2), 'unitary': np.asarray([[1, 0], [0, 1j]]), 'price':0.1},
          '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2.), 'unitary': np.asarray([[1, 0], [0, -1j]]), 'price':0.1},
          'I': {'pulses': [], 'unitary': np.asarray([[1, 0], [0, 1]]), 'price':0.1}
          }

    HZ_group = clifford.generate_group(HZ)

    ro_seq = [device.pg.pmulti(pause_length)]+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence()
    pi2_bench = interleaved_benchmarking.interleaved_benchmarking(readout_device,
            set_seq = lambda x: device.pg.set_seq(x+ro_seq))

    pi2_bench.interleavers = HZ_group

    pi2_bench.random_sequence_num = random_sequence_num
    random_sequence_ids = np.arange(random_sequence_num)

    pi2_bench.prepare_random_interleaving_sequences()
    clifford_bench = device.sweeper.sweep(pi2_bench,
                                    (seq_lengths, pi2_bench.set_sequence_length_and_regenerate, 'Gate number', ''),
                                    *params,
                                    (random_sequence_ids, pi2_bench.set_interleaved_sequence, 'Random sequence id', ''),
                                    shuffle=True,
                                    measurement_type='pi2_bench',
                                    metadata={'qubit_id':qubit_id}, references={'pi2_pulse':pi2_pulse.id})
    return clifford_bench