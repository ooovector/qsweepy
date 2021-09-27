from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
from qsweepy.libraries import clifford
from qsweepy.libraries import interleaved_benchmarking3 as interleaved_benchmarking
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.fitters import fit_dataset
from qsweepy.ponyfiles.data_structures import *
from qsweepy.fitters import exp
import numpy as np
from qsweepy.libraries.pulses import *


def create_gates_dataset(device, gates):
    gate_ids = np.arange(len(gates))
    state_ids = np.arange(gates[0]['unitary'].shape[0])
    gate_id_parameter = MeasurementParameter(gate_ids, None, 'gate_id', '')
    state_ids_in = MeasurementParameter(state_ids, None, 'state_id_in', '')
    state_ids_out = MeasurementParameter(state_ids, None, 'state_id_out', '')

    measurement = MeasurementState(sample_name=device.exdir_db.sample_name, measurement_type='gate_unitaries')

    gates_array = np.asarray([g['unitary'] for g in gates.values()])

    dataset = MeasurementDataset(parameters=(gate_id_parameter, state_ids_out, state_ids_in), data=gates_array)
    measurement.datasets['gate_unitaries'] = dataset

    device.exdir_db.save_measurement(measurement)
    return measurement


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


def benchmarking_pi2_multi(device, qubit_ids, *params, interleaver=None, two_qubit_gate=None, max_pulses=None,
                           pause_length=0, random_sequence_num=1, seq_lengths_num=400, two_qubit_num=0,
                           random_gate_num=1, seeds = None, shuffle=False):
    channel_amplitudes_ = {}
    pi2_pulses = {}
    pi_pulses = {}
    generators = {}
    if max_pulses is None:
        max_pulses = []
        for qubit_id in qubit_ids:
            coherence_measurement = Ramsey.get_Ramsey_coherence_measurement(device, qubit_id)
            T2 = float(coherence_measurement.metadata['T'])
            pi2_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi / 2.)
            pi2_pulse_length = float(pi2_pulses[qubit_id].metadata['length'])
            pi_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi)
            pi_pulse_length = float(pi_pulses[qubit_id].metadata['length'])
            max_pulses.append(T2 / pi2_pulse_length)

        if two_qubit_gate is not None:
            max_pulses = np.asarray(max_pulses)/3.

        max_pulses = min(max_pulses)

    for qubit_id in qubit_ids:
        pi2_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi / 2.)
        pi_pulses[qubit_id] = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi)
        channel_amplitudes_[qubit_id] = channel_amplitudes.channel_amplitudes(
            device.exdir_db.select_measurement_by_id(pi2_pulses[qubit_id].references['channel_amplitudes']))

    seq_lengths = np.asarray(np.round(np.linspace(0, max_pulses, seq_lengths_num)), int)

    def get_pulse_seq_z(z_phase, length, qubit_id):
        fast_control = False
        z_pulse = [(c, device.pg.virtual_z, z_phase * 360 / 2 / np.pi, fast_control) for c, a in
                   channel_amplitudes_[qubit_id].items()]
        sequence_z = [device.pg.pmulti(device, length, *tuple(z_pulse))]
        return sequence_z

    def tensor_product(unitary, qubit_id):
        U = [[1]]
        for i in qubit_ids:
            U = np.kron(U, np.identity(2) if i != qubit_id else unitary)
        return U
    #TODO
    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    generators = {}
    for qubit_id in qubit_ids:
        HZ1 = {'X': {'pulses': pi_pulses[qubit_id].get_pulse_sequence(0),
                    'unitary':  tensor_product(np.asarray([[0, 1], [1, 0]]), qubit_id), 'price': 1.0},
              'X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(0),
                      'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, -1j], [-1j, 1]]), qubit_id), 'price': 1.0},
              '-X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(np.pi),
                       'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, 1j], [1j, 1]]), qubit_id), 'price': 1.0},
              'Z': {'pulses': get_pulse_seq_z(np.pi, 0, qubit_id), 'unitary': tensor_product([[1, 0], [0, -1]], qubit_id), 'price': 0.1},
              'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2, 0, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1j]], qubit_id), 'price': 0.1},
              '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2., 0,qubit_id), 'unitary': tensor_product([[1, 0], [0, -1j]], qubit_id), 'price': 0.1},
              'I': {'pulses': get_pulse_seq_z(0,0, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1]], qubit_id), 'price': 0.1}
              }
        HZ = {
              'X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(0),
                      'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, -1j], [-1j, 1]]), qubit_id),
                      'price': 1.0},
              '-X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(np.pi),
                       'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, 1j], [1j, 1]]), qubit_id),
                       'price': 1.0},
              'Z': {'pulses': get_pulse_seq_z(np.pi, 0, qubit_id),
                    'unitary': tensor_product([[1, 0], [0, -1]], qubit_id), 'price': 0.1},
              'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2, 0, qubit_id),
                      'unitary': tensor_product([[1, 0], [0, 1j]], qubit_id), 'price': 0.1},
              '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2., 0, qubit_id),
                       'unitary': tensor_product([[1, 0], [0, -1j]], qubit_id), 'price': 0.1},
              'I': {'pulses': get_pulse_seq_z(0, 0, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1]], qubit_id),
                    'price': 0.1}
              }

        generators[qubit_id] = HZ

    if len(qubit_ids) == 2:
        #TODO
        HZ_group = clifford.two_qubit_clifford(*tuple([g for g in generators.values()]),  plus_op_parallel=device.pg.parallel, two_qubit_gate = two_qubit_gate)
    elif len(qubit_ids) == 1:
        HZ_group = clifford.generate_group(generators[qubit_ids[0]])
    else:
        raise ValueError ('More than two qubits are unsupported')
    HZ_group2 =  {
              'X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(0),
                      'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, -1j], [-1j, 1]]), qubit_id), 'price': 1.0},
              '-X/2': {'pulses': pi2_pulses[qubit_id].get_pulse_sequence(np.pi),
                       'unitary': np.sqrt(0.5) * tensor_product(np.asarray([[1, 1j], [1j, 1]]), qubit_id), 'price': 1.0},
              'Z': {'pulses': get_pulse_seq_z(np.pi, 0, qubit_id),
                    'unitary': tensor_product([[1, 0], [0, -1]], qubit_id), 'price': 0.1},
              'I': {'pulses': get_pulse_seq_z(0,0, qubit_id), 'unitary': tensor_product([[1, 0], [0, 1]], qubit_id), 'price': 0.1}
              }
    HZ_group = HZ

    print ('group length:', len(HZ_group))

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
    if seeds is None:
        seeds = np.random.randint(65536, size=(len(seq_lengths), len(ex_sequencers), random_sequence_num))
    references = {'readout_pulse': qubit_readout_pulse.id}

    #TODO readout sequence
    #ro_seq = [device.pg.pmulti(pause_length)]+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence()
    readout_sequencer = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)
    # Readout sequence
    readout_sequencer.start()

    pi2_bench = interleaved_benchmarking.interleaved_benchmarking(readout_device, ex_sequencers, seeds, seq_lengths,
                                                                  interleavers=HZ_group,
                                                                  random_sequence_num=random_sequence_num,
                                                                  two_qubit_num=two_qubit_num,
                                                                  random_gate_num=random_gate_num,
                                                                  readout_sequencer = readout_sequencer)
    gates=[]
    for name, gate in HZ_group.items():
        print(name)
        #print(gate['unitary'])
        gates.append(gate['unitary'])

    np.savez('HZ_group', np.asarray(gates))

    gates_dataset = create_gates_dataset(device, HZ_group)
    references['gate_unitaries'] = gates_dataset.id

    #TODO prepare_seq
    prepare_seq = pi2_bench.create_hdawg_generator()
    sequence_control.set_preparation_sequence(device, ex_sequencers, prepare_seq)



    pi2_bench.random_sequence_num = random_sequence_num
    seeds_ids = np.arange(seeds.shape[2])

    references.update({('pi2_pulse', qubit_id): pi2_pulses[qubit_id].id for qubit_id in qubit_ids})
    #references = references.update({('pi_pulse', qubit_id): pi_pulses[qubit_id].id for qubit_id in qubit_ids})

    # TODO
    # pi2_bench.prepare_random_interleaving_sequences()

    ### search db for previous version of the interleaver measurement
    found = False
    try:
        clifford_bench = device.exdir_db.select_measurement(measurement_type='clifford_bench',
                                    metadata={'qubit_ids': ','.join(qubit_ids)},
                                    references_that=references)
        found = True
    except IndexError:
        pass

    if random_sequence_num > 1:
        params = tuple([(seeds_ids, pi2_bench.set_interleaved_sequence, 'Random seeds id', '')]+[p for p in params])

    if (not found) or (interleaver is None):
        measurement_name = [m for m in pi2_bench.get_points().keys()][0]
        fitter_arguments = (measurement_name, exp.exp_fitter(), 0, np.arange(len(params)).tolist())

        clifford_bench = device.sweeper.sweep(pi2_bench, #device.sweeper.sweep_fit_dataset_1d_onfly(pi2_bench,
                                    (seq_lengths, pi2_bench.set_sequence_length, 'Gate number', ''),
                                    *params,
                                    #fitter_arguments=fitter_arguments,
                                    measurement_type='clifford_bench',
                                    metadata={'qubit_ids': ','.join(qubit_ids)},
                                    shuffle=shuffle,
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
                                    (seq_lengths, pi2_bench.set_sequence_length, 'Gate number', ''),
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

    pi_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi)
    pi_pulse_length = float(pi_pulse.metadata['length'])
    channel_amplitudes_ = channel_amplitudes.channel_amplitudes(
        device.exdir_db.select_measurement_by_id(pi2_pulse.references['channel_amplitudes']))
    max_pulses = T2/pi2_pulse_length
    seq_lengths = np.asarray(np.round(np.linspace(0, max_pulses, seq_lengths_num)), int)

    def get_pulse_seq_z(z_phase, length):
        fast_control = False
        z_pulse = [(c, device.pg.virtual_z, z_phase*360/2/np.pi, fast_control) for c, a in channel_amplitudes_.items()]
        sequence_z = [device.pg.pmulti(device, length, *tuple(z_pulse))]
        return sequence_z

    #TODO
    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, [qubit_id])
    #HZ = {'X': {'pulses': pi_pulse.get_pulse_sequence(0), 'unitary': np.asarray([[0, 1], [1, 0]]), 'price': 1.0},
    #    'X/2': {'pulses': pi2_pulse.get_pulse_sequence(0), 'unitary': np.sqrt(0.5) * np.asarray([[1, -1j], [-1j, 1]]), 'price': 1.0},
    #    '-X/2': {'pulses': pi2_pulse.get_pulse_sequence(np.pi), 'unitary': np.sqrt(0.5) * np.asarray([[1, 1j], [1j, 1]]), 'price': 1.0},
    #    'Z': {'pulses': get_pulse_seq_z(np.pi), 'unitary': np.asarray([[1, 0], [0, -1]]), 'price':0.1},
    #    'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2), 'unitary': np.asarray([[1, 0], [0, 1j]]), 'price':0.1},
    #    '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2.), 'unitary': np.asarray([[1, 0], [0, -1j]]), 'price':0.1},
     #   'I': {'pulses': get_pulse_seq_z(0, qubit_id), 'unitary': np.asarray([[1, 0], [0, 1]]), 'price':0.1}
     #   }
    #X_metadata = pi_pulse.metadata()
    #X2_metadata = pi2_pulse.metadata()
    #X_2_metadata = pi2_pulse.metadata()
    #X_2_metadata['amplitude'] = float(a) * float(self.metadata['amplitude']) * np.exp(1j * phase)
    HZ = {'X': {'pulses': pi_pulse.get_pulse_sequence(0), 'unitary': np.asarray([[0, 1], [1, 0]]), 'price': 1.0},
        'X/2': {'pulses': pi2_pulse.get_pulse_sequence(0), 'unitary': np.sqrt(0.5) * np.asarray([[1, -1j], [-1j, 1]]), 'price': 1.0},
        '-X/2': {'pulses': pi2_pulse.get_pulse_sequence(np.pi), 'unitary': np.sqrt(0.5) * np.asarray([[1, 1j], [1j, 1]]), 'price': 1.0},
        'Z': {'pulses': get_pulse_seq_z(np.pi, pi2_pulse_length), 'unitary': np.asarray([[1, 0], [0, -1]]), 'price':0.1},
        'Z/2': {'pulses': get_pulse_seq_z(np.pi / 2, pi2_pulse_length), 'unitary': np.asarray([[1, 0], [0, 1j]]), 'price':0.1},
        '-Z/2': {'pulses': get_pulse_seq_z(-np.pi / 2., pi2_pulse_length),  'unitary': np.asarray([[1, 0], [0, -1j]]), 'price':0.1},
        'I': {'pulses': get_pulse_seq_z(0, pi2_pulse_length),'unitary': np.asarray([[1, 0], [0, 1]]), 'price':0.1}
        }

    HZ_group = clifford.generate_group(HZ)

    #TODO qubit sequencer
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_seq_id = ex_channel.parent.sequencer_id
    else:
        control_seq_id = ex_channel.channel//2
    ex_sequencers = []

    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses = [])
        else:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)

    pi2_bench = interleaved_benchmarking.interleaved_benchmarking(readout_device, ex_sequencers, HZ_group)
    #pi2_bench.interleavers = HZ_group

    #TODO prepare_seq
    prepare_seq = pi2_bench.create_hdawg_generator()
    sequence_control.set_preparation_sequence(device, ex_sequencers, prepare_seq)

    # TODO Readout sequencer
    #ro_seq = [device.pg.pmulti(pause_length)]+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence()
    readout_sequencer = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)
    readout_sequencer.start()

    #pi2_bench = interleaved_benchmarking.interleaved_benchmarking(readout_device,
    #        set_seq = lambda x: device.pg.set_seq(device.pre_pulses+x+ro_seq))
    #pi2_bench.interleavers = HZ_group

    pi2_bench.random_sequence_num = random_sequence_num
    random_sequence_ids = np.arange(random_sequence_num)

    pi2_bench.prepare_random_interleaving_sequences()
    clifford_bench = device.sweeper.sweep(pi2_bench,
                                    (seq_lengths, pi2_bench.set_sequence_length_and_regenerate, 'Gate number', ''),
                                    *params,
                                    (random_sequence_ids, pi2_bench.set_interleaved_sequence, 'Random sequence id', ''),
                                    shuffle=True,
                                    measurement_type='pi2_bench',
                                    metadata={'qubit_id':qubit_id}, references={'pi2_pulse':pi2_pulse.id,
                                                                                'pi_pulse':pi_pulse.id})
    return clifford_bench