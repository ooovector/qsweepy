from ..ponyfiles.data_structures import *
from ..libraries import data_reduce
from qsweepy import zi_scripts
import traceback


class qubit_readout_pulse(MeasurementState):
    def __init__(self, *args, **kwargs):
        if len(args) and not len(kwargs): # copy constructor
            super().__init__(*args, **kwargs)
        else:
            super().__init__(measurement_type='qubit_readout_pulse', *args, **kwargs)
    def get_pulse_sequence(self):
        return self.definition_fragment, self.play_fragment


def get_qubit_readout_pulse_from_passthrough(device, passthrough_measurement):
    references = {'passthrough_measurement':passthrough_measurement.id}
    if 'channel_calibration' in passthrough_measurement.metadata:
        references['channel_calibration'] = passthrough_measurement.references['channel_calibration']

    compression_1db = float(passthrough_measurement.metadata['compression_1db'])
    additional_noise_appears = float(passthrough_measurement.metadata['additional_noise_appears'])
    if np.isfinite(compression_1db):
        calibration_type = 'compression_1db'
        amplitude = compression_1db
    elif np.isfinite(additional_noise_appears):
        calibration_type = 'additional_noise_appears'
        amplitude = additional_noise_appears
    else:
        calibration_type = 'maximum'
        amplitude = np.max(passthrough_measurement.datasets['S21'].parameters[0].values)
        #raise Exception('Compession_1db and additional_noise_appears not found on passthourgh scan!')
    readout_channel = passthrough_measurement.metadata['channel']
    length = float(passthrough_measurement.metadata['length'])
    metadata={'pulse_type': 'rect',
              'channel': readout_channel,
              'qubit_id': passthrough_measurement.metadata['qubit_id'],
              'amplitude':amplitude,
              'calibration_type': calibration_type,
              'length': passthrough_measurement.metadata['length']}
    try:
        readout_pulse = qubit_readout_pulse(device.exdir_db.select_measurement(measurement_type='qubit_readout_pulse', references_that=references, metadata=metadata))
    except Exception as e:
        traceback.print_exc()
        readout_pulse = qubit_readout_pulse(references=references, metadata=metadata, sample_name=device.exdir_db.sample_name)
        device.exdir_db.save_measurement(readout_pulse)
    #readout_pulse.pulse_sequence = [device.pg.p(readout_channel, length, device.pg.rect, amplitude)]
    '''Warning'''
    # TODO
    readout_pulse.definition_fragment, readout_pulse.play_fragment = device.pg.readout_rect(readout_channel,
                                                                                            length, amplitude)
    return readout_pulse


def get_multi_qubit_readout_pulse(device, qubit_ids, length=None):
    pulses = {}
    references = {}
    if not len(qubit_ids):
        raise ValueError ('No qubits to readout!')
    for qubit_id in qubit_ids:
        pulses[qubit_id] = get_qubit_readout_pulse(device, qubit_id, length)
        length = float(pulses[qubit_id].metadata['length'])
        references[('readout_pulse', qubit_id)] = pulses[qubit_id].id
        '''Warning'''
        #TODO
        pg_args = [(pulse.metadata['channel'], float(pulse.metadata['amplitude'])) for pulse in pulses.values()]
    metadata = {'qubit_ids': ','.join(sorted(qubit_ids)), 'length':length}
    try:
        multi_readout_pulse = qubit_readout_pulse(device.exdir_db.select_measurement(measurement_type='qubit_readout_pulse', references_that=references, metadata=metadata))
    except Exception as e:
        traceback.print_exc()
        multi_readout_pulse = qubit_readout_pulse(references=references, metadata=metadata, sample_name=device.exdir_db.sample_name)
        device.exdir_db.save_measurement(multi_readout_pulse)
    '''Warning'''
    # TODO
    if len(qubit_ids)==1:
        multi_readout_pulse.definition_fragment, multi_readout_pulse.play_fragment = device.pg.readout_rect(pg_args[0][0], float(length), pg_args[0][1])
    else:
        multi_readout_pulse.definition_fragment, multi_readout_pulse.play_fragment = device.pg.readout_rect_multi(float(length), *pg_args)
    return multi_readout_pulse


def get_qubit_readout_pulse(device, qubit_id, length=None):
    from .readout_passthrough2 import readout_passthrough
    from .calibrated_readout2 import readout_fidelity_scan, get_qubit_readout_pulse_from_fidelity_scan

    ## if we need to make a readout scan, here are the parameters
    points = int(device.get_qubit_constant(name='readout_passthrough_points', qubit_id=qubit_id))
    amplitude = float(device.get_qubit_constant(name='readout_passthrough_amplitude', qubit_id=qubit_id))
    if not length:
        length = float(device.get_qubit_constant(name='readout_length', qubit_id=qubit_id))
    amplitudes = np.linspace(0, amplitude, points)
    ignore_other_qubits = bool(device.get_qubit_constant(name='readout_calibration_ignore_other_qubits', qubit_id=qubit_id) == 'True')

    ## identify metadata
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    metadata_passthrough = {'qubit_id': qubit_id}
    if length:
        metadata_passthrough['length'] = str(length)
    metadata_fidelity_scan = { k:v for k,v in metadata_passthrough.items() }
    metadata_fidelity_scan['ignore_other_qubits'] = ignore_other_qubits

    references = {'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
        references['channel_calibration'] = device.awg_channels[readout_channel].get_calibration_measurement()
    ## strategy: if we have a ready readout pulse, use it.
    ## otherwise, try to calibrate a readout pulse. If that fails,
    ## jump to passthrough measurements
    use_calibrated_measurer = bool(device.get_qubit_constant(name='use_calibrated_measurer', qubit_id=qubit_id) == 'True')
    if use_calibrated_measurer:
        try:
            measurement = device.exdir_db.select_measurement(measurement_type='readout_fidelity_scan', metadata=metadata_fidelity_scan, references_that=references)
            pulse = get_qubit_readout_pulse_from_fidelity_scan(device, measurement)
        except Exception as e:
            #print (type(e), str(e))
            traceback.print_exc()
            # if there is no passthrough, calibrate passthrough
            try:
                measurement = readout_fidelity_scan(device, qubit_id, [length], amplitudes, recalibrate_excitation=False, ignore_other_qubits=ignore_other_qubits)
                pulse = get_qubit_readout_pulse_from_fidelity_scan(device, measurement)
            except Exception as e:
               print ('Failed to get readout pulse from fidelity scan, fall back to passthrough')
               traceback.print_exc()
    else:
        try:
            measurement = device.exdir_db.select_measurement(measurement_type='readout_passthrough', metadata=metadata_passthrough, references_that=references)
            pulse = get_qubit_readout_pulse_from_passthrough(device, measurement)
        except Exception as e:
            print (type(e), str(e))
            traceback.print_exc()
            # if there is no passthrough, calibrate passthrough
            measurement = readout_passthrough(device, qubit_id, length=length, amplitudes=amplitudes)
            pulse = get_qubit_readout_pulse_from_passthrough(device, measurement)
    return pulse


def get_readout_calibration(device, qubit_readout_pulse, excitation_pulse=None, recalibrate=False):
    qubit_id = qubit_readout_pulse.metadata['qubit_id']
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    metadata = {'qubit_id':qubit_id}
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'modem_calibration': device.modem.calibration_measurements[readout_channel].id}
    if excitation_pulse is not None:
        references['excitation_pulse'] = excitation_pulse.id
    try:
        assert not recalibrate
        print ()
        return device.exdir_db.select_measurement(measurement_type='readout_background_calibration', metadata=metadata, references_that=references)
    except Exception as e:
        traceback.print_exc()
        new_measurement = measure_readout(device, qubit_readout_pulse)
        new_measurement.measurement_type = 'readout_background_calibration'
        device.exdir_db.db.update_in_database(new_measurement)
        return new_measurement


def measure_readout(device, qubit_readout_pulse, excitation_pulse=None, nums=None):
    qubit_id = qubit_readout_pulse.metadata['qubit_id']
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    adc, mnames =  device.setup_adc_reducer_iq(qubit_id, raw=True)
    adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]

    if not nums:
        nums = int(device.get_qubit_constant(name='uncalibrated_readout_nums', qubit_id=qubit_id))
        adc.set_adc_nums(nums)

    mean_sample = data_reduce.data_reduce(adc)
    mean_sample.filters['Mean_Voltage_AC'] = data_reduce.mean_reducer_noavg(adc, 'Voltage', 0)
    mean_sample.filters['Std_Voltage_AC'] = data_reduce.std_reducer_noavg(adc, 'Voltage', 0, 1)
    mean_sample.filters['S21'] = data_reduce.thru(adc, mnames[qubit_id], diff=0, scale=nums)

    excitation_pulse_sequence = excitation_pulse.get_pulse_sequence(0) if excitation_pulse is not None else []
    ex_channel = device.awg_channels[exitation_channel]


    # if ex_channel.is_iq():
    #     control_awg, control_seq_id = ex_channel.parent.awg, ex_channel.parent.sequencer_id
    # else:
    #     control_awg, control_seq_id = ex_channel.parent.awg, ex_channel.channel // 2

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    '''Warning'''
    #TODO
    ex_sequencers = []
    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
        ex_seq.stop()
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_sequencers.append(ex_seq)
        device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()

    # We neet to set readout sequence to awg readout channel
    # readout pulse parameters ('length' and 'amplitude') we get from qubit_readout_pulse.metadata
    re_channel = device.awg_channels[readout_channel]
    sequence = zi_scripts.READSequence(re_channel.parent.sequencer_id, device.modem.awg)
    def_frag, play_frag = device.pg.readout_rect(channel=readout_channel,
                                                 length=float(qubit_readout_pulse.metadata['length']),
                                                 amplitude=float(qubit_readout_pulse.metadata['amplitude']))
    sequence.add_readout_pulse(def_frag, play_frag)
    sequence.stop()
    device.modem.awg.set_sequence(re_channel.parent.sequencer_id, sequence)
    sequence.set_delay(device.modem.trigger_channel.delay)
    sequence.start()

    #Here we need to set exitation pulse sequense with pre pulses for qubit control channels

    # refers to Awg_iq_multi calibrations
    metadata = {'qubit_id':qubit_id, 'averages': nums}
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'modem_calibration': device.modem.calibration_measurements[readout_channel].id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    if excitation_pulse is not None:
        references['excitation_pulse'] = excitation_pulse.id
    if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
        references[('channel_calibration', readout_channel)] = device.awg_channels[readout_channel].get_calibration_measurement()

    try:
        measurement = device.sweeper.sweep(mean_sample,
                            measurement_type='readout_response',
                            metadata=metadata,
                            references=references)
    except:
        raise

    return measurement


def get_uncalibrated_measurer(device, qubit_id, transition='01', samples = False):
    from .calibrated_readout2 import get_calibrated_measurer

    print (device, qubit_id, transition)

    try:
       assert transition == '01'
       qubit_readout_pulse_, measurer = get_calibrated_measurer(device, [qubit_id], recalibrate=False)
       reducer = data_reduce.data_reduce(measurer)
       reducer.filters['iq'+qubit_id] = data_reduce.cross_section_reducer(measurer, 'resultnumbers', 0, 1)
       return qubit_readout_pulse_, reducer
    except:
       traceback.print_exc()

    #qubit_readout_pulse_ = get_qubit_readout_pulse(device, qubit_id)
    qubit_readout_pulse_ = get_qubit_readout_pulse(device, qubit_id)
    background_calibration = get_readout_calibration(device, qubit_readout_pulse_)
    adc_reducer, mnames = device.setup_adc_reducer_iq(qubit_id, raw=samples)

    if adc_reducer.devtype == 'UHF':
        adc_reducer.set_internal_avg(True)
    else:
        adc_reducer.set_adc_nums(int(device.get_sample_global('uncalibrated_readout_nums')))

    adc_reducer.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    #measurer = adc_reducer
    #measurer.output_resnum = False
    #measurer.output_result = True
    measurer = data_reduce.data_reduce(adc_reducer)
    measurer.filters['iq'+qubit_id] = data_reduce.thru(adc_reducer,  mnames[qubit_id],
                                                        background_calibration.datasets['S21'].data, adc_reducer.get_adc_nums())
    if samples:
        #measurer.output_raw = True
    #else:
        #measurer.output_raw = False
        measurer.filters['Mean_Voltage_AC'] = data_reduce.mean_reducer_noavg(adc_reducer, 'Voltage', 0)
    # measurer.references = {'readout_background_calibration': background_calibration.id}
    nums = int(device.get_qubit_constant(name='uncalibrated_readout_nums', qubit_id=qubit_id))
    adc_reducer.set_adc_nums(nums)
    return qubit_readout_pulse_, measurer

