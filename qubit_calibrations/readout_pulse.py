from .calibrated_readout import *
from ..ponyfiles.data_structures import *
from .. import data_reduce

import traceback
class qubit_readout_pulse(MeasurementState):
    def __init__(self, *args, **kwargs):
        if len(args) and not len(kwargs): # copy constructor
            super().__init__(*args, **kwargs)
        else:
            super().__init__(measurement_type='qubit_readout_pulse', *args, **kwargs)
    def get_pulse_sequence(self):
        return self.pulse_sequence

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
        amplitude = 1.0
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
    readout_pulse.pulse_sequence = [device.pg.p(readout_channel, length, device.pg.rect, amplitude)]
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

        pg_args = [(pulse.metadata['channel'], device.pg.rect, float(pulse.metadata['amplitude'])) for pulse in pulses.values()]
    metadata = {'qubit_ids': ','.join(sorted(qubit_ids)), 'length':length}
    try:
        multi_readout_pulse = qubit_readout_pulse(device.exdir_db.select_measurement(measurement_type='qubit_readout_pulse', references_that=references, metadata=metadata))
    except Exception as e:
        traceback.print_exc()
        multi_readout_pulse = qubit_readout_pulse(references=references, metadata=metadata, sample_name=device.exdir_db.sample_name)
        device.exdir_db.save_measurement(multi_readout_pulse)
    multi_readout_pulse.pulse_sequence = [device.pg.pmulti(float(length), *pg_args)]
    return multi_readout_pulse

def get_qubit_readout_pulse(device, qubit_id, length=None):
    from .readout_passthrough import readout_passthrough

    ## if we need to make a readout scan, here are the parameters
    points = int(device.get_qubit_constant(name='readout_passthrough_points', qubit_id=qubit_id))
    amplitude = float(device.get_qubit_constant(name='readout_passthrough_amplitude', qubit_id=qubit_id))
    if not length:
        length = float(device.get_qubit_constant(name='readout_length', qubit_id=qubit_id))
    amplitudes = np.linspace(0, amplitude, points)

    ## identify metadata
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    metadata={'qubit_id':qubit_id}
    if length:
        metadata['length'] = str(length)

    references = {'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
        references['channel_calibration'] = device.awg_channels[readout_channel].get_calibration_measurement()
    ## strategy: if we have a ready readout pulse, use it.
    ## otherwise, try to calibrate a readout pulse. If that fails,
    ## jump to passthrough measurements
    try:
        measurement = device.exdir_db.select_measurement(measurement_type='readout_fidelity_scan', metadata=metadata, references_that=references)
        pulse = get_qubit_readout_pulse_from_fidelity_scan(device, measurement)
    except Exception as e:
        #print (type(e), str(e))
        traceback.print_exc()
        # if there is no passthrough, calibrate passthrough
        try:
            measurement = readout_fidelity_scan(device, qubit_id, [length], amplitudes, recalibrate_excitation=False)
            pulse = get_qubit_readout_pulse_from_fidelity_scan(device, measurement)
        except Exception as e:
            print ('Failed to get readout pulse from fidelity scan, fall back to passthrough')
            traceback.print_exc()
            try:
                measurement = device.exdir_db.select_measurement(measurement_type='readout_passthrough', metadata=metadata, references_that=references)
                pulse = get_qubit_readout_pulse_from_passthrough(device, measurement)
            except Exception as e:
                print (type(e), str(e))
                traceback.print_exc()
                # if there is no passthrough, calibrate passthrough
                measurement = readout_passthrough(device, qubit_id, length=length, amplitudes=amplitudes)
                pulse = get_qubit_readout_pulse_from_passthrough(device, measurement)
    return pulse


def get_readout_calibration(device, qubit_readout_pulse, excitation_pulse=None):
    qubit_id = qubit_readout_pulse.metadata['qubit_id']
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    metadata = {'qubit_id':qubit_id}
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'modem_calibration': device.modem.calibration_measurements[readout_channel].id}
    if excitation_pulse is not None:
        references['excitation_pulse'] = excitation_pulse.id
    try:
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
    adc.set_nop(int(device.get_sample_global('readout_adc_points')))

    if not nums:
        nums = int(device.get_qubit_constant(name='uncalibrated_readout_nums', qubit_id=qubit_id))
        adc.set_nums(nums)

    mean_sample = data_reduce.data_reduce(adc)
    mean_sample.filters['Mean_Voltage_AC'] = data_reduce.mean_reducer_noavg(adc, 'Voltage', 0)
    mean_sample.filters['Std_Voltage_AC'] = data_reduce.std_reducer_noavg(adc, 'Voltage', 0, 1)
    mean_sample.filters['S21'] = data_reduce.thru(adc, mnames[qubit_id], diff=0, scale=nums)

    excitation_pulse_sequence = excitation_pulse.get_pulse_sequence(0) if excitation_pulse is not None else []

    device.pg.set_seq(excitation_pulse_sequence+device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence())

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


def get_uncalibrated_measurer(device, qubit_id, transition='01'):
    try:
        assert transition == '01'
        qubit_readout_pulse_, measurer = get_calibrated_measurer(device, [qubit_id], recalibrate=False)
        reducer = data_reduce.data_reduce(measurer)
        reducer.filters['iq'+qubit_id] = data_reduce.cross_section_reducer(measurer, 'resultnumbers', 0, 1)
        return qubit_readout_pulse_, reducer
    except:
        traceback.print_exc()
        pass

    qubit_readout_pulse_ = get_qubit_readout_pulse(device, qubit_id)
    background_calibration = get_readout_calibration(device, qubit_readout_pulse_)
    adc_reducer, mnames = device.setup_adc_reducer_iq(qubit_id, raw=False)
    adc_reducer.set_nop(int(device.get_sample_global('readout_adc_points')))
    measurer = data_reduce.data_reduce(adc_reducer)
    measurer.filters['iq'+qubit_id] = data_reduce.thru(adc_reducer,  mnames[qubit_id],
                                                        background_calibration.datasets['S21'].data, adc_reducer.get_nums())
    # measurer.references = {'readout_background_calibration': background_calibration.id}
    nums = int(device.get_qubit_constant(name='uncalibrated_readout_nums', qubit_id=qubit_id))
    adc_reducer.set_nums(nums)
    return qubit_readout_pulse_, measurer
