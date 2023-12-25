from qsweepy.libraries import data_reduce
from qsweepy.ponyfiles.data_structures import *

def readout_passthrough(device, qubit_id, length, amplitudes):
    readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    adc, mnames = device.setup_adc_reducer_iq(qubit_id, raw=True)
    adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    adc.set_adc_nums(int(device.get_sample_global('uncalibrated_readout_nums')))
    mean_sample = data_reduce.data_reduce(adc)
    if len(adc.get_points()['Voltage']) > 1:
        mean_sample.filters['Mean_Voltage_AC'] = data_reduce.mean_reducer_noavg(adc, 'Voltage', 0)
        mean_sample.filters['Std_Voltage_AC'] = data_reduce.std_reducer_noavg(adc, 'Voltage', 0, 1)
    else:
        mean_sample.filters['Mean_Voltage_AC'] = data_reduce.thru(adc, 'Voltage')
        mean_sample.filters['Std_Voltage_AC'] = data_reduce.thru(adc, mnames[qubit_id], 0, np.inf)
    mean_sample.filters['S21'] = data_reduce.thru(adc, mnames[qubit_id])
    pre_pulse_set(device, qubit_id)
    # We need to set sequence to awg for readout passthrough calibration

    re_channel = device.awg_channels[readout_channel]
    # sequence = zi_scripts.READSequence(re_channel.parent.sequencer_id, device.modem.awg)
    sequence = zi_scripts.READSequence(device, re_channel.parent.sequencer_id, device.modem.awg)
    def_frag, play_frag = device.pg.readout_rect(channel=readout_channel, length=float(length), amplitude=0)
    sequence.add_readout_pulse(def_frag, play_frag)
    sequence.stop()
    #device.modem.awg.set_sequence(sequence.params['sequencer_id'], sequence)
    sequence.awg.set_sequence(sequence.params['sequencer_id'], sequence)
    sequence.set_delay(device.modem.trigger_channel.delay)
    sequence.start()

    def set_amplitude(amplitude):
        sequence.set_awg_amp(amplitude)
    # device.pg.set_seq(device.pre_pulses+device.trigger_readout_seq+[device.pg.p(readout_channel, length, device.pg.rect, amplitude)])

    # refers to Awg_iq_multi calibrations
    metadata = {'channel': readout_channel, 'qubit_id':qubit_id, 'averages': device.modem.adc.get_adc_nums(), 'length': length}
    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
        references['channel_calibration'] = device.awg_channels[readout_channel].get_calibration_measurement()

    def create_compression_dataset(measurement):
        parameters = [MeasurementParameter(values=amplitudes[2:], name='amplitude', setter=False)]
        measurement.datasets['compression'] = MeasurementDataset(parameters, np.zeros(len(amplitudes) - 2) * np.nan)


        parameters = [MeasurementParameter(values=amplitudes[2:], name='amplitude', setter=False)]
        measurement.datasets['cos_dist'] = MeasurementDataset(parameters, np.zeros(len(amplitudes) - 2) * np.nan)


    measurement = device.sweeper.sweep(mean_sample,
                                       (amplitudes, set_amplitude, 'amplitude'),
                                       # (lengths, set_pulse_length, 'length'),
                                       measurement_type='readout_passthrough',
                                       metadata=metadata,
                                       references=references,
                                       on_start=[(create_compression_dataset, tuple())],
                                       on_update=[(compression, tuple()),
                                                  (spread, tuple())])

    return measurement


def spread(measurement, _):
    try:
        zero_noise = np.mean(measurement.datasets['Std_Voltage_AC'].data[0, :])
        zero_noise_std = np.std(measurement.datasets['Std_Voltage_AC'].data[0, :])

        drive_amplitudes = measurement.datasets['Mean_Voltage_AC'].parameters[0].values[1:]
        noise = measurement.datasets['Std_Voltage_AC'].data[1:, :]

        additional_noise_ratio = np.mean(np.abs(noise - zero_noise), axis=1) / zero_noise_std
        # print (additional_noise_ratio)
        spread_point = np.argmax(
            additional_noise_ratio > 2)  ###TODO: statistics doesn't work this way, there is a cleaner way of checking

        if np.any(spread_point):
            measurement.metadata['additional_noise_appears'] = str(drive_amplitudes[spread_point + 1])
        else:
            measurement.metadata['additional_noise_appears'] = 'nan'  # ro_amplitude = drive_amplitudes[-1]
    except:
        measurement.metadata['additional_noise_appears'] = 'nan'


# print (spread/noise_spread)

def compression(measurement, _):
    zero_response = measurement.datasets['Mean_Voltage_AC'].data[0, :]
    drive_amplitudes = measurement.datasets['Mean_Voltage_AC'].parameters[0].values[1:]
    signal = measurement.datasets['Mean_Voltage_AC'].data[1:, :]
    if len(measurement.datasets['Std_Voltage_AC'].data.shape) == 2:
        noise = measurement.datasets['Std_Voltage_AC'].data[1:, :]
    elif len(measurement.datasets['Std_Voltage_AC'].data.shape) == 1:
        noise = measurement.datasets['Std_Voltage_AC'].data[1:]

    error = noise / np.sqrt(int(measurement.metadata['averages']))
    signal_overlap = np.sum(np.conj(signal[0, :]) * signal[1:, :], axis=1) / drive_amplitudes[1:]
    cos_dist = np.sum(np.conj(signal[0, :]) * signal[1:, :], axis=1) / np.sum(np.abs(signal[0, :] * signal[1:, :]),
                                                                              axis=1)

    signal_overlap_estimate = np.real(signal_overlap[0])
    # signal_overlap_error = 0.5 * np.sqrt(
    #     np.sum((np.abs(signal[1:, :]) * error[0, :]) ** 2, axis=1) + np.sum((np.abs(signal[0, :]) * error[1:, :]) ** 2,
    #                                                                         axis=1)) / drive_amplitudes[1:]
    # signal_overlap_estimate = (np.sum(np.abs(signal[0,:])**2) - np.sum(error[0,:]*np.abs(signal[0,:]))-np.sum(np.abs(error[0,:])**2))/drive_amplitudes[0]
    # plt.figure()
    # plt.plot(np.real(signal_overlap))
    # plt.plot(np.sum(noise**2, axis=1)/adc.get_nums())
    # plt.plot(signal_overlap_error)
    compression = 10 * np.log10(np.real(signal_overlap) / np.real(signal_overlap_estimate))
    db_compression_point1 = np.argmax(np.abs(10 * np.log10(np.real(signal_overlap) / np.real(
        signal_overlap_estimate))) > 0.8)  # -10*np.log10(1-signal_overlap_error/signal_overlap_estimate))
    db_compression_point2 = np.argmax(np.max(cos_dist) - cos_dist > 0.01)

    if np.any(db_compression_point1) and np.any(db_compression_point2):
        db_compression_point = np.min([db_compression_point1, db_compression_point2])
    elif np.any(db_compression_point1):
        db_compression_point = db_compression_point1
    elif np.any(db_compression_point2):
        db_compression_point = db_compression_point2
    else:
        db_compression_point = None

    # print("db compression point is", db_compression_point)

    # 10*np.log10(np.real(signal_overlap)/np.real(signal_overlap_estimate)),-1+10*np.log10(1-signal_overlap_error/signal_overlap_estimate)
    # 10*np.log10(np.real(signal_overlap)/np.real(signal_overlap_estimate))<-1+10*np.log10(1-signal_overlap_error/signal_overlap_estimate)
    if db_compression_point is not None:
        measurement.metadata['compression_1db'] = str(drive_amplitudes[db_compression_point + 1])
    else:
        measurement.metadata['compression_1db'] = 'nan'

    measurement.datasets['compression'].data[:] = compression[:]
    measurement.datasets['cos_dist'].data[:] = cos_dist[:]