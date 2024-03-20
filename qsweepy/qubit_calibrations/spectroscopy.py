from qsweepy.fitters import fit_dataset, resonator_tools, spectroscopy_overview
from qsweepy.ponyfiles.data_structures import *
import numpy as np
import time
import scipy.signal as ss

def calculate_delay(measurement, _):
    x = measurement.datasets['S-parameter'].parameters[-1].values
    y = measurement.datasets['S-parameter'].data

    phase = np.angle(y)

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.apply_along_axis(lambda m: np.convolve(m, box, mode='valid'), axis=-1, arr=y)
        return y_smooth

    phase_smooth = smooth(np.unwrap(phase), box_pts=21)
    xp = np.linspace(x[0], x[-1], len(phase_smooth[-1]))

    phase_smooth = np.apply_along_axis(lambda m: np.interp(x, xp, m), axis=-1, arr=phase_smooth)
    delay = np.gradient(phase_smooth, axis=-1) / (2 * np.pi * (x[1] - x[0]))

    # phase_filtered = ss.savgol_filter(np.unwrap(phase), window_length=51, polyorder=3)
    # delay = np.gradient(phase_filtered, axis=-1) / (2 * np.pi * (x[1] - x[0]))

    # delay_total = np.fft.fftfreq(len(x), x[1] - x[0])[np.argmax(np.abs(np.fft.ifft(y.ravel())))]
    # corrected_s21 = y * np.exp(1j * 2 * np.pi * delay_total * x)
    # delay = np.gradient(np.unwrap(np.angle(y)), axis=-1) / (2 * np.pi * (x[1] - x[0]))
    measurement.datasets['delay'].data[:] = delay

def single_tone_spectroscopy_overview(device, fmin, fmax, nop, *args):
    try:
        device.hardware.set_cw_mode()
        if device.hardware.lo1 is not None:
            device.hardware.lo1.set_status(0)
        if device.hardware.lo2 is not None:
            device.hardware.lo2.set_status(0)

        power = float(device.get_sample_global(name='single_tone_spectrum_power'))
        bandwidth = float(device.get_sample_global(name='single_tone_spectrum_bandwidth'))

        device.hardware.pna.set_power(power)
        device.hardware.pna.set_nop(nop)
        device.hardware.pna.set_bandwidth(bandwidth)

        device.hardware.pna.set_xlim(fmin, fmax)
        #device.hardware.pna.measure()
        #time.sleep(device.hardware.pna.get_sweep_time()*0.001)
        device.hardware.pna.measure()

        def create_delay_dataset(measurement):
            parameters = measurement.datasets['S-parameter'].parameters
            measurement.datasets['delay'] = MeasurementDataset(parameters, np.zeros(tuple([len(p.values) for p in parameters])) * np.nan)

        result = device.sweeper.sweep(device.hardware.pna, *args,
                               measurement_type='single_tone_spectroscopy_overview',
                               metadata={ 'vna_power': device.hardware.pna.get_power(),
                                          'bandwidth': bandwidth},
                               on_start=[(create_delay_dataset, tuple())],
                               on_update=[(calculate_delay, tuple())])

        fitter = spectroscopy_overview.SingleToneSpectroscopyOverviewFitter()

        fit_dataset.fit_dataset_1d(
            source_measurement=result,
            dataset_name='S-parameter',
            fitter=fitter,
            time_parameter_id=-1,
            sweep_parameter_ids=np.arange(len(args)),
            allow_unpack_complex=False,
            use_resample_x_fit=False)

        device.exdir_db.save_measurement(result.fit)

    except:
        raise

def single_tone_spectroscopy(device, qubit_id, fr_guess, *args, span=None, power=None, nop=None, bandwidth=None, **kwargs):
    print ('fr_guess: ', fr_guess)
    try:
        device.hardware.set_cw_mode()
        if device.hardware.lo1 is not None:
            device.hardware.lo1.set_status(0)
        if device.hardware.lo2 is not None:
            device.hardware.lo2.set_status(0)

        if span is None:
            span = float(device.get_qubit_constant(qubit_id=qubit_id, name='single_tone_spectrum_span'))
        if power is None:
            power = float(device.get_qubit_constant(qubit_id=qubit_id, name='single_tone_spectrum_power'))
        if nop is None:
            nop = float(device.get_qubit_constant(qubit_id=qubit_id, name='single_tone_spectrum_nop'))
        if bandwidth is None:
            bandwidth = float(device.get_qubit_constant(qubit_id=qubit_id, name='single_tone_spectrum_bandwidth'))


        fit_type = device.get_qubit_constant(qubit_id=qubit_id, name='single_tone_spectrum_fit_type')

        device.hardware.pna.set_power(power)
        device.hardware.pna.set_nop(nop)
        device.hardware.pna.set_bandwidth(bandwidth)

        device.hardware.pna.set_xlim(fr_guess - span/2, fr_guess + span/2)
        device.hardware.pna.measure()
        time.sleep(device.hardware.pna.get_sweep_time()*0.001)
        device.hardware.pna.measure()

        metadata = {'qubit_id': qubit_id,
                    'vna_power': device.hardware.pna.get_power(),
                    'bandwidth': bandwidth}
        metadata.update(kwargs)

        def create_delay_dataset(measurement):
            parameters = measurement.datasets['S-parameter'].parameters
            measurement.datasets['delay'] = MeasurementDataset(parameters, np.zeros(
                tuple([len(p.values) for p in parameters])) * np.nan)
        result = device.sweeper.sweep(device.hardware.pna, *args,
                               measurement_type='resonator_frequency',
                               metadata=metadata,
                               on_start=[(create_delay_dataset, tuple())],
                               on_update=[(calculate_delay, tuple())])
    except:
        raise

    fitter = resonator_tools.ResonatorToolsFitter(fit_type)

    fit_dataset.fit_dataset_1d(
            source_measurement=result,
            dataset_name='S-parameter',
            fitter=fitter,
            time_parameter_id=-1,
            sweep_parameter_ids=np.arange(len(args)),
            allow_unpack_complex=False,
            use_resample_x_fit=False)

    device.exdir_db.save_measurement(result.fit)

    return result


def measure_fr(device, qubit_id, fr_guess):
    result = single_tone_spectroscopy(device, qubit_id, fr_guess)
    f_range = result.datasets['S-parameter'].parameters[0].values
    if float(result.fit.metadata['fr']) < min(f_range) or float(result.fit.metadata['fr']) > max(f_range):
        result.fit.metadata['fr'] = fr_guess

    device.exdir_db.db.update_in_database(result.fit)

    return result.fit


def two_tone_spectroscopy(device, qubit_id, fq_guess, *args, power_excite=None, power_readout=None, span=None, nop=None, bandwidth=None, lo=None, **kwargs):
    try:
        device.hardware.set_cw_mode()
        # device.hardware.lo1.set_status(1)
        lo.set_status(1)

        if span is None:
            span = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_span'))
        if power_excite is None:
            power_excite = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_excite_power'))
        if power_readout is None:
            power_readout = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_readout_power'))
        if nop is None:
            nop = int(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_nop'))
        if bandwidth is None:
            bandwidth = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_bandwidth'))
        fr = float(device.get_qubit_fr(qubit_id=qubit_id))


        #fit_type = device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_fit_type')

        # device.hardware.lo1.set_power(power_excite)
        lo.set_power(power_excite)
        device.hardware.pna.set_power(power_readout)
        device.hardware.pna.set_xlim(fr, fr)
        device.hardware.pna.set_nop(1)
        device.hardware.pna.set_bandwidth(bandwidth)

        device.hardware.pna.measure()
        time.sleep(device.hardware.pna.get_sweep_time()*0.001)
        device.hardware.pna.measure()
        frequencies = np.linspace(fq_guess-span/2, fq_guess+span/2, nop)

        metadata = {'qubit_id': qubit_id, 'pna_power': device.hardware.pna.get_power(),
                                      'resonator_frequency': fr, 'excitation_power': lo.get_power()}
        metadata.update(kwargs)
        # result = device.sweeper.sweep(device.hardware.pna,
        #                     *args,
        #                     (frequencies, device.hardware.lo1.set_frequency, 'excitation_frequency'),
        #                     measurement_type='qubit_frequency',
        #                     metadata=metadata)
        result = device.sweeper.sweep(device.hardware.pna,
                                      *args,
                                      (frequencies, lo.set_frequency, 'excitation_frequency'),
                                      measurement_type='qubit_frequency',
                                      metadata=metadata)
    except:
        raise


    max_id = np.argmax(np.abs(result.datasets['S-parameter'].data-np.median(result.datasets['S-parameter'].data))**2)
    result.metadata['fq'] = result.datasets['S-parameter'].parameters[0].values[max_id%len(result.datasets['S-parameter'].parameters[0].values)]

    return result

def two_tone_spectroscopy_awg(device, qubit_id, fq_guess, *args, amp_excite=None, power_readout=None, span=None, nop=None, bandwidth=None, z_port=0, osc = 0, **kwargs,):
    try:
        device.hardware.set_cw_mode()
        # device.hardware.lo1.set_status(1)
        if span is None:
            span = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_span'))
        # if amp_excite is None:
        #     amp_excite = float(device.get_qubit_constant(qubit_id=qubit_id, name=''))
        if power_readout is None:
            power_readout = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_readout_power'))
        if nop is None:
            nop = int(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_nop'))
        if bandwidth is None:
            bandwidth = float(device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_bandwidth'))
        fr = float(device.get_qubit_fr(qubit_id=qubit_id))


        #fit_type = device.get_qubit_constant(qubit_id=qubit_id, name='two_tone_spectrum_fit_type')

        # device.hardware.lo1.set_power(power_excite)
        device.hardware.q3z.awg.set_sin_amplitude(z_port,0,amp_excite)
        device.hardware.q3z.awg.set_sin_enable(z_port, 0, 1)
        device.hardware.pna.set_power(power_readout)
        device.hardware.pna.set_xlim(fr, fr)
        device.hardware.pna.set_nop(1)
        device.hardware.pna.set_bandwidth(bandwidth)

        device.hardware.pna.measure()
        time.sleep(device.hardware.pna.get_sweep_time()*0.001)
        device.hardware.pna.measure()
        print(fq_guess)
        frequencies = np.linspace(fq_guess-span/2, fq_guess+span/2, nop)

        metadata = {'qubit_id': qubit_id, 'pna_power': device.hardware.pna.get_power(),
                                      'resonator_frequency': fr}
        metadata.update(kwargs)
        def set_frequency(fr):
            device.modem.awg.set_frequency(osc,fr)
        result = device.sweeper.sweep(device.hardware.pna,
                            *args,
                            (frequencies, set_frequency, 'excitation_frequency'),
                            measurement_type='qubit_frequency',
                            metadata=metadata)
    except:
        raise


    max_id = np.argmax(np.abs(result.datasets['S-parameter'].data-np.median(result.datasets['S-parameter'].data))**2)
    result.metadata['fq'] = result.datasets['S-parameter'].parameters[0].values[max_id%len(result.datasets['S-parameter'].parameters[0].values)]

    return result