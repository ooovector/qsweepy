import numpy as np
from scipy.signal import find_peaks_cwt
from scipy.signal import argrelextrema
from qsweepy.libraries.qubit_device import QubitDevice
from qsweepy.ponyfiles.data_structures import MeasurementState
from typing import List


class SingleToneSpectroscopyOverviewFitter:
    def __init__(self):
        self.name = 'single_tone_spectroscopy_overview_fitter'

    def fit(self, x, y, parameters_old=None):
        #phase = np.unwrap(-np.angle(y))
        #delay_total = (phase.ravel()[-1] - phase.ravel()[0]) / (2 * np.pi * (x[-1] - x[0]))
        delay_total = np.fft.fftfreq(len(x), x[1]-x[0])[np.argmax(np.abs(np.fft.ifft(y.ravel())))]

        corrected_s21 = y * np.exp(1j * 2 * np.pi * delay_total * x)

        parameters = {'delay_total': delay_total}

        return x, corrected_s21, parameters


def find_resonators(device: QubitDevice) -> List[float]:
    """
    Find peaks corresponding to notch-port connected resonators on a single-tone overview scan
    :param device: QubitDevice instance used to retrieve sensitivity threshold & minimum peak width
    :return: List[float] of resonator frequencies
    """
    delay_threshold = float(device.get_sample_global(
        'single_tone_spectroscopy_overview_fitter_delay_threshold'))
    min_peak_width_nop = int(device.get_sample_global(
        'single_tone_spectroscopy_overview_fitter_min_peak_width_nop'))

    measurement = device.exdir_db.select_measurement(measurement_type='single_tone_spectroscopy_overview')
    fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d',
                                             references_that={'fit_source': measurement.id})

    delay_total = float(fit.metadata['delay_total'])

    x = measurement.datasets['S-parameter'].parameters[0].values
    y = measurement.datasets['S-parameter'].data

    corrected_s21 = y * np.exp(1j * 2 * np.pi * delay_total * x)

    phase = np.unwrap(np.angle(corrected_s21))
    delay = np.gradient(phase) / (2 * np.pi * (x[1] - x[0]))

    data_diff = np.abs(delay)
    data_diff[data_diff < delay_threshold] = 0
    peaks = argrelextrema(data_diff, np.greater, order=min_peak_width_nop)[0]

    peak_freqs = x[peaks]

    return peak_freqs