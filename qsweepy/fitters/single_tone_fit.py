import numpy as np
from qsweepy.libraries.qubit_device import QubitDevice
from typing import List
from qsweepy.fitters.single_period_sin import sin_fit
from matplotlib import pyplot as plt
from qsweepy.fitters.single_period_sin import sin_model
from scipy.signal import argrelextrema

class single_tone_fit:

    def __init__(self, device: QubitDevice, select_type=None, id=None):
        if select_type == 'id':
            measurement = device.exdir_db.select_measurement_by_id(int(id))
        else:
            measurement = device.exdir_db.select_measurement(measurement_type='resonator_frequency')
        self.measurement = measurement
        self.device = device
        self.model_params = None
        self.metadata = None


    def find_res_points(self) -> List[float]:
        """
        Find peaks corresponding to notch-port connected resonator on a single-tone scan with flux dependence
        :param device: QubitDevice instance, select_type: measurement select type,
        id: id of the single-tone measurement if select type is 'id'
        :return: list[float] of resonator frequencies and list[float] of coil currents
        """

        peak_curs = self.measurement.datasets['S-parameter'].parameters[0].values
        peak_freqs = []
        delay_threshold = float(self.device.get_sample_global(
            'single_tone_spectroscopy_overview_fitter_delay_threshold'))
        min_peak_width_nop = int(self.device.get_sample_global(
            'single_tone_spectroscopy_overview_fitter_min_peak_width_nop'))

        for idx, value in enumerate(peak_curs):
            y = self.measurement.datasets['S-parameter'].data[idx]
            x = self.measurement.datasets['S-parameter'].parameters[1].values
            delay_total = np.fft.fftfreq(len(x), x[1] - x[0])[np.argmax(np.abs(np.fft.ifft(y.ravel())))]

            corrected_s21 = y * np.exp(1j * 2 * np.pi * delay_total * x)

            phase = np.unwrap(np.angle(corrected_s21))
            delay = np.gradient(phase) / (2 * np.pi * (x[1] - x[0]))

            data_diff = np.abs(delay)
            peak_idx = np.argmax(data_diff)
            peak_freq = x[peak_idx]

            #
            # data_diff[data_diff < delay_threshold] = 0
            # peaks = argrelextrema(data_diff, np.greater, order=min_peak_width_nop)[0]
            # peak_freq = x[peaks]
            #
            # while np.shape(peak_freq)[0] > 1:
            #     peaks = argrelextrema(data_diff, np.greater, order=min_peak_width_nop)[0]
            #     peak_freq = x[peaks]
            #     min_peak_width_nop = min_peak_width_nop + 5
            # peak_freq = x[peaks][0]


            peak_freqs.append(peak_freq)

        return peak_curs, peak_freqs

    def find_sin_model_params(self, plot=False):
        ##написать варианты использования методов
        cur_points,freq_points = self.find_res_points()
        res_fit = sin_fit(cur_points, freq_points)
        self.model_params = tuple(res_fit[1].values())
        if plot:
            plt.plot(cur_points, freq_points, 'ro')
            plt.plot(cur_points, sin_model(cur_points, *self.model_params))
        return self.model_params


    def save_sin_model_params(self):
        self.metadata = {'A': self.model_params[0], 'k': self.model_params[1],
         'phase': self.model_params[2], 'offset': self.model_params[3]}
        references = {'id':self.measurement.id, 'metadata':self.measurement.metadata,
                      'sample_name':self.measurement.sample_name}
        self.device.exdir_db.save(measurement_type='fr_sin_fit',metadata=self.metadata,
                             references=references)

    def get_sin_model_params(self):
        references_that = {'id': self.measurement.id, 'metadata': self.measurement.metadata,
                      'sample_name': self.measurement.sample_name}
        sin_model_params = self.device.exdir_db.select_measurement(measurement_type='fr_sin_fit',
                                                          metadata=self.metadata,
                                                          references_that=references_that)

        return sin_model_params.metadata
