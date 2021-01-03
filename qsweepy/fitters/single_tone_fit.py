import numpy as np
from qsweepy.libraries.qubit_device import QubitDevice
from typing import List
from matplotlib import pyplot as plt
from math import asin
from scipy.optimize import curve_fit
from scipy.signal import detrend
from qsweepy.fitters import fit_dataset
from scipy.signal import argrelextrema


def sin_model(x, A, k, phase, offset):
    return A * np.sin(2*np.pi*k * x + phase) + offset

def cf_sin_fit(x, y):
    y_max = y[np.argmax(y)]
    x_max = x[np.argmax(y)]
    y_min = y[np.argmin(y)]
    x_min = x[np.argmin(y)]
    A_0 = (y_max-y_min) #/2
    offset = np.mean(y)
    y_1 = (y_max - offset)/A_0
    y_2 = (y_min - offset)/A_0
    k_0 = 1/np.abs(x_max-x_min)
    phi_0 = asin(y_2)-k_0*x_min
    popt, pcov = curve_fit(sin_model, x, y, p0=[A_0, k_0, phi_0, offset])
    fitted_curve = sin_model(x, *popt)
    parameters = {'A': popt[0], 'k': popt[1], 'phase': popt[2], 'offset':popt[3]}

    return fitted_curve, parameters

def ls_sin_fit(x, y, parameters_old=None, mode='sync'):
    y = np.asarray(y)
    if np.any(np.logical_not(np.isfinite(y))):
        first_nan = np.argmax(np.any(np.logical_not(np.isfinite(y)), axis=0))
    else:
        first_nan = len(x)
    x_full = x.ravel()
    y_full = y
    x = x[:first_nan]
    y = y[:first_nan]

    if len(x) < 5:
        raise IndexError

    # defining model for fit
    def model(x, p):
        phase = p[0]
        freq = p[1]
        if mode == 'sync':
            inf = p[2]
            A = np.reshape(np.asarray(p[3:]), (len(p[3:]), 1))
        elif mode == 'unsync':
            A_inf = p[2:]
            inf = np.reshape(A_inf[:len(A_inf)//2], (len(A_inf)//2, 1))
            A = np.reshape(A_inf[len(A_inf)//2:], (len(A_inf)//2, 1))

        return A*(-np.cos(phase+x*freq*2*np.pi)+inf)

    # estimating frequency and amplitude from fourier-domain (for exp_sin and sin)
    ft = np.fft.fft(y, axis=0)/len(x)
    f = np.fft.fftfreq(len(x), x[1]-x[0])
    domega = (f[1]-f[0])*2*np.pi
    ft_nomean = ft.copy()
    fR_id = 2
    fR_id_conj = -2
    fR = np.abs((f[fR_id]))
    c = np.real(np.sum(ft[fR_id], axis=0))
    s = np.imag(np.sum(ft[fR_id], axis=0))
    phase = np.arctan2(c, s)#np.pi+
    A = np.sqrt(np.abs(ft[fR_id])**2+np.abs(ft[fR_id_conj])**2)*2

    #estimating asymptotics
    if mode == 'sync':
        inf = np.sqrt(np.sum(np.abs(ft[0])**2)/np.sum(A**2))
        p0 = [phase, fR, inf, A]
        parameters_flat = lambda parameters: [parameters['phi'], parameters['f'],
                                              parameters['inf'], parameters['A']]
    elif mode == 'unsync':
        inf = np.real(ft[0]/ft[fR_id])
        p0 = [phase, fR] + np.asarray(inf).tolist() + np.asarray(A).tolist()
        parameters_flat = lambda parameters: [parameters['phi'], parameters['f']] + \
                                              np.asarray(parameters['inf']).tolist() + np.asarray(parameters['A']).tolist()

    #fitting with leastsq

    from scipy.optimize import leastsq
    cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()
    fitresults = leastsq(cost, p0, maxfev=1000)
    if mode == 'sync':
        parameters_new = {'phi':fitresults[0][0], 'f':fitresults[0][1], 'inf': fitresults[0][2], 'A': fitresults[0][3:]}
    elif mode == 'unsync':
        A_inf = fitresults[0][2:]
        parameters_new = {'phi': fitresults[0][0], 'f': fitresults[0][1],
                          'inf': A_inf[:len(A_inf)//2], 'A': A_inf[len(A_inf)//2:]}

    MSE_rel_calculator = lambda parameters: np.sum(cost(parameters_flat(parameters)))/np.sum(np.abs(y.T-np.mean(y, axis=0))**2)
    MSE_rel_new = MSE_rel_calculator(parameters_new)
    #print('parameters_new:', parameters_new)
    if not parameters_old:
        MSE_rel = MSE_rel_calculator(parameters_new)
        parameters = parameters_new
    else:
        parameters_old = {k: np.asarray(v)[0] if (k != 'A' and k != 'inf') else v for k,v in parameters_old.items()}
        #print ('parameters_old:', parameters_old)
        MSE_rel_old = MSE_rel_calculator(parameters_old)
        parameters = parameters_old if MSE_rel_new > MSE_rel_old else parameters_new
        #print ('MSE_rel_new:', MSE_rel_new, 'MSE_rel_old:', MSE_rel_old)
        MSE_rel = MSE_rel_old if MSE_rel_new > MSE_rel_old else MSE_rel_new

    if parameters['f']  < 0:
        parameters['f'] = -parameters['f']
        parameters['phi'] = -parameters['phi']
    if mode == 'sync':
        if parameters['inf']<-np.sqrt(MSE_rel):
            parameters['inf'] = -parameters['inf']
            parameters['A'] = -parameters['A']
            parameters['phi'] = parameters['phi']+np.pi

    parameters['phi'] -= np.floor(parameters['phi']/(2*np.pi)+1.)*2*np.pi

    #sampling fitted curve
    fitted_curve = model(x, parameters_flat(parameters))
    #calculating MSE in relative relative units
    MSE_rel = MSE_rel_calculator(parameters)

    parameters['MSE_rel'] = MSE_rel
    parameters['num_periods_scan'] = (np.max(x)-np.min(x))*parameters['f']
    parameters['points_per_period'] = 1/((x[1]-x[0])*parameters['f'])
    print('parameters:', parameters)
#
    return fitted_curve[0], parameters

class single_tone_fit:

    def __init__(self, device: QubitDevice, select_type=None, id=None, freqs_extract_type='delay'):
        if select_type == 'id':
            measurement = device.exdir_db.select_measurement_by_id(int(id))
        else:
            measurement = device.exdir_db.select_measurement(measurement_type='resonator_frequency')
        self.measurement = measurement
        self.device = device
        self.model_params = None
        self.metadata = None
        self.freqs_extract_type = freqs_extract_type


    def find_res_points(self) -> List[float]:
        """
        Find peaks corresponding to notch-port connected resonator on a single-tone scan with flux dependence
        :param device: QubitDevice instance, select_type: measurement select type,
        id: id of the single-tone measurement if select type is 'id'
        :return: list[float] of resonator frequencies and list[float] of coil currents
        """

        peak_curs = self.measurement.datasets['S-parameter'].parameters[0].values
        peak_freqs = []
        # delay_threshold = float(self.device.get_sample_global(
        #     'single_tone_spectroscopy_overview_fitter_delay_threshold'))
        # min_peak_width_nop = int(self.device.get_sample_global(
        #     'single_tone_spectroscopy_overview_fitter_min_peak_width_nop'))

        for idx, value in enumerate(peak_curs):
            y = self.measurement.datasets['S-parameter'].data[idx]
            x = self.measurement.datasets['S-parameter'].parameters[1].values
            delay_total = np.fft.fftfreq(len(x), x[1] - x[0])[np.argmax(np.abs(np.fft.ifft(y.ravel())))]
            corrected_s21 = y * np.exp(1j * 2 * np.pi * delay_total * x)

            if self.freqs_extract_type == 'delay':
                phase = np.unwrap(np.angle(corrected_s21))
                delay = np.gradient(phase) / (2 * np.pi * (x[1] - x[0]))
                data_diff = np.abs(delay)
                peak_idx = np.argmax(data_diff)
                peak_freq = x[peak_idx]
            elif self.freqs_extract_type == 'amplitude':
                peak_idx = np.argmin(np.abs(corrected_s21))
                peak_freq = x[peak_idx]
            else:
                phase = np.unwrap(np.angle(corrected_s21))
                peak_idx = np.argmax(detrend(phase))
                peak_freq = x[peak_idx]

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
        cur_points,freq_points = self.find_res_points()
        res_fit = cf_sin_fit(cur_points, freq_points)
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
