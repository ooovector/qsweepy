import numpy as np
from . import fit_dataset
import traceback

class SinglePeriodSinFitter:
    def __init__(self, mode='sync'):
        self.name = 'single_period_sin_fit'
        self.mode = mode

    def fit(self,x,y, parameters_old=None):
        return single_period_sin_fit(x, y, parameters_old, self.mode)


def single_period_sin_fit(x, y, parameters_old=None, mode='sync'):
    y = np.asarray(y)
    if np.any(np.logical_not(np.isfinite(y))):
        first_nan = np.argmax(np.any(np.logical_not(np.isfinite(y)), axis=0))
    else:
        first_nan = len(x)
    x_full = x.ravel()
    y_full = y
    x_nonans = x_full[:first_nan]
    y_nonans = y_full[:, :first_nan]
    y_zeronans = y_full.copy()
    y_zeronans[np.logical_not(np.isfinite(y_zeronans))] = 0
    try:
        if len(x_nonans) < 2:
            raise IndexError

        # defining model for fit
        def model(x, p):
            phase = p[0]
            if mode == 'sync':
                inf = p[1]
                A = np.reshape(np.asarray(p[2:]), (len(p[2:]), 1))
            elif mode == 'unsync':
                A_inf = p[1:]
                inf = np.reshape(A_inf[:len(A_inf)//2], (len(A_inf)//2, 1))
                A = np.reshape(A_inf[len(A_inf)//2:], (len(A_inf)//2, 1))

            return A*(-np.cos(phase+x)+inf)

        # estimating frequency and amplitude from fourier-domain (for exp_sin and sin)
        ft = np.fft.fft(y_zeronans, axis=1)/len(x)
        fR_id = 2
        fR_id_conj = -2

        c = np.real(np.sum(ft[:,fR_id], axis=0))
        s = np.imag(np.sum(ft[:,fR_id], axis=0))
        phase = np.arctan2(c, s)#np.pi+
        A = np.sqrt(np.abs(ft[:,fR_id])**2+np.abs(ft[:,fR_id_conj])**2)*2

        if mode == 'sync':
            inf = np.sqrt(np.sum(np.abs(ft[:,0])**2)/np.sum(A**2))
            p0 = [phase, inf] + A.tolist()
            parameters_flat = lambda parameters: [parameters['phi'], parameters['inf']] + parameters['A'].tolist()
        elif mode == 'unsync':
            inf = np.real(ft[:,0]/ft[:,fR_id])
            p0 = [phase] + inf.tolist() + A.tolist()
            parameters_flat = lambda parameters: [parameters['phi']] + parameters['inf'].tolist() + parameters['A'].tolist()

        #fitting with leastsq

        from scipy.optimize import leastsq
        cost = lambda p: (np.abs(model(x_nonans, p)-y_nonans)**2).ravel()
        fitresults = leastsq (cost, p0, maxfev=200)
        if mode == 'sync':
            parameters_new = {'phi':fitresults[0][0], 'inf': fitresults[0][1], 'A': fitresults[0][2:]}
        elif mode == 'unsync':
            A_inf = fitresults[0][1:]
            parameters_new = {'phi': fitresults[0][0], 'inf': A_inf[:len(A_inf)//2], 'A': A_inf[len(A_inf)//2:]}


        MSE_rel_calculator = lambda parameters: np.sum(cost(parameters_flat(parameters)))/np.sum(np.abs(y_nonans-np.mean(y_nonans))**2)

        MSE_rel_new = MSE_rel_calculator(parameters_new)
        if not parameters_old:
            MSE_rel = MSE_rel_calculator(parameters_new)
            parameters = parameters_new
        else:
            MSE_rel_old = MSE_rel_calculator(parameters_old)
            parameters = parameters_old if MSE_rel_new > MSE_rel_old else parameters_new
            MSE_rel = MSE_rel_old if MSE_rel_new > MSE_rel_old else MSE_rel_new

        if mode == 'sync':
            if parameters['A']<0:
                parameters['inf'] = -parameters['inf']
                parameters['A'] = -parameters['A']
                parameters['phi'] = parameters['phi']+np.pi

        parameters['phi'] -= np.floor(parameters['phi']/(2*np.pi)+1.)*2*np.pi

        #sampling fitted curve
        fitted_curve = model(fit_dataset.resample_x_fit(x_full), parameters_flat(parameters))
        #calculating MSE in relative relative units
        MSE_rel = MSE_rel_calculator(parameters)

        parameters['MSE_rel'] = MSE_rel
    except IndexError as e:
        traceback.print_exc()
        fitted_curve = np.zeros((y_full.shape[0], len(fit_dataset.resample_x_fit(x_full))))*np.nan
        MSE_rel = np.nan
        if mode == 'sync':
            parameters = {'phi':np.nan, 'inf':np.nan, 'A':np.asarray([np.nan]*y_full.shape[0])}
        elif mode == 'unsync':
            parameters = {'phi': np.nan, 'inf': np.asarray([np.nan] * y_full.shape[0]),
                          'A': np.asarray([np.nan] * y_full.shape[0])}

        parameters['MSE_rel'] = np.nan

    phase_goodness_test = 3*MSE_rel<parameters['A']
    parameters['phase_goodness_test'] = 1 if phase_goodness_test else 0

    return fit_dataset.resample_x_fit(x_full), fitted_curve, parameters
