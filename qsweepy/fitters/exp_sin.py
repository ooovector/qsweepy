import numpy as np
from . import fit_dataset
import traceback

class exp_sin_fitter:
    def __init__(self, mode='sync'):
        self.name = 'exp_sin_fit'
        self.mode = mode

    def fit(self,x,y, parameters_old=None):
        return exp_sin_fit(x, y, parameters_old, self.mode)

def exp_sin_fit(x, y, parameters_old=None, mode='sync'):
    y = np.asarray(y)
    if np.any(np.logical_not(np.isfinite(y))):
        first_nan = np.argmax(np.any(np.logical_not(np.isfinite(y)), axis=0))
    else:
        first_nan = len(x)
    x_full = x.ravel()
    y_full = y
    x = x[:first_nan]
    y = y[:, :first_nan]
    try:
        if len(x) < 7:
            raise IndexError

        # defining model for fit
        def model(x, p):
            phase = p[0]
            freq = p[1]
            x0 = p[2]
            if mode == 'sync':
                inf = p[3]
                A = np.reshape(np.asarray(p[4:]), (len(p[4:]), 1))
            elif mode == 'unsync':
                A_inf = p[3:]
                inf = np.reshape(A_inf[:len(A_inf)//2], (len(A_inf)//2, 1))
                A = np.reshape(A_inf[len(A_inf)//2:], (len(A_inf)//2, 1))

            return A*(-np.cos(phase+x*freq*2*np.pi)*np.exp(-x/x0)+inf)

        # estimating frequency and amplitude from fourier-domain (for exp_sin and sin)
        ft = np.fft.fft(y, axis=1)/len(x)
        f = np.fft.fftfreq(len(x), x[1]-x[0])
        domega = (f[1]-f[0])*2*np.pi

        ft_nomean = ft.copy()
        ft_nomean[:, 0] = 0

        fR_id = np.argmax(np.sum(np.abs(ft_nomean)**2, axis=0))
        fR_id_conj = len(f)-fR_id

        if fR_id_conj > fR_id:
            tmp = fR_id_conj
            fR_id_conj = fR_id
            fR_id = tmp

        fR = np.abs((f[fR_id]))

        c = np.real(np.sum(ft[:,fR_id], axis=0))
        s = np.imag(np.sum(ft[:,fR_id], axis=0))
        phase = np.arctan2(c, s)#np.pi+
        A = np.sqrt(np.abs(ft[:,fR_id])**2+np.abs(ft[:,fR_id_conj])**2)*2

        #estimating decay rate from fourier-domain
        try:
            T = np.sqrt(np.mean(np.abs(ft[:,fR_id])**2)/np.mean(np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)-1)/domega/2
        except:
            #traceback.print_exc()
            T = np.nanmax(x_full) - np.nanmin(x_full)
        #estimating asymptotics
        if mode == 'sync':
            inf = np.sqrt(np.sum(np.abs(ft[:,0])**2)/np.sum(A**2))
            p0 = [phase, fR, T, inf] + np.asarray(A).tolist()
            parameters_flat = lambda parameters: [parameters['phi'], parameters['f'], parameters['T'],
                                                  parameters['inf']] + parameters['A'].tolist()
        elif mode == 'unsync':
            inf = np.real(ft[:,0]/ft[:,fR_id])
            p0 = [phase, fR, T] + np.asarray(inf).tolist() + np.asarray(A).tolist()
            parameters_flat = lambda parameters: [parameters['phi'], parameters['f'], parameters['T']] + \
                                                  np.asarray(parameters['inf']).tolist() + np.asarray(parameters['A']).tolist()

        #fitting with leastsq

        from scipy.optimize import leastsq
        cost_mod = lambda p: (np.abs(model(x, p)-y)**2/np.log(np.sum(np.std(model(x, p)**2, axis=1)))).ravel()
        residuals = lambda p: (model(x, p) - y).ravel()
        cost = lambda p: np.abs(residuals(p))**2
        fitresults = leastsq (cost_mod, p0, maxfev=200)
        # fitresults_full = leastsq(residuals, fitresults[0], full_output=True)
        #print (np.std(model(x, p0))**2, axis=1)
        if mode == 'sync':
            parameters_new = {'phi':fitresults[0][0],
                              'f':fitresults[0][1],
                              'T': fitresults[0][2],
                              'inf': fitresults[0][3],
                              'A': fitresults[0][4:]
                              }
        elif mode == 'unsync':
            A_inf = fitresults[0][3:]
            parameters_new = {'phi': fitresults[0][0],
                              'f': fitresults[0][1],
                              'T': fitresults[0][2],
                              'inf': A_inf[:len(A_inf)//2],
                              'A': A_inf[len(A_inf)//2:]}

        mse = np.sum(cost(fitresults[0])) / (len(x) - len(p0))
        MSE_rel_calculator = lambda parameters: np.sum(cost(parameters_flat(parameters)))/np.sum(np.abs(y.T-np.mean(y, axis=1))**2)

        MSE_rel_new = MSE_rel_calculator(parameters_new)
        #print('parameters_new:', parameters_new)
        if not parameters_old:
            MSE_rel = MSE_rel_calculator(parameters_new)
            parameters = parameters_new
        else:
            if mode == 'unsync':
                parameters_old = {k: np.asarray(v)[0] if (k != 'A' and k != 'inf') else v for k,v in parameters_old.items()}
            else:
                parameters_old = {k: np.asarray(v)[0] if (k != 'A') else v for k,v in parameters_old.items()}

            #print ('parameters_old:', parameters_old)
            MSE_rel_old = MSE_rel_calculator(parameters_old)
            parameters = parameters_old if MSE_rel_new > MSE_rel_old else parameters_new
            #print ('MSE_rel_new:', MSE_rel_new, 'MSE_rel_old:', MSE_rel_old)
            MSE_rel = MSE_rel_old if MSE_rel_new > MSE_rel_old else MSE_rel_new

        if parameters['f']  < 0:
            parameters['f'] = -parameters['f']
            parameters['phi'] = -parameters['phi']
        if parameters['T'] < 0:
            parameters['T'] = np.inf
        if mode == 'sync':
            if parameters['inf']<-np.sqrt(MSE_rel):
                parameters['inf'] = -parameters['inf']
                parameters['A'] = -parameters['A']
                parameters['phi'] = parameters['phi']+np.pi

        parameters['phi'] -= np.floor(parameters['phi']/(2*np.pi)+1.)*2*np.pi

        p0 = parameters_flat(parameters)
        fitresults = leastsq(cost, p0, maxfev=200)

        def errfit(hess_inv, res_variance):
            return np.sqrt(np.diag(hess_inv * res_variance))
        fit, hess_inv, infodict, errmsg, success = leastsq(residuals, p0, full_output=True)
        error_estimates = errfit(hess_inv, (residuals(fit) ** 2).sum() / np.abs(len(y) - len(p0)))
        if mode == 'sync':
            parameters = {'phi':fitresults[0][0],
                          'f':fitresults[0][1],
                          'T': fitresults[0][2],
                          'inf': fitresults[0][3],
                          'A': fitresults[0][4:]}
            parameters.update({'phi_error': error_estimates[0],
                               'f_error': error_estimates[1],
                               'T_error': error_estimates[2],
                               'inf_error': error_estimates[3],
                               'A_error': error_estimates[4:]})
            # hessian = fitresults_full[1]
            #
            # if hessian is None:
            #     hessian = np.nan*np.ones(len(fitresults[0]))
            # error_estimates = np.sqrt(np.diag(mse * hessian))
            # parameters.update({'phi_error': error_estimates[0],
            #                    'f_error': error_estimates[1],
            #                    'T_error': error_estimates[2],
            #                    'inf_error': error_estimates[3],
            #                    'A_error': error_estimates[4:]})
        elif mode == 'unsync':
            A_inf = fitresults[0][3:]
            parameters = {'phi': fitresults[0][0],
                          'f': fitresults[0][1],
                          'T': fitresults[0][2],
                          'inf': A_inf[:len(A_inf)//2],
                          'A': A_inf[len(A_inf)//2:]}
            A_inf_errors = error_estimates[3:]
            parameters.update({'phi_error': error_estimates[0],
                               'f_error': error_estimates[1],
                               'T_error': error_estimates[2],
                               'inf_error': A_inf_errors[:len(A_inf) // 2],
                               'A_error': A_inf_errors[len(A_inf) // 2:]})

            # hessian = fitresults_full[1]
            #
            # if hessian is None:
            #     hessian = np.nan*np.ones(len(fitresults[0]))
            # error_estimates = np.sqrt(np.diag(mse * hessian))
            #
            # A_inf_errors = error_estimates[3:]
            # # TODO: check these two variants
            # # parameters.update({'phi_error': error_estimates[0],
            # #                    'f_error': error_estimates[1],
            # #                    'T_error': error_estimates[2],
            # #                    'inf_error': A_inf_errors[:len(A_inf) // 2],
            # #                    'A_error': A_inf_errors[len(A_inf) // 2:]})
            # parameters.update({'phi_error': error_estimates[0],
            #                    'f_error': error_estimates[1],
            #                    'T_error': error_estimates[2],
            #                    'inf_error': error_estimates[3],
            #                    'A_error': error_estimates[4:]})

        #sampling fitted curve
        fitted_curve = model(fit_dataset.resample_x_fit(x_full), parameters_flat(parameters))
        #calculating MSE in relative relative units
        MSE_rel = MSE_rel_calculator(parameters)

        parameters['MSE_rel'] = MSE_rel
        parameters['num_periods_decay'] = parameters['T']*parameters['f']
        parameters['num_periods_scan'] = (np.max(x)-np.min(x))*parameters['f']
        parameters['points_per_period'] = 1/((x[1]-x[0])*parameters['f'])
        parameters['decays_in_scan_length'] = (np.max(x)-np.min(x))/parameters['T']
    except IndexError as e:
        traceback.print_exc()
        fitted_curve = np.zeros((y_full.shape[0], len(fit_dataset.resample_x_fit(x_full))))*np.nan
        MSE_rel = np.nan
        if mode == 'sync':
            parameters = {'phi': np.nan,
                          'f': np.nan,
                          'T': np.nan,
                          'inf': np.nan,
                          'A': np.asarray([np.nan]*y_full.shape[0]),
                          'phi_error': np.nan,
                          'f_error': np.nan,
                          'T_error': np.nan,
                          'inf_error': np.nan,
                          'A_error': np.asarray([np.nan]*y_full.shape[0])}
        elif mode == 'unsync':
            parameters = {'phi': np.nan,
                          'f': np.nan,
                          'T': np.nan,
                          'inf': np.asarray([np.nan] * y_full.shape[0]),
                          'A': np.asarray([np.nan] * y_full.shape[0]),
                          'phi_error': np.nan,
                          'f_error': np.nan,
                          'T_error': np.nan,
                          'inf_error': np.asarray([np.nan] * y_full.shape[0]),
                          'A_error': np.asarray([np.nan]*y_full.shape[0])}

        parameters['MSE_rel'] = np.nan
        parameters['num_periods_decay'] = np.nan
        parameters['num_periods_scan'] = np.nan
        parameters['points_per_period'] = np.nan
        parameters['decays_in_scan_length'] = np.nan

    frequency_goodness_test = MSE_rel<0.35 and parameters['num_periods_decay']>1.2 and parameters['num_periods_scan']>1.5 and parameters['points_per_period']>4.
    decay_goodness_test = parameters['decays_in_scan_length']>0.75 and frequency_goodness_test and np.isfinite(parameters['T'])
    parameters['frequency_goodness_test'] = 1 if frequency_goodness_test else 0
    # parameters['frequency_goodness_test'] = 1
    parameters['decay_goodness_test'] = 1 if decay_goodness_test else 0
    # print ('parameters:', parameters)

    return fit_dataset.resample_x_fit(x_full), fitted_curve, parameters