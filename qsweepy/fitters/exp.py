import numpy as np
from . import fit_dataset
import traceback


class exp_fitter:
    def __init__(self):
        self.name = 'exp_fit'

    def fit(self, x, y, parameters_old=None):
        return exp_fit(x, y)


def exp_fit(x, y):
    def model(x, p):
        x0 = p[0]
        b = p[1]
        AB = np.reshape(p[1:], (-1, 1))
        A = np.reshape(AB[:int(len(AB) / 2)], (int(len(AB) / 2), 1))
        B = np.reshape(AB[-int(len(AB) / 2):], (int(len(AB) / 2), 1))
        return A * (np.exp(-x / x0)) + B

    y = np.asarray(y)
    nonnan_x = x[np.all(np.isfinite(y), axis=0)]
    nonnan_y = y[:, np.all(np.isfinite(y), axis=0)]
    residuals = lambda p: (model(nonnan_x, p) - nonnan_y).ravel()
    cost = lambda p: np.abs(residuals(p)) ** 2

    if len(nonnan_x) == 0:
        p0 = [np.nan] * (1 + y.shape[0] * 2)
    else:
        integral = np.sum(nonnan_y, axis=1) * (min(nonnan_x) - max(nonnan_x)) / len(nonnan_x)
        y_first = nonnan_y[:, 0]
        y_last = nonnan_y[:, -1]
        x0 = np.sqrt(np.sum(np.abs(integral) ** 2) / np.sum(np.abs(y_first) ** 2))

        p0 = [x0] + y_first.tolist() + y_last.tolist()

    from scipy.optimize import leastsq
    try:
        def errfit(hess_inv, res_variance):
            return np.sqrt(np.diag(hess_inv * res_variance))

        fitresults, hess_inv, infodict, errmsg, success = leastsq(residuals, p0, full_output=True)
        error_estimates = errfit(hess_inv, (residuals(fitresults) ** 2).sum() / np.abs(len(nonnan_y) - len(p0)))
        # fitresults_full = leastsq(residuals, p0, full_output=True)
        # fitresults = fitresults_full[0]


        # mse = np.sum(cost(fitresults)) / (len(nonnan_x) - len(p0))
        # error_estimates = np.sqrt(np.diag(mse * fitresults_full[1]))
    except Exception as e:
        fitresults = p0
        error_estimates = [np.nan for i in p0]

    fitted_curve = model(fit_dataset.resample_x_fit(x), fitresults)
    MSE_rel = np.mean(cost(fitresults)) / np.mean(np.abs(nonnan_y - np.mean(nonnan_y)) ** 2)

    parameters = {'decay': fitresults[0],
                  'A': fitresults[1:-y.shape[0]],
                  'inf': fitresults[-y.shape[0]:],
                  'MSE_rel': MSE_rel,
                  }
    if error_estimates is not None:
        parameters.update({'decay_error': error_estimates[0],
                           'A_error': error_estimates[1:-y.shape[0]],
                           'inf_error': error_estimates[-y.shape[0]:]})

    return fit_dataset.resample_x_fit(x), fitted_curve, parameters
