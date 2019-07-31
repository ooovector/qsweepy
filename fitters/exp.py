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
        x0=p[0]
        AB = p[1:]
        A = np.reshape(AB[:int(len(AB)/2)],  (int(len(AB)/2), 1))
        B = np.reshape(AB[-int(len(AB)/2):], (int(len(AB)/2), 1))
        return A*np.exp(-x/x0)+B
    cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()

    y = np.asarray(y)

    integral = np.sum(y, axis=1)*(min(x)-max(x))/len(x)
    y_first = y[:,0]
    y_last = y[:, -1]
    x0 = np.sqrt(np.sum(np.abs(integral)**2)/np.sum(np.abs(y_first)**2))

    p0 = [x0]+y_first.tolist()+y_last.tolist()
    #print (p0)

    from scipy.optimize import leastsq
    fitresults = leastsq (cost, p0)
    fitted_curve = model(fit_dataset.resample_x_fit(x), fitresults[0])
    MSE_rel = cost(fitresults[0])

    parameters = {'decay': fitresults[0][0], 'A':fitresults[0][1:-y.shape[0]], 'B':fitresults[0][-y.shape[0]:], 'MSE_rel': MSE_rel}
    return fit_dataset.resample_x_fit(x), fitted_curve, parameters
