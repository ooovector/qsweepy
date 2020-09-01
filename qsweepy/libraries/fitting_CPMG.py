import numpy as np
import matplotlib.pyplot as plt

def gaussian_CPMG_predefined(x, y, gamma1, gammap, tayp, gammaphi):
	def model(x, p):
		A = p[0]
		B = p[1]
		return A*np.exp(-x*gamma1/2)*np.exp(-gammap*tayp)*np.exp(-(gammaphi*x)**2)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	p0 = []
	p0.append(y_first)
	p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
def gaussian_CPMG_exp(x, y):
	def model(x, p):
		gamma = p[0]
		A = p[1]
		B = p[2]
		return A*np.exp(-x*gamma/2)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	gamma = 10e06/2
	p0 = []
	p0.append(gamma)
	p0.append(y_first)
	p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
	
def gaussian_CPMG(x, y, gamma1, gammap, tayp):
	def model(x, p):
		gammaphi = p[0]
		A = p[1]
		B = p[2]
		return A*np.exp(-x*gamma1/2)*np.exp(-gammap*tayp)*np.exp(-(gammaphi*x)**2)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	gammaphi_0 = 1e5#np.sqrt(np.abs(np.log((y_first-y_last)/y_first)))/x[0]
	#print (x[0], y)
	p0 = []
	p0.append(gammaphi_0)
	p0.append(y_first)
	p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
	
def fit_CPMG_new(N, pi2_pulse, gamma1, gammap, tayp, t, data, fitter):
	fitdata = data
	parameters, fitted_curve = fitter(t, fitdata, gamma1, gammap, tayp)
	return fitted_curve, parameters
	
def fit_CPMG(N, pi2_pulse, gamma1, gammap, tayp, data, fitter):
	t = (np.asarray(data[1]['z'][1])[0])# + N*pi2_pulse)[0]
	fitdata = data[1]['z'][2]
	parameters, fitted_curve = fitter(t, fitdata, gamma1, gammap, tayp)
	#parameters = {'gamma': parameters[0], 'amplitudes':parameters[1:]}
	
	#measurement['fit'] = [t for t in measurement[1]['z'][1]]
	#measurement['fit'] = fitted_curve[:]
	
	#if len(measurement[1]['z'])>3:
	#	measurement['fit'][3] = dict(measurement['fit'][3])
	#	if 'scatter' in measurement['S21+ fit'][3]:
	#		del measurement['S21+ fit'][3]['scatter']
	#if len(measurement['S21- fit'])>3:
	#	measurement['S21- fit'][3] = dict(measurement['S21- fit'][3])
	#	if 'scatter' in measurement['S21- fit'][3]:
	#		del measurement['S21- fit'][3]['scatter']
			
	#measurement['S21+ fit'] = tuple(measurement['S21+ fit'])
	#measurement['S21- fit'] = tuple(measurement['S21- fit'])
	#measurement['fit'] = tuple(measurement['fit'])
	return fitted_curve, parameters
	