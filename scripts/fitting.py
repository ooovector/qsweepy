import numpy as np
import matplotlib.pyplot as plt

def exp_fit (x, y):
	def model(x,p):
		x0=p[0]
		A = np.reshape(np.asarray(p[1:]),(len(p[1:]), 1))
		return A*np.exp(-x/x0)
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()
	
	y = np.asarray(y)
	y_last = np.reshape(y[:,-1], (y.shape[0], 1))
	y = y - y_last
	
	integral = np.sum(y,axis=1)*(x[1]-x[0])
	y_first = y[:,0]	
	x0=np.sqrt(np.sum(np.abs(integral)**2)/np.sum(np.abs(y_first)**2))
	
	p0 = [x0]+y_first.tolist()
	
	from scipy.optimize import leastsq
	fitresults = leastsq (cost, p0)
	fitted_curve = model(x, fitresults[0])
	
	for i in range(4):
		plt.figure(i)
		plt.plot(x, y[i,:], label='data')
		plt.plot(x, model(x, p0)[i,:], label='initial')
		plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
		plt.legend()
	
	return fitresults[0], fitted_curve+y_last
	

def exp_sin_fit(x, y):
	# фитует результаты измерений экспонентой
	def model(x, p):
		phase = p[0]
		freq = p[1]
		x0 = p[2]
		A = np.reshape(np.asarray(p[3:]),(len(p[3:]), 1))
		return A*np.cos(phase+x*freq*2*np.pi)*np.exp(-x/x0)
	
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()
	
	means = np.reshape(np.mean(y, axis=1), (np.asarray(y).shape[0], 1))
	y = y-means
	
	ft = np.fft.fft(y-np.reshape(np.mean(y, axis=1), (y.shape[0], 1)), axis=1)/len(x)
	f = np.fft.fftfreq(len(x), x[1]-x[0])
	domega = (f[1]-f[0])*2*np.pi
	
	fR_id = np.argmax(np.sum(np.abs(ft)**2, axis=0))
	fR_id_conj = len(f)-fR_id
	if fR_id_conj > fR_id:
		tmp = fR_id_conj
		fR_id_conj = fR_id
		fR_id = tmp
	
	fR = np.abs((f[fR_id]))
	
	c = np.real(np.sum(ft[:,fR_id], axis=0))
	s = np.imag(np.sum(ft[:,fR_id], axis=0))
	phase = np.arctan2(s, c)
	x0 = np.sqrt(np.mean(np.abs(ft[:,fR_id])**2)/np.mean(np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)-1)/domega/2
	print (x0, np.abs(ft[:,fR_id])**2, np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)
	
	A = np.sqrt(np.abs(ft[:,fR_id])**2+np.abs(ft[:,fR_id_conj])**2)
	p0 = [phase, fR, x0]+A.tolist()
	
	from scipy.optimize import leastsq
	fitresults = leastsq (cost, p0)
	fitted_curve = model(x, fitresults[0])
	
	for i in range(4):
		plt.figure(i)
		plt.plot(x, y[i,:], label='data')
		plt.plot(x, model(x, p0)[i,:], label='initial')
		plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
		plt.legend()
	
	return fitresults[0], fitted_curve+means
	
def S21pm_fit(measurement, fitter):
	t = measurement['S21+'][1][0]
	fitdata = [ np.real(measurement['S21+'][2]), 
				np.imag(measurement['S21+'][2]), 
				np.real(measurement['S21-'][2]), 
				np.imag(measurement['S21-'][2])]
	parameters, fitted_curve = fitter(t, fitdata)
	if fitter is exp_sin_fit:
		parameters = {'phase':parameters[0], 'freq':parameters[1], 'decay': parameters[2], 'amplitudes':parameters[3:]}
	elif fitter is exp_fit:
		parameters = {'decay': parameters[0], 'amplitudes':parameters[1:]}
	
	measurement['S21+ fit'] = [t for t in measurement['S21+']]
	measurement['S21- fit'] = [t for t in measurement['S21-']]
	measurement['S21+ fit'][2] = fitted_curve[0,:]+fitted_curve[1,:]*1j
	measurement['S21- fit'][2] = fitted_curve[2,:]+fitted_curve[3,:]*1j
	measurement['S21+ fit'] = tuple(measurement['S21+ fit'])
	measurement['S21- fit'] = tuple(measurement['S21- fit'])
	
	return measurement, parameters
	

	
