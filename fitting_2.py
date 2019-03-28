import numpy as np
import matplotlib.pyplot as plt

def resample_x_fit(x):
	if len(x) < 500:
		return np.linspace(np.min(x), np.max(x), 501)
	else:
		return x

def exp_fit (x, y):
	def model(x,p):
		x0=p[0]
		AB = p[1:]
		A = np.reshape(AB[:int(len(AB)/2)],  (int(len(AB)/2), 1))
		B = np.reshape(AB[-int(len(AB)/2):], (int(len(AB)/2), 1))
		return A*np.exp(-x/x0)+B
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1,:]#np.reshape(y[:,-1], (y.shape[0], 1))
	#y = y - y_last
	
	integral = np.nansum(y,axis=0)*(x[1]-x[0])
	y_first = y[0,:]	
	x0=np.sqrt(np.nansum(np.abs(integral)**2)/np.nansum(np.abs(y_first)**2))
	
	p0 = [x0]+y_first.tolist()+y_last.tolist()
	
	from scipy.optimize import leastsq
	fitresults = leastsq (cost, p0)
	fitted_curve = np.transpose(model(resample_x_fit(x), fitresults[0]))
	
	#for i in range(y.shape[0]):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()
	parameters = {'decay': str(fitresults[0][0]), 'amplitudes': str(fitresults[0][1:])}
	return (resample_x_fit(x), fitted_curve), parameters

def exp_sin_fit(x, y):
	# фитует результаты измерений экспонентой
	def model(x, p):
		phase = p[0]
		freq = p[1]
		x0 = p[2]
		A = np.reshape(np.asarray(p[3:]),(len(p[3:]), 1))
		return A*np.cos(phase+x*freq*2*np.pi)*np.exp(-x/x0)
	
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()
	
	#means = np.reshape(np.nanmean(y, axis=1), (np.asarray(y).shape[0], 1))
	means = np.reshape(np.nanmean(y, axis=0), (y.shape[1], 1))
	#means = np.mean(y)
	z = np.asarray(y)[:,0]
	y = y-means
	#print(y.shape)
	
	#ft = np.fft.fft(y-np.reshape(np.nanmean(y, axis=1), (y.shape[0], 1)), axis=1)/len(x)
	ft = np.fft.fft(y - np.reshape(np.nanmean(y, axis=0), (y.shape[1], 1)), axis=0)/len(x)
	f = np.fft.fftfreq(len(x), x[1]-x[0])
	domega = (f[1]-f[0])*2*np.pi
	#print('ft', ft)
	
	#plt.figure('FFT')
	#plt.plot(f, ft.T)
	
	#plt.figure('Initial fit')
	
	fR_id = np.argmax(np.nansum(np.abs(ft)**2, axis=1))
	fR_id_conj = len(f)-fR_id
	if fR_id_conj == len(f):
		fR_id_conj = 0
	if fR_id_conj > fR_id:
		tmp = fR_id_conj
		fR_id_conj = fR_id
		fR_id = tmp
	
	fR = np.abs((f[fR_id]))
	#print(ft[fR_id, :])
	c = np.real(np.nansum(ft[fR_id, :]))
	s = np.imag(np.nansum(ft[fR_id, :]))
	phase = np.pi+np.arctan2(c, s)
	if fR_id==0 or fR_id+1==len(f):
		x0 = np.max(x)-np.min(x)
	else:
		x0 = np.sqrt(np.mean(np.abs(ft[fR_id, :])**2)/np.mean(np.abs((ft[fR_id-1, :]+ft[fR_id+1, :])/2)**2)-1)/domega/2
	
	#print (x0, np.abs(ft[:,fR_id])**2, np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)
	
	A = np.sqrt(np.abs(ft[fR_id, :])**2+np.abs(ft[fR_id_conj, :])**2)*2
	p0 = [phase, fR, x0]+A.tolist()
	#print(p0)
	
	from scipy.optimize import leastsq
	#plt.plot(x, (model(x, p0)).T)
	#plt.plot(x, (y).T, marker='o', markerfacecolor='None', linestyle='none')
	fitresults = leastsq (cost, p0)
	#print(fitresults)
	fitted_curve = np.transpose(model(resample_x_fit(x), fitresults[0]))
	#print(np.asarray(fitted_curve).shape)
	
	#for i in range(4):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()
	
	parameters = {'phase': str(fitresults[0][0]), 'freq': str(fitresults[0][1]), 'decay': str(fitresults[0][2]), 'amplitudes': str(fitresults[0][3:]), 'initial_points': str(z)}
	return (resample_x_fit(x), fitted_curve+means), parameters
