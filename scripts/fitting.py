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
	
	#for i in range(4):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()
	
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
	
def dc_squid_fit_S21(data, noise_sigma=5, fit=True, method='min', plot=False):
	from scipy.optimize import leastsq
	from resonator_tools import circuit
	
	S21 = data['S-parameter'][2]
	currents = data['S-parameter'][1][0]
	frequencies = data['S-parameter'][1][1]
	
	bgremove = np.mean(S21, axis=0)
	_S21 = S21
	S21 = S21/bgremove
	if plot:
		plt.figure('background subtracted abs')
	if method=='min':
		res_freq_id_estimate = np.argmin(np.abs(S21), axis=1)
		if plot:
			plt.pcolormesh(currents, frequencies, np.abs(S21).T)
	if method=='max_dev':
		res_freq_id_estimate = np.argmax(np.abs(S21-1), axis=1)
		if plot:
			plt.pcolormesh(currents, frequencies, np.abs(S21-1).T)
	elif method=='diff_curr':
		res_freq_id_estimate = np.argmax(np.abs(np.gradient(S21)[0]), axis=1)
		if plot:
			plt.pcolormesh(currents, frequencies, np.abs(np.gradient(S21)[0]).T)
	elif method=='diff_freq':
		res_freq_id_estimate = np.argmax(np.abs(np.gradient(S21)[1]), axis=1)
		if plot:
			plt.pcolormesh(currents, frequencies, np.abs(np.gradient(S21)[1]).T)
	if plot:
		plt.colorbar()
	res_freq_estimate = frequencies[res_freq_id_estimate]
	res_freq_estimate_unreliable = np.min(np.abs(S21), axis=1)>np.mean(np.abs(S21))-noise_sigma*np.std(np.abs(S21))
	for cur_id, cur in enumerate(currents):
		if cur_id < 1 or cur_id >= currents.size-1:
			continue
		if (np.abs(res_freq_estimate[cur_id]-np.mean(res_freq_estimate[cur_id-3:cur_id+3])))>(np.max(frequencies)-np.min(frequencies))*0.1:
			res_freq_estimate_unreliable[cur_id] = True

	res_freq_estimate[res_freq_estimate_unreliable] = np.nan
	ft = lambda I,fp,I0,L,a,b: np.real(fp*(1+b)*(1/(1/np.sqrt((1-a)*np.abs(np.cos((I-I0)*L).astype(np.complex))+a)+b)))
	ft_x = lambda I,x: ft(I, x[0], x[1], x[2], x[3], x[4])
	currents_freq_estimate = currents[np.isfinite(res_freq_estimate)]
	if plot:
		plt.plot(currents_freq_estimate, res_freq_estimate[np.isfinite(res_freq_estimate)])
	cost = lambda x: ft_x(currents_freq_estimate, x)-res_freq_estimate[np.isfinite(res_freq_estimate)]
	max_freq_cur_id = np.nanargmax(res_freq_estimate)
	max_freq = np.nanmax(res_freq_estimate)
	cur_h = 3
	d2fdI2 = (res_freq_estimate[max_freq_cur_id+cur_h]+res_freq_estimate[max_freq_cur_id-cur_h]-2*res_freq_estimate[max_freq_cur_id])/\
		((currents[max_freq_cur_id]-currents[max_freq_cur_id-cur_h])**2)
	x0 = [max_freq, currents[np.nanargmax(res_freq_estimate)], np.sqrt(np.abs(d2fdI2/max_freq)), 0.5, 0.5]
	if (np.isnan(x0[2])):
		#x0[2] = 1/(currents[-1]-currents[0])
		freq_est_null = res_freq_estimate.copy()
		freq_est_null[np.isnan(freq_est_null)] = 0
		freq_est_fft = np.fft.fft(freq_est_null)
		freq_est_fft[:2] = 0
		freq_est_fft[int(len(freq_est_fft)/2):]=0
		max_freq_id = np.argmax(np.abs(freq_est_fft))
		max_freq = 2*np.pi/(currents[-1]-currents[0])*max_freq_id
	x0[2] = max_freq
	#print(x0)
	res_freq_cur_fit = leastsq(cost, x0)
	x = res_freq_cur_fit[0]
	#plt.plot(currents, ft_x(currents, x0))
	S21 = _S21
	interpolator = lambda I: ft_x(I, x)

	if fit:
		fitresults = [None]*currents.size
		fit_unreliable = np.zeros(currents.shape, dtype=bool)
		fit_fr = np.zeros(currents.shape)
		fit_Qc = np.zeros(currents.shape)
		fit_Ql = np.zeros(currents.shape)
		fit_Qi = np.zeros(currents.shape)
		Qc_column_name = 'Qc'
		Qi_column_name = 'Qi'
		Ql_column_name = 'Ql'
		
		last_freq = False
		span = 22e6
		for cur_id, cur in enumerate(currents):
			fit_unreliable[cur_id] = True
			if not last_freq and np.isfinite(res_freq_estimate[cur_id]):
				f0 = res_freq_estimate[cur_id]
			elif np.isfinite(last_freq):
				f0 = last_freq
			else:
				continue
			
			if (f0-span/2.<np.min(frequencies) or f0+span/2.>np.max(frequencies)):
				last_freq = False
				continue
        
			freq_mask = (frequencies>f0-span/2.)*(frequencies<f0+span/2.)
			f_data = frequencies[freq_mask]
			z_data = _S21[cur_id, :][freq_mask]
			
			fitter = circuit.reflection_port(f_data=f_data, z_data_raw=z_data)
			fitter.autofit()
			fitresults[cur_id] = fitter.fitresults
			fit_fr[cur_id] = fitresults[cur_id]['fr']
			fit_Qc[cur_id] = fitresults[cur_id][Qc_column_name]
			fit_Qi[cur_id] = fitresults[cur_id][Qi_column_name]
			fit_Ql[cur_id] = fitresults[cur_id][Ql_column_name]
			fit_unreliable[cur_id] = (1-(fit_fr[cur_id]<np.nanmax(f_data))*(fit_fr[cur_id]>np.nanmin(f_data)))>0
        
		fit_fr[fit_unreliable] = np.nan
		fit_Qi[fit_unreliable] = np.nan
		fit_Qc[fit_unreliable] = np.nan
		fit_Ql[fit_unreliable] = np.nan
		#plt.plot(currents, fit_fr)
		#plt.ylim([np.min(frequencies), np.max(frequencies)])
		#plt.figure('Qc')
		fit_Qc[fit_Qc>1e5]=np.nan
		fit_Qc[fit_Qc<0]=np.nan
	
		return {'fitresults': {'fp': x[0], 'I0': x[1], 'L': x[2], 'a': x[3], 'b': x[4]}, \
			'evaluator':interpolator, \
			'fp_est': res_freq_estimate[np.isfinite(res_freq_estimate)], \
			'fp_est_xpoints': currents[np.isfinite(res_freq_estimate)], \
			'fit_fr': fit_fr, \
			'fit_Qi': fit_Qi, \
			'fit_Qc': fit_Qc, \
			'fit_Ql': fit_Ql}
	return {'fitresults': {'fp': x[0], 'I0': x[1], 'L': x[2], 'a': x[3], 'b': x[4]}, \
			'evaluator':interpolator, \
			'fp_est': res_freq_estimate[np.isfinite(res_freq_estimate)], \
			'fp_est_xpoints': currents[np.isfinite(res_freq_estimate)]}	