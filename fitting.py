import matplotlib.pyplot as plt
from .ponyfiles.data_structures import *

def resample_x_fit(x):
	if len(x) < 500:
		return np.linspace(np.min(x), np.max(x), 501)
	else:
		return x

def exp_fit1d (x, y):
	def model(x,p):
		x0=p[0]
		A =p[1]
		B =p[2]
		return A*np.exp(-x/x0)+B
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()

	y = np.asarray(y)
	y_last = np.reshape(y[:,-1], (y.shape[0], 1))
	y = y - y_last

	integral = np.sum(y,axis=1)*(x[1]-x[0])
	y_first = y[:,0]
	x0=np.sqrt(np.sum(np.abs(integral)**2)/np.sum(np.abs(y_first)**2))

	p0 = [x0]+y_first.tolist()+[0]

	from scipy.optimize import leastsq
	fitresults = leastsq (cost, p0)
	fitted_curve = model(resample_x_fit(x), fitresults[0])

	#for i in range(y.shape[0]):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()

	return fitresults[0], (resample_x_fit(x), fitted_curve+y_last)

def gaussian_CPMG(x, y, gamma1, gammap, tayp):
	def model(x, p):
		gammaphi = p[0]
		AB = p[1:]
		size = int(len(AB)/2)
		A = np.reshape(AB[:size], size)
		B = np.reshape(AB[-size:], size)
		return A*np.exp(-x*gamma1/2)*exp(-gammap*tayp)*exp(-(gammaphi*x)**2)+B
	error_function = lambda p: (np.abs(model(x,y) - y)**2).ravel()

	y = np.asarray(y)
	y_last = y[:,-1]
	y_firsy = y[:, 0]
	gammaphi_0 = np.sqrt(np.abs(np.ln(y_first-y_last))/x[0]**2)
	p0 = [gammaphi_0] + y_first.tolist() + y_last.tolist()

	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve

def exp_fit (x, y):
	def model(x,p):
		x0=p[0]
		AB = p[1:]
		A = np.reshape(AB[:int(len(AB)/2)],  (int(len(AB)/2), 1))
		B = np.reshape(AB[-int(len(AB)/2):], (int(len(AB)/2), 1))
		return A*np.exp(-x/x0)+B
	cost = lambda p: (np.abs(model(x, p)-y)**2).ravel()

	y = np.asarray(y)
	y_last = y[:,-1]#np.reshape(y[:,-1], (y.shape[0], 1))
	#y = y - y_last

	integral = np.sum(y,axis=1)*(x[1]-x[0])
	y_first = y[:,0]
	x0=np.sqrt(np.sum(np.abs(integral)**2)/np.sum(np.abs(y_first)**2))

	p0 = [x0]+y_first.tolist()+y_last.tolist()
	print (p0)

	from scipy.optimize import leastsq
	fitresults = leastsq (cost, p0)
	fitted_curve = model(resample_x_fit(x), fitresults[0])

	#for i in range(y.shape[0]):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()
	parameters = {'decay': fitresults[0][0], 'amplitudes':fitresults[0][1:]}
	return (resample_x_fit(x), fitted_curve), parameters

def sin_fit(x, y, resample=501):
# фитует результаты измерений синусом
	def model(x, p):
		phase = p[0]
		freq = p[1]
		A = np.reshape(np.asarray(p[2:]),(len(p[2:]), 1))
		return A*np.cos(phase+x*freq*2*np.pi)

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
	#x0 = np.sqrt(np.mean(np.abs(ft[:,fR_id])**2)/np.mean(np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)-1)/domega/2
	#print (np.abs(ft[:,fR_id])**2, np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)

	A = np.sqrt(np.abs(ft[:,fR_id])**2+np.abs(ft[:,fR_id_conj])**2)
	p0 = [phase, fR]+A.tolist()

	from scipy.optimize import leastsq
	fitresults_all = []
	for random_id in range(100):
		random_samples = np.random.choice(len(x), int(len(x)*0.7))
		cost = lambda p: (np.abs(model(x[random_samples], p)-y[:,random_samples])**2).ravel()

		fitresults = leastsq (cost, p0)
		fitresults_all.append(fitresults[0])
	fitted_curve = model(resample_x_fit(x), fitresults[0])
	parameters = {'phase':fitresults[0][0], 'freq':fitresults[0][1], 'amplitudes':fitresults[0][3:]}
	#resampled_x = np.linspace(np.min(x), np.max(x), resample)
	#resampled_y = model(resampled_x, fitresults[0])+means

	return (resample_x_fit(x), fitted_curve+means), parameters




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
	z = np.asarray(y)[:,0]
	y = y-means

	ft = np.fft.fft(y-np.reshape(np.mean(y, axis=1), (y.shape[0], 1)), axis=1)/len(x)
	f = np.fft.fftfreq(len(x), x[1]-x[0])
	domega = (f[1]-f[0])*2*np.pi

	#plt.figure('FFT')
	#plt.plot(f, ft.T)

	#plt.figure('Initial fit')

	fR_id = np.argmax(np.sum(np.abs(ft)**2, axis=0))
	fR_id_conj = len(f)-fR_id
	if fR_id_conj == len(f):
		fR_id_conj = 0
	if fR_id_conj > fR_id:
		tmp = fR_id_conj
		fR_id_conj = fR_id
		fR_id = tmp

	fR = np.abs((f[fR_id]))

	c = np.real(np.sum(ft[:,fR_id], axis=0))
	s = np.imag(np.sum(ft[:,fR_id], axis=0))
	phase = np.pi+np.arctan2(c, s)
	if fR_id==0 or fR_id+1==len(f):
		x0 = np.max(x)-np.min(x)
	else:
		x0 = np.sqrt(np.mean(np.abs(ft[:,fR_id])**2)/np.mean(np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)-1)/domega/2

	#print (x0, np.abs(ft[:,fR_id])**2, np.abs((ft[:,fR_id-1]+ft[:,fR_id+1])/2)**2)

	A = np.sqrt(np.abs(ft[:,fR_id])**2+np.abs(ft[:,fR_id_conj])**2)*2
	p0 = [phase, fR, x0]+A.tolist()

	from scipy.optimize import leastsq
	#plt.plot(x, (model(x, p0)).T)
	#plt.plot(x, (y).T, marker='o', markerfacecolor='None', linestyle='none')
	fitresults = leastsq (cost, p0)
	fitted_curve = model(resample_x_fit(x), fitresults[0])

	#for i in range(4):
	#	plt.figure(i)
	#	plt.plot(x, y[i,:], label='data')
	#	plt.plot(x, model(x, p0)[i,:], label='initial')
	#	plt.plot(x, model(x, fitresults[0])[i,:], label='fitted')
	#	plt.legend()

	parameters = {'phase':fitresults[0][0], 'freq':fitresults[0][1], 'decay': fitresults[0][2], 'amplitudes':fitresults[0][3:], 'initial_points':z}

	return (resample_x_fit(x), fitted_curve+means), parameters

def fit1d(measurement, fitter, dataset_name=None, db=None):
	if len(measurement.datasets) and not dataset_name:
		dataset_name = [mname for mname in measurement.datasets.keys()][0]
	elif not dataset_name:
		raise(ValueError('No measurement_name passed to fit1d. Available names are: '+', '.join([str(mname) for mname in measurement.datasets.keys()])))

	dataset = measurement.datasets[dataset_name]
	t = np.asarray(dataset.parameters[0].values)
	if np.iscomplexobj(dataset.data):
		fitdata = [ np.real(dataset.data),
			        np.imag(dataset.data) ]
	else:
		fitdata = [ dataset.data ]

	fitted_curve, parameters = fitter(t, fitdata)
	fitresult = MeasurementState(sample_name=measurement.sample_name, measurement_type=measurement.measurement_type + '-fit', references={measurement.id: 'fit_source'})

	fitted_parameter = MeasurementParameter(fitted_curve[0], False, dataset.parameters[0].name, dataset.parameters[0].unit, dataset.parameters[0].pre_setter)

	fitresult.datasets = {dataset_name+'-fit':MeasurementDataset(parameters=[fitted_parameter], data=fitted_curve[1])}
	fitresult.metadata.update({k:str(v) for k,v in parameters.items()})

	return fitresult


# def fit1d(measurement, fitter, measurement_name=None):
	# if len(measurement)==1 and not measurement_name: # if there is only one measurement name in dict, fit items
		# measurement_name = [mname for mname in measurement.keys()][0]
	# elif not measurement_name:
		# raise(ValueError('No measurement_name passed to fit1d. Available names are: '+', '.join([str(mname) for mname in measurement.keys()])))

	# t = measurement[measurement_name][1][0]
	# if np.iscomplexobj(measurement[measurement_name][2]):
		# fitdata = [ np.real(measurement[measurement_name][2]),
			        # np.imag(measurement[measurement_name][2]) ]
	# else:
		# fitdata = [ measurement[measurement_name][2]]
	# fitted_curve, parameters = fitter(t, fitdata)
	# #if fitter is exp_sin_fit:
	# #	parameters = {'phase':parameters[0], 'freq':parameters[1], 'decay': parameters[2], 'amplitudes':parameters[3:]}
	# #elif fitter is exp_fit:
	# #	parameters = {'decay': parameters[0], 'amplitudes':parameters[1:]}

	# measurement[measurement_name+' fit'] = [t for t in measurement[measurement_name]]
	# measurement[measurement_name+' fit'][1] = [a for a in measurement[measurement_name][1]]
	# measurement[measurement_name+' fit'][1][0] = fitted_curve[0]
	# if np.iscomplexobj(measurement[measurement_name][2]):
		# measurement[measurement_name+' fit'][2] = fitted_curve[1][0,:]+fitted_curve[1][1,:]*1j
	# else:
		# measurement[measurement_name+' fit'][2] = fitted_curve[1][0,:]

	# if len(measurement[measurement_name+' fit'])>3:
		# measurement[measurement_name+' fit'][3] = dict(measurement[measurement_name+' fit'][3])
		# if 'scatter' in measurement[measurement_name+' fit'][3]:
			# del measurement[measurement_name+' fit'][3]['scatter']

	# measurement[measurement_name+' fit'] = tuple(measurement[measurement_name+' fit'])

	# return measurement, parameters


def S21pm_fit(measurement, fitter):
	t = measurement['S21+'][1][0]
	fitdata = [ np.real(measurement['S21+'][2]),
				np.imag(measurement['S21+'][2]),
				np.real(measurement['S21-'][2]),
				np.imag(measurement['S21-'][2])]
	fitted_curve, parameters = fitter(t, fitdata)
	#if fitter is exp_sin_fit:
	#	parameters = {'phase':parameters[0], 'freq':parameters[1], 'decay': parameters[2], 'amplitudes':parameters[3:]}
	#elif fitter is exp_fit:
	#	parameters = {'decay': parameters[0], 'amplitudes':parameters[1:]}

	measurement['S21+ fit'] = [list(t) if type(t) is tuple else t for t in measurement['S21+']]
	measurement['S21- fit'] = [list(t) if type(t) is tuple else t for t in measurement['S21-']]
	measurement['S21+ fit'][1][0] = fitted_curve[0]
	measurement['S21- fit'][1][0] = fitted_curve[0]
	measurement['S21+ fit'][2] = fitted_curve[1][0,:]+fitted_curve[1][1,:]*1j
	measurement['S21- fit'][2] = fitted_curve[1][2,:]+fitted_curve[1][3,:]*1j

	if len(measurement['S21+ fit'])>3:
		measurement['S21+ fit'][3] = dict(measurement['S21+ fit'][3])
		if 'scatter' in measurement['S21+ fit'][3]:
			del measurement['S21+ fit'][3]['scatter']
	if len(measurement['S21- fit'])>3:
		measurement['S21- fit'][3] = dict(measurement['S21- fit'][3])
		if 'scatter' in measurement['S21- fit'][3]:
			del measurement['S21- fit'][3]['scatter']

	measurement['S21+ fit'] = tuple(measurement['S21+ fit'])
	measurement['S21- fit'] = tuple(measurement['S21- fit'])

	return measurement, parameters

def xcorr_centre_period(vec, axis=0, drop_size=5):
    f_stack = [lambda x: np.correlate(x-np.mean(x), np.flip(x-np.mean(x), axis=0), mode='full')]
    iteraxes = list(np.arange(len(vec.shape)))
    del iteraxes[axis]
    for _axis in iteraxes[::-1]:
        axis_id = len(f_stack)-1
        f_stack.append(lambda x: np.apply_along_axis(f_stack[axis_id], _axis, x))
    mpi = f_stack[-1](vec)

    f_stack = [lambda x: np.correlate(x-np.mean(x), x-np.mean(x), mode='full')]
    iteraxes = list(np.arange(len(vec.shape)))
    del iteraxes[axis]
    for _axis in iteraxes[::-1]:
        axis_id = len(f_stack)-1
        f_stack.append(lambda x: np.apply_along_axis(f_stack[axis_id], _axis, x))
    mp = f_stack[-1](vec)
    mp_scalar = np.sum(np.abs(mp)**2, axis=tuple(range(1, len(vec.shape))))
    mp_scalar[int(len(mp_scalar)/2)-drop_size:int(len(mp_scalar)/2)+drop_size]=0
    period = int(len(mp_scalar)/2)-np.argmax(mp_scalar[:int(len(mp_scalar)/2)])

    xc_max = np.argmax(np.sum(np.abs(mpi)**2, axis=tuple(range(1, len(vec.shape)))))
    plt.plot(mp_scalar)
    plt.plot(np.sum(np.abs(mpi)**2, axis=tuple(range(1, len(vec.shape)))))
    return (xc_max/2*(len(mp_scalar)*2)/(len(mp_scalar)*2-1)), period

def xcorr_scale(vec1, vec2, axis=0, thresh=0.97, centre_reference_pixels=None, max_scale=None):
	# vec1  is reference
	# vec2 is compared with

	# prefers peak near centres (avoids getting into the wrong period for periodic)
	from skimage.transform import hough_line

	iteraxes = list(np.arange(len(vec1.shape)))
	del iteraxes[axis]
	vec1 -= np.mean(vec1, axis=tuple(iteraxes), keepdims=True)
	#plt.figure()
	# vec1_4 = np.sum(np.abs(vec1-np.mean(vec1, axis=axis, keepdims=True))**2, axis=axis, keepdims=True)
	# vec2_4 = np.sum(np.abs(vec2-np.mean(vec2, axis=axis, keepdims=True))**2, axis=axis, keepdims=True)
	# vec1=vec1*vec1_4
	# vec2=vec2*vec2_4

	# plt.figure()
	# plt.pcolormesh(np.real(vec1))
	# plt.colorbar()
	# plt.figure()
	# plt.pcolormesh(np.real(vec2))
	# plt.colorbar()

	#vec1 = vec1*np.std(vec1, axis=tuple(iteraxes), keepdims=True)
	#vec2 = vec2*np.std(vec2, axis=tuple(iteraxes), keepdims=True)

	# first correlate vec1 and vec2 elements with each other
	new_axes = [axis+1]+[i+1 if i != axis else 0 for i in range(len(vec1.shape))]
	vec2_transposed = np.reshape(vec2, tuple([1]+list(vec2.shape)))
	vec2_transposed = np.transpose(vec2_transposed, new_axes)
	reduce_axes = tuple([i+1 for i in range(len(vec1.shape)) if i != axis])
	corr2d = np.sum(np.conj(vec1)*vec2_transposed, reduce_axes)
	corr2d = corr2d/np.sqrt(np.sum(np.abs(corr2d)**2, axis=0, keepdims=True)*np.sum(np.abs(corr2d)**2, axis=1, keepdims=True))
	#corr2d = -np.sum(np.abs(np.conj(vec1)-vec2_transposed)**2, reduce_axes)

	#plt.figure()
	#plt.pcolormesh(np.real(vec1))
	#plt.figure()
	#plt.pcolormesh(np.real(vec2))
	# use gradient modulus for edge detection
	#corr2d_gradient = np.sqrt(np.sum([np.abs(i)**2 for i in np.gradient(corr2d)], axis=0))
	#corr2d_gradient -= np.mean(corr2d_gradient)

	# ad-hoc crap to reduce influence of borders
	#corr2d_gradient[:int(corr2d_gradient.shape[0]/4),:] = 0
	#corr2d_gradient[-int(corr2d_gradient.shape[0]/4):,:] = 0
	#corr2d_gradient[:,:int(corr2d_gradient.shape[1]/4)] = 0
	#corr2d_gradient[:, -int(corr2d_gradient.shape[1]/4):] = 0

	#corr2d[:int(corr2d_gradient.shape[0]/5),:] = 0
	#corr2d[-int(corr2d_gradient.shape[0]/5):,:] = 0

	thresh = np.percentile(np.real(corr2d), thresh*100)
	bin = np.real(corr2d)>thresh

	#bin[:int(corr2d_gradient.shape[0]/4),:] = 0
	#bin[-int(corr2d_gradient.shape[0]/4):,:] = 0

#	corr2d[:,:int(corr2d_gradient.shape[1]/4)] = 0
#	corr2d[:, -int(corr2d_gradient.shape[1]/4):] = 0

	#binarizer using threshold
	#thresh = np.percentile(corr2d_gradient, thresh*100)
	#bin = np.real(corr2d_gradient)>thresh


	#h_real, theta, d = hough_line(np.real(corr2d))
	#h_imag, theta, d = hough_line(np.imag(corr2d))

	#h_grad, theta, d = hough_line(corr2d_gradient)

	# Hough transform: lines -> vectices
	h_bin, theta, d =  hough_line(bin)
	#h, theta, d =  hough_line(corr2d)
	#cxx, cyy = np.meshgrid(np.arange(corr2d.shape[0]), np.arange(corr2d.shape[1]))
	if not centre_reference_pixels:
		centre_reference_pixels = corr2d.shape[1]/2

	dist_to_origin = np.cos(theta)*(centre_reference_pixels)+np.sin(theta)*(corr2d.shape[0]/2)-np.reshape(d, (-1, 1))
	# plt.figure()
	# plt.pcolormesh(theta, d, np.abs(dist_to_origin))
	# plt.colorbar()
	#h_bin = h_real+1j*h_imag
	h_corrected = (h_bin/np.sqrt(10**2+np.abs(dist_to_origin)**2))/(1/np.tan(2*theta)**2+1)**(1/8)
	if max_scale: h_corrected[:, np.abs(np.tan(theta))>max_scale] = 0
	# prefer maxima near centre
	max_raveled_index = np.argmax(h_corrected)
	d_index,theta_index = np.unravel_index(max_raveled_index, h_bin.shape)
	max_theta = theta[theta_index]
	max_d = dist_to_origin[d_index,theta_index ]

	#h = h_real+1j*h_imag
	# plt.figure()
	# plt.pcolormesh(np.real(bin))
	# plt.colorbar()
	# plt.figure()
	# plt.pcolormesh(theta, d, np.real(h_bin))
	# plt.colorbar()
	# plt.figure()
	# plt.pcolormesh(theta, d, np.real(h_corrected))
	# plt.colorbar()
	# plt.figure()
	# plt.pcolormesh(theta, d, h)
	# plt.colorbar()
	# plt.figure()
	# plt.pcolormesh(np.real(corr2d))
	# plt.colorbar()
	#plt.figure('Xcorr imag')
	#plt.pcolormesh(np.imag(corr2d))
	#plt.colorbar()

	#plt.figure()
	#plt.pcolormesh(theta, d,(h_bin.T/(10**2+np.abs(d)**2)**(1/4)).T)
	#plt.colorbar()
	#x_ind = max_d*np.cos(max_theta)
	#y_ind = max_d*np.sin(max_theta)
	#print(max_d, max_theta)
	#print(x_ind, y_ind, np.tan(theta))
	scale = np.tan(max_theta)*vec2.shape[axis]/vec1.shape[axis]
	return scale, max_d

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
