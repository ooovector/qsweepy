from scipy.signal import gaussian
from scipy.signal import tukey
from scipy.signal import hann
from . import data_reduce
import numpy as np
from . import readout_classifier

class pulses:
	def __init__(self, channels = {}):
		self.channels = channels
		self.settings = {}
	
	## generate waveform of a gaussian pulse with quadrature phase mixin
	def gauss_hd (self, channel, length, amp_x, sigma, alpha=0.):
		gauss = gaussian(int(round(length*self.channels[channel].get_clock())), sigma*self.channels[channel].get_clock())
		gauss -= gauss[0]
		gauss_der = np.gradient (gauss)*self.channels[channel].get_clock()
		return amp_x*(gauss + 1j*gauss_der*alpha)
		
	# def rect_cos (self, channel, length, amp, alpha=0.):
		# alfa = 0.5
		# impulse = tukey(int(round(length*self.channels[channel].get_clock())), alfa)
		# #print(alfa*self.channels[channel].get_clock())
		# #print(length)
		# #print(round(length*self.channels[channel].get_clock()))
		# impulse -= impulse[0]
		# impulse_der = np.gradient(impulse)*self.channels[channel].get_clock()
		# return amp*(impulse + 1j*impulse_der*alpha)
		
	def rect_cos (self, channel, length, amp, length_tail, alpha=0.):
		length_of_plato = length - length_tail*2
		length_of_one_tail = int(length_tail*self.channels[channel].get_clock())
		hann_function = hann(2*length_of_one_tail)
		first = hann_function[:length_of_one_tail]
		second = hann_function[length_of_one_tail:]
		plato = np.ones(int(round(length_of_plato*self.channels[channel].get_clock())))
		final = first.tolist() 
		final.extend(plato.tolist())
		final.extend(second.tolist())
		impulse = np.asarray(final)
		impulse -= impulse[0]
		impulse_der = np.gradient(impulse)*self.channels[channel].get_clock()
		#print(self.channels[channel].get_clock())
		#print(length_tail*self.channels[channel].get_clock())
		#print(first)
		#print(second)
		#print(plato)
		return amp*(impulse + 1j*impulse_der*alpha)
		
	## generate waveform of a rectangular pulse
	def rect(self, channel, length, amplitude):
		return amplitude*np.ones(int(round(length*self.channels[channel].get_clock())), dtype=np.complex)

	def pause(self, channel, length):
		return self.rect(channel, length, 0)
		
	def p(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
		
	def ps(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
	
	def set_seq(self, seq, force=True):
		initial_delay = 1e-6
		final_delay = 1e-6
		pulse_seq_padded = [self.p(None, initial_delay, None)]+seq+[self.p(None, final_delay, None)]
	
		try:
			for channel, channel_device in self.channels.items():
				channel_device.freeze()
	
			pulse_shape = {k:[] for k in self.channels.keys()}
			for channel, channel_device in self.channels.items():
				for pulse in pulse_seq_padded:
					pulse_shape[channel].extend(pulse[channel])
				pulse_shape[channel] = np.asarray(pulse_shape[channel])
		
				if len(pulse_shape[channel])>channel_device.get_nop():
					tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
					tmp = pulse_shape[channel][-channel_device.get_nop():]
					pulse_shape[channel] = tmp
					raise(ValueError('pulse sequence too long'))
				else:
					tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
					tmp[-len(pulse_shape[channel]):]=pulse_shape[channel]
					pulse_shape[channel] = tmp
			
				channel_device.set_waveform(pulse_shape[channel])
		finally:
			for channel, channel_device in self.channels.items():
				channel_device.unfreeze()

class sz_measurer:
	def __init__(self, adc, ex_seq, ro_seq, pulse_generator, ro_delay_seq = None, ex_seq_zero=[], adc_measurement_name='Voltage'):
		self.adc = adc
		self.ro_seq = ro_seq
		self.prepare_seqs = []
		
		self.ex_seq = ex_seq
		self.ex_seq_zero = ex_seq_zero
		self.ro_delay_seq = ro_delay_seq
		self.pulse_generator = pulse_generator
		self.repeat_samples = 2
		self.save_last_samples = False
		self.train_test_split = 0.8
		self.measurement_name = ''
		self.dump_measured_samples = False
		self.cutoff_start = 0
		self.readout_classifier = readout_classifier.linear_classifier()
		self.adc_measurement_name = adc_measurement_name
	
	def measure_delay(self, ro_channel):
		import matplotlib.pyplot as plt
		from scipy.signal import resample
		
		self.pulse_generator.set_seq(self.ro_delay_seq)
		first_nonzero = int(np.nonzero(np.abs(self.pulse_generator.channels[ro_channel].get_waveform()))[0][0]/self.pulse_generator.channels[ro_channel].get_clock()*self.adc.get_clock())
		ro_dac_waveform = self.pulse_generator.channels[ro_channel].awg_I.get_waveform(channel=self.pulse_generator.channels[ro_channel].awg_ch_I)+\
					   1j*self.pulse_generator.channels[ro_channel].awg_Q.get_waveform(channel=self.pulse_generator.channels[ro_channel].awg_ch_Q)
		ro_dac_waveform = resample(ro_dac_waveform, num=int(len(ro_dac_waveform)/self.pulse_generator.channels[ro_channel].get_clock()*self.adc.get_clock()))
		ro_adc_waveform = np.mean(self.adc.measure()['Voltage'], axis=0)
		ro_dac_waveform = ro_dac_waveform - np.mean(ro_dac_waveform)
		ro_adc_waveform = ro_adc_waveform - np.mean(ro_adc_waveform)
		xc = np.abs(np.correlate(ro_dac_waveform, ro_adc_waveform, 'same'))
		xc_max = np.argmax(xc)
		delay = int((xc_max - first_nonzero)/2)
		plt.figure('delay')
		plt.plot(ro_dac_waveform[first_nonzero:])
		plt.plot(ro_adc_waveform[delay:])
		plt.plot(ro_adc_waveform)
		print ('Measured delay is {} samples'.format(delay), first_nonzero, xc_max)
		return delay
	
	def new_calibrate(self):
		X = []
		y = []
		for class_id, prepare_seq in enumerate(self.prepare_seqs):
			for i in range(self.repeat_samples):
				# pulse sequence to prepare state
				self.pulse_generator.set_seq(prepare_seq+self.ro_seq)
				measurement = self.adc.measure()
				if type(self.adc_measurement_name) is list:
					raise ValueError('Multiqubit readout not implemented') #need multiqubit readdout implementation
				else:
					X.append(measurement[self.adc_measurement_name])
				y.extend([class_id]*len(self.adc.get_points()[self.adc_measurement_name][0][1]))
		X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1]))) # last dimension is the feature dimension
		y = np.asarray(y)
		
		self.predictor_class = readout_classifier.linear_classifier()
		scores = readout_classifier.evaluate_classifier(self.predictor_class, X, y)
		self.predictor_class.fit(X, y)
		
	
	def calibrate(self):
		from sklearn.metrics import roc_auc_score, roc_curve
		
		# zero sequence
		samples_zero = []
		samples_one = []
		for i in range(self.repeat_samples):
			self.pulse_generator.set_seq(self.ex_seq_zero+self.ro_seq)
			samples_zero.append(self.adc.measure()['Voltage'])
			# pi pulse sequence
			self.pulse_generator.set_seq(self.ex_seq+self.ro_seq)
			samples_one.append(self.adc.measure()['Voltage'])
			
		samples_zero = np.asarray(samples_zero)
		samples_one  = np.asarray(samples_one )
		samples_zero = np.reshape(samples_zero, (samples_zero.shape[0]*samples_zero.shape[1], samples_zero.shape[2]))
		samples_one  = np.reshape(samples_one, (samples_one.shape[0]*samples_one.shape[1], samples_one.shape[2]))
		
		samples = np.asarray([samples_zero, samples_one])
		train = samples[:,:int(samples.shape[1]*self.train_test_split),:]
		test = samples[:,int(samples.shape[1]*self.train_test_split):,:]
		train = train - np.reshape(np.mean(train, axis=2), (train.shape[0], train.shape[1], 1))
		test = test - np.reshape(np.mean(test, axis=2), (test.shape[0], test.shape[1], 1))
		
		train[:,:,:self.cutoff_start] = 0	
		
		mean_train_signal = np.mean(np.mean(train, axis=1),axis=0)
		diff_train_signal = np.diff(np.mean(train, axis=1),axis=0).ravel()
		diff_train_signal = diff_train_signal - np.mean(diff_train_signal)
		#train_correl = np.einsum('ijk,ilk->jl', np.conj(train-np.reshape(np.mean(train, axis=1), (2,1,-1))), train-np.reshape(np.mean(train, axis=1), (2,1,-1)))

		mean_test_signal = np.mean(np.mean(test, axis=1),axis=0)
		diff_test_signal = np.diff(np.mean(test, axis=1),axis=0).ravel()
		diff_test_signal = diff_test_signal - np.mean(diff_test_signal)
		
#		diff_signal_fft = np.fft.fft(diff_signal)
#		pmax = np.argmax(diff_signal_fft)
#		diff_signal_filt_fft = np.zeros(diff_signal_fft.shape, dtype=np.complex)
#		diff_signal_filt_fft[pmax-8:pmax+8] = diff_signal_fft[pmax-8:pmax+8]
#		diff_singal = np.fft.ifft(diff_signal_filt_fft)

		# SUPERVISED PREDICTOR
		# TEST
		coefficients_train = np.einsum('k,ijk->ij', 
								np.conj(diff_train_signal), 
								test-mean_train_signal)
		coefficients_test = np.einsum('k,ijk->ij', 
								np.conj(diff_test_signal), 
								test-mean_test_signal)							
		
		scale = np.mean(np.conj(np.mean(coefficients_train[1,:]) - np.mean(coefficients_train[0,:])))
		feature = np.conj(diff_train_signal)/scale
		feature[:self.cutoff_start] = 0
		#feature2 = np.linsolve(np.conj(train_correl), np.conj(diff_train_signal))
		
		filter = data_reduce.feature_reducer(self.adc, 'Voltage', 1, mean_train_signal, feature)

		predictions = np.real(np.einsum('k,ijk->ij', feature, test-mean_train_signal))
		
		self.calib_bg = mean_train_signal
		self.calib_feature = feature
		self.calib_pred = predictions
		
		hist_all, bins = np.histogram(predictions, bins='auto')
		proba_points = (bins[1:]+bins[:-1])/2.
		hists = []
		
		for y in range(2):
			hists.append(np.histogram(predictions[y,:], bins=bins)[0])
			
		hists = np.asarray(hists, dtype=float)
		probabilities = hists/hist_all
		naive_probabilities = np.asarray([proba_points<0, proba_points>0], dtype=float)
		probabilities[np.isnan(probabilities)] = naive_probabilities[np.isnan(probabilities)]
		
		self.predictor = lambda x: np.interp(x, proba_points, probabilities[1,:], left=0., right=1.)
		self.calib_proba_points = proba_points
		self.calib_proba = probabilities[1,:]
		self.calib_hists = hists
		
		roc_curve = roc_curve([0]*test.shape[1]+[1]*test.shape[1], self.predictor(predictions.ravel()))
		roc_auc = roc_auc_score([0]*test.shape[1]+[1]*test.shape[1], self.predictor(predictions.ravel()))
		fidelity = np.mean([np.mean(self.predictor(predictions[0,:]<0)), 
							np.mean(self.predictor(predictions[1,:]>0))])

		#roc_auc_binary = roc_auc_score([0]*samples.shape[1]+[1]*samples.shape[1], (predictions.ravel()>0)*2-1)
		#fidelity_binary = np.mean([np.sqrt(np.mean(predictions[0,:]<0)), 
		#							np.sqrt(np.mean(predictions[1,:]>0))])
		
		self.calib_roc_curve = roc_curve
		self.calib_roc_auc = roc_auc
		self.calib_fidelity = fidelity

		#self.calib_roc_auc_binary = roc_auc_binary
		#self.calib_fidelity_binary = fidelity_binary
		
		self.filter = filter
		self.filter_binary = data_reduce.feature_reducer_binary(self.adc, 'Voltage', 1, mean_train_signal, feature)
		self.measurer = data_reduce.data_reduce(self.adc)
		self.measurer.filters['sz'] = self.filter
		#self.binary_measurer = data_reduce.data_reduce(self.adc)
		self.measurer.filters['sz binary'] = self.filter_binary
		
		self.filter_binary_mean = data_reduce.mean_reducer(self.measurer, 'sz binary', 0)
		self.filter_mean = data_reduce.mean_reducer(self.measurer, 'sz', 0)
		
		if self.save_last_samples:
			self.samples = samples
		
		usl = False
		if usl:
		# Unsupervised predictions
			samples = np.fft.fft(np.reshape(samples, (samples.shape[0]*samples.shape[1], samples.shape[2])), axis=1)
		
			psd = np.mean(np.abs(samples)**2,axis=0)
			mean_psd = np.abs(np.fft.fft(mean_signal))**2
			#self.calib_usl_filt = np.sqrt(mean_psd/psd) # filter out the fourier components such that the snr in each is equal
			self.calib_usl_filt = np.sqrt(psd/psd) # filter out the fourier components such that the snr in each is equal
			self.calib_usl_filt[-50:]=0.
			self.calib_usl_filt[:50]=0.
			self.calib_usl_filt = self.calib_usl_filt/np.sqrt(np.mean(self.calib_usl_filt**2))
		
			samples = samples*self.calib_usl_filt
			samples = np.fft.ifft(samples,axis=1)
		
			# create a correlation matrix
			samples = np.hstack([samples, np.ones((samples.shape[0], 1), dtype=np.complex)])
			cov = np.einsum('ij,ik->jk', np.conj(samples), samples)/samples.shape[0]
			self.calib_usl_cov = cov
			W,V = np.linalg.eigh(cov)
		
			mean_ind = np.argsort(np.abs(W))[-1]
			diff_ind = np.argsort(np.abs(W))[-2]
			
			scaled_bg = V[:-1,mean_ind]/V[-1,mean_ind]
			self.calib_usl_bg = scaled_bg
			self.calib_usl_feature = np.conj(V[:-1,diff_ind])
		
			self.calib_usl_pred = np.dot(np.reshape(samples[:,:-1], predictions.shape+self.calib_usl_feature.shape), self.calib_usl_feature)
			diff = np.diff(np.mean(self.calib_usl_pred, axis=1))[0]
			self.calib_usl_feature = self.calib_usl_feature/diff
			self.calib_usl_pred = self.calib_usl_pred / diff
		
		return filter
	
	def get_opts(self):
		return {#'Calibrated ROC AUC binary': {'log': False},
				'Calibrated ROC AUC': {'log': False},
				#'Calibrated fidelity binary': {'log': False} ,
				'Calibrated fidelity': {'log': False}}
	
	def measure(self):
		self.calibrate()
		meas = {#'Calibrated ROC AUC binary': self.calib_roc_auc_binary,
				'Calibrated ROC AUC': self.calib_roc_auc,
				#'Calibrated fidelity binary': self.calib_fidelity_binary,
				'Calibrated fidelity': self.calib_fidelity}
		if self.dump_measured_samples:
			self.dump_samples(name=self.measurement_name)
		return meas
		
	def get_points(self):
		return {#'Calibrated ROC AUC binary': {},
				'Calibrated ROC AUC': {},
				#'Calibrated fidelity binary': {},
				'Calibrated fidelity': {} }
				
	def get_dtype(self):
		return {#'Calibrated ROC AUC binary': float,
				'Calibrated ROC AUC': float,
				#'Calibrated fidelity binary': float, 
				'Calibrated fidelity': float}
	
	def dump_samples(self, name):
		from .save_pkl import save_pkl
		header = {'type':'Binary classification samples', 'name':name}
		measurement = {'Binary classification samples':(['Class', 'Sample ID', 'time'], 
				[np.asarray([0, 1]), np.arange(self.samples.shape[1]), np.arange(self.samples.shape[2])/self.adc.get_clock()],
				self.samples)}
		save_pkl(header, self.samples, plot=False)
	
class tomography:
	def __init__(self, sz_measurer, pulse_generator, proj_seq, reconstruction_basis={}):
		self.sz_measurer = sz_measurer
		#self.adc = adc
		self.pulse_generator = pulse_generator
		self.proj_seq = proj_seq
		self.reconstruction_basis=reconstruction_basis
		
		self.adc_reducer = data_reduce.data_reduce(self.sz_measurer.adc)
		self.adc_reducer.filters['SZ'] = self.sz_measurer.filter_binary
		
	def get_points(self):
		points = { p:{} for p in self.proj_seq.keys() }
		points.update({p:{} for p in self.reconstruction_basis.keys()})
		return points
	
	def get_dtype(self):
		dtypes = { p:float for p in self.proj_seq.keys() }
		dtypes.update({ p:float for p in self.reconstruction_basis.keys() })
		return dtypes
	
	def set_prepare_seq(self, seq):
		self.prepare_seq = seq
	
	def measure(self):
		meas = {}
		for p in self.proj_seq.keys():
			self.pulse_generator.set_seq(self.prepare_seq+self.proj_seq[p]['pulses'])
			meas[p] = np.real(np.mean(self.adc_reducer.measure()['SZ'])/2)

		proj_names = self.proj_seq.keys()
		basis_axes_names = self.reconstruction_basis.keys()
		#TODO: fix this norm stuff in accordance with theory
		basis_vector_norms = np.asarray([np.linalg.norm(self.reconstruction_basis[r]['operator']) for r in basis_axes_names])
		
		if len(self.reconstruction_basis.keys()):
			reconstruction_matrix = np.real(np.asarray([[np.sum(self.proj_seq[p]['operator']*np.conj(self.reconstruction_basis[r]['operator'])) \
										for r in basis_axes_names] \
										for p in proj_names]))
			projections = np.linalg.lstsq(reconstruction_matrix, [meas[p] for p in proj_names])[0]*(basis_vector_norms**2)
			meas.update({k:v for k,v in zip(basis_axes_names, projections)})
		return meas
		
	def get_opts(self):
		opts = { p:{} for p in self.proj_seq.keys()}
		opts.update ({ p:{} for p in self.reconstruction_basis.keys()})
		return opts
		