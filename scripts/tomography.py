from scipy.signal import gaussian
import data_reduce
import numpy as np

class pulses:
	def __init__(self, channels = {}):
		self.channels = channels
		self.settings = {}
	
	## generate waveform of a gaussian pulse with quadrature phase mixin
	def gauss_hd (self, channel, length, amp_x, amp_y, sigma, alpha=0.):
		gauss = gaussian(int(round(length*self.channels[channel].get_clock())), sigma*self.channels[channel].get_clock())
		gauss -= gauss[0]
		gauss_der = np.gradient (gauss)*self.channels[channel].get_clock()
		return amp_x*(gauss + 1j*gauss_der*alpha) + 1j*amp_y*(gauss + 1j*gauss_der*alpha)
		
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
	
	def set_seq(self, seq):
		initial_delay = 1e-6
		final_delay = 1e-6
		pulse_seq_padded = [self.p(None, initial_delay, None)]+seq+[self.p(None, final_delay, None)]
	
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

class sz_measurer:
	def __init__(self, adc, ex_seq, ro_seq, pulse_generator):
		self.adc = adc
		self.ex_seq = ex_seq
		self.ro_seq = ro_seq
		self.pulse_generator = pulse_generator
		self.repeat_samples = 2
		self.save_last_samples = False
	
	def calibrate(self):
		from sklearn.metrics import roc_auc_score, roc_curve
		
		# zero sequence
		samples_zero = []
		samples_one = []
		for i in range(self.repeat_samples):
			self.pulse_generator.set_seq(self.ro_seq)
			samples_zero.append(self.adc.measure()['Voltage'])
			# pi pulse sequence
			self.pulse_generator.set_seq(self.ex_seq+self.ro_seq)
			samples_one.append(self.adc.measure()['Voltage'])
			
		samples_zero = np.asarray(samples_zero)
		samples_one  = np.asarray(samples_one )
		samples_zero = np.reshape(samples_zero, (samples_zero.shape[0]*samples_zero.shape[1], samples_zero.shape[2]))
		samples_one  = np.reshape(samples_one, (samples_one.shape[0]*samples_one.shape[1], samples_one.shape[2]))

		samples = np.asarray([samples_zero, samples_one])
		
		mean_signal = np.mean(np.mean(samples, axis=1),axis=0)
		diff_signal = np.diff(np.mean(samples, axis=1),axis=0).ravel()
		
		diff_signal = diff_signal - np.mean(diff_signal)
		
#		diff_signal_fft = np.fft.fft(diff_signal)
#		pmax = np.argmax(diff_signal_fft)
#		diff_signal_filt_fft = np.zeros(diff_signal_fft.shape, dtype=np.complex)
#		diff_signal_filt_fft[pmax-8:pmax+8] = diff_signal_fft[pmax-8:pmax+8]
#		diff_singal = np.fft.ifft(diff_signal_filt_fft)

		# SUPERVISED PREDICTOR
		coefficients = np.einsum('k,ijk->ij', 
								np.conj(diff_signal), 
								samples-mean_signal)
								
		scale = np.mean(np.conj(np.mean(coefficients[1,:]) - np.mean(coefficients[0,:])))
		feature = np.conj(diff_signal)/scale
		
		filter = data_reduce.feature_reducer(self.adc, 'Voltage', 1, mean_signal, feature)

		predictions = np.real(np.einsum('k,ijk->ij', feature, samples-mean_signal))
		
		self.calib_bg = mean_signal
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
		
		roc_curve = roc_curve([0]*samples.shape[1]+[1]*samples.shape[1], self.predictor(predictions.ravel()))
		roc_auc = roc_auc_score([0]*samples.shape[1]+[1]*samples.shape[1], self.predictor(predictions.ravel()))
		fidelity = np.mean([np.sqrt(np.mean(self.predictor(predictions[0,:]))), 
							np.sqrt(np.mean(self.predictor(predictions[1,:])))])

		roc_auc_binary = roc_auc_score([0]*samples.shape[1]+[1]*samples.shape[1], (predictions.ravel()>0)*2-1)
		fidelity_binary = np.mean([np.sqrt(np.mean(predictions[0,:]<0)), 
									np.sqrt(np.mean(predictions[1,:]>0))])
		
		self.calib_roc_curve = roc_curve
		self.calib_roc_auc = roc_auc
		self.calib_fidelity = fidelity

		self.calib_roc_auc_binary = roc_auc_binary
		self.calib_fidelity_binary = fidelity_binary
		
		self.filter = filter
		self.filter_binary = data_reduce.feature_reducer_binary(self.adc, 'Voltage', 1, mean_signal, feature)
		
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
		return {'Calibrated ROC AUC binary': {'log': False},
				'Calibrated ROC AUC': {'log': False},
				'Calibrated fidelity': {'log': False},
				'Calibrated fidelity binary': {'log': False} }
	
	def measure(self):
		self.calibrate()
		meas = {'Calibrated ROC AUC binary': self.calib_roc_auc_binary,
				'Calibrated ROC AUC': self.calib_roc_auc,
				'Calibrated fidelity': self.calib_fidelity,
				'Calibrated fidelity binary': self.calib_fidelity_binary}
		return meas
		
	def get_points(self):
		return {'Calibrated ROC AUC binary': {},
				'Calibrated ROC AUC': {},
				'Calibrated fidelity': {},
				'Calibrated fidelity binary': {} }
				
	def get_dtype(self):
		return {'Calibrated ROC AUC binary': float,
				'Calibrated ROC AUC': float,
				'Calibrated fidelity': float,
				'Calibrated fidelity binary': float }
	
	
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
		