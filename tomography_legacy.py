from scipy.signal import gaussian
from . import data_reduce
import numpy as np

def ex_gauss_hd (amp_x, amp_y, length, sigma, awg_channels, delta):
	iq_ex = awg_channels['iq_ex']
	iq_ro = awg_channels['iq_ro']
	ro_trg = awg_channels['ro_trg']
	osc_trg = awg_channels['osc_trg']
	gauss = gaussian(int(round(length*iq_ex.get_clock())), sigma*iq_ex.get_clock())
	gauss_der = np.gradient (gauss)*iq_ex.get_clock()
	gauss -= gauss[0]
	return {'ex': amp_x*gauss - 1j*amp_y*gauss_der/2/np.pi/delta,
			'ro':np.zeros(int(round(length*iq_ro.get_clock())), dtype=np.complex),
			'ro_trg':np.zeros(int(round(length*ro_trg.get_clock())), dtype=int),
			'osc_trg':np.zeros(int(round(length*osc_trg.get_clock())), dtype=int)}
def ex_gauss(amplitude, length, sigma, awg_channels):
	iq_ex = awg_channels['iq_ex']
	iq_ro = awg_channels['iq_ro']
	ro_trg = awg_channels['ro_trg']
	osc_trg = awg_channels['osc_trg']
	gauss = gaussian(int(round(length*iq_ex.get_clock())), sigma*iq_ex.get_clock())
	gauss -= gauss[0]
	return {'ex': amplitude*np.asarray(gauss).astype(np.complex),
			'ro':np.zeros(int(round(length*iq_ro.get_clock())), dtype=np.complex),
			'ro_trg':np.zeros(int(round(length*ro_trg.get_clock())), dtype=int),
			'osc_trg':np.zeros(int(round(length*osc_trg.get_clock())), dtype=int)}

def ro_rect(amplitude, length, awg_channels):
	iq_ex = awg_channels['iq_ex']
	iq_ro = awg_channels['iq_ro']
	ro_trg = awg_channels['ro_trg']
	osc_trg = awg_channels['osc_trg']
	return {'ex': np.zeros(int(round(length*iq_ex.get_clock())),dtype=np.complex),
			'ro': amplitude*np.ones(int(round(length*iq_ro.get_clock())), dtype=np.complex),
			'ro_trg':np.hstack([[1, 1, 1, 1],np.zeros(int(round(length*ro_trg.get_clock()-4)), dtype=int)]),
			'osc_trg':np.hstack([[1, 1, 1, 1],np.zeros(int(round(length*osc_trg.get_clock()-4)), dtype=int)]),}

def ex_rect(amplitude, length, awg_channels):
	iq_ex = awg_channels['iq_ex']
	iq_ro = awg_channels['iq_ro']
	ro_trg = awg_channels['ro_trg']
	osc_trg = awg_channels['osc_trg']
	return {'ex': amplitude*np.ones(int(round(length*iq_ex.get_clock())),dtype=np.complex),
			'ro': np.zeros(int(round(length*iq_ro.get_clock())),dtype=np.complex),
			'ro_trg':np.zeros(int(round(length*ro_trg.get_clock())), dtype=int),
			'osc_trg':np.zeros(int(round(length*osc_trg.get_clock())), dtype=int)}

def pause(length, awg_channels):
	iq_ex = awg_channels['iq_ex1_q1_F01_min']
	iq_ro = awg_channels['iq_ro_q1']
	ro_trg = awg_channels['ro_trg']
	#osc_trg = awg_channels['osc_trg']
	return {'ex': np.zeros(int(round(length*iq_ex.get_clock())),dtype=np.complex),
			'ro': np.zeros(int(round(length*iq_ro.get_clock())),dtype=np.complex),
			'ro_trg':np.zeros(int(round(length*ro_trg.get_clock())), dtype=int)}#,
			#'osc_trg':np.zeros(int(round(length*osc_trg.get_clock())), dtype=int)}

def set_sequence(pulse_seq, awg_channels):
	initial_delay = 1e-6
	final_delay = 1e-6
	channels = {'ex':awg_channels['iq_ex1_q1_F01_min'], 'ro':awg_channels['iq_ro_q1'], 'ro_trg':awg_channels['ro_trg']}#, 'osc_trg':awg_channels['osc_trg']}
	pulse_seq_padded = [pause(initial_delay, awg_channels)]+[p for p in pulse_seq]+[pause(final_delay, awg_channels)]
	
	pulse_shape = {k:[] for k in channels.keys()}
	for channel, channel_device in channels.items():
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
	def __init__(self, adc, ex_seq, ro_seq, awg_channels):
		self.adc = adc
		self.ex_seq = ex_seq
		self.ro_seq = ro_seq
		self.awg_channels = awg_channels
	
	def calibrate(self):
		from sklearn.metrics import roc_auc_score, roc_curve
		
		# zero sequence
		set_sequence(self.ro_seq, self.awg_channels)
		samples_zero = self.adc.measure()['Voltage']
		# pi pulse sequence
		set_sequence(self.ex_seq+self.ro_seq, self.awg_channels)
		samples_one = self.adc.measure()['Voltage']

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
		roc_auc_score = roc_auc_score([0]*samples.shape[1]+[1]*samples.shape[1], self.predictor(predictions.ravel()))
		
		self.calib_roc_curve = roc_curve
		self.calib_roc_auc_score = roc_auc_score
		
		self.filter = filter
		self.filter_binary = data_reduce.feature_reducer_binary(self.adc, 'Voltage', 1, mean_signal, feature)
		
		
		# Unsupervised predictions
		usl=False
		if usl:
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
	
class tomography:
	def __init__(self, #adc, 
				sz_measurer, awg_channels, proj_seq):
		self.sz_measurer = sz_measurer
		#self.adc = adc
		self.awg_channels = awg_channels
		self.proj_seq = proj_seq
		
		self.adc_reducer = data_reduce.data_reduce(self.sz_measurer.adc)
		self.adc_reducer.filters['SZ'] = self.sz_measurer.filter_binary
		
	def get_points(self):
		return { p:{} for p in self.proj_seq.keys() }
	
	def get_dtype(self):
		return { p:float for p in self.proj_seq.keys() }
	
	def set_prepare_seq(self, seq):
		self.prepare_seq = seq
	
	def measure(self):
		meas = {}
		for p in self.proj_seq.keys():
			set_sequence(self.prepare_seq+self.proj_seq[p], self.awg_channels)
			meas[p] = np.real(np.mean(self.adc_reducer.measure()['SZ']))
		return meas
		
	def get_opts(self):
		return { p:{} for p in self.proj_seq.keys()}
		