## awg_iq class
# two channels of awgs and a local oscillator used to feed a single iq mixer

# maximum settings of the current mixer that should not be exceeded

import numpy as np
import logging
from .save_pkl import *
from .config import get_config
from matplotlib import pyplot as plt

class awg_iq:
	"""Interface for IQ modulation of RF signals wth two AWG channels.
	
	IQ modulation requires two low (intermediate) frequency arbitrary waveform generators for the I and Q 
	connectors of the mixer and one for the LO input connector.
	Modulated signal is output at RF (radio frequency).
	
	Attributes:
		awg_I (:obj:`awg`): Instance of an AWG class (for example, Tektronix AWG5014C or AWG500) 
			that is connected to the I input of the mixer. Should implement the methods get_nop, get_clock,
			set_clock, set_waveform, set_status, set_trigger_mode, run and stop.
		awg_Q (:obj:`awg`): Instance of an AWG class that is connected to the Q input of the mixer. 
			awg_I and awg_Q are normaly the same device (I and Q are connected to different channels of the same device).
		awg_ch_I (int): Channel id of the device awg_I that is connected to the I connector of the mixer.
		awg_ch_Q (int): Channel id of the device awg_I that is connected to the Q connector of the mixer.
		lo (:obj:`psg`): Instance of a sinusodial signal generator. Should implement the methods get_frequency and set_frequency.
		
	"""
	def __init__(self, awg_I, awg_Q, awg_ch_I, awg_ch_Q, lo):#, mixer):
		"""
		"""
		self.awg_I = awg_I
		self.awg_Q = awg_Q
		self.awg_ch_I = awg_ch_I
		self.awg_ch_Q = awg_ch_Q
		
		self.lo = lo
		self._if = 0
		self.frequency = lo.get_frequency()
		self.calibrations = {}
		self.sideband_id = 0
		self.ignore_calibration_drift = False
		self.frozen = False
		#self.mixer = mixer
		
	#@property 
	def get_nop(self):
		"""int: Number of samples in segment."""
		I_nop = self.awg_I.get_nop()
		Q_nop = self.awg_Q.get_nop()
		if I_nop != Q_nop:
			raise ValueError('Number of points in I channel awg and Q channel should coincide')
		return I_nop
	
	def get_clock(self):
		"""int: Sample rate of I and Q channels (complex amplitude envelope)."""
		I_clock = self.awg_I.get_clock()
		Q_clock = self.awg_Q.get_clock()
		if I_clock != Q_clock:
			raise ValueError('Clock rate in I channel awg and Q channel should coincide')
		return I_clock
		
	def set_nop(self, nop):
		"""int: Sets number of samples in segment."""
		self.awg_I.set_nop(nop)
		self.awg_Q.set_nop(nop)
		
	def set_clock(self, clock):
		"""Sets sampling rate."""
		self.awg_I.set_clock(clock)
		self.awg_Q.set_clock(clock)
		
	def set_status(self, status):
		"""Turns on and off the lo and awg channels."""
		self.lo.set_status(status)
		self.awg_I.set_status(status, channel=self.awg_ch_I)
		self.awg_Q.set_status(status, channel=self.awg_ch_Q)
	
	def set_waveform_IQ_cmplx_raw(self, waveform_cmplx):
		self.__set_waveform_IQ_cmplx(waveform_cmplx)	
		
	def __set_waveform_IQ_cmplx(self, waveform_cmplx):
		"""Sets the real part of the waveform on the I channel and the imaginary part of the 
		waveform on the Q channel.
		
		No intermediate frequency multiplication and mixer calibration corrections are performed.
		This is a low-level function that is normally only called for debugging purposes. Pulse 
		sequence generators do not normally call this function, but rather sate_waveform."""
		waveform_I = np.real(waveform_cmplx)
		waveform_Q = np.imag(waveform_cmplx)
		
		self.awg_I.stop()
		if self.awg_I != self.awg_Q:
			self.awg_Q.stop()
		
		self.awg_I.set_waveform(waveform_I, channel=self.awg_ch_I)
		self.awg_Q.set_waveform(waveform_Q, channel=self.awg_ch_Q)
		
		self.awg_I.run()
		if self.awg_I != self.awg_Q:
			self.awg_Q.run()
		#import matplotlib.pyplot as plt
		#plt.plot(waveform_I)
		#plt.plot(waveform_Q)
		
		#if np.any(np.abs(waveform_I)>1.0) or np.any(np.abs(waveform_Q)>1.0):
			#logging.warning('Waveform clipped!')
	
	def calib(self, cname):
		if self.ignore_calibration_drift:
			if cname not in self.calibrations:
				c = [calib for calib in self.calibrations.values()]
				return c[0]
		if cname not in self.calibrations:
			print ('Calibration not loaded. Use ignore_calibration_drift to use any calibration.')
		return self.calibrations[cname]
	
	def set_waveform(self, waveform_cmplx):
		"""Sets the real part of the waveform on the I channel and the imaginary part of the 
		waveform on the Q channel.
		
		This function multiplies the waveform with the intermediate frequency oscillation and sideband 
		calibration amplitudes. he function accepts a complex waveform envelope and effectively forms a 
		RF output waveform of the given envelope at the frequency given by the frequency attribute."""
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		waveform_if = waveform_cmplx*np.exp(1j*2*np.pi*t*self.get_if())
		
		waveform_I = np.real(self.calib(self.cname())['I']*waveform_if)+np.real(self.calib(self.cname())['dc'])
		waveform_Q = np.imag(self.calib(self.cname())['Q']*waveform_if)+np.imag(self.calib(self.cname())['dc'])
		
		self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)
		self.waveform = waveform_cmplx
		return np.max([np.max(np.abs(waveform_I)), np.max(np.abs(waveform_Q))])
		
	def get_waveform(self):
		return self.waveform
		
	def set_trigger_mode(self, mode):
		self.awg_I.set_trigger_mode(mode)
		self.awg_Q.set_trigger_mode(mode)
	
	# clip DC to prevent mixer damage
	def clip_dc(self, x):
		"""Clips the dc complonent of the output at both channels of the AWG to prevent mixer damage."""
		x = [np.real(x), np.imag(x)]
		for c in (0,1):
			if x[c] < -0.5:
				x[c] = -0.5
			if x[c] > 0.5:
				x[c] = 0.5
		x = x[0] + 1j * x[1]
		return x
	
	def _set_dc(self, x):
		"""Clips the dc complonent of the output at both channels of the AWG to prevent mixer damage."""
		x = self.clip_dc(x)	
		self.__set_waveform_IQ_cmplx([x]*self.get_nop())
		
	def _set_if_cw(self, dc, I, Q):
		"""Sets a CW with arbitrary calibration. This functions is invoked by _calibrate_sa 
		to find the optimal values of the I and Q complex amplitudes and dc offsets that correspond 
		to the minimum SFDR."""
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		dc = self.clip_dc(dc)
		waveform_I = np.real(I*np.exp(2*np.pi*1j*t*self.get_if()))+np.real(dc)
		waveform_Q = np.imag(Q*np.exp(2*np.pi*1j*t*self.get_if()))+np.imag(dc)
		self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)
		return np.max([np.max(np.abs(waveform_I)), np.max(np.abs(waveform_Q))])
	
	def cname(self):
		return ('if', self.get_if()), ('frequency', self.get_frequency()), ('sideband_id', self.sideband_id)
				
	def get_calibration(self, sa=None):
		"""User-level function to sort out mxer calibration matters. Checks if there is a saved calibration for the given
		LO and IF frequencies and loads it.
		When invoked with a spectrum analyzer instance as an argument it perform and save the calibration with the current 
		frequencies.
		"""
		calibration_path = get_config()['datadir']+'/calibrations/'
		filename = 'IQ-if{0:5.3g}-rf{1:5.3g}-sb-{2}'.format(self.get_if(), self.get_frequency(), self.sideband_id)
		try:
			self.calibrations[self.cname()] = load_pkl(filename, location=calibration_path)
		except Exception as e:
			if not sa:
				logging.error('No ready calibration found and no spectrum analyzer to calibrate')
			else:
				self._calibrate_cw_sa(sa)
				self.save_calibration()
		return self.calibrations[self.cname()]
			
	def save_calibration(self):
		calibration_path = get_config()['datadir']+'/calibrations/'
		print (calibration_path)
		filename = 'IQ-if{0:5.3g}-rf{1:5.3g}-sb-{2}'.format(self.get_if(), self.get_frequency(), self.sideband_id)
		save_pkl(None, self.calibrations[self.cname()], location=calibration_path, filename=filename, plot=False)
	
	def _calibrate_cw_sa(self, sa, num_sidebands = 7):
		"""Performs IQ mixer calibration with the spectrum analyzer sa with the intermediate frequency."""
		from scipy.optimize import fmin
		import time
		dc = self._calibrate_zero_sa(sa)
		res_bw = 1e5
		video_bw = 1e4
		if hasattr(sa, 'set_nop'):
			sa.set_centerfreq(self.lo.get_frequency())
			sa.set_span((num_sidebands-1)*self.get_if())
			sa.set_nop(num_sidebands)
			sa.set_detector('POS')
			sa.set_res_bw(res_bw)
			sa.set_video_bw(video_bw)
			self.set_trigger_mode('CONT')
		else:
			sa.set_detector('rms')
			sa.set_res_bw(res_bw)
			sa.set_video_bw(video_bw)
			sa.set_span(res_bw)
		sa.set_sweep_time_auto(1)
			
		self.lo.set_status(True)
		sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands-1)/2, num_sidebands), dtype=int)

		self.awg_I.run()
		self.awg_Q.run()
		solution = [np.real(dc), np.imag(dc), 0.5, 0.5, 0.5, 0.5]
		for iter_id in range(1):
			def tfunc(x):
				dc = x[0] + x[1]*1j
				I =  x[2] + x[3]*1j
				Q =  x[4] + x[5]*1j
				max_amplitude = self._set_if_cw(dc, I, Q)
				if max_amplitude < 1:
					clipping = 0
				else:
					clipping = (max_amplitude-1)
				# if we can measure all sidebands in a single sweep, do it
				if hasattr(sa, 'set_nop'):
					result = sa.measure()['Power'].ravel()
				else:
				# otherwise, sweep through each sideband
					result = []
					for sideband_id in range(num_sidebands):
						sa.set_centerfreq(self.lo.get_frequency()+(sideband_id-(num_sidebands-1)/2.)*self.get_if())
						print (sa.get_centerfreq())
						#time.sleep(0.1)
						#result.append(np.log10(np.sum(10**(sa.measure()['Power']/10)))*10)
						result.append(np.log10(np.sum(sa.measure()['Power']))*10)
						#time.sleep(0.1)
					result = np.asarray(result)
					
				bad_power = np.sum(10**((result[sideband_ids != self.sideband_id])/20))
				good_power = np.sum(10**((result[sideband_ids==self.sideband_id])/20))
				bad_power_dbm = np.log10(bad_power)*20
				good_power_dbm = np.log10(good_power)*20
				print ('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(dc, I, Q, bad_power_dbm, good_power_dbm, clipping))
				print (result)
				return -good_power/bad_power+np.abs(good_power/bad_power)*10*clipping			
			solution = fmin(tfunc, solution, maxiter=75, xtol=2**(-14))
			score = tfunc(solution)
			
		self.calibrations[self.cname()] = {'dc': self.clip_dc(solution[0]+solution[1]*1j),
							'I': solution[2]+solution[3]*1j,
							'Q': solution[4]+solution[5]*1j,
							'score': score,
							'num_sidebands': num_sidebands}
		
		return self.calibrations[self.cname()]
			
	def _calibrate_zero_sa(self, sa):
		"""Performs IQ mixer calibration for DC signals at the I and Q inputs."""
		import time
		from scipy.optimize import fmin	
		print(self.lo.get_frequency())
		res_bw = 1e5
		video_bw = 1e4
		sa.set_res_bw(res_bw)
		sa.set_video_bw(video_bw)
		sa.set_detector('rms')
		sa.set_centerfreq(self.lo.get_frequency())
		sa.set_sweep_time(1e-3)
		#time.sleep(0.1)
		if hasattr(sa, 'set_nop'):
			sa.set_span(0)
			sa.set_nop(1)
			self.set_trigger_mode('CONT')
		else:
			sa.set_span(res_bw)	
		self.lo.set_status(True)
		def tfunc(x):
			self.awg_I.stop()
			self.awg_Q.stop()
			self._set_dc(x[0]+x[1]*1j)
			self.awg_I.run()
			self.awg_Q.run()
			if hasattr(sa, 'set_nop'):
				result = sa.measure()['Power'].ravel()[0]
			else:
				#result = np.log10(np.sum(10**(sa.measure()['Power']/10)))*10
				result = np.log10(np.sum(sa.measure()['Power']))*10
			print (x, result)
			return result
		
		solution = fmin(tfunc, [0.3,0.3], maxiter=30, xtol=2**(-14))
		x = self.clip_dc(solution[0]+1j*solution[1])
		self.zero = x
		
		return x
	
	def set_if(self, _if):
		self._if = _if
	
	def get_if(self):
		return self._if
		
	def set_sideband_id(self, sideband_id):
		self.sideband_id = sideband_id
	def get_sideband_id(self):
		return self.sideband_id
	
	def set_frequency(self, frequency):
		self.lo.set_frequency(frequency-self.get_sideband_id()*self.get_if())
		self.frequency = self.lo.get_frequency()+self.get_sideband_id()*self.get_if()
		
	def set_uncal_frequency(self, frequency):
		self.lo.set_frequency(frequency-self.get_sideband_id()*self.get_if())	
	
	def get_frequency(self):
		return self.frequency
	def freeze(self):
		self.frozen = True
	def unfreeze(self):
		if self.frozen:
			self.frozen = False
			#self.assemble_waveform()