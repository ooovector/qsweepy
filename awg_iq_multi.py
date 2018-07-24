## awg_iq class
# two channels of awgs and a local oscillator used to feed a single iq mixer

# maximum settings of the current mixer that should not be exceeded

import numpy as np
import logging
from .save_pkl import *
from .config import get_config
from matplotlib import pyplot as plt

class carrier:
	def __init__(self, parent):#, mixer):
		"""
		"""
		self._if = 0
		self.frequency = parent.lo.get_frequency()
		self.parent = parent
		self.status = 1

		
	def get_nop(self):
		return self.parent.get_nop()
	def get_clock(self):
		return self.parent.get_clock()
	def set_nop(self, nop):
		self.parent.set_nop(nop)
	def set_clock(self, clock):
		self.parent.set_clock(clock)
	def set_status(self, status):
		self.status = status
		self.parent.assemble_waveform()
	def get_waveform(self):
		if not hasattr(self, 'waveform'):
			self.waveform = np.zeros(self.get_nop(), dtype=np.complex)
		return self.waveform			
	def set_if(self, _if):
		self._if = _if
	def get_if(self):
		return self._if
	def set_frequency(self, frequency):
		#self.lo.set_frequency(frequency-self.get_sideband_id()*self.get_if())
		#self.frequency = self.lo.get_frequency()+self.get_sideband_id()*self.get_if()
		self._if = frequency - self.parent.lo.get_frequency()
		self.frequency = frequency
	def set_uncal_frequency(self, frequency):
		self.lo.set_frequency(frequency-self.get_if())	
	def get_frequency(self):
		return self.frequency
	def set_waveform(self, waveform):
		self.waveform = waveform
		self.parent.assemble_waveform()
	def freeze(self):
		self.parent.freeze()
	def unfreeze(self):
		self.parent.unfreeze()

class awg_iq_multi:
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
		
		self.carriers = {}
		
		self.lo = lo
		#self._if = 0
		#self.frequency = lo.get_frequency()
		self.dc_calibrations = {}
		self.rf_calibrations = {}
		self.sideband_id = 0
		self.ignore_calibration_drift = False
		#self.mixer = mixer
		self.frozen = False
		
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
		
	
	def __set_waveform_IQ_cmplx(self, waveform_cmplx):
		"""Sets the real part of the waveform on the I channel and the imaginary part of the 
		waveform on the Q channel.
		
		No intermediate frequency multiplication and mixer calibration corrections are performed.
		This is a low-level function that is normally only called for debugging purposes. Pulse 
		sequence generators do not normally call this function, but rather sate_waveform."""
		waveform_I = np.real(waveform_cmplx)
		waveform_Q = np.imag(waveform_cmplx)
		
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
	
	def assemble_waveform(self):
		"""Takes waveforms on all carriers and sums them up."""
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		waveform_I = np.zeros(len(t), dtype=np.complex)
		waveform_Q = np.zeros(len(t), dtype=np.complex)
		waveform_I+=np.real(self.calib_dc(self.dc_cname())['dc'])
		waveform_Q+=np.imag(self.calib_dc(self.dc_cname())['dc'])
		for carrier_id, carrier in self.carriers.items():
			if not carrier.status:
				continue
			waveform_if = carrier.get_waveform()*np.exp(1j*2*np.pi*t*carrier.get_if())
		
			waveform_I += np.real(self.calib_rf(self.rf_cname(carrier))['I']*waveform_if)
			waveform_Q += np.imag(self.calib_rf(self.rf_cname(carrier))['Q']*waveform_if)
			if not self.frozen:
				self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)

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
		
	def _set_if_cw(self, dc, I, Q, _if, half_length):
		from  scipy.signal import gaussian as gaussian
		"""Sets a CW with arbitrary calibration. This functions is invoked by _calibrate_sa 
		to find the optimal values of the I and Q complex amplitudes and dc offsets that correspond 
		to the minimum SFDR."""
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		dc = self.clip_dc(dc)
		if half_length:
			envelope = gaussian(self.get_nop(), self.get_nop()/8)
		else:
			envelope = np.ones(self.get_nop())
		waveform_I = np.real(I*np.exp(2*np.pi*1j*t*_if))*envelope+np.real(dc)
		waveform_Q = np.imag(Q*np.exp(2*np.pi*1j*t*_if))*envelope+np.imag(dc)
		self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)
		return np.max([np.max(np.abs(waveform_I)), np.max(np.abs(waveform_Q))])

	def dc_cname(self):
		return ('lo_frequency', self.lo.get_frequency())
		
	def do_calibration(self, sa=None):
		"""User-level function to sort out mixer calibration matters. Checks if there is a saved calibration for the given
		LO and IF frequencies and loads it.
		When invoked with a spectrum analyzer instance as an argument it perform and save the calibration with the current 
		frequencies.
		"""
		self.get_dc_calibration(sa)
		for carrier_name, carrier in self.carriers.items():
			self.get_rf_calibration(carrier=carrier, sa=sa)
		
	def get_dc_calibration(self, sa=None):
		"""This function is no longer user-level as it only calibrates one carrier frequency. User-level function is do_calibration.
		"""
		calibration_path = get_config()['datadir']+'/calibrations/'
		filename = 'IQ-dc-lo{0:5.3f}GHz'.format(self.lo.get_frequency()/1e9)
		try:
			self.dc_calibrations[self.dc_cname()] = load_pkl(filename, location=calibration_path)
		except Exception as e:
			if not sa:
				logging.error('No ready calibration found and no spectrum analyzer to calibrate')
			else:
				self._calibrate_zero_sa(sa)
				self.save_dc_calibration()
		return self.dc_calibrations[self.dc_cname()]
				
	def save_dc_calibration(self):
		calibration_path = get_config()['datadir']+'/calibrations/'
		print (calibration_path)
		filename = 'IQ-dc-lo{0:5.3f}GHz'.format(self.lo.get_frequency()/1e9)
		save_pkl(None, self.dc_calibrations[self.dc_cname()], location=calibration_path, filename=filename, plot=False)
		
	def rf_cname(self, carrier):
		return ('if', carrier.get_if()), ('frequency', carrier.get_frequency()), ('sideband_id', self.sideband_id)
		
	def dc_cname(self):
		return ('frequency', self.lo.get_frequency())
		
	def get_rf_calibration(self, carrier, sa=None):
		"""User-level function to sort out mxer calibration matters. Checks if there is a saved calibration for the given
		LO and IF frequencies and loads it.
		When invoked with a spectrum analyzer instance as an argument it perform and save the calibration with the current 
		frequencies.
		"""
		calibration_path = get_config()['datadir']+'/calibrations/'
		filename = 'IQ-if{0:3.2g}-rf{1:3.2g}-sb-{2}'.format(carrier.get_if(), carrier.get_frequency(), self.sideband_id)
		try:
			self.rf_calibrations[self.rf_cname(carrier)] = load_pkl(filename, location=calibration_path)
		except Exception as e:
			if not sa:
				logging.error('No ready calibration found and no spectrum analyzer to calibrate')
			else:
				print (e)
				self._calibrate_cw_sa(sa, carrier)
				self.save_rf_calibration(carrier)
		return self.rf_calibrations[self.rf_cname(carrier)]
			
	def save_rf_calibration(self, carrier):
		calibration_path = get_config()['datadir']+'/calibrations/'
		print (calibration_path)
		filename = 'IQ-if{0:3.2g}-rf{1:3.2g}-sb-{2}'.format(carrier.get_if(), carrier.get_frequency(), self.sideband_id)
		save_pkl(None, self.rf_calibrations[self.rf_cname(carrier)], location=calibration_path, filename=filename, plot=False)
		
	def calib_dc(self, cname):
		if self.ignore_calibration_drift:
			if cname not in self.dc_calibrations:
				dc_c = [calib for calib in self.dc_calibrations.values()]
				return dc_c[0]
		if cname not in self.dc_calibrations:
			print ('Calibration not loaded. Use ignore_calibration_drift to use any calibration.')
		return self.dc_calibrations[cname]
	
	def calib_rf(self, cname):
		if self.ignore_calibration_drift:
			if cname not in self.rf_calibrations:
				rf_c = [calib for calib in self.rf_calibrations.values()]
				return rf_c[0]
		if cname not in self.rf_calibrations:
			print ('Calibration not loaded. Use ignore_calibration_drift to use any calibration.')
		return self.rf_calibrations[cname]
	
	def _calibrate_cw_sa(self, sa, carrier, num_sidebands = 3, use_central = False, num_sidebands_final = 9, half_length = True, use_single_sweep=False):
		"""Performs IQ mixer calibration with the spectrum analyzer sa with the intermediate frequency."""
		from scipy.optimize import fmin
		import time
		res_bw = 1e4
		video_bw = 1e3
		if hasattr(sa, 'set_nop') and use_single_sweep:
			sa.set_centerfreq(self.lo.get_frequency())
			sa.set_span((num_sidebands-1)*np.abs(carrier.get_if()))
			sa.set_nop(num_sidebands)
			sa.set_detector('POS')
			sa.set_res_bw(res_bw)
			sa.set_video_bw(video_bw)
			self.set_trigger_mode('CONT')
			sa.set_sweep_time_auto(1)
		else:
			sa.set_detector('rms')
			sa.set_res_bw(res_bw)
			sa.set_video_bw(video_bw)
			sa.set_span(res_bw)
			if hasattr(sa, 'set_nop'): 
				sa.set_sweep_time(50e-3)
				sa.set_nop(1)
		
		self.lo.set_status(True)

		self.awg_I.run()
		self.awg_Q.run()
		solution = [-0.5, 0.2]
		for iter_id in range(1):
			def tfunc(x):
				#dc = x[0] + x[1]*1j
				target_sideband_id = 1 if carrier.get_if()>0 else -1
				sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands-1)/2, num_sidebands), dtype=int)
				I =  0.5
				Q =  x[0] + x[1]*1j
				max_amplitude = self._set_if_cw(self.calib_dc(self.dc_cname())['dc'], I, Q, carrier.get_if(), half_length)
				if max_amplitude < 1:
					clipping = 0
				else:
					clipping = (max_amplitude-1)
				# if we can measure all sidebands in a single sweep, do it
				if hasattr(sa, 'set_nop') and use_single_sweep:
					result = sa.measure()['Power'].ravel()
				else:
				# otherwise, sweep through each sideband
					result = []
					for sideband_id in range(num_sidebands):
						sa.set_centerfreq(self.lo.get_frequency()+(sideband_id-(num_sidebands-1)/2.)*carrier.get_if())
						#time.sleep(0.1)
						#result.append(np.log10(np.sum(10**(sa.measure()['Power']/10)))*10)
						result.append(np.log10(np.sum(10**(sa.measure()['Power']/10)))*10)
						#time.sleep(0.1)
					result = np.asarray(result)
				if use_central:
					bad_sidebands = sideband_ids != target_sideband_id
				else:
					bad_sidebands = np.logical_and(sideband_ids != target_sideband_id, sideband_ids != 0)
				bad_power = np.sum(10**((result[bad_sidebands])/20))
				good_power = np.sum(10**((result[sideband_ids==target_sideband_id])/20))
				bad_power_dbm = np.log10(bad_power)*20
				good_power_dbm = np.log10(good_power)*20
				print ('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(self.calib_dc(self.dc_cname())['dc'], I, Q, bad_power_dbm, good_power_dbm, clipping))
				print (result)
				return -good_power/bad_power+np.abs(good_power/bad_power)*10*clipping			
			#solution = fmin(tfunc, solution, maxiter=75, xtol=2**(-13))
			solution = fmin(tfunc, solution, maxiter=30, xtol=2**(-13))
			num_sidebands = num_sidebands_final
			use_central = True
	
			sa.set_centerfreq(self.lo.get_frequency())
			sa.set_span((num_sidebands-1)*np.abs(carrier.get_if()))
			sa.set_nop(num_sidebands)
			sa.set_detector('POS')
			sa.set_res_bw(res_bw)
			sa.set_video_bw(video_bw)
			self.set_trigger_mode('CONT')
	
			score = tfunc(solution)
			
		self.rf_calibrations[self.rf_cname(carrier)] = {'I': 0.5, 
							'Q': solution[0]+solution[1]*1j,
							'score': score,
							'num_sidebands': num_sidebands}
		
		return self.rf_calibrations[self.rf_cname(carrier)]
		
	def _calibrate_zero_sa(self, sa):
		"""Performs IQ mixer calibration for DC signals at the I and Q inputs."""
		import time
		from scipy.optimize import fmin	
		print(self.lo.get_frequency())
		res_bw = 1e4
		video_bw = 1e2
		sa.set_res_bw(res_bw)
		sa.set_video_bw(video_bw)
		sa.set_detector('rms')
		sa.set_centerfreq(self.lo.get_frequency())
		sa.set_sweep_time(50e-3)
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
		
		#solution = fmin(tfunc, [0.3,0.3], maxiter=30, xtol=2**(-14))
		solution = fmin(tfunc, [0.3,0.3], maxiter=30, xtol=2**(-13))
		x = self.clip_dc(solution[0]+1j*solution[1])
		self.zero = x
		
		self.dc_calibrations[self.dc_cname()] = {'dc': solution[0]+solution[1]*1j}
			
		return self.dc_calibrations[self.dc_cname()]
		
	def freeze(self):
		self.frozen = True
	def unfreeze(self):
		if self.frozen:
			self.frozen = False
			self.assemble_waveform()