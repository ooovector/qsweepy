## awg_iq class
# two channels of awgs and a local oscillator used to feed a single iq mixer

# maximum settings of the current mixer that should not be exceeded

import numpy as np
import logging
from save_pkl import *
from config import get_config
from matplotlib import pyplot as plt

class awg_iq:
	def __init__(self, awg_I, awg_Q, awg_ch_I, awg_ch_Q, lo):#, mixer):
		self.awg_I = awg_I
		self.awg_Q = awg_Q
		self.awg_ch_I = awg_ch_I
		self.awg_ch_Q = awg_ch_Q
		
		self.lo = lo
		self._if = 0
		self.frequency = lo.get_frequency()
		self.calibrations = {}
		self.sideband_id = 0
		#self.mixer = mixer
	
	def get_nop(self):
		I_nop = self.awg_I.get_nop()
		Q_nop = self.awg_Q.get_nop()
		if I_nop != Q_nop:
			raise ValueError('Number of points in I channel awg and Q channel should coincide')
		return I_nop
	
	def get_clock(self):
		I_clock = self.awg_I.get_clock()
		Q_clock = self.awg_Q.get_clock()
		if I_clock != Q_clock:
			raise ValueError('Clock rate in I channel awg and Q channel should coincide')
		return I_clock
		
	def set_nop(self, nop):
		self.awg_I.set_nop(nop)
		self.awg_Q.set_nop(nop)
		
	def set_clock(self, clock):
		self.awg_I.set_nop(clock)
		self.awg_Q.set_nop(clock)
		
	def set_status(self, status):
		self.awg_I.set_status(status, channel=self.awg_ch_I)
		self.awg_Q.set_status(status, channel=self.awg_ch_Q)
	
	def __set_waveform_IQ_cmplx(self, waveform_cmplx):
		waveform_I = np.real(waveform_cmplx)
		waveform_Q = np.imag(waveform_cmplx)
		
		self.awg_I.set_waveform(waveform_I, channel=self.awg_ch_I)
		self.awg_Q.set_waveform(waveform_Q, channel=self.awg_ch_Q)
		
		#if np.any(np.abs(waveform_I)>1.0) or np.any(np.abs(waveform_Q)>1.0):
			#logging.warning('Waveform clipped!')
	
	def set_waveform(self, waveform_cmplx):
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		waveform_if = waveform_cmplx*np.exp(1j*2*np.pi*t*self.get_if())
		
		waveform_I = np.real(self.calibrations[self.cname()]['I']*waveform_if)+np.real(self.calibrations[self.cname()]['dc'])
		waveform_Q = np.imag(self.calibrations[self.cname()]['Q']*waveform_if)+np.imag(self.calibrations[self.cname()]['dc'])
		
		self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)
		return np.max([np.max(np.abs(waveform_I)), np.max(np.abs(waveform_Q))])
		
	def set_trigger_mode(self, mode):
		self.awg_I.set_trigger_mode(mode)
		self.awg_Q.set_trigger_mode(mode)
	
	# clip DC to prevent mixer damage
	def clip_dc(self, x):
		x = [np.real(x), np.imag(x)]
		for c in (0,1):
			if x[c] < -0.3:
				x[c] = -0.3
			if x[c] > 0.3:
				x[c] = 0.3
		x = x[0] + 1j * x[1]
		return x
	
	def _set_dc(self, x):
		x = self.clip_dc(x)	
		self.__set_waveform_IQ_cmplx([x]*self.get_nop())
		
	def _set_if_cw(self, dc, I, Q):
		t = np.linspace(0, self.get_nop()/self.get_clock(), self.get_nop(), endpoint=False)
		dc = self.clip_dc(dc)
		waveform_I = np.real(I*np.exp(2*np.pi*1j*t*self.get_if()))+np.real(dc)
		waveform_Q = np.imag(Q*np.exp(2*np.pi*1j*t*self.get_if()))+np.imag(dc)
		self.__set_waveform_IQ_cmplx(waveform_I+1j*waveform_Q)
		return np.max([np.max(np.abs(waveform_I)), np.max(np.abs(waveform_Q))])
	
	def cname(self):
		return ('if', self.get_if()), ('frequency', self.get_frequency()), ('sideband_id', self.sideband_id)
				
	def get_calibration(self, sa=None):
		calibration_path = get_config().get('datadir')+'/calibrations/'
		filename = 'IQ-if{0:3.2g}-rf{1:3.2g}-sb-{2}'.format(self.get_if(), self.get_frequency(), self.sideband_id)
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
		calibration_path = get_config().get('datadir')+'/calibrations/'
		print (calibration_path)
		filename = 'IQ-if{0:3.2g}-rf{1:3.2g}-sb-{2}'.format(self.get_if(), self.get_frequency(), self.sideband_id)
		save_pkl(None, self.calibrations[self.cname()], location=calibration_path, filename=filename, plot=False)
	
	def _calibrate_cw_sa(self, sa, num_sidebands = 7):
		from scipy.optimize import fmin
		dc = self._calibrate_zero_sa(sa)
		sa.set_centerfreq(self.lo.get_frequency())
		sa.set_span((num_sidebands-1)*self.get_if())
		sa.set_nop(num_sidebands)
		sa.set_detector('POS')
		sa.set_res_bw(1e5)
		sa.set_video_bw(1e4)
		self.set_trigger_mode('CONT')
		self.lo.set_status(True)
		sideband_ids = np.asarray(np.linspace(-(num_sidebands-1)/2, (num_sidebands-1)/2, num_sidebands), dtype=int)

		self.awg_I.run()
		self.awg_Q.run()
		solution = [np.real(dc), np.imag(dc), 0.5, 0.5, 0.5, 0.5]
		for iter_id in range(3):
			def tfunc(x):
				dc = x[0] + x[1]*1j
				I =  x[2] + x[3]*1j
				Q =  x[4] + x[5]*1j
				max_amplitude = self._set_if_cw(dc, I, Q)
				if max_amplitude < 1:
					clipping = 0
				else:
					clipping = (max_amplitude-1)
				result = sa.measure()['Power'].ravel()
				bad_power = np.sum(10**((result[sideband_ids != self.sideband_id])/20))
				good_power = np.sum(10**((result[sideband_ids==self.sideband_id])/20))
				bad_power_dbm = np.log10(bad_power)*20
				good_power_dbm = np.log10(good_power)*20
				print ('dc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\tB: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}\r'.format(dc, I, Q, bad_power_dbm, good_power_dbm, clipping))
				return -good_power/bad_power+np.abs(good_power/bad_power)*10*clipping			
			solution = fmin(tfunc, solution, maxiter=50, xtol=2**(-14))
			score = tfunc(solution)
			
		self.calibrations[self.cname()] = {'dc': self.clip_dc(solution[0]+solution[1]*1j),
							'I': solution[2]+solution[3]*1j,
							'Q': solution[4]+solution[5]*1j,
							'score': score,
							'num_sidebands': num_sidebands}
		
		return self.calibrations[self.cname()]
			
	def _calibrate_zero_sa(self, sa):
		from scipy.optimize import fmin
		
		sa.set_centerfreq(self.lo.get_frequency())
		sa.set_span(0)
		sa.set_nop(1)
		sa.set_detector('rms')
		sa.set_res_bw(2e5)
		sa.set_video_bw(4e4)
		self.set_trigger_mode('CONT')
		self.lo.set_status(True)
		
		def tfunc(x):
			self.awg_I.stop()
			self.awg_Q.stop()
			self._set_dc(x[0]+x[1]*1j)
			self.awg_I.run()
			self.awg_Q.run()
			result = sa.measure()['Power'].ravel()[0]
			print (x, result)
			return result
		
		solution = fmin(tfunc, [0.1,0.1], maxiter=30, xtol=2**(-14))
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
	
	def get_frequency(self):
		return self.frequency