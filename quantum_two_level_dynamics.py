from . import sweep
from . import save_pkl
from . import fitting
from . import qjson
import numpy as np
import warnings

# Maybe we should call this "two-level dynamics"?
class quantum_two_level_dynamics:
	def __init__(self, pulse_sequencer, readout_device, **kwargs):
		self.pulse_sequencer = pulse_sequencer
		self.readout_device = readout_device
		self.params = kwargs
		self.readout_device.zero_setter = self.set_zero_sequence
		if 'fitter' in kwargs:
			self.fitter = kwargs['fitter']
		else:
			self.fitter = fitting.S21pm_fit
		try:
			self.readout = qjson.load("setups","readout")
		except Exception as e:
			print('Failed loading readout calibration: '+str(e))
		try:
			self.rabi_fr = qjson.load('two-level','rabi')
		except Exception as e:
			print('Failed loading readout calibration: '+str(e))
			
		self.ex_amplitude = self.params['ex_ampl']
		# Rabi freq depends on excitation pulse parameters
		# If we want to save to config, we should save in 
		# ex_pulse_params=>rabi_freq pair
		# ex_pulse_params should be a serialization of a pulse params o_0
		warnings.filterwarnings('ignore')
		
	def set_zero_sequence(self):
		pg = self.pulse_sequencer
		sequence = [pg.p(None, 0), 
			pg.p('ro_trg', self.readout['trg_length'], pg.rect, 1), 
			pg.p('iq_ro', self.readout['dac_len'], pg.rect, self.readout['amp'])]
		pg.set_seq(sequence)
		
	def Rabi(self,lengths):
		rabi_freqs = dict()
		readout_begin = np.max(lengths)
		pg = self.pulse_sequencer
		ro_channel = 'iq_ro'
		def set_ex_length(length):
			sequence = [pg.p(None, readout_begin-length), 
					pg.p(ex_channel, length, pg.rect, self.ex_amplitude), 
					pg.p('ro_trg', self.readout['trg_length'], pg.rect, 1), 
					pg.p('iq_ro', self.readout['dac_len'], pg.rect, self.readout['amp'])]
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			else:
				set_seq()
		for ex_channel in self.params['ex_channels']:
			measurement_name = 'Rabi channel {}'.format(ex_channel)
			measurement = sweep.sweep(self.readout_device, (lengths, set_ex_length, 'Rabi pulse length', 's'), filename=measurement_name)
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			rabi_freqs[ex_channel]=fitted_parameters['freq']
			rabi_freqs[ex_channel+'_pi_pulse,1e-9'] = 0.5/fitted_parameters['freq']*1e9
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s, \n Excitation carrier frequency: {3:7.5g}, Readout carrier frequency: {4:7.5g}'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'],
																					 pg.channels[ex_channel].get_frequency(),
																					 pg.channels[ro_channel].get_frequency())
			save_pkl.save_pkl({'type':'Rabi','name': 'qubit{}'.format(self.params['qubit_id'])}, measurement_fitted, annotation=annotation, filename=measurement_name)
		rabi_freqs['amplitude'] = self.ex_amplitude
		if self.params['qubit_id']:
			rabi_freqs['qubit_id'] = self.params['qubit_id']
		self.rabi_fr = rabi_freqs
		qjson.dump(type='two-level',name = 'rabi', params=rabi_freqs)	
		return rabi_freqs

		
	def Ramsey(self,delays,target_freq_offset):
		ramsey = dict()
		pg = self.pulse_sequencer
		def set_delay(delay):
			sequence = [pg.p(None, readout_begin - pi2_pulse),
						pg.p(ex_channel, pi2_pulse, pg.rect, self.ex_amplitude), 
						pg.p(None, delay), 
						pg.p(ex_channel, pi2_pulse, pg.rect, self.ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi)), 
						pg.p('ro_trg', self.readout['trg_length'], pg.rect, 1), 
						pg.p('iq_ro', self.readout['dac_len'], pg.rect, self.readout['amp'])]
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			else:
				set_seq()
		
		for ex_channel in self.params['ex_channels']:
			measurement_name = 'Ramsey (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, ex_channel)
			pi2_pulse = 0.25/self.rabi_fr[ex_channel]
			readout_begin = np.max(delays)+pi2_pulse*2
			measurement = sweep.sweep(self.readout_device, (delays, set_delay, 'Ramsey delay', 's'), filename=measurement_name)
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			ramsey['offset,1e6']=fitted_parameters['freq']/1e6
			ramsey['T2,1e-6']=fitted_parameters['decay']*1e6
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Ramsey', 'name': 'qubit {}'.format(self.params['qubit_id'])}, measurement_fitted, annotation=annotation,filename=measurement_name)
		ramsey['amplitude'] = self.ex_amplitude
		if self.params['qubit_id']:
			ramsey['qubit_id'] = self.params['qubit_id']
		qjson.dump(type='two-level',name = 'ramsey', params=ramsey)
		return ramsey
		
	def Decay(self, delays):
		t1 = dict()
		pg = self.pg
		def set_delay(delay):
			sequence = [pg.p(None, readout_begin - pi_pulse-delay),
						pg.p(ex_channel, pi_pulse, pg.rect, self.ex_amplitude), 
						pg.p(None, delay), 
						pg.p('ro_trg', self.readout['trg_length'], pg.rect, 1), 
						pg.p('iq_ro', self.readout['dac_len'], pg.rect, self.readout['amp'])]
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			else:
				set_seq()
				
		for ex_channel in self.params['ex_channels']:
			measurement_name = 'Decay, excitation channel {0}'.format(ex_channel)
			pi_pulse = 0.5/self.rabi_fr[ex_channel]
			readout_begin = np.max(delays)+pi_pulse
			measurement = sweep.sweep(self.readout_device, (delays, set_delay, 'delay', 's'), filename=measurement_name)
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_fit)
			t1['T1,1e-6'] = fitted_parameters['decay']*1e6
			annotation = 'Decay: {0:4.6g} s'.format(fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Decay', 'name': 'qubit {}'.format(self.params['qubit_id'])}, measurement_fitted, annotation=annotation, filename=measurement_name)
		t1['amplitude'] = self.ex_amplitude
		if self.params['qubit_id']:
			t1['qubit_id'] = self.params['qubit_id']
		qjson.dump(type='two-level',name = 'decay', params=t1)
		return t1
	
	def SpinEcho(self,delays,target_freq_offset):
		echo = dict()
		pg = self.pulse_sequencer
		def set_delay(delay):    
			sequence = [pg.p(None, readout_begin-pi2_pulse),
					pg.p(ex_channel, pi2_pulse, pg.rect, self.ex_amplitude), 
					pg.p(None, delay), 
					pg.p(ex_channel, pi2_pulse*2, pg.rect, self.ex_amplitude), 
					pg.p(None, delay), 
					pg.p(ex_channel, pi2_pulse, pg.rect, self.ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi)), 
					pg.p('ro_trg', self.readout['trg_length'], pg.rect, 1), 
					pg.p('iq_ro', self.readout['dac_len'], pg.rect, self.readout['amp'])]
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			else:
				set_seq()
		
		for ex_channel in self.params['ex_channels']:
			measurement_name = 'SpinEcho (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, ex_channel)
			pi2_pulse = 0.25/self.rabi_fr[ex_channel]
			readout_begin = np.max(delays)+pi2_pulse*2
			measurement = sweep.sweep(self.readout_device, (delays, set_delay, 'SpinEcho delay', 's'), filename=measurement_name)
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			echo['offset,1e6']=fitted_parameters['freq']/1e6
			echo['T2_echo,1e-6']=fitted_parameters['decay']*1e6
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'SpinEcho', 'name': 'qubit {}'.format(self.params['qubit_id'])}, measurement_fitted, annotation=annotation,filename=measurement_name)
		echo['amplitude'] = self.ex_amplitude
		if self.params['qubit_id']:
			echo['qubit_id'] = self.params['qubit_id']
		qjson.dump(type='two-level',name = 'echo', params=echo)
		return echo