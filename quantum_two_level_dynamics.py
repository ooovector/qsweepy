from . import sweep
from . import save_pkl
from . import fitting
from . import qjson
import numpy as np
import warnings

# Maybe we should call this "two-level dynamics"?
class quantum_two_level_dynamics:
	def __init__(self, pulse_sequencer, readout_device, ex_channel, ro_channel, ro_sequence, ex_amplitude, qubit_id=None, shuffle=False, plot_separate_thread=True, plot=True, **kwargs):
		self.pulse_sequencer = pulse_sequencer
		self.readout_device = readout_device
		self.ro_channel = ro_channel
		self.ex_channel = ex_channel
		self.ro_sequence = ro_sequence
		self.rabi_rect_ex_amplitude = None
		self.qubit_id = qubit_id
		self.shuffle = shuffle
		self.plot_separate_thread = plot_separate_thread
		self.measurement_name_comment = ''
		self.plot = plot
		# self.params = kwargs #### TODO: FIX THIS DESIGN
		if 'fitter' in kwargs:
			self.fitter = kwargs['fitter']
		else:
			self.fitter = fitting.S21pm_fit
		try:
			self.readout = qjson.load("setups","readout")
		except Exception as e:
			print('Failed loading readout calibration: '+str(e))
		try:
			self.rabi_rect = qjson.load('two-level-rabi-rect', self.build_calibration_filename())
		except Exception as e:
			print('Failed loading rabi frequency calibration: '+str(e))
			
		self.ex_amplitude = ex_amplitude
		# Rabi freq depends on excitation pulse parameters
		# If we want to save to config, we should save in 
		# ex_pulse_params=>rabi_freq pair
		# ex_pulse_params should be a serialization of a pulse params o_0
		warnings.filterwarnings('ignore')
	
	def get_measurement_name_comment(self):
		if self.measurement_name_comment:
			return ' '+self.measurement_name_comment
		else:
			return ''
	
	def load_calibration(self):
		self.Rabi_rect_result = qjson.load(type='two-level-rabi-rect',name = self.build_calibration_filename())
	
	def set_zero_sequence(self):
		self.pulse_sequencer.set_seq(self.ro_sequence)
	
	def build_calibration_filename(self):
		return 'carrier-{0:7.5g}-amplitude-{1:7.5g}'.format(self.pulse_sequencer.channels[self.ex_channel].get_frequency(), self.ex_amplitude)
	
	def Rabi_2d_rect(self,lengths,frequencies):
		ignore_calibration_drift_previous = self.pulse_sequencer.channels[self.ex_channel].get_ignore_calibration_drift()
		frequency_previous = self.pulse_sequencer.channels[self.ex_channel].get_frequency()
		try:
			self.pulse_sequencer.channels[self.ex_channel].set_ignore_calibration_drift(True)
			readout_begin = np.max(lengths)
			pg = self.pulse_sequencer
			sequence = []
			def prepare_set_ex_length(length):
				pass
			def set_ex_length(length): 
				nonlocal sequence
				sequence = [pg.p(None, readout_begin-length), 
							pg.p(self.ex_channel, length, pg.rect, self.ex_amplitude)]+self.ro_sequence
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
					set_seq()
			def set_frequency(frequency):
				self.pulse_sequencer.channels[self.ex_channel].set_frequency(frequency)
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
					set_seq()
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
				self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

			measurement_name = 'Rabi 2D rectangular channel {}'.format(self.ex_channel)+self.get_measurement_name_comment()
			root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
			root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
			measurement = sweep.sweep(self.readout_device, (lengths, set_ex_length, 'Rabi pulse length', 's'), 
														   (frequencies, set_frequency, 'Freqeuency', 'Hz'),
														   filename=measurement_name, shuffle=self.shuffle, 
														   root_dir=root_dir,
														   plot_separate_thread= self.plot_separate_thread,
														   plot=self.plot)
			
			annotation = 'Excitation carrier frequency: {0:7.5g}, Readout carrier frequency: {1:7.5g}'.format(
																						pg.channels[self.ex_channel].get_frequency(),
																						pg.channels[self.ro_channel].get_frequency())
			save_pkl.save_pkl({'type':'Rabi 2D','name': 'qubit{}'.format(self.qubit_id)}, measurement, annotation=annotation, filename=measurement_name, location=root_dir)

			del measurement, set_ex_length, set_seq
		finally:
			self.pulse_sequencer.channels[self.ex_channel].set_ignore_calibration_drift(ignore_calibration_drift_previous)
			self.pulse_sequencer.channels[self.ex_channel].set_frequency(frequency_previous)
	
	def Rabi_rect(self,lengths):
		readout_begin = np.max(lengths)
		pg = self.pulse_sequencer
		sequence = []
		def set_ex_length(length): 
			nonlocal sequence
			sequence = [pg.p(None, readout_begin-length), 
						pg.p(self.ex_channel, length, pg.rect, self.ex_amplitude)]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

		measurement_name = 'Rabi rectangular channel {}'.format(self.ex_channel)+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		measurement = sweep.sweep(self.readout_device, (lengths, set_ex_length, 'Rabi pulse length', 's'), 
								  filename=measurement_name, 
								  shuffle=self.shuffle, 
								  root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
		self.Rabi_rect_result = {}
		self.Rabi_rect_result['rabi_rect_freq']=fitted_parameters['freq']
		self.Rabi_rect_result['rabi_rect_decay']=fitted_parameters['decay']
		self.Rabi_rect_result['rabi_carrier']=pg.channels[self.ex_channel].get_frequency()
		self.Rabi_rect_result['rabi_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.Rabi_rect_result['rabi_ex_amplitude']=self.ex_amplitude
		self.Rabi_rect_result['qubit_id']=self.qubit_id
		
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s, \n Excitation carrier frequency: {3:7.5g}, Readout carrier frequency: {4:7.5g}'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'],
																					 pg.channels[self.ex_channel].get_frequency(),
																					 pg.channels[self.ro_channel].get_frequency())
		save_pkl.save_pkl({'type':'Rabi','name': 'qubit{}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name, location=root_dir)
		qjson.dump(type='two-level-rabi-rect',name = self.build_calibration_filename(), params=self.Rabi_rect_result)	
		self.rabi_rect_ex_amplitude = self.ex_amplitude
		del measurement, measurement_fitted, set_ex_length, set_seq
		return self.Rabi_rect_result

	def Ramsey(self,delays,target_freq_offset, *params):
		if self.rabi_rect_ex_amplitude != self.ex_amplitude:
			self.load_calibration()
		pg = self.pulse_sequencer
		sequence = []
		def set_delay(delay): 
			nonlocal sequence
			sequence = [pg.p(None, readout_begin - pi2_pulse),
						pg.p(self.ex_channel, pi2_pulse, pg.rect, self.ex_amplitude), 
						pg.p(None, delay), 
						pg.p(self.ex_channel, pi2_pulse, pg.rect, self.ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
	
		measurement_name = 'Ramsey (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, self.ex_channel)+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		pi2_pulse = 0.25/self.Rabi_rect_result['rabi_rect_freq']
		readout_begin = np.max(delays)+pi2_pulse*2
		measurement = sweep.sweep(self.readout_device, 
								  (delays, set_delay, 'Ramsey delay', 's'), 
								  *params, 
								  filename=measurement_name, 
								  shuffle=self.shuffle, 
								  root_dir = root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		try:
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			self.Ramsey_result = {}
			self.Ramsey_result['Ramsey_freq']=fitted_parameters['freq']
			self.Ramsey_result['Ramsey_decay']=fitted_parameters['decay']
			self.Ramsey_result['Ramsey_carrier']=pg.channels[self.ex_channel].get_frequency()
			self.Ramsey_result['Ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.Ramsey_result['Ramsey_ex_amplitude'] = self.ex_amplitude
			self.Ramsey_result['qubit_id'] = self.qubit_id
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																				 fitted_parameters['freq'], 
																				 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Ramsey', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name,location=root_dir)
		
			qjson.dump(type='two-level-ramsey',name=self.build_calibration_filename(), params=self.Ramsey_result)
			del measurement, measurement_fitted, set_delay, set_seq
			return self.Ramsey_result
		except:
			return measurement
		
	def decay(self, delays, *params):
		pg = self.pulse_sequencer
		if self.rabi_rect_ex_amplitude != self.ex_amplitude:
			self.load_calibration()
		sequence = []
		def set_delay(delay): 
			nonlocal sequence
			sequence = [pg.p(None, readout_begin - pi_pulse-delay),
						pg.p(self.ex_channel, pi_pulse, pg.rect, self.ex_amplitude), 
						pg.p(None, delay)]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
				
		measurement_name = 'Decay, excitation channel {0}'.format(self.ex_channel)+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		pi_pulse = 0.5/self.Rabi_rect_result['rabi_rect_freq']
		readout_begin = np.max(delays)+pi_pulse
		measurement = sweep.sweep(self.readout_device, 
								  (delays, set_delay, 'delay', 's'), 
								  *params,
								  filename=measurement_name, 
								  shuffle=self.shuffle, 
								  root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		try:
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_fit)
			self.decay_result = {}
			self.decay_result['decay'] = fitted_parameters['decay']
			self.decay_result['decay_carrier']=pg.channels[self.ex_channel].get_frequency()
			self.decay_result['decay_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.decay_result['decay_ex_amplitude'] = self.ex_amplitude
			self.decay_result['qubit_id'] = self.qubit_id
			annotation = 'Decay: {0:4.6g} s'.format(fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Decay', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name,location=root_dir)
		
			qjson.dump(type='two-level-decay',name=self.build_calibration_filename(), params=self.decay_result)
			del measurement, measurement_fitted, set_delay, set_seq
			return self.decay_result
		except:
			return measurement
	
	def spin_echo(self,delays,target_freq_offset,*params):
		if self.rabi_rect_ex_amplitude != self.ex_amplitude:
			self.load_calibration()
		pg = self.pulse_sequencer
		sequence = []
		def set_delay(delay): 
			nonlocal sequence
			sequence = [pg.p(None, readout_begin-pi2_pulse),
					pg.p(self.ex_channel, pi2_pulse, pg.rect, self.ex_amplitude), 
					pg.p(None, delay), 
					pg.p(self.ex_channel, pi2_pulse*2, pg.rect, self.ex_amplitude), 
					pg.p(None, delay), 
					pg.p(self.ex_channel, pi2_pulse, pg.rect, self.ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		
		measurement_name = 'Spin echo (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, self.ex_channel)+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		pi2_pulse = 0.25/self.Rabi_rect_result['rabi_rect_freq']
		readout_begin = np.max(delays)+pi2_pulse*2
		measurement = sweep.sweep(self.readout_device, (delays, set_delay, 'Spin echo delay', 's'), *params, filename=measurement_name, shuffle=self.shuffle, root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		try:
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			self.spin_echo_result = {}
			self.spin_echo_result['ramsey_freq']=fitted_parameters['freq']
			self.spin_echo_result['ramsey_decay']=fitted_parameters['decay']
			self.spin_echo_result['ramsey_carrier']=pg.channels[self.ex_channel].get_frequency()
			self.spin_echo_result['ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.spin_echo_result['ramsey_ex_amplitude'] = self.ex_amplitude
			self.spin_echo_result['qubit_id'] = self.qubit_id
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'spin echo', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name,location=root_dir)

			qjson.dump(type='two-level-spin-echo',name=self.build_calibration_filename(), params=self.spin_echo_result)
			del measurement, measurement_fitted, set_delay, set_seq
			return self.spin_echo_result
		except:
			return measurement