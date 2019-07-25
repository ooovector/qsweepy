from . import sweep
from . import save_pkl
from . import fitting
from . import qjson
import numpy as np
import warnings
from . import pulses
# Maybe we should call this "two-level dynamics"? WUT THE FFFFFUUUUUUCK
class quantum_two_level_dynamics:
	def __init__(self, pulse_sequencer, readout_device, ex_channel, ro_channel, ro_sequence, ex_amplitude, readout_measurement_name, qubit_id=None, shuffle=False, plot_separate_thread=True, plot=True, **kwargs):
		self.pulse_sequencer = pulse_sequencer
		self.readout_device = readout_device
		self.readout_measurement_name = readout_measurement_name
		self.ro_channel = ro_channel
		if type(ex_channel) is str:
			self.ex_channels = (ex_channel, )
			self.ex_amplitudes = (ex_amplitude, )
		else:
			self.ex_channels = ex_channel
			self.ex_amplitudes = ex_amplitude
		self.ro_sequence = ro_sequence
		#self.rabi_rect_ex_amplitude = None
		self.qubit_id = qubit_id
		self.shuffle = shuffle
		self.plot_separate_thread = plot_separate_thread
		self.measurement_name_comment = ''
		self.plot = plot
		self.comment = ''
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
		try:
			self.Ramsey_result = qjson.load(type='two-level-ramsey',name=self.build_calibration_filename())
		except:
			pass
	
	def set_zero_sequence(self):
		self.pulse_sequencer.set_seq(self.ro_sequence)
	
	def build_calibration_filename(self):
		return '-'.join(['c-{0}-f-{1:.5g}-amp-r{2:.5g}-i{3:.5g}'.format(c, self.pulse_sequencer.channels[c].get_frequency(), np.real(a), np.imag(a)) for c, a in zip(self.ex_channels, self.ex_amplitudes)])+self.comment
	
	def get_pi_pulse_sequence(self, phase,max_rabi_freq=50e6):
		pg = self.pulse_sequencer
		rabi_freq = np.min([self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq])
		amp = rabi_freq/self.Rabi_rect_result['rabi_rect_freq']
		
		tail_phase_per_amplitude = self.Rabi_rect_result['rabi_rect_phase']/np.pi
		tail_phase_per_amplitude = (np.round(tail_phase_per_amplitude) - tail_phase_per_amplitude)/2.
		print (amp, tail_phase_per_amplitude, rabi_freq, self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq)
		pi_pulse_length = (0.5-tail_phase_per_amplitude)/rabi_freq
		
		print (pi_pulse_length)
		
		if self.Rabi_rect_result['rabi_tail_length']<=0:
			channel_pulses = [(c, pg.rect, amp*a*np.exp(1j*phase)) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		else:
			channel_pulses = [(c, pg.rect_cos, amp*a*np.exp(1j*phase), self.Rabi_rect_result['rabi_tail_length']) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		sequence = [pg.pmulti(pi_pulse_length, *tuple(channel_pulses))]
		return sequence
		
	def get_pi2_pulse_sequence(self, phase,max_rabi_freq=50e6):
		pg = self.pulse_sequencer
		rabi_freq = np.min([self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq])
		amp = rabi_freq/self.Rabi_rect_result['rabi_rect_freq']
		
		tail_phase_per_amplitude = self.Rabi_rect_result['rabi_rect_phase']/np.pi
		tail_phase_per_amplitude = (np.round(tail_phase_per_amplitude) - tail_phase_per_amplitude)/2.
		pi2_pulse_length = (0.25-tail_phase_per_amplitude)/rabi_freq
		#print (amp, tail_phase_per_amplitude, rabi_freq, self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq)
		#print (pi2_pulse_length)
		
		if self.Rabi_rect_result['rabi_tail_length']<=0:
			channel_pulses = [(c, pg.rect, amp*a*np.exp(1j*phase)) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		else:
			channel_pulses = [(c, pg.rect_cos, amp*a*np.exp(1j*phase), self.Rabi_rect_result['rabi_tail_length']) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		sequence = [pg.pmulti(pi2_pulse_length, *tuple(channel_pulses))]
		return sequence
	
	def get_rotation_pulse_sequence(self, rotation_angle,phase,max_rabi_freq=50e6):
		pg = self.pulse_sequencer
		rabi_freq = np.min([self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq])
		amp = rabi_freq/self.Rabi_rect_result['rabi_rect_freq']
		
		tail_phase_per_amplitude = self.Rabi_rect_result['rabi_rect_phase']/np.pi
		tail_phase_per_amplitude = (np.round(tail_phase_per_amplitude) - tail_phase_per_amplitude)/2.
		pi2_pulse_length = (rotation_angle/(2*np.pi)-tail_phase_per_amplitude)/rabi_freq
		#print (amp, tail_phase_per_amplitude, rabi_freq, self.Rabi_rect_result['rabi_rect_freq'], max_rabi_freq)
		#print (pi2_pulse_length)
		
		if self.Rabi_rect_result['rabi_tail_length']<=0:
			channel_pulses = [(c, pg.rect, amp*a*np.exp(1j*phase)) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		else:
			channel_pulses = [(c, pg.rect_cos, amp*a*np.exp(1j*phase), self.Rabi_rect_result['rabi_tail_length']) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
		sequence = [pg.pmulti(pi2_pulse_length, *tuple(channel_pulses))]
		return sequence
		
	def Rabi_2d_rect(self,lengths,frequencies):
		ignore_calibration_drift_previous = [self.pulse_sequencer.channels[c].get_ignore_calibration_drift() for c in self.ex_channels]
		frequency_previous = [self.pulse_sequencer.channels[c].get_frequency() for x in self.ex_channels]
		try:
			for c in self.ex_channels:
				self.pulse_sequencer.channels[c].set_ignore_calibration_drift(True)
			readout_begin = np.max(lengths)
			pg = self.pulse_sequencer
			sequence = []
			def prepare_set_ex_length(length):
				pass
			def set_ex_length(length): 
				nonlocal sequence
				channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
				sequence = [pg.p(None, readout_begin-length), 
							pg.pmulti(length, *tuple(channel_pulses))]+self.ro_sequence
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
					set_seq()
			def set_frequency(frequency):
				for x in self.ex_channels:
					self.pulse_sequencer.channels[c].set_frequency(frequency)
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
					set_seq()
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
				self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

			measurement_name = 'Rabi 2D rectangular channel {}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
			root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
			root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
			measurement = sweep.sweep(self.readout_device, (lengths, set_ex_length, 'Rabi pulse length', 's'), 
														   (frequencies, set_frequency, 'Freqeuency', 'Hz'),
														   filename=measurement_name, shuffle=self.shuffle, 
														   root_dir=root_dir,
														   plot_separate_thread= self.plot_separate_thread,
														   plot=self.plot)
			
			annotation = 'ex carrier f: {0}, ro carrier f: {1:7.5g}'.format(','.join(['{:7.5g}'.format(pg.channels[c].get_frequency()) for c in self.ex_channels]),
																				 pg.channels[self.ro_channel].get_frequency())
			save_pkl.save_pkl({'type':'Rabi 2D','name': 'qubit{}'.format(self.qubit_id)}, measurement, annotation=annotation, filename=measurement_name, location=root_dir)

			del measurement, set_ex_length, set_seq
		finally:
			for c_id, c in enumerate(self.ex_channels):
				self.pulse_sequencer.channels[c].set_ignore_calibration_drift(ignore_calibration_drift_previous[c_id])
				self.pulse_sequencer.channels[c].set_frequency(frequency_previous[c_id])
	
	def readout_baseline(self):
		self.set_zero_sequence()
		self.readout_baseline = self.readout_device.measure()[readout_measurement_name]
		return self.readout_baseline
	
	def Rabi_rect(self,lengths, tail_length=0e-9, pre_pulse_seq=[], post_pulse_seq=[]):
		readout_begin = np.max(lengths)
		pg = self.pulse_sequencer
		sequence = []
		def set_ex_length(length): 
			nonlocal sequence
			if tail_length > 0:
				channel_pulses = [(c, pg.rect_cos, a, tail_length) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			else:
				channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = pre_pulse_seq+[pg.pmulti(length+2*tail_length, *tuple(channel_pulses))]+post_pulse_seq+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

		measurement_name = 'Rabi rectangular channels {}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
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
		self.Rabi_rect_result['rabi_rect_initial_points']=fitted_parameters['initial_points']
		self.Rabi_rect_result['rabi_rect_phase']=fitted_parameters['phase']
		self.Rabi_rect_result['rabi_rect_amplitudes']=fitted_parameters['amplitudes']
		self.Rabi_rect_result['rabi_rect_freq']=fitted_parameters['freq']
		self.Rabi_rect_result['rabi_rect_decay']=fitted_parameters['decay']
		self.Rabi_rect_result['rabi_carriers']=[pg.channels[c].get_frequency() for c in self.ex_channels]
		self.Rabi_rect_result['rabi_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.Rabi_rect_result['rabi_ex_amplitudes']=self.ex_amplitudes
		self.Rabi_rect_result['rabi_tail_length']=tail_length
		self.Rabi_rect_result['qubit_id']=self.qubit_id
		
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s, \n ex carrier f: {3}, ro carrier f: {4:7.5g}'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'],
																					 ['{:7.5g}'.format(pg.channels[c].get_frequency()) for c in self.ex_channels],
																					 pg.channels[self.ro_channel].get_frequency())
		save_pkl.save_pkl({'type':'Rabi','name': 'qubit{}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name, location=root_dir)
		qjson.dump(type='two-level-rabi-rect',name = self.build_calibration_filename(), params=self.Rabi_rect_result)	
		#self.rabi_rect_ex_amplitude = self.ex_amplitude
		del measurement, measurement_fitted, set_ex_length, set_seq
		return self.Rabi_rect_result

	def Rabi_echo_rect(self, lengths, interleaved_pulse_seq=[], pre_pulse_seq=[]):
		pg = self.pulse_sequencer
		sequence = []
		def set_ex_length(length): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			channel_pulses_inverse = [(c, pg.rect, -a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = pre_pulse_seq+\
						[pg.pmulti(length, *tuple(channel_pulses))]+\
						interleaved_pulse_seq+\
						[pg.pmulti(length, *tuple(channel_pulses_inverse))]+\
						self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

		measurement_name = 'Rabi echo rectangular channels {}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		measurement = sweep.sweep(self.readout_device, (lengths, set_ex_length, 'Rabi pulse length', 's'), 
								  filename=measurement_name, 
								  shuffle=self.shuffle, 
								  root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
		self.Rabi_echo_rect_result = {}
		self.Rabi_echo_rect_result['rabi_rect_initial_points']=fitted_parameters['initial_points']
		self.Rabi_echo_rect_result['rabi_rect_phase']=fitted_parameters['phase']
		self.Rabi_echo_rect_result['rabi_rect_amplitudes']=fitted_parameters['amplitudes']
		self.Rabi_echo_rect_result['rabi_rect_freq']=fitted_parameters['freq']
		self.Rabi_echo_rect_result['rabi_rect_decay']=fitted_parameters['decay']
		self.Rabi_echo_rect_result['rabi_carriers']=[pg.channels[c].get_frequency() for c in self.ex_channels]
		self.Rabi_echo_rect_result['rabi_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.Rabi_echo_rect_result['rabi_ex_amplitudes']=self.ex_amplitudes
		self.Rabi_echo_rect_result['qubit_id']=self.qubit_id
		
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s, \n ex carrier f: {3}, ro carrier f: {4:7.5g}'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'],
																					 ['{:7.5g}'.format(pg.channels[c].get_frequency()) for c in self.ex_channels],
																					 pg.channels[self.ro_channel].get_frequency())
		save_pkl.save_pkl({'type':'Rabi-echo','name': 'qubit{}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name, location=root_dir)
		qjson.dump(type='two-level-rabi-echo-rect',name = self.build_calibration_filename(), params=self.Rabi_echo_rect_result)	
		#self.rabi_rect_ex_amplitude = self.ex_amplitude
		del measurement, measurement_fitted, set_ex_length, set_seq
		return self.Rabi_echo_rect_result
		
	def Rabi_rect_amplitude(self,amplitudes,length):
		readout_begin = length
		pg = self.pulse_sequencer
		sequence = []
		def set_ex_amplitude(amplitude): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a*amplitude) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = [pg.p(None, readout_begin-length), 
						pg.pmulti(length, *tuple(channel_pulses))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

		measurement_name = 'Rabi rectangular amplitude channels {}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		measurement = sweep.sweep(self.readout_device, (amplitudes, set_ex_amplitude, 'Rabi amplitude'), 
								  filename=measurement_name, 
								  shuffle=self.shuffle, 
								  root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
		self.Rabi_rect_amplitude_result = {}
		self.Rabi_rect_amplitude_result['rabi_rect_freq']=fitted_parameters['freq']
		self.Rabi_rect_amplitude_result['rabi_rect_decay']=fitted_parameters['decay']
		self.Rabi_rect_amplitude_result['rabi_carriers']=[pg.channels[c].get_frequency() for c in self.ex_channels]
		self.Rabi_rect_amplitude_result['rabi_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.Rabi_rect_amplitude_result['rabi_ex_amplitudes']=self.ex_amplitudes
		self.Rabi_rect_amplitude_result['qubit_id']=self.qubit_id
		
		annotation = 'Phase: {0:4.4g} rad, Per-amplitude-rabi-freq: {1:4.4g}, Decay: {2:4.4g}, \n ex carrier f: {3}, ro carrier f: {4:7.5g}'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq']/(length), 
																					 fitted_parameters['decay'],
																					 ['{:7.5g}'.format(pg.channels[c].get_frequency()) for c in self.ex_channels],
																					 pg.channels[self.ro_channel].get_frequency())
		save_pkl.save_pkl({'type':'Rabi amplitude','name': 'qubit{}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name, location=root_dir)
		qjson.dump(type='two-level-rabi-rect-amplitude',name = self.build_calibration_filename(), params=self.Rabi_rect_amplitude_result)	
		#self.rabi_rect_ex_amplitude = self.ex_amplitude
		#del measurement, measurement_fitted, set_ex_length, set_seq
		return measurement#self.Rabi_rect_amplitude_result
		
	def Ramsey(self,delays,target_freq_offset,*params,pre_pulse_seq=[], post_pulse_seq=[], cross_ex_device=None, dump=True):
		#if self.rabi_rect_ex_amplitude != self.ex_amplitude:
		self.load_calibration()
		pg = self.pulse_sequencer
		sequence = []
		if not cross_ex_device:
			cross_ex_device = self
			cross_ex_device_name_addition = ''
		else: 
			cross_ex_device_annotation = True
			cross_ex_device.load_calibration()
			cross_ex_device_name_addition = ' cross-ex ch '+(','.join(cross_ex_device.ex_channels))
		def set_delay(delay): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			cross_channel_pulses = [(c, pg.rect, a) for c, a in zip(cross_ex_device.ex_channels, cross_ex_device.ex_amplitudes)]
			cross_channel_pulses_vz = [(c, pulses.vz, delay*target_freq_offset*2*np.pi) for c, a in zip(cross_ex_device.ex_channels, cross_ex_device.ex_amplitudes)]
			
			sequence = [pg.p(None, readout_begin - pi2_pulse)]\
						+pre_pulse_seq+\
					   [pg.pmulti(pi2_pulse,*tuple(channel_pulses)),
					    pg.p(None, delay),
						pg.pmulti(0, *tuple(cross_channel_pulses_vz)),
						pg.pmulti(0.25/cross_ex_device.Rabi_rect_result['rabi_rect_freq'], *tuple(cross_channel_pulses))]+post_pulse_seq+self.ro_sequence

			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
	
		measurement_name = 'Ramsey (target offset {0:4.2f} MHz), ex ch {1}'.format(target_freq_offset/1e6, ','.join(self.ex_channels))+cross_ex_device_name_addition+self.get_measurement_name_comment()
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
			self.Ramsey_result['ramsey_freq']=fitted_parameters['freq']
			self.Ramsey_result['ramsey_decay']=fitted_parameters['decay']
			self.Ramsey_result['ramsey_phase']=fitted_parameters['phase']
			self.Ramsey_result['ramsey_amplitudes']=fitted_parameters['amplitudes']
			self.Ramsey_result['ramsey_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
			self.Ramsey_result['ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.Ramsey_result['ramsey_ex_amplitude'] = self.ex_amplitudes
			self.Ramsey_result['qubit_id'] = self.qubit_id
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																				 fitted_parameters['freq'], 
																				 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Ramsey', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name,location=root_dir)
			if dump:
				qjson.dump(type='two-level-ramsey',name=self.build_calibration_filename(), params=self.Ramsey_result)
			del measurement, measurement_fitted, set_delay, set_seq
			return self.Ramsey_result
		except Exception as e:
			print (e)
			return measurement
		
	def decay(self, delays, *params):
		pg = self.pulse_sequencer
		#if self.rabi_rect_ex_amplitude != self.ex_amplitude:
		self.load_calibration()
		sequence = []
		def set_delay(delay): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = [pg.p(None, readout_begin - pi_pulse-delay),
						pg.pmulti(pi_pulse, *tuple(channel_pulses)), 
						pg.p(None, delay)]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
				
		measurement_name = 'Decay, ex channels {0}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
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
			self.decay_result['decay_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
			self.decay_result['decay_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.decay_result['decay_ex_amplitude'] = self.ex_amplitudes
			self.decay_result['qubit_id'] = self.qubit_id
			annotation = 'Decay: {0:4.6g} s'.format(fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Decay', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name,location=root_dir)
		
			qjson.dump(type='two-level-decay',name=self.build_calibration_filename(), params=self.decay_result)
			del measurement, measurement_fitted, set_delay, set_seq
			return self.decay_result
		except:
			return measurement
	
	def Ramsey_probe(self,pre_pulse_seq,interleaved_pulse_seq,phase_offsets=[0],max_rabi_freq=50e6,*params):
		'''
		Suppose we want to do Z-pulse calibration (this is what this function was designed for).
		Z-pulse calibration is done as follows:
	
		(optional pi-pulse) (pi/2 along the qubit under investigation) (z-pulse) (pi/2 along along the qubit undeer investigation, perhaps phase offset)
		there three several types of sweep parameters: 
			1) z-pulse parameters (amplitude, length, etc.)
			2) pi/2 pulse phase offset
			3) external parameters (something completely unrelated to the pulse sequence)

		(1) is set by *params varargs, but the setter in the varags is not directly responsible for the setting,
			rather it is interleaved_pulse_seq, which, as it's arguments, recieves the n values of *params
		(2) is controlled via phase_offset argument
		(3) is set by *params varargs -- with the setter invoked the usual way.
		'''
		
		self.load_calibration()
		pg = self.pulse_sequencer
		sequence = []
		
		params = [(phase_offsets, lambda: None, 'Phase offset', '')]+[i for i in params] # prepend phase_offset to sweep params with zero setter
		#param_values[0] is the phase
		
		current_param_values = [None]*len(params) # pulse parameters for prepare and interleaved pulse
		# traverse scan parameters and decorate the param setter with update of current_param_values.
		params_with_decorated_setters = []
		for param_id, param in enumerate(params):
			param_with_decorated_setter = [i for i in param] # copy the param
			def decorated_param_setter(param_val, param_id=param_id):
				#print (param_id, param)
				param[1](param_val) # set param with setter passed to Ramsey_probe (propagate to sweep.sweep)
				current_param_values[param_id] = param_val # save value (so that we can pass it to pre_pulse_seq and interleave_pulse_seq)
				if not None in current_param_values:
					set_pulse_sequence(*tuple(current_param_values))
			param_with_decorated_setter[1] = decorated_param_setter
			params_with_decorated_setters.append(tuple(param_with_decorated_setter))
			
		params_with_decorated_setters = tuple(params_with_decorated_setters)
		
		def set_pulse_sequence(*param_values): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a*amplitude) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			phase_offset = param_values[0]
			channel_pulses_phase_offset = [(c, pg.rect, -a*amplitude*np.exp(1j*phase_offset)) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			# check if pre_pulse is callable
			if callable(pre_pulse_seq):
				pre_pulse_seq_inst = pre_pulse_seq(*tuple(param_values[1:])) # pass all current_param_values except for pi/2 phase offset
			else:
				pre_pulse_seq_inst = pre_pulse_seq
				
			if callable(interleaved_pulse_seq):
				interleaved_pulse_seq_inst = interleaved_pulse_seq(*tuple(param_values[1:]))
			else:
				interleaved_pulse_seq_inst = interleaved_pulse_seq
			
			sequence = pre_pulse_seq_inst+\
					[pg.pmulti(pi2_pulse, *tuple(channel_pulses))]+\
					interleaved_pulse_seq_inst+\
					[pg.pmulti(pi2_pulse, *tuple(channel_pulses_phase_offset))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
				
		def set_seq():
			#print(sequence)
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		
		measurement_name = 'Ramsey probe, probe channels {}'.format(','.join(self.ex_channels))+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		if self.Rabi_rect_result['rabi_rect_freq'] < max_rabi_freq:
			rabi_freq = self.Rabi_rect_result['rabi_rect_freq']
			amplitude = 1.0
		else:
			rabi_freq = max_rabi_freq
			amplitude = max_rabi_freq/self.Rabi_rect_result['rabi_rect_freq']
		pi2_pulse = 0.25/rabi_freq
		#readout_begin = np.max(delays)+pi2_pulse*2
		#print (params_with_decorated_setters)
		measurement = sweep.sweep(self.readout_device, *params_with_decorated_setters, filename=measurement_name, shuffle=self.shuffle, root_dir=root_dir,
								  plot_separate_thread= self.plot_separate_thread,
								  plot=self.plot)
		try:
			measurement_fitted, fitted_parameters = self.fitter(measurement, fitting.exp_sin_fit)
			Ramsey_probe_result = {}
			Ramsey_probe_result['ramsey_freq']=fitted_parameters['freq']
			Ramsey_probe_result['ramsey_decay']=fitted_parameters['decay']
			Ramsey_probe_result['ramsey_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
			Ramsey_probe_result['ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			Ramsey_probe_result['ramsey_ex_amplitude'] = self.ex_amplitudes
			Ramsey_probe_result['qubit_id'] = self.qubit_id
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																					 fitted_parameters['freq'], 
																					 fitted_parameters['decay'])
			save_pkl.save_pkl({'type':'Ramsey probe', 'name': 'qubit {}'.format(self.qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name,location=root_dir)

			#qjson.dump(type='two-level-spin-echo',name=self.build_calibration_filename(), params=self.spin_echo_result)
			#del measurement, measurement_fitted, set_delay, set_seq
			return Ramsey_probe_result#self.spin_echo_result
		except:
			return measurement
	
	def spin_echo(self,delays,target_freq_offset,*params):
		#if self.rabi_rect_ex_amplitude != self.ex_amplitude:
		self.load_calibration()
		pg = self.pulse_sequencer
		sequence = []
		def set_delay(delay): 
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			channel_pulses_phase_offset = [(c, pg.rect, a*np.exp(1j*delay*target_freq_offset*2*np.pi)) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = [pg.p(None, readout_begin-pi2_pulse),
					pg.pmulti(pi2_pulse, *tuple(channel_pulses)), 
					pg.p(None, delay), 
					pg.pmulti(pi2_pulse*2, *tuple(channel_pulses)), 
					pg.p(None, delay), 
					pg.pmulti(pi2_pulse, *tuple(channel_pulses_phase_offset))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		
		measurement_name = 'Spin echo (target offset {0:4.2f} MHz), ex channels {1}'.format(target_freq_offset/1e6, ','.join(self.ex_channels))+self.get_measurement_name_comment()
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
			self.spin_echo_result['ramsey_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
			self.spin_echo_result['ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
			self.spin_echo_result['ramsey_ex_amplitude'] = self.ex_amplitudes
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