from . import fitting
from . import qjson
import warnings
from .ponyfiles.save_exdir import*
from . import sweep_extras
from . import pulses
import imp
imp.reload(pulses)

# Maybe we should call this "two-level dynamics"?
class transmon:
	def __init__(self, pulse_sequencer, readout_device, ex_channel, ro_channel, ro_sequence, ex_amplitude, readout_measurement_name, qubit_id, exdir_db, sample_name = 'anonymous_sample', comment = '', shuffle=False, **kwargs):
		self.pulse_sequencer = pulse_sequencer
		self.readout_device = readout_device
		self.readout_measurement_name = readout_measurement_name
		self.ro_channel = ro_channel
		self.two_tone_length = 5e-06
		self.two_tone_ampl = 0
		self.two_tone_freq = 0
		if type(ex_channel) is str:
			self.ex_channels = (ex_channel, )
			self.ex_amplitudes = (ex_amplitude, )
		else:
			self.ex_channels = ex_channel
			self.ex_amplitudes = ex_amplitude
		self.ro_sequence = ro_sequence
		self.shuffle = shuffle
		self.exdir_db = exdir_db
		self.sweeper=sweep_extras.sweeper(self.exdir_db.db)#sweeper

		self.qubit_id = qubit_id
		self.sample_name = sample_name
		self.comment = comment
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
			self.Rabi_rect_result = qjson.load('two-level-rabi-rect', self.build_calibration_filename())
		except Exception as e:
			self.Rabi_rect_result = {}
			print('Failed loading rabi frequency calibration: '+str(e))
		try:
			self.Ramsey_result = qjson.load('two-level-ramsey', self.build_calibration_filename())
		except Exception as e:
			self.Ramsey_result = {}
			print('Failed loading ramsey calibration: '+str(e))


		# Rabi freq depends on excitation pulse parameters
		# If we want to save to config, we should save in
		# ex_pulse_params=>rabi_freq pair
		# ex_pulse_params should be a serialization of a pulse params o_0
		warnings.filterwarnings('ignore')

	def build_calibration_filename(self):
		return '-'.join(['c-{0}-f-{1:.5g}-amp-r{2:.5g}-i{3:.5g}'.format(c, self.pulse_sequencer.channels[c].get_frequency(), np.real(a), np.imag(a)) for c, a in zip(self.ex_channels, self.ex_amplitudes)])+self.comment


	def Rabi(self,lengths, tail_length=0e-9, pre_pulse_seq=[]):
		readout_begin = np.max(lengths)
		pg = self.pulse_sequencer
		sequence = []
		def set_ex_length(length):
			nonlocal sequence
			if tail_length > 0:
				channel_pulses = [(c, pg.rect_cos, a, tail_length) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			else:
				channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence = pre_pulse_seq+[pg.pmulti(length+2*tail_length, *tuple(channel_pulses))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		state = self.sweeper.sweep(self.readout_device,
							  (lengths, set_ex_length, 'Rabi pulse length','s'),
							  measurement_type='Rabi',
							  sample_name=self.sample_name,
							  comment=self.comment,
							  metadata={'ex_amplitudes': str(self.ex_amplitudes), 'qubit_id': str(self.qubit_id)})
		#return
		fitted_data, fitted_parameters = fitting.exp_sin_fit(np.asarray(np.memmap.tolist(state.datasets['avg_cov1'].parameters[0].values)),
															[np.asarray(np.memmap.tolist(state.datasets['avg_cov1'].data))])
		print('fitted params: ', fitted_parameters)
		state.datasets['avg_cov1_fit'] = measurement_dataset(parameters = [measurement_parameter(fitted_data[0], '', 'Rabi pulse length','s')], data = fitted_data[1][0])
		state.metadata.update({'Frequency': str(fitted_parameters['freq']), 'Decay': str(fitted_parameters['decay']), 'Phase': str(fitted_parameters['phase']),
										'Amplitude': str(fitted_parameters['amplitudes']), 'Excitation channels': str(self.ex_channels), 'Readout channel': str(self.ro_channel)})
		#state_fitted.datasets['avg_cov1'].parameters[0].values = fitted_data[0]
		#state_fitted.datasets['avg_cov1'].data = fitted_data[1]
		#for dataset in state_fitted.datasets.keys():
		#	state_fitted.datasets[dataset].data = fitted_data
		save_exdir(state)
		#db.create_in_database(state_fitted)
		print(state.id)


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

		qjson.dump(type='two-level-rabi-rect',name = self.build_calibration_filename(), params=self.Rabi_rect_result)

		#del measurement, measurement_fitted, set_ex_length, set_seq
		return self.Rabi_rect_result

	def Ramsey(self,delays,target_freq_offset,params=None,pre_pulse_seq=[], cross_ex_device=None, dump=True):
		#if self.rabi_rect_ex_amplitude != self.ex_amplitude:
		pg = self.pulse_sequencer
		sequence = []
		if self.Rabi_rect_result != {}:
			pi2_pulse = 0.25/self.Rabi_rect_result['rabi_rect_freq']
		else:
			self.Rabi_rect_result['rabi_rect_freq'] = Rabi_f
			pi2_pulse = 0.25/Rabi_f
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
			cross_channel_pulses = [(c, pg.rect, a*np.exp(1j*delay*target_freq_offset*2*np.pi)) for c, a in zip(cross_ex_device.ex_channels, cross_ex_device.ex_amplitudes)]
			sequence =  [pg.p(None, readout_begin - pi2_pulse)]\
						+pre_pulse_seq+\
						[pg.pmulti(pi2_pulse,*tuple(channel_pulses)),
						pg.p(None, delay),
						pg.pmulti(0.25/cross_ex_device.Rabi_rect_result['rabi_rect_freq'], *tuple(cross_channel_pulses))]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		readout_begin = np.max(delays)+pi2_pulse*2
		state = self.sweeper.sweep(self.readout_device, (delays, set_delay, 'Ramsey delay', 's'),#params,
								measurement_type = 'Ramsey',
								sample_name=self.sample_name,
								comment=self.comment,
								metadata={'ex_amplitudes': str(self.ex_amplitudes), 'qubit_id': str(self.qubit_id)})

		fitted_data, fitted_parameters = fitting.exp_sin_fit(np.asarray(np.memmap.tolist(state.datasets['avg_cov1'].parameters[0].values)),
															[np.asarray(np.memmap.tolist(state.datasets['avg_cov1'].data))])
		print('fitted params: ', fitted_parameters)
		state.datasets['avg_cov1_fit'] = measurement_dataset(parameters = [measurement_parameter(fitted_data[0], '', 'Rabi pulse length','s')], data = fitted_data[1][0])
		state.metadata.update({'Frequency': str(fitted_parameters['freq']), 'Decay': str(fitted_parameters['decay']), 'Phase': str(fitted_parameters['phase']), 'Amplitude': str(fitted_parameters['amplitudes']),
										'Target offset': str(target_freq_offset), 'Excitation channel': str(self.ex_channels), 'Readout channel': str(self.ro_channel)})
		save_exdir(state)
		self.Ramsey_result['ramsey_freq']=fitted_parameters['freq']
		self.Ramsey_result['ramsey_decay']=fitted_parameters['decay']
		self.Ramsey_result['ramsey_phase']=fitted_parameters['phase']
		self.Ramsey_result['ramsey_amplitudes']=fitted_parameters['amplitudes']
		self.Ramsey_result['ramsey_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
		self.Ramsey_result['ramsey_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.Ramsey_result['ramsey_ex_amplitude'] = self.ex_amplitudes
		self.Ramsey_result['qubit_id'] = self.qubit_id

		qjson.dump(type='two-level-ramsey',name=self.build_calibration_filename(), params=self.Ramsey_result)

		return self.Ramsey_result

	def decay(self,delays, Rabi_f = 0, *params,pre_pulse_seq=[], dump=True):
		#if self.rabi_rect_ex_amplitude != self.ex_amplitude:
		pg = self.pulse_sequencer
		sequence = []
		if self.Rabi_rect_result != {}:
			pi_pulse = 0.5/self.Rabi_rect_result['rabi_rect_freq']
		else:
			self.Rabi_rect_result['rabi_rect_freq'] = Rabi_f
			pi_pulse = 0.5/Rabi_f
		def set_delay(delay):
			nonlocal sequence
			channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
			sequence =  [pg.p(None, readout_begin - pi_pulse)]\
						+pre_pulse_seq+\
						[pg.pmulti(pi_pulse,*tuple(channel_pulses)),
						pg.p(None, delay),
						]+self.ro_sequence
			if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
		readout_begin = np.max(delays)+pi_pulse
		state = self.sweeper.sweep(self.readout_device, (delays, set_delay, 'Decay delay', 's'),
								measurement_type = 'Decay',
								sample_name=self.sample_name,
								comment=self.comment,
								metadata={'ex_amplitudes': str(self.ex_amplitudes), 'qubit_id': str(self.qubit_id)})

		fitted_data, fitted_parameters = fitting.exp_fit(np.asarray(np.memmap.tolist(state.datasets['avg_cov3'].parameters[0].values)),
															[np.asarray(np.memmap.tolist(state.datasets['avg_cov3'].data))])
		print('fitted params: ', fitted_parameters)
		state.datasets['avg_cov3_fit'] = measurement_dataset(parameters = [measurement_parameter(fitted_data[0], '', 'Rabi pulse length','s')], data = fitted_data[1][0])
		state.metadata.update({'Decay': str(fitted_parameters['decay']), 'Amplitude': str(fitted_parameters['amplitudes']),
								'Excitation channel': str(self.ex_channels), 'Readout channel': str(self.ro_channel)})
		save_exdir(state)
		self.decay_result['decay_decay']=fitted_parameters['decay']
		self.decay_result['decay_amplitudes']=fitted_parameters['amplitudes']
		self.decay_result['decay_carrier']=[pg.channels[c].get_frequency() for c in self.ex_channels]
		self.decay_result['decay_ro_freq']=pg.channels[self.ro_channel].get_frequency()
		self.decay_result['decay_ex_amplitude'] = self.ex_amplitudes
		self.decay_result['qubit_id'] = self.qubit_id

		qjson.dump(type='two-level-ramsey',name=self.build_calibration_filename(), params=self.decay_result)

		return self.decay_result

	def Rabi_2D(self, frequencies, lengths, ex_ch, tail_length=0e-9, pre_pulse_seq=[]):
		readout_begin = np.max(lengths)
		pg = self.pulse_sequencer
		old_calibration_drift = ex_ch.parent.ignore_calibration_drift
		old_if = ex_ch.get_if()
		ex_ch.parent.ignore_calibration_drift = True
		sequence = []
		try:
			def set_freq(f):
				ex_ch.set_if(f-ex_ch.parent.lo.get_frequency())
			def set_ex_length(length):
				nonlocal sequence
				if tail_length > 0:
					channel_pulses = [(c, pg.rect_cos, a, tail_length) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
				else:
					channel_pulses = [(c, pg.rect, a) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
				sequence = pre_pulse_seq+[pg.pmulti(length+2*tail_length, *tuple(channel_pulses))]+self.ro_sequence
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
					set_seq()
			def set_seq():
				pg.set_seq(sequence)
			if hasattr(self.readout_device, 'diff_setter'): # if this is a sifferential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
				self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout
			state = self.sweeper.sweep(self.readout_device,
								  (frequencies, set_freq, 'Frequency', 'Hz'),
								  (lengths, set_ex_length, 'Rabi pulse length','s'),
								  measurement_type='Rabi_2D',
								  sample_name=self.sample_name,
								  comment=self.comment,
								  metadata={'qubit_id': str(self.qubit_id)})

			ex_ch.parent.ignore_calibration_drift = old_calibration_drift
		except:
			raise
		finally:
			ex_ch.parent.ignore_calibration_drift = old_calibration_drift
			ex_ch.set_if(old_if)

	def two_tone(self, amplitudes, frequencies, ex_ch, tail_length=15e-9, pre_pulse_seq=[]):
		readout_begin = self.two_tone_length + 2*tail_length
		pg = self.pulse_sequencer
		old_calibration_drift = ex_ch.parent.ignore_calibration_drift
		old_if = ex_ch.get_if()
		ex_ch.parent.ignore_calibration_drift = True
		try:
			sequence = []
			def set_pulses():
				nonlocal sequence
				if tail_length > 0:
					channel_pulses = [(c, pg.rect_cos, a*self.two_tone_ampl, tail_length) for c, a in zip(self.ex_channels, self.ex_amplitudes)]
				else:
					channel_pulses = [(c, pg.rect, a*self.two_tone_ampl) for c, a in zip(self.ex_channels, self.ex_amplitudes)]

				sequence = pre_pulse_seq + [pg.pmulti(self.two_tone_length + 2*tail_length, *tuple(channel_pulses))] + self.ro_sequence
				if not hasattr(self.readout_device, 'diff_setter'): # if this is a differential measurer
					set_seq()

			def set_seq():
				pg.set_seq(sequence)

			if hasattr(self.readout_device, 'diff_setter'): # if this is a differential measurer
				self.readout_device.diff_setter = set_seq # set the measurer's diff setter
				self.readout_device.zero_setter = self.set_zero_sequence # for diff_readout

			def set_ex_amplitude(amplitude):
				self.two_tone_ampl = amplitude
				set_pulses()

			def set_ex_freq(frequency):
				ex_ch.set_if(frequency - ex_ch.parent.lo.get_frequency())
				set_pulses()

			self.sweeper.sweep(self.readout_device,
								  (amplitudes, set_ex_amplitude, 'Pump amplitude', ''),
								  (frequencies, set_ex_freq, 'Pump frequency', 'GHz'),
								  measurement_type='two-tone-pulses',
								  sample_name=self.sample_name,
								  comment=self.comment,
								  metadata={'ex_length': str(self.two_tone_length),
											'tail_length': str(tail_length),
											'ex_channels': ','.join(str(self.ex_channels)),
											'qubit_id': str(self.qubit_id)})
		except:
			raise
		finally:
			ex_ch.parent.ignore_calibration_drift = old_calibration_drift
			ex_ch.set_if(old_if)
