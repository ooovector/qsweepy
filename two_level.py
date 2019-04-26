from . import sweep
from . import save_pkl
from . import fitting
from . import qjson
import numpy as np
import warnings
from .data_structures import *
from .database import*
from .save_exdir import*
from . import sweep_extras

# Maybe we should call this "two-level dynamics"?
class quantum_two_level_dynamics:
	def __init__(self, pulse_sequencer, readout_device, ex_channel, ro_channel, ro_sequence, ex_amplitude, readout_measurement_name, qubit_id, db, sample_name = None, comment = '', shuffle=False, **kwargs):
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
		self.shuffle = shuffle
		self.db = db
		self.sweeper=sweep_extras.sweeper(db)#sweeper
		
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
			self.rabi_rect = qjson.load('two-level-rabi-rect', self.build_calibration_filename())
		except Exception as e:
			print('Failed loading rabi frequency calibration: '+str(e))
		
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
							  metadata={'ex_amplitudes': str(self.ex_amplitudes), 
										'qubit_id': str(self.qubit_id)})
		fitted_data, fitted_parameters = self.fitter(lengths, state.data, fitting.exp_sin_fit)
		print('itted params: ', fitted_parameters)
		state_fitted = state.copy()
		state_fitted.references = {state.id: 'fit'}
		state_fitted.metadata.update({'Frequency': str(fitted_parameters['freq']), 'Decay': str(fitted_parameters['decay']), 'Phase': str(fitted_parameters['phase']), 
										'Amplitude': str(fitted_parameters['amplitudes']), 'Excitation channel': str(ex_channel), 'Readout channel': str(ro_channel)})
		for dataset in state_fitted.datasets.keys():
			state_fitted.datasets[dataset].data = fitted_data
		save_exdir(state_fitted)
		db.create_in_database(state_fitted)		
		print(state.id, state_fitted.id)
		
		
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
		del measurement, measurement_fitted, set_ex_length, set_seq
		return self.Rabi_rect_result