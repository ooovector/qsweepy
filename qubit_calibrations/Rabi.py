from .readout_pulse import get_qubit_readout_pulse, get_uncalibrated_measurer
from ..fitters.exp_sin import exp_sin_fitter
from .channel_amplitudes import channel_amplitudes
import numpy as np
from . import excitation_pulse

def Rabi_rect(device, qubit_id, channel_amplitudes, lengths=None, *extra_sweep_args, tail_length=0, readout_delay=0):
	if type(lengths) is type(None):
		lengths = np.arange(0,
							float(device.get_qubit_constant(qubit_id=qubit_id, name='Rabi_rect_length')),
							float(device.get_qubit_constant(qubit_id=qubit_id, name='Rabi_rect_step')))

	#readout_pulse = get_qubit_readout_pulse(device, qubit_id)
	readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

	def set_ex_length(length):
		ex_pulse_seq = excitation_pulse.get_rect_cos_pulse_sequence(device, channel_amplitudes, tail_length, length, phase=0.)
		#if tail_length > 0:	channel_pulses = [(c, device.pg.rect_cos, a, tail_length) for c, a in channel_amplitudes.items()]
		#else:				channel_pulses = [(c, device.pg.rect,     a             ) for c, a in channel_amplitudes.items()]

		#ex_pulse_seq = [device.pg.pmulti(length+2*tail_length, *tuple(channel_pulses))]
		delay_seq = [device.pg.pmulti(readout_delay)]
		readout_trigger_seq = device.trigger_readout_seq
		readout_pulse_seq = readout_pulse.pulse_sequence

		device.pg.set_seq(ex_pulse_seq+delay_seq+readout_trigger_seq+readout_pulse_seq)

	references = {'channel_amplitudes':channel_amplitudes.id,
				  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}

	if hasattr(measurer, 'references'):
		references.update(measurer.references)

	fitter_arguments = ('iq'+qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

	measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
											  *extra_sweep_args,
											  (lengths, set_ex_length, 'Excitation length','s'),
											  fitter_arguments = fitter_arguments,
											  measurement_type='Rabi_rect',
											  metadata={'qubit_id': qubit_id, 'extra_sweep_args':str(len(extra_sweep_args)), 'tail_length':str(tail_length), 'readout_delay':str(readout_delay)},
											  references=references)

	return measurement

def Rabi_rect_adaptive(device, qubit_id, channel_amplitudes):
	# check if we have fitted Rabi measurements on this qubit-channel combo
	#Rabi_measurements = device.exdir_db.select_measurements_db(measurment_type='Rabi_rect', metadata={'qubit_id':qubit_id}, references={'channel_amplitudes': channel_amplitudes.id})
	#Rabi_fits = [exdir_db.references.this.filename for measurement in Rabi_measurements for references in measurement.reference_two if references.this.measurement_type=='fit_dataset_1d']

	#for fit in Rabi_fits:
	min_step = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_min_step'))
	scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_scan_points'))
	_range = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_range'))
	max_scan_length = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_max_scan_length'))

	good_fit = False
	lengths = np.arange(0, min_step*scan_points, min_step)
	print (0, min_step*scan_points, min_step)
	while not (good_fit or np.max(lengths)>max_scan_length):
		measurement = Rabi_rect(device, qubit_id, channel_amplitudes, lengths=lengths)
		fit_results = measurement.fit.metadata
		if int(fit_results['frequency_goodness_test']):
			return measurement
		lengths *= _range

	raise ValueError('Failed measuring Rabi frequency for qubit {} on channel_amplitudes {}'.format(qubit_id, channel_amplitudes.metadata))

def calibrate_all_single_channel_Rabi(device, _qubit_id=None, remove_bad=False):
	if _qubit_id is None:
		_qubit_id = device.get_qubit_list()
	elif type(_qubit_id) is int:
		_qubit_id = [_qubit_id]

	for qubit_id in _qubit_id:
		amplitude_default = float(device.get_qubit_constant(qubit_id=qubit_id, name='amplitude_default'))
		qubit_channel_calibrated = {}
		for channel_name, device_name in device.get_qubit_excitation_channel_list(qubit_id).items():
			ch = channel_amplitudes(device, **{channel_name:amplitude_default})
			try:
				excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=ch)
				qubit_channel_calibrated[channel_name] = device_name
			except Exception as e:
				print ('Failed to Rabi-calibrate channel ', channel_name)
				print (e)
				if remove_bad:
					print ('Removing from channel list!')
		if remove_bad:
			device.set_qubit_excitation_channel_list(qubit_id, qubit_channel_calibrated)
