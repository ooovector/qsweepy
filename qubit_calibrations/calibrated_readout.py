from ..data_structures import *
from .. import data_reduce
from .. import single_shot_readout
from . import excitation_pulse

import traceback

def get_calibrated_measurer(device, qubit_id, qubit_readout_pulse):
	readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
	metadata = {'qubit_id':qubit_id}
	references = {'readout_pulse': qubit_readout_pulse.id}
	if excitation_pulse is not None:
		references['excitation_pulse'] = excitation_pulse.id
	try:
		measurement = device.exdir_db.select_measurement(measurement_type='readout_calibration', metadata=metadata, references_that=references)
	except Exception as e:
		print (traceback.print_exc())
		measurement = calibrate_readout(device, qubit_id, qubit_readout_pulse)
		
	feature = measurement.datasets['avg_sample1'] - measurement.datasets['avg_sample0']
	threshold = np.conj(feature)*(measurement.datasets['avg_sample1'] + measurement.datasets['avg_sample0'])/2. ###TODO: ugly way of getting a threshold
	
	

def calibrate_readout(device, qubit_id, qubit_readout_pulse):
	adc, mnames =  device.setup_adc_reducer_iq(qubit_id, raw=True)
	nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
	old_nums = adc.get_nums()
	
	other_qubit_pulse_sequence = []
	references = {}
	for other_qubit_id in device.get_qubit_list():
		if other_qubit_id != qubit_id:
			half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id, rotation_angle=np.pi/2.)
			references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
			other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0))
			
	qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi)
	metadata = {'qubit_id':qubit_id, 
				'averages':nums}
				
	references.update({'readout_pulse':qubit_readout_pulse.id,
				  'excitation_pulse':qubit_excitation_pulse.id,
				  'delay_calibration':device.modem.delay_measurement.id})
	
	classifier = single_shot_readout.single_shot_readout(adc=adc, 
									prepare_seqs = [other_qubit_pulse_sequence, other_qubit_pulse_sequence+qubit_excitation_pulse.get_pulse_sequence(0)],
									ro_seq = device.trigger_readout_seq+qubit_readout_pulse.get_pulse_sequence(), 
									pulse_generator = device.pg,
									ro_delay_seq = None,
									adc_measurement_name = 'Voltage')

	classifier.readout_classifier.cov_mode = 'equal'
	
	try:
		adc.set_nums(nums)
		measurement = device.sweeper.sweep(classifier,
								measurement_type='readout_calibration',
								metadata=metadata,
								references=references)
	except:
		raise
	finally:
		adc.set_nums(old_nums)
		
	return measurement
	#classifier.repeat_samples = 2
