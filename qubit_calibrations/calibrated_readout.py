from ..data_structures import *
from .. import data_reduce
from .. import single_shot_readout
from . import excitation_pulse
from .readout_pulse import *
from .. import readout_classifier

import traceback

def get_calibrated_measurer(device, qubit_ids):
	qubit_readout_pulse = get_multi_qubit_readout_pulse(device, qubit_ids)
	features = []
	thresholds = []

	references = {'readout_pulse': qubit_readout_pulse.id}
	for qubit_id in qubit_ids:
		metadata = {'qubit_id':qubit_id}
		try:
			measurement = device.exdir_db.select_measurement(measurement_type='readout_calibration', metadata=metadata, references_that=references)
		except Exception as e:
			print (traceback.print_exc())
			measurement = calibrate_readout(device, qubit_id, qubit_readout_pulse)

		features.append(np.conj(measurement.datasets['feature'].data))
		#features.append(measurement.datasets['avg_sample1'].data - measurement.datasets['avg_sample0'].data)
		#thresholds.append(np.sum(np.conj(features[-1])*(measurement.datasets['avg_sample1'].data + measurement.datasets['avg_sample0'].data)/2.)) ###TODO: ugly way of getting a threshold

		# this works for 1d
		# which is OK I think
		cdfs = np.cumsum(measurement.datasets['hists'].data, axis=1)
		threshold = np.max(cdfs)-cdfs[1,:]-cdfs[0,:]
		threshold_bin = np.argmax(threshold<0)


	#print (features, thresholds)
	readout_device = device.set_adc_features_and_thresholds(features, thresholds, disable_rest=True)
	return qubit_readout_pulse, readout_device, features, thresholds


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
									_readout_classifier = readout_classifier.binary_linear_classifier(),
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

def get_qubit_readout_pulse_from_fidelity_scan(device, fidelity_scan):
	references = {'fidelity_scan':fidelity_scan.id}
	if 'channel_calibration' in fidelity_scan.metadata:
		references['channel_calibration'] = fidelity_scan.references['readout_channel_calibration']

	fidelity_dataset = fidelity_scan.datasets['fidelity']
	max_fidelity = np.unravel_index(np.argmax(fidelity_dataset.data.ravel()), fidelity_dataset.data.shape)
	pulse_parameters = {}
	for p, v_id in zip(fidelity_dataset.parameters, max_fidelity):
		pulse_parameters[p.name] = p.values[v_id]
#	compression_1db = float(passthrough_measurement.metadata['compression_1db'])
#	additional_noise_appears = float(passthrough_measurement.metadata['additional_noise_appears'])
#	if np.isfinite(compression_1db):
#		calibration_type = 'compression_1db'
#		amplitude = compression_1db
#	elif np.isfinite(additional_noise_appears):
#		calibration_type = 'additional_noise_appears'
#		amplitude = additional_noise_appears
#	else:
#		raise Exception('Compession_1db and additional_noise_appears not found on passthourgh scan!')
	readout_channel = fidelity_scan.metadata['channel']
	#length = float(fidelity_scan.metadata['length'])
	metadata={'pulse_type': 'rect',
			  'channel': readout_channel,
			  'qubit_id': fidelity_scan.metadata['qubit_id'],
	          #'amplitude':amplitude,
			  'calibration_type': 'fidelity_scan',
			  #'length': passthrough_measurement.metadata['length']
			  }
	metadata.update(pulse_parameters)
	length = float(metadata['length'])
	amplitude = float(metadata['amplitude'])
	try:
		readout_pulse = qubit_readout_pulse(device.exdir_db.select_measurement(measurement_type='qubit_readout_pulse', references_that=references, metadata=metadata))
	except Exception as e:
		print (type(e), str(e))
		readout_pulse = qubit_readout_pulse(references=references, metadata=metadata, sample_name=device.exdir_db.sample_name)
		device.exdir_db.save_measurement(readout_pulse)
	readout_pulse.pulse_sequence = [device.pg.p(readout_channel, length, device.pg.rect, amplitude)]
	return readout_pulse

def readout_fidelity_scan(device, qubit_id, readout_pulse_lengths, readout_pulse_amplitudes, recalibrate_excitation=True):
	adc, mnames =  device.setup_adc_reducer_iq(qubit_id, raw=True)
	nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
	old_nums = adc.get_nums()

	readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]

	other_qubit_pulse_sequence = []
	references = {}
	if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
		references['channel_calibration'] = device.awg_channels[readout_channel].get_calibration_measurement()

	for other_qubit_id in device.get_qubit_list():
		if other_qubit_id != qubit_id:
			half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id, rotation_angle=np.pi/2., recalibrate=recalibrate_excitation)
			references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
			other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0))

	qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi, recalibrate=recalibrate_excitation)
	metadata = {'qubit_id':qubit_id,
				'averages':nums,
				'channel': readout_channel}

	#print ('len(readout_pulse_lengths): ', len(readout_pulse_lengths))
	if len(readout_pulse_lengths)==1:
		metadata['length'] = str(readout_pulse_lengths[0])

	references.update({	'excitation_pulse':qubit_excitation_pulse.id,
						'delay_calibration':device.modem.delay_measurement.id})

	classifier = single_shot_readout.single_shot_readout(adc=adc,
									prepare_seqs = [other_qubit_pulse_sequence, other_qubit_pulse_sequence+qubit_excitation_pulse.get_pulse_sequence(0)],
									ro_seq = device.trigger_readout_seq,
									pulse_generator = device.pg,
									ro_delay_seq = None,
									adc_measurement_name = 'Voltage')

	classifier.readout_classifier.cov_mode = 'equal'

	# setters for sweep
	readout_amplitude = 0
	readout_length = 0
	def set_readout_amplitude(x):
		nonlocal readout_amplitude
		readout_amplitude = x
		classifier.ro_seq = device.trigger_readout_seq+[device.pg.p(readout_channel, readout_length, device.pg.rect, readout_amplitude)]
	def set_readout_length(x):
		nonlocal readout_length
		readout_length = x
		classifier.ro_seq = device.trigger_readout_seq+[device.pg.p(readout_channel, readout_length, device.pg.rect, readout_amplitude)]

	try:
		adc.set_nums(nums)
		measurement = device.sweeper.sweep(classifier,
								(readout_pulse_lengths, set_readout_length, 'length', 's'),
								(readout_pulse_amplitudes, set_readout_amplitude, 'amplitude', ''),
								measurement_type='readout_fidelity_scan',
								metadata=metadata,
								references=references)
	except:
		raise
	finally:
		adc.set_nums(old_nums)

	return measurement
	#classifier.repeat_samples = 2
