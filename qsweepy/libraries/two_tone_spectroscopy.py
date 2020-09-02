def two_tone_normalize(measurement, dataset_name='S-parameter', excitation_frequency_axis_name='Excitation frequency'):
	import copy
	import numpy as np
	'''
		Normalizes raw measurement of a two-tone spectrum.
		Performs following manipulations:
		normalized measurement = measurement[dataset_name]-<measurement[dataset_name]>_{excitation_frequency_axis} (median over real and imag parts).
	'''
	dataset = measurement[dataset_name]
	if excitation_frequency_axis_name in dataset[0]:
		excitation_frequency_axis_id = dataset[0].index(excitation_frequency_axis_name)
	else:
		raise(ValueError('Measurement doesn\'t have axis '+str(excitation_frequency_axis_name)))
	
	real_part = np.real(dataset[2])
	imag_part = np.imag(dataset[2])
	
	real_part_median = np.nanmedian(real_part, axis=excitation_frequency_axis_id)
	imag_part_median = np.nanmedian(imag_part, axis=excitation_frequency_axis_id)

	average_shape = [d for d in dataset[2].shape]
	average_shape[excitation_frequency_axis_id] = 1
	
	real_part_median = np.reshape(real_part_median, average_shape)
	imag_part_median = np.reshape(imag_part_median, average_shape)
	
	normalized_data = (real_part-real_part_median)+1j*(imag_part-imag_part_median)
	
	normalized_measurement = {dataset_name+' normalized': [copy.deepcopy(d) for d in dataset]}
	normalized_measurement[dataset_name+' normalized'][2] = normalized_data
	normalized_measurement[dataset_name+' normalized'] = tuple(normalized_measurement[dataset_name+' normalized'])
	
	
	return normalized_measurement
	
	
#def xcorr_centre_period(measurement, dataset_name='S-parameter', symmetric_periodic_parameter_name):
#	'''
#	Determines period and center of an n-d image along the axis identified 
#	by symmetric_periodic_parameter_name by cross-correlation.
#	'''
#	pass