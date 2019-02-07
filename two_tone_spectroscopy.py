def two_tone_normalize(measurement, dataset_name='S-parameter', excitation_frequency_axis_name='Excitation frequency'):
	import copy.deepcopy
	'''
		Normalizes raw measurement of a two-tone spectrum.
		Performs following manipulations:
		normalized measurement = measurement[dataset_name]-<measurement[dataset_name]>_{excitation_frequency_axis} (median over real and imag parts).
	'''
	dataset = measurement[measurement_name]
	if excitation_freqeuency_axis_name not in dataset[0]:
		excitation_frequency_axis_id = dataset[0].index(excitation_freqeuency_axis_name)
	
	real_part = np.real(dataset[2])
	imag_part = np.imag(dataset[2])
	
	real_part_median = np.nanmedian(real_part, axis=excitation_frequency_axis_id)
	imag_part_median = np.nanmedian(imag_part, axis=excitation_frequency_axis_id)

	average_shape = [d for d in dataset[2].shape]
	average_shape[excitation_frequency_axis_id] = 1
	
	real_part_median = np.reshape(real_part_median, average_shape)
	imag_part_median = np.reshape(imag_part_median, average_shape)
	
	normalized_data = (real_part-real_part_median)+1j*(imag_part-imag_part_median)
	
	normalized_measurement = {dataset_name+' normalized': deepcopy(dataset)}
	normalized_measurement[dataset_name+' normalized'] = normalized_data
	
	return normalized_measurement
	