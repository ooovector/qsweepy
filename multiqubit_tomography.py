from . import data_reduce
import numpy as np
from . import readout_classifier

class multiqubit_tomography:
	def __init__(self, measurer, pulse_generator, proj_seq, reconstruction_basis={}):
		#self.sz_measurer = sz_measurer
		#self.adc = adc
		self.pulse_generator = pulse_generator
		self.proj_seq = proj_seq
		self.reconstruction_basis=reconstruction_basis
		
		self.measurer = measurer
		#self.adc_reducer = data_reduce.data_reduce(self.sz_measurer.adc)
		#self.adc_reducer.filters['SZ'] = {k:v for k,v in self.sz_measurer.filter_binary.items()}
		#self.adc_reducer.filters['SZ']['filter'] = lambda x: 1-2*self.sz_measurer.filter_binary_func(x)
		
		## create a list of all readout names
		readout_names = []
		for readout_operators in self.proj_seq.values(): 
			for readout_name in readout_operators['operators'].keys():
				readout_names.append(readout_name)
		readout_names = list(set(readout_names))
		self.readout_names = readout_names
		
		## initialize the confusion matrix to identity so that it doesn't produce any problems
		self.confusion_matrix = np.identity(len(readout_names))

		self.output_array = []
		self.output_mode = 'array'
		self.reconstruction_output_mode = 'array'

	def get_points(self):
		if self.output_mode == 'single':
			points = { p:{} for p in self.proj_seq.keys() }
		elif self.output_mode == 'array':
			points = {'measurement': [('ax{}'.format(ax_id), np.arange(self.output_array.shape[ax_id]), 'projection') \
									   	for ax_id in range(len(self.output_array.shape))]}
		else:
			raise Exception('unknown output_mode')
		if self.reconstruction_output_mode == 'single':
			points.update({p: {} for p in self.reconstruction_basis.keys()})
		elif self.reconstruction_output_mode == 'array':
			points['reconstruction'] = [('ax{}'.format(ax_id), np.arange(self.reconstruction_output_array.shape[ax_id]), 'projection') \
									   	for ax_id in range(len(self.reconstruction_output_array.shape))]
		return points
	
	def get_dtype(self):
		if self.output_mode == 'single':
			dtypes = { p:float for p in self.proj_seq.keys() }
		elif self.output_mode == 'array':
			dtypes = {'measurement':float}
		else:
			raise Exception('unknown output_mode')
		if self.reconstruction_output_mode == 'single':
			dtypes.update({p: complex for p in self.reconstruction_basis.keys()})
		elif self.reconstruction_output_mode == 'array':
			dtypes['reconstruction'] = complex
		return dtypes
	
	def set_prepare_seq(self, seq):
		self.prepare_seq = seq
	
	def measure(self):
		meas = {}
		basis_axes_names = self.reconstruction_basis.keys()
		basis_vector_norms = np.asarray([np.linalg.norm(self.reconstruction_basis[r]['operator']) for r in basis_axes_names])
		
		reconstruction_matrix = []
		measurement_results = []
		
		reconstruction_operator_names = []
		reconstruction_operators = []
		
		for reconstruction_operator_name, reconstruction_operator in self.reconstruction_basis.items():
			reconstruction_operator_names.append(reconstruction_operator_name)
			reconstruction_operators.append(reconstruction_operator['operator'])
			#measurement_results.append(reconstruction_operator)
		
		for rot, projection in self.proj_seq.items():
			if 'pre_pulses' in self.proj_seq[rot]:
				self.pulse_generator.set_seq(self.proj_seq[rot]['pre_pulses']+self.prepare_seq+self.proj_seq[rot]['pulses'])
			else:
				self.pulse_generator.set_seq(self.prepare_seq+self.proj_seq[rot]['pulses'])
			measurement = self.measurer.measure()
			#print (projection.keys())
			measurement_ordered = [measurement[readout_name] for readout_name in self.readout_names]
			#print (rot)
			print ('uncorreted:', measurement)
			measurement_corrected = np.linalg.lstsq(self.confusion_matrix.T, measurement_ordered)[0]
			measurement = {readout_name:measurement for readout_name, measurement in zip(self.readout_names, measurement_corrected)}
			print ('corrected:', measurement)
			
			for measurement_name, projection_operator in projection['operators'].items():
				measurement_basis_coefficients = []
				projection_operator_name = str(rot)+'-P'+str(measurement_name)
				
				meas[projection_operator_name] = measurement[measurement_name]
				measurement_results.append(measurement[measurement_name])
				reconstruction_matrix.append([np.sum(projection_operator*np.conj(reconstruction_operator))/np.sum(np.abs(reconstruction_operator)**2) \
													for reconstruction_operator in reconstruction_operators])
		reconstruction_matrix_pinv = np.linalg.pinv(reconstruction_matrix)
		self.reconstruction_matrix = reconstruction_matrix
		self.reconstruction_matrix_pinv = reconstruction_matrix_pinv

		print ('reconstruction_matrix (real): ', np.real(reconstruction_matrix).astype(int))
		print('reconstruction_matrix (imag): ', np.imag(reconstruction_matrix).astype(int))
		print ('measured projections: ', measurement_results)

		projections = np.dot(reconstruction_matrix_pinv, measurement_results)
		print('reconstruction_results (real): ', np.real(projections))
		print('reconstruction_results (imag): ', np.imag(projections))

		if self.output_mode == 'array':
			it = np.nditer([self.output_array, None], flags=['refs_ok'], op_dtypes=(object, float))
			with it:
				for x, z in it:
					print(x)
					z[...] = meas[str(x)]
				meas = {'measurement': it.operands[1]}   # same as z

		reconstruction = {str(k):v for k,v in zip(basis_axes_names, projections)}
		print ('reconstruction', reconstruction)
		if self.reconstruction_output_mode == 'single':
			if len(reconstruction_operators) > 0:
				meas.update(reconstruction)

		elif self.reconstruction_output_mode == 'array':
			it = np.nditer([self.reconstruction_output_array, None], flags=['refs_ok'], op_dtypes=(object, complex))
			with it:
				for x, z in it:
					z[...] = reconstruction[str(x)]
				meas['reconstruction'] = it.operands[1]

		return meas
		
	def get_opts(self):
		if self.output_mode == 'single':
			opts = { p:{} for p in self.proj_seq.keys()}
		elif self.output_mode == 'array':
			opts = {'measurement':{}}
		opts.update({p: {} for p in self.reconstruction_basis.keys()})
		return opts
		