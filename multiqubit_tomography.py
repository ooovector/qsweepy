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
		self.reconstruction_type='cvxopt'

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

	def reconstruct(self, measurement_results):
		from cvxpy import Variable, atoms, abs, reshape, Minimize, Problem, CVXOPT
		from traceback import print_exc
		reconstruction_operator_names = []
		reconstruction_operators = []
		basis_axes_names = self.reconstruction_basis.keys()
		basis_vector_norms = np.asarray(
			[np.linalg.norm(self.reconstruction_basis[r]['operator']) for r in basis_axes_names])

		for reconstruction_operator_name, reconstruction_operator in self.reconstruction_basis.items():
			reconstruction_operator_names.append(reconstruction_operator_name)
			reconstruction_operators.append(reconstruction_operator['operator'])

		reconstruction_matrix = []
		for rot, projection in self.proj_seq.items():
			for measurement_name, projection_operator in projection['operators'].items():
				reconstruction_matrix.append([np.sum(projection_operator * np.conj(reconstruction_operator)) / np.sum(
					np.abs(reconstruction_operator) ** 2) for reconstruction_operator in reconstruction_operators])

		reconstruction_matrix_pinv = np.linalg.pinv(reconstruction_matrix)
		reconstruction_matrix = np.asarray(reconstruction_matrix)
		self.reconstruction_matrix = reconstruction_matrix
		self.reconstruction_matrix_pinv = reconstruction_matrix_pinv

		projections = np.dot(reconstruction_matrix_pinv, measurement_results)
		reconstruction = {str(k):v for k,v in zip(basis_axes_names, projections)}

		if self.reconstruction_type == 'cvxopt':
			#x = cvxpy.Variable(len(projections), complex=True)
			x = Variable(len(projections), complex=True)
			rmat_normalized = np.asarray(reconstruction_matrix/np.mean(np.abs(measurement_results)), dtype=complex)
			meas_normalized = np.asarray(measurement_results).ravel()/np.mean(np.abs(measurement_results))
			#lstsq_objective = cvxpy.atoms.sum_squares(cvxpy.abs(rmat_normalized @ x - meas_normalized))
			lstsq_objective = atoms.sum_squares(abs(rmat_normalized @ x - meas_normalized))
			matrix_size = int(np.round(np.sqrt(len(projections))))
			#x_reshaped = cvxpy.reshape(x, (matrix_size, matrix_size))
			x_reshaped = reshape(x, (matrix_size, matrix_size))
			psd_constraint = x_reshaped >> 0
			hermitian_constraint = x_reshaped.H == x_reshaped
			# Create two constraints.
			constraints = [psd_constraint, hermitian_constraint]
			# Form objective.
			#obj = cvxpy.Minimize(lstsq_objective)
			obj = Minimize(lstsq_objective)
			# Form and solve problem.
			prob = cvxpy.Problem(obj, constraints)
			try:
				prob.solve(solver=cvxpy.CVXOPT, verbose=True)
				reconstruction = {str(k): v for k, v in zip(basis_axes_names, np.asarray(x.value))}
			except ValueError as e:
				print_exc()


		if self.reconstruction_output_mode == 'array':
			it = np.nditer([self.reconstruction_output_array, None], flags=['refs_ok'], op_dtypes=(object, complex))
			with it:
				for x, z in it:
					z[...] = reconstruction[str(x)]
				reconstruction = it.operands[1]

		return reconstruction

	def measure(self):
		meas = {}
		measurement_results = []
		

		for rot, projection in self.proj_seq.items():
			if 'pre_pulses' in self.proj_seq[rot]:
				self.pulse_generator.set_seq(self.proj_seq[rot]['pre_pulses']+self.prepare_seq+self.proj_seq[rot]['pulses'])
			else:
				self.pulse_generator.set_seq(self.prepare_seq+self.proj_seq[rot]['pulses'])
			measurement = self.measurer.measure()
			#print (projection.keys())
			measurement_ordered = [measurement[readout_name] for readout_name in self.readout_names]
			#print (rot)
			#print ('uncorreted:', measurement)
			#measurement_corrected = np.linalg.lstsq(self.confusion_matrix.T, measurement_ordered)[0]
			#measurement = {readout_name:measurement for readout_name, measurement in zip(self.readout_names, measurement_corrected)}
			#print ('corrected:', measurement)
			
			for measurement_name, projection_operator in projection['operators'].items():
				measurement_basis_coefficients = []
				projection_operator_name = str(rot)+'-P'+str(measurement_name)
				
				meas[projection_operator_name] = measurement[measurement_name]
				measurement_results.append(measurement[measurement_name])

		self.measurement_results = measurement_results
		reconstruction = self.reconstruct(measurement_results)

		#reconstruction_matrix_pinv = np.linalg.pinv(reconstruction_matrix)
		#self.reconstruction_matrix = reconstruction_matrix
		#self.reconstruction_matrix_pinv = reconstruction_matrix_pinv

		#print ('reconstruction_matrix (real): ', np.real(reconstruction_matrix).astype(int))
		#print('reconstruction_matrix (imag): ', np.imag(reconstruction_matrix).astype(int))
		#print ('measured projections: ', measurement_results)

		#projections = np.dot(reconstruction_matrix_pinv, measurement_results)
		#print('reconstruction_results (real): ', np.real(projections))
		#print('reconstruction_results (imag): ', np.imag(projections))

		if self.output_mode == 'array':
			it = np.nditer([self.output_array, None], flags=['refs_ok'], op_dtypes=(object, float))
			with it:
				for x, z in it:
					print(x)
					z[...] = meas[str(x)]
				meas = {'measurement': it.operands[1]}   # same as z

		if self.reconstruction_output_mode == 'array':
			meas['reconstruction'] = reconstruction
		else:
			meas.update(reconstruction)

		return meas
		
	def get_opts(self):
		if self.output_mode == 'single':
			opts = { p:{} for p in self.proj_seq.keys()}
		elif self.output_mode == 'array':
			opts = {'measurement':{}}
		opts.update({p: {} for p in self.reconstruction_basis.keys()})
		return opts
		