import numpy as np

def measurer_point_parameters(measurer):
#	return {dataset: [measurement_parameter(dimension[1], None, dimension[0], dimension[2]) for dimension in point_parameters] for dataset, point_parameters in measurer.get_points().items}
	dataset_names = measurer.get_points().keys()
	point_parameters = {}
	for dataset_name in dataset_names:
		points = measurer.get_points()[dataset_name]
		point_parameters[dataset_name] = []
		for dimension in points:
			name, values, unit = dimension
			point_parameters[dataset_name].append(measurement_parameter(values, None, name, unit))
	return point_parameters
		
class measurement_parameter:
	'''
	Sweep parameter data structure.
	Data structure has a function (setter), which makes it
	impractical for serialization.
	'''
	def __init__(self, *param, **kwargs):
		self.values = param[0] if len(param)>0 else kwargs['values']
		self.setter = param[1] if len(param)>1 else kwargs['setter']
		self.name = param[2] if len(param)>2 else 'param_{0}'.format(param_id)
		self.unit = param[3] if len(param)>3 else ''
		self.pre_setter = param[4] if len(param)>4 else None
		self.setter_time = 0
		
		if 'name' in kwargs: self.name = kwargs['name']
		if 'unit' in kwargs: self.unit = kwargs['unit'] 
		if 'pre_setter' in kwargs: self.pre_setter = kwargs['pre_setter'] 
		
	def __str__(self):
		return '{name} ({units}): [{min}, {max}] ({num_points} points) {setter_str}'.format(
			name=self.name, 
			units=self.unit, 
			min = np.min(self.values), 
			max = np.max(self.values), 
			num_points = len(self.values),
			setter_str = 'with setter' if self.setter else 'without setter')
	def __repr__(self):
		return str(self)
		
class measurement_state():
	def __init__(self, sweep_parameters):
		self.datasets = {} ## here you have datasets
		self.parameter_values = [None for sweep_parameter in sweep_parameters] 
		self.start = time.time()
		self.measurement_time = 0
		self.started_sweeps = 0
		self.done_sweeps = 0
		self.filename = ''
		self.id = 0
		self.references = {}
		self.measurement_type = ''
		self.type_revision = 0
		### TODO: invalidation synchronization with db!!!
		self.metadata = {}
		self.total_sweeps = 0
		self.request_stop_acq = False
		self.sweep_error = None
		
	def __str__(self):
		#format = '''Sweep parameter names: {names}, Measurement: {measurement}, Measurement time: {measurement_time}, Done sweeps: {done_sweeps}, Sweep error: {sweep_error}'''
		format =  '''start: {start}, started/done/total sweeps: {started}/{done}/{total}, 
Measured data: \n{datasets}'''
		datasets_str = '\n'.join(['\'{}\': {}'.format(dataset_name, dataset.__str__()) for dataset_name, dataset in self.data.items()])
		return format.format(start=self.start, started=self.started_sweeps, done=self.done_sweeps, total=self.total_sweeps, datasets=datasets_str)
	def __repr__(self):
		return str(self)
	
class measurement_dataset:
	def __init__(self, parameters, data):
		self.parameters = parameters
		self.nonunity_parameters = [parameter for parameter in self.parameters if len(parameter.values)>1] ###TODO: rename to parameters_squeezed
		self.indices_updated = []
		self.data = data
		self.data_squeezed = np.squeeze(self.data)
	def __getattr__(attr_name):
		if attr_name != 'data':
			return self.parameters[attr_name]
		else:
			return self.data
	def __str__(self):
		format =  '''parameters: {}
data: {}'''
		#datasets_str = '\n'.join(['{}: {}'.format(dataset_name, dataset.__str__()) for dataset_name, dataset in self.data])
		return format.format('\n'.join(parameter.__str__() for parameter in self.parameters), self.data)
	def __repr__(self):
		return str(self)