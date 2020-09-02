import exdir
from .ponyfiles.data_structures import *
import os.path
from .ponyfiles.database import MyDatabase
from pony.orm import get, select


def save_exdir(state):
	parameters = []
	f = exdir.File(state.filename, 'w', allow_remove=True)
	#print(state.filename)
	try:
		for dataset in state.datasets.keys():
			parameters.append(f.create_group(str(dataset)))
			for index in range(len(state.datasets[dataset].parameters)):
				sweep_value = state.datasets[dataset].parameters[index].values
				sweep_name = state.datasets[dataset].parameters[index].name
				sweep_unit = state.datasets[dataset].parameters[index].unit
				#print(sweep_value)
				d = parameters[len(parameters)-1].create_dataset(sweep_name, dtype = complex, shape = np.shape(sweep_value))
				d.data[:] = sweep_value#str(sweep_name), data=sweep_value)
			parameters[len(parameters)-1].create_dataset('data', dtype = complex, data =  state.datasets[dataset].data)
		attrs = state.metadata.copy()
		attrs.update({'parameter values': state.parameter_values})
		f.attrs.update(attrs)
	except:
		raise
	finally:
		f.close()

def read_exdir(filename):
	result = {}
	f = exdir.File(filename, 'r')
	attributes = {}
	attr = []
	try:
		for param in f.keys():
			for k in f.attrs.items():
				attr.append(k)
			attributes = dict((a,b) for (a,b) in tuple(attr))
			variables = []
			variables_values = []
			for variable in f[str(param)].keys():
				variables.append(variable)
				variables_values.append(f[str(param) + '/' + str(variable)][:].copy())
			result[param] = (tuple(variables), tuple(variables_values))
	except:
		raise
	finally:
		f.close()
	return result, attributes


def read_exdir_new(filename):
	data = {}
	#print(filename)
	f = exdir.File(filename, 'r')
	attributes = {}
	attrs = []
	parameter_values = []
	all_parameters = []
	try:
		for k in f.attrs.items():
			#print(k)
			if k[0] != 'parameter_values': attrs.append(k)
			else: parameter_values = k[1]
		state = MeasurementState(parameter_values)
		for dataset_name in f.keys():
			all_param_help_arr = []
			for parameter in f[dataset_name].keys():
				#print(parameter, [f[str(dataset_name) + '/' + parameter][:], 'setter', parameter])
				if parameter != 'data': all_param_help_arr.append(MeasurementParameter(f[str(dataset_name) + '/' + parameter][:].copy(), 'setter', parameter))
			all_parameters = all_param_help_arr.copy()
			data[dataset_name] = f[str(dataset_name) + '/data'][:].copy()
		state.metadata = attrs
		for dataset_name in data.keys():
			state.datasets[dataset_name] = MeasurementDataset(parameters = all_parameters, data = data[dataset_name])
		db = MyDatabase()
		id = get(i.id for i in db.Data if (i.filename == filename))
		state.id = id
		state.start = db.Data[id].time_start
		state.stop = db.Data[id].time_stop
		query = select(i for i in db.Reference if (i.this.id == id))
		references = {}
		for q in query: references.update({q.that.id: q.ref_type})
		#print(references)
		state.references = references
		state.filename = filename
	except:
		raise
	finally:
		f.close()
	return state


def save_exdir_on_fly(state, measurement, indeces):
	parameters = []
	f = exdir.File(state.filename, 'w', allow_remove=True)
	try:
		if os.path.exists(state.filename):
			for dataset in state.datasets.keys():
				f[str(dataset) + '/data'][tuple(indices+[...])] = measurement[dataset]
		else:
			f = exdir.File(state.filename, 'w', allow_remove=True)
			print('I am in')
			for dataset in state.datasets.keys():
				parameters.append(f.create_group(str(dataset)))
				for index in range(len(state.datasets[dataset].parameters)):
					sweep_value = state.datasets[dataset].parameters[index].values
					sweep_name = state.datasets[dataset].parameters[index].name
					d = parameters[len(parameters)-1].create_dataset(sweep_name, dtype = complex, shape = np.shape(sweep_value))
					d.data[:] = sweep_value
				parameters[len(parameters)-1].create_dataset('data', dtype = complex, data =  state.datasets[dataset].data)
			f.attrs.update(state.metadata)
	except:
		raise
	finally:
		f.close()
