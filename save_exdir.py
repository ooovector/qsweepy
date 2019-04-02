import numpy as np
from time import gmtime, strftime
from . import save_pkl
#from . import plotting
import shutil as sh
import pathlib
import exdir
import datetime
from .data_structures import *
import os.path
from .database import database
from pony.orm import get, select
from .config import get_config
from collections import OrderedDict

#time_folder_name = now.strftime('%H-%M-%S')
#return (data_root, day_folder_name, time_folder_name)

def default_measurement_save_path(state):
	identifiers = OrderedDict()
	identifiers.update({'id':'{:06d}'.format(state.id)})
	if state.measurement_type:
		identifiers.update({'measurement_type':state.measurement_type})
	if state.sample_name:
		identifiers.update({'sample_name':state.sample_name})
	if state.comment:
		identifiers.update({'comment':state.comment})

	config = get_config()
	data_root = config['datadir']
	now = datetime.now()
	day_folder_name = now.strftime('%Y-%m-%d')

	parent = os.path.join(data_root, day_folder_name)
	#print (parent, identifiers)
	fullpath = os.path.join(parent, '-'.join(identifiers.values()))
	
	return fullpath

	
def save_exdir(state, keep_open=False):
	#parameters = []
	if not state.filename:
		state.filename = default_measurement_save_path(state)
		
	pathlib.Path(state.filename).mkdir(parents=True, exist_ok=True) 
	f = exdir.File(state.filename, 'w', allow_remove=True) 
	f.attrs.update(state.metadata)
	if keep_open:
		if hasattr(state, 'exdir'):
			close_exdir(state)
		state.exdir = f
	try:
		for dataset in state.datasets.keys():
			dataset_exdir = f.create_group(str(dataset))
			parameters_exdir = dataset_exdir.create_group('parameters')
			for index in range(len(state.datasets[dataset].parameters)):
				parameter_values = state.datasets[dataset].parameters[index].values
				parameter_name = state.datasets[dataset].parameters[index].name
				parameter_unit = state.datasets[dataset].parameters[index].unit
				has_setter = True if state.datasets[dataset].parameters[index].setter else False
				d = parameters_exdir.create_dataset(str(index), dtype = np.asarray(parameter_values).dtype, shape = np.asarray(parameter_values).shape)
				d.attrs = {'name':parameter_name, 'unit':parameter_unit, 'has_setter':has_setter}
				d.data[:] = np.asarray(parameter_values)
			data_exdir = dataset_exdir.create_dataset('data', dtype = state.datasets[dataset].data.dtype, data =  state.datasets[dataset].data)
			if keep_open:
				state.datasets[dataset].data_exdir = data_exdir
	except:
		raise
	finally:
		if not keep_open:
			f.close()
	
def update_exdir(state, indeces):
	for dataset in state.datasets.keys():
		state.exdir.attrs.update(state.metadata)
		state.datasets[dataset].data_exdir[indeces] = state.datasets[dataset].data[indeces]
	
def close_exdir(state):
	if hasattr(state, 'exdir'):
		for dataset in state.datasets.keys():
			try:
				del state.datasets[dataset].data_exdir
			except AttributeError:
				continue
			try:
				del state.references['current ref']
			except KeyError:
				continue
		state.exdir.close()
		del state.exdir
	
def load_exdir(filename, db=None):
	data = {}
	f = exdir.File(filename, 'r')
	parameter_values = []
	try:
		state = measurement_state()
		state.metadata.update(f.attrs)
		for dataset_name in f.keys():
			parameters = [None for key in f[dataset_name]['parameters'].keys()]
			for parameter_id, parameter in f[dataset_name]['parameters'].items():
				#print (parameter.attrs)
				parameter_name = parameter.attrs['name']
				parameter_setter = parameter.attrs['has_setter']
				parameter_unit = parameter.attrs['unit']
				parameter_values = parameter.data[:]
				parameters[int(parameter_id)] = measurement_parameter(parameter_values, parameter_setter, parameter_name, parameter_unit)
			data = f[dataset_name]['data'].data[:].copy()
			state.datasets[dataset_name] = measurement_dataset(parameters, data)
		if db:
			id = get(i.id for i in db.Data if (i.filename == filename))
			#print (filename)
			state.id = id
			state.start = db.Data[id].start
			state.stop = db.Data[id].stop
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
		
