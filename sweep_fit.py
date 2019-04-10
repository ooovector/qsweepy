import numpy as np
import itertools
import logging
from . import plotting
from .data_structures import *
from . import save_exdir
from . import fitting_2
from . import fitting
from . import database



def fit_on_start(state_to_fit, db):
	if not fit_condition(state_to_fit):	return
	state = measurement_state()
	state.sample_name = state_to_fit.sample_name
	state.measurement_type = state_to_fit.measurement_type
	#print(state_to_fit.datasets['random'].parameters)
	for dataset_name in state_to_fit.datasets.keys():
		if len(state_to_fit.datasets[dataset_name].data) < 500: 
			data = np.empty((501, 1), dtype = state_to_fit.datasets[dataset_name].data.dtype)
			params = [measurement_parameter(np.linspace(np.min(state_to_fit.datasets[dataset_name].parameters[0].values), np.max(state_to_fit.datasets[dataset_name].parameters[0].values), 501), 
						state_to_fit.datasets[dataset_name].parameters[0].setter, state_to_fit.datasets[dataset_name].parameters[0].name, state_to_fit.datasets[dataset_name].parameters[0].unit)]
		else: 
			data = np.empty((len(state_to_fit.datasets[dataset_name].data), 1), dtype = state_to_fit.datasets[dataset_name].data.dtype)
			params = state_to_fit.datasets[dataset_name].parameters.copy()
		data.fill(np.nan)
		state.datasets[dataset_name] = measurement_dataset(data = data, parameters = params)
	state.references = {state_to_fit.id: 'fit source'}
	db.create_in_database(state)
	state_to_fit.references.update({state.id: 'fit', 'current ref': state})
	save_exdir.save_exdir(state, True)
	db.update_in_database(state)
	#print(state)
	#print(state.references, state.filename, state_to_fit.references, state_to_fit.filename)
	

def sweep_fit(state_to_fit, indeces, db):
	if not fit_condition(state_to_fit):	return
	#print(db.Data[state_to_fit.references['fit']].filename)
	#state = save_exdir.load_exdir(db.Data[state_to_fit.references['fit']].filename)
	state = state_to_fit.references['current ref']
	parameters = {}
	for key in state_to_fit.datasets.keys():
		x = state_to_fit.datasets[key].parameters[0].values if len(state_to_fit.datasets[key].parameters[0].values) > 1 else state_to_fit.datasets[key].parameters[1].values
		y = state_to_fit.datasets[key].data#.reshape(1, len(x)) for fitting library
		#print(x, y)
		if state_to_fit.measurement_type == 'Ramsey':
			(state.datasets[key].parameters[0].values, fitted_y), parameters_for_dataset  = fitting_2.exp_sin_fit(x, y)
		elif state_to_fit.measurement_type == 'Rabi':
			(state.datasets[key].parameters[0].values, fitted_y), parameters_for_dataset  = fitting_2.exp_sin_fit(x, y)
		elif state_to_fit.measurement_type == 'Decay':
			(state.datasets[key].parameters[0].values, fitted_y), parameters_for_dataset = fitting_2.exp_fit(x, np.abs(y))
		else: return
		state.datasets[key].data = fitted_y#np.reshape(fitted_y, (len(state.datasets[key].parameters[0].values), 1)) for fitting library
		parameters.update({key: parameters_for_dataset})
		#print(state.datasets[key].data[:2])
	state.metadata.update(parameters)
	#print(parameters)
	for dataset in state.datasets.keys():
		state.exdir.attrs.update(state.metadata)
		state.datasets[dataset].data_exdir[:] = state.datasets[dataset].data[:]
	#print(state.datasets[dataset].data)

	
def fit_on_finish(state_to_fit, db):
	if not fit_condition(state_to_fit):	return
	state = state_to_fit.references['current ref']
	db.update_in_database(state)
	save_exdir.close_exdir(state)
	
	
def fit_condition(state_to_fit):
	for key in state_to_fit.datasets.keys():
		if len(state_to_fit.datasets[key].parameters[0].values) > 1 and len(state_to_fit.datasets[key].parameters[1].values) > 1:
			print('This is a 2D measurement. I cannot fit it')
			return False
	return True
	