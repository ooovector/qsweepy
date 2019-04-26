import numpy as np
import time
from time import sleep
import sys
from matplotlib import pyplot as plt
import itertools
from . import save_pkl
import logging
from . import plotting
import shutil as sh
import threading
import pathlib
import random
import gc
from .data_structures import *

def optimize(target, *params ,initial_simplex=None ,maxfun=200 ):
	from scipy.optimize import fmin
	x0 = [p[1] for p in params]
	print(x0)
	def tfunc(x):
		for xi, p in zip(x, params):
			if len(p)>2 and p[2]:
				if xi*p[1] < p[2]:
					xi = p[2]/p[1]
			if len(p)>3 and p[3]:
				if xi*p[1] > p[3]:
					xi = p[3]/p[1]
			p[0](xi*p[1]) # set current parameter
		f = target()
		print (x*x0, f)
		return f
	if initial_simplex:
		initial_simplex = np.asarray(initial_simplex)/x0
	solution = fmin(tfunc, np.ones(len(x0)), maxfun=maxfun, initial_simplex=initial_simplex)*x0
	score = tfunc(solution/x0)
	return solution, score
	
'''
sweep_state_server: generates plots, outputs remaining time, acts as server bot.
sweep_state_server is a http server thread
sweep_state is a data structure
sweep_state contains: 
1. current measurement structure,
2. current parameter_values,
3. measurement_start time,

the 'server' thread handles interactive stuff, the 'measurement' thread handles non-interactive stuff.
Interactive stuff:
- (matplotlib) UI &  & telegram bot,
- telegram bot
- plotly UI
- time_left UI

the 'server' thread functions through event hooks (callbacks):
hook types: 
- new data appeared
- measurement finished
- ?

the 'measurement' thread activates hooks and handles non-interactive stuff.
Does it make sense to delegate stuff from the 'measurement' thread to the 'server' thread? 
Since this is python which has GIL, probably no. 
Tasks for the 'measurement' thread:
- set parameter values (maybe multithreaded)
- measure stuff

- call event hooks (should be passed through to event listeners)
- save data?? -- we have save_pkl, save_hdf, ...

if there's an exception in measurement thread, stop on next iteration (and emit 'measurement finished' event) 
what about exceptions in 'server' thread? 

1. if the exception is thrown in a event handler, drop that event handler until the sweep is finished.
2. if the exception is thrown in the core server code, raise an exception in the 'measurement' thread too.

All the UI stuff is not only about 'sweep'-type measurements, but for all 'long' measurements.
A 'long' measurement, is, essenitally, any measurement => a sweep is a subclass of 'long' measurements.
'Long' measurements are characterized by the existence of an iteration loop (and the ability of emiting a 'new data appeared'
event). Examples of non-sweep 'long measurements' include:
- optimization (mixer calibration); itration progess is similar to a sweep (can be repoted on in a similar fashion)
- measuring fixed sets (tomography)??

How about integrating sweeps and tomography? Issues are:
- sweeps are non-recursive. Sweeps of sweeps go against the whole concept of sweeps.
- tomography has inherent "reductions" and is much nicer interfaced by a 'measurement'.
- whether tomography produces an array of a single measurement or a dict of measurements is a tricky question.
- probably, tomography should produce an array (more robust in terms of large tomography datasets)

Tomography should be an example of a 'long' measurement and enjoy the same online 'new data appeared' features as sweeps.
- different datasets from a single measurement should be updated separately.
- 

What about multithreaded data acquisition and parameter setting?
- we are working in python, so all this is bound to suck.
- multiple data points can be in the pipeline
- all multiple data points in the pipeline are visible only to the 'measurement' 
  thread and not to the 'server' thread. Possible exception: 
4. writes to files??
'''

''' measurement_state class
'''

'''
Creates dict of measurement_parameter data structures for each dataset of measurer device.
example: point_parameter(vna) should return
{'S-parameter':measurement_parameter(vna.get_freqpoints(), None, 'Frequency', 'Hz')}	
'''	


		
def sweep(measurer, 
			  *parameters, 
			  shuffle=False, 
			  on_start =[], 
			  on_update=[], 
			  on_finish=[], 
			  **kwargs
			  ):
	'''
	Performs a n-d parametric sweep.
	Usage: sweep(measurer, (param1_values, param1_setter, [param1_name]), (param2_values, param2_setter), ... )
	measurer: an object that supports get_points(), measure(), get_dtype() and get_opts() methods.
	
	Returns: measurement_state struct after measurement dict of ndarrays each corresponding to a measurment in the sweep
	'''
	
	sweep_parameters = [measurement_parameter(*parameter) for parameter in parameters]
	point_parameters = measurer_point_parameters(measurer)
	
	sweep_dimensions = tuple([len(sweep_parameter.values) for sweep_parameter in sweep_parameters])
	state = measurement_state(**kwargs)
	state.sweep_parameters = [None for d in sweep_dimensions]
	state.total_sweeps = np.prod([d for d in sweep_dimensions])
	#all_parameters = {dataset: sweep_parameters+_point_parameters for dataset, _point_parameters in point_parameters.items()}
	
	### initialize data
	for dataset_name, point_parameters in point_parameters.items():
		all_parameters = sweep_parameters + point_parameters
		data_dimensions = tuple([len(parameter.values) for parameter in all_parameters])
		data = np.empty(data_dimensions, dtype = measurer.get_dtype()[dataset_name])
		data.fill(np.nan)
		state.datasets[dataset_name] = measurement_dataset(parameters = all_parameters, data = data)
	
	all_indeces = itertools.product(*([i for i in range(d)] for d in sweep_dimensions))
	if shuffle:
		all_indeces = [i for i in all_indeces]
		random.shuffle(all_indeces)
	if len(sweep_dimensions)==0: # 0-d sweep case: single measurement
		all_indeces = [[]]	
	
	def set_single_measurement_result(single_measurement_result, indeces):
		nonlocal state
		indeces = list(indeces)
		for dataset in state.datasets.keys():
			state.datasets[dataset].data[tuple(indeces+[...])] = single_measurement_result[dataset]
			state.datasets[dataset].indeces_updates = tuple(indeces+[...])
		state.done_sweeps += 1
		
		for event_handler, arguments in on_update:
			try:
				event_handler(state, indeces, *arguments)
			except Exception as e:
				print (e)

	for event_handler, arguments in on_start:
		try:
			event_handler(state, *arguments)
		except Exception as e:
			print (e)
		
		################
	if hasattr(measurer, 'pre_sweep'):
		measurer.pre_sweep()
				
	for indeces in all_indeces:
		if state.request_stop_acq:
			break
				# check which values have changed this sweep
		old_parameter_values = state.parameter_values
		state.parameter_values = [sweep_parameters[parameter_id].values[value_id] for parameter_id, value_id in enumerate(indeces)]
		changed_values = np.equal(old_parameter_values, state.parameter_values)#[old_parameter_values!=state.parameter_values for old_val, val in zip(old_vals, vals)]
		# set to new param vals
		for value, sweep_parameter, changed in zip(state.parameter_values, sweep_parameters, changed_values):
			if changed:
				setter_start = time.time()
				sweep_parameter.setter(value)
				sweep_parameter.setter_time += time.time() - setter_start
		#measuring
		measurement_start = time.time()
		if hasattr(measurer, 'measure_deferred_result') and use_deferred:
			measurer.measure_deferred_result(set_single_measurement_result, (indeces, ))
		else:
			mpoint = measurer.measure()
			#saving data to containers
			set_single_measurement_result(mpoint, indeces)
			
		state.measurement_time += time.time() - measurement_start
		
	if hasattr(measurer, 'join_deferred'):
		print ('Waiting to join deferred threads:')
		measurer.join_deferred()

	for event_handler, arguments in on_finish:
		try:
			event_handler(state, *arguments)
		except Exception as e:
			print (e)
	
	return state