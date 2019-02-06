import numpy as np
import time
from time import sleep
import sys
from matplotlib import pyplot as plt
import numbers
import itertools
import cmath
from . import save_pkl
import logging
from . import plotting
import pickle as pic
import shutil as sh
import threading
import pathlib
import random
import gc
from .data_structures import *
from datetime import datetime

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
	
def sweep_new(measurer, *parameters, shuffle=False, on_start = [], on_update=[], on_finish=[], filename, meas_type, reference = {}):
	'''
	Performs a n-d parametric sweep.
	Usage: sweep(measurer, (param1_values, param1_setter, [param1_name]), (param2_values, param2_setter), ... , filename=None)
	measurer: an object that supports get_points(), measure(), get_dtype() and get_opts() methods.
	
	Returns: measurement_state struct after measurement dict of ndarrays each corresponding to a measurment in the sweep
	'''
	
	sweep_parameters = [measurement_parameter(*parameter) for parameter in parameters]
	point_parameters = measurer_point_parameters(measurer)
	
	sweep_dimensions = tuple([len(sweep_parameter.values) for sweep_parameter in sweep_parameters])
	state = measurement_state(sweep_parameters, filename, meas_type, reference)
	state.total_sweeps = np.prod([d for d in sweep_dimensions])
	#all_parameters = {dataset: sweep_parameters+_point_parameters for dataset, _point_parameters in point_parameters.items()}
	
	### initialize data
	for dataset_name, point_parameters in point_parameters.items():
		print(sweep_parameters, point_parameters)
		all_parameters = sweep_parameters + point_parameters
		data_dimensions = tuple([len(parameter.values) for parameter in all_parameters])
		data = np.empty(data_dimensions, dtype = measurer.get_dtype()[dataset_name])
		data.fill(np.nan)
		state.datasets[dataset_name] = measurement_dataset(parameters = all_parameters, data = data)
	
	all_indices = itertools.product(*([i for i in range(d)] for d in sweep_dimensions))
	if shuffle:
		all_indeces = [i for i in all_indeces]
		random.shuffle(all_indeces)
	if len(sweep_dimensions)==0: # 0-d sweep case: single measurement
		all_indeces = [[]]	
	
	def set_single_measurement_result(single_measurement_result, indices):
		nonlocal state
		indices = list(indices)
		for dataset in state.datasets.keys():
			state.datasets[dataset].data[tuple(indices+[...])] = single_measurement_result[dataset]
			state.datasets[dataset].indices_updates = tuple(indices+[...])
		state.done_sweeps += 1
		
		for event_handler, arguments in on_update:
			event_handler(state, *arguments)

	for event_handler, arguments in on_start:
		event_handler(state, *arguments)
		
		################
	if hasattr(measurer, 'pre_sweep'):
		measurer.pre_sweep()
				
	for indices in all_indices:
		if state.request_stop_acq:
			break
				# check which values have changed this sweep
		old_parameter_values = state.parameter_values
		state.parameter_values = [sweep_parameters[parameter_id].values[value_id] for parameter_id, value_id in enumerate(indices)]
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
			measurer.measure_deferred_result(set_single_measurement_result, (indices, ))
		else:
			mpoint = measurer.measure()
			#saving data to containers
			set_single_measurement_result(mpoint, indices)
			
		state.measurement_time += time.time() - measurement_start
		
	if hasattr(measurer, 'join_deferred'):
		print ('Waiting to join deferred threads:')
		measurer.join_deferred()

	for event_handler, arguments in on_finish:
		event_handler(state, *arguments)
	state.stop = datetime.now()
	return state
	
	
def sweep(measurer, 
		  *params, 
		  filename=None, 
		  root_dir=None, 
		  plot=True, 
		  plot_separate_thread=True, 
		  header=None, 
		  output=True, 
		  save = True, 
		  time_war_label=True, 
		  shuffle=False, 
		  bot=(False,0),
		  use_deferred=False,
		  loc=False):
	'''
	Performs a n-d parametric sweep.
	Usage: sweep(measurer, (param1_values, param1_setter, [param1_name]), (param2_values, param2_setter), ... , filename=None)
	measurer: an object that supports get_points(), measure(), get_dtype() and get_opts() methods.
	
	Returns: dict of ndarrays each corresponding to a measurment in the sweep
	'''
	# kill the annoying warning:
	#c:\python36\lib\site-packages\matplotlib\backend_bases.py:2445: MatplotlibDeprecationWarning: Using default event loop until function specific to this GUI is implemented
    #warnings.warn(str, mplDeprecation)
	import warnings
	warnings.filterwarnings("ignore",".*Using default event loop until function specific to this GUI is implemented*")
	if (time_war_label):
		try:
			from ipywidgets import HTML
			from IPython.display import display
			sweep_state_widget = HTML()
			display(sweep_state_widget)
		except:
			sweep_state_widget = None
	
		print("Started at: ", time.strftime("%b %d %Y %H:%M:%S", time.localtime()))
		start_time = datetime.now()#time.time()
	
	setters_time = 0
	measurement_time = 0
	
	# extracting sweep parameters from list
	sweep_dim_vals = []
	sweep_dim_names = []
	sweep_dim_units = []
	sweep_dim_setters = []
	sweep_dim_pre_setters = []
	for param_id, param in enumerate(params):
		if type(param) is tuple:
			values = param[0]
			setter = param[1]
			name = param[2] if len(param)>2 else 'param_{0}'.format(param_id)
			unit = param[3] if len(param)>3 else ''
			pre_setter = param[4] if len(param)>4 else lambda: None
		elif type(param) is dict:
			values = param['values']
			setter = param['setter']
			name = param['name'] if 'name' in param else 'param_{0}'.format(param_id)
			unit = param['unit'] if 'unit' in param else ''
			pre_setter = param['pre_setter'] if 'pre_setter' in param else lambda: None
		sweep_dim_vals.append(values)
		sweep_dim_setters.append(setter)
		sweep_dim_names.append(name)
		sweep_dim_units.append(unit)
		sweep_dim_pre_setters.append(pre_setter)
	sweep_dim = tuple(len(values) for values in sweep_dim_vals)
	sweep_dim_vals = tuple(sweep_dim_vals)
	sweep_dim_names = tuple(sweep_dim_names)
	sweep_dim_units = tuple(sweep_dim_units)
	sweep_dim_setters = tuple(sweep_dim_setters)
	sweep_dim_pre_setters = tuple(sweep_dim_pre_setters)
		
	# determining return type of measurer: in could be 2 traces (im, re)
	point_types = measurer.get_dtype()
	meas_opts = measurer.get_opts()
	points = measurer.get_points() # should return dict of meas_name:points
	point_dim_names = {mname:tuple([cname	   for cname, cvals, cunits in mxvals]) for mname, mxvals in points.items()}
	point_dim		= {mname:tuple([len(cvals) for cname, cvals, cunits in mxvals]) for mname, mxvals in points.items()}
	point_dim_vals	= {mname:tuple([cvals	   for cname, cvals, cunits in mxvals]) for mname, mxvals in points.items()}
	point_dim_units	= {mname:tuple([cunits	   for cname, cvals, cunits in mxvals]) for mname, mxvals in points.items()}
	#point_dim = {mname:tuple([len(mxvals[cname]) for cname in point_dim_names[mname]]) for mname, mxvals in points.items()}
	#point_dim_vals = {mname:tuple([mxvals[1] for cname in point_dim_names[mname]]) for mname, mxvals in points.items()}
	# initializing empty ndarrays for measurment results
	data = {mname: np.zeros(sweep_dim+point_dim[mname], dtype=point_types[mname])*np.nan for mname, mxvals in points.items()}
	
	def non_unity_dims(mname):
		return tuple(dim for dim in data[mname].shape if dim > 1)
		
	def non_unity_dim_names(mname):
		all_dim_names = sweep_dim_names+point_dim_names[mname]
		return tuple(dim_name for dim, dim_name in zip(data[mname].shape, all_dim_names) if dim > 1)

	def non_unity_dim_vals(mname):
		all_dim_vals = sweep_dim_vals+point_dim_vals[mname]
		return tuple(dim_name for dim, dim_name in zip(data[mname].shape, all_dim_vals) if dim > 1)
		
	def non_unity_dim_units(mname):
		all_dim_units = sweep_dim_units+point_dim_units[mname]
		return tuple(dim_name for dim, dim_name in zip(data[mname].shape, all_dim_units) if dim > 1)
	
	ascii_files = {mname:None for mname in points.keys()}
	mk_measurement = lambda: {mname: (non_unity_dim_names(mname), 
							non_unity_dim_vals(mname), 
							np.reshape(data[mname], non_unity_dims(mname)),
							meas_opts[mname], non_unity_dim_units(mname)) for mname in points.keys()}
	import os
	
	if not root_dir and not filename:
		data_dir = save_pkl.default_measurement_save_path()
		directory_to = save_pkl.default_measurement_save_path(unfinished=True)
	elif filename and not root_dir:
		data_dir = save_pkl.default_measurement_save_path(name=filename)
		directory_to = save_pkl.default_measurement_save_path(unfinished=True, name=filename)
	else:
		data_dir = root_dir	
		directory_to = root_dir+'/unfinished/'
	
	data_dir = data_dir+'/'
	pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True) # create top-level directory for moving
	
	#opening on-the-fly ascii files
	if not filename is None and output:
		def data_to_str(data, fmt = '{:e}\t'):
			if hasattr(data, '__iter__'):
				return ''.join([fmt.format(d) for d in data])
			else:
				return fmt.format(data)
				
		for mname in points.keys():
			# if the file has more than 1.000.000 points or more than 2 dimensions don't save to ascii
			if data[mname].size < 1e6 and len(non_unity_dims(mname))<=2:
				if point_types[mname] is complex:
					ascii_files[mname] = {np.abs: open(data_dir + filename + '_amp.dat', 'w'), 
										  np.angle: open(data_dir + filename + '_ph.dat', 'w')}
				if point_types[mname] is float:
					ascii_files[mname] = {(lambda x: x): open(data_dir + filename + '.dat', 'w')}
										  
	# opening plot windows
	if plot:
		plot_update_start = time.time()
		plot_axes = plotting.plot_measurement(mk_measurement(), name=filename)
		last_plot_update = time.time()
		plot_update_time = last_plot_update - plot_update_start
	else:
		plot_axes=None
	start_time = time.time()
	acq_thread = None
	sweep_error = False
	stop_acq = False
	if (time_war_label):
	# starting sweep
		sweep_state_print("First sweep...",sweep_state_widget)
	try:
		# looping over all indeces at the same time
		all_indeces = itertools.product(*([i for i in range(d)] for d in sweep_dim))
		if shuffle:
			all_indeces = [i for i in all_indeces]
			random.shuffle(all_indeces)
		if len(sweep_dim)==0: # 0-d sweep case: single measurement
			all_indeces = [[]] 
		vals = [None for d in sweep_dim] # set initial parameter vals to None
		done_sweeps = 0
		total_sweeps = np.prod([d for d in sweep_dim])
		#variable for Telebot
		if bot[0]==True:
			bot_time_cycle_iterator=1
			with open(data_dir+"/send_result.txt",'w') as bot_send_file:
				bot_send_file.write('0')
			bot_send_file.close()
		
		def set_single_measurement_result(mpoint, indeces):
			nonlocal done_sweeps
			nonlocal data
			#print ('indeces:', indeces)
			for mname in points.keys():
				mpoint_m = mpoint[mname]
				#print (mname, mpoint_m.shape, data[mname].shape)
				data[mname][[(i) for i in indeces]+[...]] = mpoint_m
				data_flat = np.reshape(data[mname], non_unity_dims(mname))
				mpoint_m_flat = np.reshape(mpoint_m, tuple(dim for dim in point_dim[mname] if dim > 1))
						
				#Save data to text file (on the fly) 
				#incorrect with shuffle
				if ascii_files[mname]:
					for fmt, ascii_file in ascii_files[mname].items():
						ascii_file.write(data_to_str(fmt(mpoint_m_flat))+'\n')
						ascii_file.flush()
						
				update_time_info()
			done_sweeps += 1
		
		def update_time_info():
			avg_time = (time.time() - start_time)/(done_sweeps+1)
			avg_setters_time = setters_time/(done_sweeps+1)
			avg_measurement_time = measurement_time/(done_sweeps+1)
			param_val_mes = ',\t'.join(['{0}: {1:6.4g}'.format(name, val) for name,val in zip(sweep_dim_names,vals)])
			stat_mes_fmt = '\rTime left: {0},\tparameter values: {1},\taverage cycle time: {2:4.2g}s\t,(setter_time: {4:4.2g}s, measurement_time: {5:4.2g}s)\t,plot_update_time: {3:4.2g}s'
			stat_mes = stat_mes_fmt.format(format_time_delta(avg_time*(total_sweeps-done_sweeps-1)), 
											param_val_mes, 
											avg_time, 
											plot_update_time if plot else 0.0,
											avg_setters_time,
											avg_measurement_time)
			if (time_war_label):
				sweep_state_print(stat_mes,sweep_state_widget)
		
		def main_sweep_loop():
			try:
				nonlocal measurement_time
				nonlocal setters_time
				nonlocal sweep_error
				nonlocal vals
				nonlocal data
				nonlocal plot_update_start 
				nonlocal last_plot_update
				nonlocal done_sweeps
				nonlocal bot_time_cycle_iterator
				nonlocal plot_update_time
				nonlocal acq_thread
				if hasattr(measurer, 'pre_sweep'):
					measurer.pre_sweep()
				for indeces in all_indeces:
					if stop_acq:
						break
					# check which values have changed this sweep
					old_vals = vals
					vals = [sweep_dim_vals[param_id][val_id] for param_id, val_id in enumerate(indeces)]
					changed_vals = [old_val!=val for old_val, val in zip(old_vals, vals)]
					# set to new param vals
					setters_start = time.time()
					for val, setter, changed in zip(vals, sweep_dim_setters, changed_vals):
						if changed:
							setter(val)
					setters_time += time.time() - setters_start
					#measuring
					measurement_start = time.time()
					if hasattr(measurer, 'measure_deferred_result') and use_deferred:
						measurer.measure_deferred_result(set_single_measurement_result, (indeces, ))
					else:
						mpoint = measurer.measure()
						#saving data to containers
						set_single_measurement_result(mpoint, indeces)
					measurement_time += time.time() - measurement_start
						#Plot data (on the fly)
						# measure when we started updating plot
					# if the last plot update was more than 20 times the time we need to update the plot, update
					# this is done to conserve computational resources
					if plot and not plot_separate_thread:
						if time.time() - last_plot_update >  10*plot_update_time:	
							plot_update_start = time.time()
							plotting.update_plot_measurement(mk_measurement(), plot_axes)
							last_plot_update = time.time();
							plot_update_time = last_plot_update - plot_update_start
							
					#Telebot
					if bot[0] and (done_sweeps-bot[1]*bot_time_cycle_iterator)>0:
						bot_time_cycle_iterator += 1
						for mname in points.keys():
							header = {'name': filename, 'type': mname, 'params':non_unity_dim_names(mname)}
							save_pkl.save_pkl(header, mk_measurement(), location = data_dir)
				if hasattr(measurer, 'join_deferred'):
					print ('Waiting to join deferred threads:')
					measurer.join_deferred()
				print ('Exiting sweep main thread #{} via return', threading.current_thread().ident)
				return
			except:
				sweep_error = True
				raise
			finally:
				if hasattr(measurer, 'post_sweep'):
					measurer.post_sweep()
				
		if not plot_separate_thread:
			main_sweep_loop()
		elif plot:
			def plot_thread():
				plot_thread_point_id = 0
				while True:
					if sweep_error:
						raise KeyboardInterrupt
					if plot_thread_point_id < done_sweeps:
						plotting.update_plot_measurement(mk_measurement(), plot_axes)
						plot_thread_point_id = done_sweeps
					if plot_thread_point_id >= np.prod(sweep_dim):
						return
					#print(plot_thread_point_id, done_sweeps)
					plt.gcf().canvas.start_event_loop(2.0)
				#for i in range(100):
				#	plt.gcf().canvas.start_event_loop(0.02)
				#	time.sleep(0.01)
					
			acq_thread = threading.Thread(target=main_sweep_loop)
			acq_thread.start()
			plot_thread()
			
		if (time_war_label):
			print("\nElapsed time: "+format_time_delta(time.time() - start_time))
		for mname in points.keys():
			if ascii_files[mname]:
				for fmt, ascii_file in ascii_files[mname].items():
					ascii_file.close()
					
	except KeyboardInterrupt:
		for mname in points.keys():
			if ascii_files[mname]:
				for fmt, ascii_file in ascii_files[mname].items():
					ascii_file.close()
		stop_acq = True
		pathlib.Path(directory_to).parent.mkdir(parents=True, exist_ok=True) # create top-level directory for moving
		directory_to = sh.move(data_dir,directory_to)
		data_dir = directory_to
		raise # after finally
	finally:
		if acq_thread:
			if acq_thread.isAlive():
				stop_acq = True
		header = {'name': filename}
		if (save):
			filename_location = save_pkl.save_pkl(header, mk_measurement(), location = data_dir, plot_axes=plot_axes)
		if bot[0]==True:
			with open(data_dir+"/send_result.txt",'w') as bot_send_file:
				bot_send_file.write('1')
			bot_send_file.close()	
	
	gc.collect()
	if loc==True:
		return mk_measurement(),data_dir+filename+'.pkl'
	else:
		return mk_measurement()
		
		
def sweep_vna(measurer, *params, filename=None, root_dir=None, plot=True, header=None, output=True):
	return sweep(measurer, *params, filename=filename, root_dir=root_dir, plot=plot, header=header, output=output)['S-parameter']
	
def sweep_state_print(message, widget=None):
	if widget:
		widget.value = message
	else:
		sys.stdout.write(message)
		sys.stdout.flush()

def format_time_delta(delta):
	hours, remainder = divmod(delta, 3600)
	minutes, seconds = divmod(remainder, 60)
	return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))