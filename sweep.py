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
	
def sweep(measurer, *params, filename=None, root_dir=None, plot=True, plot_separate_thread=True, header=None, output=True, save = True, time_war_label=True,bot=(False,0)):
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
		start_time = time.time()
	
	# extracting sweep parameters from list
	# check if it is a list of pairs
	sweep_dim = tuple([len(param[0]) for param in params])
	sweep_dim_vals = tuple([param[0] for param in params])
	sweep_dim_names = tuple([param[2] if len(param)>2 else 'param_{0}'.format(param_id) for param_id, param in enumerate(params)])
	sweep_dim_units = tuple([param[3] if len(param)>3 else '' for param_id, param in enumerate(params)])
	sweep_dim_setters = tuple(param[1] for param in params)
	
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
	mk_measurement = lambda x: {mname: (non_unity_dim_names(mname), 
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
		plot_axes = plotting.plot_measurement(mk_measurement(0), name=filename)
		last_plot_update = time.time()
		plot_update_time = last_plot_update - plot_update_start
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
			
		def main_sweep_loop():
			try:
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
					vals = [params[param_id][0][val_id] for param_id, val_id in enumerate(indeces)]
					changed_vals = [old_val!=val for old_val, val in zip(old_vals, vals)]
					# set to new param vals
					for val, setter, changed in zip(vals, sweep_dim_setters, changed_vals):
						if changed:
							setter(val)
					#measuring
					mpoint = measurer.measure()
					#saving data to containers
					for mname in points.keys():
						mpoint_m = mpoint[mname]
						#print (mname, mpoint_m.shape, data[mname].shape)
						data[mname][[(i) for i in indeces]+[...]] = mpoint_m
						data_flat = np.reshape(data[mname], non_unity_dims(mname))
						mpoint_m_flat = np.reshape(mpoint_m, tuple(dim for dim in point_dim[mname] if dim > 1))
						
						#Save data to text file (on the fly)
						if ascii_files[mname]:
							for fmt, ascii_file in ascii_files[mname].items():
								ascii_file.write(data_to_str(fmt(mpoint_m_flat))+'\n')
								ascii_file.flush()
						#Plot data (on the fly)
						# measure when we started updating plot
					# if the last plot update was more than 20 times the time we need to update the plot, update
					# this is done to conserve computational resources
					if plot and not plot_separate_thread:
						if time.time() - last_plot_update >  10*plot_update_time:	
							plot_update_start = time.time()
							plotting.update_plot_measurement(mk_measurement(0), plot_axes)
							last_plot_update = time.time();
							plot_update_time = last_plot_update - plot_update_start
					'''
						if plot_windows[mname]:
							for fmt, plot in plot_windows[mname].items():
								update_plot(plot, non_unity_dim_vals(mname), fmt(data_flat))
								plt.pause(0.01)
						last_plot_update = time.time()
						plot_update_time = last_plot_update - plot_update_start
					'''
						
					done_sweeps += 1
					avg_time = (time.time() - start_time)/done_sweeps
					param_val_mes = ',\t'.join(['{0}: {1:6.4g}'.format(param[2], val) for param,val in zip(params,vals)])
					stat_mes_fmt = '\rTime left: {0},\tparameter values: {1},\taverage cycle time: {2:4.2g}s\t, plot_update_time: {3:4.2g}s'
					stat_mes = stat_mes_fmt.format(format_time_delta(avg_time*(total_sweeps-done_sweeps)), param_val_mes, avg_time, plot_update_time if plot else 0.0)
					if (time_war_label):
						sweep_state_print(stat_mes,sweep_state_widget)
					#Telebot
					if bot[0] and (done_sweeps-bot[1]*bot_time_cycle_iterator)>0:
						bot_time_cycle_iterator += 1
						for mname in points.keys():
							header = {'name': filename, 'type': mname, 'params':non_unity_dim_names(mname)}
							save_pkl.save_pkl(header, mk_measurement(0), location = data_dir)
					#print ('DAQ threadsweep no: ',done_sweeps)
			except:
				sweep_error = True
				if hasattr(measurer, 'post_sweep'):
					measurer.post_sweep()
				raise
			if hasattr(measurer, 'post_sweep'):
				measurer.post_sweep()
				
		if not plot_separate_thread:
			main_sweep_loop()
		else:
			def plot_thread():
				plot_thread_point_id = 0
				while True:
					if sweep_error:
						raise KeyboardInterrupt
					if plot_thread_point_id < done_sweeps:
						plotting.update_plot_measurement(mk_measurement(0), plot_axes)
						plot_thread_point_id = done_sweeps
					if plot_thread_point_id >= np.prod(sweep_dim):
						return
					plt.pause(1.0)
					#print(plot_thread_point_id, done_sweeps)

					
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
		sh.move(data_dir,directory_to)
		directory_to = sh.move(data_dir,directory_to)
		data_dir = directory_to
	finally:
		if acq_thread:
			if acq_thread.isAlive():
				stop_acq = True
		header = {'name': filename}
		if (save):
			save_pkl.save_pkl(header, mk_measurement(0), location = data_dir, plot_axes=plot_axes)
		if bot[0]==True:
			with open(data_dir+"/send_result.txt",'w') as bot_send_file:
				bot_send_file.write('1')
			bot_send_file.close()	
			
	
	return mk_measurement(0)

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