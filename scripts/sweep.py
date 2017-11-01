import numpy as np
import time
from time import sleep
import sys
from matplotlib import pyplot as plt
import numbers
import itertools
import cmath
import save_pkl
import logging
import plotting

def optimize(target, *params, initial_simplex=None, maxfun=200):
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

def sweep(measurer, *params, filename=None, root_dir=None, plot=True, header=None, output=True):
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
	print (points)
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
	
	if root_dir is None:
		data_dir = save_pkl.mk_dir()
	else:
		data_dir = root_dir	
	
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
		plot_axes = plotting.plot_measurement(mk_measurement(0))
		last_plot_update = time.time()
		plot_update_time = plot_update_start - last_plot_update

	# starting sweep
	start_time = time.time()
	try:
		sweep_state_print("First sweep...",sweep_state_widget)
		# looping over all indeces at the same time
		all_indeces = itertools.product(*([i for i in range(d)] for d in sweep_dim))
		vals = [None for d in sweep_dim] # set initial parameter vals to None
		done_sweeps = 0
		total_sweeps = np.prod([d for d in sweep_dim])
		for indeces in all_indeces:
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
			plot_update_start = time.time()
			# if the last plot update was more than 20 times the time we need to update the plot, update
			# this is done to conserve computational resources
			if plot:
				if plot_update_start - last_plot_update > 10*plot_update_time:	
					plotting.update_plot_measurement(mk_measurement(0), plot_axes)
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
			stat_mes_fmt = '\rTime left: {0},\tparameter values: {1},\taverage cycle time: {2:4.2g}s\t'
			stat_mes = stat_mes_fmt.format(format_time_delta(avg_time*(total_sweeps-done_sweeps)), param_val_mes, avg_time)
			
			sweep_state_print(stat_mes,sweep_state_widget)
		print("\nElapsed time: "+format_time_delta(time.time() - start_time))
	
	finally:	
		for mname in points.keys():
			header = {'name': filename, 'type': mname, 'params':non_unity_dim_names(mname)}
			#sweep_xrange_pkl = non_unity_dim_vals(mname)
			#data_pkl = np.reshape(data[mname], non_unity_dims(mname))
			#if len(sweep_xrange_pkl)<2:
			#	sweep_xrange_pkl=sweep_xrange_pkl+tuple([0]*(2-len(sweep_xrange_pkl)))
			#data_pkl = tuple(sweep_xrange_pkl)+(np.abs(data_pkl), np.angle(data_pkl))

			#save_pkl.save_pkl(header, data_pkl, location = data_dir)
			save_pkl.save_pkl(header, mk_measurement(0), location = data_dir)
			
			if ascii_files[mname]:
				for fmt, ascii_file in ascii_files[mname].items():
					ascii_file.close()
					
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