# plotting script
# used for interactive plotting of measurement data without a fuss
# should be able to plot three general types of data:
# 1) 1D plots: curves (with persist)
# 2) 2D plots: pyplot imshow
# scatter plots (disconnected points)

# for complex data, should plot abs and phase and perhaps unwrapped phase - in separate windows
# should be able to plot fit in the same window
# should be able to add annotations to the plot

from matplotlib import pyplot as plt
import numpy as np

# high level function. You give the data, it plots everything that can be relatively easily plotted.
# puts complex into separate windows
# plots fits in the same axis
# puts all 2D's into separate windows
def plot_measurement(measurement, name=None, save=False, annotation=None):
	# check how many windows we need
	# cycle over the measurement dict and check stuff out
	figsize=(12,8)
	axes = {}
	remove_separate = []
	for mname, data in measurement.items():
		pnames = data[0]
		pvals = data[1]
		dims = data[2].shape
		dtype = data[2].dtype
		log = False
		unwrap = False
		if len (data)>3:
			options = data[3]
			if 'log' in options:
				log = options['log']
			if 'unwrap'	in options:
				unwrap = options['unwrap']
		
		if not log and not np.iscomplexobj(data[2]):	filter_abs = lambda x: x
		elif not log:									filter_abs = np.abs
		elif log == 10:									filter_abs = lambda x: np.log10(np.abs(x))*10
		elif log == 20:									filter_abs = lambda x: np.log10(np.abs(x))*20
		if not unwrap:									filter_phase = np.angle
		else:											filter_phase = lambda x: np.unwrap(np.angle(x))
		
		if len(dims)>2:
			continue
		
		plot_name = mname
		# check if it is complex			
		if np.iscomplexobj(data[2]):
			axes[plot_name+' amplitude'] = {'axes': plt.figure(plot_name+' amplitude', figsize=figsize).add_axes([0.1, 0.1, 0.85, 0.85]), \
											'plots':{mname:{'mname':mname, 'filter': filter_abs}}}
			axes[plot_name+' phase'] = {'axes': plt.figure(plot_name+' phase', figsize=figsize).add_axes([0.1, 0.1, 0.85, 0.85]), \
											'plots':{mname:{'mname':mname, 'filter': filter_phase}}}
		else:
			axes[plot_name] = {'axes': plt.figure(plot_name, figsize=figsize).add_axes([0.1, 0.1, 0.80, 0.85]), \
								'plots':{mname:{'mname':mname, 'filter': filter_abs}}}
		# if there is a fit of this measurement and it has the same axes, add it the plot and don't plot it separately
		if mname+' fit' in measurement.keys() and len (dims)<2:
			fit_pnames = measurement[mname+' fit'][0]
			same_axes = True
			for fit_pname in fit_pnames:
				if not fit_pname in pnames:
					same_axes = False
			# if the fit can be plotted in the same axes, plot it!
			if same_axes:
				if np.iscomplexobj(data[2]):				
					axes[plot_name+' amplitude']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_abs}
					axes[plot_name+' phase']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_phase}
					remove_separate.extend([plot_name+' fit amplitude', plot_name+' fit phase'])
				else:
					axes[plot_name]['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_abs}
					remove_separate.extend([plot_name+' fit'])
	# remove plots that are unneeded separately
	for r in remove_separate:
		if r in axes.keys():
			fig = axes[r]['axes'].get_figure()
			plt.figure(fig.canvas.get_window_title())
			plt.close()
			del axes[r]
	# from here on, no more code relating to what should be plotted where.
	# creating separate measurement structures for calling plot_measurement_sa
	for ax_name, ax in axes.items():
		meas = {}
		for plot_name, plot in ax['plots'].items():
			meas[plot_name] = (measurement[plot['mname']][0], measurement[plot['mname']][1], plot['filter'](measurement[plot['mname']][2]))
		plots = plot_measurement_sa(meas, ax['axes'])
		for plot in plots.keys():
			ax['plots'][plot]['plot'] = plots[plot]
		if annotation:
			ax['axes'].annotate(annotation, (0.1, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			
		# if we are saving the current plot
		if save:
			plt.grid()
			plt.savefig('{0}/{1}.png'.format(save, ax_name))
			
	# return all the axes, filters & plotting instructions for updates
	return axes

# update plots with new data and axes structure returned by plot_measurement
def update_plot_measurement(measurement, axes):
	for ax_name, ax in axes.items():
		meas = {}
		for plot_name, plot in ax['plots'].items():
			meas[plot_name] = (measurement[plot['mname']][0], measurement[plot['mname']][1], plot['filter'](measurement[plot['mname']][2]))
		plots = update_plot_measurement_sa(meas, ax)
	
# Plots all data on a single axis with given annotations.
def plot_measurement_sa(measurement, axes):
	#first plot 2d stuff
	fig = axes.get_figure()
	plt.figure(fig.canvas.get_window_title())
	axes_names = []
	plots = {}
	for data_dim in [2, 1]:
		for mname, data in measurement.items():
			pnames = data[0]
			pvals = data[1]
			vals = data[2]
			dims = data[2].shape
			#give axis names if there are spare axes
			for pname in pnames:
				if len(axes_names) < 2:
					if not pname in axes_names:
						axes_names.append(pname)	
			# check if non-singleton axes of the current measurement are present on the plot
			# enumerate data axes by plot axes
			axes_indeces = []
			for pname in pnames:
				if not (pname in axes_names):
					raise ValueError('Axis {0} not present on current plot. Available axes are '.format(pname)+' ,'.join(axes_names))
				axes_indeces.append(axes_names.index(pname))
			if len(dims) != data_dim: continue
			if data_dim == 1:
				if axes_indeces[0] == 0:
					plot = axes.plot(pvals[0], vals)
					# bug where plot return a list instead of a Line2D
					if hasattr(plot, '__iter__'):
						plot = plot[0]
					plot.T = False
				else:
					plot = axes.plot(vals, pvals[0])
					# bug where plot return a list instead of a Line2D
					if hasattr(plot, '__iter__'):
						plot = plot[0]
					plot.T = True
			if data_dim == 2:
				plt.clf()
				if axes_indeces[0] == 0:
					plot = plt.imshow(vals.T, aspect = 'auto', origin='lower', 
						extent = [pvals[0][0], pvals[0][-1], pvals[1][0], pvals[1][-1] ], interpolation = "none", cmap='RdBu_r')
					plot.T = True
				else:
					plot = plt.imshow(vals, aspect = 'auto', origin='lower', 
						extent = [pvals[1][0], pvals[1][-1], pvals[0][0], pvals[0][-1] ], interpolation = "none", cmap='RdBu_r')
					plot.T = False
				plot.cb = plt.colorbar()
			plots[mname] = plot
			
		if len(axes_names) >= 1:
			plt.xlabel(axes_names[0])
		if len(axes_names) >= 2:
			plt.ylabel(axes_names[1])
	plt.pause(0.05)
	return plots

def update_plot_measurement_sa(measurement, plots):
	for mname, meas in measurement.items():
		pnames = meas[0]
		pvals = meas[1]
		data = meas[2]
		
		plot = plots['plots'][mname]['plot']
		fig = plot.axes.get_figure()
		plt.figure(fig.canvas.get_window_title())
			
		if len(data.shape)==1:
			if plot.T:
				plot.set_data(data, pvals[0])
			else:
				plot.set_data(pvals[0], data)
			plot.axes.relim()
			plot.axes.autoscale_view(True,True,True)		
		if len(data.shape)==2:
			if plot.T:
				data = data.T
			plot.set_data(data)	
			plt.clim(np.nanmin(data), np.nanmax(data))
		plt.draw()	
	plt.pause(0.05)