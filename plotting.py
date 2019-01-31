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
import numbers
import matplotlib
import time

from matplotlib.ticker import EngFormatter

# high level function. You give the data, it plots everything that can be relatively easily plotted.
# puts complex into separate windows
# plots fits in the same axis
# puts all 2D's into separate windows
def plot_measurement(measurement, name=None, save=False, annotation=None, subplots=True):
	# check how many windows we need
	# cycle over the measurement dict and check stuff out
	font = {'size'   : 12}
	matplotlib.rc('font', **font)
	
	figsize=(8,4.5)
	axes = {}
	
	if subplots:
		figsize = (8, 4.5)
		num_axes = 0
		for mname, data in measurement.items():
			pnames = data[0]
			pvals = data[1]
			dims = data[2].shape
			dtype = data[2].dtype
			if not np.issubdtype(dtype, np.number):
				continue
			if len(dims)>2:
				continue
			if not isinstance(data[2].ravel()[0], numbers.Number):
				continue
			if np.iscomplexobj(data[2]):
				num_axes += 2
			else:
				num_axes += 1
			if mname+' fit' in measurement.keys() and len (dims)<2:					
				fit_pnames = measurement[mname+' fit'][0]
				same_axes = True
				for fit_pname in fit_pnames:
					if not fit_pname in pnames:
						same_axes = False
				if same_axes:
					if np.iscomplexobj(data[2]):
						num_axes -= 2
					else:
						num_axes -= 1
		
		#num_rows = np.sqrt(num_axes*(3/4))qwe
		num_cols = int(np.ceil(np.sqrt(num_axes*(4/3))))
		num_rows = int(np.ceil(num_axes/num_cols))
		
		while ((num_rows-1)*num_cols >= num_axes):
			num_rows -=1
		while ((num_cols-1)*num_rows >= num_axes):
			num_cols -=1			
			
		figure_exists = plt.fignum_exists(num=name)
			
		fig_window = plt.figure(num=name, figsize=figsize)
		if len(fig_window.get_axes()) >= num_rows*num_cols:
			subplot_axes = fig_window.get_axes()[:num_rows*num_cols]
		else:
			subplot_figs, subplot_axes = plt.subplots(num_rows, num_cols, figsize=figsize, num=name)
		subplot_axes = np.reshape(subplot_axes, (num_rows, num_cols))
		try:
			pass
			#plt.get_current_fig_manager().window.showMaximized()
		except:
			pass
	axis_id = 0
	for mname, data in measurement.items():
		pnames = data[0]
		pvals = data[1]
		dims = data[2].shape
		dtype = data[2].dtype
		if not np.issubdtype(dtype, np.number):
			continue
		log = False
		unwrap = False
		punits = ['' for i in range(len(pnames))]
		if len (data)>3:
			options = data[3]
			if 'log' in options:
				log = options['log']
			if 'unwrap'	in options:
				unwrap = options['unwrap']
			else:
				unwrap = True if len(dims) < 2 else False
			if 'scatter' in options:
				scatter = options['scatter']
			else:
				scatter = [False for i in range(len(pnames))]
			if 'realimag' in options:
				realimag = options['realimag']
			else:
				realimag = False
		if len (data) > 4:
			punits = data[4]
		
		filter_none = lambda x: x
		if not log:										filter_abs = np.abs
		elif log == 10:									filter_abs = lambda x: np.log10(np.abs(x))*10
		elif log == 20:									filter_abs = lambda x: np.log10(np.abs(x))*20
		if not unwrap:									filter_phase = np.angle
		else:											filter_phase = lambda x: np.unwrap(np.angle(x))
		if realimag:									filter_real = lambda x: np.real(x)
		if realimag:									filter_imag = lambda x: np.imag(x)
		
		if len(dims)>2:
			continue
		if not isinstance(data[2].ravel()[0], numbers.Number):
			continue
		
		plot_name = mname
		# check if it is complex			
		if np.iscomplexobj(data[2]):
			if not realimag:
				plot_filters = {plot_name+' amplitude':filter_abs, plot_name+' phase':filter_phase}
			else:
				plot_filters = {plot_name+' real':filter_real, plot_name+' imag':filter_imag}
		else:
			plot_filters = {plot_name:filter_none}
		
		# if this is a fit that can be plotted on same axes as experimental data, don't create separate axes
		if mname[-4:] == ' fit' and mname[:-4] in measurement.keys() and len (dims) < 2:
			exp_pnames = measurement[mname[:-4]][0]
			same_axes = True
			for fit_pname in pnames:
				if not fit_pname in exp_pnames:
					same_axes = False
			if same_axes:
				continue
			
		for filtered_name, filter in plot_filters.items():
			if not subplots:
				fig = plt.figure(filtered_name, figsize=figsize)
				if len(dims) == 2:
					plt.clf()
				axes[filtered_name] = {'axes': fig.add_axes([0.1, 0.1, 0.85, 0.85]), 'plots':{mname:{'mname':mname, 'filter': filter}}, 'subplots':False}
			else:
				ax = subplot_axes[int(axis_id%num_rows), int(axis_id/num_rows)]
				#if num_rows > 1: ax = subplot_axes[int(axis_id%num_rows), int(axis_id/num_rows)]
				#else: ax = subplot_axes[axis_id]
				
				axes[filtered_name] = {'axes': ax, 'plots':{mname:{'mname':mname, 'filter': filter}}, 'subplots':True}
				if len(dims) == 2:
					axes[filtered_name]['axes'].clear()
				axis_id+=1
			axes[filtered_name]['axes'].grid(True)
			axes[filtered_name]['axes'].set_title(filtered_name)
			try:
				axes[filtered_name].ticklabel_format(style='sci', scilimits=(-3,3), useOffset=False, axis='both', useLocale=True)
			except AttributeError:
				pass
				
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
					if not realimag:
						axes[plot_name+' amplitude']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_abs}
						axes[plot_name+' phase']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_phase}
					else: 
						axes[plot_name+' real']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_real}
						axes[plot_name+' imag']['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_imag}
				else:
					axes[plot_name]['plots'][mname+ ' fit'] = {'mname':mname+ ' fit', 'filter': filter_none}
				if not 'scatter' in options: # set scatter by default if there is a fit and less than 200 points in the plot.
					if len(dims) == 1:
						if dims[0] < 200:
							scatter = [True]

	# from here on, no more code relating to what should be plotted where.
	# creating separate measurement structures for calling plot_measurement_sa

	for ax_name, ax in axes.items():
		meas = {}
		for plot_name, plot in ax['plots'].items():
			meas[plot_name] = (measurement[plot['mname']][0], 
								measurement[plot['mname']][1], 
								plot['filter'](measurement[plot['mname']][2]),
								*measurement[plot['mname']][3:])
		plots = plot_measurement_sa(meas, ax['axes'])
		for plot in plots.keys():
			ax['plots'][plot]['plot'] = plots[plot]
		# if we are saving the current plot
	if subplots:
		if name and annotation:
			suptitle = name + ': ' + annotation
		elif name:
			suptitle = name
		elif annotation:
			suptitle = annotation
		else:
			suptitle = None
		if suptitle:
			plt.suptitle(suptitle)
	else:
		plot_add_annotation(axes, annotation)
		
	if not figure_exists:
		plt.figure(num=name).tight_layout(rect=[0.0, 0.03, 1, 0.95])		
	plot_save(axes, save)
	# return all the axes, filters & plotting instructions for updates
	return axes

def plot_add_annotation(axes, annotation):
	for ax_name, ax in axes.items():
		if annotation:
			ax['axes'].annotate(annotation, (0.1, 0.1), xycoords='axes fraction' if ax['subplots'] else 'figure fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
		if ax['subplots']:
			return

def plot_save(axes, save):
	plt.rcParams['svg.fonttype'] = 'none'
	for ax_name, ax in axes.items():
		if save:
			ax['axes'].get_figure().savefig('{0}/{1}.png'.format(save, ax_name))
			ax['axes'].get_figure().savefig('{0}/{1}.pdf'.format(save, ax_name))
			ax['axes'].get_figure().savefig('{0}/{1}.svg'.format(save, ax_name))
			with open('{0}/{1}.svg'.format(save, ax_name)) as fd:
				s = fd.read()
			with open('{0}/{1}.svg'.format(save, ax_name), 'w') as fd:
				fd.write(s.replace('stroke-miterlimit:100000;', ''))
			
		if ax['subplots']:
			return
			
# update plots with new data and axes structure returned by plot_measurement
def update_plot_measurement(measurement, axes):
	for ax_name, ax in axes.items():
		meas = {}
		for plot_name, plot in ax['plots'].items():
			meas[plot_name] = (measurement[plot['mname']][0], measurement[plot['mname']][1], plot['filter'](measurement[plot['mname']][2]))
		plots = update_plot_measurement_sa(meas, ax)
	
	plt.draw()	
	#plt.gcf().canvas.start_event_loop(0.001)
	plt.gcf().canvas.start_event_loop(0.01)
	
# Plots all data on a single axis with given annotations.
def plot_measurement_sa(measurement, axes):
	#first plot 2d stuff
	fig = axes.get_figure()
	#plt.figure(fig.canvas.get_window_title())
	axes_names = []
	plots = {}
	for data_dim in [2, 1]:
		for mname, data in measurement.items():
			pnames = data[0]
			pvals = data[1]
			vals = data[2]
			dims = data[2].shape
			if len(data)>3:
				opts = data[3]
			else:
				opts = {}
			if len(data)>4:
				punits = data[4]
			else:
				punits = ['' for i in range(len(pnames))]
			
			# scatter plot or lines? depends on the opts.
			if 'scatter' in opts:
				plot_kwargs = {'marker':'o', 'markerfacecolor':'None'}
			else:
				plot_kwargs = {}
			
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
					formatter = EngFormatter(unit=punits[0])
					axes.xaxis.set_major_formatter(formatter)
					plot = axes.plot(pvals[0], vals, **plot_kwargs)
					# bug where plot return a list instead of a Line2D
					if hasattr(plot, '__iter__'):
						plot = plot[0]
					plot.T = False
				else:
					formatter = EngFormatter(unit=punits[0])
					axes.yaxis.set_major_formatter(formatter)
					plot = axes.plot(vals, pvals[0], **plot_kwargs)
					# bug where plot return a list instead of a Line2D
					if hasattr(plot, '__iter__'):
						plot = plot[0]
					plot.T = True
			if data_dim == 2:
				if axes_indeces[0] == 0:
					xformatter = EngFormatter(unit=punits[0])
					axes.xaxis.set_major_formatter(xformatter)
					yformatter = EngFormatter(unit=punits[1])
					axes.yaxis.set_major_formatter(yformatter)
					plot = axes.imshow(vals.T, aspect = 'auto', origin='lower', 
						extent = [pvals[0][0], pvals[0][-1], pvals[1][0], pvals[1][-1] ], interpolation = "none", cmap='RdBu_r')
					plot.T = True
					#axes.xaxis.set_major_formatter(xformatter)
					#axes.yaxis.set_major_formatter(yformatter)
				else:
					xformatter = EngFormatter(unit=punits[0])
					axes.yaxis.set_major_formatter(xformatter)
					yformatter = EngFormatter(unit=punits[1])
					axes.xaxis.set_major_formatter(yformatter)
					plot = axes.imshow(vals, aspect = 'auto', origin='lower', 
						extent = [pvals[1][0], pvals[1][-1], pvals[0][0], pvals[0][-1] ], interpolation = "none", cmap='RdBu_r')
					plot.T = False
					#axes.xaxis.set_major_formatter(xformatter)
					#axes.yaxis.set_major_formatter(yformatter)
				if hasattr(axes, 'cb'):
					plot.cb = plt.colorbar(plot, ax=axes, cax=axes.cb.ax)
				else:
					plot.cb = plt.colorbar(plot, ax=axes)
				axes.cb = plot.cb
			plots[mname] = plot
	if len(axes_names) >= 1:
		axes.set_xlabel(axes_names[0])
	if len(axes_names) >= 2:
		axes.set_ylabel(axes_names[1])
	plt.gcf().canvas.start_event_loop(0.05)
	return plots

def update_plot_measurement_sa(measurement, plots):
	for mname, meas in measurement.items():
		pnames = meas[0]
		pvals = meas[1]
		data = meas[2]
		plot = plots['plots'][mname]['plot']
		fig = plot.axes.get_figure()
		#plt.figure(fig.canvas.get_window_title())
			
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
			plot.set_clim(np.nanmin(data), np.nanmax(data))
		plt.gcf().canvas.start_event_loop(1.0)
	
