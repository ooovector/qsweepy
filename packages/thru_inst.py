# уменьшает количество данных от оцифровщика, чтобы не было MemoryError всякого
# подлежит применению во всяких свипах когда нет сил все эти гиги хранить.

import numpy as np
import logging

class data_reduce:
	def __init__(self, source):
		self.source = source
		self.filters = {}
		
	def get_points(self):
		return { filter_name:filter['get_points']() for filter_name, filter in self.filters.items()}
	
	def get_dtype(self):
		return { filter_name:filter['get_dtype']() for filter_name, filter in self.filters.items()}
		
	def measure(self):
		data = self.source.measure()
		return { filter_name:filter['filter'](data) for filter_name, filter in self.filters.items()}
		
	def get_opts(self):
		return { filter_name:filter['get_opts']() for filter_name, filter in self.filters.items()}
		
def mean_reducer(source, src_meas, axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	filter = {'filter': lambda x:np.mean(x[src_meas], axis=axis),
			  'get_points': get_points,
			  'get_dtype': (lambda : source.get_dtype()[src_meas]),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def mean_reducer_noavg(source, src_meas, axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	filter = {'filter': lambda x:np.mean(x[src_meas], axis=axis)-np.mean(x[src_meas]),
			  'get_points': get_points,
			  'get_dtype': (lambda : source.get_dtype()[src_meas]),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def mean_reducer_freq(source, src_meas, axis_mean, freq):
	axis_dm = None
	for axis_id, axis in enumerate(source.get_points()[src_meas]):
		if axis[0] == 'Time':
			axis_dm = axis_id
	if not axis_dm:
		logging.error('mean_reducer_freq: instrument {0} has no axis "Time" for demodulation.'.format(source))

	if axis_dm>axis_mean:
		axis_dm_new = axis_dm-1
		
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [np.max([axis_dm, axis_mean])]
		del new_axes [np.min([axis_dm, axis_mean])]
		return new_axes

			
	def filter_func(x):
		dm = np.exp(1j*2*np.pi*source.get_points()[src_meas][axis_dm][1]*freq)
		mean_sample = np.mean(x[src_meas], axis=axis_mean)
		return np.mean(mean_sample*dm, axis=axis_dm_new)
	
	filter = {'filter': filter_func,
			  'get_points': get_points,
			  'get_dtype': (lambda : source.get_dtype()[src_meas]),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def feature_reducer(source, src_meas, axis_mean, bg, feature):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	new_feature_shape = [1]*len(source.get_points()[src_meas])
	new_feature_shape[axis_mean] = len(feature)
	bg  = np.reshape(bg, new_feature_shape)
	feature = np.reshape(feature, new_feature_shape)
	filter = {'filter': lambda x:np.sum((x[src_meas]-bg)*feature, axis=axis_mean),
			  'get_points': get_points,
			  'get_dtype': (lambda : source.get_dtype()[src_meas]),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter