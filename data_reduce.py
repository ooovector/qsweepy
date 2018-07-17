# уменьшает количество данных от оцифровщика, чтобы не было MemoryError всякого
# подлежит применению во всяких свипах когда нет сил все эти гиги хранить.

import numpy as np
import logging

class data_reduce:
	def __init__(self, source):
		self.source = source
		self.filters = {}
		self.extra_opts = {}
		if hasattr(self.source, 'pre_sweep'):
			self.pre_sweep = self.source.pre_sweep
		if hasattr(self.source, 'post_sweep'):
			self.post_sweep = self.source.post_sweep
		
	def get_points(self):
		return { filter_name:filter['get_points']() for filter_name, filter in self.filters.items()}
	
	def get_dtype(self):
		return { filter_name:filter['get_dtype']() for filter_name, filter in self.filters.items()}
		
	def measure(self):
		data = self.source.measure()
		return { filter_name:filter['filter'](data) for filter_name, filter in self.filters.items()}
		
	def get_opts(self):
		return { filter_name:{**filter['get_opts'](), **self.extra_opts} for filter_name, filter in self.filters.items()}
		
def downsample_reducer(source, src_meas, axis, carrier, downsample, iq=True, iq_axis=-1):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		new_axes [axis] = [i for i in new_axes [axis]]
		new_axes [axis][1] = [i for i in new_axes[axis][1]][::downsample]
		if iq:
			new_axes [iq_axis][1] = np.asarray([j for i in zip(new_axes [iq_axis][1], new_axes [iq_axis][1]) for j in i ])
		return new_axes
	intermediate_axes = [len(a[1]) for a in source.get_points()[src_meas][:axis]]+[len(source.get_points()[src_meas][axis][1][::downsample]), downsample]+[len(a[1]) for a in source.get_points()[src_meas][axis+1:]]
	filter_func = lambda x,s:np.mean(np.reshape(np.exp(s*2*np.pi*1j*source.get_points()[src_meas][axis][1]*carrier)*x[src_meas], intermediate_axes), axis=axis+1)
	
	filter = {'filter': lambda x:filter_func(x,1) if not iq else np.concatenate([filter_func(x,1), filter_func(x,-1)], axis=iq_axis),
			  'get_points': get_points,
			  'get_dtype': (lambda : complex if source.get_dtype()[src_meas] is complex else float),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
		
def mean_reducer(source, src_meas, axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	filter = {'filter': lambda x:np.mean(x[src_meas], axis=axis),
			  'get_points': get_points,
			  'get_dtype': (lambda : complex if source.get_dtype()[src_meas] is complex else float),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def std_reducer(source, src_meas, axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	filter = {'filter': lambda x:np.std(x[src_meas], axis=axis),
			  'get_points': get_points,
			  'get_dtype': (lambda : complex if source.get_dtype()[src_meas] is complex else float),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def mean_reducer_noavg(source, src_meas, axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	filter = {'filter': lambda x:np.mean(x[src_meas], axis=axis)-np.mean(x[src_meas]),
			  'get_points': get_points,
			  'get_dtype': (lambda : complex if source.get_dtype()[src_meas] is complex else float),
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
		del new_axes [axis_mean]
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
	
def feature_reducer_binary(source, src_meas, axis_mean, bg, feature):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis_mean]
		return new_axes
	new_feature_shape = [1]*len(source.get_points()[src_meas])
	new_feature_shape[axis_mean] = len(feature)
	bg  = np.reshape(bg, new_feature_shape)
	feature = np.reshape(feature, new_feature_shape)
	filter = {'filter': lambda x:(np.sum((x[src_meas]-bg)*feature, axis=axis_mean)>0)*2-1,
			  'get_points': get_points,
			  'get_dtype': (lambda : int),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter