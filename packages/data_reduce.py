# уменьшает количество данных от оцифровщика, чтобы не было MemoryError всякого
# подлежит применению во всяких свипах когда нет сил все эти гиги хранить.

import numpy as np
import logging
import threading

class data_reduce:
	def __init__(self, source, thread_limit=1):
		self.source = source
		self.filters = {}
		self.extra_opts = {}
		self.threads = []
		self.thread_limiter = threading.Semaphore(thread_limit)
		if hasattr(self.source, 'pre_sweep'):
			self.pre_sweep = self.source.pre_sweep
		if hasattr(self.source, 'post_sweep'):
			self.post_sweep = self.source.post_sweep
		
	def get_points(self):
		return { filter_name:filter['get_points']() for filter_name, filter in self.filters.items()}
	
	def get_dtype(self):
		return { filter_name:filter['get_dtype']() for filter_name, filter in self.filters.items()}
	
	def get_opts(self):
		return { filter_name:{**filter['get_opts'](), **self.extra_opts} for filter_name, filter in self.filters.items()}
		
	def measure(self):
		data = self.source.measure()
		result = { filter_name:filter['filter'](data) for filter_name, filter in self.filters.items()}
		del data
		return result
		
	def postprocess_thread_func(self, data, callback, args):
		#print ('Spawned deferred postprocessing thread with args: ', args)
		try:
			self.postprocess(data, callback, args)
			#print ('Callback finished with args: ', args)
			#print ('Worker thread list: \n', self.threads)
			#print ('Current thread: ', threading.current_thread())
		except Exception as e:	# I wouldn't recommend this, but you asked for it
			print ('Postprocessing exception occured with args: ', args)
			self.termination_cause = e	# If an Exception occurred, it will be here
			raise
		finally:
			self.threads.remove(threading.current_thread())
			self.thread_limiter.release()
			
	def postprocess(self, data, callback, args):
		result = { filter_name:filter['filter'](data) for filter_name, filter in self.filters.items()}
		#print ('Finished postprocessing with args: ', args)
		del data
		callback(result, *args)
		
	def measure_deferred_result(self, callback, args):
		if hasattr(self.source, 'measure_deferred_result'): # if underlying device supports deferred results, call it
			self.source.measure_deferred_result(self.postprocess, args=(callback, args)) 
			return
		data = self.source.measure()
		# first, traverse all threads and make sure they didn't exit with an exception, otherwise rethrow the expection in the main thread
		for t in self.threads:
			if hasattr(t, 'termination_cause'):
				print ('Postprocessing exception detected with args, joining all deferred postprocessing: ', args)
				self.join_deferred() # wait for all other threads to terminate
				print ('Reraising')
				raise(t.termination_cause)

		t= threading.Thread(target=self.postprocess_thread_func, args=(data, callback, args))
		self.thread_limiter.acquire()
		self.threads.append(t)
		t.start()
	
	def join_deferred(self):
		for t in self.threads:
			t.join()
		
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
		
def thru(source, src_meas, diff=0, scale=1):
	filter = {'filter': lambda x:x[src_meas]/scale-diff,
			  'get_points': lambda : source.get_points()[src_meas],
			  'get_dtype': (lambda : source.get_dtype()[src_meas]),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter

def cross_section_reducer(source, src_meas, axis, index):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes[axis]
		return new_axes

	cross_section_index = [slice(None) if i != axis else index for i in range(len(source.get_points()[src_meas]))]

	filter = {'filter': lambda x: np.asarray(x[src_meas])[cross_section_index],
			  'get_points': get_points,
			  'get_dtype': (lambda: complex if source.get_dtype()[src_meas] is complex else float),
			  'get_opts': (lambda: source.get_opts()[src_meas])}
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

def std_reducer_noavg(source, src_meas, axis, noavg_axis):
	def get_points():
		new_axes = source.get_points()[src_meas].copy()
		del new_axes [axis]
		return new_axes
	def filter_func(x):
		avg_dim = [len(a[1]) for a in source.get_points()[src_meas].copy()]
		avg_dim[noavg_axis] = 1
		return np.std(x[src_meas]-np.reshape(np.mean(x[src_meas], axis=noavg_axis), avg_dim), axis=axis)
	filter = {'filter': filter_func,
			  'get_points': get_points,
			  'get_dtype': (lambda : float),
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
	bg	= np.reshape(bg, new_feature_shape)
	feature = np.reshape(feature, new_feature_shape)
	def filter_func(x):
		feature_truncated_shape = tuple([slice(None) if i != axis_mean else slice(x[src_meas].shape[axis_mean]) for i in range(len(new_feature_shape))])
		feature_truncated = feature[feature_truncated_shape]
		bg_truncated = bg[feature_truncated_shape]
		#print (x[src_meas].shape, axis_mean, feature_truncated_shape, feature.shape, feature_truncated.shape)
		return np.sum((x[src_meas]-bg_truncated)*feature_truncated, axis=axis_mean)
	filter = {'filter': filter_func,
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
	bg	= np.reshape(bg, new_feature_shape)
	feature = np.reshape(feature, new_feature_shape)
	filter = {'filter': lambda x:(np.sum((x[src_meas]-bg)*feature, axis=axis_mean)>0)*2-1,
			  'get_points': get_points,
			  'get_dtype': (lambda : int),
			  'get_opts': (lambda : source.get_opts()[src_meas])}
	return filter
	
def hist_filter(source, *src_meas_values):
	def filter_func(x):
		#print (x)
		return np.mean(np.prod([x[m]==v for m, v in src_meas_values], axis=0))
	filter = {'filter': filter_func,
			  'get_points': [],
			  'get_dtype': lambda : float,
			  'get_opts': lambda : source.get_opts()[src_meas[0]]}
	return filter