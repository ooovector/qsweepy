from . import sweep
from . import save_pkl
from . import fitting
import numpy as np
import threading
import traceback
import sys

import matplotlib.pyplot as plt

class diff_readout:
	def __init__(self, source):
		self.source = source
		self.filters = {}
		self.extra_opts = {}
		if hasattr(self.source, 'pre_sweep'):
			self.pre_sweep = self.source.pre_sweep
		if hasattr(self.source, 'post_sweep'):
			self.post_sweep = self.source.post_sweep
		self.diff_setter = lambda: []
		self.thread_limiter = threading.Semaphore(1)
	
	def get_points(self):
		return self.source.get_points()
	
	def get_dtype(self):
		return self.source.get_dtype()
		
	def get_opts(self):
		return self.source.get_opts()

	def measure(self):
		self.zero_setter()
		data_zero = self.source.measure()
		self.diff_setter()
		data_one = self.source.measure()
		result = {mname: data_one[mname]-data_zero[mname] for mname in data_one.keys()}
		del data_zero, data_one
		return result
		
	def postprocess_zero(self, data, callback, args):
		self.data_zero = data
		self.zero_ready = True
		#print ('Postprocess_zero with args: ', args)
		#traceback.print_stack(file=sys.stdout)
		if self.zero_ready and self.one_ready:
			self.postprocess(callback, args)
			
	def postprocess_one(self, data, callback, args):
		self.data_one = data
		self.one_ready = True
		#print ('Postprocess_one with args: ', args)
		#traceback.print_stack(file=sys.stdout)
		if self.zero_ready and self.one_ready:
			self.postprocess(callback, args)
	
	def postprocess(self, callback, args):
		try:
			result = {mname: self.data_one[mname]-self.data_zero[mname] for mname in self.data_one.keys()}
			#print ('Finished postprocessing with args: ', args)
			callback(result, *args)
			#print ('Callback finished with args: ', args)
			self.thread_limiter.release()
		except:
			self.thread_limiter.release()
			raise
		
	def measure_deferred_result(self, callback, args):
		self.thread_limiter.acquire()
		if hasattr(self.source, 'measure_deferred_result'): # if underlying device supports deferred results, call it
			self.zero_ready = False
			self.one_ready = False
			#print ('Started measuring ', args)
			self.zero_setter()
			self.source.measure_deferred_result(self.postprocess_zero, args=(callback, args)) 
			self.diff_setter()
			self.source.measure_deferred_result(self.postprocess_one, args=(callback, args))
		else:
			# otherwise do simple measurement
			callback(self.measure(), *args)
		