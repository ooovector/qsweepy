from . import sweep
from . import save_pkl
from . import fitting
import numpy as np

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
	
	def get_points(self):
		return self.source.get_points()
	
	def get_dtype(self):
		return self.source.get_dtype()
		
	def measure(self):
		self.zero_setter()
		data_zero = self.source.measure()
		self.diff_setter()
		data_one = self.source.measure()
		result = {mname: data_one[mname]-data_zero[mname] for mname in data_one.keys()}
		del data_zero, data_one
		return result
		
	def get_opts(self):
		return self.source.get_opts()
		