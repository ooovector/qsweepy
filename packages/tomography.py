from . import data_reduce
import numpy as np
from . import readout_classifier
#import cvxopt
#import cvxpy

class tomography:
	def __init__(self, sz_measurer, pulse_generator, proj_seq, reconstruction_basis={}):
		self.sz_measurer = sz_measurer
		#self.adc = adc
		self.pulse_generator = pulse_generator
		self.proj_seq = proj_seq
		self.reconstruction_basis=reconstruction_basis
		
		self.adc_reducer = data_reduce.data_reduce(self.sz_measurer.adc)
		self.adc_reducer.filters['SZ'] = {k:v for k,v in self.sz_measurer.filter_binary.items()}
		self.adc_reducer.filters['SZ']['filter'] = lambda x: 1-2*self.sz_measurer.filter_binary_func(x)
		
	def get_points(self):
		points = { p:{} for p in self.proj_seq.keys() }
		points.update({p:{} for p in self.reconstruction_basis.keys()})
		return points
	
	def get_dtype(self):
		dtypes = { p:float for p in self.proj_seq.keys() }
		dtypes.update({ p:float for p in self.reconstruction_basis.keys() })
		return dtypes
	
	def set_prepare_seq(self, seq):
		self.prepare_seq = seq
	
	def measure(self):
		meas = {}
		for p in self.proj_seq.keys():
			self.pulse_generator.set_seq(self.prepare_seq+self.proj_seq[p]['pulses'])
			meas[p] = np.real(np.mean(self.adc_reducer.measure()['SZ'])/2)

		proj_names = self.proj_seq.keys()
		basis_axes_names = self.reconstruction_basis.keys()
		#TODO: fix this norm stuff in accordance with theory
		basis_vector_norms = np.asarray([np.linalg.norm(self.reconstruction_basis[r]['operator']) for r in basis_axes_names])
		
		if len(self.reconstruction_basis.keys()):
			reconstruction_matrix = np.real(np.asarray([[np.sum(self.proj_seq[p]['operator']*np.conj(self.reconstruction_basis[r]['operator'])) \
										for r in basis_axes_names] \
										for p in proj_names]))
			projections = np.linalg.lstsq(reconstruction_matrix, [meas[p] for p in proj_names])[0]*(basis_vector_norms**2)
			meas.update({k:v for k,v in zip(basis_axes_names, projections)})
		return meas
		
	def get_opts(self):
		opts = { p:{} for p in self.proj_seq.keys()}
		opts.update ({ p:{} for p in self.reconstruction_basis.keys()})
		return opts
		
		