import numpy as np
import os, warnings
import time

module_dir = os.path.dirname(os.path.abspath(__file__))

class TSW14J56_evm_reducer():
	def __init__(self, adc):
		self.adc = adc

		self.averaging = False #If True measure() returns averaged samples of raw samples otherwise
		#Typed of data that the measure() method should return
		self.samples = True
		self.all_cov = True
		self.last_cov = True
		self.avg_cov = True
		self.resultnumber = True
		self.dot_prods = False #If True measure() returns all dot products for each discrimination channel,
		# where dot products are calculated using adc data and feature value for each channel

		self.trig = 'ext'
		self.avg_cov_mode = 'real' ## raw results from device
		self.cov_norms = {channel_id:1 for channel_id in range(4)}
		self.cov_signals = {channel_id:None for channel_id in range(4)}
		self.resultnumbers_dimension = 16
		self.devtype = 'SK'
		self.result_source = 'avg_cov'
		self.internal_avg = True

		self._numchan = self.adc.get_num_descr_ch()
		self._nsamp_max = self.adc.get_max_pp_nsamp()

		self.save_dot_prods = False # save dot products in a file

	def set_internal_avg(self, internal_avg):
		pass

	def get_clock(self):
		return 1e9

	def get_adc_nums(self):
		return self.adc.nsegm
	def get_adc_nop(self):
		return self.adc.nsamp
	def set_adc_nums(self, nums):
		self.adc.nsegm = nums
	def set_adc_nop(self, nop):
		self.adc.nsamp = nop

	def get_points(self):
		points = {}
		if self.samples:
			if self.averaging:
				points.update({'Voltage': [('Time', np.arange(self.adc.nsamp) / self.get_clock(), 's')]})
			else:
				points.update({'Voltage': [('Sample', np.arange(self.adc.nsegm), ''),
										   ('Time', np.arange(self.adc.nsamp) / self.get_clock(), 's')]})

		if self.all_cov:
			points.update({'all_cov'+str(i):[('Sample', np.arange(self.adc.nsegm), '')] for i in range(self._numchan)})
		if self.last_cov:
			points.update({'last_cov'+str(i):[] for i in range(self._numchan)})
		if self.avg_cov:
			if self.avg_cov_mode == 'real':
				points.update({'avg_cov'+str(i):[] for i in range(self._numchan)})
			elif self.avg_cov_mode == 'iq':
				points.update({'avg_cov'+str(i):[] for i in range(self._numchan//2)})
		if self.resultnumber:
			points.update({'resultnumbers':[('State', np.arange(self.resultnumbers_dimension), '')]})
		if self.dot_prods:
			points.update({'disc_ch'+str(i):[('Sample', np.arange(self.adc.nsegm), '')] for i in range(self._numchan)})
		return (points)

	def get_dtype(self):
		dtypes = {}
		if self.samples:
			dtypes.update({'Voltage':complex})
		if self.all_cov:
			# dtypes.update({'all_cov'+str(i):float for i in range(self._numchan)})
			dtypes.update({'all_cov' + str(i): complex for i in range(self._numchan)})
		if self.last_cov:
			dtypes.update({'last_cov'+str(i):float for i in range(self._numchan)})
		if self.avg_cov:
			if self.avg_cov_mode == 'real':
				dtypes.update({'avg_cov'+str(i):float for i in range(self._numchan)})
			elif self.avg_cov_mode == 'iq':
				dtypes.update({'avg_cov'+str(i):complex for i in range(self._numchan//2)})
		if self.resultnumber:
			dtypes.update({'resultnumbers': float})
		if self.dot_prods:
			dtypes.update({'disc_ch' + str(i): float for i in range(self._numchan)})
		return (dtypes)

	def get_opts(self):
		opts = {}
		if self.samples:
			opts.update({'Voltage':{'log': None}})
		if self.all_cov:
			opts.update({'all_cov'+str(i):{'log': None} for i in range(self._numchan)})
		if self.last_cov:
			opts.update({'last_cov'+str(i):{'log': None} for i in range(self._numchan)})
		if self.avg_cov:
			if self.avg_cov_mode == 'real':
				opts.update({'avg_cov'+str(i):{'log': None} for i in range(self._numchan)})
			elif self.avg_cov_mode == 'iq':
				opts.update({'avg_cov'+str(i):{'log': None} for i in range(self._numchan//2)})
		if self.resultnumber:
			opts.update({'resultnumbers': {'log': None}})
		if self.dot_prods:
			opts.update({'disc_ch'+str(i):{'log': None} for i in range(self._numchan)})
		print (opts)
		return (opts)

	def measure(self):
		time.sleep(0.01)
		result = {}
		self.adc.trig_mode = self.trig

		if not self.adc.start_wait_done():
			warnings.warn("Capture failed! Especially for Lena.")
		dot_prods = self.adc.get_dot_prods()

		if self.save_dot_prods:
			print("DOT PRODUCTS", dot_prods)
			import datetime
			import pickle as pkl
			now = datetime.datetime.now()
			day_folder_name = now.strftime('%Y-%m-%d')
			time_folder_name = now.strftime('%H-%M-%S')

			abs_path = "D:\\data\\"+day_folder_name+'\\'
			abs_path  = "C:\\data\\"
			data_for_saving = dot_prods
			np.savez_compressed(abs_path + 'meas-'+time_folder_name, d=data_for_saving)
			print(abs_path + 'meas-'+time_folder_name)

		if self.samples:
			if self.averaging:
				result.update( {'Voltage': self.adc.get_averaged_data()} )
			else:
				result.update( {'Voltage': self.adc.get_data()} )

		if self.all_cov:
			result_raw = {'all_cov' + str(i): dot_prods[:, i] / self.cov_norms[i] for i in range(self._numchan)}
			if self.avg_cov_mode == 'real':
				result.update(result_raw)
			# else:
			# 	result.update({'avg_cov0': (result_raw['avg_cov0']+1j*result_raw['avg_cov1']),
			# 				   'avg_cov1': (result_raw['avg_cov2']+1j*result_raw['avg_cov3'])})

			else:
				result.update({'all_cov0': (result_raw['all_cov0'] + 1j * result_raw['all_cov1']),
						   'all_cov1': (result_raw['all_cov2'] + 1j * result_raw['all_cov3'])})
		if self.last_cov:
			result.update({'last_cov'+str(i):dot_prods[-1][i]/self.cov_norms[i] for i in range(self._numchan)})
		if self.avg_cov:
			dot_prods_ave = self.adc.get_dot_prod_ave()
			result_raw = {'avg_cov'+str(i):dot_prods_ave[i]/self.cov_norms[i] for i in range(self._numchan)}
			if self.avg_cov_mode == 'real':
				result.update(result_raw)
			elif self.avg_cov_mode == 'iq':
				result.update({'avg_cov0': (result_raw['avg_cov0']+1j*result_raw['avg_cov1']),
							   'avg_cov1': (result_raw['avg_cov2']+1j*result_raw['avg_cov3'])})
		if self.resultnumber:
			resultnumbers = np.zeros(self.resultnumbers_dimension)


			states = np.asarray(dot_prods > self.adc.threshold, dtype=int)
			# print(self.adc.threshold)
			# print(states.shape)
			print('DIM', dot_prods.shape)

			for s in states:
				s_int = 0
				for i in range(self._numchan):
					s_int |= s[i]<<i
				resultnumbers[s_int] += 1
			result.update({'resultnumbers':resultnumbers})

		if self.dot_prods:
			result.update({'disc_ch'+str(i): dot_prods[:, i].ravel() for i in range(self._numchan)})

		return (result)

	def set_feature_iq(self, feature_id, feature):
		self.adc.set_feature( feature, feature_id*2)
		self.adc.set_feature( feature.imag-1.j*feature.real, feature_id*2+1)

		feature = feature[:int(self._nsamp_max)] / np.max(np.abs(feature[:int(self._nsamp_max)]))
		feature = np.asarray(2 ** 13 * feature, dtype=complex)

		self.cov_norms[feature_id*2] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
		self.cov_norms[feature_id*2+1] = np.sqrt(np.mean(np.abs(feature)**2))*2**13

	def set_feature_real(self, feature_id, feature, threshold=None):
		if threshold is not None:
			threshold = threshold/np.max(np.abs(feature[:self._nsamp_max]))*(2**13)
			self.adc.set_threshold(threshold, feature_id)

		self.adc.set_feature(feature, feature_id)

		feature = feature[:self._nsamp_max] / np.max(np.abs(feature[:self._nsamp_max]))
		feature = np.asarray(2 ** 13 * feature, dtype=complex)

		self.cov_norms[feature_id] = np.sqrt(np.mean(np.abs(feature)**2))*2**13


	def disable_feature(self, feature_id):
		self.adc.set_feature( np.zeros(self._nsamp_max), feature_id)
		self.adc.set_threshold(1, feature_id)