import numpy as np
import os, warnings

"""Debug begin"""
import pickle
from contextlib import redirect_stdout
import io
import datetime
"""Debug end"""

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
		self.resultnumber_states = False
		self.dot_prods = False  # If True measure() returns all dot products for each discrimination channel,
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

		self.debug_ch_to_monitor = [False]*self._numchan
		self.debug_dot_prod_ave_prev = np.array(())
		self.debug_enable = False
		self.debug_start = False
		self.debug_delta_rel = 100


		self.save_samples = False


		self.model = None
		self.num_states = 3

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
		if self.resultnumber_states:
			points.update({'resultnumbers_states':[('State', np.arange(self.num_states), '')]})

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
		if self.resultnumber_states:
			dtypes.update({'resultnumbers_states': float})
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
		if self.resultnumber_states:
			opts.update({'resultnumbers_states': {'log': None}})
		if self.dot_prods:
			opts.update({'disc_ch'+str(i):{'log': None} for i in range(self._numchan)})
		return (opts)


	def get_data(self):
		return self.adc.get_data()


	def measure(self):
		result = {}
		self.adc.trig_mode = self.trig

		if not self.adc.start_wait_done():
			warnings.warn("Capture failed! Especially for Lena.")
		dot_prods = self.adc.get_dot_prods()

		# ### ACHTUNG NACHALO
		# print('ОСТАНОВИСЬ СОХРАНЯЮТСЯ ВООБЩЕ ВСЕ ДАННЫЕ')
		# data = self.adc.get_data()
		# print(data.shape)
		#
		# import datetime
		# now = datetime.datetime.now()
		# day_folder_name = now.strftime('%Y-%m-%d')
		# time_folder_name = now.strftime('%H-%M-%S')
		# abs_path = "C:\\data\\"
		# np.savez_compressed(abs_path + day_folder_name + '_' + time_folder_name, data=data)
		# ### ACHTUNG KONETZ

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
			for s in states:
				s_int = 0
				for i in range(self._numchan):
					s_int |= s[i]<<i
				resultnumbers[s_int] += 1
			result.update({'resultnumbers':resultnumbers})

		if self.dot_prods:
			result.update({'disc_ch'+str(i): dot_prods[:, i].ravel() for i in range(self._numchan)})

		if self.resultnumber_states:
			data = self.adc.get_data()
			# print("Model is", self.model)
			# print("Trajectories shape is ", data.shape)
			w_ = self.model.get_w_(data)
			y_hat = self.model.predict(w_)
			# print(y_hat)
			states = np.asarray([np.count_nonzero(y_hat == _class) for _class in range(self.num_states)])
			print(states)
			result.update({'resultnumbers_states': states})

		"""Debug begin"""
		# if self.debug_enable:
		# 	dot_prod_ave = np.mean(dot_prods, axis=0)
		# 	if not self.debug_start:
		# 		for ch,status in enumerate(self.debug_ch_to_monitor):
		# 			if status:
		# 				try:
		# 					if np.abs( ( dot_prod_ave[ch] - self.debug_dot_prod_ave_prev[ch] ) /
		# 						  	 self.debug_dot_prod_ave_prev[ch] ) > self.debug_delta_rel:
		# 						self.dump_device_content()
		# 				except ValueError:
		# 					print("ValueError", dot_prod_ave, self.debug_dot_prod_ave_prev)
		# 	self.debug_dot_prod_ave_prev = dot_prod_ave
		# 	self.debug_start = False
		"""Debug end"""
		return (result)

	"""Debug dump"""
	def dump_device_content(self):
		path = module_dir+'\\adc_dump'
		samples = self.adc.get_data()
		samples_ave = self.adc.get_averaged_data()
		features = self.adc.read_onchip_memory(0,8192*2*4)
		dot_prods = self.adc.get_dot_prods()
		s_out = io.StringIO()
		with redirect_stdout(s_out):
			self.adc.print_pp_regs()
		pp_reg_print = s_out.getvalue()

		obj = {"samples": samples,
			   "samples_ave":samples_ave,
			   "features":features,
			   "dot_prods":dot_prods,
			   "thresholds":self.adc.threshold,
			   "active_channels":self.debug_ch_to_monitor,
			   "dot_prod_ave_prev":self.debug_dot_prod_ave_prev,
			   "pp_reg_print":pp_reg_print}

		timestamp = datetime.datetime.now().isoformat().replace(":","_")
		with open(path+"\\"+timestamp+".pkl", 'wb') as f:
			pickle.dump(obj,f)
		del obj
		del samples

	def set_feature_iq(self, feature_id, feature):
		self.adc.set_feature( feature, feature_id*2)
		self.adc.set_feature( feature.imag-1.j*feature.real, feature_id*2+1)

		feature = feature[:int(self._nsamp_max)] / np.max(np.abs(feature[:int(self._nsamp_max)]))
		feature = np.asarray(2 ** 13 * feature, dtype=complex)

		self.cov_norms[feature_id*2] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
		self.cov_norms[feature_id*2+1] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
		"""Debug begin"""
		self.debug_enable = False
		"""Debug end"""

	def set_feature_real(self, feature_id, feature, threshold=None):
		if threshold is not None:
			threshold = threshold/np.max(np.abs(feature[:self._nsamp_max]))*(2**13)
			self.adc.set_threshold(threshold, feature_id)

		self.adc.set_feature(feature, feature_id)

		feature = feature[:self._nsamp_max] / np.max(np.abs(feature[:self._nsamp_max]))
		feature = np.asarray(2 ** 13 * feature, dtype=complex)

		self.cov_norms[feature_id] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
		"""Debug begin"""
		self.debug_ch_to_monitor[feature_id] = True
		self.debug_start = True
		self.debug_enable = True
		"""Debug end"""

	def disable_feature(self, feature_id):
		self.adc.set_feature( np.zeros(self._nsamp_max), feature_id)
		self.adc.set_threshold(1, feature_id)
		"""Debug begin"""
		self.debug_ch_to_monitor[feature_id] = False
		for status in self.debug_ch_to_monitor:
			if status: break
			else: self.debug_enable = False
		"""Debug end"""

	def set_model(self, model):
		#TODO: applicable only for one descriptor, how to make in better?
		self.model = model

# OLD ADC
# from numpy import *
# #from usb_intf import *
# #from reg_intf import *
#
# import warnings
# #from qsweepy.instrument import Instrument
# #from qsweepy.instrument_drivers.ADS54J40 import *
# from qsweepy.instrument_drivers._ADS54J40.usb_intf import *
# from qsweepy.instrument_drivers._ADS54J40.reg_intf import *
# from qsweepy.instrument_drivers.ADS54J40 import *
#
# import usb.core
# import time
# import sys
# import zlib
#
# from qsweepy.libraries import config
#
# #sys.path.append('C:\qtlab_replacement\qsweepy\instrument_drivers\_ADS54J40')
#
# class TSW14J56_evm_reducer():
# 	def __init__(self, adc):
# 		self.adc = adc
# 		self.output_raw = True
# 		self.last_cov = True
# 		self.avg_cov = True
# 		self.resultnumber = True
# 		self.trig = 'ext'
# 		self.avg_cov_mode = 'real' ## raw results from device
# 		self.cov_norms = {channel_id:1 for channel_id in range(4)}
# 		self.cov_signals = {channel_id:None for channel_id in range(4)}
# 		self.resultnumbers_dimension = 16
# 		self.devtype = 'SK'
# 		self.result_source = 'avg_cov'
# 		self.internal_avg = True
# 		#self.avg_cov_mode = 'norm_cmplx' ## normalized results in complex Volts, IQ
#
# 	def set_internal_avg(self, internal_avg):
# 		pass
#
# 	def get_clock(self):
# 		return self.adc.get_clock()
#
# 	def get_adc_nums(self):
# 		return self.adc.nsegm
# 	def get_adc_nop(self):
# 		return self.adc.nsamp
# 	def set_adc_nums(self, nums):
# 		self.adc.nsegm = nums
# 	def set_adc_nop(self, nop):
# 		self.adc.nsamp = nop
#
# 	def get_points(self):
# 		points = {}
# 		if self.output_raw:
# 			points.update({'Voltage':[('Sample',arange(self.adc.nsegm), ''),
# 								 ('Time',arange(self.adc.nsamp)/self.adc.get_clock(), 's')]})
# 		if self.last_cov:
# 			points.update({'last_cov'+str(i):[] for i in range(self.adc.num_covariances)})
# 		if self.avg_cov:
# 			if self.avg_cov_mode == 'real':
# 				points.update({'avg_cov'+str(i):[] for i in range(self.adc.num_covariances)})
# 			elif self.avg_cov_mode == 'iq':
# 				points.update({'avg_cov'+str(i):[] for i in range(self.adc.num_covariances//2)})
# 		if self.resultnumber:
# 			points.update({'resultnumbers':[('State', arange(self.resultnumbers_dimension), '')]})
# 		return (points)
#
# 	def get_dtype(self):
# 		dtypes = {}
# 		if self.output_raw:
# 			dtypes.update({'Voltage':complex})
# 		if self.last_cov:
# 			dtypes.update({'last_cov'+str(i):float for i in range(self.adc.num_covariances)})
# 		if self.avg_cov:
# 			if self.avg_cov_mode == 'real':
# 				dtypes.update({'avg_cov'+str(i):float for i in range(self.adc.num_covariances)})
# 			elif self.avg_cov_mode == 'iq':
# 				dtypes.update({'avg_cov'+str(i):complex for i in range(self.adc.num_covariances//2)})
# 		if self.resultnumber:
# 			dtypes.update({'resultnumbers': float})
# 		return (dtypes)
#
# 	def get_opts(self):
# 		opts = {}
# 		if self.output_raw:
# 			opts.update({'Voltage':{'log': None}})
# 		if self.last_cov:
# 			opts.update({'last_cov'+str(i):{'log': None} for i in range(self.adc.num_covariances)})
# 		if self.avg_cov:
# 			if self.avg_cov_mode == 'real':
# 				opts.update({'avg_cov'+str(i):{'log': None} for i in range(self.adc.num_covariances)})
# 			elif self.avg_cov_mode == 'iq':
# 				opts.update({'avg_cov'+str(i):{'log': None} for i in range(self.adc.num_covariances//2)})
# 		if self.resultnumber:
# 			opts.update({'resultnumbers': {'log': None}})
# 		return (opts)
#
# 	def measure(self):
# 		result = {}
# 		if self.avg_cov:
# 			avg_before =  {'avg_cov'+str(i):self.adc.get_cov_result_avg(i) for i in range(self.adc.num_covariances)}
# 		if self.resultnumber:
# 			resultnumbers_before = self.adc.get_resultnumbers()
# 		self.adc.capture(trig=self.trig, cov = (self.last_cov or self.avg_cov or self.resultnumber))
# 		if self.output_raw:
# 			result.update({'Voltage':self.adc.get_data()})
# 		if self.last_cov:
# 			result.update({'last_cov'+str(i):self.adc.get_cov_result(i)/self.cov_norms[i] for i in range(self.adc.num_covariances)})
# 		if self.avg_cov:
# 			result_raw = {'avg_cov'+str(i):(self.adc.get_cov_result_avg(i)-avg_before['avg_cov'+str(i)])/self.cov_norms[i] for i in range(self.adc.num_covariances)}
# 			if self.avg_cov_mode == 'real':
# 				result.update(result_raw)
# 			elif self.avg_cov_mode == 'iq':
# 				result.update({'avg_cov0': (result_raw['avg_cov0']+1j*result_raw['avg_cov1']),
# 							   'avg_cov1': (result_raw['avg_cov2']+1j*result_raw['avg_cov3'])})
# 		if self.resultnumber:
# 			result.update({'resultnumbers': [a-b for a,b in zip(self.adc.get_resultnumbers(), resultnumbers_before)][:self.resultnumbers_dimension]})
#
# 		return (result)
#
# 	def set_feature_iq(self, feature_id, feature):
# 		#self.avg_cov_mode = 'norm_cmplx'
# 		feature = feature[:self.adc.ram_size]/np.max(np.abs(feature[:self.adc.ram_size]))
# 		feature = np.asarray(2**13*feature, dtype=complex)
# 		feature_real_int = np.asarray(np.real(feature), dtype=np.int16)
# 		feature_imag_int = np.asarray(np.imag(feature), dtype=np.int16)
#
# 		self.adc.set_ram_data([feature_real_int.tolist(),     (feature_imag_int).tolist()],  feature_id*2)
# 		self.adc.set_ram_data([(feature_imag_int).tolist(),  (-feature_real_int).tolist()], feature_id*2+1)
#
# 		self.cov_norms[feature_id*2] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
# 		self.cov_norms[feature_id*2+1] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
#
# 	def set_feature_real(self, feature_id, feature, threshold=None):
# 		#self.avg_cov_mode = 'norm_cmplx'
# 		if threshold is not None:
# 			threshold = threshold/np.max(np.abs(feature[:self.adc.ram_size]))*(2**13)
# 			self.adc.set_threshold(thresh=threshold, ncov=feature_id)
#
# 		feature = feature[:self.adc.ram_size]/np.max(np.abs(feature[:self.adc.ram_size]))
# 		feature_padded = np.zeros(self.adc.ram_size, dtype=np.complex)
# 		feature_padded[:len(feature)] = feature
# 		feature = np.asarray(2**13*feature_padded, dtype=complex)
# 		feature_real_int = np.asarray(np.real(feature), dtype=np.int16)
# 		feature_imag_int = np.asarray(np.imag(feature), dtype=np.int16)
#
# 		self.adc.set_ram_data([feature_real_int.tolist(),    (feature_imag_int).tolist()],  feature_id)
# 		self.cov_norms[feature_id] = np.sqrt(np.mean(np.abs(feature)**2))*2**13
#
# 	def disable_feature(self, feature_id):
# 		self.adc.set_ram_data([np.zeros(self.adc.ram_size, dtype=np.int16).tolist(), np.zeros(self.adc.ram_size, dtype=np.int16).tolist()], feature_id)
# 		self.adc.set_threshold(thresh=1, ncov=feature_id)
#
# class TSW14J56_evm():
# 	def __init__(self, fpga_config = True):
# 		#Number of samples per channel
# 		self.nsamp = 65536
# 		self.nsegm = 1
# 		#Capture timeout
# 		self.timeout = 3
# 		self.ram_size = 2048 #in words of 32
# 		self.num_covariances = 4
#
# 		self.usb_reboot_timeout = 10
# 		self.debug_print = False
# 		# self.fpga_firmware = "_ADS54J40/qubit_daq.rbf"
# 		self.fpga_firmware = config.get_config()['TSW14J56_firmware']
# 		self.adc_reducer_hooks = []
# 		#self.cov_cnt = 0 ### TODO:what's this??? maybe delete it?
#
# 		#To do make a register readout to check ADS-programmed status"
# 		self.ads = ADS54J40()
# 		if self.ads.read_reg(ADS_CTRL_ST_ADDR) == ADS_CTRL_ST_VL:
# 			print ("ADS54J40 already programmed")
# 			self.ads.device.close()
# 		else:
# 			print ("Programming LMK clocking IC")
# 			self.ads.load_lmk_config()
# 			print("Done!")
# 			time.sleep(5)
# 			print("Programming ADC ")
# 			self.ads.load_ads_config()
# 			self.ads.device.close()
# 			print("Done!")
#
# 		self.dev=usb.core.find(idVendor=id.VENDOR, idProduct=id.PRODUCT)
# 		if self.dev is None:
# 			raise Exception('TSW14J56: Device not found!')
# 		self.dev.set_configuration(1)
# 		self.usb_reset()
# 		#Chech if FRGA already configured
# 		if fpga_config:
# 			dev_checksum = 0
# 			try:
# 				dev_checksum = self.read_reg(FX3_BASE, FX3_CRC32)
# 			except:
# 				pass
# 			try:
# 				firmware_rbf = open(self.fpga_firmware, 'rb')
# 				firmware = firmware_rbf.read()
# 			except:
# 				warnings.warn("TSW14J56: Unable to read FPGA firmware from file. The device may be unprogrammed!")
# 				return
# 			checksum = zlib.crc32(firmware)
# 			if dev_checksum != checksum:
# 				self.fpga_config(firmware = firmware)
#
# 		self.sync_req()
#
# 	def get_clock(self):
# 		###TODO: get this useless shit right
# 		return 1e9
#
# 	def usb_reset(self):
# 		#Reset USB chip and wait until it reboot.
# 		self.dev.reset()
# 		self.dev = None
# 		t0=time.time()
# 		while( time.time()-t0 < self.usb_reboot_timeout ):
# 			try:
# 				self.dev=usb.core.find(idVendor=id.VENDOR, idProduct=id.PRODUCT)
# 				if self.dev is None:
# 					time.sleep(0.05)
# 				else:
# 					self.dev.set_configuration(1)
# 					break
# 			except:
# 				self.dev = None
# 				pass
# 		if self.dev is None:
# 			raise Exception('Device not found')
#
# 	def sync_req(self):
# 		self.write_reg(JESD_BASE, JESD_CTRL, 1)
#
# 	def system_reset(self):
# 		self.write_reg(FX3_BASE, FX3_RST, 1)
# 		return
#
# 	def reset(self):
# 		self.system_reset()
# 		self.sync_req()
# 		self.usb_reset()
#
# 	def fpga_config(self, source = None, firmware = None):
# 		if source is None:
# 			source = self.fpga_firmware
# 		if firmware is None:
# 			firmware_rbf = open(source, 'rb')
# 			firmware = firmware_rbf.read()
# 		if len(firmware) > 0xFFFFFFFF:
# 			raise ValueError('TSW14J56: FPGA firmware size >0xFFFF bytes not supported!')
# 		self.usb_reset()
# 		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.FPGA_CONF_INIT, 0, 0, to_bytes(len(firmware)+2,4) )
# 		time.sleep(2)
# 		self.dev.write( endpoints.OUT, firmware+bytes([0,0]))
# 		res = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.FPGA_CONF_FIN, 0, 0, 2 )
# 		if res[0]==0:
# 			raise Exception('TSW14J56: FPGA configuration failed!')
# 		self.usb_reset()
# 		time.sleep(0.05)
# 		checksum = zlib.crc32(firmware)
# 		self.write_reg(0x10000, 20, checksum)
#
# 	def write_reg(self,base, offset, raw_data):
# 		#We just shift to avoid address modification by original FX3 chip firmware from Prakash
# 		#We will shift them back in FPGA
# 		address = (base + offset)<<2
# 		if type(raw_data) is list:
# 			data=raw_data
# 		else:
# 			data = to_bytes(raw_data, 4)
# 		Value, Index = mk_val_ind(address)
# 		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, data)
# 		if(self.debug_print):
# 			if type(raw_data) is not list:
# 				print ("Write:", hex(Value), hex(Index), hex(raw_data))
# 			else:
# 				print ("Write:", hex(Value), hex(Index), raw_data)
#
# 	def read_reg(self,base, offset):
# 		#We just shift to avoid address modification by original FX3 chip firmware from Prakash
# 		#We will shift them back in FPGA
# 		address = (base + offset)<<2
# 		Value, Index = mk_val_ind(address)
# 		data = frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ, Value,Index,4), dtype = uint32odd)[0]
# 		if(self.debug_print): print( "Read:", hex(Value), hex(Index), hex(data) )
# 		return data
#
# 	def capture(self, trig = "man", cov = False, fifo = True):
# 		'''
# 		Function starts data acquisition
#
# 		Input:
# 			trig: Way to trigger acquisition process: 'man' - after using the function, 'ext' - after external trigger occures
# 			cov:  Necessity to calculate covariance coefficient with data loaded to FPGA's OnChip Memory
# 			fifo: Necessity to write data to DDR (False used only to check that feedback loop works properly)
#
# 		Output:
# 			None
# 		'''
# 		start = time.time()
# 		self.write_reg(CAP_BASE, CAP_SEGM_NUM, int(self.nsegm))
# 		if (cov):
# 			self.write_reg(CAP_BASE, COV_LEN, int(self.nsamp/8))
# 			self.write_reg(CAP_BASE, CAP_LEN, int(self.nsamp/8))
# 			#self.cov_cnt = self.cov_cnt + 1
# 		else:
# 			self.write_reg(CAP_BASE, CAP_LEN, int(self.nsamp/8))
# 		if(trig == "man"):
# 			self.write_reg(CAP_BASE, CAP_CTRL, 1<<CAP_CTRL_START |fifo << FIFO_ST )
# 		elif(trig == "ext"):
# 			self.write_reg(CAP_BASE, CAP_CTRL, 1<<CAP_CTRL_START| 1<<CAP_CTRL_EXT_TRIG |cov << COV_ST |fifo << FIFO_ST)
# 		else: return
#
# 		t1 = time.time()
# 		while(1):
# 			if( not( self.read_reg(CAP_BASE, CAP_CTRL) & 1<<CAP_CTRL_BUSY ) ):
# 				break
# 			else:
# 				if(self.debug_print): print("Busy..")
#
# 			if(time.time()-t1>self.timeout):
# 				print ("Capture failed")
# 				self.write_reg(CAP_BASE, CAP_CTRL, 1 << CAP_CTRL_ABORT)
# 				break
# 		if(self.debug_print): print("Done!")
#
# 		stop = time.time()
# 		print('Time for capture', stop - start)
#
# 	def get_data(self):
# 		'''
# 		Transfer data from DDR
# 		'''
# 		#Number of samples per channel to read
# 		start = time.time()
# 		nsegm = self.nsegm
# 		if self.nsegm ==0:
# 			nsegm = 1
# 		data_len = self.nsamp*nsegm
# 		self.write_reg(FX3_BASE, FX3_LEN, int(data_len/16))
# 		#Trigger DDR read
# 		self.write_reg(FX3_BASE, FX3_CTRL, FX3_CTRL_START)
# 		data = frombuffer(self.dev.read(endpoints.IN, data_len*4), dtype = dtype(int16))
# 		data = reshape(data, (data_len, 2))
# 		res = reshape(data.T[1, :], (self.nsegm, self.nsamp)) + 1j * reshape(data.T[0, :], (self.nsegm, self.nsamp))
# 		stop = time.time()
# 		print('Time of data transfer', stop - start)
# 		return res
#
# 		#dataiq = reshape(data, (2, self.nsamp*self.nsegm))[0]+ 1j*reshape(data, (2, self.nsamp*self.nsegm))[1]
# 		#return (reshape(dataiq, (self.nsegm, self.nsamp)))
#
# 	def set_ram_data(self,data, ncov):
# 		'''
# 		Function transfer data to FPGA's OnChip Memory
#
# 		Input:
# 			data: TO DO write a way to load data
# 			ncov: Number of OnChip Memory to write the data (From 0 to 3)
# 		Output:
# 			None
# 		'''
#
# 		data_RAMLOAD = []
# 		### TODO: must be
# 		if len(data[0]) > self.ram_size or len(data[1]) > self.ram_size:
# 			raise ValueError('Cannot write segment larger than '+str(self.ram_size)+' as window function.')
#
# 		for i in range(len(data[0])):
# 				data_RAMLOAD.append(int(data[0][i]).to_bytes(2, byteorder = 'big', signed= True) + int(data[1][i]).to_bytes(2, byteorder = 'big', signed = True))
#
# 		for i in range(len(data[0])):
# 			Value, Index = mk_val_ind((RAM_BASE + (i + ncov*self.ram_size)*4)<<2)
# 			self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, data_RAMLOAD[i])
#
# 	def get_data_RAM(self, ncov):
# 		'''
# 		Function transfer data from FPGA's OnChip Memory to USB
#
# 		Input:
# 			ncov: Number of OnChip Memory to write the data (From 0 to 3)
# 		Output:
# 			data: ...
# 		'''
# 		data_RAM = []
# 		dty = dtype(int16)
# 		dty = dty.newbyteorder('>')
#
# 		for i in range(self.nsamp):
# 			Value, Index = mk_val_ind((RAM_BASE + (i+ ncov*self.ram_size)*4)<<2)
# 			data_RAM.append(frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ, Value, Index,4), dtype = dty))
#
# 		return (data_RAM)
#
# 	def set_threshold(self, thresh, ncov):
# 		'''
# 		Function sets the value of threshold for state discriminator
#
# 		Input:
# 			thresh: The value of threshold
# 			ncov: Number of OnChip Memory (and discriminator related to that)
# 		Output:
# 			None
# 		'''
# 		q0 = int(thresh).to_bytes(8, byteorder='big', signed = True)
#
# 		Value, Index = mk_val_ind((CAP_BASE + COV_THRESH_BASE + COV_THRESH_SUBBASE + ncov*4)<<2)
# 		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, q0[4:8])
# 		Value, Index = mk_val_ind((CAP_BASE + COV_THRESH_BASE + ncov*4)<<2)
# 		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, q0[0:4])
#
# 	def get_threshold(self, ncov):
# 		'''
# 		Function gets the value of threshold for state discriminator
#
# 		Input:
# 			ncov: Number of OnChip Memory (and discriminator related to that)
# 		Output:
# 			thresh: Threshold value
# 		'''
# 		dt = dtype(int64)
# 		dt = dt.newbyteorder('>')
# 		a,v = mk_val_ind((CAP_BASE + COV_THRESH_BASE + COV_THRESH_SUBBASE + ncov*4)<<2)
# 		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		a,v = mk_val_ind((CAP_BASE + COV_THRESH_BASE + ncov*4)<<2)
# 		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		q0 = frombuffer(b1+b0, dtype = dt)[0]
# 		return (q0)
#
# 	def get_cov_result(self, ncov):
# 		'''
# 		Function returns last covariance coefficient calculated in i-th discriminator
#
# 		Input:
# 			ncov: Number of OnChip Memory (and discriminator related to that)
# 		Output:
# 			cov = covariance coefficient value
# 		'''
# 		dt = dtype(int64)
# 		dt = dt.newbyteorder('>')
# 		a,v = mk_val_ind((CAP_BASE + COV_RES_BASE + ncov*4)<<2)
# 		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		a,v = mk_val_ind((CAP_BASE + COV_RES_BASE + COV_RES_SUBBASE + ncov*4)<<2)
# 		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		q = frombuffer(b1+b0, dtype = dt)[0]
# 		return (q)
#
# 	def get_cov_result_avg(self, ncov):
# 		'''
# 		Function returns averaged over all calculations covariance coefficient calculated in i-th discriminator
#
# 		Input:
# 			ncov: Number of OnChip Memory (and discriminator related to that)
# 		Output:
# 			cov = averaged covariance coefficient value
# 		'''
# 		start = time.time()
# 		dt = dtype(int64)
# 		dt = dt.newbyteorder('>')
# 		a,v = mk_val_ind((CAP_BASE + COV_RESAVG_BASE + ncov*4)<<2)
# 		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		a,v = mk_val_ind((CAP_BASE + COV_RESAVG_BASE + COV_RESAVG_SUBBASE + ncov*4)<<2)
# 		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
# 		q = frombuffer(b1+b0, dtype = dt)[0]
# 		stop = time.time()
# 		print('Time of cow data transfer', stop - start)
# 		return (q)
#
# 	def get_resultnumbers(self):
# 		'''
# 		Function returns amount of times each discrimination result happens
#
# 		Input:
# 			None
# 		Output:
# 			numbs: Amount of times each state happens for instance numbs[3]= numbs['0011'] - amount of times when two qubits were at |1>
# 		'''
# 		dt = dtype(int32)
# 		dt = dt.newbyteorder('>')
# 		b0 = []
# 		for i in range(16):
# 			a,v = mk_val_ind((CAP_BASE + COV_NUMB_BASE + i*4)<<2)
# 			b0.append(frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 ), dtype = dt)[0])
# 		return (b0)
#
# 	def set_trig_src_period(self, period):
# 		'''
# 		Setst pulse period of the internal pulse generator on "TRIG OUT A" SMA
#
# 		Input:
# 			period: int Period in clock cycles (125 MHz)
# 		'''
# 		period = int(period)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_PERIOD_LO, period)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_PERIOD_HI, period>>32)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_CTRL, 1<<TRIG_SRC_CTRL_UPDATE)
#
# 	def set_trig_src_width(self, width):
# 		'''
# 		Setst pulse width of the internal pulse generator on "TRIG OUT A" SMA
#
# 		Input:
# 			width: int Width in clock cycles (125 MHz)
# 		'''
# 		width = int(width)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_WIDTH_LO, width)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_WIDTH_HI, width>>32)
# 		self.write_reg(TRIG_SRC_BASE, TRIG_SRC_CTRL, 1<<TRIG_SRC_CTRL_UPDATE)
