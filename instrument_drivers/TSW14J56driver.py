from numpy import *
#from usb_intf import *
#from reg_intf import *
from ADS54J40EVM import *

from qsweepy.instrument import Instrument

from qsweepy.instrument_drivers._ADS54J40.usb_intf import *
from qsweepy.instrument_drivers._ADS54J40.reg_intf import *
from qsweepy.instrument_drivers._ADS54J40 import *

import usb.core
import time
import sys
import zlib

sys.path.append('C:\qtlab_replacement\qsweepy\instrument_drivers\_ADS54J40')

class TSW14J56_evm_reducer():
	def __init__(self, adc):
		self.adc = adc
		self.output_raw = True
		self.last_cov = True
		self.avg_cov = True
		self.resultnumber = True
		self.trig = 'ext'

	def get_clock(self):
		return self.adc.get_clock()
		
	def get_nums(self):
		return self.adc.nsegm
	def get_nop(self):
		return self.adc.nsamp
	def set_nums(self, nums):
		self.adc.nsegm = nums
	def set_nop(self, nop):
		self.adc.nsamp = nop
		
	def get_points(self):
		points = {}
		if self.output_raw:
			points.update({'Voltage':[('Sample',arange(self.adc.nsegm), ''), 
								 ('Time',arange(self.adc.nsamp)/self.adc.get_clock(), 's')]})
		if self.last_cov:
			points.update({'last_cov'+str(i):[] for i in range(self.adc.num_covariances)})
		if self.avg_cov:
			points.update({'avg_cov'+str(i):[] for i in range(self.adc.num_covariances)})
		if self.resultnumber:
			points.update({'resultnumbers':[arange(2**self.num_covariances)]})
		return (points)
		
	def get_dtype(self):
		dtypes = {}
		if self.output_raw:
			dtypes.update({'Voltage':complex})
		if self.last_cov:
			dtypes.update({'last_cov'+str(i):int for i in range(self.adc.num_covariances)})
		if self.avg_cov:
			dtypes.update({'avg_cov'+str(i):float for i in range(self.adc.num_covariances)})
		if self.resultnumber:
			dtypes.update({'resultnumbers': int})
		return (dtypes)
		
	def get_opts(self):
		opts = {}
		if self.output_raw:
			opts.update({'Voltage':{'log': None}})
		if self.last_cov:
			opts.update({'last_cov'+str(i):{'log': None} for i in range(self.adc.num_covariances)})
		if self.avg_cov:
			opts.update({'avg_cov'+str(i):{'log': None} for i in range(self.adc.num_covariances)})
		if self.resultnumber:
			opts.update({'resultnumbers': {'log': None}}) 
		return (opts)

	def measure(self):
		
		if self.avg_cov:
			avg_covs_initial = [self.adc.get_cov_result_avg(i) for i in range(self.adc.num_covariances)]
		result = {}
		self.adc.capture(trig=self.trig, cov = (self.last_cov or self.avg_cov or self.resultnumber))
		if self.output_raw:
			result.update({'Voltage':self.adc.get_data()})
		if self.last_cov:
			result.update({'last_cov'+str(i):self.adc.get_cov_result(i) for i in range(self.adc.num_covariances)})
		if self.avg_cov:
			result.update({'avg_cov'+str(i):self.adc.get_cov_result_avg(i)-avg_covs_initial[i] for i in range(self.adc.num_covariances)})
		if self.resultnumber:
			result.update({'resultnumbers':[self.adc.get_resultnumbers()]})
			
		return (result)

class TSW14J56_evm():
	def __init__(self, fpga_config = True):
		#Number of samples per channel
		self.nsamp = 65536
		self.nsegm = 1
		#Capture timeout
		self.timeout = 2
		self.ram_size = 2048 #in words of 32
		self.num_covariances = 4
		
		self.usb_reboot_timeout = 10
		self.debug_print = False
		self.fpga_firmware = "C:\qtlab_replacement\qsweepy\instrument_drivers\_ADS54J40\qubit_daq.rbf"
		self.cov_cnt = 0
		
		#To do make a register readout to check ADS-programmed status"
		self.ads = ADS54J40EVM(0)
		self.ads.load_lmk_config()
		time.sleep(10)
		self.ads.load_ads_config()
		self.ads.device.close()
		
		self.dev=usb.core.find(idVendor=id.VENDOR, idProduct=id.PRODUCT)
		if self.dev is None:
			raise Exception('TSW14J56: Device not found!')
		self.dev.set_configuration(1)
		self.usb_reset()
		#Chech if FRGA already configured
		if fpga_config:
			dev_checksum = 0
			try:
				dev_checksum = self.read_reg(FX3_BASE, FX3_CRC32)
			except:
				pass
			try:		
				firmware_rbf = open(self.fpga_firmware, 'rb')
				firmware = firmware_rbf.read()
			except:
				Warning("TSW14J56: Unable to read FPGA firmware from file. The device may be unprogrammed!")
				return
			checksum = zlib.crc32(firmware)	
			if dev_checksum != checksum:
				self.fpga_config(firmware = firmware)
				
		self.sync_req()
	
	def get_clock(self):
		###TODO: get this shit right
		return 1e9
	
	def usb_reset(self):
		#Reset USB chip and wait until it reboot. 
		self.dev.reset()
		self.dev = None
		t0=time.time()
		while( time.time()-t0 < self.usb_reboot_timeout ):
			try:
				self.dev=usb.core.find(idVendor=id.VENDOR, idProduct=id.PRODUCT)
				if self.dev is None:
					time.sleep(0.05)
				else:
					self.dev.set_configuration(1)
					break
			except:
				self.dev = None
				pass
		if self.dev is None:
			raise Exception('Device not found')
			
	def sync_req(self):
		self.write_reg(JESD_BASE, JESD_CTRL, 1)
		
	def system_reset(self):
		self.write_reg(CAP_BASE, CAP_CTRL, 1<<CAP_CTRL_ABORT) # Abort current capture process before reset 
		self.write_reg(FX3_BASE, FX3_RST, 1)
			
	def reset(self):
		self.system_reset()
		self.sync_req()
		self.usb_reset()
		
	def fpga_config(self, source = None, firmware = None):
		if source is None:
			source = self.fpga_firmware
		if firmware is None:
			firmware_rbf = open(source, 'rb')
			firmware = firmware_rbf.read()
		if len(firmware) > 0xFFFFFFFF:
			raise ValueError('TSW14J56: FPGA firmware size >0xFFFF bytes not supported!')
		self.usb_reset()
		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.FPGA_CONF_INIT, 0, 0, to_bytes(len(firmware)+2,4) )
		self.dev.write( endpoints.OUT, firmware+bytes([0,0]))
		res = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.FPGA_CONF_FIN, 0, 0, 2 )
		if res[0]==0:
			raise Exception('TSW14J56: FPGA configuration failed!')
		self.usb_reset()
		checksum = zlib.crc32(firmware)
		self.write_reg(0x10000, 20, checksum)
		
	def write_reg(self,base, offset, raw_data):
		#We just shift to avoid address modification by original FX3 chip firmware from Prakash
		#We will shift them back in FPGA	
		address = (base + offset)<<2
		if type(raw_data) is list:
			data=raw_data
		else:
			data = to_bytes(raw_data, 4)
		Value, Index = mk_val_ind(address)
		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, data)
		if(self.debug_print):
			if type(raw_data) is not list:
				print ("Write:", hex(Value), hex(Index), hex(raw_data))
			else:
				print ("Write:", hex(Value), hex(Index), raw_data)
	
	def read_reg(self,base, offset):
		#We just shift to avoid address modification by original FX3 chip firmware from Prakash
		#We will shift them back in FPGA	
		address = (base + offset)<<2
		Value, Index = mk_val_ind(address)
		data = frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ, Value,Index,4), dtype = uint32odd)[0]
		if(self.debug_print): print( "Read:", hex(Value), hex(Index), hex(data) )
		return data

	def capture(self, trig = "man", cov = False, fifo = True):
		'''
		Function starts data acquisition
		
		Input:
			trig: Way to trigger acquisition process: 'man' - after using the function, 'ext' - after external trigger occures
			cov:  Necessity to calculate covariance coefficient with data loaded to FPGA's OnChip Memory
			fifo: Necessity to write data to DDR (False used only to check that feedback loop works properly)
		
		Output:
			None
		'''
		self.write_reg(CAP_BASE, CAP_SEGM_NUM, int(self.nsegm))
		if (cov):
			self.write_reg(CAP_BASE, COV_LEN, int(self.nsamp/8))
			self.write_reg(CAP_BASE, CAP_LEN, int(self.nsamp/8))
			self.cov_cnt = self.cov_cnt + 1
		else:
			self.write_reg(CAP_BASE, CAP_LEN, int(self.nsamp/8))
		if(trig == "man"):
			self.write_reg(CAP_BASE, CAP_CTRL, 1<<CAP_CTRL_START |fifo << FIFO_ST )
		elif(trig == "ext"):
			self.write_reg(CAP_BASE, CAP_CTRL, 1<<CAP_CTRL_START| 1<<CAP_CTRL_EXT_TRIG |cov << COV_ST |fifo << FIFO_ST)
		else: return	
		
		t1 = time.time()
		while(1):
			if( not( self.read_reg(CAP_BASE, CAP_CTRL) & 1<<CAP_CTRL_BUSY ) ):
				break
			else:
				if(self.debug_print): print("Busy..")
				
			if(time.time()-t1>self.timeout):
				print ("Capture failed")
				break
		if(self.debug_print): print("Done!")		
	
	def get_data(self):
		'''
		Transfer data from DDR
		'''
		#Number of samples per channel to read
		nsegm = self.nsegm
		if self.nsegm ==0:
			nsegm = 1
		data_len = self.nsamp*nsegm
		self.write_reg(FX3_BASE, FX3_LEN, int(data_len/16))
		#Trigger DDR read
		self.write_reg(FX3_BASE, FX3_CTRL, FX3_CTRL_START)
		data = frombuffer(self.dev.read(endpoints.IN, data_len*4), dtype = dtype(int16))
		
		data = reshape(data, (data_len, 2))
		return reshape(data.T[0,:],(self.nsegm, self.nsamp))+1j*reshape(data.T[1,:], (self.nsegm, self.nsamp))

		
		#dataiq = reshape(data, (2, self.nsamp*self.nsegm))[0]+ 1j*reshape(data, (2, self.nsamp*self.nsegm))[1]
		#return (reshape(dataiq, (self.nsegm, self.nsamp)))
		
	def set_ram_data(self,data, ncov):
		'''
		Function transfer data to FPGA's OnChip Memory
		
		Input:
			data: TO DO write a way to load data
			ncov: Number of OnChip Memory to write the data (From 0 to 3)
		Output:
			None
		'''
		data_RAMLOAD = []
		### TODO: must be
		if len(data[0]) > self.ram_size or len(data[1]) > self.ram_size:
			raise ValueError('Cannot write segment larger than '+str(self.ram_size)+' as window function.')
		
		for i in range(len(data[0])):
				data_RAMLOAD.append(int(data[0][i]).to_bytes(2, byteorder = 'big', signed= True) + int(data[1][i]).to_bytes(2, byteorder = 'big', signed = True))
				
		for i in range(len(data[0])):
			Value, Index = mk_val_ind((RAM_BASE + (i + ncov*self.ram_size)*4)<<2)
			self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, data_RAMLOAD[i])
			
	def get_data_RAM(self, ncov):
		'''
		Function transfer data from FPGA's OnChip Memory to USB
		
		Input:
			ncov: Number of OnChip Memory to write the data (From 0 to 3)
		Output:
			data: ...
		'''
		data_RAM = []
		dty = dtype(int16)
		dty = dty.newbyteorder('>')

		for i in range(self.nsamp):
			Value, Index = mk_val_ind((RAM_BASE + (i+ ncov*self.ram_size)*4)<<2)
			data_RAM.append(frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ, Value, Index,4), dtype = dty))
			
		return (data_RAM)
			
	def set_threshold(self, thresh, ncov):
		'''
		Function sets the value of threshold for state discriminator
		
		Input:
			thresh: The value of threshold
			ncov: Number of OnChip Memory (and discriminator related to that)
		Output:
			None
		'''
		q0 = int(thresh).to_bytes(8, byteorder='big', signed = True)
		
		Value, Index = mk_val_ind((CAP_BASE + COV_THRESH_BASE + COV_THRESH_SUBBASE + ncov*4)<<2)
		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, q0[4:8])
		Value, Index = mk_val_ind((CAP_BASE + COV_THRESH_BASE + ncov*4)<<2)
		self.dev.ctrl_transfer(vend_req_dir.WR, vend_req.REG_WRITE, Value, Index, q0[0:4])
		
	def get_threshold(self, ncov):
		'''
		Function gets the value of threshold for state discriminator
		
		Input:
			ncov: Number of OnChip Memory (and discriminator related to that)
		Output:
			thresh: Threshold value
		'''
		dt = dtype(int64)
		dt = dt.newbyteorder('>')
		a,v = mk_val_ind((CAP_BASE + COV_THRESH_BASE + COV_THRESH_SUBBASE + ncov*4)<<2) 
		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		a,v = mk_val_ind((CAP_BASE + COV_THRESH_BASE + ncov*4)<<2) 
		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		q0 = frombuffer(b1+b0, dtype = dt)[0]
		return (q0)

	def get_cov_result(self, ncov):
		'''
		Function returns last covariance coefficient calculated in i-th discriminator
		
		Input:
			ncov: Number of OnChip Memory (and discriminator related to that)
		Output:
			cov = covariance coefficient value
		'''
		dt = dtype(int64)
		dt = dt.newbyteorder('>')
		a,v = mk_val_ind((CAP_BASE + COV_RES_BASE + ncov*4)<<2) 
		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		a,v = mk_val_ind((CAP_BASE + COV_RES_BASE + COV_RES_SUBBASE + ncov*4)<<2) 
		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		q = frombuffer(b1+b0, dtype = dt)[0]
		return (q)
		
	def get_cov_result_avg(self, ncov):
		'''
		Function returns averaged over all calculations covariance coefficient calculated in i-th discriminator
		
		Input:
			ncov: Number of OnChip Memory (and discriminator related to that)
		Output:
			cov = averaged covariance coefficient value
		'''
		dt = dtype(int64)
		dt = dt.newbyteorder('>')
		a,v = mk_val_ind((CAP_BASE + COV_RESAVG_BASE + ncov*4)<<2) 
		b0 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		a,v = mk_val_ind((CAP_BASE + COV_RESAVG_BASE + COV_RESAVG_SUBBASE + ncov*4)<<2) 
		b1 = self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 )
		q = frombuffer(b1+b0, dtype = dt)[0]
		return (q)#/self.cov_cnt)
		
	def get_resultnumbers(self):
		'''
		Function returns amount of times each discrimination result happens
		
		Input:
			None
		Output:
			numbs: Amount of times each state happens for instance numbs[3]= numbs['0011'] - amount of times when two qubits were at |1>
		'''
		dt = dtype(int32)
		dt = dt.newbyteorder('>')
		b0 = []
		for i in range(15):
			a,v = mk_val_ind((CAP_BASE + COV_NUMB_BASE + i*4)<<2) 
			b0.append(frombuffer(self.dev.ctrl_transfer(vend_req_dir.RD, vend_req.REG_READ,a,v,4 ), dtype = dt)[0])
		return (b0)