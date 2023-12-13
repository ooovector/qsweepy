import numpy as np

from qsweepy.instrument_drivers._QubitDAQ.usb_intf import *
from qsweepy.instrument_drivers._QubitDAQ.reg_intf import *
from qsweepy.instrument_drivers._QubitDAQ.ADS54J40 import *

import usb.core
import usb.util
import time
import zlib
import os
import warnings

module_dir = os.path.dirname(os.path.abspath(__file__))

class Device:
	""" This is the main class of the driver for the QubitDAQ board.

	Attributes
	---------
	dev

	serial : str
		A 16-digit hexadecimal USB serial number string.
	nsamp : int
		A number of 16-bit samples to be acquired per channel (I and Q)
	nsegm : int
		A number of segments of nsamp samples to be acquired by an external trigger.
		Must not be equal to 0.
	timeout : float
		A timeout fo data capture in seconds.
	usb_reboot_timeout : float
		A timeout for FX3 USB chip reboot used by the usb_reset() method.
	fpga_firmware : str
		A path to an *.rbf file containing FPGA firmware.
	trig_src_mode : str
		Trigger source operation mode
		"free" - free run mode;
		"when_ready" - trigger source is active only when device is ready to receive trigger
			i.e. after call of the start() method;
	"""
	def __init__(self, serial: str, fpga_config: bool = True, adc_config: bool = True, firmware: str = None) -> None:
		""" This is the constructor of the Device class of the driver for the QubitDAQ board.

		If you use Jupyter Notebook or any other interactive tool and you want to reconnect to the device, please
		delete previously created object first using the example code below. Libusb can't detach driver on Windows,
		so, if there is an active connection to the device, you'll get "can't climb interface" error if you'll not
		delete the old object.

		try: del adc
		except: pass
		adc = driver.Device("0009052001481708")

		Arguments
		---------
		serial : str
			A 16-digit hexadecimal serial number string used to identify a device.
			It corresponds to the iSerialNumber USB string descriptor of a device.
		fpga_config : bool, optional
			A flag that determines whether the FPGA configuration will be done or not.
		adc_config : bool, optional
			A flag that determines whether the ADC configuration will be done or not.
		firmware : str, optional
			A path to the FPGA firmware *.rbf file. If not specified, the firmware will be
			loaded from the same location, where this module is stored.
		"""
		#Number of samples per channel
		self.dev = None
		self.serial = serial.lower()
		self.nsamp = 65536
		self.nsegm = 1
		self.data_capture_timeout = 3
		self.usb_reboot_timeout = 10
		self.threshold = np.zeros(NUM_DESCR_CH, dtype = int64)
		self.trig_mode = "ext"
		self.trig_cap_edge = "neg"
		# self.trig_src_mode = "free"
		self.trig_src_mode = "when_ready"
		self.debug_print = False
		self.num_covariances = NUM_DESCR_CH

		if firmware is None:
			self.fpga_firmware = module_dir + "\\qubit_daq.rbf"
		else:
			self.fpga_firmware = firmware

		#To do make a register readout to check ADS-programmed status"
		if adc_config:
			self.ads = ADS54J40()
			if self.ads.read_reg(ADS_CTRL_ST_ADDR) == ADS_CTRL_ST_VL:
				print ("ADS54J40 already programmed")
				self.ads.device.close()
			else:
				print ("Programming ADS54J40 ")
				self.ads.load_lmk_config()
				time.sleep(5)
				self.ads.load_ads_config()
				self.ads.device.close()

		self.usb_connect()
		if fpga_config:
			try:
				self.fpga_config()
			except usb.core.USBError:
				self.usb_reset()
				self.fpga_config()

		self.jesd204_sync()
		self.set_trig_src_mode(self.trig_src_mode)

	def __del__(self):
		if self.dev is not None:
			usb.util.release_interface(self.dev, 0)

	def usb_connect(self) -> None:
		""" A method that opens USB connection

		This method searches for USB device with a specific VID, PID and serial number
		and establishes connection if the device is found.

		Raises
		-----
		Exception
			If device is not found.
		"""
		criteria = lambda dev: usb.util.get_string(dev, dev.iSerialNumber) == self.serial
		self.dev = usb.core.find(custom_match=criteria, idVendor=ID_VENDOR, idProduct=ID_PRODUCT)
		if self.dev is None:
			raise Exception('QubitDAQ: Device not found!')
		self.dev.set_configuration(1)

	def i2c_reset(self) -> None:
		""" A method that resets the I2C DMA channels

		This method resets I2C DMA channel of the FX3 USB chip.
		Use it to clear any I2C related errors. For example, in case
		if data exchange via I2C related control transfer is timed out.
		"""
		self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_WR, VEND_RQ_RESET_I2C, 0, 0)

	def usb_reset(self) -> None:
		""" A method that resets the FX3 USB chip

		This method resets the FX3 chip and waits until it reboots.
		Call of this method will also cause FPGA configuration lost, so,
		it will be needed to call the self.fpga_config() after.

		Raises
		-------
		Exception
			If timed out.
		"""
		#self.dev.reset() It doesn't work. It doesn't return at all.
		self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_WR, VEND_RQ_USB_RESET, 0, 0)
		t0 = time.time()
		while time.time() - t0 < self.usb_reboot_timeout :
			try:
				self.usb_connect()
			except:
				time.sleep(0.1)
				pass
		if self.dev is None:
			raise Exception("QubitDAQ: USB reset timed out, the FX3 may not have booted!")

	def gpif_dma_reset(self) -> None:
		""" A method that resets the GPIF DMA module

		This method resets the GPIF DMA module inside the FPGA.
		This module is responsible for data exchange via USB bulk endpoints.
		"""
		self.write_reg(FX3_BASE + FX3_CTRL, FX3_CTRL_ABORT)

	def fpga_reset(self) -> None:
		"""Global FPGA reset"""
		self.write_reg(FX3_BASE + FX3_CTRL, FX3_CTRL_RESET)

	def jesd204_sync(self) -> None:
		"""Issue a synchronization request for the JESD204 interface of the ADC"""
		self.write_reg(JESD_LINK_BASE + JESD_LINK_CTRL, JESD_LINK_CTRL_SYNC_REQ)

	def fpga_config(self) -> None:
		"""A method that performs FPGA configuration

		This method loads configuration into the FPGA from *.rbf file specified in the self.fpga_firmware attribute.
		First it's tries to access the FPGA register where a CRC32 checksum of the firmware is stored. If the register is
		unavailable or the checksum doesn't match the checksum of the *.rbf file, then the configuration will be performed.
		"""
		dev_checksum = 0
		try:
			dev_checksum = self.read_reg(FX3_BASE + FX3_CRC32)
		except:
			self.i2c_reset()
		try:
			firmware_rbf = open(self.fpga_firmware, 'rb')
			firmware_data = firmware_rbf.read()
		except:
			warnings.warn("QubitDAQ: Unable to read FPGA firmware from file. The device may be unprogrammed!")
			return
		checksum = zlib.crc32(firmware_data)
		if dev_checksum != checksum:
			if len(firmware_data) > 0xFFFFFFFF:
				raise ValueError('QubitDAQ: FPGA firmware size >0xFFFFFFFF bytes not supported!')
			if len(firmware_data)%4 != 0:
				raise ValueError('QubitDAQ: FPGA firmware size must be dividable by 4!')
			value, index = mk_val_ind( int(len(firmware_data)/4)+1 )
			res = self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_RD, VEND_RQ_FPGA_CONF_INIT, value, index, 2)
			n_status = res[0]  # FPGA status pin
			conf_done = res[1]  # FPGA status pin
			if conf_done != 0:
				raise Exception('QubitDAQ: FPGA configuration initialization failed! CONF_DONE != 0')
			if n_status != 1:
				raise Exception('QubitDAQ: FPGA configuration initialization failed! nSTATUS != 1')
			self.dev.write( EP_OUT, firmware_data+bytes([0,0,0,0]) )
			res = self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_RD, VEND_RQ_FPGA_CONF_FIN, 0, 0, 2 )
			n_status = res[0] #FPGA status pin
			conf_done = res[1] #FPGA status pin
			if conf_done != 1:
				raise Exception('QubitDAQ: FPGA configuration failed! CONF_DONE != 1')
			if n_status != 1:
				raise Exception('QubitDAQ: FPGA configuration failed! nSTATUS != 1')

			checksum = zlib.crc32(firmware_data)
			self.write_reg(FX3_BASE + FX3_CRC32, checksum)
		firmware_rbf.close()

	def set_ref_100mhz(self)->None:
		"""Sets the external reference frequency to 100MHz"""
		self.ads = ADS54J40()
		self.ads.load_lmk_config(filename="Config_ADC/LMK_100MHz_osc_100MHz_ref_Dpll.cfg")
		time.sleep(5)
		self.ads.load_ads_config()
		self.ads.device.close()
		self.jesd204_sync()

	def set_ref_10mhz(self)->None:
		"""Sets the external reference frequency to 10MHz"""
		self.ads = ADS54J40()
		self.ads.load_lmk_config(filename="Config_ADC/LMK_100MHz_osc_10MHz_ref_Dpll.cfg")
		time.sleep(5)
		self.ads.load_ads_config()
		self.ads.device.close()
		self.jesd204_sync()

	def write_reg(self ,address : int , raw_data: any) -> None:
		""" Writes data to an FPGA register.

		Arguments:
		-------------
		address : int
			A 32-bit register byte address.
		raw_data: a number or array-like
			A value to be written in the register(s). The value will be casted to uint32.
			If it's an array-like, then multiple 32-bit registers will be written
			starting from the specified address. The overall data size in bytes must be dividable by 4.
		"""
		if hasattr(raw_data, '__len__') :
			data = np.frombuffer( np.array(raw_data), dtype = uint8)
		else:
			data = np.frombuffer( uint32(raw_data), dtype = uint8)

		Value, Index = mk_val_ind( uint32(address) )
		self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_WR, VEND_RQ_REG_WRITE, Value, Index, data)

	def read_reg(self, address: int, dtype = uint32) -> int:
		"""Reads data from an FPGA register.

		Arguments:
		-------------
		address : int
			A 32-bit register byte address.

		Returns:
		------------
		int
			A value of the register.
		"""
		length = np.dtype(dtype).itemsize
		Value, Index = mk_val_ind(address)
		raw_data = self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_RD, VEND_RQ_REG_READ, Value, Index, length)
		data = np.frombuffer(raw_data, dtype = dtype)[0]
		if (self.debug_print): print( "Read:", hex(Value), hex(Index), hex(data) )
		return data

	def start(self)->None:
		"""Initiates data acquisition and processing"""
		if self.nsamp % 8 != 0:
			raise ValueError('Number of samples (self.nsamp) must be dividable by 8!')
		if self.nsamp < 0:
			raise ValueError('Number of samples (self.nsamp) must be positive!')
		if self.nsegm < 1:
			raise ValueError('Number of segments (self.nsegm) must be greater then 0!')
		if self.nsegm > MAX_NSEGM:
			raise ValueError('Number of segments (self.nsegm) exceeds the maximum of ({:d})!'.format(MAX_NSEGM))
		self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CAP_LEN, int(self.nsamp / 8))
		self.write_reg(PULSE_PROC_BASE + PULSE_PROC_NSEGM, int(self.nsegm))

		if self.trig_mode not in ('man', 'ext'):
			raise ValueError('Trigger mode (trig) must be \"man\" or \"ext\"!')

		if self.trig_cap_edge not in ("neg", "pos"):
			raise ValueError('Trigger capture edge (trig_cap_edge) must be \"neg\" or \"pos\"!')

		if self.trig_mode == "man":
			self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL, PULSE_PROC_CTRL_START)
		elif self.trig_mode == "ext":
			if self.trig_cap_edge == "pos":
				self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL,
							   PULSE_PROC_CTRL_START | PULSE_PROC_CTRL_EXT_TRIG_EN | PULSE_PROC_CTRL_TRIG_EDGE)
			elif self.trig_cap_edge == "neg":
				self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL,
							   PULSE_PROC_CTRL_START | PULSE_PROC_CTRL_EXT_TRIG_EN)

	def start_wait_done(self) ->bool:
		"""Starts data acquisition and waits until done

		Should be used with self.trig_mode="ext".

		Returns
		--------
			bool
				Status: True if data acquisition is successfully done, False if the function returns by timeout.
		"""
		self.start()
		t1 = time.time()
		while 1:
			time.sleep(5e-3)
			if self.check_done():
				return True

			if time.time() - t1 > self.data_capture_timeout:
				self.abort()
				return False

	def abort(self)->None:
		"""Aborts data acquisition"""
		self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL, PULSE_PROC_CTRL_ABORT)

	def check_done(self) -> bool:
		"""Checks if data acquisition is done

		Returns
		-------
			bool
				True if done.
		"""
		return not self.read_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL) & PULSE_PROC_CTRL_BUSY

	def soft_trig(self):
		"""Asserts software trigger if self.trig_mode is "man" """
		if self.trig_mode not in ('man', 'ext') :
			raise ValueError('QubitDAQ capture(): Trigger mode (trig) must be \"man\" or \"ext\"!')

		if self.trig_mode == "man":
			self.write_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL, PULSE_PROC_CTRL_SOFT_TRIG )

	def check_trig_ready(self)->bool:
		"""Checks if data capture module is ready for trigger

		Returns
		-------
			bool
				True if ready.
		"""
		return self.read_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL) & PULSE_PROC_CTRL_READY

	def read_ddr3(self, base_addr: int, length: int, dtype = np.int16 ) -> np.ndarray:
		"""Reads data from DDR3 memory

		Arguments
		---------
		base_addr : int
			A base address from which the reading will start. The address unit is 32-bit word.
		length : int
			An amount of data of the type "dtype" to read.
		dtype: optional
			A data type to interpret the data. The default value is int16.

		Returns
		-------
			numpy.ndarray
				A 1-d array of data with length and type corresponding to the "length" and "dtype" arguments.
		"""
		size = np.dtype(dtype).itemsize
		byte_len = int(length * size)
		#The DDR3 is read in 512-bit words which is 64 bytes.
		#The FX3 has maximal buffer size of 1024*16 bytes for super speed mode. Since the DMA
		#channel for DDR3 reading inside FPGA currently doesn't have the pketend signal, the FX3
		#buffer must be completely filled up in order to be committed. It sets the minimal amount of
		#data transfer to 1024*16 bytes.
		ddr_dma_len = int( ( 1024*16 * np.ceil( byte_len/(1024*16) ) )/64 )
		self.write_reg(FX3_BASE + FX3_LEN, ddr_dma_len)
		self.write_reg(FX3_BASE + FX3_ADDR, base_addr)
		# Trigger DDR read
		self.write_reg(FX3_BASE + FX3_CTRL, FX3_CTRL_START)
		return np.frombuffer( self.dev.read(EP_IN, ddr_dma_len*64 )[0:byte_len], dtype = dtype )

	def write_ddr3(self, base_addr: int, data: any) -> int:
		"""Writes data to DDR3 memory

		Arguments
		---------
		base_addr : int
			A base address at which the data will be written.
		data : array-like
			A data to be writen. A total amount in bytes must be divisible by 4.

		Returns
		-------
			int
				A number of bytes written.
		"""
		self.write_reg(FX3_BASE + FX3_ADDR, int(base_addr) )
		#Length units are 32-bit words
		data_bytes = bytes(data)
		data_len = len(data_bytes)
		if data_len % 4 != 0: Exception("A total amount of data bytes must be dividable by 4!")
		self.write_reg( FX3_BASE + FX3_LEN, int(data_len/4) )
		self.write_reg( FX3_BASE + FX3_CTRL, FX3_CTRL_START | FX3_CTRL_WR )
		return self.dev.write( EP_OUT, data_bytes )

	def get_data(self) -> tuple:
		"""Reads captured samples from DDR3 memory

		Returns
		--------
			numpy.array
				An array of complex numbers of the shape (nsegm, nsamp), where nsegm is the number of segments
				equal to self.nsegm and nsamp is the number of samples per segment equal to self.nsamp. The complex
				numbers represent a signal in quadratures: I+1j*Q, where I corresponds to the A channel of the ADC
				and Q -  to the B chanel.
		"""
		#Number of samples per channel to read
		nsegm = self.nsegm
		if self.nsegm ==0:
			nsegm = 1
		#Samples ar 16-bit
		data_len = int(self.nsamp*nsegm )
		data = self.read_ddr3(DDR_DATA_BASE, 2*data_len)
		data = np.reshape(data, (data_len, 2))
		res = np.reshape(data.T[0, :], (self.nsegm, self.nsamp)) + 1j * np.reshape(data.T[1, :], (self.nsegm, self.nsamp))
		return res

	def get_dot_prods(self):
		"""Reads dot products between acquired data and feature from device memory

		Dot products are computed in hardware between acquired data and feature, preloaded into device memory for
		each of NUM_DESCR_CH=4 qubit state discrimination channels using set_feature() method.

		Returns
		-------
			numpy.array
				An array of int64 values. The shape of the array is (self.nsegm, NUM_DESCR_CH )
				where self.nsegm is the number of acquired data segments and NUM_DESCR_CH=4 is number of state
				discrimination channels.
		"""
		return np.reshape( self.read_ddr3(DDR_RESULT_BASE, NUM_DESCR_CH * self.nsegm, dtype = int64), (self.nsegm, NUM_DESCR_CH
																									   ) )

	def get_states(self):
		"""Returns discriminated states of qubits

		Discrimination is done using a simple binary classification of dot products with threshold set by
		set_threshold() method. The dot products are computed in hardware between acquired data and feature,
		preloaded into device memory for each of NUM_DESCR_CH=4 qubit state discrimination channels using
		set_feature() method.

		Returns
		-------
			numpy.array
				An array of int values which contains only ones and zeros and represents qubits readout outcome.
				The shape of the array is (self.nsegm, NUM_DESCR_CH ) where self.nsegm is the number of acquired
				data segments and NUM_DESCR_CH=4 is the number of state discrimination channels.
		"""
		dot_prods = self.get_dot_prods()
		return np.asarray(dot_prods > self.threshold, dtype = int)

	def write_onchip_memory(self, base_addr : int, data : any) -> int:
		"""Writes data to the feature memory of the pulse processing module for state discrimination

		Arguments
		---------
			base_addr : int
				A base address from which the reading will be started. Address unit
				is a 32-bit word.
			data : array-like
				A data to be written. The overall length in bytes must be dividable
				by 4.

		Returns
		--------
			numpy.ndarray
				A 1-d array of data with length and type corresponding to the "length" and "dtype" arguments.
		"""
		data_bytes = bytes(data)
		data_len = len(data_bytes)
		if data_len % 4 != 0: Exception("A total amount of data bytes must be dividable by 4!")
		self.write_reg( FX3_BASE + FX3_ADDR, int(base_addr) )
		self.write_reg( FX3_BASE + FX3_LEN, int(data_len/4) )
		self.write_reg( FX3_BASE + FX3_CTRL, FX3_CTRL_START | FX3_CTRL_WR | FX3_CTRL_PATH )
		return self.dev.write( EP_OUT, data_bytes )

	def read_onchip_memory(self, base_addr:int, length:int, dtype = np.int16) -> np.ndarray:
		"""Reads data from the feature memory of the pulse processing module

		Arguments
		---------
			base_addr : int
				A base address from which the reading will be started. Address unit
				is a 32-bit word.
			length : int
				An amount of data of type dtype to read.
			dtype :
				A data type, the default value is int16.

		Returns
		--------
			numpy.ndarray
				A 1-d array of data with length and type corresponding to the "length" and "dtype" arguments.
		"""
		size = np.dtype( dtype ).itemsize
		byte_len = length * size
		bytes_to_read = int( np.ceil(byte_len/4)*4 )
		self.write_reg( FX3_BASE + FX3_ADDR, int(base_addr) )
		self.write_reg( FX3_BASE + FX3_LEN, int( byte_len/4 ) )
		self.write_reg( FX3_BASE + FX3_CTRL, FX3_CTRL_START | FX3_CTRL_PATH )
		return np.frombuffer( self.dev.read(EP_IN, bytes_to_read )[0:byte_len], dtype = dtype )

	def set_feature(self, feature: np.ndarray, ch:int)->None:
		"""Writes normalized feature data into feature memory associated with state discrimination channel

			The feature array is normalised in such way that the highest amplitude of a complex  value
		will be equal to the full ADC range.

		Arguments
		---------
			feature : numpy.ndarray
				An array of complex values
			ch : int
				Qubit state discrimination channel
		"""
		if ch<0 or ch >= NUM_DESCR_CH :
			raise ValueError("Channel number is out of range!")
		nsamp = FEATURE_MEM_DEPTH//4
		base_addr = ch * FEATURE_MEM_DEPTH
		feature = feature[:nsamp]
		feature_max = np.max(np.abs(feature))
		if feature_max !=0:
			feature = 2**(ADC_RESOLUTION-1) * feature/feature_max
		feature = np.reshape( np.vstack( (feature.real, feature.imag)).T, (len(feature)*2) )
		feature = np.int16(feature)
		feature = np.hstack( (feature, np.zeros(nsamp*2 - len(feature), dtype = np.int16)) )
		self.write_onchip_memory( base_addr, feature )

	def set_threshold(self, threshold: int64, chan: int) -> None:
		"""Sets a 64-bit threshold for binary classification

		Arguments
		---------
		threshold : int64
			A threshold value used for qubit state discrimination.
		chan : int
			A start discrimination channel index which starts from 0.
			There are NUM_DESCR_CH channels.
		"""
		if chan >= NUM_DESCR_CH or chan < 0:
			ValueError("Channel number is out of range!")
		self.threshold[chan] = threshold
		self.write_reg( PULSE_PROC_BASE + PULSE_PROC_THRESHOLD + 8*chan, [int64(threshold),] )

	def get_threshold(self, chan: int) -> int64:
		"""Returns a 64-bit threshold for binary classification

		Arguments
		---------
		chan : int
			A start discrimination channel index which starts from 0.
			There are NUM_DESCR_CH channels.

		Returns
		-------
			int64
			A threshold value used for qubit state discrimination.
		"""
		if chan >= NUM_DESCR_CH or chan < 0:
			raise ValueError("Channel number is out of range!")
		return self.read_reg( PULSE_PROC_BASE + PULSE_PROC_THRESHOLD + 8*chan, dtype = int64 )

	def get_dot_prod_ave(self) -> np.ndarray:
		"""Returns dot products averaged over data segments

		The dot product is calculated by the FPGA in real time between a captured segment of samples
		and features stored in memory for each of NUM_DESCR_CH discrimination channels. The averaged
		across multiple segments is accessed by this function.

		Returns
		-------
			numpy.ndarray
				An array of float values of averaged dot products for all the discrimination channels.
				Array size is equal to NUM_DESCR_CH=4.
		"""
		length = 8*NUM_DESCR_CH
		Value, Index = mk_val_ind(PULSE_PROC_BASE + PULSE_PROC_DOT_AVE)
		raw_data = self.dev.ctrl_transfer(RQ_TYPE_DEV_VEND_RD, VEND_RQ_REG_READ, Value, Index, length)
		return np.frombuffer(raw_data, dtype = np.int64)/self.nsegm

	def get_averaged_data(self) -> np.ndarray:
		"""Returns data averaged over segments

		Returns
		-------
			numpy.ndarray
				An array of complex numbers representing signal in quadratures in form of I+iQ.
				The length of the array is self.nsamp.
		"""
		data = self.read_onchip_memory( AVER_MEM_BASE_ADDR , 2*self.nsamp, dtype = np.int32)/self.nsegm
		data = np.reshape(data, (self.nsamp, 2))
		return data.T[0]  + 1j * data.T[1]

	def set_trig_src_mode(self, mode: str)->None:
		"""Sets trigger source operation mode

		Arguments
		----------
			mode: str
				"free" - free run mode;
				"when_ready" - trigger generated only when device is ready to receive trigger;
		"""
		if mode == "free":
			self.write_reg(TRIG_SRC_BASE + TRIG_SRC_CTRL, 0)
			self.trig_src_mode = "free"
		elif mode == "when_ready":
			self.write_reg(TRIG_SRC_BASE + TRIG_SRC_CTRL, TRIG_SRC_CTRL_MODE)
			self.trig_src_mode = "when_ready"
		else:
			raise ValueError("mode argument must be either \"free\" or \"when_ready\"")

	def set_trig_src_period(self, period: int) -> None:
		"""A method to set the period of trigger pulses

		This method sets the period of trigger pulsed generated by the internal pulse generator.
		The output of the generator is "TRIG OUT A" SMA connector

		Arguments
		----------
			period: int
				A period in clock cycles. Clock frequency is 125 MHz.
		"""
		period = int(period)
		self.write_reg(TRIG_SRC_BASE + TRIG_SRC_PERIOD_LO, period)
		self.write_reg(TRIG_SRC_BASE + TRIG_SRC_PERIOD_HI, period>>32)
		self._trig_src_update()

	def set_trig_src_width(self, width: int) -> None:
		"""A method to set the width of the trigger pulses

		This method sets the width of the trigger pulses generated by the internal pulse generator.
		The output of the generator is "TRIG OUT A" SMA connector

		Arguments
		----------
			width: int
				A width in clock cycles. Clock frequency is 125 MHz.
		"""
		width = int(width)
		self.write_reg(TRIG_SRC_BASE + TRIG_SRC_WIDTH_LO, width)
		self.write_reg(TRIG_SRC_BASE + TRIG_SRC_WIDTH_HI, width>>32)
		self._trig_src_update()

	def  _trig_src_update(self):
		"""Updates period and pulse width of the trigger source"""
		if self.trig_src_mode == "free":
			self.write_reg(TRIG_SRC_BASE + TRIG_SRC_CTRL, TRIG_SRC_CTRL_UPDATE)
		elif self.trig_src_mode == "when_ready":
			self.write_reg(TRIG_SRC_BASE + TRIG_SRC_CTRL, TRIG_SRC_CTRL_UPDATE|TRIG_SRC_CTRL_MODE)
		else:
			raise ValueError("mode argument must be either \"free\" or \"when_ready\"")

	def set_trig_delay(self, delay:float)->None:
		"""Sets trigger delay

		If delay is nonzero, data acquisition start has delay after external trigger.
		This option is not active in "man" (software) trigger mode.

		Arguments
		----------
			delay: float
				A delay value in seconds. The delay resolution is set by pulse processing core
				clock frequency PP_CLK_FREQ = 125 MHz (8 ns).
		"""
		if delay< 0:
			raise ValueError("Delay must be positive!")
		delay = int(delay * PP_CLK_FREQ)
		self.write_reg(PULSE_PROC_BASE + PULSE_PROC_TRIG_DLY, delay)

	def get_trig_delay(self)->float:
		"""Sets trigger delay

		If delay is nonzero, data acquisition start has delay after external trigger.
		This option is not active in "man" (software) trigger mode.

		Returns
		----------
		 	float
				A delay value in seconds.
		"""
		return self.read_reg(PULSE_PROC_BASE + PULSE_PROC_TRIG_DLY)/PP_CLK_FREQ

	def print_fx3_regs(self) -> None:
		"""Debug print of the FX3 registers content """
		fx3_ctrl = self.read_reg(FX3_BASE + FX3_CTRL)
		print("FX3_CTRL_START         :", (fx3_ctrl & FX3_CTRL_START) > 0)
		print("FX3_CTRL_WR            :", (fx3_ctrl & FX3_CTRL_WR) > 0)
		print("FX3_CTRL_ABORT         :", (fx3_ctrl & FX3_CTRL_ABORT) > 0)
		print("FX3_CTRL_DDR_WR_BUSY   :", (fx3_ctrl & FX3_CTRL_DDR_WR_BUSY) > 0)
		print("FX3_CTRL_DDR_RD_BUSY   :", (fx3_ctrl & FX3_CTRL_DDR_RD_BUSY) > 0)
		print("FX3_CTRL_RESET         :", (fx3_ctrl & FX3_CTRL_RESET) > 0)
		print("FX3_CTRL_PATH          :", (fx3_ctrl & FX3_CTRL_PATH) > 0)
		print("FX3_CTRL_ONCHIP_WR_BUSY:", (fx3_ctrl & FX3_CTRL_ONCHIP_WR_BUSY) > 0)
		print("FX3_CTRL_ONCHIP_RD_BUSY:", (fx3_ctrl & FX3_CTRL_ONCHIP_RD_BUSY) > 0)
		print()
		print("FX3_LEN:   ", self.read_reg(FX3_BASE + FX3_LEN))
		print("FX3_ADDR:  ", self.read_reg(FX3_BASE + FX3_ADDR))
		print("FX3_CRC32: ", hex(self.read_reg(FX3_BASE + FX3_CRC32)))
		print("FX3_FIFO_LVL_DDR_WR:", self.read_reg(FX3_BASE + FX3_FIFO_LVL_DDR_WR))
		print("FX3_FIFO_LVL_DDR_RD:", self.read_reg(FX3_BASE + FX3_FIFO_LVL_DDR_RD))

	def print_pp_regs(self):
		"""Debug print of the Pulse Processing module registers content"""
		pp_ctrl = self.read_reg(PULSE_PROC_BASE + PULSE_PROC_CTRL)
		print("START         :", (pp_ctrl & PULSE_PROC_CTRL_START) > 0)
		print("BUSY          :", (pp_ctrl & PULSE_PROC_CTRL_BUSY) > 0)
		print("ABORT         :", (pp_ctrl & PULSE_PROC_CTRL_ABORT) > 0)
		print("READY         :", (pp_ctrl & PULSE_PROC_CTRL_READY) > 0)
		print("EXT_TRIG_EN   :", (pp_ctrl & PULSE_PROC_CTRL_EXT_TRIG_EN) > 0)
		print("TRIG_EDGE     :", (pp_ctrl & PULSE_PROC_CTRL_TRIG_EDGE) > 0)
		print("SOFT_TRIG     :", (pp_ctrl & PULSE_PROC_CTRL_SOFT_TRIG) > 0)
		print("RES_DMA_BUSY  :", (pp_ctrl & PULSE_PROC_CTRL_RES_DMA_BUSY) > 0)
		print("RES_DMA_READY :", (pp_ctrl & PULSE_PROC_CTRL_RES_DMA_READY) > 0)
		print()
		print("CAP_LEN          :", self.read_reg(PULSE_PROC_BASE + PULSE_PROC_CAP_LEN))
		print("CAP_FIFO_LVL     :", self.read_reg(PULSE_PROC_BASE + PULSE_PROC_CAP_FIFO_LVL))
		print("NSEGM            :", self.read_reg(PULSE_PROC_BASE + PULSE_PROC_NSEGM))
		print("TRIG_DLY         :", self.read_reg(PULSE_PROC_BASE + PULSE_PROC_TRIG_DLY))
		print("RES_DMA_FIFO_LVL :", self.read_reg(PULSE_PROC_BASE + PULSE_PROC_RES_DMA_FIFO_LVL))

	def print_trig_src_regs(self):
		"""Debug print of the trigger_source module registers"""
		ctrl = self.read_reg(TRIG_SRC_BASE + TRIG_SRC_CTRL)
		print("UPDATE         :", (ctrl & TRIG_SRC_CTRL_UPDATE) > 0)
		print("MODE          :", (ctrl & TRIG_SRC_CTRL_MODE) > 0)

	def get_num_descr_ch(self)->int:
		"""Returns number of state discrimination channels"""
		return NUM_DESCR_CH
	def get_max_pp_nsamp(self)->int:
		"""Returns the maximum number of samples that can be processed by
		pulse processing module"""
		return FEATURE_MEM_DEPTH//NUM_DESCR_CH