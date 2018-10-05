import platform
import os
import platform
import sys 
import logging
from ctypes import *
import numpy

from qsweepy.instrument import Instrument

# load errors for easier access
from qsweepy.instrument_drivers._Spectrum_M4i22xx.spcerr import * 
# load registers for easier access
from qsweepy.instrument_drivers._Spectrum_M4i22xx.regs import *


class Spectrum_M4i22xx(Instrument):
	def __init__(self,name):
		logging.info(__name__ + ' : Initializing instrument Spectrum')
		Instrument.__init__(self, name, tags=['physical'])
		self._card_is_open = False
		self._load_dll()
		self._open()
		
	def _load_dll(self):
		oPlatform = platform.architecture()
		if (oPlatform[0] == '64bit'):
			bIs64Bit = 1
		else:
			bIs64Bit = 0
		#sys.stdout.write("Python Version: {0} on Windows\n\n".format (platform.python_version()))
		int8  = c_int8
		int16 = c_int16
		int32 = c_int32
		int64 = c_int64

		ptr8  = POINTER (int8)
		ptr16 = POINTER (int16)
		ptr32 = POINTER (int32)
		ptr64 = POINTER (int64)

		uint8  = c_uint8
		uint16 = c_uint16
		uint32 = c_uint32
		uint64 = c_uint64

		uptr8  = POINTER (uint8)
		uptr16 = POINTER (uint16)
		uptr32 = POINTER (uint32)
		uptr64 = POINTER (uint64)

		# define card handle type
		if (bIs64Bit):
			# for unknown reasons c_void_p gets messed up on Win7/64bit, but this works:
			drv_handle = POINTER(c_uint64)
		else:
			drv_handle = c_void_p

		# Load DLL into memory.
		# use windll because all driver access functions use _stdcall calling convention under windows
		if (bIs64Bit == 1):
			self._spcm_win32 = windll.LoadLibrary ("c:\\windows\\system32\\spcm_win64.dll")
		else:
			self._spcm_win32 = windll.LoadLibrary ("c:\\windows\\system32\\spcm_win32.dll")

		# load spcm_hOpen
		if (bIs64Bit):
			self._spcm_win32.open = getattr (self._spcm_win32, "spcm_hOpen")
		else:
			self._spcm_win32.open = getattr (self._spcm_win32, "_spcm_hOpen@4")
		self._spcm_win32.open.argtype = [c_char_p]
		self._spcm_win32.open.restype = drv_handle 

		# load spcm_vClose
		if (bIs64Bit):
			self._spcm_win32.close = getattr (self._spcm_win32, "spcm_vClose")
		else:
			self._spcm_win32.close = getattr (self._spcm_win32, "_spcm_vClose@4")
		self._spcm_win32.close.argtype = [drv_handle]
		self._spcm_win32.close.restype = None

		# load spcm_dwGetErrorInfo
		if (bIs64Bit):
			self._spcm_win32.GetErrorInfo = getattr (self._spcm_win32, "spcm_dwGetErrorInfo_i32")
		else:
			self._spcm_win32.GetErrorInfo = getattr (self._spcm_win32, "_spcm_dwGetErrorInfo_i32@16")
		self._spcm_win32.GetErrorInfo.argtype = [drv_handle, uptr32, ptr32, c_char_p]
		self._spcm_win32.GetErrorInfo.restype = uint32

		# load spcm_dwGetParam_i32
		if (bIs64Bit):
			self._spcm_win32.GetParam32 = getattr (self._spcm_win32, "spcm_dwGetParam_i32")
		else:
			self._spcm_win32.GetParam32 = getattr (self._spcm_win32, "_spcm_dwGetParam_i32@12")
		self._spcm_win32.GetParam32.argtype = [drv_handle, int32, ptr32]
		self._spcm_win32.GetParam32.restype = uint32

		# load spcm_dwGetParam_i64
		if (bIs64Bit):
			self._spcm_win32.GetParam64 = getattr (self._spcm_win32, "spcm_dwGetParam_i64")
		else:
			self._spcm_win32.GetParam64 = getattr (self._spcm_win32, "_spcm_dwGetParam_i64@12")
		self._spcm_win32.GetParam64.argtype = [drv_handle, int32, ptr64]
		self._spcm_win32.GetParam64.restype = uint32

		# load spcm_dwSetParam_i32
		if (bIs64Bit):
			self._spcm_win32.SetParam32 = getattr (self._spcm_win32, "spcm_dwSetParam_i32")
		else:
			self._spcm_win32.SetParam32 = getattr (self._spcm_win32, "_spcm_dwSetParam_i32@12")
		self._spcm_win32.SetParam32.argtype = [drv_handle, int32, int32]
		self._spcm_win32.SetParam32.restype = uint32

		# load spcm_dwSetParam_i64
		if (bIs64Bit):
			self._spcm_win32.SetParam64 = getattr (self._spcm_win32, "spcm_dwSetParam_i64")
		else:
			self._spcm_win32.SetParam64 = getattr (self._spcm_win32, "_spcm_dwSetParam_i64@16")
		self._spcm_win32.SetParam64.argtype = [drv_handle, int32, int64]
		self._spcm_win32.SetParam64.restype = uint32

		# load spcm_dwSetParam_i64m
		if (bIs64Bit):
			spcm_dwSetParam_i64m = getattr (self._spcm_win32, "spcm_dwSetParam_i64m")
		else:
			spcm_dwSetParam_i64m = getattr (self._spcm_win32, "_spcm_dwSetParam_i64m@16")
		spcm_dwSetParam_i64m.argtype = [drv_handle, int32, int32, int32]
		spcm_dwSetParam_i64m.restype = uint32

		# load spcm_dwDefTransfer_i64
		if (bIs64Bit):
			self._spcm_win32.DefTransfer64 = getattr (self._spcm_win32, "spcm_dwDefTransfer_i64")
		else:
			self._spcm_win32.DefTransfer64 = getattr (self._spcm_win32, "_spcm_dwDefTransfer_i64@36")
		self._spcm_win32.DefTransfer64.argtype = [drv_handle, uint32, uint32, uint32, c_void_p, uint64, uint64]
		self._spcm_win32.DefTransfer64.restype = uint32

		# load spcm_dwInvalidateBuf
		if (bIs64Bit):
			self._spcm_win32.InValidateBuf = getattr (self._spcm_win32, "spcm_dwInvalidateBuf")
		else:
			self._spcm_win32.InValidateBuf = getattr (self._spcm_win32, "_spcm_dwInvalidateBuf@8")
		self._spcm_win32.InValidateBuf.argtype = [drv_handle, uint32]
		self._spcm_win32.InValidateBuf.restype = uint32
		
##########################Basic PC-device communication options########################################
	def _open(self):
		logging.debug(__name__ + ' : Try to open card')
		if ( not self._card_is_open):
		  self._spcm_win32.handel = self._spcm_win32.open('spcm0')
		  self._card_is_open = True
		else:
		  logging.warning(__name__ + ' : Card is already open !')

		if (self._spcm_win32.handel==0):
			logging.error(__name__ + ' : Unable to open card')
			self._card_is_open = False
	
	def _close(self):
		logging.debug(__name__ + ' : Try to close card')
		self._spcm_win32.close(self._spcm_win32.handel)
		
	def _set_param(self, regnum, regval):
		logging.debug(__name__ + ' : Set reg %s to %s' %(regnum, regval))
		err = self._spcm_win32.SetParam32(self._spcm_win32.handel, regnum, regval)
		if (err==0):
			return err
		elif (err == 263):
			logging.error(__name__ + ' : Timeout')
			return 263
		else:
			logging.error(__name__ + ' : Error %s while setting reg %s to %s' % (err, regnum, regval))
			self._get_error()
			raise ValueError('Error communicating with device')

	def _get_param(self, regnum):
		logging.debug(__name__ + ' : Reading Reg %s' %(regnum))
		val = c_int()
		p_antw = pointer(val)

		err = self._spcm_win32.GetParam32(self._spcm_win32.handel, regnum, p_antw)
		if (err==0):
			return p_antw.contents.value
		else:
			logging.error(__name__ + ' : Error %s while getting reg %s' %(err,regnum))
			self._get_error()
			raise ValueError('Error communicating with device')
			
	def _get_error(self):
		# try to read out error
		logging.debug(__name__ + ' : Reading error')
		j = (c_char * 200)()
		e1 = c_int()
		e2 = c_int()
		p_errortekst = pointer(j)
		p_er1 = pointer(e1)
		p_er2 = pointer(e2)

		self._spcm_win32.GetErrorInfo(self._spcm_win32.handel, p_er1, p_er2, p_errortekst)

		tekst = ""

		for ii in range(200):
			tekst  = tekst + p_errortekst.contents[ii].decode()
		logging.error(__name__ + ' : ' + tekst)
		return tekst

###########################Clock options########################################		
	def set_clock(self, rate):
		logging.debug(__name__+ ' : set clock value')
		self._set_param(SPC_SAMPLERATE, int(rate))
	
	def get_clock(self):
		logging.debug(__name__+ ' : get clock value')
		return(self._get_param(SPC_SAMPLERATE))
	
	def set_clock_mode(self, mode):
		logging.debug(__name__+ ' : set clock mode') 	
		if mode == 'int':
			self._set_param(SPC_CLOCKMODE, SPC_CM_INTPLL)
		elif mode == 'ext':
			self._set_param(SPC_CLOCKMODE, SPC_CM_EXTREFCLOCK)
		elif mode == 'pxi':
			self._set_param(SPC_CLOCKMODE, SPC_CM_PXIREFCLOCK)
		else: 
			pass
	
	def get_clock_mode(self):
		logging.debug(__name__+ ' : get clock mode')
		if self._get_param(SPC_CLOCKMODE)&SPC_CM_INTPLL == 1:
			mode = 'int'
		elif self._get_param(SPC_CLOCKMODE)&SPC_CM_EXTREFCLOCK:
			mode = 'ext'
		elif self._get_param(SPC_CLOCKMODE)&SPC_CM_PXIREFCLOCK:
			mode = 'pxi'
		return (mode)
		
	def set_ext_clock(self, rate):
		logging.debug(__name__+ ' : set external clock frequency')
		self._set_param(SPC_REFERENCECLOCK, int(rate))
		
	def get_ext_clock(self):
		logging.debug(__name__+ ' : get external clock frequency')
		return (self._get_param(SPC_REFERENCECLOCK))
		
##########################Channels options########################################
	def select_channel01(self):
		logging.debug(__name__+ ' : turn on both 0 and 1 channels')
		self._set_param(SPC_CHENABLE, CHANNEL0 | CHANNEL1)
		
	def set_channel01_amps(self, amp0, amp1):
		logging.debug(__name__+ ' : set channels 0 and 1 amplitudes')
		self._set_param(SPC_AMP0, amp0)
		self._set_param(SPC_AMP1, amp1)
	
	def set_channel01_offset(self, offset0, offset1):
		logging.debug(__name__+ ' : set channels 0 and 1 offsets')
		self._set_param(SPC_OFFS0, offset0)
		self._set_param(SPC_OFFS1, offset1)
		
	def set_channel01_coupling(self, coupl0, coupl1):
		logging.debug(__name__+ ' : set channels 0 and 1 coupling mode')
		self._set_param(SPC_ACDC0, coupl0)
		self._set_param(SPC_ACDC1, coupl1)
		
	def set_aliasing_filter_status(self, status):
		logging.debug(__name__+ ' : set aliasing filter status for all channels')
		self._set_param(SPC_FILTER0, status)
		
###########################Trigger options###########################################
	def set_ext_trigger_mode(self,mode):
		logging.debug(__name__+ ' : set external triggers mode')
		self._set_param(SPC_TRIG_EXT0_MODE, mode)
		self._set_param(SPC_TRIG_EXT1_MODE, mode)
		
	def set_ext_trigger(self):
		logging.debug(__name__+ ' :set trigger be externalhard')
		self._set_param(SPC_TRIG_ORMASK,SPC_TMASK_EXT0 | SPC_TMASK_EXT1)
	
	def set_soft_trigger(self):
		logging.debug(__name__+ ' :set trigger be soft')
		self._set_param(SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)
		
	def set_trigger_termination(self, term):
		logging.debug(__name__+ ' :set trigger termination')
		self._set_param(SPC_TRIG_TERM, term)
	
	def set_trigger_delay(self, delay):
		logging.debug(__name__+ ' :set trigger dellay')
		time = delay/self._get_param(SPC_SAMPLERATE)
		self._set_param(SPC_TRIG_DELAY, int(time))
		
	def set_posttrigger(self, posttrigger):
		logging.debug(__name__+ ' :posttrigger value')
		self._set_param(SPC_POSTTRIGGER, posttrigger)

##########################Acquisistion options and state control###########################
	def set_multi_record_mode(self):
		logging.debug(__name__+ ' :set multi record mode')
		self._set_param(SPC_CARDMODE, SPC_REC_STD_MULTI)
		
	def set_timeout(self, time):
		logging.debug(__name__+ ' :set timeout value')
		self._set_param(SPC_TIMEOUT,time)
		
	def get_timeout(self):
		logging.debug(__name__+ ' :get timeout value')
		return(self._get_param(SPC_TIMEOUT))
	
	def reset(self):	
		logging.debug(__name__+ ' : Reset card')
		self._set_param(SPC_M2CMD, M2CMD_CARD_RESET)
		
	def start(self):
		logging.debug(__name__+ ' :Start the board and waiting trigger')
		self._set_param(SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_CARD_WAITREADY)
		
	def stop(self):
		logging.debug(__name__+ ' :Stop runing the board')
		self._set_P=param(SPC_M2CMD, M2CMD_CARD_STOP)
		
	def set_memsize(self, lMemsize):
		logging.debug(__name__+ ' :Set memsize')
		self._set_param(SPC_MEMSIZE, lMemsize)
	
	def get_memsize(self):
		logging.debug(__name__+ ' :Get memsize')
		return (self._get_param(SPC_MEMSIZE))
	
	def set_nop(self, nop):
		logging.debug(__name__+ ' :Set nop')
		self._set_param(SPC_SEGMENTSIZE, nop)
		
	def get_nop(self):
		logging.debug(__name__+ ' :Get nop')
		return(self._get_param(SPC_SEGMENTSIZE))
	
	def set_nums(self, nums):
		logging.debug(__name__+ ' :Set nums')
		self.nums = nums
		self.set_memsize(int(self.get_nop()*self.nums))
		
	def get_nums(self):
		logging.debug(__name__+ ' :Get nums')
		return (self.get_memsize()/self.get_nop())
				
##########################Buffer and data readout options######################################
	def _buffer_setup(self):
		logging.debug(__name__+ ' :setup buffer')
		lMemsize = self._get_param(SPC_MEMSIZE)
		numchannels = self._get_param(SPC_CHCOUNT)
		lBufsize = lMemsize*numchannels
		
		a = (c_int8 * lBufsize)()
		p_data = pointer(a)
		
		err = self._spcm_win32.DefTransfer64(self._spcm_win32.handel, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, 0, p_data, c_int64(0), c_int64(lBufsize))
		if (err!=0):
			logging.error(__name__ + ' : Error setting up buffer')
			self._get_error()
			raise ValueError('Error communicating with device')
			
		err = self._spcm_win32.SetParam32(self._spcm_win32.handel, SPC_M2CMD, M2CMD_DATA_STARTDMA)
		if (err!=0):
			logging.error(__name__ + ' : Error starting DMA transfer, error nr: %i' % err)
			self._get_error()
			raise ValueError('Error communicating with device')
			
		self._pbuffer = p_data
	
	def readout_raw_buffer(self, nr_of_channels = 1):
		logging.debug(__name__ + ' : Readout raw buffer')

		# wait for end of data transfer
		err = self._spcm_win32.SetParam32(self._spcm_win32.handel, SPC_M2CMD, M2CMD_DATA_WAITDMA)
		if (err!=0):
			logging.error(__name__ + ' : Error during read, error nr: %i' % err)
			self._get_error()
			raise ValueError('Error communicating with device')

		return self._pbuffer.contents 
	
	def get_data(self):
	
		lMemsize = int(self.get_memsize())
		lSegsize = int(self.get_nop())
		
		amp0 = float(self._get_param(SPC_AMP0))
		offset0 = float(self._get_param(SPC_OFFS0))
		amp1 = float(self._get_param(SPC_AMP1))
		offset1 = float(self._get_param(SPC_OFFS1))

		lnumber_of_segments = int(lMemsize / lSegsize)

		data = self.readout_raw_buffer(nr_of_channels=2)
		if data == 'timeout':
			return data
			
		data = numpy.array(data, numpy.int8)
		data = numpy.reshape(data, (lMemsize, 2))
		data0 = data[:,0]
		data1 = data[:,1]
		data0 = numpy.reshape(data0, (lnumber_of_segments, lSegsize))
		data1 = numpy.reshape(data1, (lnumber_of_segments, lSegsize))
		data0 = 2.0 * amp0 * (data0 / 255.0) + offset0
		data1 = 2.0 * amp1 * (data1 / 255.0) + offset1
		
		return (data0, data1)

		
	def invalidate_buffer(self, buffertype = SPCM_BUF_DATA):
		logging.debug(__name__+ ' : Invalidating buffer')
		
		# stop any running DMA transfers
		err = self._spcm_win32.SetParam32(self._spcm_win32.handel, SPC_M2CMD, M2CMD_DATA_STOPDMA)
		if (err!=0):
			logging.error(__name__+ ' : Error %s while setting reg %s to %s' % (err, regnum, regval))
			self._get_error()
			raise ValueError('Error communicating with device')
			
		# invalidate buffer
		err = self._spcm_win32.InValidateBuf(self._spcm_win32.handel, buffertype)
		if (err==0):
			return 0
		else:
			logging.error(__name__+ ' : Error %s while setting reg %s to %s' % (err, regnum, regval))
			self._get_error()
			raise ValueError('Error communicating with device')

		del self._pbuffer
		
	def buffer_setup(self):
		return self._buffer_setup()
		
	
	
		
	
	
	
		
		
	
		
	
		
		
	