from qsweepy.instrument import Instrument
import types
import logging
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from qsweepy.instrument_drivers._signal_hound import _signal_hound

import ctypes

def get_signal_hounds():
	serial_numbers, num_devices = _signal_hound.get_serial_number_list()
	devices = [s for s in serial_numbers[:num_devices]]
	return devices

class Signal_Hound_SA(Instrument):
	'''
	This is the python driver for the Signal Hound SA124 spectrum analyzer

	Usage:
	Initialise with
	<name> = instruments.create('<name>', <serial number> )
	
	'''

	def __init__(self, name, serial):
		'''
		Initializes 

		Input:
			name (string)    : name of the instrument
			serial (int) : serial number
		'''
		logging.info(__name__ + ' : Initializing instrument')
		Instrument.__init__(self, name, tags=['physical'])
		self._device = _signal_hound.open_device_by_serial_number(serial_number=serial)
		#self._device = f(serial_number=serial)

		self._serial = serial
		
		# Implement parameters
		
		self.add_parameter('ref', type=float,
			flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
			minval=-80, maxval=_signal_hound.max_ref,
			units='dBm', tags=['sweep']) 
		
		self.add_parameter('reject_if', type=bool,
			flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET, 
			tags=['sweep'])
		
		self.add_parameter('nop', type=int,
			flags=Instrument.FLAG_GET,
			tags=['sweep'])

		self.add_parameter('video_bw', type=float,
			flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
			minval=_signal_hound.min_rbw, maxval=_signal_hound.max_rbw,
			units='Hz', tags=['sweep']) 

		self.add_parameter('res_bw', type=float,
			flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
			minval=_signal_hound.min_rbw, maxval=_signal_hound.max_rbw,
			units='Hz', tags=['sweep']) 

		self.add_parameter('centerfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=_signal_hound.sa124_min_freq, maxval=_signal_hound.sa124_max_freq,
			units='Hz', tags=['sweep'])
			
		self.add_parameter('startfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=_signal_hound.sa124_min_freq, maxval=_signal_hound.sa124_max_freq,
			units='Hz', tags=['sweep'])            
			
		self.add_parameter('stopfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=_signal_hound.sa124_min_freq, maxval=_signal_hound.sa124_max_freq,
			units='Hz', tags=['sweep'])                        
			
		self.add_parameter('span', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=_signal_hound.min_span, maxval=_signal_hound.sa124_max_freq,
			units='Hz', tags=['sweep'])  

		self.add_parameter('averages', type=int,
			flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
			minval=1, maxval=1024, tags=['sweep'])					
	

#		self.add_parameter('zerospan', type=bool,
#			flags=Instrument.FLAG_GETSET)
					
		#Triggering Stuff
#		self.add_parameter('trigger_source', type=str,
#			flags=Instrument.FLAG_GETSET)
		
		
		# Implement functions
		self.add_function('get_freqpoints')
		self.add_function('get_tracedata')
		self.add_function('set_xlim')
		self.add_function('get_xlim')
		#self.add_function('avg_clear')
		#self.add_function('avg_status')
		
		#self._oldspan = self.get_span()
		#self._oldnop = self.get_nop()
		#if self._oldspan==0.002:
		#  self.set_zerospan(True)
		
		self.set_xlim(_signal_hound.sa124_min_freq, _signal_hound.sa124_max_freq)
		self.set_res_bw(_signal_hound.max_rbw)
		self.set_video_bw(_signal_hound.max_rbw)
		self.set_reject_if(True)
		self.set_averages(1)
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		
		self.get_all()
	
	def get_all(self):        
		self.get_nop()
		self.get_centerfreq()
		self.get_startfreq()
		self.get_stopfreq()
		self.get_span()
		#self.get_trigger_source()
		self.get_freqpoints()   
		#self.get_zerospan()
		
	def get_tracedata(self):
		'''
		Get the data of the current trace in mW
		'''
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		
		min = (ctypes.c_float*nop)()
		max = (ctypes.c_float*nop)()
		datamin = np.zeros(nop)
		datamax = np.zeros(nop)
		
		for _ in range(self.averages):
			end = 0
			while end < nop:
				begin, end = _signal_hound.get_partial_sweep_32f(self._device, min, max)
				plt.pause(0.05)
			datamin += 10.**(np.asarray(max, dtype=np.float)/10.)
			datamax += 10.**(np.asarray(min, dtype=np.float)/10.)
			
		datax = np.linspace(start_freq, start_freq+bin_size*(nop-1), nop)
		datamin = datamin/self.averages
		datamax = datamax/self.averages
		
		return [datax, datamin, datamax]
	  
	def get_freqpoints(self, query = False):      
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		return np.linspace(start_freq, start_freq+bin_size*(nop-1), nop)
	  
	def get_points(self):
		return {'Power':[('Frequency',self.get_freqpoints())]}
		
	def get_dtype(self):
		return {'Power':np.float}
		
	def get_opts(self):
		return {'Power': {'log': 10}}
		
	def measure(self):
		data = self.get_tracedata()
		return {'Power':data[1]}

	def set_xlim(self, start, stop):
		logging.debug(__name__ + ' : setting start freq to %s Hz and stop freq to %s Hz' % (start, stop))
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		_signal_hound.config_center_span(self._device, (start+stop)*0.5, stop-start)
		
	def get_xlim(self):
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		start = start_freq
		stop = start_freq + (nop-1)*bin_size
		return start, stop

	def set_detector(self, detector_type):	
		if detector_type == 'rms':
			_signal_hound.config_acquisition(self._device, _signal_hound.average, _signal_hound.log_scale)
			_signal_hound.config_proc_units(self._device, _signal_hound.power_units)
			_signal_hound.config_rbw_shape(self._device, _signal_hound.rbw_shape_cispr)
		else:
			_signal_hound.config_acquisition(self._device, _signal_hound.min_max, _signal_hound.log_scale)
			_signal_hound.config_proc_units(self._device, _signal_hound.bypass)
			#raise(ValueError('QtLab driver only support setting rms detector.'))

	def do_set_ref(self, ref):
		_signal_hound.config_level(self._device, ref)
			
	def do_get_nop(self):
		'''
		Get Number of Points (nop) for sweep

		Input:
			None
		Output:
			nop (int)
		'''
		logging.debug(__name__ + ' : getting Number of Points')
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		return nop 

	def do_set_averages(self, n):
		self.averages = n
	
	def do_set_centerfreq(self,cf):
		'''
		Set the center frequency

		Input:
			cf (float) :Center Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting center frequency to %s' % cf)
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		if nop<2: nop = 2
		start = start_freq
		stop = start_freq + (nop-1)*bin_size
		span = stop-start
		
		_signal_hound.config_center_span(self._device, cf, span)
		
		self.get_startfreq();
		self.get_stopfreq();
		self.get_span();
		
		
	def do_get_centerfreq(self):
		'''
		Get the center frequency

		Input:
			None

		Output:
			cf (float) :Center Frequency in Hz
		'''
		logging.debug(__name__ + ' : getting center frequency')
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		return start_freq+(nop-1)*bin_size/2.
		
	def do_set_span(self,span):
		'''
		Set Span

		Input:
			span (float) : Span in KHz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting span to %s Hz' % span)
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		start = start_freq
		if nop<2: nop = 2
		stop = start_freq + (nop-1)*bin_size
		
		_signal_hound.config_center_span(self._device, (start+stop)*0.5, span)
		
		self.get_startfreq();
		self.get_stopfreq();
		self.get_centerfreq(); 
		
	def do_get_span(self):
		'''
		Get Span
		
		Input:
			None

		Output:
			span (float) : Span in Hz
		'''
		#logging.debug(__name__ + ' : getting center frequency')
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		span = (nop-1)*bin_size #float( self.ask('SENS1:FREQ:SPAN?'))
		return span

	
	def do_set_startfreq(self,val):
		'''
		Set Start frequency

		Input:
			span (float) : Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting start freq to %s Hz' % val)
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		if nop<2: nop = 2
		stop = start_freq + (nop-1)*bin_size 
		new_center = (val+stop)*0.5
		new_span = stop - val
		
		_signal_hound.config_center_span(self._device, new_center, new_span)
		
		self.get_centerfreq();
		self.get_stopfreq();
		self.get_span();
		
		
	def do_get_startfreq(self):
		'''
		Get Start frequency
		
		Input:
			None

		Output:
			span (float) : Start Frequency in Hz
		'''
		logging.debug(__name__ + ' : getting start frequency')
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		return start_freq

	def do_set_stopfreq(self,val):
		'''
		Set STop frequency

		Input:
			val (float) : Stop Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting start freq to %s Hz' % val)
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		if nop<2: nop = 1
		#stop = start_freq + (nop-1)*bin_size 
		new_center = (val+start_freq)*0.5
		new_span = val - start_freq
		
		_signal_hound.config_center_span(self._device, new_center, new_span)
		
		self.get_centerfreq();
		self.get_stopfreq();
		self.get_span();
		
	def do_get_stopfreq(self):
		'''
		Get Stop frequency
		
		Input:
			None

		Output:
			val (float) : Start Frequency in Hz
		'''
		logging.debug(__name__ + ' : getting stop frequency')
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		stop = start_freq + (nop-1)*bin_size 
		return stop   

	def do_set_video_bw(self, video_bw):
		vbw_max = 250e3
		vbw_min = 0.1
		
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		span = (nop-1)*bin_size
		
		res_bw = self.get_res_bw()
		reject_if = self.get_reject_if()
		if not res_bw: 
			res_bw = 250e3
		if video_bw > res_bw:
			video_bw = res_bw
			
		vbw_6MHz_allowed = False
		
		if span > 100e6:
			vbw_min = 6.5e3
		if span > 200e3 and start_freq < 16e6:
			vbw_min = 6.5e3
		if start_freq > 200e6 and span > 200e6:
			vbw_6MHz_allowed = True
			
		if video_bw > vbw_max:
			if vbw_6MHz_allowed and video_bw>3e6:
				video_wb = 6e6
			else:
				video_bw = vbw_max
			
		if video_bw < vbw_min:
			video_bw = vbw_min
		_signal_hound.config_sweep_coupling(self._device, res_bw, video_bw, reject_if)
	
	def do_set_res_bw(self, res_bw):
	
		rbw_max = 250e3
		rbw_min = 0.1
		
		_signal_hound.initiate(self._device, _signal_hound.sweeping, 0)
		nop, start_freq, bin_size = _signal_hound.query_sweep_info(self._device)
		span = (nop-1)*bin_size
		video_bw = self.get_video_bw()
		
		if not video_bw: 
			video_bw = 250e3
		rbw_6MHz_allowed = False
		
		if span > 100e6:
			rbw_min = 6.5e3
		if span > 200e3 and start_freq < 16e6:
			rbw_min = 6.5e3
		if start_freq > 200e6 and span > 200e6:
			rbw_6MHz_allowed = True
			
		if res_bw > rbw_max:
			if rbw_6MHz_allowed and res_bw>3e6:
				res_wb = 6e6
			else:
				res_bw = rbw_max
			
		if res_bw < rbw_min:
			res_bw = rbw_min

		if video_bw > res_bw:
			video_bw = res_bw			
		
		reject_if = self.get_reject_if()
		self.video_bw = video_bw
		_signal_hound.config_sweep_coupling(self._device, res_bw, video_bw, reject_if)

	def do_set_reject_if (self, reject_if):
		video_bw = self.get_video_bw()
		res_bw = self.get_res_bw()
		_signal_hound.config_sweep_coupling(self._device, res_bw, video_bw, reject_if)
		
	# def do_set_trigger_source(self,source):
		# '''
		# Set Trigger Mode

		# Input:
			# source (string) : AUTO | MANual | EXTernal | REMote

		# Output:
			# None
		# '''
		# logging.debug(__name__ + ' : setting trigger source to "%s"' % source)
		# if source.upper() in ['AUTO', 'MAN', 'EXT', 'REM', 'IMM']:
			# self._visainstrument.write('TRIG:SOUR %s' % source.upper())        
		# else:
			# raise ValueError('set_trigger_source(): must be AUTO | MANual | EXTernal | REMote')
	# def do_get_trigger_source(self):
		# '''
		# Get Trigger Mode

		# Input:
			# None

		# Output:
			# source (string) : AUTO | MANual | EXTernal | REMote
		# '''
		# logging.debug(__name__ + ' : getting trigger source')
		# return self._visainstrument.ask('TRIG:SOUR?')        
