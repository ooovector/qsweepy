from qsweepy.instrument import Instrument
import visa
import types
import logging
from time import sleep
import numpy
from matplotlib import pyplot as plt

class Agilent_N9030A(Instrument):
	'''
	This is the python driver for the Agilent_N9030A

	Usage:
	Initialise with
	<name> = instruments.create('<name>', address='<GPIB address>', reset=<bool>)
	
	'''

	def __init__(self, name, address, channel_index = 1):
		'''
		Initializes 

		Input:
			name (string)    : name of the instrument
			address (string) : GPIB address
		'''
		
		logging.info(__name__ + ' : Initializing instrument')
		Instrument.__init__(self, name, tags=['physical'])

		self._address = address
		self._visainstrument = visa.ResourceManager().open_resource(self._address)# no term_chars for GPIB!!!!!
		self._visainstrument.timeout = 400000
		self._zerospan = False
		self._freqpoints = 0
		self._ci = channel_index 
		self._start = 0
		self._stop = 0
		self._nop = 0

		# Implement parameters
		
		self.add_parameter('nop', type=int,
			flags=Instrument.FLAG_GETSET,
			minval=1, maxval=100000,
			tags=['sweep'])
		
		self.add_parameter('bandwidth', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=1e9,
			units='Hz', tags=['sweep']) 

		self.add_parameter('bandwidth_video', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=1e9,
			units='Hz', tags=['sweep']) 

		self.add_parameter('video_bw', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=1e9,
			units='Hz', tags=['sweep']) 

		self.add_parameter('res_bw', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=1e9,
			units='Hz', tags=['sweep']) 

		self.add_parameter('averages', type=int,
			flags=Instrument.FLAG_GETSET,
			minval=1, maxval=1024, tags=['sweep'])                    

		self.add_parameter('average', type=bool,
			flags=Instrument.FLAG_GETSET)   
					
		self.add_parameter('centerfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=20e9,
			units='Hz', tags=['sweep'])
			
		self.add_parameter('startfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=20e9,
			units='Hz', tags=['sweep'])            
			
		self.add_parameter('stopfreq', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=20e9,
			units='Hz', tags=['sweep'])                        
			
		self.add_parameter('span', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=20e9,
			units='Hz', tags=['sweep'])        

		self.add_parameter('zerospan', type=bool,
			flags=Instrument.FLAG_GETSET)
			
		self.add_parameter('channel_index', type=int,
			flags=Instrument.FLAG_GETSET)            
					
		#Triggering Stuff
		self.add_parameter('trigger_source', type=str,
			flags=Instrument.FLAG_GETSET)
		
		
		# Implement functions
		self.add_function('get_freqpoints')
		self.add_function('get_tracedata')
		self.add_function('init')
		self.add_function('set_xlim')
		self.add_function('get_xlim')
		self.add_function('get_sweep_time')
		self.add_function('set_sweep_time')
		self.add_function('set_sweep_time_auto')
		self.add_function('get_sweep_time_auto')
		#self.add_function('avg_clear')
		#self.add_function('avg_status')
		
		#self._oldspan = self.get_span()
		#self._oldnop = self.get_nop()
		#if self._oldspan==0.002:
		#  self.set_zerospan(True)
		
		self.get_all()
	
	def get_all(self):        
		self.get_nop()
		self.get_centerfreq()
		self.get_startfreq()
		self.get_stopfreq()
		self.get_span()
		self.get_bandwidth()
		self.get_trigger_source()
		self.get_average()
		self.get_averages()
		self.get_freqpoints()   
		self.get_channel_index()
		#self.get_zerospan()
		
	###
	#Communication with device
	###	
	
	def init(self):
		if self._zerospan:
			self._visainstrument.write('INIT1;*wai')
		else:
			if self.get_average():
				for i in range(self.get_averages()):
					self._visainstrument.write('INIT1;*wai')
			else:
				self._visainstrument.write('INIT1;*wai')
		
	def reset_windows(self):
		self._visainstrument.write('DISP:WIND Off')
		self._visainstrument.write('DISP:WIND On')
	
	def set_autoscale(self):
		self._visainstrument.write("DISP:WIND:TRAC:Y:AUTO")
		
	def set_continous(self,ON=True):
		if ON:
			self._visainstrument.write( "INITiate:CONTinuous ON")
		else:
			self._visainstrument.write( "INITiate:CONTinuous Off")
	
	def get_sweep(self):
		self._visainstrument.write( "ABORT; INITiate:IMMediate;*wai")
		
	def avg_clear(self):
		self._visainstrument.write(':SENS%i:AVER:CLE' %(self._ci))

	def avg_status(self):
		# this does not work the same way than the VNA:
		#return int(self._visainstrument.ask(':SENS%i:AVER:COUN?' %(self._ci))
		pass
		
	def get_avg_status(self):
		return self._visainstrument.ask('STAT:OPER:AVER1:COND?')
			
	def still_avg(self): 
		if int(self.get_avg_status()) == 1: return True
		else: return False 
		
	def get_tracedata(self, format = 'AmpPha'):
		'''
		Get the data of the current trace

		Input:
			format (string) : 'AmpPha': Amp in dB and Phase, 'RealImag',

		Output:
			'AmpPha':_ Amplitude and Phase
		'''
		#Clear status, initiate measurement
		self.set_trigger_source("MAN")
		self.write("*ESE 1")
		self.write("*CLS")
		self.init()
			#Set bit in ESR when operation complete
			
		self.write("*OPC")
		#Wait until ready and let plots to handle events (mouse drag and so on)
		while int(self.ask("*ESR?"))==0:
			plt.pause(0.05)
		
		self._visainstrument.write(':FORMAT REAL,32; FORMat:BORDer SWAP;')
		#data = self._visainstrument.ask_for_values(':FORMAT REAL,32; FORMat:BORDer SWAP;*CLS; CALC:DATA? SDATA;*OPC',format=visa.single) 
		#data = self._visainstrument.ask_for_values(':FORMAT REAL,32;CALC:DATA? SDATA;',format=visa.double) 
		#data = self._visainstrument.ask_for_values('FORM:DATA REAL; FORM:BORD SWAPPED; CALC%i:SEL:DATA:SDAT?'%(self._ci), format = visa.double)      
		#test
		data = self._visainstrument.query_binary_values( "CALCulate:DATA?",datatype=u'f') 
		data_size = numpy.size(data)
		datax = numpy.array(data[0:data_size:2])
		datay = numpy.array(data[1:data_size:2])
		return [datax, datay]
			
		#data_size = numpy.size(data)
		
		#print datareal,dataimag,len(datareal),len(dataimag)
		#if format.upper() == 'REALIMAG':
		#  if self._zerospan:
		#    return numpy.mean(datareal), numpy.mean(dataimag)
		#  else:
		#    return datareal, dataimag
		#elif format.upper() == 'AMPPHA':
		#  if self._zerospan:
		#    datareal = numpy.mean(datareal)
		#    dataimag = numpy.mean(dataimag)
		#    dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
		#    datapha = numpy.arctan(dataimag/datareal)
		#    return dataamp, datapha
		#  else:
		#    dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
		#    datapha = numpy.arctan2(dataimag,datareal)
		#    return dataamp, datapha
		#else:
		#  raise ValueError('get_tracedata(): Format must be AmpPha or RealImag') 
	  
	def get_freqpoints(self, query = False):      
	  #if query == True:        
		#self._freqpoints = numpy.array(self._visainstrument.ask_for_values('SENS%i:FREQ:DATA:SDAT?'%self._ci,format=1)) / 1e9
		#self._freqpoints = numpy.array(self._visainstrument.ask_for_values(':FORMAT REAL,32;*CLS;CALC1:DATA:STIM?;*OPC',format=1)) / 1e9
	  self._freqpoints = numpy.linspace(self._start,self._stop,self._nop)
	  return self._freqpoints
	  
	def get_points(self):
		return {'Power':[('Frequency',self.get_freqpoints(),'Hz')]}
		
	def get_dtype(self):
		return {'Power':numpy.float}
		
	def get_opts(self):
		return {'Power':{'log': None}}
		
	def measure(self):
		return {'Power':self.get_tracedata()[1]}

	def set_xlim(self, start, stop):
		logging.debug(__name__ + ' : setting start freq to %s Hz' % start)
		self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,start))   
		self._start = start
		self.get_centerfreq();
		self.get_stopfreq();
		self.get_span();
		
		logging.debug(__name__ + ' : setting stop freq to %s Hz' % stop)
		self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,stop))  
		self._stop = stop
		self.get_startfreq();
		self.get_centerfreq();
		self.get_span();
		
	def get_xlim(self):
		return self._start, self._stop

	def set_detector(self, detector_type):	
		if detector_type == 'rms':
			self._visainstrument.write('DET:TRAC AVER')
			self._visainstrument.write('AVER:TYPE RMS')
		else:
			self._visainstrument.write('DET:TRAC {0}'.format(detector_type))
			#raise(ValueError('QtLab driver only support setting rms detector.'))
	
	def get_sweep_time(self):
		"""
		Get the time needed for one sweep
		
		Returns:
			out: float
				time in ms
		"""
		return float(self._visainstrument.ask(':SENS%i:SWE:TIME?' %(self._ci)))*1e3
		
	def set_sweep_time(self, t):
		self._visainstrument.write( ':SENS{:d}:SWE:TIME {:e}'.format(self._ci, t) )
		
	def set_sweep_time_auto(self, val):
		if val:
			self._visainstrument.write(":SWE:TIME:AUTO 1")
		else:
			self._visainstrument.write(":SWE:TIME:AUTO 0")
			
	def get_sweep_time_auto(self):
		return int(self._visainstrument.ask(":SWE:TIME:AUTO?"))
		
	# SET and GET functions
	###
	
	def do_set_nop(self, nop):
		'''
		Set Number of Points (nop) for sweep

		Input:
			nop (int) : Number of Points

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting Number of Points to %s ' % (nop))
		if self._zerospan:
		  print ('in zerospan mode, nop is 1')
		else:
		  self._visainstrument.write(':SENS%i:SWE:POIN %i' %(self._ci,nop))
		  self._nop = nop
		  self.get_freqpoints() #Update List of frequency points  
		
	def do_get_nop(self):
		'''
		Get Number of Points (nop) for sweep

		Input:
			None
		Output:
			nop (int)
		'''
		logging.debug(__name__ + ' : getting Number of Points')
		if self._zerospan:
		  return 1
		else:
			self._nop = int(self._visainstrument.ask(':SENS%i:SWE:POIN?' %(self._ci)))    
		return self._nop 
	
	def do_set_average(self, status):
		'''
		Set status of Average

		Input:
			status (string) : 'on' or 'off'

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting Average to "%s"' % (status))
		if status:
			status = 'ON'
			self._visainstrument.write('SENS%i:AVER:STAT %s' % (self._ci,status))
		elif status == False:
			status = 'OFF'
			self._visainstrument.write('SENS%i:AVER:STAT %s' % (self._ci,status))
		else:
			raise ValueError('set_Average(): can only set on or off')               
	def do_get_average(self):
		'''
		Get status of Average

		Input:
			None

		Output:
			Status of Averaging ('on' or 'off) (string)
		'''
		logging.debug(__name__ + ' : getting average status')
		return bool(int(self._visainstrument.ask('SENS%i:AVER:STAT?' %(self._ci))))
					
	def do_set_averages(self, av):
		'''
		Set number of averages

		Input:
			av (int) : Number of averages

		Output:
			None
		'''
		if self._zerospan == False:
			logging.debug(__name__ + ' : setting Number of averages to %i ' % (av))
			self._visainstrument.write('SENS%i:AVER:COUN %i' % (self._ci,av))
		else:
			self._visainstrument.write('SWE:POIN %.1f' % (self._ci,av))
			
	def do_get_averages(self):
		'''
		Get number of averages

		Input:
			None
		Output:
			number of averages
		'''
		logging.debug(__name__ + ' : getting Number of Averages')
		if self._zerospan:
		  return int(self._visainstrument.ask('SWE%i:POIN?' % self._ci))
		else:
		  return int(self._visainstrument.ask('SENS%i:AVER:COUN?' % self._ci))
				
	def do_set_centerfreq(self,cf):
		'''
		Set the center frequency

		Input:
			cf (float) :Center Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting center frequency to %s' % cf)
		self._visainstrument.write('SENS%i:FREQ:CENT %f' % (self._ci,cf))
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
		return  float(self._visainstrument.ask('SENS%i:FREQ:CENT?'%(self._ci)))
		
	def do_set_span(self,span):
		'''
		Set Span

		Input:
			span (float) : Span in KHz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting span to %s Hz' % span)
		self._visainstrument.write('SENS%i:FREQ:SPAN %i' % (self._ci,span))   
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
		span = self._visainstrument.ask('SENS%i:FREQ:SPAN?' % (self._ci) ) #float( self.ask('SENS1:FREQ:SPAN?'))
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
		self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,val))   
		self._start = val
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
		self._start = float(self._visainstrument.ask('SENS%i:FREQ:STAR?' % (self._ci)))
		return  self._start

	def do_set_stopfreq(self,val):
		'''
		Set STop frequency

		Input:
			val (float) : Stop Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting stop freq to %s Hz' % val)
		self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,val))  
		self._stop = val
		self.get_startfreq();
		self.get_centerfreq();
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
		self._stop = float(self._visainstrument.ask('SENS%i:FREQ:STOP?' %(self._ci) ))
		return  self._stop   
	def do_set_bandwidth_video(self,band):
		'''
		Set Bandwidth

		Input:
			band (float) : Bandwidth in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting video bandwidth to %s Hz' % (band))
		self._visainstrument.write('SENS%i:BWID:VID %i' % (self._ci,band))

	def do_get_bandwidth_video(self):
		'''
		Get Bandwidth

		Input:
			None

		Output:
			band (float) : Bandwidth in Hz
		'''
		logging.debug(__name__ + ' : getting video bandwidth')
		# getting value from instrument
		return  float(self._visainstrument.ask('SENS%i:BWID:VID?'%self._ci))           

	def do_set_video_bw(self, video_bw):
		self.set_bandwidth_video(video_bw)
	def do_get_video_bw(self, video_bw):
		return self.get_bandwidth_video()
	def do_set_res_bw(self, video_bw):
		self.set_bandwidth(video_bw)
	def do_get_res_bw(self):
		return self.get_bandwidth()

	def do_set_bandwidth(self,band):
		'''
		Set Bandwidth

		Input:
			band (float) : Bandwidth in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting bandwidth to %s Hz' % (band))
		self._visainstrument.write('SENS%i:BWID:RES %i' % (self._ci,band))
	def do_get_bandwidth(self):
		'''
		Get Bandwidth

		Input:
			None

		Output:
			band (float) : Bandwidth in Hz
		'''
		logging.debug(__name__ + ' : getting bandwidth')
		# getting value from instrument
		return  float(self._visainstrument.ask('SENS%i:BWID:RES?'%self._ci))                

	def do_set_zerospan(self,val):
		'''
		Zerospan is a virtual "zerospan" mode. In Zerospan physical span is set to
		the minimal possible value (2Hz) and "averages" number of points is set.

		Input:
			val (bool) : True or False

		Output:
			None
		'''
		#logging.debug(__name__ + ' : setting status to "%s"' % status)
		if val not in [True, False]:
			raise ValueError('set_zerospan(): can only set True or False')        
		if val:
			self._oldnop = self.get_nop()
			self._oldspan = self.get_span()
			if self.get_span() > 0.002:
				Warning('Setting ZVL span to 2Hz for zerospan mode')
				self.set_span(0.002)
			
		av = self.get_averages()
		self._zerospan = val
		if val:
			self.set_average(False)
			self.set_averages(av)
			if av<2:
			  av = 2
		else: 
		  self.set_Average(True)
		  self.set_span(self._oldspan)
		  self.set_nop(self._oldnop)
		  self.get_averages()
		self.get_nop()
			   
	def do_get_zerospan(self):
		'''
		Check weather the virtual zerospan mode is turned on

		Input:
			None

		Output:
			val (bool) : True or False
		'''
		return self._zerospan


	def do_set_trigger_source(self,source):
		'''
		Set Trigger Mode

		Input:
			source (string) : AUTO | MANual | EXTernal | REMote

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting trigger source to "%s"' % source)
		if source.upper() in ['AUTO', 'MAN', 'EXT', 'REM', 'IMM']:
			self._visainstrument.write('TRIG:SOUR %s' % source.upper())        
		else:
			raise ValueError('set_trigger_source(): must be AUTO | MANual | EXTernal | REMote')
	def do_get_trigger_source(self):
		'''
		Get Trigger Mode

		Input:
			None

		Output:
			source (string) : AUTO | MANual | EXTernal | REMote
		'''
		logging.debug(__name__ + ' : getting trigger source')
		return self._visainstrument.ask('TRIG:SOUR?')        
		

	def do_set_channel_index(self,val):
		'''
		Set the index of the channel to address.

		Input:
			val (int) : 1 .. number of active channels (max 16)

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting channel index to "%i"' % int)
		nop = self._visainstrument.read('DISP:COUN?')
		if val < nop:
			self._ci = val 
		else:
			raise ValueError('set_channel_index(): index must be < nop channels')
	def do_get_channel_index(self):
		'''
		Get active channel

		Input:
			None

		Output:
			channel_index (int) : 1-16
		'''
		logging.debug(__name__ + ' : getting channel index')
		return self._ci
		
	def read(self):
		return self._visainstrument.read()
	def write(self,msg):
		return self._visainstrument.write(msg)    
	def ask(self,msg):
		return self._visainstrument.ask(msg)
