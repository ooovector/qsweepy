# Anritsu_VNA.py
# hacked by Hannes Rotzinger hannes.rotzinger@kit.edu, 2011
# derived from Anritsu_VNA.py (and whatever this is derived from)
# Pascal Macha <pascalmacha@googlemail.com>, 2010
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from qsweepy.instrument import Instrument
from matplotlib import pyplot as plt
import visa
import types
import logging
from time import sleep
import numpy

class Agilent_N5242A(Instrument):
	'''
	This is the python driver for the Agilent VNA X Vector Network Analyzer

	Usage:
	Initialise with
	<name> = instruments.create('<name>', address='<GPIB address>', reset=<bool>)
	
	'''

	def __init__(self, name, address, channel_index = 1):
		'''
		Initializes 

		Input:
			name (string)	: name of the instrument
			address (string) : GPIB address
		'''
		
		logging.info(__name__ + ' : Initializing instrument')
		Instrument.__init__(self, name, tags=['physical'])

		self._address = address
		self._visainstrument = visa.ResourceManager().open_resource(self._address)# no term_chars for GPIB!!!!!
		
		self._zerospan = False
		self._freqpoints = 0
		self._ci = channel_index 
		self._start = 0
		self._stop = 0
		self._nop = 0

		# Implement parameters
		#Sweep
		self.add_parameter('sweep_mode', type=str,
			flags=Instrument.FLAG_GETSET)
		
		self.add_parameter('nop', type=int,
			flags=Instrument.FLAG_GETSET,
			minval=1, maxval=100000,tags=['sweep'])
		###########
		self.add_parameter('bandwidth', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=1e9,
			units='Hz', tags=['sweep']) 
		#Averaging
		self.add_parameter('average_mode', type=str,
			flags=Instrument.FLAG_GETSET)
		
		self.add_parameter('averages', type=int,
			flags=Instrument.FLAG_GETSET,
			minval=1, maxval=1024, tags=['sweep'])					

		self.add_parameter('average', type=bool,
			flags=Instrument.FLAG_GETSET)   
		 ##########		   
		
		self.add_parameter('frequency', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=0, maxval=20e9,
			units='Hz', tags=['sweep'])
		
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
			
		self.add_parameter('power', type=float,
			flags=Instrument.FLAG_GETSET,
			minval=-95, maxval=30,
			units='dBm', tags=['sweep'])

		self.add_parameter('zerospan', type=bool,
			flags=Instrument.FLAG_GETSET)
			
		self.add_parameter('channel_index', type=int,
			flags=Instrument.FLAG_GETSET)			
					
		#Triggering Stuff
		self.add_parameter('trigger_source', type=str,
			flags=Instrument.FLAG_GETSET)

		self.add_parameter('timeout', type=int,
			flags=Instrument.FLAG_GETSET)	
		
		self.add_parameter('status', type=bool, flags=Instrument.FLAG_GETSET)
		
		# sets the S21 setting in the PNA X
		#Do it in your script please!!
		#self.define_S21()
		#self.set_S21()
		
		# Implement functions
		self.add_function('get_freqpoints')
		self.add_function('get_tracedata')
		self.add_function('get_data')
		self.add_function('init')
		#self.add_function('set_S21')
		self.add_function('set_xlim')
		self.add_function('get_xlim')
		self.add_function('get_sweep_time')
		self.add_function('ask')
		self.add_function('write')
		#self.add_function('set_trigger_source')
		#self.add_function('get_trigger_source')
		#self.add_function('avg_clear')
		#self.add_function('avg_status')
		
		#self._oldspan = self.get_span()
		#self._oldnop = self.get_nop()
		#if self._oldspan==0.002:
		#  self.set_zerospan(True)
		
		self.clear()
		self.select_measurement(1)
		
		self.get_all()
	
	def get_all(self):
		params = {}
		if self.get_sweep_mode() =='CW\n' and self.get_span() == 0.0:
			params ['power'] = self.get_power()
			params['freq'] = self.get_startfreq()
		elif self.get_sweep_mode() == 'LIN\n' :
			params ['nop'] = self.get_nop()
			params ['power'] = self.get_power()
			params['start freq'] = self.get_startfreq()
			params ['stop freq'] = self.get_stopfreq()
			params ['span'] = self.get_span()
			params ['bw'] = self.get_bandwidth()
			if self.get_average():
				params['averages']= self.get_averages()
				params ['average mode'] = self.get_average_mode()
		params ['sweep mode'] = self.get_sweep_mode()
		return params
		#self.get_trigger_source()
		#self.get_average()
		#self.get_freqpoints()   
		#self.get_channel_index()
		#self.get_zerospan()
		
	###
	#Communication with device
	###	
	
	def pre_sweep(self):
		#self.init()
		self.set_trigger_source("MAN")
		self.write("*ESE 1")
		self.set_average_mode("POIN")
	
	def post_sweep(self):
		self.set_trigger_source("IMM")
	
	def init(self):
		self._visainstrument.write("INIT:IMM")
		#if self._zerospan:
		#  self._visainstrument.write('INIT1;*wai')
		#else:
		#  if self.get_average():
		#	for i in range(self.get_averages()):			
		#	  self._visainstrument.write('INIT1;*wai')
		#  else:
		#	  self._visainstrument.write('INIT1;*wai')
			  

	def ask(self, cmd):
	#I want just ask it motherfucka!
		return self._visainstrument.ask(cmd)
		
	def write(self, cmd):
	#I want just write it motherfucka!
		return self._visainstrument.write(cmd)	
	'''
	def set_S21(self, select='S21'):
		
		#calls the defined S21 setting
		
		self._visainstrument.write("DISP:WIND:TRAC:DEL")
		self._visainstrument.write("DISP:WIND:TRAC:FEED 'my_ch1_{0}'".format(select))
		self._visainstrument.write("CALC:PAR:SEL 'my_ch1_{0}'".format(select))
		#self._visainstrument.write("CALC:PAR:SEL 'my_ch1_{0}'".format(select))

	def define_S21(self, select='S21'):
		
		#defines the S21 measurement in the PNA X
		
		resp = self._visainstrument.ask('CALC:PAR:CAT?').strip('"')
		names = resp.split(',')
		for i in range(0, len(names), 2):
			if names[i]=='my_ch1_{0}'.format(select):
				return
				
		self._visainstrument.write( "CALCulate:PARameter:EXT 'my_ch1_{0}','{0}'".format(select))
	'''	
	def clear(self):
		self._visainstrument.write("*CLS")
	
	def select_measurement(self,Mnum):
		#Select Mnum = 1 after default preset
		self._visainstrument.write("CALC:PAR:MNUM {:d}".format(Mnum) )
		self._visainstrument.ask("*OPC?")
	
	def set_measurement(self,Mtype):
		#Mtype = "S11"|"S21"|"S22"|"S12"
		#Select measurement before doing this
		self._visainstrument.write("CALC:PAR:MOD "+Mtype)
		self._visainstrument.ask("*OPC?")
		
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
	
	def do_set_frequency(self,cwf):
		'''
		Set the cw frequency

		Input:
			cwf (float) : CW Frequency in Hz

		Output:
			None
		'''
		#logging.debug(__name__ + ' : setting CW frequency to %s' % cwf)
		self._visainstrument.write('SENS%i:FREQ:CW %f' % (self._ci,cwf))
	
	def do_get_frequency(self):
		'''
		Set the cw frequency

		Input:
			cwf (float) : CW Frequency in Hz

		Output:
			None
		'''
		return self._visainstrument.ask('SENS%i:FREQ:CW?' % (self._ci))
	
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
		
	def get_data(self):
		data = self._visainstrument.query_binary_values("CALCulate:DATA? SDATA", datatype=u'f') 
		data_size = numpy.size(data)
		datareal = numpy.array(data[0:data_size:2])
		dataimag = numpy.array(data[1:data_size:2])
		return datareal+1j*dataimag
		
	def get_tracedata(self, format = 'AmpPha'):
		'''
		Get the data of the current trace

		Input:
			format (string) : 'AmpPha': Amp in dB and Phase, 'RealImag',

		Output:
			'AmpPha':_ Amplitude and Phase
		'''
		#Clear status, initiate measurement
		self.write("*CLS")
		self.init()
		#Set bit in ESR when operation complete
		
		self.write("*OPC")
		#Wait until ready and let plots to handle events (mouse drag and so on)
		while int(self.ask("*ESR?"))==0:
			sleep(0.002)
			
		self._visainstrument.write(':FORMAT REAL,32; FORMat:BORDer SWAP;')
		#data = self._visainstrument.ask_for_values(':FORMAT REAL,32; FORMat:BORDer SWAP;*CLS; CALC:DATA? SDATA;*OPC',format=visa.single) 
		#data = self._visainstrument.ask_for_values(':FORMAT REAL,32;CALC:DATA? SDATA;',format=visa.double) 
		#data = self._visainstrument.ask_for_values('FORM:DATA REAL; FORM:BORD SWAPPED; CALC%i:SEL:DATA:SDAT?'%(self._ci), format = visa.double)	  
		#test
		data = self._visainstrument.query_binary_values("CALCulate:DATA? SDATA", datatype=u'f') 
		data_size = numpy.size(data)
		datareal = numpy.array(data[0:data_size:2])
		dataimag = numpy.array(data[1:data_size:2])
		
		#print datareal,dataimag,len(datareal),len(dataimag)
		if format.upper() == 'REALIMAG':
			if self._zerospan:
				return numpy.mean(datareal), numpy.mean(dataimag)
			else:
				return datareal, dataimag
		elif format.upper() == 'AMPPHA':
			if self._zerospan:
				datareal = numpy.mean(datareal)
				dataimag = numpy.mean(dataimag)
				dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
				datapha = numpy.arctan(dataimag/datareal)
				return dataamp, datapha
			else:
				dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
				datapha = numpy.arctan2(dataimag,datareal)
				#Unwrap
			for i in range(0, len(datapha)-1 ):
				if datapha[i+1]-datapha[i] >= numpy.pi:
					datapha[i+1] = datapha[i+1] - 2.*numpy.pi
				elif datapha[i+1]-datapha[i] <= -numpy.pi:
					datapha[i+1] = datapha[i+1] + 2.*numpy.pi
				
			return dataamp, datapha
		else:
			raise ValueError('get_tracedata(): Format must be AmpPha or RealImag') 
	  
	def get_sweep_time(self):
		"""
		Get the time needed for one sweep
		
		Returns:
			out: float
				time in ms
		"""
		#if self.get_average_mode() != "POIN":
		#	return self.get_averages()*float(self._visainstrument.ask(':SENS%i:SWE:TIME?' %(self._ci)))*1e3
		#else:
		return float(self._visainstrument.ask(':SENS%i:SWE:TIME?' %(self._ci)))*1e3
	###
	# SET and GET functions
	###
	def do_set_timeout(self,timeout):
		self._visainstrument.timeout = timeout 

	def do_get_timeout(self):
		return self._visainstrument.timeout 

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
	
	def do_set_average_mode(self, mode):
		'''
		type = poin | swe
		'''
		self._visainstrument.write("SENS:AVER:MODE "+mode)

	def do_get_average_mode(self):
		return self._visainstrument.ask('SENS:AVER:MODE?')	   

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
	
	def do_set_power(self,pow):
		'''
		Set probe power

		Input:
			pow (float) : Power in dBm

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting power to %s dBm' % pow)
		self._visainstrument.write('SOUR%i:POW1:LEV:IMM:AMPL %.1f' % (self._ci,pow))	
	def do_get_power(self):
		'''
		Get probe power

		Input:
			None

		Output:
			pow (float) : Power in dBm
		'''
		logging.debug(__name__ + ' : getting power')
		return float(self._visainstrument.ask('SOUR%i:POW1:LEV:IMM:AMPL?' % (self._ci)))
		
#Frequency	
	def get_freqpoints(self):
		self._start = self.get_startfreq()
		self._stop = self.get_stopfreq()
		self._nop = self.get_nop()
		self._freqpoints = numpy.linspace(self._start,self._stop,self._nop)
		return self._freqpoints
	
	def get_points(self):
		return {'S-parameter':[('Frequency', self.get_freqpoints(), 'Hz')]}
		
	def get_dtype(self):
		return {'S-parameter':complex}
	
	def get_opts(self):
		return {'S-parameter':{'log': 20}}

	def measure(self):
		data = self.get_tracedata(format='realimag')
		return {'S-parameter':(data[0]+1j*data[1])} 
		
	def set_xlim(self, start, stop):
		logging.debug(__name__ + ' : setting start freq to %s Hz' % start)
		self._visainstrument.write('SENS{:d}:FREQ:SPAN {:e}'.format(self._ci,stop-start)) 
		self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,start))   
		
		logging.debug(__name__ + ' : setting stop freq to %s Hz' % stop)
		self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,stop))  
		
	def get_xlim(self):
		start = self.get_startfreq();
		stop = self.get_stopfreq();
		return start, stop	
	
	def do_set_cw_freq(self,cwf):
		'''
		Set the cw frequency

		Input:
			cwf (float) : CW Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting CW frequency to %s' % cwf)
		self._visainstrument.write('SENS%i:FREQ:CW %f' % (self._ci,cwf))
	
	def do_get_cw_freq(self):
		logging.debug(__name__ + ' : getting CW frequency')
		return  float(self._visainstrument.ask('SENS%i:FREQ:CW?'%(self._ci)))
		
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
		self._cwfreq = float(self._visainstrument.ask('SENS%i:FREQ:CENT?'%(self._ci)))
		return  self._cwfreq
		
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
			self.set_Average(False)
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
			source (string) : AUTO | MANual | IMM)

		Output:
			None
		'''
		logging.debug(__name__ + ' : setting trigger source to "%s"' % source)
		if source.upper() in ["AUTO", "MAN", "IMM"]:
			self._visainstrument.write('TRIG:SOUR %s' % source.upper())		
		else:
			raise ValueError('set_trigger_source(): must be AUTO | MANual | IMM')
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
		

	def do_set_sweep_mode(self, mode):
		logging.debug(__name__ + ' : setting sweep mode to "%s"' % mode)
		if mode.upper() in ["LIN", "LOG", "POW", "CW", "SEGM", "PHASE"]:
			self._visainstrument.write('SENS:SWE:TYPE %s' % mode.upper())
		else:
			raise ValueError('set_sweep_mode(mode): mode must be LIN | LOG | POW | CW | SEGM | PHASE')	
	
	def do_get_sweep_mode(self):
		return self._visainstrument.ask('SENS:SWE:TYPE?')
	
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
		
	def do_set_status(self, status):
		'''
		Set the output status of the instrument

		Input:
			status (string) : 'On' or 'Off'

		Output:
			None
		'''
		logging.debug(__name__ + ' : set status to %s' % status)
		self._visainstrument.write('OUTP %s' % int(status))
		
	def do_get_status(self):
		'''
		Reads the output status from the instrument

		Input:
			None

		Output:
			status (string) : 'On' or 'Off'
		'''
		logging.debug(__name__ + ' : get status')
		stat = self._visainstrument.ask('OUTP?')

		if (stat=='1' or stat == 1 or stat):
		  return True
		elif (stat=='0' or stat == 0 or not stat):
		  return False
		else:
		  raise ValueError('Output status not specified : %s' % stat)
		return
	
	
	def read(self):
		return self._visainstrument.read()
	def write(self,msg):
		return self._visainstrument.write(msg)	
	def ask(self,msg):
		return self._visainstrument.ask(msg)
