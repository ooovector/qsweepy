import visa
import logging
import numpy

class rsvna():
	'''
	This is the python driver for Rohde&Schwarz vector network analyzers (written specifically for ZNA26)
	'''

	def __init__(self, name, address, channel_index =1):
		'''
		Initializes 

		Input:
			name (string)	: name of the instrument
			address (string) : GPIB address
		'''
		
		logging.info(__name__ + ' : Initializing instrument')

		self._address = address
		self._visainstrument = visa.ResourceManager().open_resource(self._address, timeout=5000)
		self._freqpoints = 0
		self._ci = channel_index 
		self._start = 0
		self._stop = 0
		self._nop = 0
		
	def ask(self, cmd):
		#I want just ask it motherfucka!
		return self._visainstrument.ask(cmd)

	def query(self, cmd):
		#I want just ask it motherfucka!
		return self._visainstrument.query(cmd)
		
	def write(self, cmd):
		#I want just write it motherfucka!
		return self._visainstrument.write(cmd)	
		
	def create_trace(self, channel, name, meas_parameter):
		'''
	Like in manual control, traces can be assigned to a channel and 
	displayed in diagram areas (see section Traces, Channels and 
	Diagram Areas in Chapter 3). 
	There are two main differences between manual and remote control:

    A trace can be created without being displayed on the screen.
    A channel must not necessarily contain a trace. 
	Channel and trace configurations are independent of each other.  

	Create new trace and new channel (if channel <Ch> does not exist yet)
	CALCulate<Ch>:PARameter:SDEFine '<Trace Name>','< Meas Parameter>
		'''
		return self._visainstrument.write("CALCulate{}:PARameter:SDEFine '{}','{}'".format(
					channel, name, meas_parameter))
	
	def get_data(self):
	
		data = self._visainstrument.query_ascii_values("CALCulate:DATA? SDATA") 
		data_size = numpy.size(data)
		datareal = numpy.array(data[0:data_size:2])
		dataimag = numpy.array(data[1:data_size:2])
		return datareal+1j*dataimag
	
	def set_auto_loss_delay(self, input_port):
		self.measure()
		self.write('SENSe%i:CORRection:LOSS%i:AUTO ONCE'%(self._ci, input_port))
	
	def setup_single_tone_spectroscopy(self, **kwargs):
		## input arguments:
		## output_port, input_port, freq_start, freq_stop, power, nop, channel=1
		
		# turn off all rf sources except for output_port
		self.write('INITiate1:CONTinuous OFF ')
		
		for port in [1,2,3,4]:
			state = 'ON' if port == kwargs['output_port'] else 'OFF'
			self.write(':SOURce{channel}:POWer{port}:STATe {state}'.format(state=state, port=port, **kwargs))
		# set output power
		self.write(':SOURce{channel}:POWer{output_port} {power}'.format(**kwargs)) #Base Power
		self.write(':SOURce{channel}:POWer{output_port}:OFFSet 0,CPADd'.format(**kwargs))# power offset
		# set sweep range
		self.write(':SENSe{channel}:FREQ:STAR {freq_start}'.format(**kwargs))
		self.write(':SENSe{channel}:FREQ:STOP {freq_stop}'.format(**kwargs))
		# set number of points in sweep
		self.write(':SENSe{channel}:SWEep:POINts {nop}'.format(**kwargs))
		# set sweep mode on output port with f=fb*1+0 and sweep mode
		self.write(':SOURce:FREQuency{output_port}:CONVersion:ARBitrary:IFRequency 1,1,0,SWEep'.format(**kwargs))
		self.write(':SOURce:FREQuency{input_port}:CONVersion:ARBitrary:IFRequency 1,1,0,SWEep'.format(**kwargs))
		# create trace for S_{input_port}{output_port}
		self.write(':CALCULATE{channel}:PARAMETER:SDEFINE "STSpectrum","B{input_port}/A{output_port}"'.format(**kwargs))
		self.write(':DISPLAY:WINDOW1:TRACE1:FEED "STSpectrum"')			
		
		#bandwidth
		self.write('SENSe{channel}:BANDwidth:RESolution {bandwidth}'.format(**kwargs))
		#trigger
		self.write('SENSe{channel}:SWEep:TIME:AUTO ON'.format(**kwargs))
		
		if 'set_auto_loss_delay' in kwargs:
			if kwargs['set_auto_loss_delay']:
				self.set_auto_loss_delay(kwargs['input_port'])
		
	def delete_single_tone_spectroscopy(self):
		self.write('CALCulate%i:PARameter:DELete "STSpectrum"'%(self._ci,))
		self.write('INITiate%i:CONTinuous ON'%(self._ci,))
	
	def setup_two_tone_spectroscopy(self, **kwargs):
		## input arguments:
		## output_port_probe, output_port_pump, input_port, freq_start_pump, freq_stop_pump, freq_probe
		## power_probe, power_pump, nop, channel=1
		
		# turn off all rf sources except for output_port
		self.write('INITiate{channel}:CONTinuous OFF '.format(**kwargs))
		
		for port in [1,2,3,4]:
			state = 'ON' if port == kwargs['output_port_pump'] or port == kwargs['output_port_probe'] else 'OFF'
			self.write(':SOURce{channel}:POWer{port}:STATe {state}'.format(state=state, port=port, **kwargs))
			
		self.write(':SOURce{channel}:POWer{output_port_pump}:PERManent:STATe ON'.format(**kwargs))
		# set output power
		self.write(':SOURce{channel}:POWer{output_port_pump} {power_pump}'.format(**kwargs)) #Base Power
		self.write(':SOURce{channel}:POWer{output_port_pump}:OFFSet 0,CPADd'.format(**kwargs))# power offset
		self.write(':SOURce{channel}:POWer{output_port_probe}:OFFSet {power_probe},ONLY'.format(**kwargs))# power offset
		# set sweep range
		self.write(':SENSe{channel}:FREQ:STAR {freq_start_pump}'.format(**kwargs))
		self.write(':SENSe{channel}:FREQ:STOP {freq_stop_pump}'.format(**kwargs))
		# set number of points in sweep
		self.write(':SENSe{channel}:SWEep:POINts {nop}'.format(**kwargs))
		# set sweep mode on output port with f=fb*1+0 and sweep mode
		self.write(':SOURce:FREQuency{output_port_pump}:CONVersion:ARBitrary:IFRequency 1,1,0,SWEep'.format(**kwargs))
		self.write(':SOURce:FREQuency{output_port_probe}:CONVersion:ARBitrary:IFRequency 0,1,{freq_probe},CW'.format(**kwargs))
		self.write(':SOURce:FREQuency{input_port}:CONVersion:ARBitrary:IFRequency 0,1,{freq_probe},CW'.format(**kwargs))

		# create trace for S_{input_port}{output_port}
		self.write(':CALCULATE{channel}:PARAMETER:SDEFINE "STSpectrum","B{input_port}/A{output_port_probe}"'.format(**kwargs))
		self.write(':DISPLAY:WINDOW1:TRACE1:FEED "TTSpectrum"')
		
		#turn off delay because it only makes sense for single-tone
		self.write(':SENSe{channel}:CORRection:EDELay{input_port}:TIME 0'.format(**kwargs))
		
		#bandwidth
		self.write('SENSe{channel}:BANDwidth:RESolution {bandwidth}'.format(**kwargs))
		#trigger
		self.write('SENSe{channel}:SWEep:TIME:AUTO ON'.format(**kwargs))
	
	def delete_two_tone_spectroscopy(self):
		#delete trace
		self.write('CALCulate%i:PARameter:DELete "TTSpectrum"'%(self._ci,))
		self.write('INITiate%i:CONTinuous ON'%(self._ci,))
	
	def set_power(self, power):
		self.write(':SOURce%i:POWer %f'%(self._ci, power)) #Base Power
	
	def get_power(self):
		return float(self.ask(':SOURce%i:POWer?'%(self._ci)))
	
	def get_span(self):
		'''
		Get Span
		
		Input:
			None

		Output:
			span (float) : Span in Hz
		'''
		span = self._visainstrument.ask('SENS%i:FREQ:SPAN?' % (self._ci) ) #float( self.ask('SENS1:FREQ:SPAN?'))
		return span
	
	def set_startfreq(self, val):
		'''
		Set Start frequency

		Input:
			span (float) : Frequency in Hz

		Output:
			None
		'''
		self._visainstrument.write(':SENS%i:FREQ:STAR %f' % (self._ci, val))   
		self._start = val
		self.get_centerfreq();
		self.get_stopfreq();
		self.get_span();
	
	def get_startfreq(self):
		'''
		Get Start frequency
		
		Input:
			None

		Output:
			span (float) : Start Frequency in Hz
		'''
		self._start = float(self._visainstrument.ask(':SENS%i:FREQ:STAR?' % (self._ci)))
		return  self._start
	
	def get_centerfreq(self):
		'''
		Get the center frequency

		Input:
			None

		Output:
			cf (float) :Center Frequency in Hz
		'''
		self._cwfreq = float(self._visainstrument.ask(':SENS%i:FREQ:CENT?'%(self._ci)))
		return  self._cwfreq
	
	def set_stopfreq(self, val):
		'''
		Set STop frequency

		Input:
			val (float) : Stop Frequency in Hz

		Output:
			None
		'''
		self._visainstrument.write(':SENS%i:FREQ:STOP %f' % (self._ci, val))  
		self._stop = val
		self.get_startfreq();
		self.get_centerfreq();
		self.get_span();
	
	def measure(self):
		sweep_time = float(self.ask('SENSe1:SWEep:TIME?'))
		
		self.write('TRIGger%i:SEQuence:SOURce IMMediate'%(self._ci,))
		self.write('INITiate%i:IMMediate'%(self._ci,))
		
		old_timeout = self._visainstrument.timeout
		self._visainstrument.timeout = sweep_time*1000+5000
		self.ask('*OPC?')
		self._visainstrument.timeout = old_timeout
	
		return {'S-parameter': self.get_data()}
		
	
	def get_stopfreq(self):
		'''
		Get Stop frequency
		
		Input:
			None

		Output:
			val (float) : Start Frequency in Hz
		'''
		self._stop = float(self._visainstrument.ask(':SENS%i:FREQ:STOP?' %(self._ci) ))
		return  self._stop
	
	
	def get_freqpoints(self):
		self._start = self.get_startfreq()
		self._stop = self.get_stopfreq()
		self._nop = self.get_nop()
		self._freqpoints = numpy.linspace(self._start,self._stop,self._nop)
		return self._freqpoints
	
	def get_nop(self):
		self._nop = int(self._visainstrument.ask(':SENSe%i:SWEep:POINts?'%self._ci))
		return self._nop
	
	def get_points(self):
		return {'S-parameter':[('Frequency', self.get_freqpoints(), 'Hz')]}
		
	def get_dtype(self):
		return {'S-parameter':complex}
	
	def get_opts(self):
		return {'S-parameter':{'log': 20}}