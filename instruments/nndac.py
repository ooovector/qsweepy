from instrument import Instrument

import visa
import numpy

class nndac(Instrument):
	def __init__(self):
		self._visainstrument = visa.ResourceManager().open_resource('DAC')
		self._visainstrument.write_termination = '\n'
		self._visainstrument.read_termination = '\n'
		self._visainstrument.timeout = 10000 
		self.step = 0.002
	
	def set_voltage(self,channel_number,value):
		self._visainstrument.ask('VOLT {:d},{:f}'.format(channel_number,value))
	def get_voltage(self,channel_number):
		return(self._visainstrument.ask('VOLT {:d}?'.format(channel_number)))

		
		
		