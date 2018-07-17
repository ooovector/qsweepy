from qsweepy.instrument import Instrument
import visa
import numpy
import time

class nndac(Instrument):
	def __init__(self):
		self._visainstrument = visa.ResourceManager().open_resource('DAC')
		self._visainstrument.write_termination = '\n'
		self._visainstrument.read_termination = '\n'
		self._visainstrument.timeout = 10000 
		self.max_abs = 4.094
		self.step = 0.002
		self.cached_voltages = [self.get_voltage(channel_number=i) for i in range(24)]
		self.use_cache = True
	
	def set_voltage(self,channel_number,value):
		if numpy.abs(value) < self.max_abs:
			#print ('Channel {}, current voltage: {}, setting: {}'.format(channel_number, self.cached_voltages[channel_number], value))
			self._visainstrument.ask('VOLT {:d},{:f}'.format(channel_number,value))
			self.cached_voltages[channel_number] = value
			
			
	def get_voltage(self,channel_number):
		return(float(self._visainstrument.ask('VOLT {:d}?'.format(channel_number))))

	def get_voltage_cache(self, channel_number):
		return self.cached_voltages[channel_number]
		
	def set_voltage_safe(self, channel_number, value):
		if value == self.get_voltage_cache(channel_number):
			return
		#print type(self.get_voltage_cache(channel_number))
		while numpy.abs(self.get_voltage_cache(channel_number) - value)>self.step:
			#if numpy.abs(self.get_voltage_cache(channel_number)) > numpy.abs(value):
			step_abs = self.step*32.
			if step_abs < self.step: 
				step_abs = self.step
			if step_abs > numpy.abs(value - self.get_voltage_cache(channel_number)):
				step_abs = numpy.abs(value - self.get_voltage_cache(channel_number))
			step_sign = numpy.sign(value - self.get_voltage_cache(channel_number))
			self.set_voltage(channel_number, self.get_voltage_cache(channel_number)+step_abs*step_sign)
			print ('step_abs: ', step_abs, 'step_sign: ', step_sign)
			time.sleep(0.01)