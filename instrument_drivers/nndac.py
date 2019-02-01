from qsweepy.instrument import Instrument
import visa
import numpy
import time

class nndac(Instrument):
	def __init__(self, resource):
		self._resource = resource
		self._visainstrument = visa.ResourceManager().open_resource(self._resource)
		self._visainstrument.write_termination = '\n'
		self._visainstrument.read_termination = '\n'
		self._visainstrument.timeout = 10000 
		self.max_abs = 4.094
		self.step = 0.002
		self.cached_voltages = [self.get_voltage(channel=i) for i in range(24)]
		self.use_cache = True
	
	def set_voltage(self,value,channel):
		if numpy.abs(value) < self.max_abs:
			#print ('Channel {}, current voltage: {}, setting: {}'.format(channel_number, self.cached_voltages[channel_number], value))
			try:
				self._visainstrument.ask('VOLT {:d},{:f}'.format(channel,value))
			except:
				time.sleep(1)
				self._visainstrument = visa.ResourceManager().open_resource(self._resource)
				time.sleep(1)
				self._visainstrument.ask('VOLT {:d},{:f}'.format(channel,value))
			self.cached_voltages[channel] = value
			
	def get_voltage(self,channel):
		try:
			return(float(self._visainstrument.ask('VOLT {:d}?'.format(channel))))
		except:
			time.sleep(1)
			self._visainstrument = visa.ResourceManager().open_resource(self._resource)
			time.sleep(1)
			return(float(self._visainstrument.ask('VOLT {:d}?'.format(channel))))

	def get_voltage_cache(self, channel):
		return self.cached_voltages[channel]
		
	def set_voltage_safe(self, value, channel):
		if value == self.get_voltage_cache(channel):
			return
		#print type(self.get_voltage_cache(channel_number))
		while numpy.abs(self.get_voltage_cache(channel) - value)>self.step:
			#if numpy.abs(self.get_voltage_cache(channel_number)) > numpy.abs(value):
			step_abs = self.step*32.
			if step_abs < self.step: 
				step_abs = self.step
			if step_abs > numpy.abs(value - self.get_voltage_cache(channel)):
				step_abs = numpy.abs(value - self.get_voltage_cache(channel))
			step_sign = numpy.sign(value - self.get_voltage_cache(channel))
			self.set_voltage(channel, self.get_voltage_cache(channel)+step_abs*step_sign)
			print ('step_abs: ', step_abs, 'step_sign: ', step_sign)
			time.sleep(0.01)