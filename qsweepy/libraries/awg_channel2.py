import numpy as np

class awg_channel_carrier:
	def __init__(self, parent, frequency):#, mixer):
		"""
		"""
		self.awg = parent.awg
		self.channel = parent.channel
		self.frequency = frequency
		self.status = 1
		self.parent = parent
		self.parent.carriers.append(self)
		self.waveform = None
	def is_iq(self):
		return False

	def set_frequency(self, frequency):
		self.frequency = frequency
	def get_frequency(self):
		return self.frequency
	def set_offset(self, offset):
		return self.awg.set_offset(offset, channel=self.channel)
	def get_offset(self):
		return self.awg.get_offset(channel=self.channel)
	def set_amplitude(self, amplitude):
		return self.awg.set_amplitude(amplitude, channel=self.channel)
	def get_amplitude(self):
		return self.awg.get_amplitude()
	def get_nop(self):
		return self.awg.get_nop()
	def get_clock(self):
		return self.awg.get_clock()
	def set_nop(self, nop):
		self.awg.set_nop(nop)
	def set_clock(self, clock):
		self.awg.set_clock(clock)
	def set_status(self, status):
		self.status = status

	def get_phase(self, time):
		return 2*np.pi*self.frequency*time

	def get_physical_devices(self):
		return [self.awg]


class awg_channel:
	'''
	Class for a single awg channel. Typically, awg are multi-channel, and set_waveform
	operates with the require channel kwarg. The synchronized multi-awg class
	(pulses from pulses.py) assumes that each 'channel' has
		- set_waveform(waveform ndarray),
		- freeze()
		- unfreeze()
	functions.
	This class takes a parent awg (for example Tektronix AWG5014C) class and a channel number,
	and acts as an interface between pulses and the awg.
	'''
	def __init__(self, awg, channel):#, mixer):
		"""
		"""
		self.awg = awg
		self.channel = channel
		self.status = 1
		self.carriers = []
		self.frozen = False
	def is_iq(self):
		return False

	def set_offset(self, offset):
		return self.awg.set_offset(offset, channel=self.channel)
	def get_offset(self):
		return self.awg.get_offset(channel=self.channel)
	def set_amplitude(self, amplitude):
		return self.awg.set_amplitude(amplitude, channel=self.channel)
	def get_amplitude(self):
		return self.awg.get_amplitude()
	def get_nop(self):
		return self.awg.get_nop()
	def get_clock(self):
		return self.awg.get_clock()
	def set_nop(self, nop):
		self.awg.set_nop(nop)
	def set_clock(self, clock):
		self.awg.set_clock(clock)
	def set_status(self, status):
		self.status = status

	def get_physical_devices(self):
		return [self.awg]

