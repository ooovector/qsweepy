from matplotlib import pyplot as plt
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

	def set_offset(self, offset):
		return self.awg.set_offset(offset, channel=self.channel)
	def get_offset(self):
		return self.awg.get_offset(channel=self.channel)
	def set_amplitude(self, amplitude):
		return self.awg.set_amplitude(amplitude, channel=self.channel)
	def get_amplitude(self):
		return self.awg.get_amplitude(amplitude)
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
	def get_waveform(self):
		#if not hasattr(self, 'waveform'):
			#self.waveform = np.zeros(self.get_nop(), dtype=np.complex)
		return self.awg.get_waveform(channel=self.channel)
	def set_waveform(self, waveform):
		#self.waveform = waveform
		#plt.figure(self.channel)
		#plt.plot(waveform)
		self.awg.set_waveform(waveform, channel=self.channel)
	def freeze(self):
		''' not implemented yet. Does nothing -- which is OK, but slightly slow.
		consider replaceing with the logic from awg_iq_multi for faster performance.
		freeze/unfreeze makes sense only together with each other'''
		pass
		#self.parent.freeze()
	def unfreeze(self):
		''' not implemented yet. Does nothing -- which is OK, but slightly slow.
		consider replaceing with the logic from Awg_iq_multi for faster performance.
		freeze/unfreeze makes sense only together with each other'''
		pass
		#self.parent.unfreeze()
	def get_physical_devices(self):
		return [self.awg]
