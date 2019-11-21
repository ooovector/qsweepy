import numpy as np
class DummyAWG:
	'''
	Has the pythonic interfaces of an AWG, but doesn't have an actual piece of hardware on the other end.
	'''
	def __init__(self, channels):#, mixer):
		"""
		"""
		self.channel = channels
		self.status = 1
		self.carriers = []
		self.frozen = False
		self.amplitude = np.zeros(channels)
		self.offset = np.zeros(channels)
		self.nop = 0
		self.waveform = np.zeros((channels, self.nop))

	def set_offset(self, offset, channel):
		self.offset[channel] = offset
	def get_offset(self, channel):
		return self.offset[channel]
	def set_amplitude(self, amplitude):
		self.amplitude = amplitude
	def get_amplitude(self):
		return self.amplitude
	def get_nop(self):
		return self.nop
	def get_clock(self):
		return self.clock
	def set_nop(self, nop):
		self.nop = nop
	def set_clock(self, clock):
		self.clock = clock
	def set_status(self, status):
		self.status = status
	def get_waveform(self, channel):
		return self.waveform[channel, :]
	#	#if not hasattr(self, 'waveform'):
	#		#self.waveform = np.zeros(self.get_nop(), dtype=np.complex)
	#	return self.awg.get_waveform(channel=self.channel)
	def set_waveform(self, waveform, channel):
		self.waveform[channel, :] = waveform
	#	#plt.figure(self.channel)
	#	#plt.plot(waveform)
	#	self.awg.set_waveform(waveform, channel=self.channel)
	def freeze(self):
		''' not implemented yet. Does nothing -- which is OK, but slightly slow.
		consider replaceing with the logic from awg_iq_multi for faster performance.
		freeze/unfreeze makes sense only together with each other'''
		#pass
		#self.parent.freeze()
		self.frozen = True
	def unfreeze(self):
		''' not implemented yet. Does nothing -- which is OK, but slightly slow.
		consider replaceing with the logic from Awg_iq_multi for faster performance.
		freeze/unfreeze makes sense only together with each other'''
		#pass
		#self.parent.unfreeze()
		self.frozen = False
	def get_physical_devices(self):
		return []
