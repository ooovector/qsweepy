from instrument import Instrument
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
from _Keysight_M3202A import simple_sync
import keysightSD1
class Keysight_M3202A_Base(Instrument):
	def __init__(self, name, chassis, slot):
	## identify chassis, slot id & so on
		self.mask = 0
		self.module = keysightSD1.SD_AOU()
		self.module_id = self.module.openWithSlot("M3202A", chassis, slot)
		self.amplitudes = [0.2]*4
		self.offsets = [0.0]*4
		self.add_parameter('amplitude_channel_{}', type=float,
							flags=Instrument.FLAG_SOFTGET,
							channels = (1, 4),
						   unit='Volts', minval=-2, maxval=2, channel_prefix='ch%d_')
						   vals=validator.Numbers(-1.5, 1.5))
		self.add_parameter('offset_channel_{}', type=float,
							flags=Instrument.FLAG_SOFTGET,
							channels = (1, 4),
						   unit='Volts', minval=-2, maxval=2, channel_prefix='ch%d_')
						   vals=validator.Numbers(-1.5, 1.5)
		for channel_id in range(4):
			self.set_amplitude(0.2, channel_id)
			self.set_offset(0.0, channel_id)
	def set_output(self, output, channel):
		if output:
			self.mask = self.mask | (1 << (channel))
		else:
			self.mask = self.mask & (0xFF ^ (1 << (channel)))
	def set_trigger_mode(self, mode):
		pass
	def run(self):
		self.module.AWGstartMultiple(self.mask)
	def stop(self):
		self.module.AWGstopMultiple(0xF)
	def do_set_amplitude(self, amplitude, channel):
		return self.set_amplitude(amplitude, channel)
	def set_amplitude(self, amplitude, channel):
		self.amplitudes[channel] = amplitude
		self.module.channelAmplitude(channel, amplitude)
	def do_get_amplitude(self, channel):
		return self.get_amplitude(channel)
	def get_amplitude(self, channel):
		return self.amplitudes[channel]
	def do_set_offset(self, offset, channel):
		return self.set_offset(offset, channel)
	def set_offset(self, offset, channel):
		self.offsets[channel] = offset
		self.module.channelOffset(channel, offset)
	def do_get_offset(self, channel):
		return self.get_offset(channel)
	def get_offset(self, channel):
		return self.amplitudes[channel]

