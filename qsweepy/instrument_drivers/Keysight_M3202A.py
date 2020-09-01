from qsweepy.instrument import Instrument
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1
class Keysight_M3202A_Base(Instrument):
	def __init__(self, name, chassis, slot):
	## identify chassis, slot id & so on
		super().__init__(name, tags=['physical'])
		self.mask = 0
		self.module = keysightSD1.SD_AOU()
		self.module_id = self.module.openWithSlotCompatibility("M3202A", chassis, slot, compatibility=keysightSD1.SD_Compatibility.LEGACY)
		self.amplitudes = [0.2]*4
		self.offsets = [0.0]*4
		self.clock = None
		self.add_parameter('amplitude_channel_{}', type=float,
							flags=Instrument.FLAG_SOFTGET,
							channels = (1, 4),
						    unit='Volts', minval=-2, maxval=2, channel_prefix='ch%d_')
		self.add_parameter('offset_channel_{}', type=float,
							flags=Instrument.FLAG_SOFTGET,
							channels = (1, 4),
						    unit='Volts', minval=-2, maxval=2, channel_prefix='ch%d_')
		for channel_id in range(4):
			self.set_amplitude(0.2, channel_id)
			self.set_offset(0.0, channel_id)
	def set_output(self, output, channel):
		if output:
			self.mask = self.mask | (1 << (channel))
			self.module.channelWaveShape(channel, keysightSD1.SD_Waveshapes.AOU_AWG);
		else:
			self.mask = self.mask & (0xFF ^ (1 << (channel)))
			self.module.channelWaveShape(channel, keysightSD1.SD_Waveshapes.AOU_HIZ);
	def set_trigger_mode(self, mode):
		self.module.triggerIOconfig(mode & 0x1)
		self.module.triggerIOwrite((mode & 0x1)==0, mode & 0x2)
	def set_clock_output(self, output):
		self.module.clockIOconfig(output)
	def set_clock(self, clock):
		self.module.clockSetFrequency(clock)
		self.clock=clock
	def run(self):
		self.module.AWGstartMultiple(self.mask)
	def stop(self):
		self.module.AWGstopMultiple(0xF)
	def do_set_amplitude(self, amplitude, channel):
		return self.set_amplitude(amplitude, channel)
	def set_amplitude(self, amplitude, channel):
		self.stop()
		self.amplitudes[channel] = amplitude
		self.module.channelAmplitude(channel, amplitude)
	def do_get_amplitude(self, channel):
		return self.get_amplitude(channel)
	def get_amplitude(self, channel):
		return self.amplitudes[channel]
	def do_set_offset(self, offset, channel):
		return self.set_offset(offset, channel)
	def set_offset(self, offset, channel):
		self.stop()
		self.offsets[channel] = offset
		self.module.channelOffset(channel, offset)
	def do_get_offset(self, channel):
		return self.get_offsets(channel)
	def get_offset(self, channel):
		return self.offset[channel]
	def get_clock(self):
		return self.clock

from qsweepy.instrument_drivers._Keysight_M3202A.simple_sync import *
