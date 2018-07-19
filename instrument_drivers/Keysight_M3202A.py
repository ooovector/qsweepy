from instrument import Instrument
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')

import keysightSD1
import matplotlib.pyplot as plt
import numpy as np
import time

# CREATE AND OPEN MODULE IN
class Keysight_M3202A(Instrument):
	def __init__(self, name, chassis, slot):
	## identify chassis, slot id & so on
		self.mask = 0
		self.module = keysightSD1.SD_AOU()
		
		self.module_id = self.module.openWithSlot("M3202A", chassis, slot)
		self.amplitudes = [0.2]*4
		self.offsets = [0.0]*4
		self.waveform_ids = [None]*4
		for channel_id in range(4):
			self.set_amplitude(0.2, channel_id)
			self.set_offset(0.0, channel_id)
		
		self.master_channel = None
		self.repetition_period = 50000
		self.trigger_source_types = [0]*4
		self.trigger_source_channels = [0]*4
		self.trigger_delays = [0]*4
		self.trigger_behaviours = [0]*4
		self.waveforms = [None]*4
		self.marker_delay = [None]*4
		self.marker_length = [None]*4
	
	def set_waveform(self, waveform, channel):
		self.module.channelWaveShape(channel, keysightSD1.SD_Waveshapes.AOU_AWG);
		wave = keysightSD1.SD_Wave()
		if self.waveform_ids[channel] is None:
			wave.newFromArrayDouble(0, np.zeros((50000,)).tolist()) # WAVE_ANALOG_32
			self.module.waveformLoad(wave, channel)
			wave = keysightSD1.SD_Wave()
			self.waveform_ids[channel] = channel
		waveform_id = self.waveform_ids[channel];

		trigger_source_type = self.trigger_source_types[channel]
		trigger_source_channel = self.trigger_source_channels[channel]
		trigger_delay = self.trigger_delays[channel]
		trigger_behaviour = self.trigger_behaviours[channel]
		self.module.AWGtriggerExternalConfig(channel, trigger_source_channel, trigger_behaviour)
		#trigger = 0
		
		self.module.AWGflush(channel)
		wave = keysightSD1.SD_Wave()
		wave.newFromArrayDouble(0, np.asarray(waveform).tolist()) # WAVE_ANALOG_32
		self.module.waveformReLoad(wave, waveform_id, 0)
		
		self.module.AWGqueueConfig(channel, 1) # inifnite cycles
		#self.module.AWGfromArray(channel, trigger, 0, 0, 0, keysightSD1.SD_WaveformTypes.WAVE_ANALOG, waveform[:32576])
		self.module.AWGqueueWaveform(channel, 
											waveform_id, 
											trigger_source_type,#keysightSD1.SD_TriggerModes.AUTOTRIG, 
											trigger_delay, 
											1, 
											0)
		if self.marker_delay[channel]:
			self.module.AWGqueueMarkerConfig(channel, # nAWG
											2, # each cycle
											1<<channel, # PXI channels
											0, #trigIOmask
											1, #value (0 is low, 1 is high)
											0, #syncmode
											self.marker_length[channel], #length5Tclk 
											self.marker_delay[channel]); #delay5Tclk
		self.module.AWGqueueSyncMode(channel, 1)
		self.waveforms[channel] = waveform
										
		#time.sleep(0.05)
		
	def set_marker(self, delay, length, channel):
		
		self.module.AWGflush(channel)
		self.marker_length[channel] = length
		self.marker_delay[channel] = delay

		trigger_source_type = self.trigger_source_types[channel]
		trigger_source_channel = self.trigger_source_channels[channel]
		trigger_delay = self.trigger_delays[channel]
		if self.waveforms[channel] is not None:		
			self.module.AWGqueueConfig(channel, 1) # inifnite cycles
			#self.module.AWGfromArray(channel, trigger, 0, 0, 0, keysightSD1.SD_WaveformTypes.WAVE_ANALOG, waveform[:32576])
			self.module.AWGqueueWaveform(channel, 
												channel, 
												trigger_source_type,#keysightSD1.SD_TriggerModes.AUTOTRIG, 
												trigger_delay, 
												1, 
												0)
			#(self, nAWG, markerMode, trgPXImask, trgIOmask, value, syncMode, length, delay)
			print(self.marker_length[channel], self.marker_delay[channel])
			self.module.AWGqueueMarkerConfig(channel, # nAWG
											2, # each cycle
											1<<channel, # PXI channels
											0, #trigIOmask
											1, #value (0 is low, 1 is high)
											0, #syncmode
											self.marker_length[channel], #length5Tclk 
											self.marker_delay[channel]); #delay5Tclk
			self.module.AWGqueueSyncMode(channel, 1)
		
	def set_output(self, output, channel):
		if output:
			self.mask = self.mask | (1 << (channel))
		else:
			self.mask = self.mask & (0xFF ^ (1 << (channel)))
		
	def set_trigger_mode(self, mode):
		pass
		
	def set_clock(self):
		print('Set clock not implemented')
		pass
	def get_clock(self):
		return 1e9
		
	def run(self):
		self.module.AWGstartMultiple(self.mask)

	def stop(self):
		self.module.AWGstopMultiple(0xF)
		
	def set_amplitude(self, amplitude, channel):
		self.amplitudes[channel] = amplitude
		self.module.channelAmplitude(channel, amplitude)
	
	def get_amplitude(self, channel):
		return self.amplitudes[channel]
		
	def set_offset(self, offset, channel):
		self.offsets[channel] = offset
		self.module.channelOffset(channel, offset)
	
	def get_offset(self, channel):
		return self.amplitudes[channel]
	
	## TODO: this function is broken
	def do_set_repetition_period(self, repetition_period):
		pass
		
	## TODO: this function is broken
	def get_repetition_period(self):
		return 50000*1e9

	## TODO: this function is broken
	def get_nop(self):
		return 50000

	## TODO: this function is broken
	def set_nop(self, repetition_period):
		pass