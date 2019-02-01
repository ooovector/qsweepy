from  qsweepy.instrument_drivers.Keysight_M3202A import *
import numpy as np

# CREATE AND OPEN MODULE IN
class Keysight_M3202A_HVISync(Keysight_M3202A_Base):
	def __init__(self, name, chassis, slot):
		super().__init__(name, chassis, slot)
		#self.master_channel = None
		#self.repetition_period = 50000
		#self.trigger_source_types = [0]*4
		#self.trigger_source_channels = [0]*4
		#self.trigger_delays = [0]*4
		#self.trigger_behaviours = [0]*4
		self.waveforms = [None]*4
		self.waveform_ids = [None]*4
		self.marker_delay = [None]*4
		self.marker_length = [None]*4
	
	### infinite cycles of a single waveform mode with syncronisation across channels
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
			self.module.AWGqueueConfig(channel, 1) # infnite cycles
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
		
	## TODO: this function is broken
	def do_set_repetition_period(self, repetition_period):
		pass
	## TODO: this function is broken
	def get_repetition_period(self):
		return 50000*1e9
	## TODO: this function is broken
	def set_nop(self, nop):
		pass
	def get_nop(self):
		return 50000
	def set_trigger_mode(self, mode):
		pass		
	def set_clock(self):
		print('Set clock not implemented')
		pass
	def get_clock(self):
		return 1e9
	## TODO: this function is broken
	def set_nop(self, repetition_period):
		pass
		