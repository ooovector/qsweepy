import numpy as np
import logging
from .save_pkl import *
from .config import get_config

class awg_digital:
	def __init__(self, awg, channel):#, mixer):
		self.awg = awg
		self.channel = channel
		self.frozen = False
	
	def get_nop(self):
		return self.awg.get_nop()
	
	def get_clock(self):
		return self.awg.get_clock()
	
	def set_nop(self, nop):
		return self.awg.set_nop(nop)
	
	def set_clock(self, clock):
		return self.awg.set_clock(clock)
	
	def set_waveform(self, waveform):
		if self.mode == 'waveform':
			self.awg.set_digital(waveform, channel=self.channel)
		if self.mode == 'marker':
			delay_tock = np.where(waveform)[0][0]
			delay = int(np.ceil(delay_tock / 10));
			length_tock = np.where(1-waveform[delay_tock:])[0][0]
			length = int(np.ceil(length_tock/10))
			self.awg.set_marker(delay, length, channel=self.channel)
		if self.mode == 'set_delay':
			delay_tock = np.where(waveform)[0][0]
			delay = int(np.ceil(delay_tock));
			self.delay_setter(delay)
		
	def freeze(self):
		self.frozen = True
	def unfreeze(self):
		if self.frozen:
			self.frozen = False
			#self.assemble_waveform()
