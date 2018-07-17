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
		self.awg.set_digital(waveform, channel=self.channel)
		
	def freeze(self):
		self.frozen = True
	def unfreeze(self):
		if self.frozen:
			self.frozen = False
			#self.assemble_waveform()