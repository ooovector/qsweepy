from qsweepy.instrument import Instrument
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')

import keysightSD1
import matplotlib.pyplot as plt
import numpy
import logging

# CREATE AND OPEN MODULE IN
class Keysight_M3102A(Instrument):
	def __init__(self, name, chassis, slot_in):
		logging.info(__name__ + ' : Initializing instrument Spectrum')
		Instrument.__init__(self, name, tags=['physical'])
		self.moduleIn = keysightSD1.SD_AIN()
		self.moduleInID = self.moduleIn.openWithSlot("M3102A", chassis, slot_in)
		self.mask = 0
		self.trigger_delay = 0
		self.software_nums_multi = 1
		self.software_averages = 1
		self.nop = 256
		self.nums = 1
		self.trigger = keysightSD1.SD_TriggerModes.EXTTRIG
		self.moduleIn.triggerIOconfig(1)
		self.timeout = 10000
		self.impedances = [0]*5
		self.couplings = [0]*5
		self.fullscale = [0.0625]*5
		for ch in range(4): self.set_channelinputconfig(ch+1);
			
		
	def set_channelinputconfig(self, channel):
		print(self.moduleIn.channelInputConfig(channel, self.fullscale[channel], self.impedances[channel], self.couplings[channel]))

	def set_coupling(self, coupling, channel): # 0 for hiZ, 1 for 50Ohms
		self.couplings[channel] = coupling
		self.set_channelinputconfig(channel)

	def set_fullscale(self, fullscale, channel): # 0 for hiZ, 1 for 50Ohms
		self.fullscale[channel] = fullscale
		self.set_channelinputconfig(channel)
		
	def set_impedance(self, impedance, channel): # 0 for hiZ, 1 for 50Ohms
		self.impedances[channel] = impedance
		self.set_channelinputconfig(channel)
	
	def _daqconfig(self, channel):
		#print (channel, self.nop, self.nums, self.trigger_delay, self.trigger)
		self.moduleIn.DAQconfig(channel, self.nop, self.nums, self.trigger_delay, self.trigger)
	
	def set_trigger_external(self, channel, trigger_source='ext'):
		self.trigger = keysightSD1.SD_TriggerModes.EXTTRIG
		# digitalTriggerMode = keysightSD1.SD_TriggerModes.HWDIGTRIG
		# if trigger_source == 'ext'
			# digitalTriggerSource = keysightSD1.SD_TriggerModes.TRIG_EXTERNAL
		# else:
			# digitalTriggerSource = keysightSD1.SD_TriggerModes.TRIG_PXI
		self.moduleIn.triggerIOconfig(1)
		# self.moduleIn.DAQtriggerConfig(channel, digitalTriggerMode, digitalTriggerSource, analogTriggerMask)
		self.moduleIn.DAQdigitalTriggerConfig(channel, trigger_source, keysightSD1.SD_TriggerBehaviors.TRIGGER_RISE)
		# self.moduleIn.DAQtriggerExternalConfig(channel, externalSource, triggerBehavior, sync = SD_SyncModes.SYNC_NONE)
		self._daqconfig(channel)
		
	def set_trigger_delay(self, delay):
		self.trigger_delay = delay
		for c in range(1, 5):
			self._daqconfig(c)
	
	def get_clock(self):
		return 0.5e9
	
	def set_clock(self, clock):
		print('set_clock not implemented')
		
	def set_nop(self, nop):
		self.nop = nop
		for c in range(1, 5):
			self._daqconfig(c)
			
	def get_nop(self):
		return self.nop
		
	def set_nums(self, nums):
		self.nums = nums
		for c in range(1, 5):
			self._daqconfig(c)
			
	def get_nums(self):
		return self.nums
	
	def set_input(self, input, channel):
		if input:
			self.mask = self.mask | (1 << (channel-1))
		else:
			self.mask = self.mask & (0xFF ^ (1 << (channel-1)))
	
	def get_points(self):
		return {'Voltage':[('Sample',numpy.arange(self.get_nums()*self.software_nums_multi), ''), 
							('Time',numpy.arange(self.get_nop())/self.get_clock(), 's')]}
		
	def get_dtype(self):
		return {'Voltage':complex}
	
	def get_opts(self):
		return {'Voltage':{'log': None}}

	def get_channel_number(self):
		return sum([(self.mask & 1<<i) > 0 for i in range(4)])
		
	def measure(self):
		data = numpy.zeros((self.get_channel_number(), self.get_nums()*self.software_nums_multi, self.get_nop()), dtype=numpy.float)

		for i in range(self.software_averages):
			for j in range(self.software_nums_multi):
				self.moduleIn.DAQstartMultiple(self.mask)
				for channel in range(0,self.get_channel_number()):
					#self.start_with_trigger_and_waitready()
					read = self.moduleIn.DAQread(channel+1, self.get_nums()*self.get_nop(), self.timeout)
					data[channel,j*self.get_nums():(j+1)*self.get_nums(),:] += numpy.reshape(read, (self.get_nums(), self.get_nop()))/float(self.software_averages)
					#self.stop()
				self.moduleIn.DAQstopMultiple(self.mask)
		return {'Voltage':(data[0,:,:]+1j*data[1,:,:])}