import numpy as np
import logging
from .save_pkl import *
from .config import get_config

import traceback

class awg_digital:
	def __init__(self, awg, channel, delay_tolerance=20e-9):#, mixer):
		self.awg = awg
		self.channel = channel
		self.frozen = False
		self.delay = 0
		self.measured_delay = None
		self.delay_tolerance = delay_tolerance
	
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
			self.awg.set_digital(np.roll(waveform, self.delay), channel=self.channel)
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

	def get_physical_devices(self):
		return [self.awg]
		
	def validate_modem_delay_calibration(self, modem, ex_channel_name):
		measured_delay = modem.calibrate_delay(ex_channel_name)
		# if this delay validation has been already been obtained with this calibration, don't save
		calibration_references, calibration_metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
		calibration_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_calibration', references_that=calibration_references, metadata=calibration_metadata)
		validation_metadata = {'measured_delay':measured_delay}
		try:
			validation_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_validation', 
																				references_that={'calibration':calibration_measurement.id}, metadata=validation_metadata)
		except Exception as e:
			print(traceback.format_exc())
			validation_measurement = modem.exdir_db.save(measurement_type='modem_readout_delay_validation', references={'calibration':calibration_measurement.id}, metadata=validation_metadata)
		print ('Validation measurement (delay):', measured_delay)
		assert (abs(measured_delay)<self.delay_tolerance)
		modem.delay_measurement = validation_measurement 
		return measured_delay

	def get_modem_delay_calibration_references(self, modem, ex_channel_name):
		references = {}
		ex_channel = modem.pulse_sequencer.channels[ex_channel_name]
		if (hasattr(ex_channel, 'get_calibration_measurement')):
			references[('channel_calibration', ex_channel_name)] = ex_channel.get_calibration_measurement()
		metadata = {'ex_channel':ex_channel_name}
		return references, metadata
	
	def modem_delay_calibrate(self, modem, ex_channel_name):
		old_delay = self.delay
		self.delay = 0
		self.measured_delay = modem.calibrate_delay(ex_channel_name)
		if modem.exdir_db:
			references,metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
			metadata['measured_delay'] = self.measured_delay
			calibration_measurement = modem.exdir_db.save(measurement_type='modem_readout_delay_calibration', references=references, metadata = metadata)
			#modem.delay_calibration = calibration_measurement
		print ('Calibration measurement (delay):', self.measured_delay)
		self.delay = int(-self.measured_delay*self.get_clock()-10)
			
	def get_modem_delay_calibration(self, modem, ex_channel_name):
		try:
			references, metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
			calibration_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_calibration', references_that=references, metadata=metadata)
			#modem.delay_calibration = calibration_measurement
			self.measured_delay = float(calibration_measurement.metadata['measured_delay'])
			result = self.measured_delay
			self.delay = int(-self.measured_delay*self.get_clock()-10)
		except Exception as e:
			print(str(e), type(e))
			self.modem_delay_calibrate(modem, ex_channel_name)
				# save calibration to database
				
		
		return self.measured_delay