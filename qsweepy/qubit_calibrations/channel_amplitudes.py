from ..ponyfiles.data_structures import *


class channel_amplitudes(MeasurementState):
	def __init__(self, *args, **kwargs):
		if len(args) == 1 and isinstance(args[0], MeasurementState) and not len(kwargs): # copy constructor
			super().__init__(*args, **kwargs)
		else: # otherwise initialize from dict and device
			device = args[0]
			metadata = {channel:str(amplitude) for channel, amplitude in kwargs.items()}
			references = {('channel_calibration', channel):device.awg_channels[channel].get_calibration_measurement()
							for channel in kwargs.keys() if hasattr(device.awg_channels[channel], 'get_calibration_measurement')}
			print ('Requested: ', 'metadata:', metadata, 'references:', references)

			# check if such measurement exists
			try:
				measurement = device.exdir_db.select_measurement(measurement_type='channel_amplitudes', metadata=metadata, references_that=references)
				super().__init__(measurement)
			except:
				#traceback.print_exc()
				super().__init__(measurement_type='channel_amplitudes', sample_name=device.exdir_db.sample_name, metadata=metadata, references=references)
				device.exdir_db.save_measurement(self)
			print ('Obtained: ', 'metadata:', self.metadata, 'references:', self.references)

	def items(self): # just use metadata for now
		for channel, amplitude in self.metadata.items():
			yield channel, complex(amplitude)
