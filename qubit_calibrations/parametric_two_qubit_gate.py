from ..ponyfiles.data_structures import *
from . import channel_amplitudes
import traceback
from .import
from .. import pulses
from .excitation_pulse import get_rect_cos_pulse_sequence

def calibrate_parametric_iswap(device, )

class ParametricTwoQubitGate(MeasurementState):
	def __init__(self, *args, **kwargs):
		self.device = args[0]
		if len(args) == 2 and isinstance(args[1], MeasurementState) and not len(kwargs): # copy constructor
			super().__init__(args[1])
		else: # otherwise initialize from dict and device
			metadata = {'rotation_angle': str(kwargs['rotation_angle']),
						'pulse_type': 'rect',
						'length': str(kwargs['length']),
						'tail_length': str(kwargs['tail_length'])
						'phase_q1': str(kwargs['phase_q1']),
						'phase_q2': str(kwargs['phase_q2']),
						'q1': str(kwargs['q1']),
						'q2': str(kwargs['q2'])}

			if 'calibration_type' in kwargs:
				metadata['calibration_type'] = kwargs['calibration_type']

			references = {'channel_amplitudes': int(kwargs['channel_amplitudes']),
						  'Rabi_rect': int(kwargs['Rabi_rect_measurement'])}

			# check if such measurement exists
			try:
				measurement = self.device.exdir_db.select_measurement(measurement_type='parameteric_two_qubit_gate', metadata=metadata, references_that=references)
				super().__init__(measurement)
			except:
				traceback.print_exc()
				super().__init__(measurement_type='parametric_two_qubit_gate', sample_name=self.device.exdir_db.sample_name, metadata=metadata, references=references)
				self.device.exdir_db.save_measurement(self)

		#inverse_references = {v:k for k,v in self.references.items()}
		#print ('inverse_references in __init__:', inverse_references)
		self.channel_amplitudes = channel_amplitudes.channel_amplitudes(self.device.exdir_db.select_measurement_by_id(self.references['channel_amplitudes']))

	def get_pulse_sequence(self, phase):

		return get_rect_cos_pulse_sequence(device = self.device,
										   channel_amplitudes = self.channel_amplitudes,
										   tail_length = float(self.metadata['tail_length']),
										   length = float(self.metadata['length']),
										   phase = phase)