from qsweepy.qubit_calibrations.readout_pulse import get_qubit_readout_pulse, get_uncalibrated_measurer
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse

def print_Rabi_frequencies(device, qubit_id):
	device.exdir_db.get_measurements_db(measurement_type='Rabi_rect', 
										metadata={'qubit_id': qubit_id})
	pass


def calibrate_cr_compensation(device, qubit_id, channels):
	# (1) get rabi with highest Rabi frequency for target and control
	
	pass