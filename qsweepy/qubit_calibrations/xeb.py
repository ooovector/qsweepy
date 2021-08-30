from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np


def xeb_single_qubit_rotation(device, qubit_id):
    from .readout_pulse2 import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse

    qubit_readout_pulse, readout_device = get_uncalibrated_measurer(device, qubit_id)
    rotation = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi / 2.)

    random_phase_gats = []
    # TODO: define random phase gates for single-qubit XEBs