from .readout_pulse import get_qubit_readout_pulse, get_uncalibrated_measurer
from .calibrated_readout import get_calibrated_measurer
from ..fitters.exp_sin import exp_sin_fitter
from .channel_amplitudes import channel_amplitudes
import numpy as np
from . import excitation_pulse
import traceback


def Rabi_rect(device, qubit_id, channel_amplitudes, transition='01', lengths=None, *extra_sweep_args, tail_length=0, readout_delay=0,
              pre_pulses=tuple(), repeats=1, measurement_type='Rabi_rect', additional_metadata={}):
    if type(qubit_id) is not list and type(qubit_id) is not tuple:  # if we are working with a single qubit, use uncalibrated measurer
        readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition=transition)
        measurement_name = 'iq'+qubit_id
        qubit_id = [qubit_id]
        exp_sin_fitter_mode = 'sync'
    else: # otherwise use calibrated measurer
        readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)
        measurement_name = 'resultnumbers'
        exp_sin_fitter_mode = 'unsync'

    def set_ex_length(length):
        pre_pulse_sequences = []
        for pulse in pre_pulses:
            if hasattr(pulse, 'get_pulse_sequence'):
                pre_pulse_sequences += pulse.get_pulse_sequence(0.0)
            else:
                pre_pulse_sequences += pulse
        #pre_pulse_sequences = [p for pulse in pre_pulses for p in pulse.get_pulse_sequence(0)]
        ex_pulse_seq = excitation_pulse.get_rect_cos_pulse_sequence(device, channel_amplitudes, tail_length, length, phase=0.)*repeats
        delay_seq = [device.pg.pmulti(readout_delay)]
        readout_trigger_seq = device.trigger_readout_seq
        readout_pulse_seq = readout_pulse.pulse_sequence

        device.pg.set_seq(pre_pulse_sequences+delay_seq+ex_pulse_seq+delay_seq+readout_trigger_seq+readout_pulse_seq)

    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for qubit_id_ in qubit_id}
    references['channel_amplitudes'] = channel_amplitudes.id

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    for pre_pulse_id, pre_pulse in enumerate(pre_pulses):
        if hasattr(pre_pulse, 'id'):
            references.update({('pre_pulse', str(pre_pulse_id)): pre_pulse.id})

    if len(qubit_id)>1:
        arg_id = -2 # the last parameter is resultnumbers, so the time-like argument is -2
    else:
        arg_id = -1
    fitter_arguments = (measurement_name, exp_sin_fitter(mode=exp_sin_fitter_mode), arg_id, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition':transition}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (lengths, set_ex_length, 'Excitation length', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references)

    return measurement


def Rabi_rect_adaptive(device, qubit_id, channel_amplitudes, transition='01', measurement_type='Rabi_rect', pre_pulses=tuple(),
                       repeats=1, tail_length = 0, additional_metadata={}, expected_frequency=None):
    # check if we have fitted Rabi measurements on this qubit-channel combo
    #Rabi_measurements = device.exdir_db.select_measurements_db(measurment_type='Rabi_rect', metadata={'qubit_id':qubit_id}, references={'channel_amplitudes': channel_amplitudes.id})
    #Rabi_fits = [exdir_db.references.this.filename for measurement in Rabi_measurements for references in measurement.reference_two if references.this.measurement_type=='fit_dataset_1d']

    #for fit in Rabi_fits:
    max_scan_length = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_max_scan_length'))
    min_step = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_min_step'))
    scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_scan_points'))
    _range = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_range'))

    if expected_frequency is None:
        lengths = np.arange(0, min_step * scan_points, min_step)
    else:
        num_periods = int(np.round(np.sqrt(scan_points)))
        lengths = np.arange(0, num_periods/expected_frequency, 1/(num_periods*expected_frequency))

    good_fit = False

    print (0, min_step*scan_points, min_step)
    while not (good_fit or np.max(lengths)>max_scan_length):
        measurement = Rabi_rect(device, qubit_id, channel_amplitudes, transition=transition, lengths=lengths,
                                measurement_type=measurement_type, pre_pulses=pre_pulses, repeats=1,
                                tail_length=tail_length, additional_metadata=additional_metadata)
        fit_results = measurement.fit.metadata
        if int(fit_results['frequency_goodness_test']):
            return measurement
        lengths *= _range

    raise ValueError('Failed measuring Rabi frequency for qubit {} on channel_amplitudes {}'.format(qubit_id, channel_amplitudes.metadata))


def calibrate_all_single_channel_Rabi(device, _qubit_id=None, transition='01', remove_bad=False):
    if _qubit_id is None:
        _qubit_id = device.get_qubit_list()
    elif type(_qubit_id) is int:
        _qubit_id = [_qubit_id]

    for qubit_id in _qubit_id:
        amplitude_default = float(device.get_qubit_constant(qubit_id=qubit_id, name='amplitude_default'))
        qubit_channel_calibrated = {}
        for channel_name, device_name in device.get_qubit_excitation_channel_list(qubit_id,
                    transition=transition).items():
            ch = channel_amplitudes(device, **{channel_name: amplitude_default})
            try:
                excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi/2., transition=transition, channel_amplitudes_override=ch)
                qubit_channel_calibrated[channel_name] = device_name
            except Exception as e:
                print ('Failed to Rabi-calibrate channel ', channel_name)
                traceback.print_exc()
                if remove_bad:
                    print ('Removing from channel list!')
        if remove_bad:
            device.set_qubit_excitation_channel_list(qubit_id, qubit_channel_calibrated)
