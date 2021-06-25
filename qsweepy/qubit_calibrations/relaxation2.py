from qsweepy.qubit_calibrations import excitation_pulse
import numpy as np

def relaxation(device, qubit_id, transition='01', *extra_sweep_args, channel_amplitudes=None, lengths=None,
           readout_delay=0, delay_seq_generator=None, measurement_type='decay', ex_pulse=None,
           additional_references = {}, additional_metadata = {}):
    from .readout_pulse import get_uncalibrated_measurer
    from ..fitters.exp import exp_fitter
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition)
    if ex_pulse is None:
        ex_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi, channel_amplitudes_override=channel_amplitudes)

    def set_delay(length):
        #ex_pulse_seq = [device.pg.pmulti(length+2*tail_length, *tuple(channel_pulses))]
        if delay_seq_generator is None:
            delay_seq = [device.pg.pmulti(length)]
        else:
            delay_seq = delay_seq_generator(length)
        readout_delay_seq = [device.pg.pmulti(readout_delay)]
        readout_trigger_seq = device.trigger_readout_seq
        readout_pulse_seq = readout_pulse.pulse_sequence

        device.pg.set_seq(device.pre_pulses+
                          ex_pulse.get_pulse_sequence(0)+
                          delay_seq+
                          readout_delay_seq+
                          readout_trigger_seq+
                          readout_pulse_seq)

    references = {'ex_pulse':ex_pulse.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+qubit_id, exp_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': qubit_id,
                'transition': transition,
              'extra_sweep_args':str(len(extra_sweep_args)),
              'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                              *extra_sweep_args,
                                              (lengths, set_delay, 'Delay','s'),
                                              fitter_arguments = fitter_arguments,
                                              measurement_type=measurement_type,
                                              metadata=metadata,
                                              references=references)

    return measurement


def relaxation_adaptive(device, qubit_id, transition='01', delay_seq_generator=None, measurement_type='decay', additional_references = {}, additional_metadata={}, expected_T1=None):
    # check if we have fitted Rabi measurements on this qubit-channel combo
    #Rabi_measurements = device.exdir_db.select_measurements_db(measurment_type='Rabi_rect', metadata={'qubit_id':qubit_id}, references={'channel_amplitudes': channel_amplitudes.id})
    #Rabi_fits = [exdir_db.references.this.filename for measurement in Rabi_measurements for references in measurement.reference_two if references.this.measurement_type=='fit_dataset_1d']

    #for fit in Rabi_fits:
    min_step = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_min_step'))
    scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_scan_points'))
    _range = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_range'))
    max_scan_length = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Ramsey_max_scan_length'))
    target_offset_correction = 0

    if expected_T1 is None:
        lengths = np.arange(0, min_step*scan_points, min_step)
    else:
        lengths = np.linspace(0, expected_T1*2, scan_points)

    while not np.max(lengths) > max_scan_length:
        # go down the rabbit hole for
        measurement = relaxation(device, qubit_id, transition=transition, lengths=lengths,
                             delay_seq_generator=delay_seq_generator, measurement_type=measurement_type,
                             additional_references=additional_references, additional_metadata=additional_metadata)
        fit_results = measurement.fit.metadata

        if int(fit_results['decay_goodness_test']):
            return device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':measurement.id})

        lengths *= _range