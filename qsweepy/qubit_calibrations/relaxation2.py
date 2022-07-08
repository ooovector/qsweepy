from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np

def relaxation(device, qubit_id, transition='01', *extra_sweep_args, channel_amplitudes=None, lengths=None,
           readout_delay=0, delay_seq_generator=None, measurement_type='decay', ex_pulse=None,
           additional_references = {}, additional_metadata = {}):
    from .readout_pulse2 import get_uncalibrated_measurer
    from ..fitters.exp import exp_fitter
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition)
    if ex_pulse is None:
        ex_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi, channel_amplitudes_override=channel_amplitudes)

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel//2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            #ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            #print('control_sequence=',control_sequence)
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock / 8))
    lengths = times * 8 / control_sequence.clock

    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.delay_seq_generator = delay_seq_generator
            self.lengths = lengths
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            # Create preparation sequence
            self.prepare_seq.extend(ex_pulse.get_pulse_sequence(0))
            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def set_delay(self, length):
            if self.delay_seq_generator is None:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)

            else:
                if length == self.lengths[0]:
                    self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                    self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                    self.prepare_seq[-2] = self.delay_sequence[0]
                    sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
                                                              self.control_sequence)
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length)
                    self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                else:
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length)

    setter = ParameterSetter()


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
                                              (lengths, setter.set_delay, 'Delay','s'),
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


def relaxation_prepulse(device, qubit_id, transition='01', pre_pulse_delays = None, channel_amplitudes=None, lengths=None,
       readout_delay=0, delay_seq_generator=None, measurement_type='decay', ex_pulse=None,
       additional_references = {}, additional_metadata = {}):
    from .readout_pulse2 import get_uncalibrated_measurer
    from ..fitters.exp import exp_fitter
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    # readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition)
    if ex_pulse is None:
        ex_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi,
                                                         channel_amplitudes_override=channel_amplitudes)

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
            # ex_seq.start(holder=1)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            # print('control_sequence=',control_sequence)
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock / 8))
    lengths = times * 8 / control_sequence.clock

    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers = ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.delay_seq_generator = delay_seq_generator
            self.lengths = lengths
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence
            # Create preparation sequence
            self.prepare_seq.extend(ex_pulse.get_pulse_sequence(0))
            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def set_pre_pulse_delay(self, delay):

            for ex_seq in self.ex_sequencers:
                ex_seq.set_prepulse_delay(delay)

        def set_delay(self, length):
            if self.delay_seq_generator is None:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)

            else:
                if length == self.lengths[0]:
                    self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                    self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                    self.prepare_seq[-2] = self.delay_sequence[0]
                    sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
                                                              self.control_sequence)
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length)
                    self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                else:
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length)

    setter = ParameterSetter()

    references = {'ex_pulse': ex_pulse.id,
                  'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq' + qubit_id, exp_fitter(), -1,
                        np.arange(len((pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'))))

    metadata = {'qubit_id': qubit_id,
                'transition': transition,
                'extra_sweep_args': str(
                    len((pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'))),
                'readout_delay': str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'),
                                                            (lengths, setter.set_delay, 'Delay', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=10)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()

    return measurement