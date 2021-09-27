
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations.channel_amplitudes import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse

def echo_process(device, qubit_id1, qubit_id2, process, channel_amplitudes1=None, channel_amplitudes2=None):
    '''
    :param device QubitDevice:
    '''
    from .readout_pulse2 import get_uncalibrated_measurer
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id2) # we want to measure qubit 2 because otherwise wtf are we are doing the second pi/2 pulse for
    phase_scan_points = int(device.get_sample_global(name='process_phase_scan_points'))
    phases = np.linspace(0, 2*np.pi, phase_scan_points, endpoint=False)
    ex_pulse1 = excitation_pulse.get_excitation_pulse(device, qubit_id1, np.pi/2., channel_amplitudes_override=channel_amplitudes1)
    ex_pulse_pi = excitation_pulse.get_excitation_pulse(device, qubit_id1, np.pi, channel_amplitudes_override=channel_amplitudes1)
    ex_pulse2 = excitation_pulse.get_excitation_pulse(device, qubit_id2, np.pi/2., channel_amplitudes_override=channel_amplitudes2)

    def set_phase(phase):
        device.pg.set_seq(ex_pulse1.get_pulse_sequence(0.0) +
                          process.get_pulse_sequence() +
                          ex_pulse_pi(phase) +
                          process.get_pulse_sequence() +
                          ex_pulse2.get_pulse_sequence(phase) +
                          device.trigger_readout_seq +
                          readout_pulse.get_pulse_sequence())

    references = {'ex_pulse1':ex_pulse1.id,
                  'ex_pulse_pi': ex_pulse_pi.id,
                  'ex_pulse2':ex_pulse2.id,
                  ('frequency_controls', qubit_id1): device.get_frequency_control_measurement_id(qubit_id=qubit_id1),
                  ('frequency_controls', qubit_id2): device.get_frequency_control_measurement_id(qubit_id=qubit_id2),
                  'process': process.id}
    metadata = {'q1': qubit_id1,
                'q2': qubit_id2}
    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+qubit_id2, SinglePeriodSinFitter(), -1, [])

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (phases, set_phase, 'Phase','radians'),
                                                            fitter_arguments = fitter_arguments,
                                                            measurement_type='echo_process',
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=10)

    return measurement


def echo_crosstalk(device, target_qubit_id, control_qubit_id, *extra_sweep_args,
                   channel_amplitudes1=None, channel_amplitudes_pi=None, channel_amplitudes2=None, lengths=None,
                   target_freq_offset=None, readout_delay=0, delay_seq_generator=None, measurement_type='echo_crosstalk',
                   additional_references = {}, additional_metadata = {}):
    from .readout_pulse2 import get_uncalibrated_measurer
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=target_qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=target_qubit_id, name='Ramsey_step')))

    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, target_qubit_id)
    ex_pulse1 = excitation_pulse.get_excitation_pulse(device, target_qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes1)
    ex_pulse_pi = excitation_pulse.get_excitation_pulse(device, target_qubit_id, np.pi, channel_amplitudes_override=channel_amplitudes_pi)
    ex_pulse_control = excitation_pulse.get_excitation_pulse(device, control_qubit_id, np.pi)
    ex_pulse2 = excitation_pulse.get_excitation_pulse(device, target_qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes2)

    # Define qubit and readout sequencers
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(target_qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_seq_id = ex_channel.parent.sequencer_id
    else:
        control_seq_id = ex_channel.channel // 2
    ex_sequencers = []

    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()


    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.delay_seq_generator = delay_seq_generator
            self.lengths = lengths/2
            self.control_sequence = control_sequence
            # Create preparation sequence
            self.prepare_seq.extend(ex_pulse1.get_pulse_sequence(0))
            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            self.prepare_seq.extend(device.pg.parallel(ex_pulse_pi.get_pulse_sequence(0), ex_pulse_control.get_pulse_sequence(0)))

            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            self.prepare_seq.extend(excitation_pulse.get_s(device, target_qubit_id,
                                                           phase=(64/self.control_sequence.clock)*target_freq_offset*360 % 360,
                                                           fast_control='quasi-binary'))
            self.prepare_seq.extend(ex_pulse2.get_pulse_sequence(0))
            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def set_delay(self, length):
            phase = int(np.round((length+140e-9)*self.control_sequence.clock)+64)/self.control_sequence.clock*target_freq_offset*360 % 360
            #print ('length: ', length, ', phase: ', phase, ', phase register: ', int(phase/360*(2**6)))

            if self.delay_seq_generator is None:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length/2)
                self.control_sequence.set_phase(int(phase/360*(2**6)))
            else:
                if length == self.lengths[0]*2:
                    self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                    self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                    self.prepare_seq[-4] = self.delay_sequence[0]
                    self.prepare_seq[-8] = self.delay_sequence[0]
                    sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
                                                              self.control_sequence)
                    #sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)

                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length/2)
                    self.control_sequence.set_phase(int(phase/360*(2**6)))
                    self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

                else:
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length/2)
                    self.control_sequence.set_phase(int(phase/360*(2**6)))

    setter = ParameterSetter()


    references = {'ex_pulse_control': ex_pulse_control.id,
                  'ex_pulse1': ex_pulse1.id,
                  'ex_pulse_pi': ex_pulse_pi.id,
                  'ex_pulse2': ex_pulse2.id,
                  'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=target_qubit_id)}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+target_qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'target_qubit_id': target_qubit_id,
                'control_qubit_id': control_qubit_id,
                'extra_sweep_args':str(len(extra_sweep_args)),
                'target_offset_freq': str(target_freq_offset),
                'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                              *extra_sweep_args,
                                              (lengths, setter.set_delay, 'Delay', 's'),
                                              fitter_arguments = fitter_arguments,
                                              measurement_type=measurement_type,
                                              metadata=metadata,
                                              references=references,
                                              on_update_divider=10)

    return measurement


def echo(device, qubit_id, transition='01', *extra_sweep_args, channel_amplitudes1=None, channel_amplitudes_pi=None, channel_amplitudes2=None, lengths=None,
           target_freq_offset=None, readout_delay=0, delay_seq_generator=None, measurement_type='echo',
           additional_references = {}, additional_metadata = {}):
    from .readout_pulse2 import get_uncalibrated_measurer
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition)
    ex_pulse1 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes1)
    ex_pulse_pi = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi, channel_amplitudes_override=channel_amplitudes_pi)
    ex_pulse2 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes2)

    # Define qubit and readout sequencers
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_seq_id = ex_channel.parent.sequencer_id
    else:
        control_seq_id = ex_channel.channel // 2
    ex_sequencers = []

    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()


    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.delay_seq_generator = delay_seq_generator
            self.lengths = lengths/2
            self.control_sequence = control_sequence
            # Create preparation sequence
            self.prepare_seq.extend(ex_pulse1.get_pulse_sequence(0))
            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            self.prepare_seq.extend(ex_pulse_pi.get_pulse_sequence(0))

            if self.delay_seq_generator is None:
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
                self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            else:
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=(64/self.control_sequence.clock)*target_freq_offset*360 % 360,
                                                           fast_control='quasi-binary'))
            self.prepare_seq.extend(ex_pulse2.get_pulse_sequence(0))
            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def set_delay(self, length):
            phase = int(np.round((length+140e-9)*self.control_sequence.clock)+64)/self.control_sequence.clock*target_freq_offset*360 % 360
            #print ('length: ', length, ', phase: ', phase, ', phase register: ', int(phase/360*(2**6)))

            if self.delay_seq_generator is None:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length/2)
                self.control_sequence.set_phase(int(phase/360*(2**6)))
            else:
                if length == self.lengths[0]*2:
                    self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                    self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
                    self.prepare_seq[-4] = self.delay_sequence[0]
                    self.prepare_seq[-8] = self.delay_sequence[0]
                    sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
                                                              self.control_sequence)
                    #sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)

                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length/2)
                    self.control_sequence.set_phase(int(phase/360*(2**6)))
                    self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

                else:
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(length/2)
                    self.control_sequence.set_phase(int(phase/360*(2**6)))

    setter = ParameterSetter()

    references = {'ex_pulse1':ex_pulse1.id,
                  'ex_pulse_pi': ex_pulse_pi.id,
                  'ex_pulse2':ex_pulse2.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': qubit_id,
                'transition': transition,
              'extra_sweep_args':str(len(extra_sweep_args)),
              'target_offset_freq': str(target_freq_offset),
              'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                              *extra_sweep_args,
                                              (lengths, setter.set_delay, 'Delay','s'),
                                              fitter_arguments = fitter_arguments,
                                              measurement_type=measurement_type,
                                              metadata=metadata,
                                              references=references,
                                              on_update_divider=10)

    return measurement

def echo_prepulse(device, qubit_id, transition='01', pre_pulse_delays = None, channel_amplitudes1=None,
                  channel_amplitudes_pi=None, channel_amplitudes2=None, lengths=None, target_freq_offset=None,
                  readout_delay=0, measurement_type='echo_prepulse_delay', additional_references = {},
                  additional_metadata = {}):
    from .readout_pulse2 import get_uncalibrated_measurer
    if type(lengths) is type(None):
        lengths = np.arange(0,
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_length')),
                            float(device.get_qubit_constant(qubit_id=qubit_id, name='Ramsey_step')))

    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id, transition)
    ex_pulse1 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes1)
    ex_pulse_pi = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi, channel_amplitudes_override=channel_amplitudes_pi)
    ex_pulse2 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=channel_amplitudes2)


    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_seq_id = ex_channel.parent.sequencer_id
    else:
        control_seq_id = ex_channel.channel//2
    ex_sequencers = []

    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses = [])
        else:
            ex_seq = zi_scripts.SIMPLESequence(sequencer_id=seq_id, awg=device.modem.awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
            print('control_sequence=',control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        #device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            self.ex_sequencers=ex_sequencers
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.lengths = lengths/2
            self.control_sequence = control_sequence
            # Create preparation sequence
            self.prepare_seq.extend(ex_pulse1.get_pulse_sequence(0))

            self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
            self.prepare_seq.extend([device.pg.pmulti(device, 0)])

            self.prepare_seq.extend(ex_pulse_pi.get_pulse_sequence(0))

            self.prepare_seq.extend([device.pg.pmulti(device, 0)])
            self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
            self.prepare_seq.extend([device.pg.pmulti(device, 0)])


            self.prepare_seq.extend(excitation_pulse.get_s(device, qubit_id,
                                                           phase=(64/self.control_sequence.clock)*target_freq_offset*360 % 360,
                                                           fast_control='quasi-binary'))
            self.prepare_seq.extend(ex_pulse2.get_pulse_sequence(0))
            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()
        def set_pre_pulse_delay(self, delay):

            for ex_seq in self.ex_sequencers:
                ex_seq.set_prepulse_delay(delay)

        def set_delay(self, length):
            phase = int(np.round((length+140e-9)*self.control_sequence.clock)+64)/self.control_sequence.clock*target_freq_offset*360 % 360
            #print ('length: ', length, ', phase: ', phase, ', phase register: ', int(phase/360*(2**6)))

            for ex_seq in self.ex_sequencers:
                ex_seq.set_length(length/2)
                    #ex_seq.set_phase(int(phase / 360 * (2 ** 8)))
            if phase >= 0:
                self.control_sequence.set_phase(int(phase/360*(2**6)))
            else:
                self.control_sequence.set_phase(int((360+phase) / 360 * (2 ** 6)))

    setter = ParameterSetter()

    references = {'ex_pulse1':ex_pulse1.id,
                  'ex_pulse_pi': ex_pulse_pi.id,
                  'ex_pulse2':ex_pulse2.id,
                  'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    references.update(additional_references)

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    fitter_arguments = ('iq'+qubit_id, exp_sin_fitter(), -1, np.arange(len((pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'))))

    metadata = {'qubit_id': qubit_id,
                'transition': transition,
              'extra_sweep_args':str(len((pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'))),
              'target_offset_freq': str(target_freq_offset),
              'readout_delay':str(readout_delay)}
    metadata.update(additional_metadata)

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                              (pre_pulse_delays, setter.set_pre_pulse_delay, 'Pre pulse Delay', 's'),
                                              (lengths, setter.set_delay, 'Delay','s'),
                                              fitter_arguments = fitter_arguments,
                                              measurement_type=measurement_type,
                                              metadata=metadata,
                                              references=references,
                                              on_update_divider=10)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def get_echo_pulse_frequency(device, echo_measurement):
    ex_pulse1 = device.exdir_db.select_measurement_by_id(echo_measurement.references['ex_pulse1'])
    ex_pulse_pi = device.exdir_db.select_measurement_by_id(echo_measurement.references['ex_pulse_pi'])
    ex_pulse2 = device.exdir_db.select_measurement_by_id(echo_measurement.references['ex_pulse2'])
    qubit_id = echo_measurement.metadata['qubit_id']
    frequency_rounding = float(device.get_qubit_constant(qubit_id=qubit_id, name='frequency_rounding'))
    channel1_frequencies = []
    channel_pi_frequencies = []
    channel2_frequencies = []
    for reference in ex_pulse1.references:
        if 'channel_calibration_' in reference.name:
            channel1_frequencies.append([float(m.value) for m in reference.that.metadata if m.name == 'frequency'][0])
    for reference in ex_pulse_pi.references:
        if 'channel_calibration_' in reference.name:
            channel_pi_frequencies.append([float(m.value) for m in reference.that.metadata if m.name == 'frequency'][0])
    for reference in ex_pulse2.references:
        if 'channel_calibration_' in reference.name:
            channel2_frequencies.append([float(m.value) for m in reference.that.metadata if m.name == 'frequency'][0])
    if np.max(channel1_frequencies) - np.min(channel1_frequencies) < frequency_rounding:
        raise Exception('Channel 1 frequency spreads are larger than frequency rounding')
    if np.max(channel_pi_frequencies) - np.min(channel_pi_frequencies) < frequency_rounding:
        raise Exception('Channel pi frequency spreads are larger than frequency rounding')
    if np.max(channel2_frequencies) - np.min(channel2_frequencies) < frequency_rounding:
        raise Exception('Channel 2 frequency spreads are larger than frequency rounding')

def echo_adaptive(device, qubit_id, transition='01', set_frequency=True, delay_seq_generator=None, measurement_type='echo', additional_references = {}, additional_metadata={}, expected_T2=None):
    # check if we have fitted Rabi measurements on this qubit-channel combo
    #Rabi_measurements = device.exdir_db.select_measurements_db(measurment_type='Rabi_rect', metadata={'qubit_id':qubit_id}, references={'channel_amplitudes': channel_amplitudes.id})
    #Rabi_fits = [exdir_db.references.this.filename for measurement in Rabi_measurements for references in measurement.reference_two if references.this.measurement_type=='fit_dataset_1d']

    #for fit in Rabi_fits:
    min_step = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_min_step'))
    scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_scan_points'))
    _range = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_range'))
    max_scan_length = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Ramsey_max_scan_length'))
    points_per_oscillation_target = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Ramsey_points_per_oscillation'))
    frequency_rounding = float(device.get_qubit_constant(qubit_id=qubit_id, name='frequency_rounding'))
    target_offset_correction = 0

    if expected_T2 is None:
        lengths = np.arange(0, min_step*scan_points, min_step)
        target_offset = 1. / (min_step * points_per_oscillation_target)
    else:
        lengths = np.linspace(0, expected_T2*2, scan_points)
        target_offset = scan_points/(expected_T2*2*points_per_oscillation_target)
    qubit_frequency_guess = device.get_qubit_fq(qubit_id)

    while not np.max(lengths)>max_scan_length:
        # go down the rabbit hole for
        measurement = echo(device, qubit_id, transition=transition, lengths=lengths, target_freq_offset=(target_offset+target_offset_correction),
                             delay_seq_generator=delay_seq_generator, measurement_type=measurement_type,
                             additional_references=additional_references, additional_metadata=additional_metadata)
        fit_results = measurement.fit.metadata
        if not (int(fit_results['frequency_goodness_test'])):
            raise Exception('Failed to calibrate echo: frequency_goodness_test failed on qubit {}, measurement {}'.format(qubit_id, measurement.id))

        # reset frequency?
        frequency_delta = - float(fit_results['f']) + (target_offset)
        num_delta_periods = np.abs(frequency_delta*(np.max(measurement.datasets['iq'+qubit_id].parameters[0].values)-np.min(measurement.datasets['iq'+qubit_id].parameters[0].values)))
        num_decay_periods = np.abs(frequency_delta*float(fit_results['T']))
        if num_delta_periods > 1/8 and num_decay_periods > 1/8 and set_frequency:
            old_frequency = device.get_qubit_fq(qubit_id)
            new_frequency = round((old_frequency+frequency_delta)/frequency_rounding)*frequency_rounding
            device.set_qubit_fq(new_frequency, qubit_id=qubit_id)
            assert(np.abs(frequency_delta) < np.abs(target_offset/2.))
            print('Updating qubit frequency: old frequency {}, new frequency {}'.format(old_frequency, new_frequency))
            device.update_pulsed_frequencies()
        elif num_delta_periods > 1/8 and num_decay_periods > 1/8:
            target_offset_correction = -frequency_delta+target_offset_correction

        if int(fit_results['decay_goodness_test']):
            return device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':measurement.id})

        lengths *= _range
        target_offset /= _range

    #raise ValueError('Failed measuring Rabi frequency for qubit {} on channel_amplitudes {}'.format(qubit_id, channel_amplitudes.metadata))

def calibrate_all_T2(device, force_recalibration=False):
    for qubit_id in device.get_qubit_list():
        get_echo_coherence_measurement(device, qubit_id, force_recalibration=force_recalibration)
        #if True: ### TODO check if echo measurement is already there
            #echo_adaptive(device, qubit_id, set_frequency=True)

def calibrate_all_crosstalks(device):
    for control_qubit_id in device.get_qubit_list():
        for target_qubit_id in device.get_qubit_list():
            if control_qubit_id != target_qubit_id:
                echo_crosstalk(device, target_qubit_id, control_qubit_id)

def get_echo_crosstalk_measurement(device,
                                     target_qubit_id,
                                     control_qubit_id,
                                     frequency_controls=None,
                                     recalibrate=True,
                                     force_recalibration=False):
    if frequency_controls is None:
        frequency_controls = device.get_frequency_control_measurement_id(target_qubit_id)

    echo_crosstalk_measurements = device.exdir_db.db.Data.select_by_sql('''
    SELECT
        echo_fit.*
        FROM data echo_fit
        INNER JOIN Reference fit_measurement_reference ON
            fit_measurement_reference.this = echo_fit.id and fit_measurement_reference.ref_type='fit_source'
        INNER JOIN Data measurement ON
            measurement.id = fit_measurement_reference.that
        INNER JOIN Metadata measurement_target_qubit_id ON
            (measurement_qubit_id.data_id = measurement.id
                AND measurement_qubit_id.name='target_qubit_id'
                AND measurement_qubit_id.value='{target_qubit_id}')
            INNER JOIN Metadata measurement_control_qubit_id ON
                (measurement_qubit_id.data_id = measurement.id
                    AND measurement_qubit_id.name='control_qubit_id'
                    AND measurement_qubit_id.value='{control_qubit_id}')
        INNER JOIN Metadata fit_decay_goodness_test ON
            (fit_decay_goodness_test.data_id = echo_fit.id
                AND fit_decay_goodness_test.name='decay_goodness_test'
                AND fit_decay_goodness_test.value='1'
            )
        INNER JOIN Metadata transition ON 
            transition.data_id = measurement.id AND
            transition.name = 'transition' AND
            transition.value = '01'
        INNER JOIN reference frequency_controls ON
            (frequency_controls.this = measurement.id AND
             frequency_controls.that = {frequency_controls} AND
             frequency_controls.ref_type = 'frequency_controls')
        WHERE (NOT measurement.invalid OR measurement.invalid IS NULL) AND
              (NOT measurement.incomplete OR measurement.incomplete IS NULL) AND
            measurement.measurement_type='echo_crosstalk'
    ;
    '''.format(target_qubit_id=target_qubit_id,
               control_qubit_id=control_qubit_id,
               frequency_controls=device.get_frequency_control_measurement_id(qubit_id=target_qubit_id)))

    for measurement in echo_crosstalk_measurements:
        try:
            assert not force_recalibration
            return device.exdir_db.select_measurement_by_id(measurement.id)
        except:
            print ('Failed loading echo coherence measurement {}'.format(measurement.id))

    if recalibrate:
        return echo_crosstalk(device, target_qubit_id, control_qubit_id)
    else:
        raise ValueError('No crosstalk measurement available for qubit {}, recalibrate is False, so fail'.format(target_qubit_id))

def get_echo_coherence_measurement(device,
                                     qubit_id,
                                     transition='01',
                                     frequency_controls=None,
                                     recalibrate=True,
                                     force_recalibration=False):
    if frequency_controls is None:
        frequency_controls = device.get_frequency_control_measurement_id(qubit_id)

    echo_coherence_measurements = device.exdir_db.db.Data.select_by_sql(
    '''
    SELECT
        echo_fit.*
        FROM data echo_fit
        INNER JOIN Reference fit_measurement_reference ON
            fit_measurement_reference.this = echo_fit.id and fit_measurement_reference.ref_type='fit_source'
        INNER JOIN Data measurement ON
            measurement.id = fit_measurement_reference.that
        INNER JOIN Metadata measurement_qubit_id ON
            (measurement_qubit_id.data_id = measurement.id
                AND measurement_qubit_id.name='qubit_id'
                AND measurement_qubit_id.value='{qubit_id}')
        INNER JOIN Metadata fit_decay_goodness_test ON
            (fit_decay_goodness_test.data_id = echo_fit.id
                AND fit_decay_goodness_test.name='decay_goodness_test'
                AND fit_decay_goodness_test.value='1'
            )
        INNER JOIN Metadata transition ON 
            transition.data_id = measurement.id AND
            transition.name = 'transition' AND
            transition.value = '{transition}'
        INNER JOIN reference frequency_controls ON
            (frequency_controls.this = measurement.id AND
             frequency_controls.that = {frequency_controls} AND
             frequency_controls.ref_type = 'frequency_controls')
        WHERE (NOT measurement.invalid OR measurement.invalid IS NULL) AND
              (NOT measurement.incomplete OR measurement.incomplete IS NULL) AND
            measurement.measurement_type='echo'
    ;
    '''.format(qubit_id=qubit_id, transition=transition, frequency_controls=frequency_controls))

    for measurement in echo_coherence_measurements:
        try:
            assert not force_recalibration
            return device.exdir_db.select_measurement_by_id(measurement.id)
        except:
            print ('Failed loading echo coherence measurement {}'.format(measurement.id))

    if recalibrate:
        return echo_adaptive(device, qubit_id, transition)
    else:
        raise ValueError('No coherence measurement available for qubit {}, recalibrate is False, so fail'.format(qubit_id))


def calibrate_all_cross_echo(device):
    cross_echo_fits = {}

    for qubit_id in device.get_qubit_list():
        min_step = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_min_step'))
        scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_scan_points'))
        points_per_oscillation_target = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Ramsey_points_per_oscillation'))
        initial_delay = float(device.get_qubit_constant(qubit_id=qubit_id, name='cross_Ramsey_initial_delay'))
        target_offset = 1./(min_step*points_per_oscillation_target)
        amplitude_default = float(device.get_qubit_constant(qubit_id=qubit_id, name='amplitude_default'))

        lengths = np.arange(initial_delay, min_step*scan_points+initial_delay, min_step)

        cross_echo_fits[qubit_id] = {}

        for channel_name1, device_name1 in device.get_qubit_excitation_channel_list(qubit_id).items():
            ch1 = channel_amplitudes(device, **{channel_name1:amplitude_default})
            try:
                pulse1 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=ch1)
            except:
                print ('Failed to Rabi-calibrate channel ', channel_name)
                continue

            cross_echo_fits[qubit_id][channel_name1] = {}

            for channel_name2, device_name2 in device.get_qubit_excitation_channel_list(qubit_id).items():
                ch2 = channel_amplitudes(device, **{channel_name2:amplitude_default})
                try:
                    pulse2 = excitation_pulse.get_excitation_pulse(device, qubit_id, np.pi/2., channel_amplitudes_override=ch2)
                except:
                    print ('Failed to Rabi-calibrate channel ', channel_name)
                    continue

                #cross_echo_measurement_results = {}
                cross_echo_measurements = device.exdir_db.select_measurements_db(measurement_type='echo', references_that={'ex_pulse1': pulse1.id, 'ex_pulse2': pulse2.id})
                for m in cross_echo_measurements:
                    for r in m.reference_two:
                        if r.ref_type=='fit_source':
                            fit = r.this
                            if int(device.exdir_db.db.Metadata[fit, 'frequency_goodness_test'].value):
                                cross_echo_fits[qubit_id][channel_name1][channel_name2] = device.exdir_db.select_measurement_by_id(fit.id)
                if channel_name2 not in cross_echo_fits[qubit_id][channel_name1]: # hasn't been found in db
                    echo(device, qubit_id, channel_amplitudes1=ch1, channel_amplitudes2=ch2, lengths=lengths, target_freq_offset=target_offset)

    return cross_echo_fits
