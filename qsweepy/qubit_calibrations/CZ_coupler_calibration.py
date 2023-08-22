from qsweepy.ponyfiles.data_structures import *
import traceback
#from .import
from qsweepy.libraries import pulses2 as pulses
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import Rabi2 as Rabi
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
from qsweepy.qubit_calibrations.readout_pulse2 import *
import textwrap

def CZ_coupler_rabi(device, qubit_id, coupler_id, gate, amplitude, frequencies, transition='01', lengths=None, tail_length=0e-9, readout_delay=0,
                    *extra_sweep_args, repeats=1, gate_type='rect',  post_pulse=None, pre_pulse1=None, pre_pulse2=None, additional_metadata={}):

    assert gate_type == 'rect' or gate_type == 'gauss'
    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    #sequence difinition

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(coupler_id).keys()][0]
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
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{gate.metadata[
                                                                        'carrier_name']: amplitude})
    class ParameterSetter:
        def __init__(self):

            self.lengths = lengths
            self.tail_length = tail_length
            self.amplitude = amplitude
            self.frequency = frequencies[0]


            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.post_pulse = post_pulse
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            self.length_setter(self.lengths[0])

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def length_setter(self, length):

            # this is kostyl again
            if gate_type == 'rect':
                length_tmp = self.lengths[0]
            elif gate_type == 'gauss':
                length_tmp = length


            if length == length_tmp:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []

                # if self.pre_pulse1 is not None:
                #     if self.pre_pulse2 is not None:
                #         self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                #                                                     self.pre_pulse2.get_pulse_sequence(0)[0])])
                #         self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                #                                                     self.pre_pulse2.get_pulse_sequence(0)[0])])
                #     else:
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # elif self.pre_pulse2 is not None:
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                if self.pre_pulse1 is not None:
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                if self.pre_pulse2 is not None:
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                # Gate 1
                if gate_type == 'rect':
                    # gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                    #                                                            channel_amplitudes=channel_amplitudes1_,
                    #                                                            tail_length=self.tail_length,
                    #                                                            length=self.lengths,
                    #                                                            phase=0.0,
                    #                                                            fast_control=True)
                    # for _ in range(repeats):
                    #     self.prepare_seq.extend(gate1_pulse)
                    #     self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])

                    if 'pulse_type' in gate.metadata:
                        if gate.metadata['pulse_type'] == 'cos':
                            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                                        **{gate.metadata[
                                                                                               'carrier_name']: self.amplitude})
                            frequency = self.frequency  # float(gate.metadata['frequency'])
                            tail_length = float(gate.metadata['tail_length'])
                            phase = 0.0
                            fast_control = True
                            channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length,
                                               fast_control, frequency) for
                                              c, a in channel_amplitudes_.items()]
                            gate1_pulse = [device.pg.pmulti(device, length_tmp, *tuple(channel_pulses))]
                    else:
                        channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                                    **{gate.metadata[
                                                                                           'carrier_name']: self.amplitude})
                        gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                                  channel_amplitudes=channel_amplitudes_,
                                                                                  tail_length=float(
                                                                                      gate.metadata['tail_length']),
                                                                                  length=length_tmp,
                                                                                  phase=0.0,
                                                                                  fast_control=True)


                    for _ in range(repeats):
                        self.prepare_seq.extend(gate1_pulse)
                        self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])


                elif gate_type == 'gauss':
                    gate1_pulse = [(c, device.pg.gauss_hd, float(a)*1., length/4, 0., 0., False)
                                    for c, a in channel_amplitudes1_.metadata.items()]
                    for _ in range(repeats):
                        self.prepare_seq.append(device.pg.pmulti(device, length, *tuple(gate1_pulse)))
                        self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])


                #self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                if post_pulse is not None:
                    self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                    # self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)

    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for qubit_id_ in coupler_id}
    references['channel_amplitudes'] = channel_amplitudes1_.id
    references['readout_pulse'] = readout_pulse.id
    measurement_type = 'Rabi_'+gate_type

    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id #[0]+qubit_id[1]

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    if len(coupler_id)>1:
        arg_id = -2 # the last parameter is resultnumbers, so the time-like argument is -2
    else:
        arg_id = -1
    fitter_arguments = (measurement_name, exp_sin_fitter(mode=exp_sin_fitter_mode), arg_id, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(coupler_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(gate.metadata['tail_length']),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition':transition}
    metadata.update(additional_metadata)

    setter = ParameterSetter()

    references['long_process'] =  gate.id
    references['readout_pulse'] = readout_pulse.id
    if post_pulse is not None:
        references['post_pulse'] = post_pulse.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            (frequencies, setter.frequency_setter, 'frequency'),
                                                            (lengths, setter.length_setter, 'Excitation length', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement




def CZ_coupler_double_rabi(device, qubit_id, coupler_id, gate, amplitude, transition='01', lengths=None, tail_length=0e-9, readout_delay=0,
                    *extra_sweep_args, repeats=1, gate_type='rect',  post_pulse=None, pre_pulse1=None, pre_pulse2=None, additional_metadata={}):

    assert gate_type == 'rect' or gate_type == 'gauss'
    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    #sequence difinition

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(coupler_id).keys()][0]
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
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{gate.metadata[
                                                                        'carrier_name']: amplitude})
    class ParameterSetter:
        def __init__(self):

            self.lengths = lengths
            self.tail_length = tail_length
            self.amplitude = amplitude
            self.frequency = 0


            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.post_pulse = post_pulse
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            self.length_setter(self.lengths[0])

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def length_setter(self, length):

            # this is kostyl again
            if gate_type == 'rect':
                length_tmp = self.lengths[0]
            elif gate_type == 'gauss':
                length_tmp = length


            if length == length_tmp:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []

                # if self.pre_pulse1 is not None:
                #     if self.pre_pulse2 is not None:
                #         self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                #                                                     self.pre_pulse2.get_pulse_sequence(0)[0])])
                #         self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                #                                                     self.pre_pulse2.get_pulse_sequence(0)[0])])
                #     else:
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # elif self.pre_pulse2 is not None:
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                if self.pre_pulse1 is not None:
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                if self.pre_pulse2 is not None:
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                # Gate 1
                if gate_type == 'rect':
                    gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                               channel_amplitudes=channel_amplitudes1_,
                                                                               tail_length=self.tail_length,
                                                                               length=length,
                                                                               phase=0.0,
                                                                               fast_control=True)
                    self.prepare_seq.extend(gate1_pulse)

                    s_pulse = [(c, device.pg.virtual_z, 180, False) for c, p in
                               channel_amplitudes1_.items()]
                    sequence_z = [device.pg.pmulti(device, 0, *tuple(s_pulse))]
                    self.prepare_seq.extend(sequence_z)

                    self.prepare_seq.extend(gate1_pulse)

                    self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])

                elif gate_type == 'gauss':
                    gate1_pulse = [(c, device.pg.gauss_hd, float(a)*1., length/4, 0., 0., False)
                                    for c, a in channel_amplitudes1_.metadata.items()]
                    for _ in range(repeats):
                        self.prepare_seq.append(device.pg.pmulti(device, length, *tuple(gate1_pulse)))
                        self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])


                #self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                if post_pulse is not None:
                    self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)

    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for qubit_id_ in coupler_id}
    references['channel_amplitudes'] = channel_amplitudes1_.id
    references['readout_pulse'] = readout_pulse.id
    measurement_type = 'Rabi_'+gate_type

    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id #[0]+qubit_id[1]

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    if len(coupler_id)>1:
        arg_id = -2 # the last parameter is resultnumbers, so the time-like argument is -2
    else:
        arg_id = -1
    fitter_arguments = (measurement_name, exp_sin_fitter(mode=exp_sin_fitter_mode), arg_id, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': ','.join(coupler_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition':transition}
    metadata.update(additional_metadata)

    setter = ParameterSetter()

    references['long_process'] =  gate.id
    references['readout_pulse'] = readout_pulse.id
    if post_pulse is not None:
        references['post_pulse'] = post_pulse.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (lengths, setter.length_setter, 'Excitation length', 's'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def CZ_coupler_parametric_rabi(device, qubit_ids, gate, amplitude, lengths=None, frequencies=None,
                               tail_length=0e-9, readout_delay=0, *extra_sweep_args, repeats=1, post_pulse=None,
                               pre_pulse1=None, pre_pulse2=None, additional_metadata={}, post_pause=None):

    pre_pause = float(gate.metadata['pre_pause'])
    if post_pause is None:
        post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    #readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    #sequence difinition

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    times = np.zeros(len(lengths))
    for _i in range(len(lengths)):
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock

    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{gate.metadata[
                                                                        'carrier_name']: amplitude})
    class ParameterSetter:
        def __init__(self):

            self.lengths = lengths
            self.frequencies = frequencies
            self.tail_length = tail_length
            self.amplitude = amplitude
            self.length = lengths[0]
            self.frequency = frequencies[0]

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.post_pulse = post_pulse
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            # self.length_setter(self.lengths[0])
            # self.frequency(self.frequencies[0])

            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])

            if self.pre_pulse1 is not None:
                self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
            if self.pre_pulse2 is not None:
                self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

            self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.length)

            for _ in range(repeats):
                # self.prepare_seq.extend(gate1_pulse)
                # self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])
                self.prepare_seq.extend(self.pre_pause)
                self.prepare_seq.extend(self.delay_sequence)
                self.prepare_seq.extend(self.post_pause)

            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def delay_seq_generator(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate.metadata[
                                                                                'carrier_name']: self.amplitude})
            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = self.frequency  # float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [
                        (c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                        c, a in channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]

                    print('pulse_type=cos')
            else:
                frequency = self.frequency
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes_,
                                                                          tail_length=float(
                                                                              gate.metadata['tail_length']),
                                                                          length=self.length,
                                                                          phase=0.0,
                                                                          fast_control=True,
                                                                          control_frequency=frequency)

            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{'iq_ex1_q7': 0})
            channel_pulses1 = [(c, device.pg.pause, True) for c, a in channel_amplitudes1_.items()]
            pause_pulse1 = [device.pg.pmulti(device, self.lengths, *tuple(channel_pulses1))]

            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{'iq_ex1_q8': 0})
            channel_pulses2 = [(c, device.pg.pause, True) for c, a in channel_amplitudes2_.items()]
            pause_pulse2 = [device.pg.pmulti(device, self.lengths, *tuple(channel_pulses2))]

            self.pre_pause_seq1 = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq1 = [device.pg.parallel(gate_pulse[0], pause_pulse1[0], pause_pulse2[0])]
            self.post_pause_seq1 = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq1, self.gate_pulse_seq1, self.post_pause_seq1


        def length_setter(self, length):

            if length == self.lengths[0]:

                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(length)
                print(self.delay_sequence)
                self.prepare_seq[-2] = self.delay_sequence[0]
            # sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
            # self.control_sequence)
            # if True:
            #     channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
            #                                                                  **{gate.metadata[
            #                                                                         'carrier_name']: self.amplitude})
            #     self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            #     self.prepare_seq = []
            #
            #     if self.pre_pulse1 is not None:
            #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
            #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
            #     if self.pre_pulse2 is not None:
            #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
            #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
            #
            #     # Gate 1
            #
            #     if 'pulse_type' in gate.metadata:
            #         if gate.metadata['pulse_type'] == 'cos':
            #             frequency = self.frequency  # float(gate.metadata['frequency'])
            #             tail_length = float(gate.metadata['tail_length'])
            #             phase = 0.0
            #             fast_control = True
            #             channel_pulses = [
            #                 (c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
            #                 c, a in channel_amplitudes1_.items()]
            #             gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            #     else:
            #         frequency = self.frequency
            #         gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
            #                                                                   channel_amplitudes=channel_amplitudes1_,
            #                                                                   tail_length=float(
            #                                                                       gate.metadata['tail_length']),
            #                                                                   length=self.lengths,
            #                                                                   phase=0.0,
            #                                                                   fast_control=True,
            #                                                                   control_frequency=frequency)
            #     self.prepare_seq.extend([device.pg.pmulti(device, pre_pause)])
            #     self.prepare_seq.extend(gate_pulse)
            #     self.prepare_seq.extend([device.pg.pmulti(device, post_pause)])
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
                #print(self.prepare_seq)
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)

    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for qubit_id_ in qubit_ids}
    references['channel_amplitudes'] = channel_amplitudes1_.id
    references['readout_pulse'] = readout_pulse.id
    measurement_type = 'Rabi_coupler_parametric'

    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'resultnumbers' # 'iq' + qubit_ids[0]#+qubit_ids[1]
    # if len(qubit_ids) > 1:
    #     arg_id = -2  # the last parameter is resultnumbers, so the time-like argument is -2
    # else:
    #     arg_id = -1

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    # fitter_arguments = (measurement_name, exp_sin_fitter(mode=exp_sin_fitter_mode), -1, np.arange(len(extra_sweep_args)))

    metadata = {'qubit_id': qubit_ids[0],
                'qubit_ids': ','.join(qubit_ids),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'repeats': str(repeats)}
    metadata.update(additional_metadata)

    setter = ParameterSetter()

    references['long_process'] =  gate.id
    references['readout_pulse'] = readout_pulse.id
    if post_pulse is not None:
        references['post_pulse'] = post_pulse.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id

    # measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
    #                                                         *extra_sweep_args,
    #                                                         (frequencies, setter.frequency_setter, 'Modulation frequencies', 'Hz'),
    #                                                         (lengths, setter.length_setter, 'Excitation length', 's'),
    #                                                         # fitter_arguments=fitter_arguments,
    #                                                         measurement_type=measurement_type,
    #                                                         metadata=metadata,
    #                                                         references=references,
    #                                                         on_update_divider=5)


    measurement = device.sweeper.sweep(measurer,
                                        *extra_sweep_args,
                                        (frequencies, setter.frequency_setter, 'Modulation frequencies', 'Hz'),
                                        (lengths, setter.length_setter, 'Excitation length', 's'),
                                        # fitter_arguments=fitter_arguments,
                                        measurement_type=measurement_type,
                                        metadata=metadata,
                                        references=references,
                                        on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement




def CZ_coupler_ramsey(device, qubit_id, coupler_id,  gate, *extra_sweep_args, lengths, target_freq_offset,  post_pulse=None,
                      pre_pulse1=None, pre_pulse2=None, additional_metadata={}):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    #sequence difinition

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(coupler_id).keys()][0]
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
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock


    class ParameterSetter:
        def __init__(self):

            self.lengths = lengths

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.post_pulse = post_pulse
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2
            self.target_freq_offset=target_freq_offset


            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            #self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
            #self.set_delay(self.lengths[0])


        def set_delay(self, length):

            phase = int(np.round((length) * self.control_sequence.clock))/self.control_sequence.clock*target_freq_offset*360%360
            if length==self.lengths[0]:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []
                if self.pre_pulse1 is not None:
                    if self.pre_pulse2 is not None:
                        self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                                                                    self.pre_pulse2.get_pulse_sequence(0)[0])])
                        self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                                                                    self.pre_pulse2.get_pulse_sequence(0)[0])])
                    else:
                        self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                        self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                elif self.pre_pulse2 is not None:
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                # Gate 1
                channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                             **{gate.metadata[
                                                                                    'carrier_name']: float(gate.metadata['amplitude'])})
                gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                           channel_amplitudes=channel_amplitudes1_,
                                                                           tail_length=float(gate.metadata['tail_length']),
                                                                           length=float(gate.metadata['length']),
                                                                           # length=self.lengths,
                                                                           phase=0.0,
                                                                           fast_control=False)
                self.prepare_seq.extend(gate1_pulse)
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])

                fast_control = 'quasi-binary'
                #phases = {c: (phase if c in channel_amplitudes1_.keys() else 0) for c in device.awg_channels}
                s_pulse = [(c, device.pg.virtual_z, int(phase / 360 * (2 ** 8)), fast_control) for c, p in channel_amplitudes1_.items()]
                sequence_z = [device.pg.pmulti(device, 0, *tuple(s_pulse))]

                self.prepare_seq.extend(sequence_z)
                self.prepare_seq.extend(gate1_pulse)

                self.prepare_seq.extend([device.pg.pmulti(device, 500e-9)])
                self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
                if phase >= 0:
                    self.control_qubit_sequence.set_phase(int(phase / 360 * (2 ** 8)))
                else:
                    self.control_qubit_sequence.set_phase(int((360 + phase) / 360 * (2 ** 8)))
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
                if phase >= 0:
                    self.control_qubit_sequence.set_phase(int(phase / 360 * (2 ** 8)))
                else:
                    self.control_qubit_sequence.set_phase(int((360 + phase) / 360 * (2 ** 8)))


    metadata = {'qubit_id':coupler_id,
                'tail': float(gate.metadata['tail_length']),
                'amplitude':float(gate.metadata['amplitude'])}
    metadata.update(additional_metadata)
    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if post_pulse is not None:
        references['post_pulse'] = post_pulse.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id
    measurement_type = 'Ramsey_coupler'
    fitter_arguments = ('iq' + qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                       *extra_sweep_args,
                                       (lengths, setter.set_delay, 'Delay', 's'),
                                       fitter_arguments=fitter_arguments,
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references,
                                       on_update_divider=10)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def CZ_coupler_ramsey2(device, qubit_id, coupler_id,  gate, *extra_sweep_args, lengths, target_freq_offsets,  post_pulse=None, pre_pulse1=None, pre_pulse2=None):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    #sequence difinition

    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(coupler_id).keys()][0]
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
        times[_i] = int(round(lengths[_i] * control_sequence.clock))
    lengths = times / control_sequence.clock


    class ParameterSetter:
        def __init__(self):

            self.lengths = lengths

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.post_pulse = post_pulse
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2
            self.target_freq_offsets=target_freq_offsets
            self.target_freq_offset = target_freq_offsets[0]


            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            #self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
            #self.set_delay(self.lengths[0])

        def set_freq_ofset(self, freq_ofset):
            self.target_freq_offset=freq_ofset



        def set_delay(self, length):

            phase = int(np.round((length) * self.control_sequence.clock))/self.control_sequence.clock*self.target_freq_offset*360%360
            if length==self.lengths[0]:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []
                # if self.pre_pulse1 is not None:
                #     if self.pre_pulse2 is not None:
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #     else:
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # elif self.pre_pulse2 is not None:
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #mif self.pre_pulse1 is not None:
                #     self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #     self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # if self.pre_pulse2 is not None:
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                if self.pre_pulse1 is not None:
                    if self.pre_pulse2 is not None:
                        self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                                                                    self.pre_pulse2.get_pulse_sequence(0)[0])])
                        self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0],
                                                                    self.pre_pulse2.get_pulse_sequence(0)[0])])
                    else:
                        self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                        self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                elif self.pre_pulse2 is not None:
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                # Gate 1
                channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                             **{gate.metadata[
                                                                                    'carrier_name']: float(gate.metadata['amplitude'])})
                gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                           channel_amplitudes=channel_amplitudes1_,
                                                                           tail_length=float(gate.metadata['tail_length']),
                                                                           length=float(gate.metadata['length']),
                                                                           phase=0.0,
                                                                           fast_control=False)
                self.prepare_seq.extend(gate1_pulse)
                self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])

                fast_control = 'quasi-binary'
                #phases = {c: (phase if c in channel_amplitudes1_.keys() else 0) for c in device.awg_channels}
                s_pulse = [(c, device.pg.virtual_z, int(phase / 360 * (2 ** 8)), fast_control) for c, p in channel_amplitudes1_.items()]
                sequence_z = [device.pg.pmulti(device, 0, *tuple(s_pulse))]

                self.prepare_seq.extend(sequence_z)
                self.prepare_seq.extend(gate1_pulse)

                self.prepare_seq.extend([device.pg.pmulti(device, 500e-9)])
                self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                self.prepare_seq.extend(self.post_pulse.get_pulse_sequence(0))
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
                if phase >= 0:
                    self.control_qubit_sequence.set_phase(int(phase / 360 * (2 ** 8)))
                else:
                    self.control_qubit_sequence.set_phase(int((360 + phase) / 360 * (2 ** 8)))
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(length)
                if phase >= 0:
                    self.control_qubit_sequence.set_phase(int(phase / 360 * (2 ** 8)))
                else:
                    self.control_qubit_sequence.set_phase(int((360 + phase) / 360 * (2 ** 8)))


    metadata = {'qubit_id':coupler_id,
                'tail': float(gate.metadata['tail_length']),
                'amplitude':float(gate.metadata['amplitude'])}

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if post_pulse is not None:
        references['post_pulse'] = post_pulse.id

    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id

    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id

    measurement_type = 'Ramsey_coupler'
    fitter_arguments = ('iq' + qubit_id, exp_sin_fitter(), -1, np.arange(len(extra_sweep_args)))
    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                       *extra_sweep_args,
                                        (target_freq_offsets, setter.set_freq_ofset, 'Frequency offset', 'Hz'),
                                       (lengths, setter.set_delay, 'Delay', 's'),
                                       fitter_arguments=fitter_arguments,
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references,
                                       on_update_divider=10)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement