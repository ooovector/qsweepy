from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import relaxation2 as relaxation
from qsweepy.qubit_calibrations import echo2 as echo
from qsweepy.qubit_calibrations import channel_amplitudes
import numpy as np
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts


def zgate_ramsey(device, gate):
    def filler_func(length):
        channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                              **{gate.metadata['carrier_name']: float(gate.metadata['amplitude'])})
        return excitation_pulse.get_rect_cos_pulse_sequence(device = device,
                                           channel_amplitudes = channel_amplitudes_ ,
                                           tail_length = float(gate.metadata['tail_length']),
                                           length = length,
                                           phase = 0.0)

    return Ramsey.Ramsey_adaptive(device=device, qubit_id=gate.metadata['target_qubit_id'], set_frequency=False,
                           delay_seq_generator=filler_func, measurement_type='Ramsey_long_process', additional_references={'gate':gate.id})


def zgate_amplitude_ramsey(device, gate, lengths, amplitudes,additional_metadata={},  target_freq_offset=100e9, pre_pulse_gate=None, sort='best',
                           readout_delay=None, gauss=False):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude = amplitudes[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []

        def amplitude_setter(self, amplitude):
            self.amplitude = 1j*amplitude

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                                      c, a in  channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0,
                                                                              fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    setter = ParameterSetter()

    return Ramsey.Ramsey(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths, target_freq_offset=target_freq_offset, readout_delay=readout_delay,
                           delay_seq_generator=setter.filler_func, pre_pulse_gate=pre_pulse_gate, measurement_type='Ramsey_amplitude_scan',
                           additional_references={'gate':gate.id}, additional_metadata= additional_metadata, sort=sort, gauss=gauss)


def zgate_amplitude_ramsey_sweep(device, gate, lengths, amplitudes, frequencies, additional_metadata={},  target_freq_offset=100e9, pre_pulse_gate=None, sort='best'):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude = amplitudes[0]
            self.freq = frequencies[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []
            self.amplitude_setter(amplitudes[0])

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def frequency_setter(self, freq):
            self.freq = freq

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = self.freq #float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                                      c, a in  channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0,
                                                                              fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    setter = ParameterSetter()

    return Ramsey.Ramsey(device, gate.metadata['target_qubit_id'], '01',
                           (frequencies, setter.frequency_setter, 'frequency'),
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, pre_pulse_gate=pre_pulse_gate, measurement_type='Ramsey_parametric_freq_scan',
                           additional_references={'gate':gate.id}, additional_metadata= additional_metadata, sort=sort)

def zgate_coupler_rabi_sweep(device, gate, lengths, amplitudes, frequencies, transition='01',additional_metadata={},
                             pre_pulse_gate1=None,pre_pulse_gate2=None):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    class GateParameterSetter:
        def __init__(self):
            self.amplitude = amplitudes[0]
            self.freq = frequencies[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []
            self.amplitude_setter(amplitudes[0])

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def frequency_setter(self, freq):
            self.freq = freq

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                        **{gate.metadata[
                                                                               'carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = self.freq  # float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [
                        (c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                        c, a in channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes_,
                                                                          tail_length=float(
                                                                              gate.metadata['tail_length']),
                                                                          length=self.length,
                                                                          phase=0.0,
                                                                          fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    setter = GateParameterSetter()

    # setter = ParameterSetter()

    return Ramsey.Ramsey(device, gate.metadata['target_qubit_id'], '01',
                           (frequencies, setter.frequency_setter, 'frequency'),
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, pre_pulse_gate=pre_pulse_gate, measurement_type='Ramsey_parametric_freq_scan',
                           additional_references={'gate':gate.id}, additional_metadata= additional_metadata, sort=sort)



    # from .readout_pulse2 import get_uncalibrated_measurer
    # readout_pulse, measurer = get_uncalibrated_measurer(device, gate.metadata['target_qubit_id'], transition)
    #
    #
    # exitation_channel = [i for i in device.get_qubit_excitation_channel_list(gate.metadata['target_qubit_id']).keys()][0]
    # ex_channel = device.awg_channels[exitation_channel]
    # if ex_channel.is_iq():
    #     control_qubit_awg = ex_channel.parent.awg
    #     control_qubit_seq_id = ex_channel.parent.sequencer_id
    # else:
    #     control_qubit_awg = ex_channel.parent.awg
    #     control_qubit_seq_id = ex_channel.channel//2
    # control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    # ex_sequencers = []
    #
    #
    # for awg, seq_id in device.pre_pulses.seq_in_use:
    #     if [awg, seq_id] != [control_awg, control_seq_id]:
    #         ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
    #                                            awg_amp=1, use_modulation=True, pre_pulses=[])
    #         #ex_seq.start(holder=1)
    #     else:
    #         ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
    #                                            awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
    #         control_sequence = ex_seq
    #         #print('control_sequence=',control_sequence)
    #     if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
    #         control_qubit_sequence = ex_seq
    #     device.pre_pulses.set_seq_offsets(ex_seq)
    #     device.pre_pulses.set_seq_prepulses(ex_seq)
    #     if ex_seq.params['is_iq']:
    #         ex_seq.start()
    #     else:
    #         ex_seq.start(holder=1)
    #     ex_sequencers.append(ex_seq)
    # readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    # readout_sequencer.start()
    #
    # times = np.zeros(len(lengths))
    # for _i in range(len(lengths)):
    #     times[_i] = int(round(lengths[_i] * control_sequence.clock))
    # lengths = times / control_sequence.clock
    #
    #
    # class ParameterSetter:
    #     def __init__(self):
    #         self.ex_sequencers = ex_sequencers
    #         self.prepare_seq = []
    #         self.readout_sequencer = readout_sequencer
    #         self.delay_seq_generator = gate_setter.filler_func
    #         self.lengths = lengths
    #         self.control_sequence = control_sequence
    #         self.control_qubit_sequence = control_qubit_sequence
    #         # Create preparation sequence
    #         if pre_pulse_gate1 is not None:
    #             self.prepare_seq.extend(pre_pulse_gate1.get_pulse_sequence(0))
    #         if pre_pulse_gate2 is not None:
    #             self.prepare_seq.extend(pre_pulse_gate2.get_pulse_sequence(0))
    #         if self.delay_seq_generator is None:
    #             self.prepare_seq.extend([device.pg.pmulti(device, 0)])
    #             self.prepare_seq.extend([device.pg.pmulti(device, self.lengths)])
    #             self.prepare_seq.extend([device.pg.pmulti(device, 0)])
    #         else:
    #             self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
    #             self.prepare_seq.extend(self.pre_pause)
    #             self.prepare_seq.extend(self.delay_sequence)
    #             self.prepare_seq.extend(self.post_pause)
    #
    #         # Set preparation sequence
    #         sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
    #         self.readout_sequencer.start()
    #
    #
    #     def set_delay(self, length):
    #
    #         if self.delay_seq_generator is None:
    #             #self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
    #             for ex_seq in self.ex_sequencers:
    #                 ex_seq.set_length(length)
    #                 #ex_seq.set_phase(int(phase / 360 * (2 ** 8)))
    #
    #         else:
    #             if length == self.lengths[0]:
    #                 self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
    #                 self.pre_pause, self.delay_sequence, self.post_pause = self.delay_seq_generator(self.lengths)
    #                 self.prepare_seq[-4] = self.delay_sequence[0]
    #                 #sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq,
    #                                                           #self.control_sequence)
    #                 sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
    #
    #                 for ex_seq in self.ex_sequencers:
    #                     ex_seq.set_length(length)
    #                 self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
    #
    #             else:
    #                 for ex_seq in self.ex_sequencers:
    #                     ex_seq.set_length(length)
    #
    #
    # setter = ParameterSetter()
    #
    # references = {'pre_pulse1':pre_pulse_gate1,
    #               'pre_pulse2':pre_pulse_gate2,
    #               'frequency_controls':device.get_frequency_control_measurement_id(qubit_id=gate.metadata['target_qubit_id']),
    #               'readout_pulse':readout_pulse.id }
    # # references.update(additional_references)
    #
    # if hasattr(measurer, 'references'):
    #     references.update(measurer.references)
    #
    # fitter_arguments = ('iq'+gate.metadata['target_qubit_id'], exp_sin_fitter(), -1,
    #                     np.arange(len((frequencies, gate_setter.frequency_setter, 'frequency'))))
    #
    # metadata = {'qubit_id': gate.metadata['target_qubit_id'],
    #             'transition': transition,
    #           'extra_sweep_args':str(len((frequencies, gate_setter.frequency_setter, 'frequency')))}
    # metadata.update(additional_metadata)
    #
    #
    # measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
    #                                           (frequencies, gate_setter.frequency_setter, 'frequency'),
    #                                           (lengths, setter.set_delay, 'Delay','s'),
    #                                           fitter_arguments = fitter_arguments,
    #                                           measurement_type='Coupler Rabi with pre pulses',
    #                                           metadata=metadata,
    #                                           references=references,
    #                                           on_update_divider=10)
    #
    # for ex_seq in ex_sequencers:
    #     ex_seq.stop()
    # readout_sequencer.stop()
    # return measurement


    #
    # pre_pause = float(gate.metadata['pre_pause'])
    # post_pause = float(gate.metadata['post_pause'])
    # class ParameterSetter:
    #     def __init__(self):
    #         self.amplitude = amplitudes[0]
    #         self.freq = frequencies[0]
    #         self.length = lengths[0]
    #         self.pre_pause_seq = []
    #         self.gate_pulse_seq = []
    #         self.post_pause_seq = []
    #         self.amplitude_setter(amplitudes[0])
    #
    #     def amplitude_setter(self, amplitude):
    #         self.amplitude = amplitude
    #
    #     def frequency_setter(self, freq):
    #         self.freq = freq
    #
    #     def filler_func(self, length):
    #         self.length = length
    #         channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
    #                                               **{gate.metadata['carrier_name']: self.amplitude})
    #
    #         if 'pulse_type' in gate.metadata:
    #             if gate.metadata['pulse_type'] == 'cos':
    #                 frequency = self.freq #float(gate.metadata['frequency'])
    #                 tail_length = float(gate.metadata['tail_length'])
    #                 phase = 0.0
    #                 fast_control = True
    #                 channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
    #                                   c, a in  channel_amplitudes_.items()]
    #                 gate_pulse = [device.pg.pmulti(device, self.length, *tuple(channel_pulses))]
    #         else:
    #             gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
    #                                                                           channel_amplitudes=channel_amplitudes_,
    #                                                                           tail_length=float(
    #                                                                               gate.metadata['tail_length']),
    #                                                                           length=self.length,
    #                                                                           phase=0.0,
    #                                                                           fast_control=True)
    #         self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
    #         self.gate_pulse_seq = gate_pulse
    #         self.post_pause_seq = [device.pg.pmulti(device, post_pause)]
    #
    #         return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    # setter = ParameterSetter()
    #
    # return relaxation.relaxation(device, gate.metadata['target_qubit_id'], '01',
    #                              (frequencies, setter.frequency_setter, 'frequency'),
    #                              lengths=lengths,  delay_seq_generator=setter.filler_func,
    #                              pre_pulse_gate=pre_pulse_gate, measurement_type='Decay_parametric_freq_scan',
    #        additional_references = {'gate':gate.id}, additional_metadata = additional_metadata, sort=sort)




def zgate_amplitude_echo(device, gate, lengths, amplitudes,additional_metadata={}, parallel_gate=None, target_freq_offset=100e9):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude=amplitudes[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []

        def amplitude_setter(self, amplitude):
            self.amplitude = 1j*amplitude

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    #print(frequency)
                    #print(self.length)
                    #channel_pulses = [(c, device.pg.sin, self.amplitude, frequency) for c, a in
                    #                  channel_amplitudes_.metadata.items()]
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                                      c, a in  channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0,
                                                                              fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]
            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

        def filler_func2(self, length):
                self.length = length
                channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                            **{gate.metadata[
                                                                                   'carrier_name']: 0*self.amplitude})

                if 'pulse_type' in gate.metadata:
                    if gate.metadata['pulse_type'] == 'cos':
                        frequency = float(gate.metadata['frequency'])
                        tail_length = float(gate.metadata['tail_length'])
                        phase = 0.0
                        # print(frequency)
                        # print(self.length)
                        # channel_pulses = [(c, device.pg.sin, self.amplitude, frequency) for c, a in
                        #                  channel_amplitudes_.metadata.items()]
                        fast_control = True
                        channel_pulses = [
                            (c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                            c, a in channel_amplitudes_.items()]
                        gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
                else:
                    gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0,
                                                                              fast_control=True)
                self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
                self.gate_pulse_seq = gate_pulse
                self.post_pause_seq = [device.pg.pmulti(device, post_pause)]
                return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    setter = ParameterSetter()
    if parallel_gate is not None:
        return echo.echo_zz(device, gate.metadata['target_qubit_id'], parallel_gate, '01',
                         (amplitudes, setter.amplitude_setter, 'amplitude'),
                         lengths=lengths, target_freq_offset=target_freq_offset,
                         delay_seq_generator=setter.filler_func, delay_seq_generator2=setter.filler_func2, measurement_type='Echo_amplitude_scan_zz',
                         additional_references={'gate': gate.id}, additional_metadata=additional_metadata)
    else:
        return echo.echo(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, measurement_type='Echo_amplitude_scan',
                           additional_references={'gate':gate.id}, additional_metadata=additional_metadata)

def zgate_amplitude_ramsey_crosstalk(device, gate, target_qubit_id, control_qubit_id, lengths, amplitudes,
                                     target_freq_offset=100e6, measurement = 'Ramsey'):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude = amplitudes[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []

        def amplitude_setter(self, amplitude):
            self.amplitude = 1j*amplitude

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{gate.metadata[
                                                                               'carrier_name']: self.amplitude})

            gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes_,
                                                                      tail_length=float(
                                                                          gate.metadata['tail_length']),
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=True)

            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    setter = ParameterSetter()
    if measurement == 'Ramsey':
        measurement_type = 'Ramsey_crosstalk_amplitude_scan'
        function = Ramsey.Ramsey_crosstalk
    elif measurement == 'echo':
        measurement_type = 'echo_crosstalk_amplitude_scan'
        function = echo.echo_crosstalk

    return function(device, target_qubit_id, control_qubit_id, (amplitudes, setter.amplitude_setter, 'amplitude'),
                                       lengths = lengths, target_freq_offset=target_freq_offset,
                                       delay_seq_generator=setter.filler_func, measurement_type=measurement_type,
                                       additional_references={'gate':gate.id})


def zgate_amplitude_relaxation(device, gate, lengths, amplitudes, ex_pulse=None, additional_metadata={}):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude = amplitudes[0]
            self.length = lengths[0]
            self.pre_pause_seq = []
            self.gate_pulse_seq = []
            self.post_pause_seq = []

        def amplitude_setter(self, amplitude):
            self.amplitude = 1j*amplitude

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = float(gate.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    #print(frequency)
                    #print(self.length)
                    #channel_pulses = [(c, device.pg.sin, self.amplitude, frequency) for c, a in
                    #                  channel_amplitudes_.metadata.items()]
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                                      c, a in  channel_amplitudes_.items()]
                    gate_pulse = [device.pg.pmulti(device, self.length + 2 * tail_length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0,
                                                                              fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = gate_pulse
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    setter = ParameterSetter()

    return relaxation.relaxation(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths,
                           delay_seq_generator=setter.filler_func, measurement_type='relaxation_amplitude_scan',
                           ex_pulse=ex_pulse,
                           additional_metadata=additional_metadata,
                           additional_references={'gate':gate.id})
