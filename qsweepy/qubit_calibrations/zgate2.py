from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import relaxation2 as relaxation
from qsweepy.qubit_calibrations import echo2 as echo
from qsweepy.qubit_calibrations import channel_amplitudes
import numpy as np
#from qsweepy.qubit_calibrations import sequence_control
#from qsweepy import zi_scripts


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


def zgate_amplitude_ramsey(device, gate, lengths, amplitudes,additional_metadata={}, target_freq_offset=100e9):
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
                    gate_pulse = [device.pg.pmulti(device, self.length, *tuple(channel_pulses))]
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
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, measurement_type='Ramsey_amplitude_scan',
                           additional_references={'gate':gate.id}, additional_metadata= additional_metadata)

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
                    gate_pulse = [device.pg.pmulti(device, self.length, *tuple(channel_pulses))]
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
                        gate_pulse = [device.pg.pmulti(device, self.length, *tuple(channel_pulses))]
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
                    gate_pulse = [device.pg.pmulti(device, self.length, *tuple(channel_pulses))]
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
