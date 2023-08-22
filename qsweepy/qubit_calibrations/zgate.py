from qsweepy.qubit_calibrations import excitation_pulse
from qsweepy.qubit_calibrations import Ramsey, relaxation, echo
from qsweepy.qubit_calibrations import channel_amplitudes


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


def zgate_amplitude_ramsey(device, gate, lengths, amplitudes, target_freq_offset=100e9):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude=None
            self.length = None

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def filler_func(self, length):
            self.length = length
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})

            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'cos':
                    frequency = float(gate.metadata['frequency'])
                    #print(frequency)
                    #print(self.length)
                    channel_pulses = [(c, device.pg.sin, self.amplitude, frequency) for c, a in
                                      channel_amplitudes_.metadata.items()]
                    gate_pulse = [device.pg.pmulti(self.length, *tuple(channel_pulses))]
            else:
                gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes_,
                                                                              tail_length=float(
                                                                                  gate.metadata['tail_length']),
                                                                              length=self.length,
                                                                              phase=0.0)

            return [device.pg.pmulti(pre_pause)]+gate_pulse+[device.pg.pmulti(post_pause)]
    setter = ParameterSetter()

    return Ramsey.Ramsey(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, measurement_type='Ramsey_amplitude_scan',
                           additional_references={'gate':gate.id}, sort=sort)

def zgate_amplitude_ramsey_crosstalk(device, gate, target_qubit_id, control_qubit_id, lengths, amplitudes,
                                     target_freq_offset=100e6, measurement = 'Ramsey'):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude=None

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def filler_func(self, length):
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})
            return [device.pg.pmulti(pre_pause)]+excitation_pulse.get_rect_cos_pulse_sequence(device = device,
                                               channel_amplitudes = channel_amplitudes_,
                                               tail_length = float(gate.metadata['tail_length']),
                                               length = length,
                                               phase = 0.0)+[device.pg.pmulti(post_pause)]
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


def zgate_amplitude_relaxation(device, gate, lengths, amplitudes):
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])
    class ParameterSetter:
        def __init__(self):
            self.amplitude=None

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude

        def filler_func(self, length):
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})
            return [device.pg.pmulti(pre_pause)]+excitation_pulse.get_rect_cos_pulse_sequence(device = device,
                                               channel_amplitudes = channel_amplitudes_ ,
                                               tail_length = float(gate.metadata['tail_length']),
                                               length = length,
                                               phase = 0.0)+[device.pg.pmulti(post_pause)]
    setter = ParameterSetter()

    return relaxation.relaxation(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths,
                           delay_seq_generator=setter.filler_func, measurement_type='relaxation_amplitude_scan',
                           additional_references={'gate':gate.id})
