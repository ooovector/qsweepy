from . import excitation_pulse
from . import Ramsey
from . import channel_amplitudes

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


def zgate_amplitude_ramsey(device, gate, lengths, amplitudes, target_freq_offset=100e69):
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

    return Ramsey.Ramsey(device, gate.metadata['target_qubit_id'], '01', (amplitudes, setter.amplitude_setter, 'amplitude'),
                           lengths = lengths, target_freq_offset=target_freq_offset,
                           delay_seq_generator=setter.filler_func, measurement_type='Ramsey_amplitude_scan',
                           additional_references={'gate':gate.id})