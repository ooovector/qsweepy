from qsweepy.ponyfiles.data_structures import *
import traceback
#from .import
from qsweepy.libraries import pulses2 as pulses
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import Rabi2 as Rabi
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
import textwrap


def iswap_rabi(device, qubit_id,  gate, amplitudes, lengths, gate2=None, pre_pulse = None, gate_nums = 1, Naxuy=True):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)

    #sequence difinition
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

            self.fast_control=True

            self.lengths = lengths
            self.amplitudes = amplitudes

            self.amplitude = 0
            self.frequency = 0
            self.length = 0
            self.full_length=0

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            self.prepare_seq = []
            self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
            self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
            for i in range(gate_nums):
                if not Naxuy:
                    self.prepare_seq.extend(self.pre_pause)
                    self.prepare_seq.extend(self.delay_sequence)
                    self.prepare_seq.extend(self.post_pause)
                else:
                    self.prepare_seq.extend(self.delay_sequence)

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def frequency_setter(self, frequency):
            self.frequency = frequency
            #self.filler_func()

        def amplitude_setter(self, amplitude):
            if self.fast_control:
                self.amplitude = amplitude
            else:
                self.amplitude = amplitude
                self.length_setter(self.length)
            #self.filler_func()

        def length_setter(self, length):
            self.length = length
            #if length == self.lengths[0]:
            if not self.fast_control:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []
                self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
                self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
                for i in range(gate_nums):
                    if not Naxuy:
                        self.prepare_seq.extend(self.pre_pause)
                        self.prepare_seq.extend(self.delay_sequence)
                        self.prepare_seq.extend(self.post_pause)
                    else:
                        self.prepare_seq.extend(self.delay_sequence)
                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(self.length)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            else:
                if length == self.lengths[0]:
                    self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                    self.prepare_seq = []
                    self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
                    self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
                    for i in range(gate_nums):
                        if not Naxuy:
                            self.prepare_seq.extend(self.pre_pause)
                            self.prepare_seq.extend(self.delay_sequence)
                            self.prepare_seq.extend(self.post_pause)
                        else:
                            self.prepare_seq.extend(self.delay_sequence)
                    sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(self.length)
                    self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                else:
                    for ex_seq in self.ex_sequencers:
                        ex_seq.set_length(self.length)

        def filler_func(self, length):
            self.length = length
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: 1j*self.amplitude})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=float(gate.metadata['tail_length']),
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=True)
            #Gate 2
            amplitude2 = 1j*float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate2.metadata['carrier_name']: amplitude2})
            if 'pulse_type' in gate2.metadata:
                if gate2.metadata['pulse_type'] == 'cos':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    self.fast_control = True #True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, self.fast_control, frequency2) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.length+2*tail_length, *tuple(channel_pulses))]
                elif gate2.metadata['pulse_type'] == 'sin':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length = float(gate.metadata['tail_length'])
                    phase = 0.0
                    initial_phase = 0
                    self.fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, self.fast_control) for
                                      c, a in channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.length+2*tail_length, *tuple(channel_pulses))]

            else:
                gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudes2_,
                                                                        tail_length=float(gate.metadata['tail_length']),
                                                                        length=self.length,
                                                                        phase=0.0,
                                                                        fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
    measurement_type = 'iswap_rabi'
    measurement = device.sweeper.sweep(measurer,
                                       #(frequencies, setter.frequency_setter, 'Frequency', 'Hz'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       measurement_type = measurement_type,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def sqrt_iswap_z_rabi(device, qubit_id,  gate, tail_lengths, amplitudes, lengths, z_amplitudes, gate2=None,
                      z_gate=None, pre_pulse = None, gate_nums = 1, sign=1, Naxuy=True):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)

    #sequence difinition
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

            self.lengths = lengths
            self.amplitudes = amplitudes
            self.z_amps = z_amplitudes
            self.sign=sign
            self.tail_lengths = tail_lengths

            self.amplitude = amplitudes[0]
            self.frequency = 0
            self.full_length = self.lengths[0]
            self.tail_length = self.tail_lengths[0]
            self.length = self.full_length - 2*self.tail_length
            self.z_amp = z_amplitudes[0]

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.pre_pulse = pre_pulse

            self.prepare_seq = []
            self.prepare_seq.extend(self.pre_pulse.get_pulse_sequence(0))
            self.length_setter(self.full_length)

        def create_z_pulse(self):
            #Z_Gate
            amplitudeZ = 1j*self.z_amp
            channel_amplitudesZ_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{z_gate.metadata['carrier_name']: amplitudeZ})
            if 'pulse_type' in z_gate.metadata:
                if z_gate.metadata['pulse_type'] == 'cos':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * z_phase), z_tail, fast_control, z_freq) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
                elif z_gate.metadata['pulse_type'] == 'sin':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * z_phase), z_freq, initial_phase, fast_control) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
            else:
                z_gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudesZ_,
                                                                        tail_length=0,
                                                                        length=float(z_gate.metadata['length']),
                                                                        phase=0.0,
                                                                        fast_control=False)
            return z_gate_pulse

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def z_amp_setter(self, z_amp):
            self.z_amp = z_amp
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def tail_setter(self, tail_length):
            self.tail_length = tail_length
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def length_setter(self, full_length):
            self.full_length = full_length
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            self.prepare_seq = []
            self.prepare_seq.extend(self.pre_pulse.get_pulse_sequence(0))
            self.pre_pause1, self.delay_sequence1, self.post_pause1 = self.filler_func()
            self.pre_pause2, self.delay_sequence2, self.post_pause2 = self.filler_func(self.sign)
            z_gate_pulse = self.create_z_pulse()
            for i in range(gate_nums):
                if not Naxuy:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause1)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(self.post_pause1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause2)
                    self.prepare_seq.extend(self.delay_sequence2)
                    self.prepare_seq.extend(self.post_pause2)
                else:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence2)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def filler_func(self, sign=1):

            self.length = self.full_length - 2*self.tail_length
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: sign*self.amplitude})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=self.tail_length,
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=False)
            #Gate 2
            amplitude2 = 1j*float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate2.metadata['carrier_name']: amplitude2})
            gate2_pulse=[]
            if 'pulse_type' in gate2.metadata:
                if gate2.metadata['pulse_type'] == 'cos':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length2, fast_control, frequency2) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
                elif gate2.metadata['pulse_type'] == 'sin':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
            else:
                gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudes2_,
                                                                        tail_length=self.tail_length,
                                                                        length=self.length,
                                                                        phase=0.0,
                                                                        fast_control=False)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq
    metadata = {'qubit_id':qubit_id,
                'gate_nums':gate_nums,
                'sign': sign,
                'tail': tail_lengths[0],
                'length':lengths[0]}

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
    measurement_type = 'iswap_rabi_z'
    measurement = device.sweeper.sweep(measurer,
                                       (tail_lengths, setter.tail_setter, 'Tail_length', 's'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       (z_amplitudes, setter.z_amp_setter, 'Z_amp', ''),
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement

def sqrt_iswap_z_rabi_11(device, qubit_ids,  gate, tail_lengths, amplitudes, lengths, z_amplitudes, gate2=None,
                      z_gate=None, pre_pulse1 = None, pre_pulse2 = None, gate_nums = 1, sign=1, Naxuy=True,
                      fill=1):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    #sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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

            self.lengths = lengths
            self.amplitudes = amplitudes
            self.z_amps = z_amplitudes
            self.sign=sign
            self.fill = fill
            self.tail_lengths = tail_lengths

            self.amplitude = amplitudes[0]
            self.frequency = 0
            self.full_length = self.lengths[0]
            self.tail_length = self.tail_lengths[0]
            self.length = self.full_length - 2*self.tail_length
            self.z_amp = z_amplitudes[0]

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2

            self.prepare_seq = []
            self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0], self.pre_pulse2.get_pulse_sequence(0)[0])])
            self.length_setter(self.full_length)

        def create_z_pulse(self):
            #Z_Gate
            amplitudeZ = 1j*self.z_amp
            channel_amplitudesZ_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{z_gate.metadata['carrier_name']: amplitudeZ*self.fill})
            if 'pulse_type' in z_gate.metadata:
                if z_gate.metadata['pulse_type'] == 'cos':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * z_phase), z_tail, fast_control, z_freq) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
                elif z_gate.metadata['pulse_type'] == 'sin':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * z_phase), z_freq, initial_phase, fast_control) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
            else:
                z_gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudesZ_,
                                                                        tail_length=0,
                                                                        length=float(z_gate.metadata['length']),
                                                                        phase=0.0,
                                                                        fast_control=False)
            return z_gate_pulse

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def z_amp_setter(self, z_amp):
            self.z_amp = z_amp
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def tail_setter(self, tail_length):
            self.tail_length = tail_length
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def length_setter(self, full_length):
            self.full_length = full_length
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            self.prepare_seq = []
            self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0], self.pre_pulse2.get_pulse_sequence(0)[0])])
            self.pre_pause1, self.delay_sequence1, self.post_pause1 = self.filler_func()
            self.pre_pause2, self.delay_sequence2, self.post_pause2 = self.filler_func(self.sign)
            z_gate_pulse = self.create_z_pulse()
            for i in range(gate_nums):
                if not Naxuy:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause1)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(self.post_pause1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause2)
                    self.prepare_seq.extend(self.delay_sequence2)
                    self.prepare_seq.extend(self.post_pause2)
                else:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence2)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def filler_func(self, sign=1):

            self.length = self.full_length - 2*self.tail_length
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: sign*self.amplitude*self.fill})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=self.tail_length,
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=False)
            #Gate 2
            amplitude2 = 1j*float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate2.metadata['carrier_name']: amplitude2*self.fill})
            gate2_pulse=[]
            if 'pulse_type' in gate2.metadata:
                if gate2.metadata['pulse_type'] == 'cos':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length2, fast_control, frequency2) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
                elif gate2.metadata['pulse_type'] == 'sin':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
            else:
                gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudes2_,
                                                                        tail_length=self.tail_length,
                                                                        length=self.length,
                                                                        phase=0.0,
                                                                        fast_control=False)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    metadata = {'qubit_id':qubit_ids,
                'gate_nums':gate_nums,
                'sign': sign,
                'tail': tail_lengths[0],
                'length':lengths[0]}

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id
    measurement_type = 'iswap_rabi_z'
    measurement = device.sweeper.sweep(measurer,
                                       (tail_lengths, setter.tail_setter, 'Tail_length', 's'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       (z_amplitudes, setter.z_amp_setter, 'Z_amp', ''),
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement

def sqrt_iswap_z_rabi_11_command_table(device, qubit_ids,  gate, tail_lengths, amplitudes, lengths, z_amplitudes, gate2=None,
                      z_gate=None, pre_pulse1 = None, pre_pulse2 = None, gate_nums = 1, sign=1, Naxuy=True,
                      fill=1):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    #sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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

            self.lengths = lengths
            self.amplitudes = amplitudes
            self.z_amps = z_amplitudes
            self.sign=sign
            self.fill = fill
            self.tail_lengths = tail_lengths

            self.amplitude = amplitudes[0]
            self.frequency = 0
            self.full_length = self.lengths[0]
            self.tail_length = self.tail_lengths[0]
            self.length = self.full_length - 2*self.tail_length
            self.z_amp = z_amplitudes[0]

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.pre_pulse1 = pre_pulse1
            self.pre_pulse2 = pre_pulse2

            self.prepare_seq = []
            self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0], self.pre_pulse2.get_pulse_sequence(0)[0])])
            self.length_setter(self.full_length)

        def create_z_pulse(self):
            #Z_Gate
            amplitudeZ = 1j*self.z_amp
            channel_amplitudesZ_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{z_gate.metadata['carrier_name']: amplitudeZ*self.fill})
            if 'pulse_type' in z_gate.metadata:
                if z_gate.metadata['pulse_type'] == 'cos':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * z_phase), z_tail, fast_control, z_freq) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
                elif z_gate.metadata['pulse_type'] == 'sin':
                    z_freq = float(z_gate.metadata['frequency'])
                    z_len = float(z_gate.metadata['length'])
                    z_tail = 0
                    z_phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * z_phase), z_freq, initial_phase, fast_control) for
                                      c, a in  channel_amplitudesZ_.items()]
                    z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]
            else:
                z_gate_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudesZ_,
                                                                        tail_length=0,
                                                                        length=float(z_gate.metadata['length']),
                                                                        phase=0.0,
                                                                        fast_control=False)
            return z_gate_pulse

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def z_amp_setter(self, z_amp):
            self.z_amp = z_amp
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def tail_setter(self, tail_length):
            self.tail_length = tail_length
            #if len(self.lengths) == 1:
            self.length_setter(self.full_length)

        def length_setter(self, full_length):
            self.full_length = full_length
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            self.prepare_seq = []
            self.prepare_seq.extend([device.pg.parallel(self.pre_pulse1.get_pulse_sequence(0)[0], self.pre_pulse2.get_pulse_sequence(0)[0])])
            self.pre_pause1, self.delay_sequence1, self.post_pause1 = self.filler_func()
            self.pre_pause2, self.delay_sequence2, self.post_pause2 = self.filler_func(self.sign)
            z_gate_pulse = self.create_z_pulse()
            for i in range(gate_nums):
                if not Naxuy:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause1)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(self.post_pause1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.pre_pause2)
                    self.prepare_seq.extend(self.delay_sequence2)
                    self.prepare_seq.extend(self.post_pause2)
                else:
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence1)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(z_gate_pulse)
                    self.prepare_seq.extend(self.delay_sequence2)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def filler_func(self, sign=1):

            self.length = self.full_length - 2*self.tail_length
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: sign*self.amplitude*self.fill})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=self.tail_length,
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=False)
            #Gate 2
            amplitude2 = 1j*float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate2.metadata['carrier_name']: amplitude2*self.fill})
            gate2_pulse=[]
            if 'pulse_type' in gate2.metadata:
                if gate2.metadata['pulse_type'] == 'cos':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    fast_control = False
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length2, fast_control, frequency2) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
                elif gate2.metadata['pulse_type'] == 'sin':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length2 = self.tail_length
                    phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]
            else:
                gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudes2_,
                                                                        tail_length=self.tail_length,
                                                                        length=self.length,
                                                                        phase=0.0,
                                                                        fast_control=False)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    metadata = {'qubit_id':qubit_ids,
                'gate_nums':gate_nums,
                'sign': sign,
                'tail': tail_lengths[0],
                'length':lengths[0]}

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse1 is not None:
        references['pre_pulse1'] = pre_pulse1.id
    if pre_pulse2 is not None:
        references['pre_pulse2'] = pre_pulse2.id
    measurement_type = 'iswap_rabi_z'
    measurement = device.sweeper.sweep(measurer,
                                       (tail_lengths, setter.tail_setter, 'Tail_length', 's'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       (z_amplitudes, setter.z_amp_setter, 'Z_amp', ''),
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement

def sqrt_iswap_rabi(device, qubit_id,  gate, amplitudes, lengths, delays, gate2=None, pre_pulse = None,
                      gate_nums = 1, Naxuy=True):

    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)

    #sequence difinition
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

            self.lengths = lengths
            self.amplitudes = amplitudes
            self.delays = delays

            self.amplitude = 0
            self.frequency = 0
            self.length = 0
            self.delay = 0

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

            self.prepare_seq = []
            self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))

            self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
            for i in range(gate_nums):
                if not Naxuy:
                    self.prepare_seq.extend(self.pre_pause)
                    self.prepare_seq.extend(self.delay_sequence)
                    self.prepare_seq.extend(self.post_pause)
                else:
                    self.prepare_seq.extend(self.delay_sequence)

            # Set preparation sequence
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
            self.readout_sequencer.start()

        def frequency_setter(self, frequency):
            self.frequency = frequency
            #self.filler_func()

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            #self.filler_func()

        def delay_setter(self, delay):
            self.delay = delay
            #self.filler_func()

        def length_setter(self, length):
            self.length = length
            if length == self.lengths[0]:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                #self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
                #self.prepare_seq[-2] = self.delay_sequence[0]
                self.prepare_seq = []
                self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
                self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func(self.lengths[0])
                for i in range(gate_nums):
                    if not Naxuy:
                        self.prepare_seq.extend(self.pre_pause)
                        self.prepare_seq.extend(self.delay_sequence)
                        self.prepare_seq.extend(self.post_pause)
                        self.prepare_seq.extend([device.pg.pmulti(device, -self.delay)])
                        self.prepare_seq.extend(self.pre_pause)
                        self.prepare_seq.extend(self.delay_sequence)
                        self.prepare_seq.extend(self.post_pause)
                        self.prepare_seq.extend([device.pg.pmulti(device, -self.delay)])
                    else:
                        self.prepare_seq.extend(self.delay_sequence)
                        self.prepare_seq.extend([device.pg.pmulti(device, -self.delay)])
                        self.prepare_seq.extend(self.delay_sequence)
                        self.prepare_seq.extend([device.pg.pmulti(device, -self.delay)])

                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(self.length)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
            else:
                for ex_seq in self.ex_sequencers:
                    ex_seq.set_length(self.length)

        def filler_func(self, length):
            self.length = length
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=float(gate.metadata['tail_length']),
                                                                      length=self.length,
                                                                      phase=0.0,
                                                                      fast_control=True)
            #Gate 2
            amplitude2 = 1j*float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate2.metadata['carrier_name']: amplitude2})
            if 'pulse_type' in gate2.metadata:
                if gate2.metadata['pulse_type'] == 'cos':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length = float(gate2.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency2) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.length+2*tail_length, *tuple(channel_pulses))]
                elif gate2.metadata['pulse_type'] == 'sin':
                    frequency2 = float(gate2.metadata['frequency'])
                    tail_length = float(gate2.metadata['tail_length'])
                    phase = 0.0
                    initial_phase = 0
                    fast_control = False
                    channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                                      c, a in  channel_amplitudes2_.items()]
                    gate2_pulse = [device.pg.pmulti(device, self.length+2*tail_length, *tuple(channel_pulses))]
            else:
                gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                        channel_amplitudes=channel_amplitudes2_,
                                                                        tail_length=float(gate.metadata['tail_length']),
                                                                        length=self.length,
                                                                        phase=0.0,
                                                                        fast_control=True)
            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
    measurement_type = 'iswap_rabi'
    measurement = device.sweeper.sweep(measurer,
                                       #(frequencies, setter.frequency_setter, 'Frequency', 'Hz'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       (delays, setter.delay_setter, 'Pause', 's'),
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       measurement_type = measurement_type,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def sqrt_iswap_z_rabi_confuse_matrix(device, qubit_ids, correspondence,
                                     phases_1=[0], phases_2=[0], phases_x=[0], phases_3=[0], phases_4=[0],
                                     number_of_circles=1):

    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    # sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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
            print('control_sequence=', control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()


    class ParameterSetter:
        def __init__(self):
            #self.device = device
            self.control = 4
            if len(phases_4) > 1:
                self.control = 4
            elif len(phases_3) > 1:
                self.control = 3
            elif len(phases_x) > 1:
                self.control = 2
            elif len(phases_2) > 1:
                self.control = 1
            elif len(phases_1) > 1:
                self.control = 0

            self.phase_1 = phases_1[0]
            self.phase_2 = phases_2[0]
            self.phase_x = phases_x[0]
            self.phase_3 = phases_3[0]
            self.phase_4 = phases_4[0]

            self.number_of_circles = number_of_circles

            self.interleavers = {}
            self.instructions = []
            self.qubit_ids = qubit_ids
            self.correspondence = correspondence

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

        def set_phase_1(self, phase):
            self.phase_1 = phase
            if self.control==0:
                self.create_program()

        def set_phase_2(self, phase):
            self.phase_2 = phase
            if self.control==1:
                self.create_program()

        def set_phase_x(self, phase):
            self.phase_x = phase
            if self.control==2:
                self.create_program()

        def set_phase_3(self, phase):
            self.phase_3 = phase
            if self.control == 3:
                self.create_program()

        def set_phase_4(self, phase):
            self.phase_4 = phase
            self.create_program()

        def create_program(self):
            self.interleavers = self.create_interleavers()
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            prepare_seq = self.create_hdawg_generator()
            for ex_seq in self.ex_sequencers:
                for register in range(7):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, prepare_seq,
                                                      instructions=self.instructions)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def set_target_state(self, state):
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            for _id, qubit_id in enumerate(self.qubit_ids):
                state_register = 0
                qubit_state = (1 << _id) & state
                if qubit_state:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 1)
                else:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 0)
            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def return_hdawg_program(self, ex_seq):
            random_gate_num = len(self.interleavers)
            assign_waveform_indexes = {}
            definition_part = ''''''
            command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                            "header": {"version": "0.2"},
                            "table": []}

            random_command_id = 0
            waveform_id = -1
            for name, gate in self.interleavers.items():
                for j in range(len(gate['pulses'])):
                    for seq_id, part in gate['pulses'][j][0].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            #if part[0] not in definition_part:
                                #definition_part += part[0]
                                # for entry_table_index_constant in part[2]:
                            table_entry = {'index': random_command_id}
                            random_command_id += 1

                            entry_table_index_constant = part[2][0]
                            # if entry_table_index_constant not in definition_part:
                            # if entry_table_index_constant not in definition_part:
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                definition_part += part[0]
                                definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                definition_part += part[3]
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}


                            random_pulse = part[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)
            two_qubit_gate_index = 6

            table_entry = {'index': random_gate_num}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': False}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': random_gate_num + 1}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': True}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            play_part = textwrap.dedent('''
//  Confuse play part
    executeTableEntry({random_gate_num});
    wait(5);

//Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    repeat({repeat}){{
// First Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(2);
    
//First two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});
    
//Middle X gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(3);
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(4);

//Second two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});

// Second Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(5);}}
    
//Post pulses - Not necessary here
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register1); // variable_register1 = 0 or 1
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register2); // variable_register1 = 0 or 1

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num, repeat=self.number_of_circles))

            self.instructions.append(command_table)
            print(command_table)

            return definition_part, play_part
        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            self.instructions = []
            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])

            return [[pulses, control_seq_ids]]

        def create_interleavers(self):
            interleavers = {}
            # Preparation gates
            # Exitation
            ex1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            ex2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            preparation1 = {'X/2': {'pulses': [device.pg.parallel(ex1.get_pulse_sequence(0)[0], ex2.get_pulse_sequence(0)[0])]}}
            # Identical
            channel_pulses_I1 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            channel_pulses_I2 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            I1_pulse = [device.pg.pmulti(device, float(ex1.metadata['length']), *tuple(channel_pulses_I1))]
            I2_pulse = [device.pg.pmulti(device, float(ex2.metadata['length']), *tuple(channel_pulses_I2))]
            preparation0 = {'I': {'pulses': [device.pg.parallel(I1_pulse[0], I2_pulse[0])]}}

            # First Hadamars definition
            h1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi/2)
            channel_pulses_h1 = [(c, device.pg.gauss_hd_modulation, float(a)*1j*float(h1.metadata['amplitude'])*np.exp(1j*0),
            float(h1.metadata['sigma']), float(h1.metadata['alpha']), self.phase_1)
            for c, a in h1.channel_amplitudes.metadata.items()]
            h1_pulse = [device.pg.pmulti(device, float(h1.metadata['length']), *tuple(channel_pulses_h1))]

            h2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi/2)
            channel_pulses_h2 = [(c, device.pg.gauss_hd_modulation, float(a)*1j*float(h2.metadata['amplitude'])*np.exp(1j*0),
            float(h2.metadata['sigma']), float(h2.metadata['alpha']), self.phase_2)
            for c, a in h2.channel_amplitudes.metadata.items()]
            h2_pulse = [device.pg.pmulti(device, float(h2.metadata['length']), *tuple(channel_pulses_h2))]

            hadamars_1 = {'H1': {'pulses': [device.pg.parallel(h1_pulse[0], h2_pulse[0])]},}

            # Two qubit gate definition
            # Gate 1
            gate1 = device.get_two_qubit_gates()['iSWAP(1,2)2']
            gate2 = device.get_zgates()['z2p_sin']
            full_length = float(gate1.metadata['length'])
            tail_length = float(gate1.metadata['tail_length'])
            length = full_length - 2 * tail_length
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                **{gate1.metadata['carrier_name']: float(gate1.metadata['amplitude'])})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes1_,
                                                                       tail_length=tail_length,
                                                                       length=length,
                                                                       phase=0.0,
                                                                       fast_control=False)
            # Gate 2
            amplitude2 = 1j * float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate2.metadata['carrier_name']: amplitude2})
            frequency2 = float(gate2.metadata['frequency'])
            phase = 0.0
            initial_phase = 0
            fast_control = False
            channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                              c, a in channel_amplitudes2_.items()]
            gate2_pulse = [device.pg.pmulti(device, full_length, *tuple(channel_pulses))]

            two_qubit_gate = {'fSIM': {'pulses': [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]},}

            # Middle_pulse qubit 1
            # Middle_pulse X/2 first
            ex_pulse = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            channel_pulses = [(c, device.pg.gauss_hd_modulation,float(a)*1j*float(ex_pulse.metadata['amplitude'])*np.exp(1j*0),
            float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']), self.phase_x)
            for c, a in ex_pulse.channel_amplitudes.metadata.items()]
            pulse1 = [device.pg.pmulti(device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]
            # Middle_pulse X/2 second
            channel_pulses = [(c, device.pg.gauss_hd_modulation,float(a)*1j*float(ex_pulse.metadata['amplitude'])*np.exp(1j*0),
            float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']),
            self.phase_x + float(ex_pulse.metadata['phase']) + 2 * np.pi * (round(float(ex_pulse.metadata['length'])* 2.4e9/16)*16/2.4e9 * device.pg.channels[c].get_frequency() % 1))
            for c, a in ex_pulse.channel_amplitudes.metadata.items()]
            pulse2 = [device.pg.pmulti(device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]

            middle_pulse = {'X1': {'pulses': pulse1}, 'X2': {'pulses': pulse2}}

            # Second Hadamars definition
            h3 = excitation_pulse.get_excitation_pulse(device, '1', np.pi/2)
            channel_pulses_h3 = [(c, device.pg.gauss_hd_modulation, float(a)*1j*float(h3.metadata['amplitude'])*np.exp(1j*0),
            float(h3.metadata['sigma']), float(h3.metadata['alpha']), self.phase_3)
            for c, a in h3.channel_amplitudes.metadata.items()]
            h3_pulse = [device.pg.pmulti(device, float(h3.metadata['length']), *tuple(channel_pulses_h3))]

            h4 = excitation_pulse.get_excitation_pulse(device, '2', np.pi/2)
            channel_pulses_h4 = [(c, device.pg.gauss_hd_modulation, float(a)*1j*float(h4.metadata['amplitude'])*np.exp(1j*0),
            float(h4.metadata['sigma']), float(h4.metadata['alpha']), self.phase_4)
            for c, a in h4.channel_amplitudes.metadata.items()]
            h4_pulse = [device.pg.pmulti(device, float(h4.metadata['length']), *tuple(channel_pulses_h4))]

            hadamars_2 = {'H2': {'pulses': [device.pg.parallel(h3_pulse[0], h4_pulse[0])]}}

            interleavers.update(preparation0)  # 0
            interleavers.update(preparation1)  # 1
            interleavers.update(hadamars_1)  # 2
            interleavers.update(middle_pulse)  # 3,4
            interleavers.update(hadamars_2)  # 5
            interleavers.update(two_qubit_gate)  # 6s
            return interleavers

    metadata = {'qubit_id': qubit_ids,
                'number_of_circles': number_of_circles}

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id}
    measurement_type = 'iswap_rabi_z_confuse_scan'

    measurement = device.sweeper.sweep(measurer,
                                       (phases_1, setter.set_phase_1, 'Phase 1', ''),
                                       (phases_2, setter.set_phase_2, 'Phase 2', ''),
                                       (phases_x, setter.set_phase_x, 'Phase X', ''),
                                       (phases_3, setter.set_phase_3, 'Phase 3', ''),
                                       (phases_4, setter.set_phase_4, 'Phase 4', ''),
                                       (np.arange(2 ** len(qubit_ids)), setter.set_target_state, 'Target state', ''),
                                        measurement_type=measurement_type,
                                        references=references,
                                        metadata=metadata)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def CZ_confuse_matrix_amplitude(device, qubit_ids, correspondence, amplitudes,
                                     phase_1=0, phase_2=0, phase_x=0, phase_3=0, phase_4=0,
                                     number_of_circles=1,sign=1):
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    # sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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
            print('control_sequence=', control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            self.amplitude = 0

            self.phase_1 = phase_1
            self.phase_2 = phase_2
            self.phase_x = phase_x
            self.phase_3 = phase_3
            self.phase_4 = phase_4

            self.number_of_circles = number_of_circles

            self.interleavers = {}
            self.instructions = []
            self.qubit_ids = qubit_ids
            self.correspondence = correspondence

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer


        def set_amplitude(self, amplitude):
            self.amplitude = amplitude
            self.create_program()

        def create_program(self):
            self.interleavers = self.create_interleavers()
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            prepare_seq = self.create_hdawg_generator()
            for ex_seq in self.ex_sequencers:
                for register in range(7):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, prepare_seq,
                                                      instructions=self.instructions)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def set_target_state(self, state):
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            for _id, qubit_id in enumerate(self.qubit_ids):
                state_register = 0
                qubit_state = (1 << _id) & state
                if qubit_state:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 1)
                else:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 0)
            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def return_hdawg_program(self, ex_seq):
            random_gate_num = len(self.interleavers)
            assign_waveform_indexes = {}
            definition_part = ''''''
            command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                             "header": {"version": "0.2"},
                             "table": []}

            random_command_id = 0
            waveform_id = -1
            for name, gate in self.interleavers.items():
                for j in range(len(gate['pulses'])):
                    for seq_id, part in gate['pulses'][j][0].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            #if part[0] not in definition_part:
                                #definition_part += part[0]
                                # for entry_table_index_constant in part[2]:
                            table_entry = {'index': random_command_id}
                            random_command_id += 1

                            entry_table_index_constant = part[2][0]
                            # if entry_table_index_constant not in definition_part:
                            # if entry_table_index_constant not in definition_part:
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                definition_part += part[0]

                                definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                definition_part += part[3]
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = part[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)
            two_qubit_gate_index = 6

            table_entry = {'index': random_gate_num}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': False}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': random_gate_num + 1}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': True}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            play_part = textwrap.dedent('''
//  Confuse play part
    executeTableEntry({random_gate_num});
    wait(5);

//Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    repeat({repeat}){{
// First Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(2);

//First two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});

//Middle X gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(3);
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(4);

//Second two qubit gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});

// Second Hadamars group
        executeTableEntry({random_gate_num}+1);
        executeTableEntry(5);}}

//Post pulses - Not necessary here
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register1); // variable_register1 = 0 or 1
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register2); // variable_register1 = 0 or 1

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num,
                               repeat=self.number_of_circles))

            self.instructions.append(command_table)
            print(command_table)

            return definition_part, play_part

        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            self.instructions = []
            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])

            return [[pulses, control_seq_ids]]

        def create_interleavers(self):
            interleavers = {}
            # Preparation gates
            # Exitation
            ex1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            ex2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            preparation1 = {
                'X/2': {'pulses': [device.pg.parallel(ex1.get_pulse_sequence(0)[0], ex2.get_pulse_sequence(0)[0])]}}
            # Identical
            channel_pulses_I1 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            channel_pulses_I2 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            I1_pulse = [device.pg.pmulti(device, float(ex1.metadata['length']), *tuple(channel_pulses_I1))]
            I2_pulse = [device.pg.pmulti(device, float(ex2.metadata['length']), *tuple(channel_pulses_I2))]
            preparation0 = {'I': {'pulses': [device.pg.parallel(I1_pulse[0], I2_pulse[0])]}}

            # First Hadamars definition
            h1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            channel_pulses_h1 = [
                (c, device.pg.gauss_hd_modulation, float(a) * 1j * float(h1.metadata['amplitude']) * np.exp(1j * 0),
                 float(h1.metadata['sigma']), float(h1.metadata['alpha']), self.phase_1)
                for c, a in h1.channel_amplitudes.metadata.items()]
            h1_pulse = [device.pg.pmulti(device, float(h1.metadata['length']), *tuple(channel_pulses_h1))]

            h2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            channel_pulses_h2 = [
                (c, device.pg.gauss_hd_modulation, float(a) * 1j * float(h2.metadata['amplitude']) * np.exp(1j * 0),
                 float(h2.metadata['sigma']), float(h2.metadata['alpha']), self.phase_2)
                for c, a in h2.channel_amplitudes.metadata.items()]
            h2_pulse = [device.pg.pmulti(device, float(h2.metadata['length']), *tuple(channel_pulses_h2))]

            hadamars_1 = {'H1': {'pulses': [device.pg.parallel(h1_pulse[0], h2_pulse[0])]}, }

            # Two qubit gate definition
            # Gate 1
            # gate1 = device.get_two_qubit_gates()['iSWAP(1,2)2']
            # gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)']
            gate1 = device.get_two_qubit_gates()['iSWAP(1,2)_CZ']
            gate2 = device.get_zgates()['z2p_sin']
            full_length = float(gate1.metadata['length'])
            tail_length = float(gate1.metadata['tail_length'])
            length = full_length - 2 * tail_length
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate1.metadata['carrier_name']: self.amplitude})# float(gate1.metadata['amplitude'])})
            gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                       channel_amplitudes=channel_amplitudes1_,
                                                                       tail_length=tail_length,
                                                                       length=length,
                                                                       phase=0.0,
                                                                       fast_control=False)
            # Gate 2
            amplitude2 = 1j * float(gate2.metadata['amplitude'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate2.metadata['carrier_name']: amplitude2})
            frequency2 = float(gate2.metadata['frequency'])
            phase = 0.0
            initial_phase = 0
            fast_control = False
            channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                              c, a in channel_amplitudes2_.items()]
            gate2_pulse = [device.pg.pmulti(device, full_length, *tuple(channel_pulses))]

            two_qubit_gate = {'fSIM': {'pulses': [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]}, }

            # Middle_pulse qubit 1
            # Middle_pulse X/2 first
            ex_pulse = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            channel_pulses = [(c, device.pg.gauss_hd_modulation,float(a) * 1j * float(ex_pulse.metadata['amplitude']) * np.exp(1j * 0),
            float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']), self.phase_x)
            for c, a in ex_pulse.channel_amplitudes.metadata.items()]
            pulse1 = [device.pg.pmulti(device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]
            # Middle_pulse X/2 second
            channel_pulses = [(c, device.pg.gauss_hd_modulation,float(a) * 1j * float(ex_pulse.metadata['amplitude']) * np.exp(1j * 0),
            float(ex_pulse.metadata['sigma']), float(ex_pulse.metadata['alpha']),self.phase_x +
            float(ex_pulse.metadata['phase']) + 2 * np.pi * (round(float(ex_pulse.metadata['length'])* 2.4e9/16)*16/2.4e9 * device.pg.channels[c].get_frequency() % 1))
            for c, a in ex_pulse.channel_amplitudes.metadata.items()]
            pulse2 = [device.pg.pmulti(device, float(ex_pulse.metadata['length']), *tuple(channel_pulses))]

            middle_pulse = {'X1': {'pulses': pulse1}, 'X2': {'pulses': pulse2}}

            # Second Hadamars definition
            h3 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            channel_pulses_h3 = [
                (c, device.pg.gauss_hd_modulation, float(a) * 1j * float(h3.metadata['amplitude']) * np.exp(1j * 0),
                 float(h3.metadata['sigma']), float(h3.metadata['alpha']), self.phase_3)
                for c, a in h3.channel_amplitudes.metadata.items()]
            h3_pulse = [device.pg.pmulti(device, float(h3.metadata['length']), *tuple(channel_pulses_h3))]

            h4 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            channel_pulses_h4 = [
                (c, device.pg.gauss_hd_modulation, float(a) * 1j * float(h4.metadata['amplitude']) * np.exp(1j * 0),
                 float(h4.metadata['sigma']), float(h4.metadata['alpha']), self.phase_4)
                for c, a in h4.channel_amplitudes.metadata.items()]
            h4_pulse = [device.pg.pmulti(device, float(h4.metadata['length']), *tuple(channel_pulses_h4))]

            hadamars_2 = {'H2': {'pulses': [device.pg.parallel(h3_pulse[0], h4_pulse[0])]}}

            interleavers.update(preparation0)  # 0
            interleavers.update(preparation1)  # 1
            interleavers.update(hadamars_1)  # 2
            interleavers.update(middle_pulse)  # 3,4
            interleavers.update(hadamars_2)  # 5
            interleavers.update(two_qubit_gate)  # 6s
            return interleavers

    metadata = {'qubit_id': qubit_ids,
                'number_of_circles': number_of_circles}

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id}
    measurement_type = 'iswap_rabi_z_confuse_scan'

    measurement = device.sweeper.sweep(measurer,
                                       (amplitudes, setter.set_amplitude, 'Amplitude iswap', 'V'),
                                       (np.arange(2 ** len(qubit_ids)), setter.set_target_state, 'Target state', ''),
                                       measurement_type=measurement_type,
                                       references=references,
                                       metadata=metadata)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def cz_unite_gate_confuse_matrix(device, qubit_ids, correspondence,
                                     phases_1=[0], phases_2=[0], phases_x=[0], phases_3=[0], phases_4=[0],
                                     number_of_circles=1, virtual_phase1=0, virtual_phase2=0,sign=1):
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    # sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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
            print('control_sequence=', control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            # self.device = device
            self.control = 4
            if len(phases_4) > 1:
                self.control = 4
            elif len(phases_3) > 1:
                self.control = 3
            elif len(phases_x) > 1:
                self.control = 2
            elif len(phases_2) > 1:
                self.control = 1
            elif len(phases_1) > 1:
                self.control = 0

            self.phase_1 = phases_1[0]
            self.phase_2 = phases_2[0]
            self.phase_x = phases_x[0]
            self.phase_3 = phases_3[0]
            self.phase_4 = phases_4[0]

            self.virtual_phase_1 = virtual_phase1
            self.virtual_phase_2 = virtual_phase2

            self.number_of_circles = number_of_circles

            self.interleavers = {}
            self.instructions = []
            self.qubit_ids = qubit_ids
            self.correspondence = correspondence

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

        def set_phase_1(self, phase):
            self.phase_1 = phase
            if self.control == 0:
                self.create_program()

        def set_phase_2(self, phase):
            self.phase_2 = phase
            if self.control == 1:
                self.create_program()

        def set_phase_x(self, phase):
            self.phase_x = phase
            if self.control == 2:
                self.create_program()

        def set_phase_3(self, phase):
            self.phase_3 = phase
            if self.control == 3:
                self.create_program()

        def set_phase_4(self, phase):
            self.phase_4 = phase
            self.create_program()

        def create_program(self):
            self.interleavers = self.create_interleavers()
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            prepare_seq = self.create_hdawg_generator()
            for ex_seq in self.ex_sequencers:
                for register in range(7):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, prepare_seq,
                                                      instructions=self.instructions)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def set_target_state(self, state):
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            for _id, qubit_id in enumerate(self.qubit_ids):
                state_register = 0
                qubit_state = (1 << _id) & state
                if qubit_state:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 1)
                else:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 0)
            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def return_hdawg_program(self, ex_seq):
            random_gate_num = len(self.interleavers)
            assign_waveform_indexes = {}
            definition_part = ''''''
            command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                             "header": {"version": "0.2"},
                             "table": []}

            random_command_id = 0
            waveform_id = -1
            for name, gate in self.interleavers.items():
                for j in range(len(gate['pulses'])):
                    for seq_id, part in gate['pulses'][j][0].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            # if part[0] not in definition_part:
                            # definition_part += part[0]
                            # for entry_table_index_constant in part[2]:
                            table_entry = {'index': random_command_id}
                            random_command_id += 1

                            entry_table_index_constant = part[2][0]
                            # if entry_table_index_constant not in definition_part:
                            # if entry_table_index_constant not in definition_part:
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                if part[0] not in definition_part:
                                    definition_part += part[0]
                                definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                definition_part += part[3]
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = part[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)
            two_qubit_gate_index = 2

            table_entry = {'index': random_gate_num}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': False}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': random_gate_num + 1}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': True}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            play_part = textwrap.dedent('''
//  Confuse play part
    executeTableEntry({random_gate_num});
    wait(5);

//Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    
    repeat({repeat}){{
// CZ group gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});}}

//Post pulses - Not necessary here
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register1); // variable_register1 = 0 or 1
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register2); // variable_register1 = 0 or 1

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num,
                               repeat=self.number_of_circles))

            self.instructions.append(command_table)
            print(command_table)

            return definition_part, play_part

        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            self.instructions = []
            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])

            return [[pulses, control_seq_ids]]

        def create_interleavers(self):
            interleavers = {}
            # Preparation gates
            # Exitation
            ex1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            ex2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            preparation1 = {
                'X/2': {'pulses': [device.pg.parallel(ex1.get_pulse_sequence(0)[0], ex2.get_pulse_sequence(0)[0])]}}
            # Identical
            channel_pulses_I1 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            channel_pulses_I2 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            I1_pulse = [device.pg.pmulti(device, float(ex1.metadata['length']), *tuple(channel_pulses_I1))]
            I2_pulse = [device.pg.pmulti(device, float(ex2.metadata['length']), *tuple(channel_pulses_I2))]
            preparation0 = {'I': {'pulses': [device.pg.parallel(I1_pulse[0], I2_pulse[0])]}}

            # Two qubit gate definition

            ex_pulse1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2, preferred_length=13.3e-9)
            ex_pulse2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2, preferred_length=13.3e-9)
            # gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)2']
            # gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)']
            gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)_CZ']
            z_gate2 = device.get_zgates()['z2p_sin']

            # Qubit 1 gate

            # hadamar_cz_sequence_qubit_1(self, channel, length, length1, length_fsim, amp_x, sigma, phase_1=0,
            #                             phase_x1=0, phase_x1=0, phase_3=0, virtual_phase=0):
            channel_pulses1 = [(c, device.pg.hadamar_cz_sequence_qubit_1, float(ex_pulse1.metadata['length']),
                               float(gate_fsim.metadata['length']), float(a)*1j*float(ex_pulse1.metadata['amplitude'])*np.exp(1j*0),
                               float(ex_pulse1.metadata['sigma']), self.phase_1, 0*self.phase_x,
                               0*self.phase_x + 0*float(ex_pulse1.metadata['phase'])+0*2*np.pi*(round(float(ex_pulse1.metadata['length'])*2.4e9/16)*16/2.4e9*device.pg.channels[c].get_frequency()%1),
                               self.phase_3,
                               self.virtual_phase_1) for c, a in ex_pulse1.channel_amplitudes.metadata.items()]
            qubit1_pulse = [device.pg.pmulti(device, 4*float(ex_pulse1.metadata['length'])+2*float(gate_fsim.metadata['length']), *tuple(channel_pulses1))]

            # Qubit 2 gate

            # hadamar_cz_sequence_qubit_2(self, channel, length, length1, length_fsim, amp_x, sigma, freq, amp_sin,
            #                             phase_2=0, phase_4=0, virtual_phase=0)
            channel_pulses2 = [(c, device.pg.hadamar_cz_sequence_qubit_2, float(ex_pulse2.metadata['length']),
                               float(gate_fsim.metadata['length']), float(a)*1j*float(ex_pulse2.metadata['amplitude'])*np.exp(1j*0),
                               float(ex_pulse2.metadata['sigma']), float(z_gate2.metadata['frequency']),
                               1j*float(z_gate2.metadata['amplitude']), self.phase_2, 0*self.phase_x,
                               self.phase_x + float(ex_pulse2.metadata['phase'])+2*np.pi*(round(float(ex_pulse2.metadata['length'])*2.4e9/16)*16/2.4e9*device.pg.channels[c].get_frequency()%1), self.phase_4,
                               self.virtual_phase_2) for c, a in ex_pulse2.channel_amplitudes.metadata.items()]
            qubit2_pulse = [device.pg.pmulti(device, 4*float(ex_pulse2.metadata['length'])+2*float(gate_fsim.metadata['length']), *tuple(channel_pulses2))]

            #Coupler gate
            #hadamar_cz_sequence_coupler(self, channel, length, length1, length_fsim, amp, length_tail):
            channel_amplitudesC_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate_fsim.metadata['carrier_name']: float(
                                                                             gate_fsim.metadata['amplitude'])})
            channel_pulsesC = [(c, device.pg.hadamar_cz_sequence_coupler, float(ex_pulse2.metadata['length']),
                               float(gate_fsim.metadata['length']), a*np.exp(1j*0),
                               float(gate_fsim.metadata['tail_length']), sign) for c, a in channel_amplitudesC_.items()]
            coupler_pulse = [device.pg.pmulti(device, 4*float(ex_pulse2.metadata['length'])+2*float(gate_fsim.metadata['length']), *tuple(channel_pulsesC))]

            two_qubit_gate_group = {'CZ': {'pulses': [device.pg.parallel(qubit1_pulse[0], qubit2_pulse[0], coupler_pulse[0])]},}

            interleavers.update(preparation0)  # 0
            interleavers.update(preparation1)  # 1
            interleavers.update(two_qubit_gate_group)  # 6s
            return interleavers

    metadata = {'qubit_id': qubit_ids,
                'number_of_circles': number_of_circles}

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id}
    measurement_type = 'iswap_rabi_z_confuse_scan'

    measurement = device.sweeper.sweep(measurer,
                                       (phases_1, setter.set_phase_1, 'Phase 1', ''),
                                       (phases_2, setter.set_phase_2, 'Phase 2', ''),
                                       (phases_x, setter.set_phase_x, 'Phase X', ''),
                                       (phases_3, setter.set_phase_3, 'Phase 3', ''),
                                       (phases_4, setter.set_phase_4, 'Phase 4', ''),
                                       (np.arange(2 ** len(qubit_ids)), setter.set_target_state, 'Target state', ''),
                                       measurement_type=measurement_type,
                                       references=references,
                                       metadata=metadata)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement


def cz_unite_gate_confuse_matrix_amplitude(device, qubit_ids, correspondence,amplitudes,
                                     phase_1=0, phase_2=0, phase_x=0, phase_3=0, phase_4=0,
                                     number_of_circles=1, sign=1):
    readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    # sequence difinition
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
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
            print('control_sequence=', control_sequence)
        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        # device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()

    class ParameterSetter:
        def __init__(self):
            # self.device = device


            self.phase_1 = phase_1
            self.phase_2 = phase_2
            self.phase_x = phase_x
            self.phase_3 = phase_3
            self.phase_4 = phase_4
            self.amplitude = amplitudes[0]

            self.number_of_circles = number_of_circles

            self.interleavers = {}
            self.instructions = []
            self.qubit_ids = qubit_ids
            self.correspondence = correspondence

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer

        def set_amplitude(self, amplitude):
            self.amplitude = amplitude
            self.create_program()

        def create_program(self):
            self.interleavers = self.create_interleavers()
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            prepare_seq = self.create_hdawg_generator()
            for ex_seq in self.ex_sequencers:
                for register in range(7):
                    ex_seq.awg.set_register(ex_seq.params['sequencer_id'], register, 0)
            sequence_control.set_preparation_sequence(device, self.ex_sequencers, prepare_seq,
                                                      instructions=self.instructions)
            self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def set_target_state(self, state):
            # self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            for _id, qubit_id in enumerate(self.qubit_ids):
                state_register = 0
                qubit_state = (1 << _id) & state
                if qubit_state:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 1)
                else:
                    for ex_seq in self.ex_sequencers:
                        if self.correspondence[qubit_id]['sequencer_id'] == ex_seq.params['sequencer_id']:
                            ex_seq.awg.set_register(ex_seq.params['sequencer_id'], state_register, 0)
            # self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])

        def return_hdawg_program(self, ex_seq):
            random_gate_num = len(self.interleavers)
            assign_waveform_indexes = {}
            definition_part = ''''''
            command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
                             "header": {"version": "0.2"},
                             "table": []}

            random_command_id = 0
            waveform_id = -1
            for name, gate in self.interleavers.items():
                for j in range(len(gate['pulses'])):
                    for seq_id, part in gate['pulses'][j][0].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            # if part[0] not in definition_part:
                            # definition_part += part[0]
                            # for entry_table_index_constant in part[2]:
                            table_entry = {'index': random_command_id}
                            random_command_id += 1

                            entry_table_index_constant = part[2][0]
                            # if entry_table_index_constant not in definition_part:
                            # if entry_table_index_constant not in definition_part:
                            if entry_table_index_constant not in assign_waveform_indexes.keys():
                                waveform_id += 1
                                assign_waveform_indexes[entry_table_index_constant] = waveform_id
                                if part[0] not in definition_part:
                                    definition_part += part[0]
                                definition_part += 'const ' + entry_table_index_constant + '={_id};'.format(
                                    _id=waveform_id)
                                definition_part += part[3]
                                table_entry['waveform'] = {'index': waveform_id}
                            else:
                                table_entry['waveform'] = {'index': assign_waveform_indexes[entry_table_index_constant]}

                            random_pulse = part[4]
                            if 'phase0' in random_pulse:
                                table_entry['phase0'] = random_pulse['phase0']
                            if 'phase1' in random_pulse:
                                table_entry['phase1'] = random_pulse['phase1']
                            command_table['table'].append(table_entry)
            two_qubit_gate_index = 2

            table_entry = {'index': random_gate_num}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': False}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            table_entry = {'index': random_gate_num + 1}
            # table_entry['amplitude0'] = {'value': 1}
            table_entry['phase0'] = {'value': 0.0, 'increment': True}
            # table_entry['amplitude1'] = {'value': 1}
            table_entry['phase1'] = {'value': 90.0, 'increment': False}
            command_table['table'].append(table_entry)

            play_part = textwrap.dedent('''
//  Confuse play part
    executeTableEntry({random_gate_num});
    wait(5);

//Pre pulses
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1
    executeTableEntry({random_gate_num}+1);
    executeTableEntry(variable_register0); // variable_register0=0 or 1

    repeat({repeat}){{
// CZ group gate
        executeTableEntry({random_gate_num}+1);
        executeTableEntry({two_qubit_gate_index});}}

//Post pulses - Not necessary here
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register1); // variable_register1 = 0 or 1
    //executeTableEntry({random_gate_num}+1);
    //executeTableEntry(variable_register2); // variable_register1 = 0 or 1

    executeTableEntry({random_gate_num});
    resetOscPhase();'''.format(two_qubit_gate_index=two_qubit_gate_index, random_gate_num=random_gate_num,
                               repeat=self.number_of_circles))

            self.instructions.append(command_table)
            print(command_table)

            return definition_part, play_part

        def create_hdawg_generator(self):
            pulses = {}
            control_seq_ids = []
            self.instructions = []
            for ex_seq in self.ex_sequencers:
                pulses[ex_seq.params['sequencer_id']] = self.return_hdawg_program(ex_seq)
                control_seq_ids.append(ex_seq.params['sequencer_id'])

            return [[pulses, control_seq_ids]]

        def create_interleavers(self):
            interleavers = {}
            # Preparation gates
            # Exitation
            ex1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2)
            ex2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2)
            preparation1 = {
                'X/2': {'pulses': [device.pg.parallel(ex1.get_pulse_sequence(0)[0], ex2.get_pulse_sequence(0)[0])]}}
            # Identical
            channel_pulses_I1 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            channel_pulses_I2 = [(c, device.pg.virtual_z, 0, False) for c, a in ex1.channel_amplitudes.metadata.items()]
            I1_pulse = [device.pg.pmulti(device, float(ex1.metadata['length']), *tuple(channel_pulses_I1))]
            I2_pulse = [device.pg.pmulti(device, float(ex2.metadata['length']), *tuple(channel_pulses_I2))]
            preparation0 = {'I': {'pulses': [device.pg.parallel(I1_pulse[0], I2_pulse[0])]}}

            # Two qubit gate definition

            ex_pulse1 = excitation_pulse.get_excitation_pulse(device, '1', np.pi / 2, preferred_length=13.3e-9)
            ex_pulse2 = excitation_pulse.get_excitation_pulse(device, '2', np.pi / 2, preferred_length=13.3e-9)
            # gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)2']
            #gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)']
            gate_fsim = device.get_two_qubit_gates()['iSWAP(1,2)_CZ']
            z_gate2 = device.get_zgates()['z2p_sin']

            # Qubit 1 gate
            self.virtual_phase_1 = 0
            # hadamar_cz_sequence_qubit_1(self, channel, length, length1, length_fsim, amp_x, sigma, phase_1=0,
            #                             phase_x1=0, phase_x1=0, phase_3=0, virtual_phase=0):
            channel_pulses1 = [(c, device.pg.hadamar_cz_sequence_qubit_1, float(ex_pulse1.metadata['length']),
                                float(gate_fsim.metadata['length']),
                                float(a) * 1j * float(ex_pulse1.metadata['amplitude']) * np.exp(1j * 0),
                                float(ex_pulse1.metadata['sigma']), self.phase_1, 0 * self.phase_x,
                                self.phase_x + float(ex_pulse1.metadata['phase']) + 2 * np.pi * (
                                            round(float(ex_pulse1.metadata['length']) * 2.4e9 / 16) * 16 / 2.4e9 *
                                            device.pg.channels[c].get_frequency() % 1),
                                self.phase_3,
                                self.virtual_phase_1) for c, a in ex_pulse1.channel_amplitudes.metadata.items()]
            qubit1_pulse = [device.pg.pmulti(device, 4 * float(ex_pulse1.metadata['length']) + 2 * float(
                gate_fsim.metadata['length']), *tuple(channel_pulses1))]

            # Qubit 2 gate
            self.virtual_phase_2 = 0
            # hadamar_cz_sequence_qubit_2(self, channel, length, length1, length_fsim, amp_x, sigma, freq, amp_sin,
            #                             phase_2=0, phase_4=0, virtual_phase=0)
            channel_pulses2 = [(c, device.pg.hadamar_cz_sequence_qubit_2, float(ex_pulse2.metadata['length']),
                                float(gate_fsim.metadata['length']),
                                float(a) * 1j * float(ex_pulse2.metadata['amplitude']) * np.exp(1j * 0),
                                float(ex_pulse2.metadata['sigma']), float(z_gate2.metadata['frequency']),
                                1j * float(z_gate2.metadata['amplitude']), self.phase_2, self.phase_4,
                                self.virtual_phase_2) for c, a in ex_pulse2.channel_amplitudes.metadata.items()]
            qubit2_pulse = [device.pg.pmulti(device, 4 * float(ex_pulse2.metadata['length']) + 2 * float(
                gate_fsim.metadata['length']), *tuple(channel_pulses2))]

            # Coupler gate
            # hadamar_cz_sequence_coupler(self, channel, length, length1, length_fsim, amp, length_tail):
            channel_amplitudesC_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{gate_fsim.metadata['carrier_name']: float(self.amplitude)})
            channel_pulsesC = [(c, device.pg.hadamar_cz_sequence_coupler, float(ex_pulse2.metadata['length']),
                                float(gate_fsim.metadata['length']), a * np.exp(1j * 0),
                                float(gate_fsim.metadata['tail_length']),sign) for c, a in channel_amplitudesC_.items()]
            coupler_pulse = [device.pg.pmulti(device, 4 * float(ex_pulse2.metadata['length']) + 2 * float(
                gate_fsim.metadata['length']), *tuple(channel_pulsesC))]

            two_qubit_gate_group = {
                'CZ': {'pulses': [device.pg.parallel(qubit1_pulse[0], qubit2_pulse[0], coupler_pulse[0])]}, }

            interleavers.update(preparation0)  # 0
            interleavers.update(preparation1)  # 1
            interleavers.update(two_qubit_gate_group)  # 6s
            return interleavers

    metadata = {'qubit_id': qubit_ids,
                'number_of_circles': number_of_circles}

    setter = ParameterSetter()

    references = {'readout_pulse': readout_pulse.id}
    measurement_type = 'iswap_rabi_z_confuse_scan'

    measurement = device.sweeper.sweep(measurer,
                                       (amplitudes, setter.set_amplitude, 'Amplitude', 'V'),
                                       (np.arange(2 ** len(qubit_ids)), setter.set_target_state, 'Target state', ''),
                                       measurement_type=measurement_type,
                                       references=references,
                                       metadata=metadata)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement
