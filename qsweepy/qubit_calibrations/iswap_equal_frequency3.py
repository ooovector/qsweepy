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

            self.lengths = lengths
            self.amplitudes = amplitudes

            self.amplitude = 0
            self.frequency = 0
            self.length = 0

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
                    tail_length = float(gate2.metadata['tail_length'])
                    phase = 0.0
                    fast_control = True
                    channel_pulses = [(c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency2) for
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
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       measurement_type = measurement_type,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement



def sqrt_iswap_amplitude_slice(device, qubit_id,  gate, amplitudes, length, tail_length, z_amplitude, gate2=None,
                      z_gate=None, pre_pulse = None, gate_nums=1, sign=1, Naxuy=True):

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

            self.amplitudes = amplitudes
            self.z_amp = z_amplitude
            self.sign=sign
            self.tail_length = tail_length
            self.amplitude = amplitudes[0]
            self.frequency = 0
            self.full_length = length
            print('self.full_length', self.full_length)

            self.length = self.full_length - 2*self.tail_length

            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            self.pre_pulse = pre_pulse
            self.prepare_seq = []
            self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
            self.pre_pause, self.delay_sequence, self.post_pause = self.filler_func()
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

        def create_z_pulse(self):
            #Z_Gate
            amplitudeZ = 1j*self.z_amp
            channel_amplitudesZ_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{z_gate.metadata['carrier_name']: amplitudeZ})
            z_freq = float(z_gate.metadata['frequency'])
            z_len = float(z_gate.metadata['length'])
            z_tail = 0
            z_phase = 0.0
            initial_phase = 0
            fast_control = False
            channel_pulses = [(c, device.pg.sin, a * np.exp(1j * z_phase), z_freq, initial_phase, fast_control) for
                                      c, a in  channel_amplitudesZ_.items()]
            z_gate_pulse = [device.pg.pmulti(device, z_len+2*z_tail, *tuple(channel_pulses))]

            return z_gate_pulse

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
            self.prepare_seq = []
            self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))
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
            print('self.length', self.length)
            print('self.full_length', self.full_length)
            #Gate 1
            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: 1j*sign*self.amplitude})
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
            frequency2 = float(gate2.metadata['frequency'])
            phase = 0.0
            initial_phase = 0
            fast_control = False
            channel_pulses = [(c, device.pg.sin, a * np.exp(1j * phase), frequency2, initial_phase, fast_control) for
                                      c, a in  channel_amplitudes2_.items()]
            gate2_pulse = [device.pg.pmulti(device, self.full_length, *tuple(channel_pulses))]

            self.pre_pause_seq = [device.pg.pmulti(device, pre_pause)]
            self.gate_pulse_seq = [device.pg.parallel(gate1_pulse[0], gate2_pulse[0])]
            self.post_pause_seq = [device.pg.pmulti(device, post_pause)]

            return self.pre_pause_seq, self.gate_pulse_seq, self.post_pause_seq

    metadata = {'qubit_id': qubit_id,
                'gate_nums': gate_nums,
                'sign': sign,
                'teil': tail_length,
                'full_length': length}

    setter = ParameterSetter()

    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
    measurement_type = 'iswap_rabi_amplitude_slice'

    measurement = device.sweeper.sweep(measurer,
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement