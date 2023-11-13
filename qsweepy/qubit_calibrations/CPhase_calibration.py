from qsweepy.ponyfiles.data_structures import *
import traceback
#from .import
from qsweepy.libraries import pulses2 as pulses
from qsweepy.fitters.exp_sin import exp_sin_fitter
from qsweepy.fitters.single_period_sin import SinglePeriodSinFitter
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.qubit_calibrations import Rabi2 as Rabi
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
from qsweepy.qubit_calibrations import Ramsey2 as Ramsey
from qsweepy.qubit_calibrations import calibrated_readout2 as calibrated_readout
from qsweepy.qubit_calibrations.readout_pulse2 import *
import textwrap

def CPhase_calibration(device, qubit_ids, gate, amplitude, length, transition='01', phis=None, tail_length=0e-9, readout_delay=0e-9,
                    *extra_sweep_args, repeats=1, pulse_U=None, sign=False, additional_metadata={}, gate_freq=None):
    """
    Provide CPhase calibration
    :param device:
    :param qubit_ids: [control qubit id in CPhase, target qubit in CPhase]
    :param pulse_U: additional prepulse on a control qubit identity  or X gate
    """
    control_qubit_id, qubit_id = qubit_ids
    assert(pulse_U == 'I' or pulse_U == 'X')
    # assert(qubit_id == '1' or qubit_id == '2')


    #from calibrated_readout import get_calibrated_measurer

    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])


    #readout_pulse, measurer = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    #sequence difinition

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
        ex_seq.start(holder=1)
        ex_sequencers.append(ex_seq)
    readout_sequencer = sequence_control.define_readout_control_seq(device, readout_pulse)
    readout_sequencer.start()


    # lengths = np.array([length])
    # times = np.zeros(len(lengths))
    # for _i in range(len(lengths)):
    #     times[_i] = int(round(lengths[_i] * control_sequence.clock))
    # lengths = times / control_sequence.clock
    # length = lengths[0]

    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{gate.metadata['carrier_name']: 1j*amplitude})
    channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{gate.metadata['carrier_name']: -1j * amplitude})
    class ParameterSetter:
        def __init__(self):

            self.length = length - 2*tail_length
            self.phis = np.asarray(phis)
            self.tail_length = tail_length
            self.amplitude = amplitude
            self.gate_freq = gate_freq
            if self.gate_freq is None:
                self.gate_freq = float(gate.metadata['frequency'])
            self.frequency = 0
            self.sign = sign


            self.ex_sequencers = ex_sequencers
            self.readout_sequencer = readout_sequencer
            # self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id='1', rotation_angle=np.pi/2)
            # self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id='2', rotation_angle=np.pi/2)
            self.pre_pulse1 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id,
                                                                    rotation_angle=np.pi / 2)
            self.pre_pulse2 = excitation_pulse.get_excitation_pulse(device=device, qubit_id=control_qubit_id,
                                                                    rotation_angle=np.pi / 2)
            self.control_sequence = control_sequence
            self.control_qubit_sequence = control_qubit_sequence

            self.prepare_seq = []
            self.phi_setter(self.phis[0])

        def frequency_setter(self, frequency):
            self.frequency = frequency

        def phi_setter(self, phi):

            if phi==self.phis[0]:
                self.readout_sequencer.awg.stop_seq(self.readout_sequencer.params['sequencer_id'])
                self.prepare_seq = []

                if pulse_U == 'I':
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                elif pulse_U == 'X':
                    self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                    self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                else:
                    raise ValueError('Prepulse can be only I or X!')

                # if qubit_id == '1':
                #     if pulse_U == 'I':
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #     else:
                #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #         # self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # else:
                #     if pulse_U == 'I':
                #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                #     else:
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                #         # self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])
                #         self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))
                # self.prepare_seq.extend([device.pg.pmulti(device, 20e-9)])

                full_length = float(gate.metadata['length'])
                tail_length = float(gate.metadata['tail_length'])
                length = full_length - 2 * tail_length

                if 'pulse_type' in gate.metadata:
                    if gate.metadata['pulse_type'] == 'cos':

                        channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                                    **{gate.metadata[
                                                                                           'carrier_name']: self.amplitude})
                        frequency = self.gate_freq#float(gate.metadata['frequency'])
                        tail_length = float(gate.metadata['tail_length'])
                        phase = 0.0
                        fast_control = True
                        channel_pulses = [
                            (c, device.pg.rect_cos, a * np.exp(1j * phase), tail_length, fast_control, frequency) for
                            c, a in channel_amplitudes_.items()]
                        gate1_pulse = [device.pg.pmulti(device, length + 2 * tail_length, *tuple(channel_pulses))]


                        print('pulse_type=cos')

                    else:
                        gate1_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                                   channel_amplitudes=channel_amplitudes1_,
                                                                                   tail_length=self.tail_length,
                                                                                   length=self.length,
                                                                                   phase=0.0,
                                                                                   fast_control=False)
                        gate2_pulse = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                                   channel_amplitudes=channel_amplitudes2_,
                                                                                   tail_length=self.tail_length,
                                                                                   length=self.length,
                                                                                   phase=0.0,
                                                                                   fast_control=False)





                for repeat in range(repeats):
                    if self.sign==True:
                        self.prepare_seq.extend(gate1_pulse)
                        self.prepare_seq.extend(gate2_pulse)
                    else:
                        self.prepare_seq.extend(gate1_pulse)
                    # self.prepare_seq.extend([device.pg.pmulti(device, 10e-9)])

                fast_control = 'quasi-binary'
                phase = phi%(2*np.pi)
                # if qubit_id == '1':
                #     s_pulse = [(c, device.pg.virtual_z, int(phase / (2*np.pi) * (2 ** 8)), fast_control) for c, p in
                #                self.pre_pulse1.channel_amplitudes.metadata.items()]
                # else:
                #     s_pulse = [(c, device.pg.virtual_z, int(phase / (2 * np.pi) * (2 ** 8)), fast_control) for c, p in
                #                self.pre_pulse2.channel_amplitudes.metadata.items()]

                s_pulse = [(c, device.pg.virtual_z, int(phase / (2 * np.pi) * (2 ** 8)), fast_control) for c, p in
                           self.pre_pulse1.channel_amplitudes.metadata.items()]

                sequence_z = [device.pg.pmulti(device, 0, *tuple(s_pulse))]
                self.prepare_seq.extend(sequence_z)

                # self.control_qubit_sequence.set_phase(int(phase / (2 * np.pi) * (2 ** 8)))
                #
                # if qubit_id == '1':
                #     self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))
                # else:
                #     self.prepare_seq.extend(self.pre_pulse2.get_pulse_sequence(0))

                self.prepare_seq.extend(self.pre_pulse1.get_pulse_sequence(0))

                sequence_control.set_preparation_sequence(device, self.ex_sequencers, self.prepare_seq)
                self.readout_sequencer.awg.start_seq(self.readout_sequencer.params['sequencer_id'])
                #for ex_seq in self.ex_sequencers:
                    #ex_seq.set_length(self.length)
            else:
                phase = phi % (2 * np.pi)
                self.control_qubit_sequence.set_phase(int(phase / (2 * np.pi) * (2 ** 8)))
                #for ex_seq in self.ex_sequencers:
                    #ex_seq.set_length(self.length)

    references = {('frequency_controls', qubit_id_): device.get_frequency_control_measurement_id(qubit_id=qubit_id_) for qubit_id_ in qubit_id}
    references['channel_amplitudes'] = channel_amplitudes1_.id
    references['readout_pulse'] = readout_pulse.id
    measurement_type = 'CPhase_calibration'
    exp_sin_fitter_mode = 'unsync'
    measurement_name = 'iq' + qubit_id

    if hasattr(measurer, 'references'):
        references.update(measurer.references)

    if len(qubit_id)>1:
        arg_id = -2 # the last parameter is resultnumbers, so the time-like argument is -2
    else:
        arg_id = -1
    fitter_arguments = (measurement_name, SinglePeriodSinFitter(), arg_id, np.arange(len(extra_sweep_args)))


    metadata = {'qubit_id': ','.join(qubit_id),
                'extra_sweep_args': str(len(extra_sweep_args)),
                'tail_length': str(tail_length),
                'readout_delay': str(readout_delay),
                'repeats': str(repeats),
                'transition':transition}
    metadata.update(additional_metadata)

    setter = ParameterSetter()

    references['long_process'] =  gate.id
    references['readout_pulse'] = readout_pulse.id

    measurement = device.sweeper.sweep_fit_dataset_1d_onfly(measurer,
                                                            *extra_sweep_args,
                                                            (phis, setter.phi_setter, 'Z phase', 'rad'),
                                                            fitter_arguments=fitter_arguments,
                                                            measurement_type=measurement_type,
                                                            metadata=metadata,
                                                            references=references,
                                                            on_update_divider=5)

    for ex_seq in ex_sequencers:
        ex_seq.stop()
    readout_sequencer.stop()
    return measurement