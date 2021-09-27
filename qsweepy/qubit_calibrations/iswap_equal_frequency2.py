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






def iswap_rabi(device, qubit_id,  gate, frequencies, amplitudes, lengths,
               gate2=None, pre_pulse = None, filter_parameters = None, gate_nums = 1):
    from .calibrated_readout import get_calibrated_measurer
    pre_pause = float(gate.metadata['pre_pause'])
    post_pause = float(gate.metadata['post_pause'])

    readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)

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
            self.prepare_seq = []
            self.readout_sequencer = readout_sequencer
            self.prepare_seq.extend(pre_pulse.get_pulse_sequence(0))

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


        def frequency_setter(self, frequency):
            self.frequency = frequency
            #self.filler_func()

        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            self.filler_func()

        def length_setter(self, length):
            self.length = length
            #self.filler_func()


        def filler_func(self):

            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate.metadata['carrier_name']: self.amplitude})
            #print(channel_amplitudes1_)
            #vf_pulse = [device.pg.pmulti(0, (gate.metadata['carrier_name'], pulses.vf, self.frequency))]

            pre_sequence = [] #vf_pulse
            if pre_pulse is not None:
                pre_sequence = pre_sequence + pre_pulse.get_pulse_sequence(0)
            sequence=[]
            pulse_seq1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=float(gate.metadata['tail_length']),
                                                                      length=self.length,
                                                                      phase=0.0)

            if filter_parameters is not None:
                pass

            if gate2 is not None:
                carrier2 = gate2.metadata['carrier_name']
                amplitude2 = float(gate2.metadata['amplitude'])
                frequency2 = float(gate2.metadata['frequency'])
                #frequency2 = self.frequency
                channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})
                if gate2.metadata['pulse_type'] == 'parametric':
                    pulse_seq2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes2_,
                                                                              tail_length=float(0),
                                                                              length=self.length,
                                                                              phase=0.0)
                else:
                    #amplitude2 = -0.410 - self.amplitude
                    #channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in
                    #                   channel_amplitudes2_.metadata.items()]
                    #pulse_seq2 = [device.pg.pmulti((float(gate.metadata['tail_length'])),*tuple(channel_pulses2_1))]
                    channel_pulses2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in
                                       channel_amplitudes2_.metadata.items()]
                    pulse_seq2 = [device.pg.pmulti((self.length+2*float(gate.metadata['tail_length'])), *tuple(channel_pulses2))]
                    #channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in
                    #                     channel_amplitudes2_.metadata.items()]
                    #pulse_seq2 += [device.pg.pmulti((float(gate.metadata['tail_length'])), *tuple(channel_pulses2_1))]

                sequence = sequence + [device.pg.pmulti(pre_pause)] + device.pg.parallel(pulse_seq1, pulse_seq2) + [
                    device.pg.pmulti(post_pause)]
            else:
                sequence = sequence + [device.pg.pmulti(pre_pause)] + pulse_seq1 + [device.pg.pmulti(post_pause)]


            #raise ValueError('fallos')
            #device.pg.set_seq(device.pre_pulses + pre_sequence + sequence*gate_nums + readout_trigger_seq + readout_pulse_seq)

    setter = ParameterSetter()



    references = {'long_process': gate.id,
                  'readout_pulse': readout_pulse.id}
    if gate2 is not None:
        references['gate2'] = gate2.id
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
    measurement_type = 'iswap_rabi'
    measurement = device.sweeper.sweep(measurer,
                                       #*extra_sweep_args,
                                       (lengths, setter.length_setter, 'Delay', 's'),
                                       (frequencies, setter.frequency_setter, 'Frequency', 'Hz'),
                                       (amplitudes, setter.amplitude_setter, 'Amplitude', ''),
                                       measurement_type = measurement_type,
                                       references=references)


def get_iswap_simple_sequence(device,coupler_gate, frequency_delta_gate):
    amplitude_coupler = float(coupler_gate.metadata['amplitude'])
    length_coupler = float(coupler_gate.metadata['length'])
    tail_length_coupler = float(coupler_gate.metadata['tail_length'])
    pre_pause_coupler = float(coupler_gate.metadata['pre_pause'])
    post_pause_coupler = float(coupler_gate.metadata['post_pause'])
    carrier_coupler = coupler_gate.metadata['carrier_name']
    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{carrier_coupler: amplitude_coupler})

    pulse_seq_coupler = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                              channel_amplitudes=channel_amplitudes1_,
                                                              tail_length=tail_length_coupler,
                                                              length=length_coupler,
                                                              phase=0.0)


    carrier_delta = frequency_delta_gate.metadata['carrier_name']
    amplitude_delta = float(frequency_delta_gate.metadata['amplitude'])
    frequency_delta = float(frequency_delta_gate.metadata['frequency'])
    channel_amplitudes_delta = channel_amplitudes.channel_amplitudes(device, **{carrier_delta: amplitude_delta})
    pulse_seq_delta = []
    if frequency_delta_gate.metadata['pulse_type'] == 'parametric':
        pulse_seq_delta = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes_delta,
                                                                          tail_length=0,
                                                                          length=length_coupler,
                                                                          phase=0.0)
    else:
        channel_pulses_delta = [(c, device.pg.sin, amplitude_delta, frequency_delta) for c, a in
                                channel_amplitudes_delta.metadata.items()]
        pulse_seq_delta = [device.pg.pmulti(length_coupler+2*tail_length_coupler, *tuple(channel_pulses_delta))]
    pulse_seq = [device.pg.pmulti(pre_pause_coupler)] + device.pg.parallel(pulse_seq_coupler, pulse_seq_delta) + \
                [device.pg.pmulti(post_pause_coupler)]
    return pulse_seq


def get_zz_from_fsim_pi4_phi_sequence(device, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                      x_phase = 0.0, virt_phase_z1=0.0, virt_phase_z2=0,  num_pulses=1):

    qubit_id1 = x_gate.metadata['qubit_id']
    #readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id1)
    qubit_id2 = z_gate.metadata['target_qubit_id']

    references = {'x_gate': x_gate.id}
    references['z_gate'] = z_gate.id

    HZ1 = excitation_pulse.get_hadamard(device=device, qubit_id='1')
    HZ2 = excitation_pulse.get_hadamard(device=device, qubit_id='2')

    # Coupler gate
    references['coupler_gate']: coupler_gate.id
    amplitude_coupler = float(coupler_gate.metadata['amplitude'])
    length_coupler = float(coupler_gate.metadata['length'])
    tail_length_coupler = float(coupler_gate.metadata['tail_length'])
    pre_pause_coupler = float(coupler_gate.metadata['pre_pause'])
    post_pause_coupler = float(coupler_gate.metadata['post_pause'])
    carrier_coupler = coupler_gate.metadata['carrier_name']

    channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                 **{carrier_coupler: amplitude_coupler})

    pulse_seq_coupler = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                              channel_amplitudes=channel_amplitudes1_,
                                                              tail_length=tail_length_coupler,
                                                              length=length_coupler,
                                                              phase=0.0)

    # Frequency_delta_gate for frequency difference compensation
    references['frequency_delta_gate']: frequency_delta_gate.id
    carrier_delta = frequency_delta_gate.metadata['carrier_name']
    amplitude_delta = float(frequency_delta_gate.metadata['amplitude'])
    frequency_delta = float(frequency_delta_gate.metadata['frequency'])
    channel_amplitudes_delta = channel_amplitudes.channel_amplitudes(device, **{carrier_delta: amplitude_delta})

    if frequency_delta_gate.metadata['pulse_type'] == 'parametric':
        pulse_seq_delta = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes_delta,
                                                                          tail_length=0,
                                                                          length=length_coupler,
                                                                          phase=0.0)
    else:
        channel_pulses_delta = [(c, device.pg.sin, amplitude_delta, frequency_delta) for c, a in
                                channel_amplitudes_delta.metadata.items()]
        pulse_seq_delta = [device.pg.pmulti(length_coupler+2*tail_length_coupler, *tuple(channel_pulses_delta))]

    virt_phsae_q1 = -2 * np.pi * (length_coupler + 2 * tail_length_coupler + post_pause_coupler + pre_pause_coupler)*device.get_qubit_fq(qubit_id1)
    virtual_Q1 = excitation_pulse.get_s(device=device, qubit_id=qubit_id1, phase=virt_phsae_q1)

    virt_phsae_q2 = -2 * np.pi * (length_coupler + 2 * tail_length_coupler) * device.get_qubit_fq(qubit_id1) - \
                    2 * np.pi * (post_pause_coupler+ pre_pause_coupler) * device.get_qubit_fq(qubit_id2)
    virtual_Q2 = excitation_pulse.get_s(device=device, qubit_id=qubit_id2, phase=virt_phsae_q2)

    virtual_Q12 = device.pg.parallel(virtual_Q1, virtual_Q2)

    pulse_seq = [device.pg.pmulti(pre_pause_coupler)] + device.pg.parallel(pulse_seq_coupler, pulse_seq_delta) + \
                [device.pg.pmulti(post_pause_coupler)] + virtual_Q12

    # Gate sequence between two sqrt(fSIM)
    # x_gate
    x_gate_seq = x_gate.get_pulse_sequence(x_phase)

    z_carrier = z_gate.metadata['carrier_name']
    z_amplitude = float(z_gate.metadata['amplitude'])
    z_frequency = float(z_gate.metadata['frequency'])
    z_length = float(z_gate.metadata['length'])
    z_channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{z_carrier: z_amplitude})

    z_channel_pulses = [(c, device.pg.sin, z_amplitude, z_frequency) for c, a in
                                   z_channel_amplitudes_.metadata.items()]
    pulse_seq_z = [device.pg.pmulti(z_length, *tuple(z_channel_pulses))]

    virt2_phsae_q1 = -2 * np.pi * (z_length) * device.get_qubit_fq(qubit_id1) -x_phase
    virtual2_Q1 = excitation_pulse.get_s(device=device, qubit_id=qubit_id1, phase=virt2_phsae_q1)
    virt2_phsae_q2 = -2 * np.pi * (z_length)  * device.get_qubit_fq(qubit_id2)
    virtual2_Q2 = excitation_pulse.get_s(device=device, qubit_id=qubit_id2, phase=virt2_phsae_q2)
    virtual2_Q12 = device.pg.parallel(virtual2_Q1, virtual2_Q2)

    #alt_gate = x_gate + pulse_seq_z
    #alt_gate = device.pg.parallel(x_gate_seq, pulse_seq_z)
    alt_gate = x_gate_seq + virtual2_Q12

    #full_length = (float(x_gate.metadata['length']) + length_coupler)*2
    full_length = float(x_gate.metadata['length']) + length_coupler*2+4*tail_length_coupler
    phase_shift_q2 = device.get_qubit_fq(qubit_id1) - device.get_qubit_fq(qubit_id2)

    #virtual_gate = excitation_pulse.get_s(device=device, qubit_id='2', phase = 2*np.pi*phase_shift_q2*full_length%2*np.pi*)
    virtual_gate1 = excitation_pulse.get_s(device=device, qubit_id='1', phase= virt_phase_z1)
    virtual_gate2 = excitation_pulse.get_s(device=device, qubit_id='2', phase= virt_phase_z2)
    virtual_gate = device.pg.parallel(virtual_gate1, virtual_gate2)
    pre_h_pulse = device.pg.parallel(HZ1, HZ2)
    post_h_pulse = device.pg.parallel(HZ1, HZ2)
    #work_sequence = (pulse_seq+alt_gate)*num_pulses
    #work_sequence = (pre_h_pulse + pulse_seq + alt_gate + pulse_seq + alt_gate + virtual_gate + post_h_pulse) * num_pulses
    #work_sequence = (pre_h_pulse + pulse_seq + alt_gate + pulse_seq + virtual_gate + post_h_pulse) * num_pulses
    work_sequence = (pulse_seq + alt_gate + pulse_seq + virtual_gate) * num_pulses

    return work_sequence

def h_fSIM_h_phase_scan(device, qubit_id, coupler_gate, frequency_delta_gate, x_gate, z_gate, phase_values,
                        virt_phase_values = np.asarray(0.0), num_pulses = 1, pre_pulse = None, calibrated = False):

    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations import calibrated_readout
    # readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    if not calibrated:
        readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)
    else:
        qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, qubit_id)
    metadata = {'qubit_id': qubit_id,
                'num_pulses': num_pulses,}
    references = {'x_gate': x_gate.id}
    references['z_gate'] = z_gate.id
    references['coupler_gate'] = z_gate.id
    references['frequency_delta_gate'] = z_gate.id
    if pre_pulse is not None:
        metadata = {'qubit_id': qubit_id,
                    'num_pulses': num_pulses,
                    'pre_pulse_qubit': pre_pulse.metadata['qubit_id']}
        references['pre_pulse'] = pre_pulse.id
        pre_pulse = pre_pulse.get_pulse_sequence(0)
    else:
        pre_pulse = []

    class ParameterSetter:
        def __init__(self):
            self.x_phase = 0
            self.virt_phase_z1 = 0
            self.virt_phase_z2 = 0
        def set_virt_phase_z1(self, virt_phase):
            self.virt_phase_z1 = virt_phase
        def set_virt_phase_z2(self, virt_phase):
            self.virt_phase_z2 = virt_phase
        def set_x_phase(self, x_phase):
            self.x_phase = x_phase
            self.set_seq()
        def set_seq(self):

            work_sequence = get_zz_from_fsim_pi4_phi_sequence(device, coupler_gate,
                                        frequency_delta_gate, x_gate, z_gate, x_phase=self.x_phase,
                                                          virt_phase_z1 =self.virt_phase_z1, virt_phase_z2 =self.virt_phase_z2,
                                                          num_pulses=num_pulses)

            device.pg.set_seq(device.pre_pulses + pre_pulse + work_sequence +
                          device.trigger_readout_seq + readout_pulse.get_pulse_sequence())

    setter = ParameterSetter ()
    measurement = device.sweeper.sweep(measurer,
                                       (virt_phase_values, setter.set_virt_phase_z1, 'virt_phase_z1'),
                                       (phase_values, setter.set_x_phase, 'x_phase'),
                                       measurement_type='fSIM_phase_scan',
                                       metadata=metadata,
                                       references=references)
    return measurement

def parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                  parameter_name, parameter_values,
                                  phase_value = 0, virt_phase_z1_value = 0, virt_phase_z2_value = 0,
                                  num_pulses = 1, pre_pulse = None):
    #from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations import calibrated_readout
    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, qubit_ids)



    qubit_ids = qubit_readout_pulse.metadata['qubit_ids'].split(',')
    target_qubit_states = [0] * len(qubit_ids)
    excitation_pulses = {qubit_id: excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi) for
                         qubit_id in qubit_ids}
    references = {('excitation_pulse', qubit_id): pulse.id for qubit_id, pulse in excitation_pulses.items()}
    references['readout_pulse'] = qubit_readout_pulse.id

    metadata = {'qubit_ids': qubit_ids,
                'num_pulses': num_pulses,
                'initial_phase_value': phase_value,
                'initial_phase_z1_value': virt_phase_z1_value,
                'initial_phase_z2_value': virt_phase_z2_value,}
    references = {'x_gate': x_gate.id}
    references['z_gate'] = z_gate.id
    references['coupler_gate'] = z_gate.id
    references['frequency_delta_gate'] = z_gate.id

    class ParameterSetter:
        def __init__(self):
            self.x_phase = phase_value
            self.virt_phase_z1 = virt_phase_z1_value
            self.virt_phase_z2 = virt_phase_z2_value
            self.parameter_name = parameter_name

        def filler_funct(self):
            work_sequence = get_zz_from_fsim_pi4_phi_sequence(device, coupler_gate,
                                                                 frequency_delta_gate, x_gate,
                                                                 z_gate, x_phase=self.x_phase,
                                                                 virt_phase_z1=self.virt_phase_z1,
                                                                 virt_phase_z2=self.virt_phase_z2,
                                                                 num_pulses=num_pulses)
            return work_sequence

        def set_parameter(self, parameter_value):
            if self.parameter_name == 'x_phase':
                self.x_phase = parameter_value
            elif self.parameter_name == 'virt_z1_phase':
                self.virt_phase_z1 = parameter_value
            elif self.parameter_name == 'virt_z2_phase':
                self.virt_phase_z2 = parameter_value

    setter = ParameterSetter()
    measurement = calibrated_readout.calibrate_preparation_and_readout_confusion(device, qubit_readout_pulse,
                                                            readout_device,
                                                            (parameter_values, setter.set_parameter, parameter_name),
                                                            pause_length=0, middle_seq_generator=setter.filler_funct,
                                                            additional_metadata=metadata,
                                                            additional_references=references)
    return measurement

def parametric_confuse_calibration(device, qubit_ids, coupler_gate, frequency_delta_gate,
                                   x_gate, z_gate, num_pulses = 1, scan_points=41, max_num_pulses = 64):

    adaptive_measurements_x = []
    adaptive_measurements_z1 = []
    adaptive_measurements_z2 = []
    def data_reducer(matrix_array):
        reduce_data = np.zeros(len(matrix_array))
        for i in range(len(matrix_array)):
            #reduce_data[i] = np.trace(np.fliplr(matrix_array[i]))
            reduce_data[i] = np.trace(matrix_array[i])
        return reduce_data

    def infer_params_from_measurement(measurement):
        parametric_scan = measurement
        matrix_array = np.asarray(parametric_scan.datasets['resultnumbers'].data)
        params = np.asarray(parametric_scan.datasets['resultnumbers'].parameters[0].values)

        reduce_data = data_reducer(matrix_array)
        return params[np.argmax(reduce_data)]

    def infer_params_from_measurements(adaptive_measurements):
        parametric_scan = adaptive_measurements[-1]
        matrix_array = np.asarray(parametric_scan.datasets['resultnumbers'].data)
        params = np.asarray(parametric_scan.datasets['resultnumbers'].parameters[0].values)

        reduce_data = data_reducer(matrix_array)
        measurement_interpolated_combined = np.zeros(params.shape)
        measurement_projector = np.conj(reduce_data)
        for measurement in adaptive_measurements:
            measurement_interpolated_combined += np.interp(params,
                      np.asarray(measurement.datasets['resultnumbers'].parameters[0].values),
                      np.real(data_reducer(measurement.datasets['resultnumbers'].data)*measurement_projector),)
        return params[np.argmax(measurement_interpolated_combined)]
        #return params[np.argmax(reduce_data)]

    x_phase_guess = 0*np.pi
    z1_phase_guess = 0*np.pi
    z2_phase_guess = 0*np.pi
    phase_range = 2*np.pi
    _range = 2
    num_pulses_init = num_pulses
    scan_points_first = 41
    #first cycle
    x_phase_values = np.linspace(x_phase_guess - 0.5 * phase_range, x_phase_guess + 0.5 * phase_range, scan_points_first)
    measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                                parameter_name='x_phase', parameter_values=x_phase_values,
                                                phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
                                                virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    x_phase_guess = infer_params_from_measurement(measurement)

    z1_phase_values = np.linspace(z1_phase_guess - 0.5*phase_range, z1_phase_guess + 0.5*phase_range, scan_points_first)
    measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                                parameter_name='virt_z1_phase', parameter_values=z1_phase_values,
                                                phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
                                                virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    z1_phase_guess = infer_params_from_measurement(measurement)

    z2_phase_values = np.linspace(z2_phase_guess - 0.5*phase_range, z2_phase_guess + 0.5*phase_range, scan_points_first)

    measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                                parameter_name='virt_z2_phase', parameter_values=z2_phase_values,
                                                phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
                                                virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    z2_phase_guess = infer_params_from_measurement(measurement)
    phase_range = 2*np.pi
    #second cycle
    #x_phase_values = np.linspace(x_phase_guess - 0.5 * phase_range, x_phase_guess + 0.5 * phase_range, scan_points)
    #measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
    #                                            parameter_name='x_phase', parameter_values=x_phase_values,
    #                                            phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
    #                                            virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    #x_phase_guess = infer_params_from_measurement(measurement)
    #
    #z1_phase_values = np.linspace(z1_phase_guess - 0.5 * phase_range, z1_phase_guess + 0.5 * phase_range, scan_points)
    #measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
    #                                            parameter_name='virt_z1_phase', parameter_values=z1_phase_values,
    #                                            phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
    #                                            virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    #z1_phase_guess = infer_params_from_measurement(measurement)
    #
    #z2_phase_values = np.linspace(z2_phase_guess - 0.5 * phase_range, z2_phase_guess + 0.5 * phase_range, scan_points)
    #
    #measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
    #                                            parameter_name='virt_z2_phase', parameter_values=z2_phase_values,
    #                                            phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
    #                                            virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
    #z2_phase_guess = infer_params_from_measurement(measurement)

    x_phase_guess_one_pulse = x_phase_guess
    z1_phase_guess_one_pulse = z1_phase_guess
    z2_phase_guess_one_pulse = z2_phase_guess
    #cycle with different number of pulses
    phase_range = 2*np.pi
    while (num_pulses <= max_num_pulses):
        x_phase_values = np.linspace(x_phase_guess-0.5*phase_range, x_phase_guess+0.5*phase_range, scan_points)
        measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                        parameter_name = 'x_phase', parameter_values = x_phase_values,
                                        phase_value = x_phase_guess, virt_phase_z1_value = z1_phase_guess,
                                        virt_phase_z2_value = z2_phase_guess, num_pulses = num_pulses)
        adaptive_measurements_x.append(measurement)
        x_phase_guess = infer_params_from_measurements(adaptive_measurements_x)
        #num_pulses *= int(_range)
        #phase_range /= int(_range)
    #num_pulses = num_pulses_init
    #phase_range = np.pi
    #while (num_pulses <= max_num_pulses):
        z1_phase_values = np.linspace(z1_phase_guess - 0.5*phase_range, z1_phase_guess + 0.5*phase_range, scan_points)
        measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate, z_gate,
                                        parameter_name = 'virt_z1_phase', parameter_values = z1_phase_values,
                                        phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
                                        virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses)
        adaptive_measurements_z1.append(measurement)
        z1_phase_guess = infer_params_from_measurements(adaptive_measurements_z1)
        #num_pulses *= int(_range)
        #phase_range /= int(_range)

    #num_pulses = num_pulses_init
    #phase_range = np.pi
    #while (num_pulses <= max_num_pulses):
        z2_phase_values = np.linspace(z2_phase_guess - 0.5*phase_range, z2_phase_guess + 0.5*phase_range, scan_points)
        measurement = parametric_iswap_confuse_scan(device, qubit_ids, coupler_gate, frequency_delta_gate, x_gate,z_gate,
                                        parameter_name = 'virt_z2_phase', parameter_values = z2_phase_values,
                                        phase_value=x_phase_guess, virt_phase_z1_value=z1_phase_guess,
                                        virt_phase_z2_value=z2_phase_guess, num_pulses=num_pulses, )
        adaptive_measurements_z2.append(measurement)
        z2_phase_guess = infer_params_from_measurements(adaptive_measurements_z2)
        num_pulses *= int(_range)
        phase_range /= int(_range)

    references = {('confuse_x_phase', measurement.metadata['num_pulses']): measurement.id
                  for measurement in adaptive_measurements_x}
    references1 = {('confuse_z2_phase', measurement.metadata['num_pulses']): measurement.id
                  for measurement in adaptive_measurements_z1}
    references2 = {('confuse_z2_phase', measurement.metadata['num_pulses']): measurement.id
                  for measurement in adaptive_measurements_z2}
    references.update(references1)
    references.update(references2)

    references['z_gate'] = z_gate.id
    references['coupler_gate'] = z_gate.id
    references['frequency_delta_gate'] = z_gate.id
    metadata = {'qubit_ids': qubit_ids,
                'max_num_pulses': max_num_pulses,
                'x_phase_guess': x_phase_guess,
                'z1_phase_guess': z1_phase_guess,
                'z2_phase_guess': z2_phase_guess,
                'x_phase_guess_one_pulse': x_phase_guess_one_pulse,
                'z1_phase_guess_one_pulse': z1_phase_guess_one_pulse,
                'z2_phase_guess_one_pulse': z2_phase_guess_one_pulse,
                }



    return device.exdir_db.save(measurement_type='parametrix_cz_confuse_phases_adaptive',
                                references=references,
                                metadata=metadata)



def filter_param_scan(device, qubit_id, gate1, gate2, parameter_name, parameter_values, lengths, amplitudes,
                      pre_pulse = None, init_filter_param = None, gate_nums=1):
    # We whant to define the filter parameters in coupler DC line
    # We search filter curv in the form S21 = exp(-1j*2*np.pi*freq*delay) * exp(-[attenuation+1j*phase_coef]*freq**degree)
    from .calibrated_readout import get_calibrated_measurer
    pre_pause = float(gate1.metadata['pre_pause'])
    post_pause = float(gate1.metadata['post_pause'])

    readout_pulse, measurer = get_calibrated_measurer(device, qubit_id)

    metadata = {'qubit_id': qubit_id,
                'initial_filter_parameters':init_filter_param,}

    references = {'gate1': gate1.id,
                  'gate2': gate2.id,
                  'readout_pulse': readout_pulse.id}
    pre_pulse_seq =[]
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        pre_pulse_seq = pre_pulse.get_pulse_sequence(0)

    class ParameterSetter:
        def __init__(self):
            self.delay = 0*10
            self.degree = 1.0
            self.attenuation = 4.4
            self.phase_coeff = 8.0
            self.freq_cut_off = 0.6
            self.length = float(gate1.metadata['length'])
            self.amplitude = float(gate1.metadata['amplitude'])
            if init_filter_param is not None:
                self.delay = init_filter_param[0]
                self.degree = init_filter_param[1]
                self.attenuation = init_filter_param[2]
                self.phase_coeff = init_filter_param[3]


        def set_parameter(self, parameter_value):

            if parameter_name == 'delay':
                self.delay = parameter_value
            elif parameter_name == 'degree':
                self.degree = parameter_value
            elif parameter_name == 'attenuation':
                self.attenuation = parameter_value
            elif parameter_name == 'phase_coeff':
                self.phase_coeff = parameter_value
                #self.filler_func()
            #self.filler_func()
        #def set_phase_coef(self, coeff):
        #   self.phase_coeff = coeff
        #    self.filler_func()
        def length_setter(self, length):
            self.length = length
            #self.filler_func()
        def amplitude_setter(self, amplitude):
            self.amplitude = amplitude
            self.filler_func()
        def filter_function(self, x):
            sign = np.sign(x)
            x_new = np.abs(x/1e9)
            delay_part = np.exp(-1j*2*np.pi*sign*x_new*self.delay)
            attenuation_part = np.exp(-x_new**(self.degree)*self.attenuation)
            phase_part = np.exp(1j*sign*x_new**(self.degree)*self.phase_coeff)
            S21 = delay_part*attenuation_part*phase_part

            S21_new = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(S21), norm='ortho'))
            #S21_new[:(len(x_new) // 2 - len(x_new) // 4)] = 0
            S21_new = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(S21_new), norm='ortho'))

            return S21_new

        def filter_correction(self, sig):

            pre_sig = np.zeros(len(sig) // 4)
            sig_full_1 = np.append(pre_sig, sig / 2)
            sig_full_1 = np.append(sig_full_1, pre_sig)
            # sig_full_1 = np.append(sig_full_1, [0])
            #
            sig_full_1 = np.append(sig_full_1, pre_sig)
            sig_full_1 = np.append(sig_full_1, sig / 2)
            sig_full_1 = np.append(sig_full_1, pre_sig)

            sig_full_2 = np.append(pre_sig, -sig / 2)
            sig_full_2 = np.append(sig_full_2, pre_sig)
            # sig_full_2 = np.append(sig_full_2, [0])
            #
            sig_full_2 = np.append(sig_full_2, pre_sig)
            sig_full_2 = np.append(sig_full_2, sig / 2)
            sig_full_2 = np.append(sig_full_2, pre_sig)
            x = np.linspace(-1 / 2, 1 / 2, len(sig_full_1)) * 2.4e9

            function = self.filter_function(x)

            function_zero = np.zeros((len(sig_full_2) - int(self.freq_cut_off*len(sig_full_2))) // 2, dtype='complex')
            function_zero = np.append(function_zero, np.ones(int(self.freq_cut_off*len(sig_full_2))))
            function_zero = np.append(function_zero, np.zeros((len(sig_full_2) - len(function_zero)), dtype='complex'))

            fft_sig_filt_1 = (np.fft.fftshift(np.fft.fft(sig_full_1))) / (function) * function_zero
            fft_sig_filt_2 = (np.fft.fftshift(np.fft.fft(sig_full_2))) / (function) * function_zero
            #fft_sig_filt = fft_sig_filt_1 + fft_sig_filt_2
            # sig_filt = np.fft.ifft(np.fft.fftshift(fft_sig_filt))
            sig_filt1 = np.fft.ifft(np.fft.fftshift((fft_sig_filt_1)))
            sig_filt2 = np.fft.ifft(np.fft.fftshift((fft_sig_filt_2)))
            sig_filt = (sig_filt1) + (sig_filt2)

            return sig_filt[len(sig)*7//4:len(sig)*11//4]


        def filler_func(self):

            #amplitude1 = float(gate1.metadata['amplitude'])
            amplitude1 = self.amplitude
            #frequency1 = float(gate1.metadata['frequency'])

            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{gate1.metadata['carrier_name']: amplitude1})
            #print (channel_amplitudes1_)
            #vf_pulse = [device.pg.pmulti(0, (gate1.metadata['carrier_name'], pulses.vf, frequency1))]

            #sequence = vf_pulse
            sequence = []
            pulse_seq1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=float(gate1.metadata['tail_length']),
                                                                      length=self.length,
                                                                      phase=0.0)
            #channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{carrier: 1.0})
            #channel_pulses = [(c, device.pg.rect_cos, amplitude, tail_length) for c, a in
                              #channel_amplitudes_.metadata.items()]
            #pulse_seq1 = [device.pg.pmulti(length, *tuple(channel_pulses))]

            sig_filt = self.filter_correction(pulse_seq1[0][gate1.metadata['carrier_name']])

            pulse_seq1[0][gate1.metadata['carrier_name']] = sig_filt

            carrier2 = gate2.metadata['carrier_name']
            amplitude2 = float(gate2.metadata['amplitude'])
            frequency2 = float(gate2.metadata['frequency'])
            channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})
            if gate2.metadata['pulse_type'] == 'parametric':
                pulse_seq2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes2_,
                                                                              tail_length=float(0),
                                                                              length=self.length,
                                                                              phase=0.0)
            else:
                channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in
                                       channel_amplitudes2_.metadata.items()]

                pulse_seq2 = [device.pg.pmulti((float(gate1.metadata['tail_length'])), *tuple(channel_pulses2_1))]
                channel_pulses2_2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in
                                   channel_amplitudes2_.metadata.items()]
                pulse_seq2 += [device.pg.pmulti((self.length), *tuple(channel_pulses2_2))]

                channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in
                                       channel_amplitudes2_.metadata.items()]
                pulse_seq2 += [device.pg.pmulti((float(gate1.metadata['tail_length'])), *tuple(channel_pulses2_1))]

            sequence = sequence + [device.pg.pmulti(pre_pause)] + device.pg.parallel(pulse_seq1, pulse_seq2) + [
                    device.pg.pmulti(post_pause)]


            readout_trigger_seq = device.trigger_readout_seq
            readout_pulse_seq = readout_pulse.pulse_sequence
            #raise ValueError('fallos')
            device.pg.set_seq(device.pre_pulses + pre_pulse_seq + sequence*gate_nums + readout_trigger_seq + readout_pulse_seq)
    setter = ParameterSetter()


    measurement_type = 'filter_reconstruction_rabi'
    measurement = device.sweeper.sweep(measurer,
                                       #*extra_sweep_args,
                                       (parameter_values, setter.set_parameter, parameter_name),
                                       #(np.linspace(0,10,11), setter.set_phase_coef, 'phase_coeff'),
                                       #(lengths, setter.length_setter, 'Delay', 's'),#
                                       (amplitudes, setter.amplitude_setter, 'amplitude', ''),
                                       measurement_type = measurement_type,
                                       metadata = metadata,
                                       references=references)
    return measurement

def alt_gate_calibration(device, alt_gate, pre_pulse, phase_values, z_gate1=None, alt_length=None, num_pulses = 2):
    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    qubit_id = pre_pulse.metadata['qubit_id']
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)
    metadata = {'qubit_id': qubit_id,
                'num_pulses': num_pulses}
    references = {'alt_gate': alt_gate.id,}

    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        references['post_pulse'] = pre_pulse.id
        pre_pulse = pre_pulse.get_pulse_sequence(0)

    else:
        pre_pulse = []

    def set_virt_phase(phase_value):

        carrier2 = alt_gate.metadata['carrier_name']
        amplitude2 = float(alt_gate.metadata['amplitude'])
        frequency2 = float(alt_gate.metadata['frequency'])
        length2 = float(alt_gate.metadata['length'])
        if alt_length is not None:
            length2 = alt_length
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})

        pulse=[]
        pulse_seq_alt=[]
        pulse_seq1 = []
        pulse_seq2 = []

        virtual_gate = excitation_pulse.get_s(device=device, qubit_id=alt_gate.metadata['target_qubit_id'],
                                              phase=phase_value)

        if z_gate1 is not None:
            pre_pause = float(z_gate1.metadata['pre_pause'])
            post_pause = 0*float(z_gate1.metadata['post_pause'])
            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                                        **{z_gate1.metadata[
                                                                               'carrier_name']: 0.8})
            pulse_seq1 = [device.pg.pmulti(pre_pause)]

            pulse_seq1 += excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                    channel_amplitudes=channel_amplitudes_,
                                                                    tail_length=float(z_gate1.metadata['tail_length']),
                                                                    length=alt_length*num_pulses,
                                                                    phase=0.0)
            pulse_seq1 = [device.pg.pmulti(post_pause)]


        if alt_gate.metadata['pulse_type'] == 'parametric':
                pulse_seq_alt = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes2_,
                                                                          tail_length=float(0),
                                                                          length=alt_length,
                                                                          phase=0.0)
        else:
                channel_pulses2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in
                                   channel_amplitudes2_.metadata.items()]
                pulse_seq_alt = [device.pg.pmulti(alt_length, *tuple(channel_pulses2))]
        pulse_seq2 = (pulse_seq_alt + virtual_gate)*num_pulses

        pulse = device.pg.parallel(pulse_seq1, pulse_seq2)

        device.pg.set_seq(device.pre_pulses + pre_pulse + pulse + pre_pulse + device.trigger_readout_seq + readout_pulse.get_pulse_sequence())

    measurement = device.sweeper.sweep(measurer,
                                       (phase_values, set_virt_phase, 'virt_phase'),
                                       measurement_type='alt_gate_scan',
                                       metadata=metadata,
                                       references=references)
    return measurement

def alt_gate_scan(device, alt_gate, pre_pulse, scan_points, init_phase = np.pi, z_gate1=None, alt_length=None, max_num_pulses = 64):

    qubit_id = pre_pulse.metadata['qubit_id']
    adaptive_measurements = []


    def infer_amplitude_from_measurements():
        amplitudes = adaptive_measurements[-1].datasets['iq'+qubit_id].parameters[0].values
        measurement_interpolated_combined = np.zeros(amplitudes.shape)
        measurement_projector = np.conj(np.mean(adaptive_measurements[0].datasets['iq'+qubit_id].data))
        for measurement in adaptive_measurements:
            measurement_interpolated_combined += np.interp(amplitudes,
                      measurement.datasets['iq' + qubit_id].parameters[0].values,
                      np.real(measurement.datasets['iq' + qubit_id].data*measurement_projector),)
        return amplitudes[np.argmax(measurement_interpolated_combined)]

    _range = 2
    phase_range = 2*np.pi
    num_pulses = 2
    phase_guess = init_phase
    while (num_pulses <= max_num_pulses):
        phase_values = np.linspace(phase_guess-0.5*phase_range, phase_guess+0.5*phase_range, scan_points)
        measurement = alt_gate_calibration(device, alt_gate, pre_pulse, phase_values, z_gate1, alt_length, num_pulses)
        adaptive_measurements.append(measurement)
        phase_guess = infer_amplitude_from_measurements()
        num_pulses *= int(_range)
        phase_range /= int(_range)

    references = {('alt_gate_scan', measurement.metadata['num_pulses']): measurement.id
                  for measurement in adaptive_measurements}
    references['pre_pulse'] = pre_pulse.id
    references['alt_gate'] = alt_gate.id

    metadata = {'phase_guess': phase_guess,
                'qubit_id': qubit_id,
                'max_num_pulses': max_num_pulses,
                }

    return device.exdir_db.save(measurement_type='alt_gate_scan_phase_adaptive',
                                references=references,
                                metadata=metadata)


def parametric_gate_scan(device, gate1, num_pulses, parameter_name, parameter_values,
                         pre_pulse = None, gate2=None, filter_parameters = None, delay=False):
    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    #readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    qubit_id = gate1.metadata['q2']
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    metadata = {'qubit_id':qubit_id,
                'num_pulses':num_pulses}

    references = {'gate': gate1.id}
    if gate2 is not None:
        references['gate2']: gate2.id
        carrier2 = gate2.metadata['carrier_name']
        amplitude2 = float(gate2.metadata['amplitude'])
        frequency2 = float(gate2.metadata['frequency'])
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})
    #time_length0 = 0
    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        #time_length0 += float(pre_pulse.metadata['length'])
        pre_pulse_seq = pre_pulse.get_pulse_sequence(0)

    else:
        pre_pulse_seq = []

    def filter_function(x, param):
        sign = np.sign(x)
        x_new = np.abs(x / 1e9)
        delay_part = np.exp(-1j * 2 * np.pi * sign * x_new * param[0])
        attenuation_part = np.exp(-x_new ** (param[1]) * param[2])
        phase_part = np.exp(1j * sign * x_new ** (param[1]) * param[3])
        S21 = delay_part * attenuation_part * phase_part
        S21_new = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(S21), norm='ortho'))
        # S21_new[:(len(x_new) // 2 - len(x_new) // 4)] = 0
        S21_new = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(S21_new), norm='ortho'))
        return S21_new

    def filter_correction(sig, param, freq_cut_off=0.6):

        pre_sig = np.zeros(len(sig) // 4)
        sig_full_1 = np.append(pre_sig, sig / 2)
        sig_full_1 = np.append(sig_full_1, pre_sig)
        sig_full_1 = np.append(sig_full_1, pre_sig)
        sig_full_1 = np.append(sig_full_1, sig / 2)
        sig_full_1 = np.append(sig_full_1, pre_sig)

        sig_full_2 = np.append(pre_sig, -sig / 2)
        sig_full_2 = np.append(sig_full_2, pre_sig)
        sig_full_2 = np.append(sig_full_2, pre_sig)
        sig_full_2 = np.append(sig_full_2, sig / 2)
        sig_full_2 = np.append(sig_full_2, pre_sig)
        x = np.linspace(-1 / 2, 1 / 2, len(sig_full_1)) * 2.4e9
        function = filter_function(x, param)
        function_zero = np.zeros((len(sig_full_2) - int(freq_cut_off * len(sig_full_2))) // 2, dtype='complex')
        function_zero = np.append(function_zero, np.ones(int(freq_cut_off * len(sig_full_2))))
        function_zero = np.append(function_zero, np.zeros((len(sig_full_2) - len(function_zero)), dtype='complex'))
        fft_sig_filt_1 = (np.fft.fftshift(np.fft.fft(sig_full_1))) / (function) * function_zero
        fft_sig_filt_2 = (np.fft.fftshift(np.fft.fft(sig_full_2))) / (function) * function_zero
        sig_filt1 = np.fft.ifft(np.fft.fftshift((fft_sig_filt_1)))
        sig_filt2 = np.fft.ifft(np.fft.fftshift((fft_sig_filt_2)))
        sig_filt = (sig_filt1) + (sig_filt2)

        return sig_filt[len(sig) * 7 // 4:len(sig) * 11 // 4]

    def set_parameter(parameter_value):
        amplitude = float(gate1.metadata['amplitude'])
        length = float(gate1.metadata['length'])
        tail_length1 = float(gate1.metadata['tail_length'])
        pre_pause = float(gate1.metadata['pre_pause'])
        post_pause = float(gate1.metadata['post_pause'])
        carrier1 = gate1.metadata['carrier_name']

        if parameter_name == 'amplitude':
            amplitude = parameter_value
        elif parameter_name == 'length':
            length = parameter_value
        elif parameter_name == 'tail_length':
            tail_length1 = parameter_value
        elif parameter_name == 'post_pause':
            post_pause = parameter_value

        time_length0 = 0
        if pre_pulse is not None:
            time_length0 += float(pre_pulse.metadata['length'])

        pulse = []

        for _i in range(num_pulses):
            #channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{carrier: 1.0})
            #channel_pulses = [(c, device.pg.rect_cos, amplitude, tail_length) for c, a in channel_amplitudes_.metadata.items()]
            #pulse_seq1 = [device.pg.pmulti(length, *tuple(channel_pulses))]

            channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                         **{carrier1: amplitude})

            pulse_seq1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes1_,
                                                                      tail_length=tail_length1,
                                                                      length=length,
                                                                      phase=0.0)

            if filter_parameters is not None:
                sig_filt = filter_correction(pulse_seq1[0][gate1.metadata['carrier_name']], filter_parameters)
                pulse_seq1[0][gate1.metadata['carrier_name']] = sig_filt
            if delay:
                phase_shift_q2 = device.get_qubit_fq(gate1.metadata['q1']) - device.get_qubit_fq(gate1.metadata['q2'])

                phase_diff = 2*np.pi*time_length0*phase_shift_q2/(2*np.pi) - int(2*np.pi*time_length0*phase_shift_q2/(2*np.pi))
                pre_pause = 1/phase_shift_q2 - phase_diff/phase_shift_q2#/(2*np.pi)
                post_pause = 1/phase_shift_q2 - pre_pause# phase_diff/phase_shift_q2/(2*np.pi)
                time_length0 +=pre_pause + post_pause# + length + 2*tail_length1
                #print(post_pause, pre_pause)

            #pulse += [device.pg.pmulti(pre_pause)]
            pulse_seq =[]
            if gate2 is not None:
                virt_phsae_q1 = -2 * np.pi * (length + 2 * tail_length1+pre_pause + post_pause) * device.get_qubit_fq(gate1.metadata['q1'])

                virtual_gate1 = excitation_pulse.get_s(device=device, qubit_id=gate2.metadata['target_qubit_id'],
                                                      phase=virt_phsae_q1)
                virt_phsae_q2 = -2*np.pi*(length+2*tail_length1)*device.get_qubit_fq(gate1.metadata['q1']) -\
                                2*np.pi*(pre_pause + post_pause)* device.get_qubit_fq(gate1.metadata['q2'])

                virtual_gate2 = excitation_pulse.get_s(device=device, qubit_id=gate2.metadata['target_qubit_id'],
                                                   phase=virt_phsae_q2)
                virtual_gate = device.pg.parallel(virtual_gate1, virtual_gate2)
                if gate2.metadata['pulse_type'] == 'parametric':
                    pulse_seq2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes2_,
                                                                      tail_length=float(0),
                                                                      length=length,
                                                                      phase=0.0)
                else:
                    #channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in channel_amplitudes2_.metadata.items()]
                    #pulse_seq2 = [device.pg.pmulti(tail_length1, *tuple(channel_pulses2_1))]

                    channel_pulses2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in channel_amplitudes2_.metadata.items()]
                    pulse_seq2 = [device.pg.pmulti((length+2*tail_length1), *tuple(channel_pulses2))]

                    #channel_pulses2_1 = [(c, device.pg.sin, 0, frequency2) for c, a in channel_amplitudes2_.metadata.items()]
                    #pulse_seq2 += [device.pg.pmulti(tail_length1, *tuple(channel_pulses2_1))]
                pulse_seq = [device.pg.pmulti(pre_pause)] + device.pg.parallel(pulse_seq1, pulse_seq2) +\
                            [device.pg.pmulti(post_pause)]# + virtual_gate
            else:
                pulse_seq = [device.pg.pmulti(pre_pause)] + pulse_seq1 + [device.pg.pmulti(post_pause)]

            pulse += pulse_seq #[device.pg.pmulti(pre_pause)]+pulse_seq+[device.pg.pmulti(post_pause)]
            #pulse = ([device.pg.pmulti(pre_pause)] + pulse_seq+[device.pg.pmulti(post_pause)])*num_pulses
        device.pg.set_seq(device.pre_pulses+pre_pulse_seq+pulse+device.trigger_readout_seq+readout_pulse.get_pulse_sequence())

    measurement = device.sweeper.sweep(measurer,
                                        (parameter_values, set_parameter, parameter_name),
                                        measurement_type='parametric_gate_scan',
                                        metadata=metadata,
                                        references=references)
    return measurement

def gate_fSIM_pulse_sequence(device, gate, num_pulses, pause_values, phase_values, gate1=None, gate2=None, pre_pulse=None):
    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    # readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    qubit_id = gate.metadata['qubit_id']
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)
    metadata = {'qubit_id': qubit_id,
                'num_pulses': num_pulses}
    references = {'gate': gate.id}


    if gate1 is not None:
        references['gate1']: gate1.id
        amplitude1 = float(gate1.metadata['amplitude'])
        length1 = float(gate1.metadata['length'])
        tail_length1 = float(gate1.metadata['tail_length'])
        pre_pause1 = float(gate1.metadata['pre_pause'])
        post_pause1 = float(gate1.metadata['post_pause'])
        carrier1 = gate1.metadata['carrier_name']

    if gate2 is not None:
        references['gate2']: gate2.id
        carrier2 = gate2.metadata['carrier_name']
        amplitude2 = float(gate2.metadata['amplitude'])
        frequency2 = float(gate2.metadata['frequency'])
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})

    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        pre_pulse = pre_pulse.get_pulse_sequence(0)
    else:
        pre_pulse = []

    class ParameterSetter:
        def __init__(self):
            self.pause_length = pre_pause1
            self.phase = 0.0

        def phase_setter(self, phase):
            self.phase = phase

        def pause_setter(self, pause):
            self.pause_length = pause
            self.create_sequence()

        def create_sequence(self):

            if gate1 is not None:
                channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{carrier1: amplitude1})
                channel_pulses = [(c, device.pg.rect_cos, amplitude1, tail_length1) for c, a in
                                channel_amplitudes_.metadata.items()]
                pulse_seq1 = [device.pg.pmulti(length1, *tuple(channel_pulses))]
            else:
                pulse_seq1 = []
            if gate2 is not None:
                if gate2.metadata['pulse_type'] == 'parametric':
                    pulse_seq2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                              channel_amplitudes=channel_amplitudes2_,
                                                                              tail_length=float(0),
                                                                              length=length1,
                                                                              phase=0.0)
                else:
                    channel_pulses2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in
                                       channel_amplitudes2_.metadata.items()]
                    pulse_seq2 = [device.pg.pmulti(length1+2*tail_length1, *tuple(channel_pulses2))]
                pulse_seq = device.pg.parallel(pulse_seq1, pulse_seq2)
            else:
                pulse_seq = pulse_seq1

            pulse = ([device.pg.pmulti(self.pause_length)] + pulse_seq + [device.pg.pmulti(post_pause1)]) * num_pulses
            work_sequence = []
            for _i in range(int(num_pulses)):
                x_pre_pulse = gate.get_pulse_sequence(self.phase*_i)
                work_sequence = (work_sequence + x_pre_pulse + pulse)
            device.pg.set_seq(device.pre_pulses + pre_pulse + work_sequence +
                              device.trigger_readout_seq + readout_pulse.get_pulse_sequence())

    setter = ParameterSetter()

    measurement = device.sweeper.sweep(measurer,
                                       (phase_values, setter.phase_setter, 'gate_phase'),
                                       (pause_values, setter.pause_setter, 'pause_length'),
                                       measurement_type='parametric_gate_scan',
                                       metadata=metadata,
                                       references=references)
    return measurement


def gate_fSIM_pulse_sequence2(device, alt_gate1, num_pulses, gate1_amplitude, alt_gate2_amplitude, gate1=None, gate2=None, alt_gate2=None, pre_pulse=None):
    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    # readout_pulse = get_qubit_readout_pulse(device, qubit_id)
    qubit_id = alt_gate1.metadata['qubit_id']
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)
    metadata = {'qubit_id': qubit_id,
                'num_pulses': num_pulses}
    references = {'alt_gate1': alt_gate1.id}

    if alt_gate2 is not None:
        references['alt_gate2'] = alt_gate2.id
        alt_carrier2 = alt_gate2.metadata['carrier_name']
        alt_amplitude2 = float(alt_gate2.metadata['amplitude'])
        alt_frequency2 = float(alt_gate2.metadata['frequency'])
        alt_length2 = float(alt_gate2.metadata['length'])
        alt_channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{alt_carrier2: alt_amplitude2})

    if gate1 is not None:
        references['gate1'] = gate1.id
        amplitude1 = float(gate1.metadata['amplitude'])
        length1 = float(gate1.metadata['length'])
        tail_length1 = float(gate1.metadata['tail_length'])
        pre_pause1 = float(gate1.metadata['pre_pause'])
        post_pause1 = float(gate1.metadata['post_pause'])
        carrier1 = gate1.metadata['carrier_name']

    if gate2 is not None:
        references['gate2'] = gate2.id
        carrier2 = gate2.metadata['carrier_name']
        amplitude2 = float(gate2.metadata['amplitude'])
        frequency2 = float(gate2.metadata['frequency'])
        channel_amplitudes2_ = channel_amplitudes.channel_amplitudes(device, **{carrier2: amplitude2})

    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        pre_pulse = pre_pulse.get_pulse_sequence(0)
    else:
        pre_pulse = []

    class ParameterSetter:
        def __init__(self):
            self.alt_gate2_amplitude = 0.
            self.amplitude1 = amplitude1

        def alt_gate2_amplitude_setter(self, amplitude):
            self.alt_gate2_amplitude = amplitude
            self.create_sequence()

        def amplitude_setter(self, amplitude):
            self.amplitude1 = amplitude
            self.create_sequence()

        def create_sequence(self):

            if gate1 is not None:
                #channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device, **{carrier1: 1.0})
                #channel_pulses = [(c, device.pg.rect_cos, self.amplitude1, tail_length1) for c, a in
                #                channel_amplitudes_.metadata.items()]
                #pulse_seq1 = [device.pg.pmulti(length1, *tuple(channel_pulses))]

                channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                             **{carrier1: self.amplitude1})

                pulse_seq1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                          channel_amplitudes=channel_amplitudes1_,
                                                                          tail_length=tail_length1,
                                                                          length=length1,
                                                                          phase=0.0)
            else:
                pulse_seq1 = []
            if gate2 is not None:
                if gate2.metadata['pulse_type'] == 'parametric':
                    pulse_seq2 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                      channel_amplitudes=channel_amplitudes2_,
                                                                      tail_length=float(0),
                                                                      length=length1,
                                                                      phase=0.0)
                else:
                    channel_pulses2 = [(c, device.pg.sin, amplitude2, frequency2) for c, a in
                                       channel_amplitudes2_.metadata.items()]
                    pulse_seq2 = [device.pg.pmulti((length1+2*tail_length1), *tuple(channel_pulses2))]

                pulse_seq = [device.pg.pmulti(pre_pause1)] + device.pg.parallel(pulse_seq1, pulse_seq2) +\
                            [device.pg.pmulti(post_pause1)]
            else:
                pulse_seq = [device.pg.pmulti(pre_pause1)] + pulse_seq1 + [device.pg.pmulti(post_pause1)]

            alt_gate = alt_gate1.get_pulse_sequence(0.0)
            if alt_gate2 is not None:
                alt_channel_pulses2 = [(c, device.pg.sin, self.alt_gate2_amplitude, alt_frequency2) for c, a in
                                        alt_channel_amplitudes2_.metadata.items()]
                pulse_seq_alt = [device.pg.pmulti(alt_length2, *tuple(alt_channel_pulses2))]
                #alt_gate = alt_gate+pulse_seq2
                alt_gate = device.pg.parallel(alt_gate, pulse_seq_alt)


            work_sequence = (alt_gate + pulse_seq) * num_pulses

            device.pg.set_seq(device.pre_pulses + pre_pulse + work_sequence +
                              device.trigger_readout_seq + readout_pulse.get_pulse_sequence())

    setter = ParameterSetter()

    measurement = device.sweeper.sweep(measurer,
                                       (alt_gate2_amplitude, setter.alt_gate2_amplitude_setter, 'alt_gate2 amplitude'),
                                       (gate1_amplitude, setter.amplitude_setter, 'gate1 amplitude'),
                                       measurement_type='parametric_gate_scan',
                                       metadata=metadata,
                                       references=references)
    return measurement




def parametric_adaptive_scan_2d(device, qubit_id, num_seq, x_phase_range, length_range, scan_points = 21):
    from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations import excitation_pulse

    gate = excitation_pulse.get_excitation_pulse(device=device, qubit_id=qubit_id, rotation_angle=np.pi)
    gate2 = device.get_zgates()['z2p']
    gate1 = device.get_two_qubit_gates()['iSWAP(1,2)']
    readout_pulse, measurer = get_uncalibrated_measurer(device, qubit_id)

    scipy.interpolate.interp2d

    adaptive_measurements = []
    def infer_parameters_from_measurements():
        phases = adaptive_measurements[-1].datasets['iq'+qubit_id].parameters[0].values
        lengths = adaptive_measurements[-1].datasets['iq' + qubit_id].parameters[1].values
        measurement_interpolated_combined = np.zeros(phases.shape, lengths.shape)
        measurement_projector = np.conj(np.mean(adaptive_measurements[0].datasets['iq'+qubit_id].data))
        for measurement in adaptive_measurements:
            measurement_interpolated_combined += np.interp(amplitudes,
                      measurement.datasets['iq' + qubit_id].parameters[0].values,
                      np.real(measurement.datasets['iq' + qubit_id].data*measurement_projector),)
        return phases[np.argmin(measurement_interpolated_combined)[0]], lengths[np.argmin(measurement_interpolated_combined)[1]],

    pause_guess = float(pause_range*0.5)
    pause_range = pause_range
    phase_guess = float(phase_range * 0.5)
    phase_range = phase_range


    while (num_seq <= max_num_pulses):
        phase_values = np.linspace(phase_guess-0.5*phase_range, phase_guess+0.5*phase_range, scan_points)
        pause_values = np.linspace(pause_guess-0.5*pause_range, pause_guess+0.5*pause_range, scan_points)
        measurement = gate_fSIM_pulse_sequence(device, gate, num_pulses, pause_values, phase_values, gate1, gate2)

        adaptive_measurements.append(measurement)
        phase_guess, length_guess = infer_parameters_from_measurements()
        num_pulses *= int(_range)
        phase_range /= int(l_range)




def parametric_adaptive_scan(device, gate1, inverse_rotation_cycles, parameter_name, parameter_values, gate2=None):
    qubit_id = gate1['q1']
    scan_points = int(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_amplitude_scan_points'))
    max_scan_length = float(device.get_qubit_constant(qubit_id=qubit_id, name='adaptive_Rabi_max_scan_length'))

    adaptive_measurements = []

    def infer_amplitude_from_measurements():
        amplitudes = adaptive_measurements[-1].datasets['iq'+qubit_id].parameters[0].values
        measurement_interpolated_combined = np.zeros(amplitudes.shape)
        measurement_projector = np.conj(np.mean(adaptive_measurements[0].datasets['iq'+qubit_id].data))
        for measurement in adaptive_measurements:
            measurement_interpolated_combined += np.interp(amplitudes,
                      measurement.datasets['iq' + qubit_id].parameters[0].values,
                      np.real(measurement.datasets['iq' + qubit_id].data*measurement_projector),)
        return amplitudes[np.argmin(measurement_interpolated_combined)]

    rotation_angle = 2*np.pi/inverse_rotation_cycles
    rect_pulse = get_rect_excitation_pulse(device, qubit_id, rotation_angle, transition=transition)
    channel_amplitudes = device.exdir_db.select_measurement_by_id(rect_pulse.references['channel_amplitudes'])
    if len(channel_amplitudes.metadata)> 2:
        raise ValueError('Default excitation pulse has more than one excitation channel')
    channel = [channel for channel in channel_amplitudes.metadata.keys()][0]
    if preferred_length is None:
        pulse_length = get_preferred_length(device, qubit_id, channel)
    else:
        pulse_length = preferred_length

    num_pulses = int(inverse_rotation_cycles)
    max_num_pulses = max_scan_length / pulse_length

    amplitude_guess = float(rect_pulse.metadata['length'])/per_amplitude_angle_guess(pulse_length, pulse_length/sigmas_in_gauss)
    amplitude_range = amplitude_guess
    #print ('rect_pulse.metadata[length]:', rect_pulse.metadata['length'])
    #print ('rotation_angle: ', rotation_angle)
    #print ('amplitude_guess: ', amplitude_guess)
    sigma = pulse_length/sigmas_in_gauss

    while (num_pulses <= max_num_pulses):
        amplitudes = np.linspace(amplitude_guess-0.5*amplitude_range, amplitude_guess+0.5*amplitude_range, scan_points)
        measurement = gauss_hd_Rabi_amplitude(device, qubit_id, channel_amplitudes, rotation_angle, amplitudes,
                                              pulse_length, sigma, alpha, num_pulses)
        adaptive_measurements.append(measurement)
        amplitude_guess = infer_amplitude_from_measurements()
        num_pulses *= int(_range)
        amplitude_range /= int(_range)

    references = {('gauss_hd_Rabi_amplitude', measurement.metadata['num_pulses']): measurement.id
                    for measurement in adaptive_measurements}
    references['channel_amplitudes'] = channel_amplitudes.id
    references['frequency_controls'] = device.get_frequency_control_measurement_id(qubit_id)
    metadata = {'amplitude_guess': amplitude_guess,
                'qubit_id':qubit_id,
                'alpha':alpha,
                'inverse_rotation_cycles':inverse_rotation_cycles,
                'length': pulse_length,
                'sigma': sigma,
                'transition': transition}

    return device.exdir_db.save(measurement_type='gauss_hd_Rabi_amplitude_adaptive',
                                references=references,
                                metadata=metadata)


def parametric_bswap_confuse_scan(device, qubit_ids, coupler_gate,
                                  parameter_name, parameter_values, lengths, pre_pulse = None):
    #from qsweepy.qubit_calibrations.readout_pulse import get_uncalibrated_measurer
    from qsweepy.qubit_calibrations import calibrated_readout
    qubit_readout_pulse, readout_device = calibrated_readout.get_calibrated_measurer(device, qubit_ids)

    qubit_ids = qubit_readout_pulse.metadata['qubit_ids'].split(',')
    target_qubit_states = [0] * len(qubit_ids)
    excitation_pulses = {qubit_id: excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi) for
                         qubit_id in qubit_ids}
    references = {('excitation_pulse', qubit_id): pulse.id for qubit_id, pulse in excitation_pulses.items()}
    references['readout_pulse'] = qubit_readout_pulse.id

    metadata = {'qubit_ids': qubit_ids}
    references['coupler_gate'] = coupler_gate.id

    if pre_pulse is not None:
        references['pre_pulse'] = pre_pulse.id
        pre_pulse_seq = pre_pulse.get_pulse_sequence(0)
    else:
        pre_pulse_seq = []


    class ParameterSetter:
        def __init__(self):
            self.frequency = float(coupler_gate.metadata['frequency'])
            self.amplitude = float(coupler_gate.metadata['amplitude'])
            self.length = 0
            self.parameter_name = parameter_name

        def filler_funct(self):


            work_sequence = []

            channel_amplitudes_ = channel_amplitudes.channel_amplitudes(device,
                                                  **{coupler_gate.metadata['carrier_name']: self.amplitude})
            frequency_delta = self.frequency
            #vf_pulse = [device.pg.pmulti(0, (coupler_gate.metadata['carrier_name'], pulses.vf, frequency_delta))]
            #pulse_seq = vf_pulse + excitation_pulse.get_rect_cos_pulse_sequence(device=device,
             #                                            channel_amplitudes=channel_amplitudes_,
             #                                            tail_length=0,
             #                                            length=self.length,
            #                                             phase=0.0)

            channel_pulses = [(c, device.pg.sin, self.amplitude, frequency_delta) for c, a in
                               channel_amplitudes_.metadata.items()]
            pulse_seq = [device.pg.pmulti(self.length, *tuple(channel_pulses))]

            work_sequence = pre_pulse_seq + pulse_seq

            return work_sequence

        def set_parameter(self, parameter_value):
            if self.parameter_name == 'amplitude':
                self.amplitude = parameter_value
            elif self.parameter_name == 'frequency':
                self.frequency += parameter_value
        def set_length(self, length):
            self.length = length


    setter = ParameterSetter()
    measurement = calibrated_readout.calibrate_preparation_and_readout_confusion(device, qubit_readout_pulse,
                                                            readout_device,
                                                            (parameter_values, setter.set_parameter, parameter_name),
                                                            (lengths, setter.set_length, 'length'),
                                                            pause_length=0, middle_seq_generator=setter.filler_funct,
                                                            additional_metadata=metadata,
                                                            additional_references=references)
    return measurement