from ..ponyfiles.data_structures import *
from . import channel_amplitudes
import traceback
#from .import
from .. import pulses
from . import excitation_pulse
from . import Rabi
from . import channel_amplitudes
from . import Ramsey


def iswap_frequency_scan(device, gate, q):
    channel_amplitudes_ = two_qubit_gate_channel_amplitudes(device, gate)
    frequency_delta = float(device.get_sample_global(name='parametric_frequency_shift_calibration_frequency_offset'))
    vf_pulse = [device.pg.pmulti(0, (gate.metadata['carrier_name'], pulses.vf, frequency_delta))]

    def filler_func(length):
        return vf_pulse + \
               excitation_pulse.get_rect_cos_pulse_sequence(device = device,
                                           channel_amplitudes = channel_amplitudes_,
                                           tail_length = float(gate.metadata['tail_length']),
                                           length = length,
                                           phase = 0.0) #+ \
               #excitation_pulse.get_rect_cos_pulse_sequence(device=device,
               #                            channel_amplitudes = channel_amplitudes_,
               #                            tail_length=float(gate.metadata['tail_length']),
               #                            length=length / 2,
               #                            phase=np.pi*float(gate.metadata['carrier_harmonic']))

    return Ramsey.Ramsey_adaptive(device=device, qubit_id=gate.metadata[q], set_frequency=False,
                           delay_seq_generator=filler_func, measurement_type='Ramsey_long_process', additional_references={'long_process': gate.id})


def calibrate_iswap_phase_single_pulse(device, gate_pulse, num_pulses):
    # excite qubit 1 with pi/2 pulse, perform iswap, apply pi./2 pulse to qubit 2
    phase_scan1 = Ramsey.Ramsey_process(device, qubit_id1=gate_pulse.metadata['q1'], qubit_id2=gate_pulse.metadata['q2'],
                                     process=gate_pulse)
    phase_scan2 = Ramsey.Ramsey_process(device, qubit_id1=gate_pulse.metadata['q2'], qubit_id2=gate_pulse.metadata['q1'],
                                     process=gate_pulse)
    return phase_scan1, phase_scan2, phase_scan1.fit, phase_scan2.fit


def two_qubit_gate_channel_amplitudes (device, gate):
    return channel_amplitudes.channel_amplitudes(device, **{gate.metadata['carrier_name']:float(gate.metadata['amplitude'])})


def get_frequency_shift(device, gate, recalibrate=True, force_recalibration=False):
    # frequency difference calibration
    try:
        if force_recalibration:
            raise IndexError('force_recalibration')
        Ramsey_process_q1 = device.exdir_db.select_measurement(measurement_type='Ramsey_long_process', references_that={'long_process': gate.id}, metadata={'qubit_id': gate.metadata['q1']})
        Ramsey_process_q1_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':Ramsey_process_q1.id})
    except IndexError as e:
        if not recalibrate:
            raise
        traceback.print_exc()
        iswap_frequency_scan(device, gate, q='q1')
        Ramsey_process_q1 = device.exdir_db.select_measurement(measurement_type='Ramsey_long_process', references_that={'long_process': gate.id}, metadata={'qubit_id': gate.metadata['q1']})
        Ramsey_process_q1_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':Ramsey_process_q1.id})
    frequency_delta_q1 = float(Ramsey_process_q1_fit.metadata['f']) - float(Ramsey_process_q1.metadata['target_offset_freq'])

    try:
        if force_recalibration:
            raise IndexError('force_recalibration')
        Ramsey_process_q2 = device.exdir_db.select_measurement(measurement_type='Ramsey_long_process', references_that={'long_process': gate.id}, metadata={'qubit_id': gate.metadata['q2']})
        Ramsey_process_q2_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':Ramsey_process_q2.id})
    except IndexError as e:
        if not recalibrate:
            raise
        traceback.print_exc()
        iswap_frequency_scan(device, gate, q='q2')
        Ramsey_process_q2 = device.exdir_db.select_measurement(measurement_type='Ramsey_long_process', references_that={'long_process': gate.id}, metadata={'qubit_id': gate.metadata['q2']})
        Ramsey_process_q2_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source':Ramsey_process_q2.id})
    frequency_delta_q2 = float(Ramsey_process_q2_fit.metadata['f']) - float(Ramsey_process_q2.metadata['target_offset_freq'])

    frequency_shift = frequency_delta_q1-frequency_delta_q2

    print('frequency_delta_q1: ', frequency_delta_q1)
    print('frequency_delta_q2: ', frequency_delta_q2)
    print('frequency_shift: ', frequency_shift)

    return frequency_delta_q1, frequency_delta_q2


def get_long_process_vf(device, gate, inverse=False, recalibrate=True, force_recalibration=False):
    frequency_delta_q1, frequency_delta_q2 = get_frequency_shift(device, gate, recalibrate=recalibrate, force_recalibration=force_recalibration)
    factor = -1 if not inverse else 1
    vf1 = excitation_pulse.get_vf(device, gate.metadata['q1'], frequency_delta_q1*factor)
    vf2 = excitation_pulse.get_vf(device, gate.metadata['q2'], frequency_delta_q2*factor)
    return vf1+vf2


def calibrate_parametric_iswap_length(device, gate, calibration_qubit='1'):
    if calibration_qubit == '1':
        excite = excitation_pulse.get_excitation_pulse(device=device, qubit_id=gate.metadata['q1'], rotation_angle=np.pi)
    else:
        excite = excitation_pulse.get_excitation_pulse(device=device, qubit_id=gate.metadata['q2'],
                                                       rotation_angle=np.pi)

    vf_pulse = get_long_process_vf(device, gate)
    Rabi_rect_measurement_q1 = Rabi.Rabi_rect_adaptive(device=device, qubit_id=[gate.metadata['q1'], gate.metadata['q2']], tail_length = float(gate.metadata['tail_length']),
            channel_amplitudes=two_qubit_gate_channel_amplitudes(device, gate), measurement_type=gate.metadata['physical_type']+'_calibration', pre_pulses=(excite,  vf_pulse))
    ## TODO: fix this shit!!!!! it doesn't use the frequency shifts after all.

    fit_m1 = Rabi_rect_measurement_q1.fit

    return Rabi_rect_measurement_q1, fit_m1


def get_gate_calibration(device, gate, recalibrate=True, force_recalibration=False):
    channel_amplitudes_ = two_qubit_gate_channel_amplitudes(device, gate)

    frequency_delta_q1, frequency_delta_q2 = get_frequency_shift(device, gate, recalibrate, force_recalibration)
    frequency_shift = frequency_delta_q1 - frequency_delta_q2

    try:
        if force_recalibration:
            raise IndexError('force_recalibration')
        query = two_qubit_Rabi_measurements_query(qubit_ids=[gate.metadata['q1'], gate.metadata['q2']],
                                                   measurement_type=gate.metadata['physical_type']+'_calibration',
                                                   frequency=device.awg_channels[list(channel_amplitudes_.metadata.keys())[0]].get_frequency(),
                                                   frequency_tolerance=device.get_qubit_constant(qubit_id=gate.metadata['q1'], name='frequency_rounding'),
                                                   frequency_controls=device.get_frequency_control_measurement_id(qubit_id=gate.metadata['q1']),
                                                   channel_amplitudes_=channel_amplitudes_.id,
                                                   frequency_shift=frequency_shift)
        # print (query)
        fit_id = device.exdir_db.db.db.select(query)
        fit_m1 = device.exdir_db.select_measurement_by_id(fit_id[0])
        m1 = device.exdir_db.select_measurement_by_id(fit_m1.references['fit_source'])
        #m1 = device.exdir_db.select_measurement(measurement_type='two_qubit_Rabi',
        #										references_that={'channel_amplitudes':channel_amplitudes_.id})
        #fit_m1 = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d',
        #											references_that={'fit_source':m1.id})
    except IndexError as e:
        traceback.print_exc()
        if not recalibrate:
            raise
        m1, fit_m1, = calibrate_parametric_iswap_length(device, gate)
    rotation_angle = float(gate.metadata['rotation_angle'])
    f = float(fit_m1.metadata['f'])
    phi = float(fit_m1.metadata['phi'])
    candidate_num_periods = [(rotation_angle-phi)/(2*np.pi), (rotation_angle+phi)/(2*np.pi),
                         (rotation_angle-phi+np.pi)/(2*np.pi), (rotation_angle+phi+np.pi)/(2*np.pi)]
    candidate_lengths_minimal_positive = [(num_periods-np.floor(num_periods))/f for num_periods in candidate_num_periods]
    # now look at our fit and check which phase most corresponds to what is expected from iSWAp
    # we know the qubit order and that this measurement should have state 01 at minimum at pi, and state 10 at max.
    target = fit_m1.datasets['resultnumbers'].data[:,1]-fit_m1.datasets['resultnumbers'].data[:,2]
    target_values = np.interp(fit_m1.datasets['resultnumbers'].parameters[-2].values, target, candidate_lengths_minimal_positive)
    print ('candidates: ', candidate_lengths_minimal_positive)
    print ('target values of candidates: ', target_values)
    length = candidate_lengths_minimal_positive[np.argmin(target_values)]
    print ('length: ', length)

    gate_without_phase = ParametricTwoQubitGate(device,
                                  q1=gate.metadata['q1'],
                                  q2=gate.metadata['q2'],
                                  phase_q1=np.nan, #gate['q1'],
                                  phase_q2=np.nan, #gate['q2'],
                                  rotation_angle = np.pi,
                                  length=length, tail_length=gate.metadata['tail_length'],
                                  channel_amplitudes=channel_amplitudes_.id,
                                  Rabi_rect_measurement=m1.id,
                                  gate_settings=gate.id,
                                  frequency_shift=frequency_shift)

    try:
        references = {'process': gate_without_phase.id}
        metadata_scan1 = {'q1': gate.metadata['q1'],
                          'q2': gate.metadata['q2']}
        metadata_scan2 = {'q1': gate.metadata['q2'],
                          'q2': gate.metadata['q1']}
        phase_scan1 = device.exdir_db.select_measurement(measurement_type='Ramsey_process', metadata=metadata_scan1, references_that=references)
        phase_scan2 = device.exdir_db.select_measurement(measurement_type='Ramsey_process', metadata=metadata_scan2, references_that=references)
        phase_scan1_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source': phase_scan1.id})
        phase_scan2_fit = device.exdir_db.select_measurement(measurement_type='fit_dataset_1d', references_that={'fit_source': phase_scan2.id})
    except IndexError as e:
        traceback.print_exc()
        if not recalibrate:
            raise
        phase_scan1, phase_scan2, phase_scan1_fit, phase_scan2_fit = calibrate_iswap_phase_single_pulse(device, gate_without_phase, 1)

    # we want -np.pi/2. phase from iSWAP; exchange qubits since phase is applied after iSWAP
    phase_q2 = -float(phase_scan1_fit.metadata['phi']) - np.pi/2.
    phase_q1 = -float(phase_scan2_fit.metadata['phi']) - np.pi/2.

    gate_with_phase = ParametricTwoQubitGate(device,
                                  q1=gate.metadata['q1'],
                                  q2=gate.metadata['q2'],
                                  phase_q1=phase_q1, #gate['q1'],
                                  phase_q2=phase_q2, #gate['q2'],
                                  rotation_angle = np.pi,
                                  length=length, tail_length=0,
                                  channel_amplitudes=channel_amplitudes_.id,
                                  Rabi_rect_measurement=m1.id,
                                  phase_scan_q1=phase_scan1.id,
                                  phase_scan_q2=phase_scan2.id,
                                  gate_settings=gate.id,
                                  frequency_shift=frequency_shift)
    return gate_with_phase


class ParametricTwoQubitGate(MeasurementState):
    def __init__(self, *args, **kwargs):
        self.device = args[0]
        if len(args) == 2 and isinstance(args[1], MeasurementState) and not len(kwargs): # copy constructor
            super().__init__(args[1])
        else: # otherwise initialize from dict and device
            metadata = {'rotation_angle': str(kwargs['rotation_angle']),
                        'pulse_type': 'rect',
                        'length': str(kwargs['length']),
                        'tail_length': str(kwargs['tail_length']),
                        'phase_q1': str(kwargs['phase_q1']),
                        'phase_q2': str(kwargs['phase_q2']),
                        'q1': str(kwargs['q1']),
                        'q2': str(kwargs['q2']),
                        'frequency_shift': str(kwargs['frequency_shift'])}

            if 'calibration_type' in kwargs:
                metadata['calibration_type'] = kwargs['calibration_type']

            references = {'channel_amplitudes': int(kwargs['channel_amplitudes']),
                          'Rabi_rect': int(kwargs['Rabi_rect_measurement']),
                          'gate_settings': int(kwargs['gate_settings'])}
            if 'phase_scan_q1' in kwargs:
                references['phase_scan_q1'] = int(kwargs['phase_scan_q1'])
            if 'phase_scan_q2' in kwargs:
                references['phase_scan_q2'] = int(kwargs['phase_scan_q2'])

            # check if such measurement exists
            try:
                measurement = self.device.exdir_db.select_measurement(measurement_type='parametric_two_qubit_gate', metadata=metadata, references_that=references)
                super().__init__(measurement)
            except IndexError as e:
                traceback.print_exc()
                super().__init__(measurement_type='parametric_two_qubit_gate', sample_name=self.device.exdir_db.sample_name, metadata=metadata, references=references)
                self.device.exdir_db.save_measurement(self)

        #inverse_references = {v:k for k,v in self.references.items()}
        #print ('inverse_references in __init__:', inverse_references)
        self.channel_amplitudes = channel_amplitudes.channel_amplitudes(self.device.exdir_db.select_measurement_by_id(self.references['channel_amplitudes']))

    def get_pulse_sequence(self, phase=0.0):
        pulse_sequence = get_long_process_vf(self.device, self, inverse=False)
        pulse_sequence += excitation_pulse.get_rect_cos_pulse_sequence(device = self.device,
                                           channel_amplitudes = self.channel_amplitudes,
                                           tail_length = float(self.metadata['tail_length']),
                                           length = float(self.metadata['length']),
                                           phase = phase)
        if np.isfinite(float(self.metadata['phase_q1'])):
            pulse_sequence += excitation_pulse.get_s(self.device, self.metadata['q1'],
                                                     phase=float(self.metadata['phase_q1']))
        if np.isfinite(float(self.metadata['phase_q2'])):
            pulse_sequence += excitation_pulse.get_s(self.device, self.metadata['q2'],
                                                     phase=float(self.metadata['phase_q2']))
        pulse_sequence += get_long_process_vf(self.device, self, inverse=True)

        post_pause_length = float(self.device.get_sample_global(name='two_qubit_pulse_post_pause'))
        pulse_sequence += [self.device.pg.pmulti(post_pause_length)]
        return pulse_sequence


def two_qubit_Rabi_measurements_query (qubit_ids, measurement_type, frequency, frequency_tolerance, frequency_controls, channel_amplitudes_, frequency_shift):
    '''
    Perfectly ugly query for retrieving Rabi oscillation measurements corresponding to a qubit, and, possibly, a 'channel'
    '''
    channel_amplitudes_clause = 'AND channel_amplitudes.id = {}'.format(channel_amplitudes_)

    query = '''SELECT fit_id FROM (
    SELECT
        channel_amplitudes.id channel_amplitudes_id,
        fit.id fit_id,
         LEAST(CAST(num_periods_scan.value AS DECIMAL),
               CASE WHEN num_periods_decay.value='inf' THEN 3 ELSE CAST(num_periods_decay.value AS DECIMAL) END)	 score
    --fit.id,
    --Rabi_freq.value as f,
    --num_periods_decay.value,
    --num_periods_scan.value,
    --MIN(CAST (channel_calibration_frequency.value AS DECIMAL)) min_freq,
    --MAX(CAST (channel_calibration_frequency.value AS DECIMAL)) max_freq

    --MAX(fit.id) fit_id

    --channel_amplitudes.*

    FROM
    data measurement

    INNER JOIN reference fit_measurement_reference ON
        fit_measurement_reference.ref_type = 'fit_source' AND
        fit_measurement_reference.that = measurement.id
    INNER JOIN data fit ON
        fit_measurement_reference.this = fit.id

    INNER JOIN metadata qubit_id_metadata ON
        qubit_id_metadata.data_id = measurement.id AND
        qubit_id_metadata.name = 'qubit_id' AND
        qubit_id_metadata.value = '{qubit_ids}'

    INNER JOIN metadata frequency_goodness_test_metadata ON
        frequency_goodness_test_metadata.data_id = fit.id AND
        frequency_goodness_test_metadata.name = 'frequency_goodness_test' AND
        frequency_goodness_test_metadata.value = '1'

    INNER JOIN metadata MSE_rel_metadata ON
        MSE_rel_metadata.data_id = fit.id AND
        MSE_rel_metadata.name = 'MSE_rel'

    INNER JOIN reference channel_amplitudes_reference ON
        channel_amplitudes_reference.this = measurement.id AND
        channel_amplitudes_reference.ref_type = 'channel_amplitudes'

    INNER JOIN data channel_amplitudes ON
        channel_amplitudes.id = channel_amplitudes_reference.that

    INNER JOIN reference frequency_controls ON
        frequency_controls.this = measurement.id AND
        frequency_controls.ref_type = 'frequency_controls' AND
        frequency_controls.that = {frequency_controls}

    INNER JOIN metadata channel ON
        channel.data_id = channel_amplitudes.id

--    INNER JOIN reference channel_calibration_reference ON
--        channel_calibration_reference.ref_type='channel_calibration' AND
--        channel_calibration_reference.this = channel.data_id

--    INNER JOIN data channel_calibration ON
--        channel_calibration.id = channel_calibration_reference.that

--    INNER JOIN metadata channel_calibration_frequency ON
--        channel_calibration_frequency.data_id = channel_calibration.id AND
--        channel_calibration_frequency.name = 'frequency' --AND
--        --ABS(CAST (channel_calibration_frequency.value AS DECIMAL) - {frequency}) < {frequency_tolerance}

    INNER JOIN metadata Rabi_freq ON
        Rabi_freq.data_id = fit.id AND
        Rabi_freq.name = 'f'

    INNER JOIN metadata num_periods_decay ON
        num_periods_decay.data_id = fit.id AND
        num_periods_decay.name = 'num_periods_decay'

    INNER JOIN metadata num_periods_scan ON
        num_periods_scan.data_id = fit.id AND
        num_periods_scan.name = 'num_periods_scan'
    
    INNER JOIN metadata frequency_shift ON
        frequency_shift.data_id = measurement.id AND
        frequency_shift.name = 'frequency_shift' AND
        ABS(CAST(frequency_shift.value AS DECIMAL) - ({frequency_shift})) < {frequency_tolerance}

    WHERE
        measurement.measurement_type = '{measurement_type}' AND
        (NOT measurement.invalid OR (measurement.invalid IS NULL))
        {channel_amplitudes_clause}
--		  (SELECT COUNT(*) FROM metadata
--			  WHERE metadata.data_id = channel_amplitudes.id) = 1

    GROUP BY channel_amplitudes_id,
             fit.id,
             LEAST(CAST(num_periods_scan.value AS DECIMAL),
                   CASE WHEN num_periods_decay.value='inf' THEN 3 ELSE CAST(num_periods_decay.value AS DECIMAL) END)

--    HAVING
--        ABS(MIN(CAST (channel_calibration_frequency.value AS DECIMAL))- {frequency})<{frequency_tolerance}
--        AND ABS(MAX(CAST (channel_calibration_frequency.value AS DECIMAL))- {frequency})<{frequency_tolerance}

    ORDER BY
         LEAST(CAST(num_periods_scan.value AS DECIMAL),
               CASE WHEN num_periods_decay.value='inf' THEN 3 ELSE CAST(num_periods_decay.value AS DECIMAL) END)	 DESC

    ) Rabi_measurements
'''
    return query.format(measurement_type=measurement_type,
        qubit_ids=','.join(qubit_ids),
        frequency=frequency,
        frequency_tolerance = frequency_tolerance,
        channel_amplitudes_clause = channel_amplitudes_clause,
        frequency_controls = frequency_controls,
        frequency_shift = frequency_shift)
