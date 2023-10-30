from qsweepy.qubit_calibrations.readout_pulse import *
from qsweepy.libraries import readout_classifier
from qsweepy.qubit_calibrations import channel_amplitudes
from qsweepy.qubit_calibrations import excitation_pulse2 as excitation_pulse
from qsweepy.libraries import single_shot_readout2 as single_shot_readout
from qsweepy.libraries import single_shot_readout3 as single_shot_readout3
from qsweepy.libraries import logistic_regression_classifier
from qsweepy.libraries import logistic_regression_classifier2 as logistic_regression_classifier2
from qsweepy.qubit_calibrations import sequence_control
from qsweepy import zi_scripts
import numpy as np

import traceback


def get_confusion_matrix(device, qubit_ids, pause_length=0, recalibrate=True, force_recalibration=False,gauss=True, sort='best'):
    qubit_readout_pulse, readout_device = get_calibrated_measurer(device, qubit_ids)
    # TODO
    '''Warning'''
    excitation_pulses = {qubit_id: excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi,gauss=gauss, sort=sort) for
                         qubit_id in qubit_ids}
    ''''''
    references = {('excitation_pulse', qubit_id): pulse.id for qubit_id, pulse in excitation_pulses.items()}
    references['readout_pulse'] = qubit_readout_pulse.id
    metadata = {'qubit_ids': qubit_readout_pulse.metadata['qubit_ids'], 'pause_length': str(pause_length)}
    try:
        assert not force_recalibration
        confusion_matrix = device.exdir_db.select_measurement(measurement_type='confusion_matrix',
                                                              references_that=references, metadata=metadata)
    except:
        if not recalibrate:
            raise
        confusion_matrix = calibrate_preparation_and_readout_confusion(device=device, qubit_readout_pulse=qubit_readout_pulse,
                                                                       readout_device=readout_device,
                                                                       pause_length=pause_length, gauss=gauss)

    return qubit_readout_pulse, readout_device, confusion_matrix


def calibrate_preparation_and_readout_confusion(device, qubit_readout_pulse, readout_device, *extra_sweep_args,
                                                pause_length=0, middle_seq_generator = None,
                                                additional_references = {}, additional_metadata = {},
                                                gauss=True, sort='best'):
    qubit_ids = qubit_readout_pulse.metadata['qubit_ids'].split(',')
    target_qubit_states = [0] * len(qubit_ids)
    # TODO
    '''Warning'''
    excitation_pulses = {qubit_id: excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi/2.,gauss=gauss, sort=sort) for
                         qubit_id in qubit_ids}
    ''''''
    references = {('excitation_pulse', qubit_id): pulse.id for qubit_id, pulse in excitation_pulses.items()}
    references['readout_pulse'] = qubit_readout_pulse.id

    re_sequence = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)

    #TODO
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_ids[0]).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():

        control_qubit_awg = ex_channel.parent
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:

        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq

        if awg == control_qubit_awg:
            if seq_id == control_qubit_seq_id:
                control_qubit_sequence = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)

    def set_target_state(state):
        re_sequence.awg.stop_seq(re_sequence.params['sequencer_id'])
        preparation_sequence = []
        for _id, qubit_id in enumerate(qubit_ids):
            qubit_state = (1 << _id) & state
            if qubit_state:
                # TODO
                '''Warning'''
                preparation_sequence.extend(excitation_pulses[qubit_id].get_pulse_sequence(0))
                preparation_sequence.extend(excitation_pulses[qubit_id].get_pulse_sequence(0))
        if middle_seq_generator is not None:
            # TODO
            '''Warning'''
            middle_pulse = middle_seq_generator()
            preparation_sequence.extend(middle_pulse)
        else:
            middle_pulse = []
        # TODO
        '''Warning'''
        sequence_control.set_preparation_sequence(device, ex_sequencers, preparation_sequence)
        re_sequence.awg.start_seq(re_sequence.params['sequencer_id'])
    if middle_seq_generator is not None:
        measurement_type = 'confusion_matrix_middle_seq'
    else:
        measurement_type = 'confusion_matrix'
    metadata = {'qubit_ids': qubit_readout_pulse.metadata['qubit_ids'],
                'pause_length': str(pause_length)}
    references.update(additional_references)
    metadata.update(additional_metadata)

    return device.sweeper.sweep(readout_device,
                                *extra_sweep_args,
                                (np.arange(2 ** len(qubit_ids)), set_target_state, 'Target state', ''),
                                measurement_type=measurement_type,
                                references=references,
                                metadata=metadata)


def get_calibrated_measurer(device, qubit_ids, qubit_readout_pulse=None, transition='01', recalibrate=True, force_recalibration=False, raw=False,
                            internal_avg=False, readouts_per_repetition=1, get_thresholds=False):
    from .readout_pulse2 import get_multi_qubit_readout_pulse

    if qubit_readout_pulse is None:
        qubit_readout_pulse = get_multi_qubit_readout_pulse(device, qubit_ids)
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'delay_calibration': device.modem.delay_measurement.id}
    features = []
    thresholds = []

    for qubit_id in qubit_ids:
        metadata = {'qubit_id': qubit_id, 'transition': transition}
        try:
            if force_recalibration:
                raise ValueError('Forcing recalibration')
            print (metadata)
            measurement = device.exdir_db.select_measurement(measurement_type='readout_calibration', metadata=metadata,
                                                             references_that=references)
            print("reference: readout_calibration: {}".format(measurement.id))
            print('\x1b[1;30;44m' + 'GET CALIBRATED MEASURER!' + '\x1b[0m')
        except Exception as e:
            print(traceback.print_exc())
            if not recalibrate:
                raise
            measurement = calibrate_readout(device, qubit_id, qubit_readout_pulse)

        features.append(measurement.datasets['feature'].data)
        thresholds.append(measurement.datasets['threshold'].data.ravel()[0])


    readout_device = device.set_adc_features_and_thresholds(features, thresholds, disable_rest=True, raw=raw, internal_avg=internal_avg)
    nums = int(device.get_sample_global(name='calibrated_readout_nums'))
    readout_device.set_adc_nums(nums * readouts_per_repetition)
    readout_device.set_adc_nop(int(device.get_sample_global('readout_adc_points')))


    # print(dir(readout_device))
    # print(readout_device.get_data().shape)


    #readout_device.output_raw = False
    #readout_device.output_result = False
    #readout_device.output_resnum = True

    if not get_thresholds:
        return qubit_readout_pulse, readout_device  # , features, thresholds
    else:
        return qubit_readout_pulse, readout_device, thresholds, features


def get_calibrated_measurer2(device, qubit_id, qubit_readout_pulse=None, recalibrate=True, force_recalibration=False, raw=False,
                             internal_avg=False, readouts_per_repetition=1, qutrit=False, M0_id=None):
    """
    Train and set classifier model to adc device
    """
    from .readout_pulse2 import get_multi_qubit_readout_pulse

    if qubit_readout_pulse is None:
        qubit_readout_pulse = get_multi_qubit_readout_pulse(device, [qubit_id])
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'delay_calibration': device.modem.delay_measurement.id}

    metadata = {'qubit_id': qubit_id, 'logistic_regression_classifier': str(True), 'post_selection': str(False)}
    try:
        if force_recalibration:
            raise ValueError('Forcing recalibration')
        print(metadata)
        if qutrit:
            if M0_id == None:
                measurement = device.exdir_db.select_measurement(measurement_type='readout_qutrit_calibration', metadata=metadata,
                                                                references_that=references)
            else:
                measurement = device.exdir_db.select_measurement_by_id(M0_id)
        else:
            if M0_id == None:
                measurement = device.exdir_db.select_measurement(measurement_type='readout_calibration', metadata=metadata,
                                                            references_that=references)
            else:
                measurement = device.exdir_db.select_measurement_by_id(M0_id)
        print('\x1b[1;30;44m' + 'Get calibrated measurer and train model!' + '\x1b[0m')

    except Exception as e:
        print(traceback.print_exc())
        if not recalibrate:
            raise
        if qutrit:
            raise ValueError("Calibrate qutrit readout")
        else:
            measurement = calibrate_readout(device, qubit_id, qubit_readout_pulse, internal_avg=internal_avg,
                                        classifier_type='logistic_regression_classifier')

    print("reference: readout_calibration: {}".format(measurement.id))

    # get w and markers from measurement and train model
    w, marker = measurement.datasets['w'].data, measurement.datasets['marker'].data
    feature0, feature1 = measurement.datasets['feature0'].data, measurement.datasets['feature1'].data
    if qutrit:
        model = logistic_regression_classifier2.LogisticRegressionReadoutClassifier(states=3)
    else:
        model = logistic_regression_classifier2.LogisticRegressionReadoutClassifier(states=2)
    model.feature0, model.feature1 = feature0, feature1
    model.train(w=w, marker=marker)


    model.nums_adc = int(device.get_sample_global('readout_adc_points'))

    # set model
    readout_device = device.set_adc_model(model, raw=raw)
    nums = int(device.get_sample_global(name='calibrated_readout_nums'))
    readout_device.set_adc_nums(nums * readouts_per_repetition)
    readout_device.set_adc_nop(int(device.get_sample_global('readout_adc_points')))

    return qubit_readout_pulse, readout_device, model


def get_qutrit_calibrated_measurer(device, qubit_id, qubit_readout_pulse=None, recalibrate=True, force_recalibration=False):
    from .readout_pulse2 import get_multi_qubit_readout_pulse
    print('GET CALIBRATED MEASURER')
    if qubit_readout_pulse is None:
        qubit_readout_pulse = get_multi_qubit_readout_pulse(device, qubit_id)
    references = {'readout_pulse': qubit_readout_pulse.id,
                  'delay_calibration': device.modem.delay_measurement.id}
    features = []
    thresholds = []

    models = []
    features0 = []
    features1 = []
    if len(qubit_id) > 1:
        raise ValueError("List length has to be equal one!")
    else:
        metadata = {'qubit_id': qubit_id[0]}
        try:
            if force_recalibration:
                raise ValueError('Forcing recalibration')
            print (metadata)
            measurement = device.exdir_db.select_measurement(measurement_type='readout_qutrit_calibration',
                                                             metadata=metadata,
                                                             references_that=references)
        except Exception as e:
            print(traceback.print_exc())
            if not recalibrate:
                raise
            raise ValueError("Calibrate qutrit readout!")
        print("reference: readout_qutrit_calibration: {}".format(measurement.id))
        # get and train model
        w, marker = measurement.datasets['w'].data, measurement.datasets['marker'].data
        feature0, feature1 = measurement.datasets['feature0'].data, measurement.datasets['feature1'].data
        model = logistic_regression_classifier.LogisticRegressionReadoutClassifier()
        model.feature0, model.feature1 = feature0, feature1
        model.train(w=w, marker=marker)
        model.nums_adc = int(device.get_sample_global('readout_adc_points'))

    # set model and features
    readout_device = device.set_adc_model(model)
    nums = int(device.get_sample_global(name='calibrated_readout_nums'))
    readout_device.set_adc_nums(nums)
    readout_device.set_adc_nop(int(device.get_sample_global('readout_adc_points')))


    # print(dir(readout_device))
    # print(readout_device.get_data().shape)


    #readout_device.output_raw = False
    #readout_device.output_result = False
    #readout_device.output_resnum = True

    return qubit_readout_pulse, readout_device  # , features, thresholds


def calibrate_readout(device, qubit_id, qubit_readout_pulse, transition='01', ignore_other_qubits=None, dbg_storage=False,
                      gauss=True, sort='best', internal_avg=True, ex_pre_pulse=None, dbg_storage_samples=False,
                      post_selection_flag=False, classifier_type=None, pre_pulse_delay=None, M0_id=None):
    """
    Calibrate readout
    :param device:
    :param qubit_id:
    :param qubit_readout_pulse:
    :param transition:
    :param ignore_other_qubits:
    :param dbg_storage:
    :param gauss:
    :param sort: type of sorting for Rabi references "best" or "newest"
    :param internal_avg: internal avg of a measurement, if True then dbg_storage=False and averaging is performed
    :param ex_pre_pulse: excitations pre pulse before pulse sequence
    :param dbg_storage_samples: if True then save samples in m.datasets['x'] and markers in m.datasets['y']
    :param post_selection_flag:
    :param classifier_type:
    :return:
    """
    if not post_selection_flag:
        adc, mnames = device.setup_adc_reducer_iq(qubit_id, raw=True,  internal_avg=internal_avg)
        adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    else:
        # in case of post selection use features and thresholds or model from previous readout calibration
        if not classifier_type:
            qubit_readout_pulse_, adc = get_calibrated_measurer(device, qubit_id, raw=True, internal_avg=False,
                                                                readouts_per_repetition=2)
        elif classifier_type == 'logistic_regression_classifier':
            qubit_readout_pulse_, adc, _ = get_calibrated_measurer2(device, qubit_id, raw=True, internal_avg=False,
                                                                readouts_per_repetition=2, M0_id=M0_id)
        else:
            raise ValueError

    nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
    old_nums = adc.get_adc_nums()

    if ignore_other_qubits is None:
        ignore_other_qubits = (device.get_qubit_constant(qubit_id=qubit_id, name='readout_calibration_ignore_other_qubits') == 'True')
    print ('ignore_other_qubits', ignore_other_qubits)

    other_qubit_pulse_sequence = []
    references = {}
    if not ignore_other_qubits:
        for other_qubit_id in device.get_qubit_list():
            if other_qubit_id != qubit_id:
                # TODO
                '''Warning'''
                half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id,
                                                                           rotation_angle=np.pi / 2,gauss=gauss,
                                                                           sort = sort)
                references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
                other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0))
    # TODO
    '''Warning'''
    if gauss:
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi/2,
                                                                       transition=transition, gauss=gauss,
                                                                       sort = sort)
    else:
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi,
                                                                       gauss=gauss, transition=transition,
                                                                       sort = sort)

    if ex_pre_pulse:
        metadata = {'qubit_id': qubit_id,
                    'transition': transition,
                    'averages': nums,
                    'ignore_other_qubits': ignore_other_qubits,
                    'ex_pre_pulse': ex_pre_pulse.id,
                    'ex_pre_pulse_length': ex_pre_pulse.metadata['length']}
    else:
        metadata = {'qubit_id': qubit_id,
                    'transition': transition,
                    'averages': nums,
                    'ignore_other_qubits': ignore_other_qubits}

    if post_selection_flag:
        metadata.update({'post_selection': str(True)})
    else:
        metadata.update({'post_selection': str(False)})


    references.update({'readout_pulse': qubit_readout_pulse.id,
                       'excitation_pulse': qubit_excitation_pulse.id,
                       'delay_calibration': device.modem.delay_measurement.id})

    #TODO
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=transition).keys()][0]
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
                                               awg_amp=1, use_modulation=True, pre_pulses=[],
                                               post_selection_flag=post_selection_flag)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
                                               post_selection_flag=post_selection_flag)
            control_sequence = ex_seq


        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        if pre_pulse_delay:
            ex_seq.set_prepulse_delay(pre_pulse_delay)
        ex_sequencers.append(ex_seq)

    if gauss:
        excitation = qubit_excitation_pulse.get_pulse_sequence(0) + qubit_excitation_pulse.get_pulse_sequence(0)
    else:
        excitation = qubit_excitation_pulse.get_pulse_sequence(0)

    if ex_pre_pulse:
        pre_excitation = ex_pre_pulse.get_pulse_sequence(0)
    else:
        pre_excitation = []

    sequence_control.set_preparation_sequence(device, ex_sequencers, other_qubit_pulse_sequence + pre_excitation + excitation)
                                                                       # qubit_excitation_pulse.get_pulse_sequence(0)+
                                                                       #  qubit_excitation_pulse.get_pulse_sequence(0))

    '''Warning'''
    #readout_sequencer = sequence_control.define_readout_control_seq(device, readout_channel)
    #raise ValueError('fallos')

    readout_sequencer = sequence_control.define_readout_control_seq(device, qubit_readout_pulse, post_selection_flag=post_selection_flag)
    readout_sequencer.start()

    if not classifier_type:
        classifier = single_shot_readout.single_shot_readout(device=device,
                                                             adc=adc,
                                                             prepare_seqs=[other_qubit_pulse_sequence,
                                                                           other_qubit_pulse_sequence + pre_excitation +
                                                                           excitation],
                                                                           # qubit_excitation_pulse.get_pulse_sequence(0)+
                                                                           # qubit_excitation_pulse.get_pulse_sequence(0)],
                                                             ex_seqs=ex_sequencers,
                                                             ro_seq=readout_sequencer,
                                                             control_seq = control_qubit_sequence,
                                                             #pulse_generator=device.pg,
                                                             ro_delay_seq=None,
                                                             _readout_classifier=readout_classifier.binary_linear_classifier(),
                                                             adc_measurement_name='Voltage',
                                                             dbg_storage=dbg_storage, dbg_storage_samples=dbg_storage_samples,
                                                             post_selection_flag=post_selection_flag)

        classifier.readout_classifier.cov_mode = 'equal'

    elif classifier_type == 'logistic_regression_classifier':
        from qsweepy.libraries import single_shot_readout_postselection as single_shot_readout_postselection
        metadata.update({'logistic_regression_classifier': str(True)})
        classifier = single_shot_readout_postselection.SingleShotReadoutPostSelection(device=device,
                                                             adc=adc,
                                                             prepare_seqs=[other_qubit_pulse_sequence,
                                                                           other_qubit_pulse_sequence + pre_excitation +
                                                                           excitation],
                                                             # qubit_excitation_pulse.get_pulse_sequence(0)+
                                                             # qubit_excitation_pulse.get_pulse_sequence(0)],
                                                             ex_seqs=ex_sequencers,
                                                             ro_seq=readout_sequencer,
                                                             control_seq=control_qubit_sequence,
                                                             # pulse_generator=device.pg,
                                                             ro_delay_seq=None,
                                                             _readout_classifier=None,
                                                             adc_measurement_name='Voltage',
                                                             dbg_storage=dbg_storage,
                                                             dbg_storage_samples=dbg_storage_samples,
                                                             post_selection_flag=post_selection_flag)
    else:
        raise ValueError
    try:
        if not post_selection_flag:
            adc.set_adc_nums(nums)
        measurement = device.sweeper.sweep(classifier,
                                           measurement_type='readout_calibration',
                                           metadata=metadata,
                                           references=references)

    except:
        raise
    finally:
        if not post_selection_flag:
            adc.set_adc_nums(old_nums)

    if classifier_type:
        print('fidelity', np.mean(np.diag(classifier.confusion_matrix)))
        metadata.update({'fidelity': str(np.mean(np.diag(classifier.confusion_matrix)))})

    return measurement
    # classifier.repeat_samples = 2


# def calibrate_qutrit_readout(device, qubit_transitions_id, qubit_readout_pulse, ignore_other_qubits=None, dbg_storage=False,
#                              gauss=True, sort='best', dbg_storage_samples=False, post_selection_flag=False,
#                              pre_pulse_delay=None, M0_id=None):

def calibrate_qutrit_readout(device, qubit_id, qubit_readout_pulse, ignore_other_qubits=None,
                             dbg_storage=False,
                             gauss=True, sort='best', dbg_storage_samples=False, post_selection_flag=False,
                             pre_pulse_delay=None, M0_id=None):
    """
    Calibrate readout
    :param device:
    :param qubit_id:
    :param qubit_readout_pulse:
    :param ignore_other_qubits:
    :param dbg_storage:
    :param gauss:
    :param sort: type of sorting for Rabi references "best" or "newest"
    :param dbg_storage_samples: if True then save samples in m.datasets['x'] and markers in m.datasets['y']
    :return:
    """
    print('\x1b[5;35;46m' + 'Start qutrit calibration!' + '\x1b[0m')

    # qubit_id = qubit_transitions_id['01'] # for transition 01
    internal_avg = False
    transitions_list = ['01', '12']

    # qubit_ids = [qubit_transitions_id[t] for t in list(qubit_transitions_id.keys())]
    # print(qubit_ids)

    if not post_selection_flag:
        adc, mnames = device.setup_adc_reducer_iq(qubit_id, raw=True,  internal_avg=internal_avg)
        adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
        adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    else:
        # in case of post selection use features and thresholds or model from previous readout calibration
        qubit_readout_pulse_, adc, _ = get_calibrated_measurer2(device, qubit_id, raw=True, internal_avg=internal_avg,
                                                                readouts_per_repetition=2, qutrit=True, M0_id=M0_id)


    nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
    old_nums = adc.get_adc_nums()


    if ignore_other_qubits is None:
        ignore_other_qubits = (device.get_qubit_constant(qubit_id=qubit_id, name='readout_calibration_ignore_other_qubits') == 'True')
    print ('ignore_other_qubits', ignore_other_qubits)

    other_qubit_pulse_sequence = []
    references = {}
    if not ignore_other_qubits:
        for other_qubit_id in device.get_qubit_list():
            if other_qubit_id not in [qubit_id]:
                # TODO
                '''Warning'''
                half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id,
                                                                           rotation_angle=np.pi / 2,gauss=gauss,
                                                                           sort = sort)
                references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
                other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0))
    # TODO
    '''Warning'''
    qubit_excitation_pulses = {}
    if gauss:
        for t in transitions_list:
            qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id,
                                                                           rotation_angle=np.pi / 2, transition=t,
                                                                           gauss=gauss,
                                                                           sort=sort)
            qubit_excitation_pulses[t] = qubit_excitation_pulse
    else:
        for t in transitions_list:
            qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id,
                                                                           rotation_angle=np.pi, transition=t,
                                                                           gauss=gauss,
                                                                           sort=sort)
            qubit_excitation_pulses[t] = qubit_excitation_pulse

    metadata = {'qubit_id': qubit_id,
                'averages': nums,
                'ignore_other_qubits': ignore_other_qubits}

    if post_selection_flag:
        metadata.update({'post_selection': str(True)})
    else:
        metadata.update({'post_selection': str(False)})

    references.update({'readout_pulse': qubit_readout_pulse.id,
                       'delay_calibration': device.modem.delay_measurement.id})

    for t in list(transitions_list):
        references.update({'excitation_pulse' + str(t): qubit_excitation_pulses[t].id})

    #TODO
    exitation_channels = {}
    for t in transitions_list:
        exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=t).keys()][0]
        exitation_channels[t] = exitation_channel

    ex_channels = {}
    for t in transitions_list:
        exitation_channel = exitation_channels[t]
        ex_channel = device.awg_channels[exitation_channel]
        ex_channels[t] = ex_channel

    awg_and_seq_id = {} # [awg, seq_id]
    for t in transitions_list:
        ex_channel = ex_channels[t]
        if ex_channel.is_iq():
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.parent.sequencer_id
        else:
            control_qubit_awg = ex_channel.parent.awg
            control_qubit_seq_id = ex_channel.channel // 2
        awg_and_seq_id[t] = [control_qubit_awg, control_qubit_seq_id]

    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    control_qubit_sequence = {}
    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[],
                                               post_selection_flag=post_selection_flag)
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True,
                                               post_selection_flag=post_selection_flag)
            control_sequence = ex_seq
        for t in transitions_list:
            control_qubit_awg, control_qubit_seq_id = awg_and_seq_id[t]
            if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
                control_qubit_sequence[t] = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        if pre_pulse_delay:
            ex_seq.set_prepulse_delay(pre_pulse_delay)
        ex_sequencers.append(ex_seq)

    excitations = {}
    if gauss:
        for t in transitions_list:
            excitation = qubit_excitation_pulses[t].get_pulse_sequence(0) + qubit_excitation_pulses[t].get_pulse_sequence(0)
            excitations[t] = excitation
    else:
        for t in transitions_list:
            excitation = qubit_excitation_pulses[t].get_pulse_sequence(0)
            excitations[t] = excitation

    qubit_pulse_sequence = []
    for t in transitions_list:
        qubit_pulse_sequence += excitations[t]

    prepare_seqs = {0: other_qubit_pulse_sequence,
                    1: other_qubit_pulse_sequence + excitations['01'],
                    2: other_qubit_pulse_sequence + excitations['01'] + excitations['12']}

    control_seq = {0: {0: [control_qubit_sequence['01'], control_qubit_sequence['12']], 1: []},
                   1: {0: [control_qubit_sequence['12']], 1: [control_qubit_sequence['01']]},
                   2: {0: [], 1: [control_qubit_sequence['01'], control_qubit_sequence['12']]}
                   }

    sequence_control.set_preparation_sequence(device, ex_sequencers, other_qubit_pulse_sequence +  qubit_pulse_sequence)

    readout_sequencer = sequence_control.define_readout_control_seq(device, qubit_readout_pulse, post_selection_flag=post_selection_flag)
    readout_sequencer.start()

    # nums = readout_fidelity_scan
    metadata.update({'logistic_regression_classifier': str(True)})
    classifier = single_shot_readout3.SingleShotReadout(device=device,
                                                         adc=adc,
                                                         prepare_seqs=prepare_seqs,
                                                         ex_seqs=ex_sequencers,
                                                         ro_seq=readout_sequencer,
                                                         control_seq = control_seq,
                                                         ro_delay_seq=None,
                                                         _readout_classifier=None,
                                                         adc_measurement_name='Voltage',
                                                         dbg_storage=dbg_storage, dbg_storage_samples=dbg_storage_samples,
                                                         post_selection_flag=post_selection_flag)

    classifier.readout_classifier.cov_mode = 'equal'

    try:
        if not post_selection_flag:
            adc.set_adc_nums(nums)
        measurement = device.sweeper.sweep(classifier,
                                           measurement_type='readout_qutrit_calibration',
                                           metadata=metadata,
                                           references=references)
    except:
        raise
    finally:
        if not post_selection_flag:
            adc.set_adc_nums(old_nums)

    return measurement
    # classifier.repeat_samples = 2


def get_qubit_readout_pulse_from_fidelity_scan(device, fidelity_scan):
    """
    Get qubit readout pulse from fidelity scan
    """
    from .readout_pulse2 import qubit_readout_pulse
    references = {'fidelity_scan': fidelity_scan.id}
    print('GET QUBIT READOUT PULSE FROM FIDELITY SCAN!')
    if 'channel_calibration' in fidelity_scan.metadata:
        references['channel_calibration'] = fidelity_scan.references['readout_channel_calibration']

    fidelity_dataset = fidelity_scan.datasets['fidelity']
    max_fidelity = np.unravel_index(np.argmax(fidelity_dataset.data.ravel()), fidelity_dataset.data.shape)
    pulse_parameters = {}
    for p, v_id in zip(fidelity_dataset.parameters, max_fidelity):
        pulse_parameters[p.name] = p.values[v_id]
    #   compression_1db = float(passthrough_measurement.metadata['compression_1db'])
    #   additional_noise_appears = float(passthrough_measurement.metadata['additional_noise_appears'])
    #   if np.isfinite(compression_1db):
    #       calibration_type = 'compression_1db'
    #       amplitude = compression_1db
    #   elif np.isfinite(additional_noise_appears):
    #       calibration_type = 'additional_noise_appears'
    #       amplitude = additional_noise_appears
    #   else:
    #       raise Exception('Compession_1db and additional_noise_appears not found on passthourgh scan!')
    readout_channel = fidelity_scan.metadata['channel']
    # length = float(fidelity_scan.metadata['length'])
    metadata = {'pulse_type': 'rect',
                'channel': readout_channel,
                'qubit_id': fidelity_scan.metadata['qubit_id'],
                # 'amplitude':amplitude,
                'calibration_type': 'fidelity_scan',
                # 'length': passthrough_measurement.metadata['length']
                }
    metadata.update(pulse_parameters)
    length = float(metadata['length'])
    amplitude = float(metadata['amplitude'])
    try:
        readout_pulse = qubit_readout_pulse(
            device.exdir_db.select_measurement(measurement_type='qubit_readout_pulse', references_that=references,
                                               metadata=metadata))
    except Exception as e:
        # print(type(e), str(e))
        readout_pulse = qubit_readout_pulse(references=references, metadata=metadata,
                                            sample_name=device.exdir_db.sample_name)
        device.exdir_db.save_measurement(readout_pulse)

    readout_pulse.definition_fragment, readout_pulse.play_fragment = device.pg.readout_rect(fidelity_scan.metadata['channel'], length, amplitude)

    #readout_pulse.pulse_sequence = [device.pg.p(readout_channel, length, device.pg.rect, amplitude)]
    return readout_pulse


def readout_fidelity_scan(device, qubit_id, readout_pulse_lengths, readout_pulse_amplitudes, readout_frequency_offsets=None,
                          recalibrate_excitation=True, ignore_other_qubits=False, channel_amplitudes=None,
                          transition='01', dbg_storage=False,
                          gauss=True, sort = 'best', ex_pre_pulse=None, ro_channel=None, set_sequence=None, set_control_qubit_sequence=None):

    if not ro_channel:
        adc, mnames = device.setup_adc_reducer_iq(qubit_id, raw=True)
    else:
        adc, mnames = device.setup_adc_reducer_iq(ro_channel, raw=True)
    nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
    adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    old_nums = adc.get_adc_nums()

    if not ro_channel:
        readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
    else:
        readout_channel = [i for i in device.get_qubit_readout_channel_list(ro_channel).keys()][0]

    other_qubit_pulse_sequence = []
    references = {'frequency_controls': device.get_frequency_control_measurement_id(qubit_id=qubit_id)}
    if hasattr(device.awg_channels[readout_channel], 'get_calibration_measurement'):
        references['channel_calibration'] = device.awg_channels[readout_channel].get_calibration_measurement()

    if not ignore_other_qubits:
        for other_qubit_id in device.get_qubit_list():
            if other_qubit_id != qubit_id:

                '''Warning'''
                #TODO
                half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id,
                                                                           rotation_angle=np.pi / 2.,
                                                                           recalibrate=recalibrate_excitation,
                                                                           gauss = gauss,sort = sort)
                references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
                other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0))

    '''Warning'''
    # TODO
    if gauss:
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi/2,
                                                                   channel_amplitudes_override=channel_amplitudes,
                                                                   transition=transition,
                                                                   recalibrate=recalibrate_excitation,
                                                                   gauss=gauss,sort = sort)
    else:
        qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi,
                                                                   channel_amplitudes_override=channel_amplitudes,
                                                                   transition=transition,
                                                                   recalibrate=recalibrate_excitation,sort = sort,
                                                                   gauss=gauss)
    if not ex_pre_pulse:
            metadata = {'qubit_id': qubit_id,
                    'averages': nums,
                    'channel': readout_channel,
                    'ignore_other_qubits': ignore_other_qubits}
    else:
        metadata = {'qubit_id': qubit_id,
                    'averages': nums,
                    'channel': readout_channel,
                    'ignore_other_qubits': ignore_other_qubits,
                    'ex_pre_pulse': ex_pre_pulse.id,
                    'ex_pre_pulse_length': ex_pre_pulse.metadata['length']}

    # print ('len(readout_pulse_lengths): ', len(readout_pulse_lengths))
    if len(readout_pulse_lengths) == 1:
        metadata['length'] = str(readout_pulse_lengths[0])

    references.update({'excitation_pulse': qubit_excitation_pulse.id,
                       'delay_calibration': device.modem.delay_measurement.id})

    #TODO
    exitation_channel = [i for i in device.get_qubit_excitation_channel_list(qubit_id, transition=transition).keys()][0]
    ex_channel = device.awg_channels[exitation_channel]
    if ex_channel.is_iq():
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.parent.sequencer_id
    else:
        control_qubit_awg = ex_channel.parent.awg
        control_qubit_seq_id = ex_channel.channel // 2
    control_awg, control_seq_id = device.pre_pulses.seq_in_use[0]
    ex_sequencers = []

    for awg, seq_id in device.pre_pulses.seq_in_use:
        if [awg, seq_id] != [control_awg, control_seq_id]:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[])
        else:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=awg,
                                               awg_amp=1, use_modulation=True, pre_pulses=[], control=True)
            control_sequence = ex_seq
        if [awg, seq_id] == [control_qubit_awg, control_qubit_seq_id]:
            control_qubit_sequence = ex_seq

        device.pre_pulses.set_seq_offsets(ex_seq)
        device.pre_pulses.set_seq_prepulses(ex_seq)
        ex_seq.start()
        ex_sequencers.append(ex_seq)

    if ex_pre_pulse:
        pre_excitation = ex_pre_pulse.get_pulse_sequence(0)
    else:
        pre_excitation = []

    if gauss:
        excitation = qubit_excitation_pulse.get_pulse_sequence(0) + qubit_excitation_pulse.get_pulse_sequence(0)
    else:
        excitation = qubit_excitation_pulse.get_pulse_sequence(0)

    if set_sequence:
        excitation = set_sequence

    # TODO: not optimal
    if set_control_qubit_sequence:
        control_qubit_sequence = []
        for ex_seq in ex_sequencers:
            control_qubit_sequence.append(ex_seq)

    sequence_control.set_preparation_sequence(device, ex_sequencers, other_qubit_pulse_sequence + pre_excitation +
                                                                        excitation)
                                                                       # qubit_excitation_pulse.get_pulse_sequence(0)+
                                                                       # qubit_excitation_pulse.get_pulse_sequence(0))


    '''Warning'''
    #readout_sequencer = sequence_control.define_readout_control_seq(device, readout_channel)
    #raise ValueError('fallos')
    re_channel = device.awg_channels[readout_channel]
    readout_sequencer = zi_scripts.READSequence(device,re_channel.parent.sequencer_id, device.modem.awg)

    def_frag, play_frag = device.pg.readout_rect(channel=readout_channel, length=readout_pulse_lengths[0],
                                                 amplitude=readout_pulse_amplitudes[0])
    readout_sequencer.add_readout_pulse(def_frag, play_frag)
    readout_sequencer.stop()
    device.modem.awg.set_sequence(readout_sequencer.params['sequencer_id'], readout_sequencer)
    readout_sequencer.set_delay(device.modem.trigger_channel.delay)
    readout_sequencer.start()

    classifier = single_shot_readout.single_shot_readout(device=device,
                                                         adc=adc,
                                                         prepare_seqs=[other_qubit_pulse_sequence + pre_excitation,
                                                                       other_qubit_pulse_sequence + pre_excitation +
                                                                       excitation],
                                                                       # qubit_excitation_pulse.get_pulse_sequence(0)+
                                                                       # qubit_excitation_pulse.get_pulse_sequence(0)],
                                                         ex_seqs=ex_sequencers,
                                                         ro_seq=readout_sequencer,
                                                         control_seq=control_qubit_sequence,
                                                         #pulse_generator=device.pg,
                                                         ro_delay_seq=None,
                                                         _readout_classifier=readout_classifier.binary_linear_classifier(),
                                                         adc_measurement_name='Voltage',
                                                         dbg_storage=dbg_storage)

    classifier.readout_classifier.cov_mode = 'equal'

    # setters for sweep
    readout_amplitude = 0
    readout_length = 0

    class ParameterSetter:
        def __init__(self):
            self.readout_amplitude = 0
            self.readout_length = 0
            self.channel = readout_channel

        def set_readout_amplitude(self, x):
            #nonlocal readout_amplitude
            self.readout_amplitude = x
            #classifier.ro_seq = device.trigger_readout_seq + [
            #    device.pg.p(readout_channel, readout_length, device.pg.rect, readout_amplitude)]
            classifier.ro_seq.set_awg_amp(x)

        def set_readout_length(self, x):
            #nonlocal readout_length
            self.readout_length = x
            #classifier.ro_seq = device.trigger_readout_seq + [
            #    device.pg.p(readout_channel, readout_length, device.pg.rect, readout_amplitude)]
            classifier.ro_seq.awg.stop_seq(classifier.ro_seq.params['sequencer_id'])
            def_frag, play_frag = device.pg.readout_rect(channel=self.channel,
                                                        length=self.readout_length,
                                                        amplitude=self.readout_amplitude)
            classifier.ro_seq.clear_readout_pulse()
            classifier.ro_seq.add_readout_pulse(def_frag, play_frag)
            device.modem.awg.set_sequence(classifier.ro_seq.params['sequencer_id'], classifier.ro_seq)
            classifier.ro_seq.awg.start_seq(classifier.ro_seq.params['sequencer_id'])

    setter = ParameterSetter()

    sweep_params = [(readout_pulse_lengths, setter.set_readout_length, 'length', 's'),
                    (readout_pulse_amplitudes, setter.set_readout_amplitude, 'amplitude', '')]

    lo_frequency = device.awg_channels[readout_channel].parent.lo.get_frequency()
    ro_frequency = device.awg_channels[readout_channel].get_frequency()

    if readout_frequency_offsets is not None:
        frequency_setter = device.awg_channels[readout_channel].set_uncal_frequency
        sweep_params.append((readout_frequency_offsets + ro_frequency, frequency_setter, 'frequency', 'Hz'))
    try:
        adc.set_adc_nums(nums)
        measurement = device.sweeper.sweep(classifier,
                                           *sweep_params,
                                           measurement_type='readout_fidelity_scan',
                                           metadata=metadata,
                                           references=references)
    except:
        raise
    finally:
        adc.set_adc_nums(old_nums)
        device.awg_channels[readout_channel].parent.lo.set_frequency(lo_frequency)

    return measurement
    # classifier.repeat_samples = 2



def readout_Zgate_scan(device, qubit_id, qubit_readout_pulse, Zgate, amplitudes,  transition='01', ignore_other_qubits=None):
    adc, mnames = device.setup_adc_reducer_iq(qubit_id, raw=True,gauss=True,sort = 'best')
    nums = int(device.get_qubit_constant(qubit_id=qubit_id, name='readout_background_nums'))
    old_nums = adc.get_adc_nums()
    adc.set_adc_nop(int(device.get_sample_global('readout_adc_points')))
    if ignore_other_qubits is None:
        ignore_other_qubits = (device.get_qubit_constant(qubit_id=qubit_id, name='readout_calibration_ignore_other_qubits') == 'True')
    print ('ignore_other_qubits', ignore_other_qubits)

    other_qubit_pulse_sequence = []
    references = {}
    if not ignore_other_qubits:
        for other_qubit_id in device.get_qubit_list():
            if other_qubit_id != qubit_id:
                half_excited_pulse = excitation_pulse.get_excitation_pulse(device, other_qubit_id,
                                                                           rotation_angle=np.pi / 2.,gauss=gauss,
                                                                           sort = sort)
                references[('other_qubit_pulse', other_qubit_id)] = half_excited_pulse.id
                other_qubit_pulse_sequence.extend(half_excited_pulse.get_pulse_sequence(0)[0])

    qubit_excitation_pulse = excitation_pulse.get_excitation_pulse(device, qubit_id, rotation_angle=np.pi,gauss=gauss,
                                                                   sort = sort)
    metadata = {'qubit_id': qubit_id,
                'transition': transition,
                'averages': nums,
                'ignore_other_qubits': ignore_other_qubits}

    references.update({'readout_pulse': qubit_readout_pulse.id,
                       'excitation_pulse': qubit_excitation_pulse.id,
                       'delay_calibration': device.modem.delay_measurement.id,
                      'Zgate': Zgate.id},
                      )

    classifier = single_shot_readout.single_shot_readout(adc=adc,
                                                         prepare_seqs=[
                                                             device.pre_pulses + other_qubit_pulse_sequence,
                                                             device.pre_pulses + other_qubit_pulse_sequence + qubit_excitation_pulse.get_pulse_sequence(0)],
                                                         ro_seq=device.trigger_readout_seq + qubit_readout_pulse.get_pulse_sequence(),
                                                         control_seq=None,
                                                         # pulse_generator=device.pg,
                                                         ro_delay_seq=None,
                                                         _readout_classifier=readout_classifier.binary_linear_classifier(),
                                                         adc_measurement_name='Voltage')

    def set_Zgate_amplitude(x):
        channel_amplitudes1_ = channel_amplitudes.channel_amplitudes(device,
                                                                     **{Zgate.metadata[
                                                                            'carrier_name']: x})
        print(channel_amplitudes1_)

        pulse_seq1 = excitation_pulse.get_rect_cos_pulse_sequence(device=device,
                                                                  channel_amplitudes=channel_amplitudes1_,
                                                                  tail_length=float(Zgate.metadata['tail_length']),
                                                                  length=float(qubit_readout_pulse.metadata['length']),
                                                                  phase=0.0)

        classifier.ro_seq =  device.trigger_readout_seq + device.pg.parallel(pulse_seq1,qubit_readout_pulse.get_pulse_sequence())

    classifier.readout_classifier.cov_mode = 'equal'

    try:
        adc.set_adc_nums(nums)
        measurement = device.sweeper.sweep(classifier,
                                           (amplitudes, set_Zgate_amplitude, 'amplitude', 'Voltage'),
                                           measurement_type='readout_Zgate_scan',
                                           metadata=metadata,
                                           references=references)
    except:
        raise
    finally:
        adc.set_adc_nums(old_nums)

    return measurement
    # classifier.repeat_samples = 2