from qsweepy.libraries import data_reduce
import numpy as np
from qsweepy.libraries import readout_classifier
from qsweepy.qubit_calibrations import sequence_control


class single_shot_readout:
    """
    Single shot readout class

    Args:
        adc (Instrument): a device that measures a complex vector for each readout trigger (an ADC)
        prepare_seqs (list of pulses.sequence): a dict of sequences of control pulses. The keys are use for state identification.
        ro_seq (pulses.sequence): a sequence of control pulses that is used to generate the reaout pulse of the DAC.
        pulse_generator (pulses.pulse_generator): pulse generator used to concatenate and set waveform sequences on the DAC.
        ro_delay_seq (pulses.sequence): Sequence used to align the DAC and ADC (readout delay compensation)
        adc_measurement_name (str): name of measurement on ADC
    """

    def __init__(self, device, adc, prepare_seqs, ex_seqs, ro_seq, control_seq, ro_delay_seq=None, _readout_classifier=None,
                 adc_measurement_name='Voltage', dbg_storage=False):
        self.adc = adc
        self.device = device
        self.ro_seq = ro_seq
        self.ex_seqs = ex_seqs
        self.prepare_seqs = prepare_seqs

        self.ro_delay_seq = ro_delay_seq
        #self.pulse_generator = pulse_generator
        self.control_seq = control_seq
        self.repeat_samples = 3
        self.save_last_samples = False
        self.train_test_split = 0.8
        self.measurement_name = ''
        # self.dump_measured_samples = False

        self.measure_avg_samples = True
        self.dbg_storage = dbg_storage
        if self.dbg_storage:
            import pickle as pkl
            from time import gmtime, strftime
        # self.measure_cov_samples = False
        self.measure_hists = True
        self.measure_feature_w_threshold = True
        self.return_scores = False
        # self.measure_features = True

        # self.cutoff_start = 0
        if not _readout_classifier:
            self.readout_classifier = readout_classifier.linear_classifier()
        else:
            self.readout_classifier = _readout_classifier
        self.adc_measurement_name = adc_measurement_name

        self.filter_binary = dict(get_points=lambda: (self.adc.get_points()[adc_measurement_name][0],),
                                  get_dtype=lambda: int, get_opts=lambda: {}, filter=self.filter_binary_func)

    def calibrate(self):
        X = []
        y = []
        for i in range(self.repeat_samples):
            for class_id, prepare_seq in enumerate(self.prepare_seqs):
                # pulse sequence to prepare state
                self.adc.set_internal_avg(True)
                '''Warning'''
                #self.ro_seq.awg.stop_seq(self.ro_seq.params['sequencer_id'])
                # self.pulse_generator.set_seq(prepare_seq + self.ro_seq)
                #sequence_control.set_preparation_sequence(self.device, self.ex_seqs, prepare_seq, self.control_seq)
                self.control_seq.set_awg_amp(float(class_id))

                #for ex_seq in self.ex_seqs:
                    #if ex_seq.params['sequencer_id'] == self.control_seq.params['sequencer_id']:
                        #ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
                        #ex_seq.clear_pulse_sequence()
                        #for prep_seq in prepare_seq:
                            #for seq_id, single_sequence in prep_seq[0].items():
                                #if seq_id == ex_seq.params['sequencer_id']:
                                    #ex_seq.add_definition_fragment(single_sequence[0])
                                    #ex_seq.add_play_fragment(single_sequence[1])
                        #self.device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
                        #ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])

                #self.ro_seq.awg.start_seq(self.ro_seq.params['sequencer_id'])
                #re_sequence = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)
                #re_sequence.start()

                if self.adc.devtype == 'SK':
                    measurement = self.adc.measure()
                    X.append(measurement[self.adc_measurement_name])
                    y.extend([class_id] * len(self.adc.get_points()[self.adc_measurement_name][0][1]))
                elif self.adc.devtype == 'UHF':
                    # UHF scenario
                    nums = self.adc.get_nums()
                    self.adc.set_nums(1024)
                    repeats = nums // 1024
                    for inner_repeat_id in range(repeats):
                        measurement = self.adc.measure()
                        X.append(measurement[self.adc_measurement_name])
                        y.append(class_id)
                    self.adc.set_nums(nums)

        # print (np.asarray(X).shape, np.asarray(y).shape)
        X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
        y = np.asarray(y)

        if self.dbg_storage:
            abs_path = "'D:\\qtlab_replacement\\dbg_data"  # TODO make dependent path
            with open(abs_path + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '.pkl', 'wb') as f: pkl.dump((X, y), f)

        # last dimension is the feature dimension
        # y = np.asarray(y)
        # print(np.asarray(X).shape, np.asarray(y).shape)
        # print (X, y)
        self.readout_classifier.fit(X, y)

        if self.adc.devtype == 'SK':
            scores = readout_classifier.evaluate_classifier(self.readout_classifier, X, y)
            self.scores = scores
            self.confusion_matrix = readout_classifier.confusion_matrix(y, self.readout_classifier.predict(X))
        elif self.adc.devtype == 'UHF':
            # UHF scenario
            #x0 = np.mean(X[y == 0, :], axis=0)
            #x1 = np.mean(X[y == 1, :], axis=0)
            #feature = x1 - x0
            #feature = feature - np.mean(feature)
            threshold = 0
            #self.readout_classifier.feature = feature
            self.adc.set_internal_avg(False)
            self.readout_classifier.feature = self.readout_classifier.feature/np.max(np.abs(self.readout_classifier.feature))
            self.adc.set_feature_iq(feature_id=0, feature=self.readout_classifier.feature)
            x = []
            y = []
            for i in range(self.repeat_samples):
                for class_id, prepare_seq in enumerate(self.prepare_seqs):
                    '''Warning'''
                    #self.ro_seq.awg.stop_seq(self.ro_seq.params['sequencer_id'])
                    #self.pulse_generator.set_seq(prepare_seq + self.ro_seq)

                    #sequence_control.set_preparation_sequence(self.device, self.ex_seqs, prepare_seq, self.control_seq)
                    self.control_seq.set_awg_amp(float(class_id))
                    #for ex_seq in self.ex_seqs:
                    #    if ex_seq.params['sequencer_id'] == self.control_seq.params['sequencer_id']:
                    #        ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
                    #        ex_seq.clear_pulse_sequence()
                    #        for prep_seq in prepare_seq:
                    #            for seq_id, single_sequence in prep_seq.items():
                    #                if seq_id == ex_seq.params['sequencer_id']:
                    #                    ex_seq.add_definition_fragment(single_sequence[0])
                    #                    ex_seq.add_play_fragment(single_sequence[1])
                    #        self.device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
                    #        ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])

                    #self.ro_seq.awg.start_seq(self.ro_seq.params['sequencer_id'])
                    #re_sequence = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)
                    #re_sequence.start()
                    # TODO do we need to calibrate for all discriminators?
                    j = self.adc.measure()[self.adc.result_source + str(0)]
                    x.extend((np.real(j)+np.imag(j)).tolist())
                    #x.extend((np.real(j)).tolist())
                    y.extend([class_id] * len(j))

            self.readout_classifier.naive_bayes_reduced(x, y)
            self.scores = self.readout_classifier.scores
            self.confusion_matrix = readout_classifier.confusion_matrix(np.asarray(y),
                                                                        np.asarray(self.readout_classifier.predict_reduced(x)))
            self.x = x
            self.y = y
            x = (np.real(x)+np.imag(x)).tolist()

    def get_opts(self):
        opts = {}
        scores = {score_name: {'log': False} for score_name in readout_classifier.readout_classifier_scores}
        opts.update(scores)

        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
            # features = {'feature'+str(_class):{'log':False} for _class in self.readout_classifier.class_list}
            #opts['sigma0'] = {'log': False}
            #opts['sigma1'] = {'log': False}
            opts.update(avg_samples)

        opts['fidelities'] = {'log': False}
        opts['thresholds'] = {'log': False}
        if self.measure_feature_w_threshold:
            opts['feature'] = {'log': False}
            opts['threshold'] = {'log': False}
        if self.adc.devtype == 'UHF' and self.return_scores:
            opts['x'] = {'log': False}
            opts['y'] = {'log': False}


        return opts

    def measure(self):
        self.calibrate()
        meas = {}
        # if self.dump_measured_samples:
        # self.dump_samples(name=self.measurement_name)
        meas.update(self.scores)
        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): self.readout_classifier.class_averages[_class] for _class in
                           self.readout_classifier.class_list}
            # features = {'feature'+str(_class):self.readout_classifier.class_features[_class] for _class in self.readout_classifier.class_list}
            #meas['sigma0'] = self.readout_classifier.sigma0
            #meas['sigma1'] = self.readout_classifier.sigma1
            meas.update(avg_samples)

        meas['fidelities'] = self.readout_classifier.fidelities
        meas['thresholds'] = self.readout_classifier.thresholds
        if self.measure_feature_w_threshold:
            meas['feature'] = self.readout_classifier.feature
            meas['threshold'] = self.readout_classifier.threshold

        if self.adc.devtype == 'UHF' and self.return_scores:
            meas['x'] = self.x
            meas['y'] = self.y

        return meas

    def get_points(self):
        points = {}
        scores = {score_name: [] for score_name in readout_classifier.readout_classifier_scores}
        points.update(scores)

        if self.measure_avg_samples:
            avg_samples = {
                'avg_sample' + str(_class): [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')] for
                _class in self.readout_classifier.class_list}
            # features = {'feature'+str(_class):[('Time',np.arange(self.adc.get_nop())/self.adc.get_clock(), 's')] for _class in self.readout_classifier.class_list}
            points.update(avg_samples)
            #points['sigma0'] = [('Time1', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's'),
            #                    ('Time2', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            #points['sigma1'] =  [('Time1', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's'),
            #                    ('Time2', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
        # points.update(features)
        # if self.measure_hists:
        #    points['hists'] = [('class', self.readout_classifier.class_list, ''),
        #                       ('bin', np.arange(self.readout_classifier.nbins), '')]
        #    points['proba_points'] = [('bin', np.arange(self.readout_classifier.nbins), '')]
        points['fidelities'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs)), '')]
        points['thresholds'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs)), '')]
        if self.measure_feature_w_threshold:
            points['feature'] = [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            points['threshold'] = []
        if self.adc.devtype == 'UHF' and self.return_scores:
            points['x'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs)), '')]
            points['y'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs)), '')]

        return points

    def get_dtype(self):
        dtypes = {}
        scores = {score_name: float for score_name in readout_classifier.readout_classifier_scores}
        dtypes.update(scores)

        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
                           self.readout_classifier.class_list}
            features = {'feature' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
                        self.readout_classifier.class_list}
            dtypes.update(avg_samples)
            dtypes.update(features)
            #dtypes['sigma0'] = np.complex
            #dtypes['sigma1'] = np.complex
        # if self.measure_hists:
        #    dtypes['hists'] = float
        #    dtypes['proba_points'] = float
        dtypes['fidelities'] = float
        dtypes['thresholds'] = float
        if self.measure_feature_w_threshold:
            dtypes['feature'] = np.complex
            dtypes['threshold'] = float

        if self.adc.devtype == 'UHF' and self.return_scores:
            dtypes['x'] = np.complex
            dtypes['y'] = float

        return dtypes

    # def dump_samples(self, name):
    # from .save_pkl import save_pkl
    # header = {'type':'Readout classification X', 'name':name}
    # measurement = {'Readout classification X':(['Sample ID', 'time'],
    # [np.arange(self.calib_X.shape[0]), np.arange(self.calib_X.shape[1])/self.adc.get_clock()],
    # self.calib_X),
    # 'Readout classification y':(['Sample ID'],
    # [np.arange(self.calib_X.shape[0])],
    # self.calib_y)}
    # save_pkl(header, measurement, plot=False)

    def filter_binary_func(self, x):
        return self.readout_classifier.predict(x[self.adc_measurement_name])
