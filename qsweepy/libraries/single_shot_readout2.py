from qsweepy.libraries import data_reduce
import numpy as np
from qsweepy.libraries import readout_classifier
from qsweepy.qubit_calibrations import sequence_control
import pickle as pkl
from time import gmtime, strftime

import datetime
import time


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
                 adc_measurement_name='Voltage', dbg_storage=False,  dbg_storage_samples=False,
                 post_selection_flag=False):
        self.adc = adc
        self.num_desc = self.adc.adc.num_covariances
        self.device = device
        self.ro_seq = ro_seq
        self.ex_seqs = ex_seqs
        self.prepare_seqs = prepare_seqs

        self.ro_delay_seq = ro_delay_seq
        #self.pulse_generator = pulse_generator
        self.control_seq = control_seq
        self.repeat_samples = 5 #5
        self.save_last_samples = False
        self.train_test_split = 0.8
        self.measurement_name = ''
        # self.dump_measured_samples = False

        self.measure_avg_samples = True
        self.dbg_storage = dbg_storage
        self.dbg_storage_samples = dbg_storage_samples
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

        self.post_selection_flag = post_selection_flag
        if self.post_selection_flag:
            self.readouts_per_repetition = 2
        else:
            self.readouts_per_repetition = 1

        # readout shots for one repetition
        self.nums= self.readouts_per_repetition * int(
            device.get_sample_global(name='calibrated_readout_nums'))

    def calibrate(self):
        X = []
        y = []

        if not self.post_selection_flag:
            for i in range(self.repeat_samples):
                for class_id, prepare_seq in enumerate(self.prepare_seqs):
                    # pulse sequence to prepare state
                    self.adc.set_internal_avg(True)

                    '''Warning'''
                    # self.ro_seq.awg.stop_seq(self.ro_seq.params['sequencer_id'])
                    # self.pulse_generator.set_seq(prepare_seq + self.ro_seq)
                    # sequence_control.set_preparation_sequence(self.device, self.ex_seqs, prepare_seq, self.control_seq)
                    if type(self.control_seq) == list:
                        for control_seq in self.control_seq:
                            control_seq.set_awg_amp(float(class_id ))
                    else:
                        self.control_seq.set_awg_amp(float(class_id ))

                    # for ex_seq in self.ex_seqs:
                    # if ex_seq.params['sequencer_id'] == self.control_seq.params['sequencer_id']:
                    # ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
                    # ex_seq.clear_pulse_sequence()
                    # for prep_seq in prepare_seq:
                    # for seq_id, single_sequence in prep_seq[0].items():
                    # if seq_id == ex_seq.params['sequencer_id']:
                    # ex_seq.add_definition_fragment(single_sequence[0])
                    # ex_seq.add_play_fragment(single_sequence[1])
                    # self.device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
                    # ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])

                    # self.ro_seq.awg.start_seq(self.ro_seq.params['sequencer_id'])
                    # re_sequence = sequence_control.define_readout_control_seq(device, qubit_readout_pulse)
                    # re_sequence.start()
                    # time.sleep(2)
                    if self.adc.devtype == 'SK':
                        measurement = self.adc.measure()
                        X.append(measurement[self.adc_measurement_name])
                        if len(self.adc.get_points()[self.adc_measurement_name]) > 1:
                            y.extend([class_id] * len(self.adc.get_points()[self.adc_measurement_name][0][1]))
                        else:
                            y.extend([class_id])

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

        else:
            states_meas0 = []  # first measurement (for post selection procedure)
            # states_meas1 = []  # second measurement

            # self.adc.set_adc_nums(self.nums)

            for i in range(self.repeat_samples):
                for class_id, prepare_seq in enumerate(self.prepare_seqs):
                    # pulse sequence to prepare state

                    # TODO: somehow add it and averaging=False
                    self.adc.dot_prods = True

                    if type(self.control_seq) == list:
                        for control_seq in self.control_seq:
                            control_seq.set_awg_amp(float(class_id))
                    else:
                        self.control_seq.set_awg_amp(float(class_id))

                    if self.adc.devtype == 'SK':
                        measurement = self.adc.measure()
                        samples = measurement[self.adc_measurement_name]
                        # print(samples.shape)

                        dot_products = measurement['disc_ch0']

                        # list of threshold for all discrimination channels, we use only disc_ch0
                        threshold = self.adc.adc.threshold
                        states = np.asarray(dot_products > threshold[0], dtype=int)


                        # dot_products = np.zeros((self.nums, self.num_desc))
                        # for disc_ch in range(self.num_desc):
                        #     dot_products[:, disc_ch] = measurement['disc_ch' + str(disc_ch)]
                        # threshold = self.adc.adc.threshold
                        # states = np.asarray(dot_products > threshold, dtype=int)

                        states_meas0.append(states[::2])
                        # states_meas1.append(states[1::2])

                        # print(self.adc.adc.threshold)
                        # print(states_meas0, states_meas1)

                        X.append(measurement[self.adc_measurement_name][1::2, :]) # append samples only for meas1
                        y.extend([class_id] * (len(self.adc.get_points()[self.adc_measurement_name][0][1]) // 2))

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

        print (np.asarray(X).shape, np.asarray(y).shape)
        X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
        y = np.asarray(y)
        print(X.shape, y.shape)


        if self.post_selection_flag:
            print(np.asarray(states_meas0).shape)
            states_meas0 = np.asarray(states_meas0).ravel()
            print(states_meas0.shape)
            self.states_meas0 = states_meas0
            self.x = X
            self.y = y

        if self.dbg_storage:
            print("Save projections!")
            self.x = X
            self.y = y

        if self.dbg_storage_samples:
            print('ВНИМАНИЕ, СОХРАНЯЕТСЯ МНОГО ДАННЫХ!!! ОТКЛЮЧИ dbg_storage_samples!!!')
            self.x = X
            self.y = y
        self.readout_classifier.fit(X, y)

        if len(self.adc.get_points()[self.adc_measurement_name]) > 1:
            scores = readout_classifier.evaluate_classifier(self.readout_classifier, X, y)
            self.scores = scores
            self.confusion_matrix = readout_classifier.confusion_matrix(y, self.readout_classifier.predict(X))
        elif len(self.adc.get_points()[self.adc_measurement_name]) == 1:
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

                    if type(self.control_seq) == list:
                        for control_seq in self.control_seq:
                            control_seq.set_awg_amp(float(class_id))
                    else:
                        self.control_seq.set_awg_amp(float(class_id ))

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
                    if self.adc.devtype == 'UHF':
                        j = self.adc.measure()[self.adc.result_source + str(0)]
                        x.extend((np.real(j) + np.imag(j)).tolist())
                        y.extend([class_id] * len(j))
                    else:
                        j = self.adc.measure()['all_cov0']
                        x.extend(j.tolist())
                        y.extend([class_id] * len(j))

                    #x.extend((np.real(j)).tolist())


            self.readout_classifier.naive_bayes_reduced(x, y)
            self.scores = self.readout_classifier.scores
            self.confusion_matrix = readout_classifier.confusion_matrix(np.asarray(y),
                                                                        np.asarray(self.readout_classifier.predict_reduced(x)))
            # self.x = x
            # self.y = y
            # x = (np.real(x)+np.imag(x)).tolist()

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
        # if self.adc.devtype == 'UHF' and self.return_scores:
        if self.dbg_storage:
            # opts['x'] = {'log': False}
            # opts['y'] = {'log': False}
            opts['x0'] = {'log': False}
            opts['x1'] = {'log': False}
        if self.dbg_storage_samples:
            opts['x'] = {'log': False}
            opts['y'] = {'log': False}

        if self.post_selection_flag:
            opts['meas0']  = {'log': False}
            opts['y'] = {'log': False}
            opts['w'] = {'log': False}
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

        if self.dbg_storage:
        # if self.adc.devtype == 'UHF' and self.return_scores:
        #     meas['x'], meas['y'] = self.count_clouds(self.x, self.y)
        #     W, marker = self.count_clouds(self.x, self.y)
        #     meas['x0'] = W[marker == 0, :]
        #     meas['x1'] = W[marker == 1, :]
            x0, x1 = self.calculate_projections(self.x, self.y)
            meas['x0'] = x0
            meas['x1'] = x1
        if self.dbg_storage_samples:
            meas['x'], meas['y'] = self.x, self.y

        if self.post_selection_flag:
            meas['meas0']  = self.states_meas0
            meas['y'] = self.y

            w = self.calculate_projections2(self.x, self.y)
            meas['w'] = w
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
        points['fidelities'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs) // self.readouts_per_repetition), '')]
        points['thresholds'] = [('bin', np.arange(self.adc.get_adc_nums()*self.repeat_samples*len(self.prepare_seqs) // self.readouts_per_repetition), '')]
        if self.measure_feature_w_threshold:
            points['feature'] = [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            points['threshold'] = []
        if self.dbg_storage:
        # if self.adc.devtype == 'UHF' and self.return_scores:
            adc_points = self.adc.get_points()[self.adc_measurement_name]
        #     adc_points = self.adc.get_points()[self.adc_measurement_name][0][1]
            if len(adc_points) < 2:
                raise IndexError('dbg_storage not available with on-digitizer averaging')
            num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1])*self.repeat_samples*len(self.prepare_seqs) // self.readouts_per_repetition
            # points['x'] = [('Segment', np.arange(num_shots), ''),
            #                ('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            # points['x'] = [('Segment', np.arange(num_shots), ''),
            #                  ('Segment', np.arange(2), '')]
            # points['y'] = [('Segment', np.arange(num_shots), '')]

            # points['x0'] = [('Segment', np.arange(num_shots // 2), ''),
            #                 ('Segment', np.arange(2), '')]
            # points['x1'] = [('Segment', np.arange(num_shots // 2), ''),
            #                 ('Segment', np.arange(2), '')]

            points['x0'] = [('Segment', np.arange(num_shots // 4), ''),
                            ('Segment', np.arange(2), '')]
            points['x1'] = [('Segment', np.arange(num_shots // 4), ''),
                            ('Segment', np.arange(2), '')]

        if self.dbg_storage_samples:
            # if self.adc.devtype == 'UHF' and self.return_scores:
            adc_points = self.adc.get_points()[self.adc_measurement_name]
            #     adc_points = self.adc.get_points()[self.adc_measurement_name][0][1]
            if len(adc_points) < 2:
                raise IndexError('dbg_storage not available with on-digitizer averaging')
            num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1]) * self.repeat_samples * len(
                self.prepare_seqs)
            points['x'] = [('Segment', np.arange(num_shots), ''),
                           ('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            points['y'] = [('Segment', np.arange(num_shots), '')]

        if self.post_selection_flag:
            nums_  = self.nums // 2 * self.repeat_samples * len(self.prepare_seqs)
            points['meas0']  = [('bin', np.arange(nums_), '')]

            num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1]) * self.repeat_samples * len(
                self.prepare_seqs) // self.readouts_per_repetition
            points['y'] = [('Segment', np.arange(num_shots), '')]

            points['w'] = [('Segment', np.arange(num_shots), ''),
                           ('axe', np.arange(2), '')]

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
            #dtypes['sigma0'] = complex
            #dtypes['sigma1'] = complex
        # if self.measure_hists:
        #    dtypes['hists'] = float
        #    dtypes['proba_points'] = float
        dtypes['fidelities'] = float
        dtypes['thresholds'] = float
        if self.measure_feature_w_threshold:
            dtypes['feature'] = complex
            dtypes['threshold'] = float

        if self.dbg_storage:
        # if self.adc.devtype == 'UHF' and self.return_scores:
        #     dtypes['x'] = complex
        #     dtypes['x'] = float
        #     dtypes['y'] = float
            dtypes['x0'] = float
            dtypes['x1'] = float

        if self.dbg_storage_samples:
        # if self.adc.devtype == 'UHF' and self.return_scores:
            dtypes['x'] = complex
            dtypes['y'] = float

        if self.post_selection_flag:
            dtypes['meas0'] = float

            dtypes['y'] = float

            dtypes['w'] = float

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

    def calculate_projections(self, X, y):
        """
        Calculate projections of samples
        """
        from qsweepy.libraries import logistic_regression_classifier2 as logistic_regression_classifier
        clf = logistic_regression_classifier.LogisticRegressionReadoutClassifier(nums_adc=self.adc.get_adc_nop(),
                                                                                 states=2, num_shots=len(y))
        clf.fit(X, y)

        w_ = clf.get_w_(trajectories=X)

        w, marker = clf.get_w_and_markers()
        x0 = w[: len(marker) // 2]
        x1 = w[len(marker) // 2 :]

        # x0 = np.vstack((clf.projections0[0], clf.projections1[0])).T
        # x1 = np.vstack((clf.projections0[1], clf.projections1[1])).T
        return x0, x1

    def calculate_projections2(self, X, y):
        """
        Calculate projections of samples
        """
        from qsweepy.libraries import logistic_regression_classifier2 as logistic_regression_classifier
        clf = logistic_regression_classifier.LogisticRegressionReadoutClassifier(nums_adc=self.adc.get_adc_nop(),
                                                                                 states=2, num_shots=len(y))
        clf.fit(X, y)
        w_ = clf.get_w_(trajectories=X)
        return w_


    # def Function(self, x0, x1):
    #     """
    #     Feature function
    #     """
    #     N = np.shape(x0)[1]
    #     n = np.shape(x0)[0]
    #     F = np.zeros(N)
    #     F = np.mean(x1[:n], axis=0) - np.mean(x0[:n], axis=0)
    #
    #     return (F)

    # def count_projection(self, F, x):
    #     return (np.sum(x * F))
    #
    # def make_statistic_of_projects(self, F, x):
    #     n = x.shape[0]
    #     array_pr = np.zeros(n)
    #     for i in range(n):
    #         array_pr[i] = self.count_projection(F, x[i])
    #
    #     return (array_pr)
    #
    # def count_clouds(self, x, y):
    #
    #     num_samples = np.shape(x)[0]
    #     l = np.shape(x)[1]
    #     ind_0 = [i for i in range(num_samples) if y[i] == 0]
    #     x0 = x[ind_0]
    #     ind_1 = [i for i in range(num_samples) if y[i] == 1]
    #     x1 = x[ind_1]
    #
    #     # x0_train, x0_test = train_test_split(x0, test_size=0.5,
    #     #                                      train_size=0.5, random_state=42)
    #     # x1_train, x1_test = train_test_split(x1, test_size=0.5,
    #     #                                      train_size=0.5, random_state=42)
    #
    #     x0_train, x0_test = x0, x0
    #     x1_train, x1_test = x1, x1
    #
    #     a0 = np.real(x0_train)
    #     a1 = np.real(x1_train)
    #     a0_wave = np.real(x0_test)
    #     a1_wave = np.real(x1_test)
    #     b0 = np.imag(x0_train)
    #     b1 = np.imag(x1_train)
    #     b0_wave = np.imag(x0_test)
    #     b1_wave = np.imag(x1_test)
    #
    #     Fa = self.Function(a0, a1)
    #     Fb = self.Function(b0, b1)
    #
    #     t0a = self.make_statistic_of_projects(Fa, a0_wave)
    #     t1a = self.make_statistic_of_projects(Fa, a1_wave)
    #     t0b = self.make_statistic_of_projects(Fb, b0_wave)
    #     t1b = self.make_statistic_of_projects(Fb, b1_wave)
    #
    #     la = len(t0a)
    #     l = len(t0a) + len(t1a)
    #     W = np.zeros((l, 2))
    #     W[:la, 0] = t0a
    #     W[:la, 1] = t0b
    #     W[la:, 0] = t1a
    #     W[la:, 1] = t1b
    #     marker = np.zeros(l)
    #     marker[:la] = 0
    #     marker[la:] = 1
    #
    #     return W, marker
