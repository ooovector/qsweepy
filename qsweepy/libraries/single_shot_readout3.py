from qsweepy.libraries import data_reduce
import numpy as np
from qsweepy.qubit_calibrations import sequence_control
# from qsweepy.libraries import logistic_regression_classifier
from qsweepy.libraries import logistic_regression_classifier2 as logistic_regression_classifier
import pickle as pkl
from time import gmtime, strftime
import datetime

class SingleShotReadout:
    """
    Single shot readout class for qutrits systems
    """

    def __init__(self, device, adc, prepare_seqs, ex_seqs, ro_seq, control_seq, ro_delay_seq=None,
                 _readout_classifier=None,
                 adc_measurement_name='Voltage', dbg_storage=False, dbg_storage_samples=False,
                 measure_projections=False, measure_features=True, post_selection_flag=False):
        self.adc = adc
        self.adc_measurement_name = adc_measurement_name
        self.device = device
        self.ro_seq = ro_seq
        self.ex_seqs = ex_seqs
        self.prepare_seqs = prepare_seqs

        self.ro_delay_seq = ro_delay_seq
        self.control_seq = control_seq
        self.repeat_samples = 3 #5
        self.save_last_samples = False
        # self.train_test_split = 0.8 # 0.8

        self.measurement_name = ''
        # self.dump_measured_samples = False

        self.measure_avg_samples = True
        self.dbg_storage = dbg_storage
        self.dbg_storage_samples = dbg_storage_samples
        self.measure_projections =measure_projections
        self.measure_features = measure_features

        # self.measure_cov_samples = False
        self.measure_hists = True
        self.measure_feature_w_threshold = True
        self.return_scores = False
        # self.measure_features = True

        self.confusion_matrix =None

        self.post_selection_flag = post_selection_flag
        if self.post_selection_flag:
            self.readouts_per_repetition = 2
        else:
            self.readouts_per_repetition = 1

        # readout shots for one repetition
        self.nums = self.readouts_per_repetition * int(device.get_sample_global(name='calibrated_readout_nums'))

        # readout shots for calibration
        self.num_shots = int(device.get_sample_global(name='calibrated_readout_nums')) * self.repeat_samples * len(
            self.prepare_seqs)

        if not _readout_classifier:
            self.readout_classifier = logistic_regression_classifier.LogisticRegressionReadoutClassifier(
                nums_adc=self.adc.get_adc_nop(),
                states=3, num_shots=self.num_shots)

        # self.cutoff = 0.97
        self.cutoff = 0.9

    def calibrate(self):
        X = []
        y = []

        print(self.adc.adc.nsegm)

        if not self.post_selection_flag:
            for i in range(self.repeat_samples):
                for class_id, prepare_seq in enumerate(self.prepare_seqs):
                    #
                    self.adc.set_internal_avg(False)

                    #TODO: optimize it somehow
                    for seq in self.control_seq[class_id][0]:
                        seq.set_awg_amp(0. * (1+1j))
                    for seq in self.control_seq[class_id][1]:
                        seq.set_awg_amp(1. * (1+1j))

                    if self.adc.devtype == 'SK':
                        measurement = self.adc.measure()
                        X.append(measurement[self.adc_measurement_name])
                        if len(self.adc.get_points()[self.adc_measurement_name]) > 1:
                            y.extend([class_id] * len(self.adc.get_points()[self.adc_measurement_name][0][1]))
                        else:
                            y.extend([class_id])

        else:
            X_ = []  # samples for first measurement (for post selection procedure)

            for i in range(self.repeat_samples):
                for class_id, prepare_seq in enumerate(self.prepare_seqs):

                    # TODO: optimize it somehow
                    for seq in self.control_seq[class_id][0]:
                        seq.set_awg_amp(0. * (1 + 1j))
                    for seq in self.control_seq[class_id][1]:
                        seq.set_awg_amp(1. * (1 + 1j))

                    if self.adc.devtype == 'SK':
                        measurement = self.adc.measure()
                        samples = measurement[self.adc_measurement_name]
                        # print(samples.shape)

                        X_.append(measurement[self.adc_measurement_name][::2, :]) # append samples only for meas0: odd



                        X.append(measurement[self.adc_measurement_name][1::2, :])  # append samples only for meas1: even
                        y.extend([class_id] * (len(self.adc.get_points()[self.adc_measurement_name][0][1]) // 2))


        print(np.asarray(X).shape, np.asarray(y).shape)
        X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
        y = np.asarray(y)
        print("X shape is {}, y shape is {} internal adc avg is True".format(X.shape, y.shape))

        self.readout_classifier.fit(X, y)
        self.readout_classifier.train()
        self.confusion_matrix = self.readout_classifier.get_confusion_matrix()

        if self.post_selection_flag:
            X_ = np.reshape(X_, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
            w_meas0 = self.adc.model.get_w_(X_)
            w_meas1 = self.adc.model.get_w_(X)

            self.w_meas0, self.w_meas1 = w_meas0, w_meas1
            self.y = y

            # self.meas0 = self.adc.model.clf.predict(w_meas0)

            prob = self.adc.model.clf.predict_proba(self.w_meas0)
            self.meas0 = np.asarray(prob[:, 0] < self.cutoff, dtype=int)


        if self.dbg_storage_samples:
            print('ВНИМАНИЕ, СОХРАНЯЕТСЯ МНОГО ДАННЫХ!!! ОТКЛЮЧИ dbg_storage_samples!!!')
            self.x = X
            self.y = y


            if self.post_selection_flag:
                self.X_ = X_

    def get_opts(self):
        opts = {}
        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
            opts.update(avg_samples)

        if self.dbg_storage_samples:
            opts['x'] = {'log': False}
            opts['y'] = {'log': False}

        # train_samples = {'avg_sample_train' + str(_class): {'log': False} for _class in self.readout_classifier.class_train}
        # opts.update(train_samples)
        if self.measure_features:
            feature0 = {'feature0': {'log': False}}
            opts.update(feature0)
            feature1 = {'feature1': {'log': False}}
            opts.update(feature1)

        # if self.measure_projections:
        #     states_projections = {'x' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
        #     opts.update(states_projections)
        if self.measure_hists:
            hists0 = {'hist_axe_0_state' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
            hists1 = {'hist_axe_1_state' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
            opts.update(hists0)
            opts.update(hists1)

        opts.update({'confusion_matrix': {'log': False}})
        opts.update({'w': {'log': False}})
        opts.update({'marker': {'log': False}})

        if self.post_selection_flag:
            # opts.update({'w_': {'log': False}})
            opts.update({'w_meas0': {'log': False}})
            opts.update({'w_meas1': {'log': False}})

            opts.update({'y': {'log': False}})
            opts.update({'meas0': {'log': False}})

            if self.dbg_storage_samples:
                opts['x_'] = {'log': False}

        return opts

    def measure(self):
        self.calibrate()
        meas = {}
        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): self.readout_classifier.class_averages[_class] for _class in
                           self.readout_classifier.class_list}
            meas.update(avg_samples)

        if self.dbg_storage_samples:
            meas['x'], meas['y'] = self.x, self.y

        # train_samples = {'avg_sample_train' + str(_class): self.readout_classifier.class_train[_class] for _class in
        #          self.readout_classifier.class_list}
        # meas.update(train_samples)
        if self.measure_features:
            feature0 = {'feature0': self.readout_classifier.feature0}
            feature1 = {'feature1': self.readout_classifier.feature1}
            meas.update(feature0)
            meas.update(feature1)

        # if self.measure_projections:
        #     states_projections = {'x' + str(_class): self.readout_classifier.projections0[_class] + 1j *
        #                                              self.readout_classifier.projections1[_class] for _class in
        #                           self.readout_classifier.class_list}
        #     meas.update(states_projections)
        if self.measure_hists:
            hists0 = {'hist_axe_0_state' + str(_class): self.readout_classifier.hist_projections0[_class] for _class in self.readout_classifier.class_list}
            hists1 = {'hist_axe_1_state' + str(_class): self.readout_classifier.hist_projections1[_class] for _class in self.readout_classifier.class_list}
            meas.update(hists0)
            meas.update(hists1)

        meas.update({'confusion_matrix': self.readout_classifier.confusion_matrix})
        meas.update({'w': self.readout_classifier.w})
        meas.update({'marker': self.readout_classifier.marker})

        if self.post_selection_flag:
            meas.update({'w_meas0': self.w_meas0})
            meas.update({'w_meas1': self.w_meas1})
            meas.update({'y': self.y})
            meas.update({'meas_0': self.meas0})

            if self.dbg_storage_samples:
                meas['x_'] = self.x_
        return meas

    def get_points(self):
        points = {}
        if self.measure_avg_samples:
            avg_samples = {
                'avg_sample' + str(_class): [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')] for
                _class in self.readout_classifier.class_list}
            points.update(avg_samples)

        if self.dbg_storage_samples:
            adc_points = self.adc.get_points()[self.adc_measurement_name]
            if len(adc_points) < 2:
                raise IndexError('dbg_storage not available with on-digitizer averaging')
            num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1]) * self.repeat_samples * (len(self.prepare_seqs) // self.readouts_per_repetition)
            points['x'] = [('Segment', np.arange(num_shots), ''),
                           ('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
            points['y'] = [('Segment', np.arange(num_shots), '')]

        # train_samples = {
        #     'avg_sample_train' + str(_class): [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')] for
        #     _class in self.readout_classifier.class_list}
        # points.update(train_samples)
        if self.measure_features:
            feature0 = {'feature0': [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]}
            feature1 = {'feature1': [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]}
            points.update(feature0)
            points.update(feature1)
        # if self.measure_projections:
        #     states_projections = {'x' + str(_class): [('Segment', np.arange(int(self.num_shots * self.train_test_split)), '')] for _class in
        #                           self.readout_classifier.class_list}
        #     points.update(states_projections)
        if self.measure_hists:
            hists0 = {'hist_axe_0_state' + str(_class): [('bins', np.arange(50), '')] for _class in
                      self.readout_classifier.class_list}
            hists1 = {'hist_axe_1_state' + str(_class): [('bins', np.arange(50), '')] for _class in
                      self.readout_classifier.class_list}
            points.update(hists0)
            points.update(hists1)
        points.update({'confusion_matrix': [('State', np.arange(len(self.readout_classifier.class_list)), ''),
                                            ('State', np.arange(len(self.readout_classifier.class_list)), '')]})

        points.update({'w': [('Segment', np.arange(self.num_shots // 2), ''),
                             ('axe', np.arange(2), '')]})
        points.update({'marker': [('States', np.arange(self.num_shots // 2), '')]})

        if self.post_selection_flag:
            points.update({'w_meas0': [('Segment', np.arange(self.num_shots), ''),
                                  ('axe', np.arange(2), '')]})
            points.update({'w_meas1': [('Segment', np.arange(self.num_shots), ''),
                                  ('axe', np.arange(2), '')]})
            points.update({'y': [('States', np.arange(self.num_shots), '')]})

            points.update({'meas_0': [('Segment', np.arange(self.num_shots), '')]})

            if self.dbg_storage_samples:
                points['x_'] = [('Segment', np.arange(num_shots), ''),
                               ('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]

        return points

    def get_dtype(self):
        dtypes = {}

        if self.measure_avg_samples:
            avg_samples = {'avg_sample' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
                           self.readout_classifier.class_list}
            dtypes.update(avg_samples)

        if self.dbg_storage_samples:
        # if self.adc.devtype == 'UHF' and self.return_scores:
            dtypes['x'] = np.complex64
            dtypes['y'] = float
        # train_samples = {'avg_sample_train' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
        #                self.readout_classifier.class_list}
        # dtypes.update(train_samples)
        if self.measure_features:
            dtypes['feature0'], dtypes['feature1'] = np.complex64, np.complex64

        # if self.measure_projections:
        #     states_projections = {'x' + str(_class): np.complex64 for _class in
        #                           self.readout_classifier.class_list}
        #     dtypes.update(states_projections)
        if self.measure_hists:
            hists0 = {'hist_axe_0_state' + str(_class): float for _class in
                      self.readout_classifier.class_list}
            hists1 = {'hist_axe_1_state' + str(_class): float for _class in
                      self.readout_classifier.class_list}
            dtypes.update(hists0)
            dtypes.update(hists1)
        dtypes.update({'confusion_matrix': float})
        dtypes.update({'w': float})
        dtypes.update({'marker': float})

        if self.post_selection_flag:
            # dtypes.update({'w_': float})
            dtypes.update({'w_meas0': float})
            dtypes.update({'w_meas1': float})
            dtypes.update({'y': float})

            dtypes.update({'meas_0': float})

            if self.dbg_storage_samples:
                dtypes['x_'] = np.complex64

        return dtypes

    def filter_binary_func(self, x):
        return self.readout_classifier.predict(x[self.adc_measurement_name])


# from qsweepy.libraries import data_reduce
# import numpy as np
# from qsweepy.qubit_calibrations import sequence_control
# # from qsweepy.libraries import logistic_regression_classifier
# from qsweepy.libraries import logistic_regression_classifier2 as logistic_regression_classifier
# import pickle as pkl
# from time import gmtime, strftime
# import datetime
#
# class SingleShotReadout:
#     """
#     Single shot readout class
#     Args:
#         adc (Instrument): a device that measures a complex vector for each readout trigger (an ADC)
#         prepare_seqs (dict of pulses.sequence): a dict of sequences of control pulses. The keys are use for state identification.
#         ro_seq (pulses.sequence): a sequence of control pulses that is used to generate the reaout pulse of the DAC.
#         ro_delay_seq (pulses.sequence): Sequence used to align the DAC and ADC (readout delay compensation)
#         adc_measurement_name (str): name of measurement on ADC
#     """
#
#     def __init__(self, device, adc, prepare_seqs, ex_seqs, ro_seq, control_seq, ro_delay_seq=None,
#                  _readout_classifier=None,
#                  adc_measurement_name='Voltage', dbg_storage=False, dbg_storage_samples=False,
#                  measure_projections=False, measure_features=True, double_train=True):
#         self.adc = adc
#         self.adc_measurement_name = adc_measurement_name
#         self.device = device
#         self.ro_seq = ro_seq
#         self.ex_seqs = ex_seqs
#         self.prepare_seqs = prepare_seqs
#
#         self.ro_delay_seq = ro_delay_seq
#         self.control_seq = control_seq
#         self.repeat_samples = 5
#         self.save_last_samples = False
#         self.train_test_split = 0.8 # 0.8
#
#         self.measurement_name = ''
#         # self.dump_measured_samples = False
#
#         self.measure_avg_samples = True
#         self.dbg_storage = dbg_storage
#         self.dbg_storage_samples = dbg_storage_samples
#         self.measure_projections =measure_projections
#         self.measure_features = measure_features
#
#         # self.measure_cov_samples = False
#         self.measure_hists = True
#         self.measure_feature_w_threshold = True
#         self.return_scores = False
#         # self.measure_features = True
#
#         # self.cutoff_start = 0
#         # self.num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1]) * self.repeat_samples * len(
#         #     self.prepare_seqs)
#         self.num_shots = int(device.get_sample_global(name='calibrated_readout_nums')) * self.repeat_samples * len(
#             self.prepare_seqs)
#
#         self.num_test = int(self.num_shots * self.train_test_split)
#         self.num_train = self.num_shots - self.num_test
#         if not _readout_classifier:
#             self.readout_classifier = logistic_regression_classifier.LogisticRegressionReadoutClassifier(nums_adc=adc.get_adc_nop(),
#                                                                           num_shots=self.num_shots,
#                                                                           train_size=self.train_test_split)
#
#         self.double_train = double_train
#
#         self.filter_binary = dict(get_points=lambda: (self.adc.get_points()[adc_measurement_name][0],),
#                                   get_dtype=lambda: int, get_opts=lambda: {}, filter=self.filter_binary_func)
#
#     def calibrate(self):
#         X = []
#         y = []
#         for i in range(self.repeat_samples):
#             for class_id, prepare_seq in enumerate(self.prepare_seqs):
#                 #
#                 self.adc.set_internal_avg(False)
#
#                 for seq in self.control_seq[class_id][0]:
#                     seq.set_awg_amp(0. * (1+1j))
#                 for seq in self.control_seq[class_id][1]:
#                     seq.set_awg_amp(1. * (1+1j))
#
#                 if self.adc.devtype == 'SK':
#                     measurement = self.adc.measure()
#                     X.append(measurement[self.adc_measurement_name])
#                     if len(self.adc.get_points()[self.adc_measurement_name]) > 1:
#                         y.extend([class_id] * len(self.adc.get_points()[self.adc_measurement_name][0][1]))
#                     else:
#                         y.extend([class_id])
#
#
#         print(np.asarray(X).shape, np.asarray(y).shape)
#         X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
#         y = np.asarray(y)
#         print("X shape is {}, y shape is {} internal adc avg is True".format(X.shape, y.shape))
#
#         if self.dbg_storage_samples:
#             print('ВНИМАНИЕ, СОХРАНЯЕТСЯ МНОГО ДАННЫХ!!! ОТКЛЮЧИ dbg_storage_samples!!!')
#             self.x = X
#             self.y = y
#
#         self.readout_classifier.fit(X, y)
#         self.readout_classifier.train()
#
#
#         confusion_matrix = self.readout_classifier.confusion_matrix
#         print("CONFUSION MATRIX IS ", confusion_matrix)
#
#     def get_opts(self):
#         opts = {}
#         if self.measure_avg_samples:
#             avg_samples = {'avg_sample' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
#             opts.update(avg_samples)
#
#         if self.dbg_storage_samples:
#             opts['x'] = {'log': False}
#             opts['y'] = {'log': False}
#
#         # train_samples = {'avg_sample_train' + str(_class): {'log': False} for _class in self.readout_classifier.class_train}
#         # opts.update(train_samples)
#         if self.measure_features:
#             feature0 = {'feature0': {'log': False}}
#             opts.update(feature0)
#             feature1 = {'feature1': {'log': False}}
#             opts.update(feature1)
#
#         # if self.measure_projections:
#         #     states_projections = {'x' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
#         #     opts.update(states_projections)
#
#         hists0 = {'hist_axe_0_state' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
#         hists1 = {'hist_axe_1_state' + str(_class): {'log': False} for _class in self.readout_classifier.class_list}
#         opts.update(hists0)
#         opts.update(hists1)
#
#         opts.update({'confusion_matrix': {'log': False}})
#         opts.update({'w': {'log': False}})
#         opts.update({'marker': {'log': False}})
#         return opts
#
#     def measure(self):
#         self.calibrate()
#         meas = {}
#         if self.measure_avg_samples:
#             avg_samples = {'avg_sample' + str(_class): self.readout_classifier.class_averages[_class] for _class in
#                            self.readout_classifier.class_list}
#             meas.update(avg_samples)
#
#         if self.dbg_storage_samples:
#             meas['x'], meas['y'] = self.x, self.y
#
#         # train_samples = {'avg_sample_train' + str(_class): self.readout_classifier.class_train[_class] for _class in
#         #          self.readout_classifier.class_list}
#         # meas.update(train_samples)
#         if self.measure_features:
#             feature0 = {'feature0': self.readout_classifier.feature0}
#             feature1 = {'feature1': self.readout_classifier.feature1}
#             meas.update(feature0)
#             meas.update(feature1)
#
#         # if self.measure_projections:
#         #     states_projections = {'x' + str(_class): self.readout_classifier.projections0[_class] + 1j *
#         #                                              self.readout_classifier.projections1[_class] for _class in
#         #                           self.readout_classifier.class_list}
#         #     meas.update(states_projections)
#
#         hists0 = {'hist_axe_0_state' + str(_class): self.readout_classifier.hist_projections0[_class] for _class in self.readout_classifier.class_list}
#         hists1 = {'hist_axe_1_state' + str(_class): self.readout_classifier.hist_projections1[_class] for _class in self.readout_classifier.class_list}
#         meas.update(hists0)
#         meas.update(hists1)
#         meas.update({'confusion_matrix': self.readout_classifier.confusion_matrix})
#         meas.update({'w': self.readout_classifier.w})
#         meas.update({'marker': self.readout_classifier.marker})
#         return meas
#
#     def get_points(self):
#         points = {}
#         if self.measure_avg_samples:
#             avg_samples = {
#                 'avg_sample' + str(_class): [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')] for
#                 _class in self.readout_classifier.class_list}
#             points.update(avg_samples)
#
#         if self.dbg_storage_samples:
#             adc_points = self.adc.get_points()[self.adc_measurement_name]
#             if len(adc_points) < 2:
#                 raise IndexError('dbg_storage not available with on-digitizer averaging')
#             num_shots = len(self.adc.get_points()[self.adc_measurement_name][0][1]) * self.repeat_samples * len(
#                 self.prepare_seqs)
#             points['x'] = [('Segment', np.arange(num_shots), ''),
#                            ('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]
#             points['y'] = [('Segment', np.arange(num_shots), '')]
#
#         # train_samples = {
#         #     'avg_sample_train' + str(_class): [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')] for
#         #     _class in self.readout_classifier.class_list}
#         # points.update(train_samples)
#         if self.measure_features:
#             feature0 = {'feature0': [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]}
#             feature1 = {'feature1': [('Time', np.arange(self.adc.get_adc_nop()) / self.adc.get_clock(), 's')]}
#             points.update(feature0)
#             points.update(feature1)
#         # if self.measure_projections:
#         #     states_projections = {'x' + str(_class): [('Segment', np.arange(int(self.num_shots * self.train_test_split)), '')] for _class in
#         #                           self.readout_classifier.class_list}
#         #     points.update(states_projections)
#
#         hists0 = {'hist_axe_0_state' + str(_class): [('bins', np.arange(50), '')] for _class in
#                   self.readout_classifier.class_list}
#         hists1 = {'hist_axe_1_state' + str(_class): [('bins', np.arange(50), '')] for _class in
#                   self.readout_classifier.class_list}
#         points.update(hists0)
#         points.update(hists1)
#         points.update({'confusion_matrix': [('State', np.arange(len(self.readout_classifier.class_list)), ''),
#                                             ('State', np.arange(len(self.readout_classifier.class_list)), '')]})
#
#         points.update({'w': [('Segment', np.arange(self.num_train), ''),
#                              ('axe', np.arange(2), '')]})
#         points.update({'marker': [('States', np.arange(self.num_train), '')]})
#
#         # if not self.double_train:
#         #     points.update({'w': [('Segment', np.arange(self.num_train), ''),
#         #                          ('axe', np.arange(2), '')]})
#         #     points.update({'marker': [('States', np.arange(self.num_train), '')]})
#         # else:
#         #     points.update({'w': [('Segment', np.arange(self.num_shots), ''),
#         #                          ('axe', np.arange(2), '')]})
#         #     points.update({'marker': [('States', np.arange(self.num_shots), '')]})
#         return points
#
#     def get_dtype(self):
#         dtypes = {}
#
#         if self.measure_avg_samples:
#             avg_samples = {'avg_sample' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
#                            self.readout_classifier.class_list}
#             dtypes.update(avg_samples)
#
#         if self.dbg_storage_samples:
#         # if self.adc.devtype == 'UHF' and self.return_scores:
#             dtypes['x'] = np.complex64
#             dtypes['y'] = float
#         # train_samples = {'avg_sample_train' + str(_class): self.adc.get_dtype()[self.adc_measurement_name] for _class in
#         #                self.readout_classifier.class_list}
#         # dtypes.update(train_samples)
#         if self.measure_features:
#             dtypes['feature0'], dtypes['feature1'] = np.complex64, np.complex64
#
#         # if self.measure_projections:
#         #     states_projections = {'x' + str(_class): np.complex64 for _class in
#         #                           self.readout_classifier.class_list}
#         #     dtypes.update(states_projections)
#
#         hists0 = {'hist_axe_0_state' + str(_class): float for _class in
#                   self.readout_classifier.class_list}
#         hists1 = {'hist_axe_1_state' + str(_class): float for _class in
#                   self.readout_classifier.class_list}
#         dtypes.update(hists0)
#         dtypes.update(hists1)
#         dtypes.update({'confusion_matrix': float})
#         dtypes.update({'w': float})
#         dtypes.update({'marker': float})
#         return dtypes
#
#     def filter_binary_func(self, x):
#         return self.readout_classifier.predict(x[self.adc_measurement_name])
