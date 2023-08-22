import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

class LogisticRegressionReadoutClassifier:
    """
    Logistic Regression readout classifier
    """
    def __init__(self, nums_adc=None, num_shots=None, train_size=0.8,  name='', states=3, standardize=False):
        """
        :param nums_adc:
        :param num_shots:
        """
        self.nums_adc = nums_adc
        self.num_shots = num_shots
        self.name = name
        self.states = states
        self.standardize = standardize

        self.train_size = train_size

        self.class_list = list(np.arange(self.states))
        self.class_averages = {}
        self.class_samples = {}
        self.class_test = {}
        self.class_train = {}
        self.class_num_shots = {}

        self.feature0 = None
        self.feature1 = None

        self.x = None
        self.y = None

        self.projections0 = {}
        self.projections1 = {}

        self.hist_projections0 = {}
        self.hist_projections1 = {}
        self.clf = None
        self.marker = None
        self.w = None

        self.confusion_matrix = None

    def fit(self, x, y):
        """
        Fit statistic data
        """
        self.x = x
        self.y = y

        for class_id in self.class_list:
            ind_ = [i for i in range(self.num_shots) if y[i] == class_id]
            x_class_id = x[ind_]  # samples with class_id

            x_class_id_train, x_class_id_test =  train_test_split(x_class_id, test_size=1 - self.train_size,
                                                                   train_size=self.train_size, random_state=42)

            print(x_class_id_train.shape, x_class_id_test.shape)
            self.class_test.update({class_id: x_class_id_test})
            self.class_train.update({class_id: np.mean(x_class_id_train, axis=0)})
            self.class_num_shots.update({class_id: len(x_class_id_test)})
            self.class_averages.update({class_id : np.mean(x[y==class_id, :], axis=0)})

        print(self.class_num_shots)


        self.feature0, self.feature1 = self.get_features(self.class_train[0], self.class_train[1], self.class_train[2])

    def get_feature_projection(self, feature, trajectories):
        return np.real((trajectories @ feature.reshape(self.nums_adc, 1)).ravel())

    def get_features(self, avg_sample0, avg_sample1, avg_sample2):
        """
        Get features
        """
        f_10 = avg_sample1.conj() - avg_sample0.conj()
        f_20 = avg_sample2.conj() - avg_sample0.conj()

        numerator = ((avg_sample2 - avg_sample0) @ f_10.reshape(self.nums_adc, 1)).ravel()[0]
        denominator = (f_10.conj() @ f_10.reshape(self.nums_adc, 1)).ravel()[0]
        return f_10, f_20 - numerator / denominator * f_10

    def train(self, w = np.asarray([]), marker = np.asarray([])):
        """
        Train model using data w and markers
        """
        if w.any() and marker.any():
            self.w, self.marker = w, marker
            num_shots_test = self.marker.shape[0] // 3
        else:
            self.w, self.marker = self.get_w_and_markers()
            num_shots_test = self.class_test[0].shape[0]


        clf = LogisticRegression(random_state=0).fit(self.w, self.marker)
        y_hat = clf.predict(self.w)
        clf.score(self.w, self.marker)
        self.clf = clf
        self.confusion_matrix = confusion_matrix(self.marker, y_hat) / num_shots_test

    def get_w_and_markers(self):
        """
        Get w and markers for train model
        """
        for class_id in self.class_list:
            self.projections0.update({class_id: self.get_feature_projection(self.feature0, self.class_test[class_id])})
            self.projections1.update({class_id: self.get_feature_projection(self.feature1, self.class_test[class_id])})

            self.hist_projections0.update({class_id: np.histogram(self.projections0[class_id], bins=50)[0]})
            self.hist_projections1.update({class_id: np.histogram(self.projections1[class_id], bins=50)[0]})


        # num_shots_test = self.class_test[0].shape[0]
        # w = np.zeros((num_shots_test * self.states, 2))
        #
        # w[:num_shots_test, 0], w[:num_shots_test, 1] = self.projections0[0], self.projections1[0]
        # w[num_shots_test:2 * num_shots_test, 0], w[num_shots_test:2 * num_shots_test, 1] = self.projections0[1], \
        #                                                                                    self.projections1[1]
        # w[2 * num_shots_test:, 0], w[2 * num_shots_test:, 1] = self.projections0[2], self.projections1[2]

        # marker = np.zeros(3 * num_shots_test)
        # marker[:num_shots_test] = 0
        # marker[num_shots_test: 2 * num_shots_test] = 1
        # marker[2 * num_shots_test:] = 2


        w = np.zeros((np.sum(list(self.class_num_shots.values())), 2))
        marker = np.zeros(np.sum(list(self.class_num_shots.values())))

        i = 0
        j = self.class_num_shots[self.class_list[0]]
        for class_id in self.class_list:
            w[i:j, 0], w[i:j, 1] = self.projections0[class_id], self.projections1[class_id]
            marker[i:j] = class_id

            if class_id < self.class_list[-1]:
                i += self.class_num_shots[class_id]
                j += self.class_num_shots[class_id + 1]




        # if self.standardize:
        #     ss = StandardScaler()
        #     ss.fit(w)
        #     w = ss.transform(w)
        return w, marker

    def get_w_(self, trajectories):
        """
        Get w_ for predictions data from trajectories
        """
        projection0, projection1 = self.get_feature_projection(self.feature0, trajectories), self.get_feature_projection(self.feature1,
                                                                                                          trajectories)
        w_ = np.zeros((trajectories.shape[0], 2))
        w_[:, 0], w_[:, 1] = projection0, projection1
        return w_

    def predict(self, w_):
        y_hat = self.clf.predict(w_)
        return y_hat

    def set_w_and_markers(self, w, marker):
        self.marker = marker
        self.w = w


