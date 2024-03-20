import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
# from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

class LogisticRegressionReadoutClassifier:
    """
    Logistic Regression readout classifier for qubit and qutrit readout
    """
    def __init__(self, nums_adc=None, num_shots=None, train_size=0.5, name='', states=3, standardize=False):
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
        self.class_num_train = {}
        self.class_num_test = {}

        self.class_train_averages ={}

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
        Fit statistic data, create train and test data, calculate features
        """
        self.x = x
        self.y = y

        for class_id in self.class_list:
            ind_ = [i for i in range(self.num_shots) if y[i] == class_id]
            x_class_id = x[ind_]  # samples with class_id

            x_class_id_train, x_class_id_test = train_test_split(x_class_id, test_size=1 - self.train_size,
                                                                 train_size=self.train_size, random_state=42)

            self.class_test.update({class_id: x_class_id_test})
            self.class_train.update({class_id: x_class_id_train})

            self.class_num_train.update({class_id: len(x_class_id_train)})
            self.class_num_test.update({class_id: len(x_class_id_test)})

            self.class_train_averages.update({class_id: np.mean(x_class_id_test, axis=0)})
            self.class_averages.update({class_id: np.mean(x_class_id, axis=0)})
        if self.states == 2:
            self.feature0, self.feature1 = self.get_features(self.class_train_averages[0], self.class_train_averages[1])
        elif self.states == 3:
            self.feature0, self.feature1 = self.get_features(self.class_train_averages[0], self.class_train_averages[1],
                                                             self.class_train_averages[2])
        else:
            raise ValueError("Logistic Regression readout classifier supports only qubit and qutrit readout!")

    def get_feature_projection(self, feature, trajectories):
        return np.real((trajectories @ feature.reshape(self.nums_adc, 1)).ravel())

    def get_features(self, avg_sample0, avg_sample1, avg_sample2=None):
        """
        Get features f0 and f1
        """
        if self.states == 2:
            f0 = avg_sample1.real - avg_sample0.real
            f1 = avg_sample1.imag - avg_sample0.imag
            return f0, f1
        elif self.states == 3:
            f_10 = avg_sample1.conj() - avg_sample0.conj()
            f_20 = avg_sample2.conj() - avg_sample0.conj()
            # f_21 = avg_sample2.conj() - avg_sample1.conj()

            numerator = ((avg_sample2 - avg_sample0) @ f_10.reshape(self.nums_adc, 1)).ravel()[0]
            denominator = (f_10.conj() @ f_10.reshape(self.nums_adc, 1)).ravel()[0]
            return f_10, f_20 - numerator / denominator * f_10

        else:
            raise ValueError("Logistic Regression readout classifier supports only qubit and qutrit readout!")

    def train(self, w=np.asarray([]), marker=np.asarray([])):
        """
        Train model using data w and markers
        """
        if w.any() and marker.any():
            self.w, self.marker = w, marker

        else:
            self.w, self.marker = self.get_w_and_markers()

        clf = LogisticRegression(random_state=0).fit(self.w, self.marker)
        self.clf = clf

        # y_hat = clf.predict(self.w)
        # clf.score(self.w, self.marker)
        # self.clf = clf
        # self.confusion_matrix = confusion_matrix(self.marker, y_hat) / num_shots_test

    def get_w_and_markers(self):
        """
        Get w and markers for train model
        """
        if self.states == 2:
            for class_id in self.class_list:
                self.projections0.update({class_id: self.get_feature_projection(self.feature0,
                                                                                self.class_train[class_id].real)})
                self.projections1.update({class_id: self.get_feature_projection(self.feature1,
                                                                                self.class_train[class_id].imag)})
        elif self.states == 3:
            for class_id in self.class_list:
                self.projections0.update({class_id: self.get_feature_projection(self.feature0, self.class_train[class_id])})
                self.projections1.update({class_id: self.get_feature_projection(self.feature1, self.class_train[class_id])})
        else:
            raise ValueError("Logistic Regression readout classifier supports only qubit and qutrit readout!")

        for class_id in self.class_list:
            self.hist_projections0.update({class_id: np.histogram(self.projections0[class_id], bins=50)[0]})
            self.hist_projections1.update({class_id: np.histogram(self.projections1[class_id], bins=50)[0]})

        w = np.zeros((np.sum(list(self.class_num_train.values())), 2))
        marker = np.zeros(np.sum(list(self.class_num_train.values())))

        i = 0
        j = self.class_num_train[self.class_list[0]]

        for class_id in self.class_list:
            w[i:j, 0], w[i:j, 1] = self.projections0[class_id], self.projections1[class_id]
            marker[i:j] = class_id

            if class_id < self.class_list[-1]:
                i += self.class_num_train[class_id]
                j += self.class_num_train[class_id + 1]

        # if self.standardize:
        #     ss = StandardScaler()
        #     ss.fit(w)
        #     w = ss.transform(w)
        return w, marker

    def get_w_(self, trajectories):
        """
        Get w_ for predictions data from trajectories
        """
        if self.states == 2:
            projection0, projection1 = self.get_feature_projection(self.feature0, trajectories.real), \
                                       self.get_feature_projection(self.feature1, trajectories.imag)
        elif self.states == 3:
            projection0, projection1 = self.get_feature_projection(self.feature0, trajectories), \
                                       self.get_feature_projection(self.feature1, trajectories)
        else:
            raise ValueError("Logistic Regression readout classifier supports only qubit and qutrit readout!")
        w_ = np.zeros((trajectories.shape[0], 2))
        w_[:, 0], w_[:, 1] = projection0, projection1
        return w_

    def predict(self, w_):
        y_hat = self.clf.predict(w_)
        return y_hat

    def set_w_and_markers(self, w, marker):
        self.marker = marker
        self.w = w

    def get_confusion_matrix(self, w_test=np.asarray([]), test_marker=np.asarray([]), plot=False):
        if not (w_test.any() and test_marker.any()):

            test_trajectories = np.concatenate([self.class_test[class_id] for class_id in self.class_list])
            test_marker = np.asarray(
                [[class_id] * list(self.class_num_test.values())[class_id] for class_id in self.class_list]).ravel()

            w_test = self.get_w_(test_trajectories)

        y_hat = self.clf.predict(w_test)
        num_shots_test = test_marker.shape[0] // self.states
        self.clf.score(w_test, test_marker)
        self.confusion_matrix = confusion_matrix(test_marker, y_hat) / num_shots_test


        if plot:
            import pandas as pd
            import seaborn as sns
            sns.set_theme(style="ticks")

            # state_names = ['|0>', '|1>', '|2>']
            state_names  = ['|' + str(i) + '>' for i in self.class_list]
            df = pd.DataFrame(data=self.confusion_matrix, index=state_names, columns=state_names)

            ax = sns.heatmap(df / np.sum(df.T), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), fmt='g')
            ax.set_xlabel('Prepared state')
            ax.set_ylabel('Assigned state')


        # y_hat = self.clf.predict(self.w)
        # self.clf.score(self.w, self.marker)
        # # self.clf = clf
        # num_shots_test = self.marker.shape[0] // self.states
        # self.confusion_matrix = confusion_matrix(self.marker, y_hat) / num_shots_test

        return self.confusion_matrix

    def get_fidelity(self):
        """
        Get readout fidelity from confusion matrix
        """
        return np.mean(np.diag(self.confusion_matrix))

    # def display_boundaries(self):
    #     disp = DecisionBoundaryDisplay.from_estimator(
    #         self.clf, self.w, response_method="auto",
    #         xlabel="I", ylabel="Q",
    #         alpha=0.2, plot_method="contourf",
    #         # shading="auto",
    #         eps=0.1,
    #         grid_resolution=500)
    #     return disp

    def get_w_and_markers_from_w_meas(self, w_meas, y):
        """
        Calculate confusion matrix from projections
        :param w_meas: array of projections
        :param y: array of markers
        """
        class_w = {}
        class_w_test = {}
        class_w_train = {}

        markers = []
        test_marker = []

        for class_id in self.class_list:
            ind_ = [i for i in range(len(y)) if y[i] == class_id]
            class_w.update({class_id: w_meas[ind_, :]})
            w_class_id_train, w_class_id_test = train_test_split(class_w[class_id], test_size=1 - self.train_size,
                                                                 train_size=self.train_size, random_state=42)

            # self.projections0.update({class_id: w_class_id_train[:, 0]})
            # self.projections1.update({class_id: w_class_id_train[:, 1]})
            class_w_train.update({class_id: w_class_id_train})
            class_w_test.update({class_id: w_class_id_test})

            markers.extend([class_id] * w_class_id_train.shape[0])
            test_marker.extend([class_id] * w_class_id_test.shape[0])

        w_test = class_w_test[0]
        w_train = class_w_train[0]

        for class_id in self.class_list[1:]:
            w_train = np.vstack((w_train, class_w_train[class_id]))
            w_test = np.vstack((w_test, class_w_test[class_id]))

        return w_train, w_test, np.asarray(markers), np.asarray(test_marker)