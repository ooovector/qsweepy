import numpy as np
from qsweepy.ponyfiles import exdir_db, data_structures, database
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from scipy.linalg import expm
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


db = database.MyDatabase()


class ReadoutPostProcessing:
    """
    A class for readout postprocessing
    """

    def __init__(self, meas_id, sample_name):
        self.meas_id = meas_id
        self.sample_name = sample_name
        self.exdir_db_inst = exdir_db.Exdir_db(db=db, sample_name=sample_name)

        self.x = np.asarray([])
        self.y = np.asarray([])

        self.x0 = None
        self.x1 = None

    def get_and_separate_data(self, max_fid=True):
        """
        Get data from measurement
        :param max_fid: if True choose amplitude with max fidelity
        :return:
        """
        m = self.exdir_db_inst.select_measurement_by_id(self.meas_id)
        # if hasattr(m.datasets['fidelity'].data, '__iter__'):
        if m.datasets['fidelity'].parameters:
            ampl = m.datasets['fidelity'].parameters[1].values
            fidelity = m.datasets['fidelity'].data[0]
            if len(ampl) > 0:
                if max_fid:
                    indx = np.argmax(fidelity)
            else:
                indx = 0
            self.x = m.datasets['x'].data[0][indx, :, :]
            self.y = m.datasets['y'].data[0][indx, :]
        else:
            self.x = m.datasets['x'].data
            self.y = m.datasets['y'].data

        num_samples = len(self.y)
        ind_0 = [i for i in range(num_samples) if self.y[i] == 0]
        self.x0 = self.x[ind_0] # samples without pi pulse
        ind_1 = [i for i in range(num_samples) if self.y[i] == 1]
        self.x1 = self.x[ind_1] # samples with pi pulse
        # return self.x0, self.x1

    def data_postselection(self):
        m = self.exdir_db_inst.select_measurement_by_id(self.meas_id)


    def get_features(self, fraction_train=0.5, fraction_test=0.5):
        if not (self.x0.any() or self.x1.any()):
            raise ValueError("Get and separate data before readout postprocessing!")
        x0_train, x0_test = train_test_split(self.x0, test_size=fraction_test, train_size=fraction_train, random_state=42)
        x1_train, x1_test = train_test_split(self.x1, test_size=fraction_test, train_size=fraction_train, random_state=42)

        a0, a1, a0_wave, a1_wave  = np.real(x0_train), np.real(x1_train), np.real(x0_test), np.real(x1_test)
        b0, b1, b0_wave, b1_wave = np.imag(x0_train), np.imag(x1_train), np.imag(x0_test), np.imag(x1_test)
        fa = function(a0, a1)
        fb = function(b0, b1)

        t0a = make_statistic_of_projects(fa, a0_wave)
        t1a = make_statistic_of_projects(fa, a1_wave)
        t0b = make_statistic_of_projects(fb, b0_wave)
        t1b = make_statistic_of_projects(fb, b1_wave)
        l = len(t0a)
        W = np.zeros((2 * l, 2))
        W[:l, 0] = t0a
        W[:l, 1] = t0b
        W[l:, 0] = t1a
        W[l:, 1] = t1b
        marker = np.zeros(2 * l)
        marker[:l] = 0
        marker[l:] = 1
        return W, marker, l

    def get_features_and_standardize(self, fraction_train=0.5, fraction_test=0.5):
        W, marker, l = self.get_features(fraction_train, fraction_test)
        ss = StandardScaler()
        ss.fit(W)
        W = ss.transform(W)
        return W, marker, l


# Функция выдаёт массив проекций (n чисел) array_pr
def make_statistic_of_projects(f, x):
    n = x.shape[0]
    array_pr = [count_projection(f, x[i]) for i in range(n)]
    return array_pr

# Функция принимает массив из n траекторий J для случаев, когда кубит изначально
# Находился в состоянии 1 и 0
def function(x0, x1):
    n = np.shape(x0)[0]
    f = np.mean(x1[:n], axis=0) - np.mean(x0[:n], axis=0)
    return f

def count_fidelity_logistic(marker, y_hat):
    f1 = np.sum(y_hat * marker) * 2 / len(marker)
    f0 = np.sum((1 - y_hat) * (1 - marker)) * 2 / len(marker)
    return f1, f0

def count_projection(f, x):
    return np.sum(x * f)




