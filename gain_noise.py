from scipy.constants import Planck, Boltzmann
import numpy as np

def planck_function(f, Ts, gains):
       return np.sum([Planck*f*(0.5+1./(np.exp(Planck*f/(Boltzmann*T))-1))*gain for T, gain in zip(Ts, gains)])

def gain_noise(f, P_meas, T1, T2):
    G_ = np.zeros_like(f)
    TN_ = np.zeros_like(f)
    P_meas = np.asarray(P_meas)
    for f_id, f_ in enumerate(f):
        P_in = [planck_function(f_,[t[0] for t in T1],[t[1] for t in T1]), 
                planck_function(f_,[t[0] for t in T2],[t[1] for t in T2])] # input noise powers
        a = np.asarray([[1, P_in[0]], [1, P_in[1]]])
        b = P_meas[:, f_id].T
        solution = np.linalg.solve(a, b)
        GkTNbw, G = solution
        TN = GkTNbw/(Boltzmann*G)
        G_[f_id] = G
        TN_[f_id] = TN
    return G_, TN_