from scipy.constants import Planck, Boltzmann
import numpy as np

def planck_function(f, Ts, gains):
       return np.sum([Planck*f*(0.5+1./(np.exp(Planck*f/(Boltzmann*T))-1))*gain for T, gain in zip(Ts, gains)])

def gain_noise(f, P_meas, T1, T2, bw):
    G_ = np.zeros_like(f)
    TN_ = np.zeros_like(f)
    P_meas = np.asarray(P_meas)
    for f_id, f_ in enumerate(f):
        P_in = [planck_function(f_, 
                                     [t[0] for t in T1], 
                                     [t[1] for t in T1]), 
                planck_function(f_, 
                                     [t[0] for t in T2], 
                                     [t[1] for t in T2])] # input noise powers
        a = np.asarray([[1, P_in[0]*bw], [1, P_in[1]*bw]])
        b = P_meas[:, f_id].T
        #print (f_id, a.shape, b.shape)
        solution = np.linalg.solve(a, b)
        #print (solution)
        #if (f_id<10):
            #print (a,b, solution)
        GkTNbw, G = solution
        TN = GkTNbw/(Boltzmann*G*bw)
        G_[f_id] = G
        TN_[f_id] = TN
    return G_, TN_