import numpy as np
import pandas as pd
from qsweepy.ponyfiles.data_structures import *

def data_abs(meas):
    for i in range(len(meas)):
        meas[i]=meas[i]-np.median(np.real(meas[i]))-1j*np.median(np.imag(meas[i]))
    return meas
	
def fq_coil(p,x):
    frb, Cc, EJ1, EJ2, EC, phi0,L = p
    return fq_r(x, frb, Cc, EJ1, EJ2, EC, phi0, L)/1e9
	
def fq_r (x, frb, Cc, EJ1, EJ2, EC, phi0,L): 
	return (fqb(x, EJ1, EJ2, EC, phi0, L)+frb)*0.5-(((fqb(x, EJ1, EJ2, EC, phi0, L)-frb)*0.5)**2+Cc**2*fqb(x, EJ1, EJ2, EC, phi0, L)*frb)**0.5*np.sign(frb-fqb(x, EJ1, EJ2, EC, phi0, L))

def fqb (x, EJ1, EJ2, EC, phi0, L): 
	return (8*EC)**0.5*((EJ1-EJ2)**2*np.sin(np.pi*x*L+phi0*np.pi)**2+(EJ1+EJ2)**2*np.cos(np.pi*x*L+phi0*np.pi)**2)**0.25

def find_freqs_abs(data,Y):
    arg=[np.argmax((np.abs(data.T)-np.mean(np.abs(data.T)))[:,i]) for i in range(len(data))]
    return Y[arg]
	
def find_freqs_min(data,Y):
    arg=[np.argmin(data[i]) for i in range(len(data))]
#     print(arg)
    return Y[arg]

def two_tone_spectrum_frequency_extract(meas, remove_outliers_ = True):
    x = meas.datasets['S-parameter'].parameters[0].values[:]
    y = meas.datasets['S-parameter'].parameters[1].values[:]
    z = meas.datasets['S-parameter'].data[:,:,0]
    metadata = meas.metadata

    if metadata['resonator_id']!='9':
        fit = data_abs(z)
        fit = find_freqs_abs(fit,y)
    if metadata['resonator_id']=='9':
        fit = z
        for i in range(len(fit)):
            fit[i]=fit[i]-np.median(np.real(fit[i]))-1j*np.median(np.imag(fit[i]))
        fit = find_freqs_min(np.imag(fit),y)
    if remove_outliers_:
        x, fit = remove_outliers(x, fit)
    
    return x, fit, metadata ### TODO: MHZ -> Hz??
	
	
def remove_outliers(x,y):
    d2_abs = np.abs(np.gradient(np.gradient(y)))
    d2_abs_av = np.mean(d2_abs)
    good_points= d2_abs<d2_abs_av*2
    return x[good_points], y[good_points]
	
def get_Adaptive_two_tone_sprectoscopy_measurement(exdir_db_inst, _id):
    try:
        meas = exdir_db_inst.select_measurement(measurement_type="Adaptive_two_tone_spectroscopy_2", 
                                metadata={'resonator_id':_id})
    except:
        meas = exdir_db_inst.select_measurement(measurement_type="Adaptive_two_tone_spectroscopy", 
                                metadata={'resonator_id':_id})
        
    p = exdir_db_inst.select_measurement(measurement_type="Adaptive_two_tone_spectroscopy_parameters", 
                            metadata={'qubit_id':_id}).metadata ## TODO: should be loaded by reference
    
    return meas, p

def get_Nondiag_two_tone_spectroscopy_measurement(exdir_db_inst, _id, qubit_id):
    coil_id = exdir_db_inst.select_measurement(measurement_type='nndac_coil_parameters', metadata={'qubit_id':qubit_id}).metadata['coil_id']
    meas = exdir_db_inst.select_measurement(measurement_type="Nondiag_two_tone_spectroscopy", 
                            metadata={'resonator_id':_id,'non_diag_coil':"NNDAC-"+coil_id})

    return meas

def get_Adaptive_two_tone_spectroscopy(exdir_db_inst, _id,remove_outliers_ = True):
    meas, p = get_Adaptive_two_tone_sprectoscopy_measurement(exdir_db_inst, _id)
    
    p0 = [float(p[key]) for key in ['frb', 'Cc', 'EJ1', 'EJ2', 'EC', 'phi0', 'L']]
    
    x, fit, metadata = two_tone_spectrum_frequency_extract(meas,remove_outliers_)
    fr_diag=fq_coil(p0,x)
    return x, (fit/1e3+fr_diag), metadata ### TODO: MHZ -> Hz??
	
def get_Nondiag_two_tone_spectroscopy(exdir_db_inst, _id, qubit_id,remove_outliers_ = True):
    meas = get_Nondiag_two_tone_spectroscopy_measurement(exdir_db_inst, _id, qubit_id)

    return two_tone_spectrum_frequency_extract(meas,remove_outliers_)
	
	
	
def load_spectrum_data_for_fit(exdir_db_inst, qubit_ids, qubit_coil_ids=None, full_qubit_list=None):
    if qubit_coil_ids is None:
        qubit_coil_ids = [i for i in qubit_ids]
    
    if full_qubit_list is None:
        full_qubit_list = [i for i in qubit_ids]
    
    coil_qubit = { qubit_id: exdir_db_inst.select_measurement(measurement_type='nndac_coil_parameters', 
                            metadata={'qubit_id':qubit_id}).metadata['coil_id'] for qubit_id in full_qubit_list }
    qubit_coil = { coil_id: qubit_id for qubit_id, coil_id in coil_qubit.items() }
    
    data = pd.DataFrame()

    # loading diagonal data
    for qubit_id in qubit_ids:
        if qubit_id not in qubit_coil_ids:
            continue
        x, y, metadata = get_Adaptive_two_tone_spectroscopy(exdir_db_inst, qubit_id)

        V = pd.DataFrame(np.zeros((len(y), len(full_qubit_list))), columns=full_qubit_list)
        V[qubit_id] = x
        coilL = metadata['coilL']
        coilR = metadata['coilR']
        if coilL != 'False' and qubit_coil[coilL[6:]] in full_qubit_list:
            V[qubit_coil[coilL[6:]]] = float(metadata['voltL'])
        if coilR != 'False' and qubit_coil[coilR[6:]] in full_qubit_list:
            V[qubit_coil[coilR[6:]]] = float(metadata['voltR'])
        V['qubit_id'] = int(qubit_id)
        V['f'] = y
        data = pd.concat([data, V])
        
    for qubit_id  in qubit_ids:
        for qubit_coil_id in qubit_coil_ids:
            try:
                x,y, metadata = get_Nondiag_two_tone_spectroscopy(exdir_db_inst, qubit_id, qubit_coil_id)
                y=y/1e9
            except IndexError as e:
                continue

            V = pd.DataFrame(np.zeros((len(y), len(full_qubit_list))), columns=full_qubit_list)
            V[qubit_coil[metadata['non_diag_coil'][6:]]] = x
            V[qubit_coil[metadata['coil'][6:]]] = float(metadata['volt_coil'])
            
            V['qubit_id'] = int(qubit_id)
            V['f'] = y
            data = pd.concat([data, V])
    return data

fqbare = lambda EJ1, EJ2, EC,flux: (8*EC)**0.5*((EJ1-EJ2)**2*np.sin(np.pi*flux)**2+\
                                                    (EJ1+EJ2)**2*np.cos(np.pi*flux)**2)**0.25-EC

def build_inductance_matrix(parameters):
    qubits = parameters['qubits']
    num_qubits = len(qubits)    
    if parameters['inductance_matrix_type'] == 'chain-nn':
        inductance_matrix = np.zeros((num_qubits, num_qubits))
        for qubit_id_, qubit_id in enumerate(qubits):
            if qubit_id_ > 0:
                inductance_matrix[qubit_id_, qubit_id_-1] = parameters['inductances'][qubit_id]['left']
            inductance_matrix[qubit_id_, qubit_id_] = parameters['inductances'][qubit_id]['central']
            if qubit_id_ < num_qubits-1:
                inductance_matrix[qubit_id_, qubit_id_+1] = parameters['inductances'][qubit_id]['right']
    return inductance_matrix

def model(parameters, spectra, print_=False):
    
    qubits = parameters['qubits']
    num_qubits = len(qubits)    
    
    resonator_freqs = [parameters['fr'][qubit_id] for qubit_id in qubits]
    inductance_matrix = build_inductance_matrix(parameters)
    
    frequencies = []
    
    for s_id, data in spectra.iterrows():
        try:
            induced_flux = inductance_matrix@np.asarray(data[qubits].tolist())
        except:
            print ('inductance_matrix:', inductance_matrix)
            print ('data[qubits].tolist():', data[qubits].tolist())
            raise
            
        #print (induced_flux)
        
        qubit_freqs = [fqbare(parameters['EJ1'][qubit_id], 
                              parameters['EJ2'][qubit_id], 
                              parameters['EC'], 
                              induced_flux[qubit_id_]+parameters['phi0'][qubit_id]) 
                           for qubit_id_, qubit_id in enumerate(qubits)]
        
        #print (qubit_freqs)
        
        linear_oscillator_matrix = np.diag(resonator_freqs+qubit_freqs)
        
        # equal qubit-resonator coupling
        if parameters['qubit_resonator_individual'] == 'equal_claws':
            for qubit_id_, qubit_id in enumerate(qubits):
                linear_oscillator_matrix[qubit_id_, qubit_id_+num_qubits] = \
                    parameters['g']*np.sqrt(qubit_freqs[qubit_id_])
                linear_oscillator_matrix[qubit_id_+num_qubits, qubit_id_] = \
                    parameters['g']*np.sqrt(qubit_freqs[qubit_id_])
        
        # qubit-qubit coupling
        if parameters['qubit_qubit_coupling'] == 'alternating-chain-nn':
            for qubit_id_, qubit_id in enumerate(qubits):
                if (qubit_id_ % 2):
                    Jl, Jr = parameters['J1'], parameters['J2']
                else:
                    Jl, Jr = parameters['J2'], parameters['J1']
                    
                if qubit_id_ > 0:
                    linear_oscillator_matrix[num_qubits+qubit_id_, num_qubits+qubit_id_-1] = \
                        Jl*np.sqrt(qubit_freqs[qubit_id_]*qubit_freqs[qubit_id_-1])
                if qubit_id_ < num_qubits-1:
                    linear_oscillator_matrix[num_qubits+qubit_id_, num_qubits+qubit_id_+1] = \
                        Jr*np.sqrt(qubit_freqs[qubit_id_]*qubit_freqs[qubit_id_+1])
#         print(linear_oscillator_matrix)
        #print ('linear_oscillator_matrix:', linear_oscillator_matrix)
        qubit_freqs[int(qubit_id)-1]
        try:
            w, v = np.linalg.eigh(linear_oscillator_matrix/1e9)
            #print ('w:', w)
            #print ('v:', v)
            participations = np.abs(v)**2
            qubit_like_mode_id = np.argmax(participations[num_qubits+int(data['qubit_id'])-1,:])
            #print (qubit_like_mode_id )
            frequencies.append(w[qubit_like_mode_id])
            if print_:
                print (w)
        except Exception as e:
            #print (e)
            #print (linear_oscillator_matrix/1e9)
            frequencies.append(qubit_freqs[int(qubit_id)-1])

    return frequencies

def save_parameters_dict(exdir_db_inst, parameters_dict):    
    metadata = {'inductance_matrix_type': parameters_dict['inductance_matrix_type'],
                'qubit_resonator_individual': parameters_dict['qubit_resonator_individual'],
                'qubit_qubit_coupling': parameters_dict['qubit_qubit_coupling']}
    metadata['EC'] = parameters_dict['EC']
    metadata['g'] = parameters_dict['g']
    metadata['J1'] = parameters_dict['J1']
    metadata['J2'] = parameters_dict['J2']
    
    qubits = parameters_dict['qubits']
    qubit_id_parameter = MeasurementParameter(name='qubit_id', values=[int(qubit_id) for qubit_id in qubits], 
                                              setter=False, unit='')
    coil_id_parameter = MeasurementParameter(name='coil_id', values=[int(qubit_id) for qubit_id in qubits], 
                                              setter=False, unit='')

    EJ1 = MeasurementDataset(parameters=[qubit_id_parameter], 
                             data=np.asarray([parameters_dict['EJ1'][str(qubit_id)] for qubit_id in qubit_id_parameter.values]))
    EJ2 = MeasurementDataset(parameters=[qubit_id_parameter], 
                             data=np.asarray([parameters_dict['EJ2'][str(qubit_id)] for qubit_id in qubit_id_parameter.values]))
    fr  = MeasurementDataset(parameters=[qubit_id_parameter], 
                             data=np.asarray([parameters_dict['fr'][str(qubit_id)] for qubit_id in qubit_id_parameter.values]))
    phi0= MeasurementDataset(parameters=[qubit_id_parameter], 
                             data=np.asarray([parameters_dict['phi0'][str(qubit_id)] for qubit_id in qubit_id_parameter.values]))
    L   = MeasurementDataset(parameters=[coil_id_parameter, qubit_id_parameter], data=build_inductance_matrix(parameters_dict))
    #dataset = MeasurementDataset?

    m = MeasurementState(datasets={'EJ1':EJ1, 'EJ2':EJ2, 'fr':fr, 'phi0':phi0, 'L':L}, 
                         metadata=metadata,
                         measurement_type='linear_oscillator_model_parameters',
                         sample_name=exdir_db_inst.sample_name)
    exdir_db_inst.save_measurement(m)
    return m

def load_parameters_dict(exdir_db_inst):    
    meas = exdir_db_inst.select_measurement(measurement_type='linear_oscillator_model_parameters')
    
    parameters_dict = {}
    parameters_dict.update(meas.metadata)
    
    L = meas.datasets['L']
    
    qubits = [str(qubit_id) for qubit_id in meas.datasets['EJ1'].parameters[0].values]
    parameters_dict['EJ1'] = {}
    parameters_dict['EJ2'] = {}
    parameters_dict['fr'] = {}
    parameters_dict['phi0'] = {}
    parameters_dict['inductances'] = {}
    parameters_dict['J1'] = float(parameters_dict['J1'])
    parameters_dict['J2'] = float(parameters_dict['J2'])
    parameters_dict['EC'] = float(parameters_dict['EC'])
    parameters_dict['g'] = float(parameters_dict['g'])
    
    for qubit_id_, qubit_id in enumerate(qubits):
        parameters_dict['EJ1'][qubit_id] = float(meas.datasets['EJ1'].data[qubit_id_])
        parameters_dict['EJ2'][qubit_id] = float(meas.datasets['EJ2'].data[qubit_id_])
        parameters_dict['fr'][qubit_id] = float(meas.datasets['fr'].data[qubit_id_])
        parameters_dict['phi0'][qubit_id] = float(meas.datasets['phi0'].data[qubit_id_])
        parameters_dict['inductances'][qubit_id] = {'central': float(L.data[qubit_id_, qubit_id_])}
        if qubit_id_ > 0:
            parameters_dict['inductances'][qubit_id]['left'] = float(L.data[qubit_id_, qubit_id_-1])
        if qubit_id_ < len(qubits)-1:
            parameters_dict['inductances'][qubit_id]['right'] = float(L.data[qubit_id_, qubit_id_+1])
        
    return parameters_dict

def build_p0_dict(qubits):
    L =   {'1':-0.11182, '2': -0.11217, '3': -0.11249, '4': -0.11300, '5': -0.11103, '6': -0.10783, 
           '7':-0.10364, '8': -0.11033, '9': -0.11038, '10': -0.10766, '11': -0.11556}
    EJ1 = {'1': 11.816e9, '2': 11.168e9, '3': 11.815e9, '4': 11.263e9, '5': 10.981e9, '6': 10.909e9,
           '7': 11.666e9, '8': 10.903e9, '9': 11.563e9, '10': 11.045e9, '11': 11.265e9}
    EJ2 = {'1': 1.9435e9, '2': 1.6731e9, '3': 1.9114e9, '4': 1.7694e9, '5': 1.8107e9, '6': 1.6079e9,
           '7': 1.9888e9, '8': 2.1891e9, '9': 2.0010e9, '10': 1.9410e9, '11': 1.9456e9}
    phi0 = {'1': 0.0796, '2': 0.0869, '3': 0.0828, '4': 0.0845, '5': 0.0861, '6': 0.0919,
           '7': 0.0759, '8': 0.0815, '9': 0.0758, '10': 0.0703, '11': 0.0813}
    parameters = {
    'qubits': [qubit_id for qubit_id in qubits.keys()],
    'fr': {qubit_id: qubit['r']['Fr'] for qubit_id, qubit in qubits.items()},
    'EJ1': EJ1,
    'EJ2': EJ2,
#      'EC': 155e6,
    'phi0':phi0,
    'J1':0.015,#0.0129,
    'J2':0.004,#0.0033,
    'g':0.008*np.sqrt(6.6e9),
    'inductances':{qubit_id:{'central': L[qubit_id], 'left':1e-3, 'right':1e-3} for qubit_id in qubits.keys()},
    'inductance_matrix_type':'chain-nn',
    'qubit_resonator_individual': 'equal_claws',
    'qubit_qubit_coupling': 'alternating-chain-nn',             
                 }
    return parameters

def build_bounds_from_parameters_dict(parameters):
    num_qubits = len(parameters['qubits'])
    qubit_ids = parameters['qubits']
    
    podgon_low = []
    #podgon.extend([parameters['fr'][qubit_id]/1e9 for qubit_id in qubit_ids])
    podgon_low.extend([parameters['EJ1'][qubit_id]/1e9/2 for qubit_id in qubit_ids])
    podgon_low.extend([parameters['EJ2'][qubit_id]/1e9/2 for qubit_id in qubit_ids])
    podgon_low.extend([parameters['phi0'][qubit_id]/2 for qubit_id in qubit_ids])
    
    
    podgon_low.extend([-np.abs(parameters['inductances'][qubit_id]['central'])*2 for qubit_id in qubit_ids])
    podgon_low.extend([-np.abs(parameters['inductances'][qubit_id]['right'])*5 \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ < num_qubits-1])
    podgon_low.extend([-np.abs(parameters['inductances'][qubit_id]['left'])*5 \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ > 0])
    
    #podgon_low.append(parameters['EC']/1e9/2)
    podgon_low.append(parameters['g']/2)
    podgon_low.append(parameters['J1']/1.5)
    podgon_low.append(parameters['J2']/1.5)
    
    
    podgon_up = []
    #podgon.extend([parameters['fr'][qubit_id]/1e9 for qubit_id in qubit_ids])
    podgon_up.extend([parameters['EJ1'][qubit_id]/1e9*2 for qubit_id in qubit_ids])
    podgon_up.extend([parameters['EJ2'][qubit_id]/1e9*2 for qubit_id in qubit_ids])
    podgon_up.extend([parameters['phi0'][qubit_id]*2 for qubit_id in qubit_ids])
    
    podgon_up.extend([np.abs(parameters['inductances'][qubit_id]['central'])*2 for qubit_id in qubit_ids])
    podgon_up.extend([np.abs(parameters['inductances'][qubit_id]['right'])*5 \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ < num_qubits-1])
    podgon_up.extend([np.abs(parameters['inductances'][qubit_id]['left'])*5 \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ > 0])
    
    #podgon_up.append(parameters['EC']/1e9*2)
    podgon_up.append(parameters['g']*2)
    podgon_up.append(parameters['J1']*1.5)
    podgon_up.append(parameters['J2']*1.5)
    
    
    return tuple([tuple(podgon_low),tuple(podgon_up)])

def build_podgon_list_from_parameters_dict(parameters):
    num_qubits = len(parameters['qubits'])
    qubit_ids = parameters['qubits']
    
    podgon = []
    #podgon.extend([parameters['fr'][qubit_id]/1e9 for qubit_id in qubit_ids])
    podgon.extend([parameters['EJ1'][qubit_id]/1e9 for qubit_id in qubit_ids])
    podgon.extend([parameters['EJ2'][qubit_id]/1e9 for qubit_id in qubit_ids])
    podgon.extend([parameters['phi0'][qubit_id] for qubit_id in qubit_ids])
    
    podgon.extend([parameters['inductances'][qubit_id]['central'] for qubit_id in qubit_ids])
    podgon.extend([parameters['inductances'][qubit_id]['right'] \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ < num_qubits-1])
    podgon.extend([parameters['inductances'][qubit_id]['left'] \
                       for qubit_id_, qubit_id in enumerate(qubit_ids) if qubit_id_ > 0])
    
    #podgon.append(parameters['EC']/1e9)
    podgon.append(parameters['g'])
    podgon.append(parameters['J1'])
    podgon.append(parameters['J2'])
    
    return podgon

def build_parameters_dict_from_podgon_list(p,parameters_fixed, qubit_ids):
    num_qubits = len(qubit_ids)
    
    #print (p)
    
    #fr   = p[:num_qubits]
    #p = p[num_qubits:]
    EJ1  = p[:num_qubits]
    p = p[num_qubits:]
    EJ2  = p[:num_qubits]
    p = p[num_qubits:]
    phi0 = p[:num_qubits]
    p = p[num_qubits:]
    
    L = p[:(3*num_qubits-2)]
    p = p[(3*num_qubits-2):]
    #EC   = p[0]
    g    = p[0]
    J1   = p[1]
    J2   = p[2]
    
    inductances = {}
    for qubit_id_, qubit_id in enumerate(qubit_ids):
        inductances[qubit_id] = {'central':L[qubit_id_]}
        if qubit_id_ < num_qubits-1:
            inductances[qubit_id]['right'] = L[num_qubits+qubit_id_]
        if qubit_id_ > 0:
            inductances[qubit_id]['left'] = L[2*num_qubits-2+qubit_id_]
            
    ## inductances are saved as [L_11, L_22, ... L_nn, L_12, L_23, L_34, ..., L_{n-1},n, L_21, L_32, ..., L_n,{n-1}]
    
    parameters = {'qubits':qubit_ids,
              'inductance_matrix_type':'chain-nn',
              'qubit_resonator_individual': 'equal_claws',
              'qubit_qubit_coupling': 'alternating-chain-nn',
              'inductances':inductances,
              #'fr': {qubit_id: fr_*1e9 for qubit_id, fr_ in zip(qubit_ids, fr)},
              'EJ1': {qubit_id: EJ1_*1e9 for qubit_id, EJ1_ in zip(qubit_ids, EJ1)},
              'EJ2': {qubit_id: EJ2_*1e9 for qubit_id, EJ2_ in zip(qubit_ids, EJ2)},
              'phi0': {qubit_id: phi0_ for qubit_id, phi0_ in zip(qubit_ids, phi0)},
              #'EC': EC*1e9,
              'g': g,
              'J1': J1,
              'J2': J2}
    parameters.update(parameters_fixed)
    return parameters