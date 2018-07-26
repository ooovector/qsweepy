from . import sweep
from . import save_pkl
from . import fitting
import numpy as np


def Rabi(ex_channels,ex_amplitude,lengths,ro_par,qubit_id=None):
	rabi_freqs = list()
	trg_length = ro_par['trg_length']
	ro_dac_length = ro_par['ro_dac_length']
	ro_amplitude = ro_par['ro_amplitude']
	dac = ro_par['dac']
	adc = ro_par['adc']
	pg = ro_par['pulse gen']
	def set_ex_length(length):
			sequence = [pg.p(None, readout_begin-length), 
					pg.p(rabi_channel, length, pg.rect, ex_amplitude), 
					pg.p('ro_trg', trg_length, pg.rect, 1), 
					pg.p('iq_ro', ro_dac_length, pg.rect, ro_amplitude)]
			pg.set_seq(sequence)
			dac.run()
	for rabi_channel in ex_channels:
		measurement_name = 'Rabi channel {}'.format(rabi_channel)
		readout_begin = np.max(lengths)
		measurement = sweep.sweep(adc, (lengths, set_ex_length, 'Rabi pulse length', 's'), filename=measurement_name)
		measurement_fitted, fitted_parameters = fitting.S21pm_fit(measurement, fitting.exp_sin_fit)
		rabi_freqs.append(fitted_parameters['freq'])
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																				 fitted_parameters['freq'], 
																				 fitted_parameters['decay'])
		save_pkl.save_pkl({'type':'Rabi','name': 'qubit{}'.format(qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name)
	return rabi_freqs

	
def Ramsey(ex_channels,ex_amplitude,delays,target_freq_offset,rabi_freqs,ro_par,qubit_id=None):
	offset_freqs = list()
	t2 = list()
	trg_length = ro_par['trg_length']
	ro_dac_length = ro_par['ro_dac_length']
	ro_amplitude = ro_par['ro_amplitude']
	dac = ro_par['dac']
	adc = ro_par['adc']
	pg = ro_par['pulse gen']
	def set_delay(delay):
		sequence = [pg.p(None, readout_begin-pi2_pulse),
					pg.p(ex_channel, pi2_pulse, pg.rect, ex_amplitude), 
					pg.p(None, delay), 
					pg.p(ex_channel, pi2_pulse, pg.rect, ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi)), 
					pg.p('ro_trg', trg_length, pg.rect, 1), 
					pg.p('iq_ro', ro_dac_length, pg.rect, ro_amplitude)]
		pg.set_seq(sequence)
		dac.run()
	
	for ex_channel, rabi_freq in zip(ex_channels, rabi_freqs):
		measurement_name = 'Ramsey (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, ex_channel)
		pi2_pulse = 0.25/rabi_freq
		readout_begin = np.max(delays)+pi2_pulse*2
		measurement = sweep.sweep(adc, (delays, set_delay, 'Ramsey delay', 's'), filename=measurement_name)
		measurement_fitted, fitted_parameters = fitting.S21pm_fit(measurement, fitting.exp_sin_fit)
		offset_freqs.append(fitted_parameters['freq'])
		t2.append(fitted_parameters['decay'])
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
																				 fitted_parameters['freq'], 
																				 fitted_parameters['decay'])
		save_pkl.save_pkl({'type':'Ramsey', 'name': 'qubit {}'.format(qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name)
	return offset_freqs,t2
def T1(ex_channels,ex_amplitude,delays,rabi_freqs,ro_par,qubit_id=None):
	t1 = list()
	trg_length = ro_par['trg_length']
	ro_dac_length = ro_par['ro_dac_length']
	ro_amplitude = ro_par['ro_amplitude']
	dac = ro_par['dac']
	adc = ro_par['adc']
	pg = ro_par['pulse gen']
	def set_delay(delay):
			sequence = [pg.p(None, readout_begin-ro_dac_length),
                    pg.p(ex_channel, pi_pulse, pg.rect, ex_amplitude), 
                    pg.p(None, delay), 
                    pg.p('ro_trg', trg_length, pg.rect, 1), 
                    pg.p('iq_ro', ro_dac_length, pg.rect, ro_amplitude)]
			pg.set_seq(sequence)
			dac.run()
	for ex_channel, rabi_freq in zip(ex_channels, rabi_freqs):
		measurement_name = 'Decay, excitation channel {0}'.format(ex_channel)
		pi_pulse = 0.5/rabi_freq
		readout_begin = np.max(delays)+pi_pulse
		measurement = sweep.sweep(adc, (delays, set_delay, 'delay', 's'), filename=measurement_name)
		measurement_fitted, fitted_parameters = fitting.S21pm_fit(measurement, fitting.exp_fit)
		t1.append(fitted_parameters['decay'])
		annotation = 'Decay: {0:4.6g} s'.format(fitted_parameters['decay'])
		save_pkl.save_pkl({'type':'Decay', 'name': 'qubit {}'.format(qubit_id)}, measurement_fitted, annotation=annotation, filename=measurement_name)
	return t1
	
def SpinEcho(ex_channels,ex_amplitude,delays,target_freq_offset,rabi_freqs,ro_par,qubit_id=None):
	offset_freqs = list()
	t2_echo = list()
	trg_length = ro_par['trg_length']
	ro_dac_length = ro_par['ro_dac_length']
	ro_amplitude = ro_par['ro_amplitude']
	dac = ro_par['dac']
	adc = ro_par['adc']
	pg = ro_par['pulse gen']
	def set_delay(delay):    
		sequence = [pg.p(None, readout_begin-pi2_pulse),
                pg.p(ex_channel, pi2_pulse, pg.rect, ex_amplitude), 
                pg.p(None, delay), 
                pg.p(ex_channel, pi2_pulse*2, pg.rect, ex_amplitude), 
                pg.p(None, delay), 
                pg.p(ex_channel, pi2_pulse, pg.rect, ex_amplitude*np.exp(1j*delay*target_freq_offset*2*np.pi)), 
                pg.p('ro_trg', trg_length, pg.rect, 1), 
                pg.p('iq_ro', ro_dac_length, pg.rect, ro_amplitude)]
		pg.set_seq(sequence)
		dac.run()
	for ex_channel, rabi_freq in zip(ex_channels, rabi_freqs):
		measurement_name = 'Spin echo (target offset {0:4.2f} MHz), excitation channel {1}'.format(target_freq_offset/1e6, ex_channel)
		pi2_pulse = 0.25/rabi_freq
		readout_begin = np.max(delays)+pi2_pulse*2
		measurement = sweep.sweep(adc, (delays, set_delay, 'Spin echo full delay','s'), filename=measurement_name)
		measurement_fitted, fitted_parameters = fitting.S21pm_fit(measurement, fitting.exp_sin_fit)
		offset_freqs.append(fitted_parameters['freq'])
		t2_echo.append(fitted_parameters['decay'])
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}, Decay: {2:4.4g} s'.format(fitted_parameters['phase'], 
                                                                             fitted_parameters['freq'], 
                                                                             fitted_parameters['decay'])
		save_pkl.save_pkl({'type':'Spin echo', 'name': 'qubit {}'.format(qubit_id)}, measurement_fitted, annotation=annotation,filename=measurement_name)
	return offset_freqs,t2_echo