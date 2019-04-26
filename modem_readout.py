from . import sweep
from . import save_pkl
from . import fitting
import numpy as np
from qsweepy import data_reduce

import matplotlib.pyplot as plt

# Several carriers for readout??
# data_reduce.data_reduce does all the stuff but still...
# This class doesn't know about the excitation tone and can be used for any systems
class modem_readout(data_reduce.data_reduce):
### Frequency multiplexing: 
### Implements measure(), get_points(), get_opts() and  so on.
### 1) Calibrates f*** S+ and S- channels into I and Q by measureing how how in-phase and out-phase signals on DAC look on ADC and zero-level signal
### 2) Uses data_reduce.data_reduce for I, Q on carrier frequencies.
### 3) Uses data_reduce.data_reduce for downsampling.
### 4) Doesn't perform classification.
### 5) 
	# trigger sequence:
	#pg.p('ro_trg', trg_length, pg.rect, 1), 
	
	def __init__(self, pulse_sequencer, adc, trigger_daq_seq, src_meas='Voltage', axis_mean = 0, trigger_delay=0):
		self.pulse_sequencer = pulse_sequencer
		self.adc = adc
		self.src_meas = src_meas 
		self.axis_mean = axis_mean
		self.trigger_daq_seq = trigger_daq_seq
		self.trigger_delay = trigger_delay
		
		self.readout_channels = {}
		self.delay_calibrations = {}
		super().__init__(adc) # modem_readout is a 
	# random pulse sequence for wideband calibration of everything
	# shot noise with characterisitic timescale 30 time slots should be like 640K of RAM from microsoft
	def random_alignment_sequence(self,ex_channel, shot_noise_time=30):
		from scipy.signal import resample
		sequence_length = int(np.ceil(ex_channel.get_nop()/2))
		dac_sequence = []
		while len(dac_sequence) < sequence_length:
			dac_sequence.extend([1]*np.random.randint(shot_noise_time)+[-1]*np.random.randint(shot_noise_time))
		dac_sequence = np.asarray(dac_sequence[:sequence_length])
		dac_sequence_adc_time = resample(dac_sequence, int(np.ceil(sequence_length*self.adc.get_clock()/ex_channel.get_clock())))

		return dac_sequence, dac_sequence_adc_time
	
	def demodulation(self, ex_channel, sign=True):
		readout_time_axis = self.adc.get_points()[self.src_meas][1-self.axis_mean][1]
		if hasattr(ex_channel, 'get_if'):	# has get_if method => works on carrier, need to demodulate
			demodulation = np.exp(1j*np.asarray(readout_time_axis)*2*np.pi*ex_channel.get_if()*(1 if sign else -1))
		else:
			demodulation = np.ones(len(readout_time_axis)) #otherwise demodulation with unity (multiply by one)
		return demodulation
	
	def calibrate_delay(self, save=True):
		from scipy.signal import correlate
		# delay is calibrated on all lines (we probably don't really need that, but whatever)
		readout_delays = {}
		for ex_channel_name, ex_channel in self.readout_channels.items():
			# delay calibration pulse sequence
			dac_sequence, dac_sequence_adc_time = self.random_alignment_sequence(ex_channel)
			
			# set sequence in sequencer
			seq = self.trigger_daq_seq+[self.pulse_sequencer.p(ex_channel_name, len(dac_sequence)/ex_channel.get_clock(), self.pulse_sequencer.awg, dac_sequence)]
			self.pulse_sequencer.set_seq(seq)
			# readout
			adc_sequence = np.mean(self.adc.measure()[self.src_meas], axis=self.axis_mean)
			# demodulate
			demodulation = self.demodulation(ex_channel, sign=True)
			# depending on how the cables are plugged in, measure
			xc1 = correlate(adc_sequence*np.conj(demodulation), dac_sequence_adc_time, mode='full') # magic is here
			xc2 = correlate(adc_sequence*np.conj(demodulation), dac_sequence_adc_time, mode='full') # magic is here
			xc3 = correlate(np.conj(adc_sequence)*demodulation, dac_sequence_adc_time, mode='full') # magic is here
			xc4 = correlate(np.conj(adc_sequence)*demodulation, dac_sequence_adc_time, mode='full') # magic is here
			abs_xc = np.abs(xc1)+np.abs(xc2)+np.abs(xc3)+np.abs(xc4)
			# maximum correlation:
			readout_delays[ex_channel] = -(np.argmax(abs_xc)-len(dac_sequence_adc_time))/self.adc.get_clock() # get delay time in absolute units
			# plt.plot(xc1)
			# plt.plot(xc2)
			# plt.plot(xc3)
			# plt.plot(xc4)
			# plt.plot(abs_xc)
			# plt.plot(np.real(adc_sequence))
			# plt.plot(np.imag(adc_sequence))
			# plt.plot(dac_sequence_adc_time)
			if save:
				self.delay_calibrations[ex_channel_name] = readout_delays[ex_channel]
		return readout_delays
	
	def calibrate_dc(self, amplitude=1.0, save=True):
		# send nothing
		self.pulse_sequencer.set_seq(self.trigger_daq_seq)
		bg = np.mean(self.adc.measure()[self.src_meas], axis=self.axis_mean)
		self.bg = bg
		iq_readout_calibrations = {}
		calibrated_filters = {}
		# send I and Q pulses 
		for ex_channel_name, ex_channel in self.readout_channels.items():
			# delay calibration pulse sequence
			dac_sequence, dac_sequence_adc_time = self.random_alignment_sequence(ex_channel)
			dac_sequence=np.asarray(dac_sequence,dtype=np.float)*amplitude
			demodulation = self.demodulation(ex_channel, sign=True)
			# set sequence in sequencer with amplitude & phase
			seq_I = self.trigger_daq_seq+[self.pulse_sequencer.p(ex_channel_name, len(dac_sequence)/ex_channel.get_clock(), self.pulse_sequencer.awg, dac_sequence)]
			#seq_Q = self.trigger_daq_seq+[self.pulse_sequencer.p(ex_channel_name, sequence_length/ex_channel.get_clock(), self.pulse_sequencer.awg, 1j*dac_sequence)]
			# measure response on I sequence
			self.pulse_sequencer.set_seq(seq_I)
			calibration_measurement = self.adc.measure()
			meas_I = (np.mean(calibration_measurement[self.src_meas], axis=self.axis_mean) - bg)[:len(dac_sequence_adc_time)]
			# measure response on Q sequence
			#self.pulse_sequencer.set_seq(seq_Q)
			#meas_Q = np.mean(self.adc.measure()[self.src_meas], axis=self.axis_mean)		
			# solving equation: response(channel, time, complex) * sum(over demodulation_sign){demodulation(time, demodulation_sign, complex)*calibration(demodulation_sign, channel, complex)} = probe(channel, time, real)
			# In Ax=b form: A(time,demodulation_sign, channel)*x(demodulation_sign, channel)=b(time, channel), (response(channel, time, complex)*demodulation(time, demodulation_sign, complex))*calibration(emodulation_sign, channel, complex)=probe(time, channel)
			A_I = (meas_I*np.asarray([demodulation[:len(dac_sequence_adc_time)], np.conj(demodulation[:len(dac_sequence_adc_time)])])).T
			#A_Q = meas_Q*np.asarray([demodulation, np.conj(demodulation)])
			b_I = dac_sequence_adc_time
			#print (A_I.shape, b_I.shape)
			print (A_I.shape, b_I.shape)
			iq_calibration = np.linalg.lstsq(A_I, b_I)[0]
			feature = demodulation*iq_calibration[0]+np.conj(demodulation)*iq_calibration[1]
			iq_readout_calibrations[ex_channel_name] = {'iq_calibration': iq_calibration,
												   'coherent_background': self.bg,
												   'feature': feature}
			calibrated_filters[ex_channel_name] = data_reduce.feature_reducer(self.adc, self.src_meas, 1-self.axis_mean, self.bg, feature)
			
			del calibration_measurement, meas_I, A_I, b_I, seq_I
		if save:
			self.iq_readout_calibrations = iq_readout_calibrations
			self.calibrated_filters = calibrated_filters
		return iq_readout_calibrations
		
			# real parts of "calibration" correspond to response to real sources,
			# imaginary part of "calibration" corresponds to response to imaginary sources.
			
			#b_Q = dac_sequence*1j
			#np.dot(dac_sequence_adc_time, meas_I)
			#np.dot(dac_sequence_adc_time, meas_Q)
	# adc_reducer = data_reduce.data_reduce(adc)
	# adc_reducer.filters['Mean Voltage (AC)'] = data_reduce.mean_reducer_noavg(adc, 'Voltage', 0)
	# # adc_reducer.filters['Std Voltage (AC)'] = data_reduce.mean_reducer_noavg(adc, 'Voltage std', 0)
	# adc_reducer.filters['S21+'] = data_reduce.mean_reducer_freq(adc, 'Voltage', 0, iq_ro.get_if())
	# adc_reducer.filters['S21-'] = data_reduce.mean_reducer_freq(adc, 'Voltage', 0, -iq_ro.get_if())
	# adc_downsampler = data_reduce.data_reduce(adc)
	# adc_downsampler.filters['Voltage'] = data_reduce.downsample_reducer(adc, 'Voltage', 0, iq_ro.get_if(), 4)
	# # Этот измеритель мы как правило используем когда точек не слишком много и все результаты его жизнедеятельности как правило 
	# # выглядят как ломаные. Чтобы было красиво, давайте лучше сделаем точки (а кривые потом получим фитованые)
	# adc_reducer.extra_opts['scatter'] = True

	# lo1.set_frequency(lo_freq)
	# ex_if = lo_freq-qubit_params[qubit_id]['F00-1-01']
	# iq_ex = awg_iq_multi.awg_iq_multi(awg_tek, awg_tek, 0, 1, lo_ex)
	# # iq_ex = awg_iq.awg_iq(awg_tek, awg_tek, 2, 1, lo_ex) calibrated
	# iq_ro = awg_iq.awg_iq(awg_tek, awg_tek, 2, 3, lo_ro)
	# iq_ex.carriers['00-1-01'] = awg_iq_multi.carrier(iq_ex)
	# # iq_ex.carriers['00-1-10'] = awg_iq_multi.carrier(iq_ex)
	# iq_ex.carriers['01-1-02'] = awg_iq_multi.carrier(iq_ex)
	# iq_ex.carriers['00-2-02'] = awg_iq_multi.carrier(iq_ex)
	# iq_ex.carriers['00-1-01'].set_frequency(qubit_params[qubit_id]['F00-1-01'])
	# # iq_ex.carriers['00-1-10'].set_frequency(qubit_params[qubit_id]['F00-1-10'])
	# iq_ex.carriers['01-1-02'].set_frequency(qubit_params[qubit_id]['F01-1-02'])
	# iq_ex.carriers['00-2-02'].set_frequency(qubit_params[qubit_id]['F00-2-02'])
	# iq_ro.set_if(ro_if)
	# iq_ro.set_sideband_id(-1)
	# iq_ro.set_frequency(qubit_params[qubit_id]['Fr'])


	# awg_channels = {'iq_ex_00-1-01':iq_ex.carriers['00-1-01'], 
	               # # 'iq_ex_00-1-10':iq_ex.carriers['00-1-10'], 
					# 'iq_ex_01-1-02':iq_ex.carriers['01-1-02'], 
					# 'iq_ex_00-2-02':iq_ex.carriers['00-2-02'], 
					# 'iq_ro':iq_ro, 
					# 'ro_trg':ro_trg}

	# pg = pulses.pulses(awg_channels)

	# def __init__(self, pulse_sequencer, readout_device, **kwargs):
		# self.pulse_sequencer = pulse_sequencer
		# self.readout_device = readout_device
		# # self.params = **kwargs
		
		# # Rabi freq depends on excitation pulse parameters
		# # If we want to save to config, we should save in 
		# # ex_pulse_params=>rabi_freq pair
		# # ex_pulse_params should be a serialization of a pulse params o_0
		
		# adc_reducer.filters['Std Voltage (AC)'] = data_reduce.std_reducer_noavg(adc, 'Voltage', 0, 1)
	# def set_pulse_amplitude(x):
		# awg_tek.set_nop(awg_tek.get_clock()/rep_rate)
		# pg.set_seq([pg.p('ro_trg', trg_length, pg.rect, 1), pg.p('iq_ro', ro_dac_length, pg.rect, x)])
		# awg_tek.run()
		# measurement = sweep.sweep(adc_reducer, (np.linspace(0, 0.3, 31), set_pulse_amplitude, 'Readout amplitude'), filename='Readout pulse passthrough')
		# del adc_reducer.filters['Std Voltage (AC)']