from . import data_reduce
import numpy as np
from . import readout_classifier

class single_shot_readout:
    """
    Single shot readout class

    Args:
		adc (Instrument): a device that measures a complex vector for each readout trigger (an ADC)
        ex_seq (dict of pulses.sequence): a dict of sequences of control pulses. The keys are use for state identification.
        ro_seq (pulses.sequence): a sequence of control pulses that is used to generate the reaout pulse of the DAC.
        pulse_generator (pulses.pulse_generator): pulse generator used to concatenate and set waveform sequences on the DAC.
		ro_delay_seq (pulses.sequence): Sequence used to align the DAC and ADC (readout delay compensation)
		adc_measurement_name (str): name of measurement on ADC
    """
	def __init__(self, adc, ex_seq, ro_seq, pulse_generator, ro_delay_seq = None, _readout_classifier = readout_classifier.linear_classifier(), adc_measurement_name='Voltage'):
		self.adc = adc
		self.ro_seq = ro_seq
		self.ex_seq = ex_seq
		
		self.ro_delay_seq = ro_delay_seq
		self.pulse_generator = pulse_generator
		self.repeat_samples = 2
		self.save_last_samples = False
		self.train_test_split = 0.8
		self.measurement_name = ''
		self.dump_measured_samples = False
		#self.cutoff_start = 0
		self.readout_classifier = _readout_classifier
		self.adc_measurement_name = adc_measurement_name
	
	def measure_delay(self, ro_channel):
		import matplotlib.pyplot as plt
		from scipy.signal import resample
		
		self.pulse_generator.set_seq(self.ro_delay_seq)
		first_nonzero = int(np.nonzero(np.abs(self.pulse_generator.channels[ro_channel].get_waveform()))[0][0]/self.pulse_generator.channels[ro_channel].get_clock()*self.adc.get_clock())
		ro_dac_waveform = self.pulse_generator.channels[ro_channel].awg_I.get_waveform(channel=self.pulse_generator.channels[ro_channel].awg_ch_I)+\
					   1j*self.pulse_generator.channels[ro_channel].awg_Q.get_waveform(channel=self.pulse_generator.channels[ro_channel].awg_ch_Q)
		ro_dac_waveform = resample(ro_dac_waveform, num=int(len(ro_dac_waveform)/self.pulse_generator.channels[ro_channel].get_clock()*self.adc.get_clock()))
		ro_adc_waveform = np.mean(self.adc.measure()['Voltage'], axis=0)
		ro_dac_waveform = ro_dac_waveform - np.mean(ro_dac_waveform)
		ro_adc_waveform = ro_adc_waveform - np.mean(ro_adc_waveform)
		xc = np.abs(np.correlate(ro_dac_waveform, ro_adc_waveform, 'same'))
		xc_max = np.argmax(xc)
		delay = int((xc_max - first_nonzero)/2)
		#plt.figure('delay')
		#plt.plot(ro_dac_waveform[first_nonzero:])
		#plt.plot(ro_adc_waveform[delay:])
		#plt.plot(ro_adc_waveform)
		#print ('Measured delay is {} samples'.format(delay), first_nonzero, xc_max)
		return delay
	
	def calibrate(self):
		X = []
		y = []
		for class_id, prepare_seq in enumerate(self.prepare_seqs):
			for i in range(self.repeat_samples):
				# pulse sequence to prepare state
				self.pulse_generator.set_seq(prepare_seq+self.ro_seq)
				measurement = self.adc.measure()
				if type(self.adc_measurement_name) is list:
					raise ValueError('Multiqubit readout not implemented') #need multiqubit readdout implementation
				else:
					X.append(measurement[self.adc_measurement_name])
				y.extend([class_id]*len(self.adc.get_points()[self.adc_measurement_name][0][1]))
		X = np.reshape(X, (-1, len(self.adc.get_points()[self.adc_measurement_name][-1][1]))) # last dimension is the feature dimension
		y = np.asarray(y)
		if self.dump_measured_samples:
			self.samples = np.reshape(X, (len(self.prepare_seqs), -1, len(self.adc.get_points()[self.adc_measurement_name][-1][1])))
		
		self.predictor_class = readout_classifier.linear_classifier()
		scores = readout_classifier.evaluate_classifier(self.predictor_class, X, y)
		self.predictor_class.fit(X, y)
		self.scores = scores
		
	def get_opts(self):
		scores = readout_classifier.readout_classifier_scores(lambda X, y: 0, [[0]], [0])
		return {score_name:{'log':False} for score_name in scores}
	
	def measure(self):
		self.calibrate()
		meas = self.scores
		if self.dump_measured_samples:
			self.dump_samples(name=self.measurement_name)
		return meas
		
	def get_points(self):
		scores = readout_classifier.readout_classifier_scores(lambda X, y: 0, [[0]], [0])
		return {score_name:{}}
				
	def get_dtype(self):
		scores = readout_classifier.readout_classifier_scores(lambda X, y: 0, [[0]], [0])
		return {score_name:float}
	
	def dump_samples(self, name):
		from .save_pkl import save_pkl
		header = {'type':'Readout classification samples', 'name':name}
		measurement = {'Readout classification samples':(['Class', 'Sample ID', 'time'], 
				[np.arange(self.samples), np.arange(self.samples.shape[1]), np.arange(self.samples.shape[2])/self.adc.get_clock()],
				self.samples)}
		save_pkl(header, self.samples, plot=False)
	