import numpy as np
from . import sweep
import time
import pickle


class paramp:
	def __init__(self, vna, pump_src, bias_src):
		self.vna = vna
		self.pump_src = pump_src
		self.bias_src = bias_src
		self.voltage_setter = None
		self.voltage_getter = None
		self.optimal_points = {}

		self.target_f = None
		self.target_bw = None
		self.target_power = None
		self.num_restarts = 1

		self.hint_fp_offset = 100e6
		self.hint_rel_errors = (1e-4, 1e-2, 1e-2)
		self.hint_maxfun = 200
		self.hint_nop = 1000
		self.hint_span_bw = 200.
		self.optimize_pump_frequency = True
		self.bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))

		self.pump_on = True

	def measure(self):
	# maximize SNR of S21 at given frequency, excitation power and bandwidth
		if hasattr(self.vna, 'pre_sweep'):
			self.vna.pre_sweep()
		self.vna.set_nop(self.hint_nop)
		self.vna.set_bandwidth(self.target_bw)
		self.vna.set_power(self.target_power)
		self.vna.set_centerfreq(self.target_f)
		self.vna.set_span(self.target_bw/self.hint_span_bw)

		params = [(self.pump_src.set_frequency, self.pump_src.get_frequency()),
				  (self.pump_src.set_power if not self.power_setter else self.power_setter,
				   self.pump_src.get_power()),
				  (self.bias_src.set_current if not self.voltage_setter else self.voltage_setter,
				   self.bias_src.get_current() if not self.voltage_setter else self.voltage_getter())]

		if not self.optimize_pump_frequency:
			params = params[1:]
			self.pump_src.set_frequency(self.target_f-self.target_bw*3)

		params = tuple(params)

		def create_initial_simplex(x0):
			if hasattr(self, 'hint_abs_errors'):
				initial_simplex = [
					[p[1] + self.hint_abs_errors[p_id] if v == p_id else p[1] for p_id, p in enumerate(x0)]
					for v in range(len(params) + 1)]
			else:
				initial_simplex = [
					[p[1] * (1 + self.hint_rel_errors[p_id]) if v == p_id else p[1] for p_id, p in enumerate(x0)]
					for v in range(len(params) + 1)]
			return initial_simplex

		self.pump_src.set_status(self.pump_on)
		if hasattr(self.bias_src, 'set_status'):
			self.bias_src.set_status(True)

		def target():
			time.sleep(0.2)
			measurements = self.vna.measure()['S-parameter'].ravel()
			print ('Gain, dB:', np.log10(np.abs(np.mean(measurements)))*20,
                   'SNR, dB:', np.log10(np.abs(np.mean(measurements))/np.std(measurements))*10)
			return np.std(measurements)/np.abs(np.mean(measurements))

		costs = {tuple([initial for setter, initial in params]): target()}
		for restart_id in range(self.num_restarts):
			initial_simplex = create_initial_simplex(params)
			res = sweep.optimize(target, *params, initial_simplex=initial_simplex, maxfun=self.hint_maxfun, bounds=self.bounds)
			for x, p in zip(res[0], params): p[0](x)
			costs[tuple(res[0])] = target()
			min_ = np.inf
			for x, y in costs.items():
				if y < min_:
					argmin = x
					params = [(p[0], v) for p, v in zip(params, argmin)]

		# iterate over all restarts and find best

		for x, p in zip(argmin, params): p[0](x)
		measurements = self.vna.measure()['S-parameter'].ravel()
		S21 = np.mean(measurements)
		measurement = {'S-parameter':np.asarray(S21),
					   'SNR':np.asarray(np.abs(np.mean(measurements))/np.std(measurements))}

		if self.optimize_pump_frequency:
			measurement['Pump_frequency'] = np.asarray(argmin[0])
			measurement['Pump_power'] = np.asarray(argmin[1])
			measurement['Bias'] = np.asarray(argmin[2])
		else:
			measurement['Pump_power'] = np.asarray(argmin[0])
			measurement['Bias'] = np.asarray(argmin[1])

		if hasattr(self.vna, 'post_sweep'):
			self.vna.post_sweep()
		return measurement

	def set_target_f(self, value):
		self.target_f = value

	def set_target_bw(self, value):
		self.target_bw = value

	def set_target_power(self, value):
		self.target_power = value

	def get_opts(self):
		res =  {'S-parameter':{'log': 20},
				'SNR':{'log': 20},
				'Pump_power':{'log': False},
				'Bias':{'log': False}}
		if self.optimize_pump_frequency:
			res['Pump_frequency'] = {'log': False}

	def get_points(self):
		res = {'S-parameter': [],
				'SNR': [],
				'Pump_power': [],
				'Bias': []}
		if self.optimize_pump_frequency:
			res['Pump_frequency'] = []
		return res

	def get_dtype(self):
		res =  {'S-parameter': np.complex,
				'SNR': np.float,
				'Pump_power': np.float,
				'Bias': np.float}
		if self.optimize_pump_frequency:
			res['Pump_frequency'] = np.float
		return res

	def set_parameters(self, f, power, bias):
		import time
		params = self.measure()
		pump_src.set_frequency(params['Pump frequency'])
		pump_src.set_power(params['Pump power'])
		if not self.voltage_setter:
			bias_src.set_current(params['Bias'])
		else:
			self.voltage_setter(params['Bias'])

	def load_calibration(self, calibration_measurement):
		import pickle
		from scipy.interpolate import interp1d
		#bias_name = 'Bias Paramp VNA calib'
		#pumpfreq_name = 'Pump frequency Paramp VNA calib'
		#pumppower_name = 'Pump power Paramp VNA calib'
		#calib_file = pickle.load(open('{0}\\{1}.pkl'.format(path, name), 'rb'))[1]
		self.calib_targetfreq = calibration_measurement.datasets['Bias'].parameters[0].values
		self.calib_bias = calibration_measurement.datasets['Bias'].data
		if 'Pump_frequency' in calibration_measurement.datasets:
			self.calib_pumpfreq = calibration_measurement.datasets['Pump_frequency'].data
		else:
			self.calib_pumpfreq = self.calib_targetfreq - 3*float(calibration_measurement.metadata['target_bw'])
		self.calib_pumppower = calibration_measurement.datasets['Pump_power'].data

		self.pump_frequency_by_target_frequency = interp1d(self.calib_targetfreq, self.calib_pumpfreq)
		self.pump_power_by_target_frequency = interp1d(self.calib_targetfreq, self.calib_pumppower)
		self.bias_by_target_frequency = interp1d(self.calib_targetfreq, self.calib_bias)

	def load_noise_measurement(self, temp_files, bw):
		import pickle
		self.GN_P = []
		for T, f in temp_files:
			noise_powers = pickle.load(open(f, 'rb'))[1]['Power']
			self.GN_target_frequencies = noise_powers[1][0]
			self.GN_frequencies = noise_powers[1][1]
			self.GN_P.append(10**(noise_powers[2]/10))
		self.G = np.zeros_like(self.GN_P[0])
		self.TN = np.zeros_like(self.GN_P[0])
		self.G_1d = np.zeros(self.GN_target_frequencies.shape[0])
		self.TN_1d = np.zeros(self.GN_target_frequencies.shape[0])
		for tf_id, tf in enumerate(self.GN_target_frequencies):
			gain, noise_T = self.gain_noise(self.GN_frequencies, np.asarray(self.GN_P)[:, tf_id,:], temp_files[0][0], temp_files[1][0], bw)
			self.G[tf_id, :] = gain
			self.TN[tf_id, :] = noise_T
			self.G_1d[tf_id] = np.interp(tf, self.GN_frequencies, gain)
			self.TN_1d[tf_id] = np.interp(tf, self.GN_frequencies, noise_T)
			#print (noise_powers)

		self.GN_meas = {'Gain frequency dependence': (('Target frequency', 'Probe frequency'),
						 (self.GN_target_frequencies, self.GN_frequencies),
						 self.G, {'log':10}),
						'Noise T frequency dependence': (('Target frequency', 'Probe frequency'),
						 (self.GN_target_frequencies, self.GN_frequencies),
						 self.TN),
						'Gain': (('Frequency',), (self.GN_target_frequencies,), self.G_1d, {'log':10}),
						'Noise T': (('Frequency',), (self.GN_target_frequencies,), self.TN_1d),}
		self.off_gain = np.nanmedian(np.nanmedian(self.G))
		return self.GN_meas

	def load_vna_noise_measurement(self, filename, vna_power, bw, atten, transpose=False):
		from scipy.constants import Boltzmann
		noise_vna = open(filename, 'rb')
		data_vna = pickle.load(noise_vna)
		noise_vna.close()
		noise_rel = (np.real(data_vna[1]['S-parameter std'][2])/np.abs(data_vna[1]['S-parameter'][2]))**2
		noise_vna_inp = 1e-3*(10**(vna_power/10.))
		noise_T = noise_rel*noise_vna_inp*(10**(atten/10))/(bw*Boltzmann)
		gain = (np.abs(data_vna[1]['S-parameter'][2])**2)*10**(-atten/10)
		self.TN_m = noise_T.T if transpose else noise_T
		self.G_m = gain.T if transpose else gain
		self.GN_m_target_frequencies = data_vna[1]['S-parameter'][1][0]
		self.GN_m_probe_frequencies = data_vna[1]['S-parameter'][1][1]
		self.G_m_1d = np.zeros(self.GN_m_target_frequencies.shape[0])
		self.TN_m_1d = np.zeros(self.GN_m_target_frequencies.shape[0])
		for tf_id, tf in enumerate(self.GN_m_target_frequencies):
			self.G_m_1d[tf_id] = np.interp(tf, self.GN_m_probe_frequencies, gain[tf_id,:])
			self.TN_m_1d[tf_id] = np.interp(tf, self.GN_m_probe_frequencies, noise_T[tf_id,:])

		self.off_gain_m = np.nanmedian(self.G_m)

		self.GN_m_meas = {'Monochromatic gain frequency dependence': (('Target frequency', 'Probe frequency'),
						 (self.GN_m_target_frequencies, self.GN_m_probe_frequencies),
						 self.G_m, {'log':10}),
						'Monochromatic noise T frequency dependence': (('Target frequency', 'Probe frequency'),
						 (self.GN_m_target_frequencies, self.GN_m_probe_frequencies),
						 self.TN_m),
						'Monochromatic gain': (('Frequency',), (self.GN_m_target_frequencies,), self.G_m_1d, {'log':10}),
						'Monochromatic noise T': (('Frequency',), (self.GN_m_target_frequencies,), self.TN_m_1d),}
		self.off_gain = np.nanmedian(np.nanmedian(self.G))
		return self.GN_m_meas

	def load_gain_saturation_measurement(self, filename, filter_kernel = (3,3,3)):
		import pickle
		from scipy.signal import medfilt, convolve
		file = open(filename, 'rb')
		data = pickle.load(file)
		data_power = np.abs(data[1]['S-parameter'][2])**2
		filter_kernel = np.ones(filter_kernel)
		data_power_filt = np.transpose(convolve(data_power, filter_kernel/np.sum(filter_kernel), mode='same'), axes=(1,0,2))
		file.close()
		data_filt_diff = np.log10(data_power_filt/data_power_filt[0,:,:])*10
		data_filt_uncompressed = (data_filt_diff>-1)
		pp, tf, pf = np.meshgrid(data[1]['S-parameter'][1][1],
								 data[1]['S-parameter'][1][0],
								 data[1]['S-parameter'][1][2],
								 indexing='ij')
		pp[data_filt_uncompressed]=np.nan
		pp_compression = np.nanmin(pp, axis=0)

		compression_1db_1d = np.zeros(data[1]['S-parameter'][1][0].shape, dtype=float)
		for tf_id, _tf in enumerate(data[1]['S-parameter'][1][0]):
			compression_1db_1d[tf_id] = np.interp(_tf, data[1]['S-parameter'][1][2], pp_compression[tf_id,:])

		self.sat_1db_ft_freq = pp_compression
		self.sat_1db_ft = compression_1db_1d

		self.sat_target_freq = data[1]['S-parameter'][1][0]
		self.sat_freq = data[1]['S-parameter'][1][2]

		self.sat_meas = data[1]
		self.sat_meas['1 dB compression point on frequency'] = (('Target frequency', 'Probe frequency'), \
													   (self.sat_target_freq, self.sat_freq), self.sat_1db_ft_freq)
		self.sat_meas['1 dB compression point'] = (('Frequency',), (self.sat_target_freq,), self.sat_1db_ft)

		return self.sat_meas

	def save_gain_saturation_plots(self, name):
		import save_pkl
		calibration_path = get_config().get('datadir')+'/calibrations/paramp/saturation/{0}/'.format(name)
		header = {'name':name, 'type':'gain saturation'}
		save_pkl.save_pkl(header, self.sat_meas, location=calibration_path)

	def save_gain_noise_plots(self, name):
		import save_pkl
		calibration_path = get_config().get('datadir')+'/calibrations/paramp/gain & noise/{0}/'.format(name)
		header = {'name':name, 'type':'gain & noise'}
		save_pkl.save_pkl(header, self.GN_meas, location=calibration_path)

	def save_vna_gain_noise_plots(self, name):
		import save_pkl
		calibration_path = get_config().get('datadir')+'/calibrations/paramp/monochromatic gain & noise/{0}/'.format(name)
		header = {'name':name, 'type':'Monochromatic gain & noise'}
		save_pkl.save_pkl(header, self.GN_m_meas, location=calibration_path)

	def planck_function(self, f, Ts, gains):
		from scipy.constants import Planck, Boltzmann
		return np.sum([Planck*f*(0.5+1./(np.exp(Planck*f/(Boltzmann*T))-1))*gain for T, gain in zip(Ts, gains)])

	def gain_noise(self, f, P_meas, T1, T2, bw):
		from scipy.constants import Boltzmann
		G_ = np.zeros_like(f)
		TN_ = np.zeros_like(f)
		P_meas = np.asarray(P_meas)*1e-3
		for f_id, f_ in enumerate(f):
			P_in = [self.planck_function(f_,
										 [t[0] for t in T1],
										 [t[1] for t in T1]),
					self.planck_function(f_,
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

	def set_target_freq_calib(self, f):
		self.pump_src.set_frequency(self.pump_frequency_by_target_frequency(f))
		self.pump_src.set_power(self.pump_power_by_target_frequency(f))
		if not self.voltage_setter:
			self.bias_src.set_current(self.bias_by_target_frequency(f))
		else:
			self.voltage_setter(self.bias_by_target_frequency(f))

	def calibrate_vna(self, frequencies, name):
		calibration_path = get_config().get('datadir')+'/calibrations/paramp/calibration'
		calibration = sweep.sweep(self,
								  (freqs, paramp.set_target_f, 'Target frequency'),
								  filename='Paramp VNA calibration {0}'.format(name),
								  output=False,
								  location=calibration_path)
		self.load_calibration(calibration_path, name)