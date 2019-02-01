from qsweepy import save_pkl
from qsweepy import plotting
from qsweepy import sweep
from qsweepy import fitting
from qsweepy import qjson
from qsweepy import tomography
from qsweepy import clifford
from qsweepy import interleaved_benchmarking
from matplotlib import pyplot as plt
import numpy as np
class gauss_hd_calibration:
	def __init__(self, tld, sigma, length, alpha, anharmonicity):
		self.tld = tld
		#warnings.filterwarnings('ignore')
		self.sigma = sigma
		self.length = length
		self.alpha = alpha
		self.anharmonicity = anharmonicity
	def rabi_amplitudes(self,ex_amps,num_pulses,*params):
		pg=self.tld.pulse_sequencer
		sequence = []
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.tld.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.tld.readout_device.zero_setter = self.tld.set_zero_sequence # for diff_readout
		def set_ex_amp(amp):
			nonlocal sequence
			pulse = [(c, pg.gauss_hd, a*amp, self.sigma, self.alpha/(self.anharmonicity*2*np.pi)) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			sequence = [pg.pmulti(self.length, *tuple(pulse))]*num_pulses+self.tld.ro_sequence
			if not hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		measurement_name = 'Rabi ampl excitation channel {}'.format(','.join(self.tld.ex_channels))#+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		measurement = sweep.sweep(self.tld.readout_device, 
								  (ex_amps, set_ex_amp, 'Rabi pulse amplitude', 'v'), 
								  *params, 
								  filename=measurement_name, 
								  shuffle=self.tld.shuffle, 
								  root_dir = root_dir,
								  plot_separate_thread= self.tld.plot_separate_thread,
								  plot=self.tld.plot)
		measurement_fitted, fitted_parameters_rabi = self.tld.fitter(measurement, fitting.sin_fit)
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}'.format(fitted_parameters_rabi['phase'], 
																			 fitted_parameters_rabi['freq'])
		#print( measurement_fitted)
		plotting.plot_measurement(measurement_fitted, measurement_name,save=root_dir,annotation=annotation,\
								  subplots=True)
		return (measurement_fitted,fitted_parameters_rabi)
	
	def ape(self,num_pulses_array=None,y_sign='+',ape_phase=np.pi/2.,*params):
		y_sign_mul = 1 if y_sign == '-' else -1
		pg=self.tld.pulse_sequencer
		sequence = []
		ex_amp = self.get_amplitude(np.pi/2.)
		print ('ex_amp is: ', ex_amp)
		num_pulses_max = int(np.floor(self.guess_max_num_pulses()/2))-1
		if not num_pulses_array:
			num_pulses_array = np.arange(num_pulses_max)
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.tld.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.tld.readout_device.zero_setter = self.tld.set_zero_sequence # for diff_readout
		def set_ape_length(num_pulses):
			nonlocal sequence
			#pulse = [(c, pg.gauss_hd, a*ex_amp, self.sigma, self.alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			#pulse_inv = [(c, pg.gauss_hd, -a*ex_amp, self.sigma, self.alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			#pulse_y = [(c, pg.gauss_hd, 1j*y_sign_mul*a*ex_amp, self.sigma, self.alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			#sequence = [pg.pmulti(self.length, *tuple(pulse))]+\
			#   [pg.pmulti(self.length, *tuple(pulse)), pg.pmulti(self.length, *tuple(pulse_inv))]*num_pulses+\
			#   [pg.pmulti(self.length, *tuple(pulse_y))]+self.tld.ro_sequence
			sequence = self.get_pulse_seq(np.pi/2., 0)+\
						(self.get_pulse_seq(ape_phase, 0)+self.get_pulse_seq(ape_phase, np.pi))*num_pulses+\
						self.get_pulse_seq(np.pi/2., np.pi/2.*y_sign_mul)+self.tld.ro_sequence
			if not hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		measurement_name = 'APE ch {} Y'.format(','.join(self.tld.ex_channels))+y_sign#+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		measurement = sweep.sweep(self.tld.readout_device, 
								  (num_pulses_array, set_ape_length, 'APE identity pulse num'), 
								  *params, 
								  filename=measurement_name, 
								  shuffle=self.tld.shuffle, 
								  root_dir = root_dir,
								  plot_separate_thread= self.tld.plot_separate_thread,
								  plot=self.tld.plot)
		measurement_fitted, fitted_parameters_rabi = self.tld.fitter(measurement, fitting.exp_sin_fit)
		annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}'.format(fitted_parameters_rabi['phase'], 
																			 fitted_parameters_rabi['freq'])
		plotting.plot_measurement(measurement_fitted, measurement_name,save=root_dir,annotation=annotation,\
								  subplots=True)
		return measurement_fitted
	
	def alpha_skan(self,num_pulses_array,alphas,y_sign='+',chid=None,*params):
		y_sign_mul = 1 if y_sign == '+' else -1
		measurement_array=[]
		pg=self.tld.pulse_sequencer
		sequence = []
		ex_amp = self.get_amplitude(np.pi/2.)
		def find_alpha(meas,chid):
			y_real=[]
			y_imag=[]
			for i in meas:
				y_real.append(np.real(i['S21_r'+str(chid)][2]))
				y_imag.append(np.imag(i['S21_r'+str(chid)][2]))
			x = meas[0]['S21_r'+str(chid)][1][0]
			yr=[0]
			yi=[0]
			for i in range(len(meas)):
				yr+=y_real[i]
				yi+=y_imag[i]
			real=np.asarray(yr)/len(meas)
			imag=np.asarray(yi)/len(meas)
			dif_r=0
			dif_i=0
			for i in range(len(meas)):
				dif_r+=np.abs(real-y_real[i])
				dif_i+=np.abs(imag-y_imag[i])
			alpha_r=x[np.where(dif_r==min(dif_r))]
			alpha_i=x[np.where(dif_i==min(dif_i))]
			return (alpha_r,alpha_i)
		
		def set_seq():
			pg.set_seq(sequence)
		if hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
			self.tld.readout_device.diff_setter = set_seq # set the measurer's diff setter
			self.tld.readout_device.zero_setter = self.tld.set_zero_sequence # for diff_readout
		def set_alpha(alpha_dimensionless,num_pulses):
			nonlocal sequence
			alpha = alpha_dimensionless/(self.anharmonicity*2*np.pi)
			pulse = [(c, pg.gauss_hd, a*ex_amp, self.sigma, alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			pulse_inv = [(c, pg.gauss_hd, -a*ex_amp, self.sigma, alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			pulse_y = [(c, pg.gauss_hd, 1j*y_sign_mul*a*ex_amp, self.sigma, alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			sequence = [pg.pmulti(self.length, *tuple(pulse))]+\
			   [pg.pmulti(self.length, *tuple(pulse)), pg.pmulti(self.length, *tuple(pulse_inv))]*num_pulses+\
			   [pg.pmulti(self.length, *tuple(pulse_y))]+self.tld.ro_sequence
			if not hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
				set_seq()
		measurement_name = 'APE ch {} Y'.format(','.join(self.tld.ex_channels))+\
				y_sign+' dependence on HD DRAG alpha'#+self.get_measurement_name_comment()
		root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
		root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
		for i in num_pulses_array:    
			set_alphas = lambda amp: set_alpha(amp,i)
			measurement = sweep.sweep(self.tld.readout_device, 
								  (alphas, set_alphas, 'HD DRAG alpha'), 
								  *params, 
								  filename=measurement_name, 
								  shuffle=self.tld.shuffle, 
								  root_dir = root_dir,
								  plot_separate_thread= self.tld.plot_separate_thread,
								  plot=self.tld.plot)
			measurement_array.append(measurement)
			measurement_fitted, fitted_parameters_rabi = self.tld.fitter(measurement, fitting.sin_fit)
			annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}'.format(fitted_parameters_rabi['phase'], 
																			 fitted_parameters_rabi['freq'])
			plotting.plot_measurement(measurement_fitted, measurement_name,save=root_dir,annotation=annotation,\
								  subplots=True)
		alpha = find_alpha(measurement_array,chid)
		print(alpha)
		return measurement_array,alpha
	
	#pognali
	
	def guess_max_num_pulses(self):
		num_pulses_max = int(np.floor(self.tld.Ramsey_result['ramsey_decay']/self.length*4))
		return num_pulses_max
	
	def calibrate_amplitude(self, max_angle = np.pi):
		from scipy.interpolate import interp1d
		from scipy.special import erf
		num_pulses_max = self.guess_max_num_pulses()
		num_pulses_log2_max = int(np.floor(np.log(num_pulses_max)/np.log(2)))
		min_pulse_num_log2 = int(np.ceil(np.log(2*np.pi/max_angle)/np.log(2)))
		min_pulse_num = 2**min_pulse_num_log2
		num_pulses = 2**np.arange(min_pulse_num_log2, num_pulses_log2_max)
		print ('num_pulses_in_scan: ', num_pulses)
		amplitude_scan_points = 12
		
		# rabi_amplitude_calibration
		# rabi freqeuncy guess from tld
		rabi_freq_rect_per_amp = self.tld.Rabi_rect_result['rabi_rect_freq']
		ex_amplitudes = np.linspace(0, 1.0, 21)
		
		def per_amplitude_angle_guess(length, sigma):
			erf_arg = length/(np.sqrt(8)*sigma)
			cutoff = np.exp(-erf_arg**2)
			erf_results=0.5*(1+erf(erf_arg))
			print ('length: ', length)
			print ('sigma: ', sigma)
			print('cutoff: ', cutoff)
			print('erf_arg: ', erf_arg)
			print('erf result', erf_results)
			print ('infinite length result: ', sigma*np.sqrt(2*np.pi))
			result = 1/(1-cutoff)*(sigma*np.sqrt(2*np.pi)*erf_results-length*np.exp(-erf_arg**2))
			print ('finite length result', result)

			return result
		
		max_amp_angle_guess = per_amplitude_angle_guess(self.length, self.sigma)*rabi_freq_rect_per_amp*2*np.pi
		print (per_amplitude_angle_guess(self.length, self.sigma), max_amp_angle_guess)
		print ('Rabi freq:', rabi_freq_rect_per_amp)
		amplitude_guesses = {0:0., max_amp_angle_guess:1.}
		
		for num_pulses_id, num_pulses in enumerate(num_pulses):
			print (num_pulses)
			target_angles_padded = max_angle*np.linspace(0, 1+1/(num_pulses/min_pulse_num), int(num_pulses/min_pulse_num)+2)
			print ('target_angles:', target_angles_padded)
			amplitude_guesses_kv = np.asarray([(k,v) for k,v in amplitude_guesses.items()])
			print ('amplitude_guesses_kv: ', amplitude_guesses)
			amplitude_guesses_resampled = interp1d(amplitude_guesses_kv[:,0], amplitude_guesses_kv[:,1], fill_value='extrapolate')(target_angles_padded)
			range_edges = (amplitude_guesses_resampled[1:]+amplitude_guesses_resampled[:-1])/2.
			range_mins = range_edges[:-1]
			range_maxs = range_edges[ 1:]
			print ('range_mins:', range_mins)
			print ('range_maxs:', range_maxs)
			
			thermal_state_readout_result = np.complex(*tuple(self.tld.Rabi_rect_result['rabi_rect_initial_points']))
			
			amplitude_guesses = {0:0.0}
			
			for angle_id, angle in enumerate(target_angles_padded[1:-1]):
				amplitude_scan_range = np.linspace(range_mins[angle_id], range_maxs[angle_id], amplitude_scan_points, endpoint=False)
				measurement_result, fitresults = self.rabi_amplitudes(amplitude_scan_range, num_pulses)
				amplitude_max = measurement_result[self.tld.readout_measurement_name][1][0][np.argmin(np.abs(measurement_result[self.tld.readout_measurement_name][2]-thermal_state_readout_result))]
				amplitude_guesses[angle] = amplitude_max
		self.amplitude_guesses = amplitude_guesses
		return amplitude_guesses
			#self.rabi_amplitudes(ex_amplitudes, )
			#measurement, fitresults = self.rabi_amplitudes(ex_amplitudes, num_pulses)
			
			
			#sweep.sweep(self.tld.readout_device, (), )
	
	def build_calibration_filename(self):
		tld_part = self.tld.build_calibration_filename()
		gauss_hd_part = '-s-'+str(int(np.round(self.sigma*1e9)))+'T-'+str(int(np.round(self.length*1e9)))
		return tld_part+gauss_hd_part
	
	def save_calibration(self):
		qjson.dump(type='gauss_amp', name=self.build_calibration_filename(), params=self.amplitude_guesses)
	
	def load_calibration(self):
		self.amplitude_guesses = {float(angle):float(amp) for angle, amp in qjson.load(type='gauss_amp', name=self.build_calibration_filename()).items()}
	
	def get_amplitude(self, angle):
		from scipy.interpolate import interp1d
		if not hasattr(self, 'amplitude_guesses'):
			self.load_calibration()
		amplitude_guesses_kv = np.asarray([(k,v) for k,v in self.amplitude_guesses.items()])
		return interp1d(amplitude_guesses_kv[:,0], amplitude_guesses_kv[:,1], fill_value='extrapolate')(angle)
	
	def get_pulse_seq(self, angle, phase):
		pg = self.tld.pulse_sequencer
		amplitude = self.get_amplitude(angle)*np.exp(1j*phase)
		pulse = [(c, pg.gauss_hd, a*amplitude, self.sigma, self.alpha/(self.anharmonicity*2*np.pi)) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
		sequence_with_alpha = [pg.pmulti(self.length, *tuple(pulse))]
		#pulse = [(c, pg.gauss_hd, a*amplitude, self.sigma, 0.0) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
		#sequence_without_alpha = [pg.pmulti(self.length, *tuple(pulse))]
		#print ('alpha is: ', self.alpha)
		#print ('with alpha:', sequence_with_alpha)
		#print ('without alpha:', sequence_without_alpha)
		return sequence_with_alpha
	
	def randomized_clifford_benchmarking(self, single_shot_readout, seq_lengths, *params, random_sequence_num=20):
		observables = { 'X': 0.5*np.asarray([[0, 1],   [1, 0]]),
						'Y': 0.5*np.asarray([[0, -1j],   [1j, 0]]),
						'-X': 0.5*np.asarray([[0, -1],   [-1, 0]]),
						'-Y': 0.5*np.asarray([[0, 1j],   [-1j, 0]]),
						'Z': 0.5*np.asarray([[1, 0],   [0, -1]])}
		ro_seq = single_shot_readout.ro_seq
		proj_seq = {'Xo':{'pulses': self.get_pulse_seq(np.pi/2., -np.pi/2.)+ro_seq, 
						 'operator': observables['X']},
					'Yo':{'pulses': self.get_pulse_seq(np.pi/2., 0)+ro_seq,
						 'operator': observables['Y']},
					'-Xo':{'pulses': self.get_pulse_seq(np.pi/2., np.pi/2.)+ro_seq,
						 'operator': observables['-X']},
					'-Yo':{'pulses': self.get_pulse_seq(np.pi/2., np.pi)+ro_seq,
						 'operator': observables['-Y']},
					'Zo': {'pulses':ro_seq, 'operator':observables['Z']} }
		reconstruction_basis={'x':{'operator':observables['X']},
							  'y':{'operator':observables['Y']},
							  'z':{'operator':observables['Z']}}
							  
		#multiqubit_single_shot.filter_binary = multiqubit_single_shot.filters['1']
		proj_seq = {'Z': {'pulses':ro_seq, 'operator':observables['Z']}}
		reconstruction_basis={'z':{'operator':observables['Z']}}
		tomoz = tomography.tomography(single_shot_readout, self.tld.pulse_sequencer, proj_seq, reconstruction_basis=reconstruction_basis)

		pi2 = {'X/2': {'pulses':self.get_pulse_seq(np.pi/2., np.pi),
					   'unitary': np.sqrt(0.5)*np.asarray([[1, -1j],  [-1j, 1]])},
			   'Y/2': {'pulses':self.get_pulse_seq(np.pi/2., np.pi/2.),
					   'unitary': np.sqrt(0.5)*np.asarray([[1, -1],    [1, 1]])},
			   '-X/2':{'pulses':self.get_pulse_seq(np.pi/2., 0),
					  'unitary': np.sqrt(0.5)*np.asarray([[1, 1j],   [1j, 1]])},
			   '-Y/2':{'pulses':self.get_pulse_seq(np.pi/2., -np.pi/2.),
					   'unitary': np.sqrt(0.5)*np.asarray([[1, 1],   [-1, 1]])},
			   'I':   {'pulses':[], 'unitary':np.asarray([[1, 0], [0,1]])}
			}
		clifford_group = clifford.generate_group(pi2)
		#print(clifford_group.keys(), len(clifford_group))
		
		self.pi2_bench = interleaved_benchmarking.interleaved_benchmarking(tomoz)
		self.pi2_bench.interleavers = clifford_group
		
		self.pi2_bench.random_sequence_num = random_sequence_num
		random_sequence_ids = np.arange(self.pi2_bench.random_sequence_num)
		
		self.pi2_bench.prepare_random_interleaving_sequences()
		clifford_bench = sweep.sweep(self.pi2_bench,
                             (seq_lengths, self.pi2_bench.set_sequence_length_and_regenerate, 'Gate number', ''), 
							 *params,
                             (random_sequence_ids, self.pi2_bench.set_interleaved_sequence, 'Random sequence id', ''))
		try:
			clifford_fitresults, clifford_expfit = fitting.exp_fit1d(clifford_bench['distance'][1][0], 
                                                        np.reshape(np.mean(1-clifford_bench['distance'][2], axis=1), 
                                                                   (1, len(clifford_bench['distance'][1][0]))))
																   
			plt.figure()
			plt.plot(clifford_bench['distance'][1][0], 1-clifford_bench['distance'][2], 
				marker='o', 
				color='black', 
				linestyle='')
			plt.plot(clifford_expfit[0].ravel(), clifford_expfit[1].ravel(), label='Clifford set')
			plt.grid()
			plt.xlabel('Number of gates')
			plt.ylabel('Fidelity')
			plt.legend()
			print('Clifford fidelity: {0:6.3f}'.format(np.exp(-1/clifford_fitresults[0])))
			self.clifford_fidelity = np.exp(-1/clifford_fitresults[0])
			return self.clifford_fidelity
		except:
			return clifford_bench
		
	
	# def Rabi_amplitudes_calibration(self,num_pulses,ex_amps,chid):
		
		# from scipy.optimize import curve_fit
		# pg=self.tld.pulse_sequencer
		# sequence=[]
		# def set_seq():
			# pg.set_seq(sequence)
		# if hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
			# self.tld.readout_device.diff_setter = set_seq # set the measurer's diff setter
			# self.tld.readout_device.zero_setter = self.tld.set_zero_sequence # for diff_readout
		# def set_ex_amp(amp,num_pulses):
			# nonlocal sequence
			# pulse = [(c, pg.gauss_hd, a*amp, self.sigma, self.alpha) for c,a in zip(self.tld.ex_channels, self.tld.ex_amplitudes)]
			# sequence = [pg.pmulti(self.length, *tuple(pulse))]*num_pulses+self.tld.ro_sequence
			# if not hasattr(self.tld.readout_device, 'diff_setter'): # if this is a sifferential measurer
				# set_seq()

		# def fit_cal(data):
			# x = [data[i][0] for i in range(len(data))]
			# y = [data[i][1] for i in range(len(data))]
			# func_for_fit_cal = lambda x,a,b,c : a*x**(5/3)+b*x**(4/3)+c
			# popt = curve_fit(func_for_fit_cal,x,y)[0]
			# angles = np.linspace(x[np.argmin(x)],x[np.argmax(x)],1000)
			# amps = func_for_fit_cal(angles,popt[0],popt[1],popt[2])
			# return angles,amps,lambda x:popt[0]*x**(5/3)+popt[1]*x**(4/3)+popt[2]
		
		# def find_min(x,y):
			# func = lambda x,a,b,c: a*x**2+c*x+b
			# popt=curve_fit(func,x,y)[0]
			# n1 = np.linspace(x[np.argmin(x)],x[np.argmax(x)],100)
			# n2 = func(n1,popt[0],popt[1],popt[2])
# #             data = {keys[0],}
# #             plotting.plot_measurement((n1,n2), measurement_name,save=root_dir,subplots=True)
			# min_amp=n1[np.argmin(n2)]
			# return min_amp
	
	
		# def set_amps(num_pulses,ex_amps_first,chid): 
			# fitted_parameters = self.rabi_amplitudes(ex_amps_first,num_pulses)[1]
			# x_max_0,x_min_0=(np.pi-fitted_parameters['phase'])/(2*np.pi*fitted_parameters['freq']),\
							# (np.pi*2-fitted_parameters['phase'])/(2*np.pi*fitted_parameters['freq'])
			# cut=np.abs((x_max_0-x_min_0))/4
			# ex_amps=np.linspace(x_max_0-cut,x_max_0+cut,10)
			# if min(ex_amps)<=0:
				# ex_amps=np.linspace(x_max_0*0.5,x_max_0+cut,10)
			# T0=1/fitted_parameters['freq']
			# amp_for_angle=[]
			# k=0
			# set_ex_next = lambda amp: set_ex_amp(amp,num_pulses)
			# for i in range(int(num_pulses/2)):
				# measurement_name = 'Rabi ampl excitation ch {}'.format(','.join(self.tld.ex_channels))
				# root_dir, day_folder_name, time_folder_name = save_pkl.get_location()
				# root_dir = '{}/{}/{}-{}'.format(root_dir, day_folder_name, time_folder_name, measurement_name)
				# measurement = sweep.sweep(self.tld.readout_device, 
										  # (ex_amps, set_ex_next, measurement_name), 
										  # filename=measurement_name, 
										# shuffle=self.tld.shuffle, 
										# root_dir = root_dir,
										# plot_separate_thread= self.tld.plot_separate_thread,
										# plot=self.tld.plot)
# #                 measurement_fitted, fitted_parameters_rabi = self.fitter(measurement, find_max)
# #                 annotation = 'Phase: {0:4.4g} rad, Freq: {1:4.4g}'.format(fitted_parameters_rabi['phase'], 
# #                                                                              fitted_parameters_rabi['freq'])
# #                 plotting.plot_measurement(measurement_fitted, measurement_name,save=root_dir,annotation=None,\
# #                                   subplots=True)
				# x = measurement['S21_r'+str(chid)][1][0]
				# y=np.real(measurement['S21_r'+str(chid)][2])
				# max_amp=find_min(x,y)
				# if x_min_0>1/fitted_parameters['freq']:
					# angle_step=(1+2*i)
				# else:
					# angle_step=2*(i+1)
				# amp_for_angle.append([angle_step,max_amp])
				# if k>0:
					# T=amp_for_angle[i][1]-amp_for_angle[i-1][1]
# #                     print(T)
					# ex_amps=np.linspace(max_amp+T-cut,max_amp+T+cut,10)
				# else:
					# ex_amps=np.linspace(max_amp+T0-cut,max_amp+T0+cut,10)
					# k=1
				# print('angle: {}pi, amplitude: {}'.format(angle_step/num_pulses,max_amp))
				# print(max(ex_amps))
				# if max(ex_amps)>=1:
					# break
			# return amp_for_angle
		
		# amps=set_amps(num_pulses,ex_amps,chid)
		# for i in range(len(amps)):
			# amps[i][0]/=num_pulses
		# x,y,amp_angle=fit_cal(amps)
		# plt.figure()
		# plt.plot(x,y,"-")
		# plt.plot([amps[0][0],amps[-1][0]],[amps[0][1],amps[-1][1]])
		# plt.plot([amps[i][0] for i in range(len(amps))],[amps[i][1] for i in range(len(amps))],'o')
		# plt.show()
		# return x,y,amp_angle,amps
   