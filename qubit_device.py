from . import pulses, awg_iq_multi, modem_readout
from .instrument_drivers.TSW14J56driver import TSW14J56_evm_reducer
import numpy as np

class qubit_device:
	def __init__(self, exdir_db, sweeper):
		self.exdir_db = exdir_db
		self.ftol = 100
		self.sweeper = sweeper
	def set_qubits_from_dict(self, _dict):
		try:
			assert set(_dict.keys()) == set(self.get_qubit_list())
		except Exception as e:
			print(str(e), type(e))
			self.set_qubit_list(_dict)
		for qubit_id, qubit in _dict.items():
			if 'r' in qubit:
				if 'Fr' in qubit['r']:
					try:
						assert(self.get_qubit_fr(qubit_id)-qubit['r']['Fr'])<self.ftol
					except Exception as e:
						print(str(e), type(e))
						self.set_qubit_fr(qubit_id=qubit_id, fr=qubit['r']['Fr'])
				if 'iq_devices' in qubit['r']:
					try:
						assert(qubit['r']['iq_devices'] == self.get_qubit_readout_channel_list(qubit_id))
					except Exception as e:
						print(str(e), type(e))
						self.set_qubit_readout_channel_list(qubit_id, qubit['r']['iq_devices'])
			if 'q' in qubit:
				if 'F01_min' in qubit['q']['F']:
					try:
						assert(self.get_qubit_fq(qubit_id, transition_name='01')-qubit['q']['F']['F01_min'])<self.ftol
					except Exception as e:
						print(str(e), type(e))
						self.set_qubit_fq(qubit_id=qubit_id, fq=qubit['q']['F']['F01_min'], transition_name='01')
				if 'iq_devices' in qubit['q']:
					try:
						assert(qubit['q']['iq_devices'] == self.get_qubit_excitation_channel_list(qubit_id))
					except Exception as e:
						print(str(e), type(e))
						self.set_qubit_excitation_channel_list(qubit_id, qubit['q']['iq_devices'])
			for key, value in qubit.items():
				if key != 'r' and key != 'q':
					try:
						last_measurement = self.exdir_db.select_measurement(measurement_type=key, metadata={'qubit_id':qubit_id})
						assert(last_measurement.metadata[key] == str(value))
					except:
						self.exdir_db.save(measurement_type=key, metadata={key: str(value), 'qubit_id': qubit_id})

	def set_sample_globals(self, globals):
		for global_name, global_value in globals.items():
			try:
				last_measurement = self.exdir_db.select_measurement(measurement_type=global_name, metadata={'scope':'sample'})
				assert(last_measurement.metadata[global_name] == str(global_value))
			except:
				self.exdir_db.save(measurement_type=global_name, metadata={global_name: str(global_value), 'scope':'sample'})


	def get_sample_global(self, name):
		return self.exdir_db.select_measurement(measurement_type=name, metadata={'scope':'sample'}).metadata[name]

	def get_qubit_constant(self, name, qubit_id):
		try:
			return self.exdir_db.select_measurement(measurement_type=name, metadata={'qubit_id': qubit_id}).metadata[name]
		except:
			return self.exdir_db.select_measurement(measurement_type=name, metadata={'scope': 'sample'}).metadata[name]

	def get_qubit_fq(self, qubit_id, transition_name='01'):
		fq_measurement = self.exdir_db.select_measurement(measurement_type='qubit_fq',
														  metadata={'qubit_id':qubit_id,
																	'transition_name':transition_name})
		return float(fq_measurement.metadata['fq'])
	def set_qubit_fq(self, fq, qubit_id, transition_name='01'):
		self.exdir_db.save(measurement_type='qubit_fq', metadata={'qubit_id':qubit_id,
																  'transition_name':transition_name,
																  'fq':str(fq)})

	# def get_Rabi_fr(self, qubit_id, channel_amplitudes, transition_name='01'):
		# try:

		# except:


	def get_qubit_fr(self, qubit_id):
		fr_measurement = self.exdir_db.select_measurement(measurement_type='qubit_fr',
														  metadata={'qubit_id':qubit_id})
		return float(fr_measurement.metadata['fr'])
	def set_qubit_fr(self, fr, qubit_id):
		self.exdir_db.save(measurement_type='qubit_fr',
						   metadata={'qubit_id':qubit_id,
									 'fr': str(fr)})

	def set_qubit_list(self, qubit_list):
		self.exdir_db.save(measurement_type='qubit_list', metadata={key:key for key in qubit_list})

	def get_qubit_list(self):
		return [i for i in	self.exdir_db.select_measurement(measurement_type='qubit_list').metadata.keys()]

	def set_qubit_excitation_channel_list(self, qubit_id, device_list):
		metadata = {k:v for k,v in device_list.items()}
		metadata['qubit_id'] = qubit_id
		self.exdir_db.save(measurement_type='qubit_excitation_channel_list', metadata=metadata)

	def get_qubit_excitation_channel_list(self, qubit_id):
		return {channel_name:device_name for channel_name, device_name in
				self.exdir_db.select_measurement(measurement_type='qubit_excitation_channel_list',
												 metadata={'qubit_id': qubit_id}).metadata.items() if channel_name != 'qubit_id'}

	def set_qubit_readout_channel_list(self, qubit_id, device_list):
		metadata = {k:v for k,v in device_list.items()}
		metadata['qubit_id'] = qubit_id
		self.exdir_db.save(measurement_type='qubit_readout_channel_list', metadata=metadata)

	def get_qubit_readout_channel_list(self, qubit_id):
		return {channel_name:device_name for channel_name, device_name in
				self.exdir_db.select_measurement(measurement_type='qubit_readout_channel_list',
												 metadata={'qubit_id': qubit_id}).metadata.items() if channel_name != 'qubit_id'}

	def create_pulsed_interfaces(self, iq_devices, extra_channels={}):
		self.awg_channels = {}
		self.readout_channels = {}
		for _qubit_id in self.get_qubit_list():
			fr = self.get_qubit_fr(_qubit_id)
			fq = self.get_qubit_fq(_qubit_id)
			for channel_name, device_name in self.get_qubit_excitation_channel_list(_qubit_id).items():
				iq_devices[device_name].carriers[channel_name] = awg_iq_multi.carrier(iq_devices[device_name])
				#iq_devices[device_name].carriers[channel_name].set_frequency(fq)
				self.awg_channels[channel_name] = iq_devices[device_name].carriers[channel_name]

			for channel_name, device_name in self.get_qubit_readout_channel_list(_qubit_id).items():
				iq_devices[device_name].carriers[channel_name] = awg_iq_multi.carrier(iq_devices[device_name])
				#iq_devices[device_name].carriers[channel_name].set_frequency(fr)
				self.awg_channels[channel_name] = iq_devices[device_name].carriers[channel_name]
				self.readout_channels[channel_name] = iq_devices[device_name].carriers[channel_name]

		self.awg_channels.update(extra_channels)
		self.pg = pulses.pulses(self.awg_channels)
		self.update_pulsed_frequencies()

	def update_pulsed_frequencies(self):
		iq_devices = []
		for _qubit_id in self.get_qubit_list():
			fr = self.get_qubit_fr(_qubit_id)
			fq = self.get_qubit_fq(_qubit_id)
			for channel_name, device_name in self.get_qubit_excitation_channel_list(_qubit_id).items():
				self.awg_channels[channel_name].set_frequency(fq)
				iq_devices.append(self.awg_channels[channel_name].parent)
			for channel_name, device_name in self.get_qubit_readout_channel_list(_qubit_id).items():
				self.awg_channels[channel_name].set_frequency(fr)
				iq_devices.append(self.awg_channels[channel_name].parent)
		for d in list(set(iq_devices)):
			d.do_calibration(d.sa)

	def setup_modem_readout(self, hardware):
		old_settings = hardware.set_readout_delay_calibration_mode() ## set adc settings to modem delay calibration mode
		try:
			self.trigger_readout_seq = [self.pg.p('ro_trg', hardware.get_readout_trigger_pulse_length(), self.pg.rect, 1)]

			self.modem = modem_readout.modem_readout(self.pg, hardware.adc, self.trigger_readout_seq, axis_mean=0, exdir_db=self.exdir_db)
			self.modem.readout_channels = self.readout_channels
			self.modem.adc_device = hardware.adc_device

			self.modem_delay_calibration_channel = [channel_name for channel_name in self.modem.readout_channels.keys()][0]
			self.ro_trg = hardware.ro_trg

			readout_trigger_delay = self.ro_trg.get_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
			print ('Got delay calibration:', readout_trigger_delay)
			try:
				self.ro_trg.validate_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
			except Exception as e:
				print(type(e), str(e))
				self.ro_trg.modem_delay_calibrate(self.modem, self.modem_delay_calibration_channel)
				self.ro_trg.validate_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
			self.modem.get_dc_bg_calibration()
			self.modem.get_dc_calibrations(amplitude=hardware.get_modem_dc_calibration_amplitude())
		except:
			raise
		finally:
			hardware.revert_setup(old_settings) ## revert sdc settings from modem delay calibration mode

		return self.modem

	def setup_adc_reducer_iq(self, qubits, raw=False): ### pimp this code to make it more universal. All the hardware belongs to the hardware
	# file, but how do we do that here without too much boilerplate???
		feature_id = 0

		adc_reducer = TSW14J56_evm_reducer(self.modem.adc_device)
		adc_reducer.output_raw = raw
		adc_reducer.last_cov = False
		adc_reducer.avg_cov = True
		adc_reducer.resultnumber = False

		adc_reducer.avg_cov_mode = 'iq'

		qubit_measurement_dict = {}
		for qubit_id in qubits:
			if feature_id > 1:
				raise ValueError('Cannot setup hardware adc_reducer for more that 2 qubits')
			readout_channel_name = [i for i in self.get_qubit_readout_channel_list(qubit_id=qubit_id).keys()][0]
			calibration = self.modem.iq_readout_calibrations[readout_channel_name]
			qubit_measurement_dict[qubit_id] = 'avg_cov'+str(feature_id)

			adc_reducer.set_feature_iq(feature_id=feature_id, feature=calibration['feature'])
			feature_id += 1
		return adc_reducer, qubit_measurement_dict

	def set_adc_features_and_thresholds(self, features, thresholds, disable_rest=True, raw=False):
		adc_reducer = TSW14J56_evm_reducer(self.modem.adc_device)
		adc_reducer.output_raw = raw
		adc_reducer.last_cov = False
		adc_reducer.avg_cov = False
		adc_reducer.resultnumber = True

		adc_reducer.avg_cov_mode = 'real'

		feature_id = 0
		for feature, threshold in zip(features, thresholds):
			adc_reducer.set_feature_real(feature_id=feature_id, feature=feature, threshold=threshold)
			feature_id += 1

		if disable_rest:
			while (feature_id < self.modem.adc_device.num_covariances):
				adc_reducer.disable_feature(feature_id=feature_id)
				feature_id += 1
		return adc_reducer
