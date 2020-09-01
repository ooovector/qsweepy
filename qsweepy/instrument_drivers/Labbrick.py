from qsweepy.instrument import Instrument
import types
import logging
import numpy
import ctypes

from qsweepy.instrument_drivers._labbrick import _labbrick

def get_labbricks():
	num_devices = _labbrick.get_num_devices()
	devices_ids_type = ctypes.c_uint*num_devices
	get_dev_info_proto = ctypes.WINFUNCTYPE (ctypes.c_int, devices_ids_type)
	get_dev_info = get_dev_info_proto (("fnLMS_GetDevInfo", _labbrick.labbrick_dll), ((2, 'ActiveDevices'),) )
	#device_ids_buffer = devices_ids_type()
	device_ids = get_dev_info()

	devices = {}
	for device_id in device_ids:
		max_modelname = 32
		model_name_unicode = ctypes.create_unicode_buffer(32)
		_labbrick.get_model_name_unicode(device_id, model_name_unicode)
		devices[device_id] = {'name': model_name_unicode.value, 'serial_number': _labbrick.get_serial_number(device_id)}
		#print (model_name_unicode.value)
		#print(get_serial_number(device_id))
	return devices

class Labbrick(Instrument):
	def __init__(self, name, serial):
		'''
		Initializes the Labbrick, and communicates with the wrapper.

		Input:
		  name (string)    : name of the instrument
		  serial (int)  : serial number
		'''
		logging.info(__name__ + ' : Initializing instrument Labbrick')
		Instrument.__init__(self, name, tags=['physical'])
		
		devices = get_labbricks()
		for device_id, device in devices.items():
			if device['serial_number'] == serial:
				self._device_id = device_id
				self._device_name = device['name']
				
		if not hasattr(self, '_device_id'):
			raise ValueError('Labbrick with serial number {0} not found'.format(serial))
			
		_labbrick.set_test_mode(False)
		_labbrick.init_device(self._device_id)
		
		# Add some global constants
		self._serial_number = serial

		self.add_parameter('power',
			flags=Instrument.FLAG_GETSET, units='dBm', minval=4*_labbrick.get_min_pwr(self._device_id), \
			maxval=4*_labbrick.get_max_pwr(self._device_id), type=float)
		self.add_parameter('frequency',
			flags=Instrument.FLAG_GETSET, units='Hz', minval=_labbrick.get_min_freq(self._device_id)*10, \
			maxval=_labbrick.get_max_freq(self._device_id)*10, type=float)
		self.add_parameter('status',
			flags=Instrument.FLAG_GETSET, type=bool)

		self.add_function ('get_all')
		self.get_all()

	def get_all(self):
		'''
		Reads all implemented parameters from the instrument,
		and updates the wrapper.

		Input:
			None

		Output:
			None
		'''
		logging.info(__name__ + ' : get all')
		self.get_power()
		self.get_frequency()
		self.get_status()

	def do_get_power(self):
		'''
		Reads the power of the signal from the instrument

		Input:
			None

		Output:
			ampl (?) : power in ?
		'''
		logging.debug(__name__ + ' : get power')
		return float(_labbrick.get_abs_power_level(self._device_id)/4)

	def do_set_power(self, amp):
		'''
		Set the power of the signal

		Input:
			amp (float) : power in ??

		Output:
			None
		'''
		logging.debug(__name__ + ' : set power to %f' % amp)
		_labbrick.set_power_level(self._device_id, int(amp*4))

	def do_get_frequency(self):
		'''
		Reads the frequency of the signal from the instrument

		Input:
			None

		Output:
			freq (float) : Frequency in Hz
		'''
		logging.debug(__name__ + ' : get frequency')
		return float(_labbrick.get_frequency(self._device_id)*10)

	def do_set_frequency(self, freq):
		'''
		Set the frequency of the instrument

		Input:
			freq (float) : Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : set frequency to %f' % freq)
		_labbrick.set_frequency(self._device_id, int(freq/10))

	def do_get_status(self):
		'''
		Reads the output status from the instrument

		Input:
			None

		Output:
			status (string) : 'On' or 'Off'
		'''
		logging.debug(__name__ + ' : get status')
		return _labbrick.get_rf_on(self._device_id)

	def do_set_status(self, status):
		'''
		Set the output status of the instrument

		Input:
			status (string) : 'On' or 'Off'

		Output:
			None
		'''
		logging.debug(__name__ + ' : set status to %s' % status)
		_labbrick.set_rf_on(self._device_id, status)

	# shortcuts
	def off(self):
		'''
		Set status to 'off'

		Input:
			None

		Output:
			None
		'''
		self.set_status(False)

	def on(self):
		'''
		Set status to 'on'

		Input:
			None

		Output:
			None
		'''
		self.set_status(True)

