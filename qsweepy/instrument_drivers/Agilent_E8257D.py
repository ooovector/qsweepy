# Agilent_E8257D.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA	 02110-1301	 USA

from qsweepy.instrument_drivers.instrument import Instrument
import pyvisa as visa
import types
import logging
import numpy

class Agilent_E8257D(Instrument):
	'''
	This is the driver for the Agilent E8257D Signal Genarator

	Usage:
	Initialize with
	<name> = instruments.create('<name>', 'Agilent_E8257D', address='<GBIP address>, reset=<bool>')
	'''

	def __init__(self, name, address, reset=False):
		'''
		Initializes the Agilent_E8257D, and communicates with the wrapper.

		Input:
		  name (string)	   : name of the instrument
		  address (string) : GPIB address
		  reset (bool)	   : resets to default values, default=False
		'''
		logging.info(__name__ + ' : Initializing instrument Agilent_E8257D')
		Instrument.__init__(self, name, tags=['physical'])

		# Add some global constants
		self._address = address
		self._visainstrument = visa.ResourceManager().open_resource(self._address)

		self.add_parameter('power',
			flags=Instrument.FLAG_GETSET, units='dBm', minval=-20, maxval=18, type=float)
		self.add_parameter('phase',
			flags=Instrument.FLAG_GETSET, units='rad', minval=-numpy.inf, maxval=numpy.inf, type=float)
		self.add_parameter('frequency',
			flags=Instrument.FLAG_GETSET, units='Hz', minval=1e5, maxval=20e9, type=float)
		self.add_parameter('status',
			flags=Instrument.FLAG_GETSET, type=bool)

		self.add_function('reset')
		self.add_function ('get_all')


		if (reset):
			self.reset()
		else:
			self.get_all()

	def reset(self):
		'''
		Resets the instrument to default values

		Input:
			None

		Output:
			None
		'''
		logging.info(__name__ + ' : resetting instrument')
		self._visainstrument.write('*RST')
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
		params = {}
		params ['status'] = self.get_status()
		params ['freq'] = self.get_frequency()
		params ['power'] = self.get_power()
		return params
		#self.get_power()
		#self.get_phase()
		#self.get_frequency()
		

	def do_get_power(self):
		'''
		Reads the power of the signal from the instrument

		Input:
			None

		Output:
			ampl (?) : power in ?
		'''
		logging.debug(__name__ + ' : get power')
		return float(self._visainstrument.query('POW:AMPL?'))

	def do_set_power(self, amp):
		'''
		Set the power of the signal

		Input:
			amp (float) : power in ??

		Output:
			None
		'''
		logging.debug(__name__ + ' : set power to %f' % amp)
		self._visainstrument.write('POW:AMPL %s;*OPC?' % amp)

	def do_get_phase(self):
		'''
		Reads the phase of the signal from the instrument

		Input:
			None

		Output:
			phase (float) : Phase in radians
		'''
		logging.debug(__name__ + ' : get phase')
		return float(self._visainstrument.query('PHASE?'))

	def do_set_phase(self, phase):
		'''
		Set the phase of the signal

		Input:
			phase (float) : Phase in radians

		Output:
			None
		'''
		logging.debug(__name__ + ' : set phase to %f' % phase)
		self._visainstrument.write('PHASE %s' % phase)

	def do_get_frequency(self):
		'''
		Reads the frequency of the signal from the instrument

		Input:
			None

		Output:
			freq (float) : Frequency in Hz
		'''
		logging.debug(__name__ + ' : get frequency')
		return float(self._visainstrument.query('FREQ:CW?'))

	def do_set_frequency(self, freq):
		'''
		Set the frequency of the instrument

		Input:
			freq (float) : Frequency in Hz

		Output:
			None
		'''
		logging.debug(__name__ + ' : set frequency to %f' % freq)
		self._visainstrument.query('FREQ:CW %s;*OPC?' % freq)

	def do_get_status(self):
		'''
		Reads the output status from the instrument

		Input:
			None

		Output:
			status (string) : 'On' or 'Off'
		'''
		logging.debug(__name__ + ' : get status')
		stat = self._visainstrument.query('OUTP?')
		
		if (stat[0]=='1' or stat == 1):
		  return True
		elif (stat[0]=='0' or stat == 0 or not stat):
		  return False
		else:
		  raise ValueError('Output status not specified : %s' % stat)
		return

	def do_set_status(self, status):
		'''
		Set the output status of the instrument

		Input:
			status (string) : 'On' or 'Off'

		Output:
			None
		'''
		logging.debug(__name__ + ' : set status to %s' % status)
		self._visainstrument.write('OUTP %s' % int(status))

	# shortcuts
	def off(self):
		'''
		Set status to 'off'

		Input:
			None

		Output:
			None
		'''
		self.set_status('off')

	def on(self):
		'''
		Set status to 'on'

		Input:
			None

		Output:
			None
		'''
		self.set_status('on')

