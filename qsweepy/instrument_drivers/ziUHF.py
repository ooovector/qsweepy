import numpy as np
from qsweepy.instrument_drivers.zihdawg import ZIDevice

import time

MAPPINGS = {
    "result_source": {
        0: "Crosstalk",
        1: "Threshold",
        2: "Rotation",
        4: "Crosstalk Correlation",
        5: "Threshold Correlation",
        7: "Integration",
    },
    "averaging_mode": {0: "Cyclic", 1: "Sequential", },
}


class ziUHF(ZIDevice):
	def __init__(self, ch_num) -> None:
		super(ziUHF, self).__init__(device_id='dev2491', devtype='UHF')
		# Set number of different channels for signal demodulation
		self.ch_num = ch_num
		# self.dev.enable_readout_channels(list(range(ch_num)))
		# Set parameters required to be returned
		self.output_raw = True
		self.output_result = True
		# Set default result source to be Integration
		self.result_source = 7
		self.timeout = 10

	@property
	def nsamp(self) -> int:
		'''
		Number of samples recorded and used to get the result values
		'''
		return self.daq.getInt('/' + self.device + '/qas/0/monitor/length')

	@nsamp.setter
	def nsamp(self, nsamp):
		if nsamp > 4096:
			raise ValueError("Maximum number samples is 4096!")
		# Set both recording and integration length
		self.daq.setInt('/' + self.device + '/qas/0/monitor/length', nsamp)
		self.daq.setInt('/' + self.device + '/qas/0/integration/length', nsamp)

	@property
	def nsegm(self) -> int:
		'''
		Amount of repetitions to get the result
		'''
		return self.daq.getInt('/' + self.device + '/qas/0/result/length')

	@nsegm.setter
	def nsegm(self, nsegm):
		self.rep_num = nsegm
		self.daq.setInt('/' + self.device + '/qas/0/result/length', nsegm)

	@property
	def averages(self) -> int:
		'''
		Amount of each repetition averages
		'''
		return self.daq.getInt('/' + self.device + '/qas/0/result/averages')

	@averages.setter
	def averages(self, averages):
		self.daq.setInt('/' + self.device + '/qas/0/result/averages', averages)

	@property
	def trigger_result(self) -> int:
		# TODO write definition for each trigger type
		return self.daq.getInt('/' + self.device + '/qas/0/integration/trigger/channel')

	@trigger_result.setter
	def trigger_result(self, trigger):
		self.daq.setInt('/' + self.device + '/qas/0/integration/trigger/channel', trigger)

	@property
	def trigger_monitor(self) -> int:
		return self.daq.getInt('/' + self.device + '/qas/0/monitor/trigger/channel')

	@trigger_monitor.setter
	def trigger_monitor(self, trigger):
		self.daq.setInt('/' + self.device + '/qas/0/monitor/trigger/channel', trigger)

	@property
	def result_source(self) -> str:
		return MAPPINGS['result_source'][self.daq.getInt('/' + self.device + '/qas/0/result/source')]

	@result_source.setter
	def result_source(self, result_source):
		self.daq.setInt('/' + self.device + '/qas/0/result/source', result_source)

	@property
	def trigger_channel0_dir(self):
		return self.daq.getInt('/' + self.device + '/triggers/out/0/drive')

	@property
	def trigger_channel1_dir(self):
		return self.daq.getInt('/' + self.device + '/triggers/out/1/drive')

	@trigger_channel0_dir.setter
	def trigger_channel0_dir(self, dir):
		self.daq.setInt('/' + self.device + '/triggers/out/0/drive', dir)

	@trigger_channel1_dir.setter
	def trigger_channel1_dir(self, dir):
		self.daq.setInt('/' + self.device + '/triggers/out/1/drive', dir)

	def get_points(self) -> dict:
		points = {}
		if self.output_raw:
			points.update({'Voltage': [('Sample', np.arange(self.nsegm), ''),
						('Time', np.arange(self.nsamp)/self.get_clock(), 's')]})
		if self.output_result:
			points.update({self.result_source + str(channel): [] for channel in range(self.ch_num)})

		return points

	def get_opts(self) -> dict:
		opts = {}
		if self.output_raw:
			opts.update({'Voltage': {'log': None}})
		if self.output_result:
			opts.update({self.result_source + str(channel): {'log': None} for channel in range(self.ch_num)})

		return opts

	def get_type(self) -> dict:
		dtypes = {}
		if self.output_raw:
			dtypes.update({'Voltage': complex})
		if self.output_result:
			# Not sure if it's right to do it this way
			dtypes.update({self.result_source + str(channel): type(
				self.daq.getList('/' + self.device + '/qas/0/result/data/' + str(channel) + '/wave')[0][1][0]['vector']
			)
			for channel in range(self.ch_num)})

		return dtypes

	def get_status(self):
		return self.daq.getInt('/' + self.device + '/awgs/0/sequencer/status')

	# Main measurer method TODO write a proper docstring
	def measure(self) -> dict:
		result = {}

		# toggle node value from 0 to 1 for result reset
		self.daq.setInt('/' + self.device + '/qas/0/result/reset', 0)
		self.daq.setInt('/' + self.device + '/qas/0/result/reset', 1)
		# and for monitor reset
		self.daq.setInt('/' + self.device + '/qas/0/monitor/reset', 0)
		self.daq.setInt('/' + self.device + '/qas/0/monitor/reset', 1)
		# enable both digitizer regimes
		self.daq.setInt('/' + self.device + '/qas/0/result/enable', 1)
		self.daq.setInt('/' + self.device + '/qas/0/monitor/enable', 1)

		# and start the sequencer execution
		self.run()

		# Sleep to correctly capture initial status TODO mb there is a better way to do it
		time.sleep(0.5)
		t1 = time.time()

		while(1):
			if(self.get_status() == 0):
				break
			else:
				pass

			if(time.time()-t1>self.timeout):
				print ("Acquisition failed")
				break

		self.daq.setInt('/' + self.device + '/qas/0/result/enable', 0)
		self.daq.setInt('/' + self.device + '/qas/0/monitor/enable', 0)

		if self.output_raw:
			# Acquire data from the device:
			result.update({'Voltage': (self.daq.getList('/' + self.device + '/qas/0/monitor/inputs/0/wave')[0][1][0]['vector'] +
						1j * self.daq.getList('/' + self.device + '/qas/0/monitor/inputs/1/wave')[0][1][0]['vector'])[:self.nsamp]})

		# Readout result and store it with key depending on result source
		if self.output_result:
			result.update({self.result_source + str(channel):
						self.daq.getList('/' + self.device + '/qas/0/result/data/' + str(channel) + '/wave')[0][1][0]['vector']
						for channel in range(self.ch_num)})

		return result

	def set_feature_iq(self, channel, feature_real, feature_imag) -> None:
		'''
		Use API to upload the demodulation weights
		:param channel: number of channel used to demodulate
		:param feature_real: I part of the weights
		:param feature_imag: Q part of the weights
		# Had to separate due to strange ZI setVector method issue
		# Should be defined separately
		'''

		self.daq.setVector('/' + self.device + '/qas/0/integration/weights/' + str(channel) + '/real', feature_real)
		self.daq.setVector('/' + self.device + '/qas/0/integration/weights/' + str(channel) + '/imag', feature_imag)



