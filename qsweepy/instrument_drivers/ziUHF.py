import numpy as np
import logging
import textwrap

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
		self.sync_mode = False  # True only during mixers calibration
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
		self.internal_average = True
		self.default_delay_reg = 9

	def set_adc_nop(self, nop):
		self.nsamp = nop

	def get_adc_nop(self):
		return self.nsamp

	def set_adc_nums(self, nums):
		self.nsegm = nums

	def get_adc_nums(self):
		return self.nsegm

	def get_nums(self):
		return self.get_adc_nums()

	def set_nums(self, nums):
		self.set_adc_nums(nums)

	def get_clock(self):
		return self.daq.getDouble('/' + self.device + '/clockbase')

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
		if nsegm > int(2**15):
			logging.warning('Number of segments is higher then the maximum possible number of trace averages')
		# First tell QA to integrate nsegm traces
		self.daq.setInt('/' + self.device + '/qas/0/result/averages', nsegm)
		# Then make monitor average raw trace nsegm times
		self.daq.setInt('/' + self.device + '/qas/0/monitor/averages', nsegm)
		# Set the repetition user register to nsegm
		self.daq.setInt('/' + self.device + '/awgs/0/userregs/2', nsegm)

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
	def trigger_channel0_dir(self) -> int:
		return self.daq.getInt('/' + self.device + '/triggers/out/0/drive')

	@property
	def trigger_channel1_dir(self) -> int:
		return self.daq.getInt('/' + self.device + '/triggers/out/1/drive')

	@trigger_channel0_dir.setter
	def trigger_channel0_dir(self, dir):
		self.daq.setInt('/' + self.device + '/triggers/out/0/drive', dir)

	@trigger_channel1_dir.setter
	def trigger_channel1_dir(self, dir):
		self.daq.setInt('/' + self.device + '/triggers/out/1/drive', dir)

	@property
	def delay(self):
		delay_samp = self.daq.getInt('/' + self.device + '/qas/0/delay')
		return delay_samp / self.get_clock()

	@delay.setter
	def delay(self, delay):
		delay_samp = delay * self.get_clock()
		if delay_samp > 1020 or delay_samp < 0:
			logging.warning('Delay can not be bigger than {} or negative'.format(1020 / self.get_clock()))
		self.daq.setInt('/' + self.device + '/qas/0/delay', int(np.abs(delay_samp)))
	'''
	@property
	def default_delay(self):
		# Since the digitizing window length is limited define an additional delay, which will be user in the sequencer
		return self.daq.getInt('/' + self.device + '/awgs/0/userregs/{}'.format(self.default_delay_reg))

	@default_delay.setter
	def default_delay(self, delay):
		self.daq.setInt('/' + self.device + '/awgs/0/userregs/{}'.format(self.default_delay_reg), delay)
	'''
	def get_points(self) -> dict:
		points = {}
		if self.output_raw:
			points.update({'Voltage': [('Sample', np.arange(self.nsegm), ''),  # UHFQA stores only the averaged trace
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

	def get_dtype(self) -> dict:
		dtypes = {}
		if self.output_raw:
			dtypes.update({'Voltage': complex})
		if self.output_result:
			# Not sure if it's right to do it this way
			# it isn't
			dtypes.update({self.result_source + str(channel):
				self.daq.getList('/' + self.device + '/qas/0/result/data/' + str(channel) + '/wave')[0][1][0]['vector'].dtype
			for channel in range(self.ch_num)})

		return dtypes

	def get_status(self) -> int:
		return self.daq.getInt('/' + self.device + '/awgs/0/sequencer/status')

	# Main measurer method TODO write a proper docstring
	def measure(self) -> dict:
		# Just in case
		self.stop()

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
				print ("Acquisition failed with status {}".format(self.get_status()))
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

	def set_feature_iq(self, feature_id, feature) -> None:
		'''
		Use API to upload the demodulation weights
		:param channel: number of channel used to demodulate
		:param feature_real: I part of the weights
		:param feature_imag: Q part of the weights
		# Had to separate due to strange ZI setVector method issue
		# Should be defined separately
		'''
		feature = feature[:self.nsamp]/np.max(np.abs(feature[:self.nsamp]))
		feature_real = np.ascontiguousarray(np.real(feature))
		feature_imag = np.ascontiguousarray(np.imag(feature))
		self.daq.setVector('/' + self.device + '/qas/0/integration/weights/' + str(feature_id) + '/real', feature_real)
		self.daq.setVector('/' + self.device + '/qas/0/integration/weights/' + str(feature_id) + '/imag', feature_imag)

	@property
	def crosstalk_matrix(self) -> np.ndarray:
		matrix = np.zeros((10, 10), np.float)
		for raw in range(10):
			for column in range(10):
				matrix[raw][column] = self.daq.getDouble('/' + self.device + '/qas/0/crosstalk/rows/{}/cols/{}'.format(raw, column))

		return matrix

	@crosstalk_matrix.setter
	def crosstalk_matrix(self, matrix):
		raws, columns = np.asarray(matrix).shape
		if raws>10 or columns >10:
			raise ValueError('Maximum matrix size should be 10x10, while the given is {}x{}'.format(raws, columns))
		for raw_idx in range(raws):
			for col_idx in range(columns):
				self.daq.setDouble('/' + self.device + '/qas/0/crosstalk/rows/{}/cols/{}'.format(raw_idx, col_idx),
								matrix[raw_idx][col_idx])

	@property
	def crosstalk_bypass(self) -> bool:
		return self.daq.getInt('/' + self.device + '/qas/0/crosstalk/bypass') == 1

	@crosstalk_bypass.setter
	def crosstalk_bypass(self, status):
		self.daq.setInt('/' + self.device + '/qas/0/crosstalk/bypass', int(status))

	# UHFQA has it's own seqeunce
	def set_cur_prog(self, parameters, sequencer_idx):
		definition_fragments = []
		play_fragments = []

		if self.sync_mode:
			repetition_fragment = 'repeat(getUserReg(2))'
			#pre_wave_fragment = '''waitDigTrigger(2, 1);
			pre_wave_fragment = '''wait(2);
			setTrigger(AWG_INTEGRATION_ARM);
			'''
			post_wave_fragment = '''wait(getUserReg(9));
			setTrigger(AWG_MONITOR_TRIGGER + AWG_INTEGRATION_ARM + AWG_INTEGRATION_TRIGGER);			
			setTrigger(AWG_INTEGRATION_ARM);'''
		else:
			repetition_fragment = 'while(true)'
			pre_wave_fragment = ''
			post_wave_fragment = ''

		for wave_length_id, wave_length in enumerate(self.wave_lengths):
			if wave_length != 0:
				definition_fragments.append(textwrap.dedent('''
				wave w_marker_I_{wave_length} = join(marker(10, 1), marker({wave_length} - 10, 0));
				wave w_marker_Q_{wave_length} = join(marker(10, 2), marker({wave_length} - 10, 0));
				wave w_I_{wave_length} = zeros({wave_length}) + w_marker_I_{wave_length};
				wave w_Q_{wave_length} = zeros({wave_length}) + w_marker_Q_{wave_length};
				'''.format(wave_length=wave_length)))
				play_fragments.append(textwrap.dedent('''
				if (getUserReg({wave_length_reg}) == {wave_length_supersamp}) {{
					{repetition_fragment} {{
						setTrigger(1);
						{pre_wave_fragment}
						wait(getUserReg({pre_delay_reg}));
						playWave(w_I_{wave_length},w_Q_{wave_length});
						{post_wave_fragment}
						waitWave();
						wait({nsupersamp}-getUserReg({pre_delay_reg})-getUserReg({wave_length_reg}));
					}}
				}}
				''').format(repetition_fragment=repetition_fragment,
							pre_wave_fragment=pre_wave_fragment,
							post_wave_fragment=post_wave_fragment,
							wave_length=wave_length,
							wave_length_supersamp=wave_length // 8,
							**parameters))
			else:
				f = textwrap.dedent('''
				if (getUserReg({wave_length_reg}) == 0) {{
					{repetition_fragment} {{
						setTrigger(1);
						{pre_wave_fragment}
						{post_wave_fragment}
						wait({nsupersamp});
						waitWave();
					}}
				}}
				''').format(repetition_fragment=repetition_fragment,
							pre_wave_fragment=pre_wave_fragment,
							post_wave_fragment=post_wave_fragment,
							**parameters)
				play_fragments.append(f)
		self.current_programs[0] = ''.join(definition_fragments + play_fragments)
