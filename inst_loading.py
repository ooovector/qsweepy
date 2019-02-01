from . import data_reduce
import numpy as np
from . import qjson


#you should run this script and every device start to working rigth
class instrument_setups:
	def __init__(self, pna, lo, dac, adc, **kwargs):
		self.lo_ro = pna
		self.lo_ex = lo
		self.awg = dac
		self.awg_chanels = dict()
		self.adc = adc
		#???think about every None man
		self.iq_ex = None
		self.iq_ro = None
		self.pg = None
		##
		self.params = kwargs#qubit_id
		self.qubits = qjson.load("qubits","qubit_params")
		#self.readout = qjson.load('setups','readout')
		#maybe should we write all constant params in one place? into config?
		#for osc
		self.lo_ex_pow = 14
		self.lo_ro_pow = 16
		#for awg
		self.ex_clock = 1e9
		self.ro_clock = 1e9
		self.rep_rate = 20e3 # частота повторений эксперимента
		#interm fr
		self.ro_if = 75e6
		self.lo_freq = 4.17e9
		self.ex_if = None
		#for trigger
		self.marker_length = 100
		self.readout_trigger_delay = 185
		self.trg_length = 10e-9
		#run all functions
		self.setter()
		
	def setter(self):
		self.set_osc()
		self.set_awg_PXI()
		self.set_adc_PXI()
		self.set_pulses_PXI()
		self.trigger()
		self.set_adc_filters()
	def set_osc(self):
		self.lo_ex.set_status(1)
		self.lo_ex.set_power(self.lo_ex_pow)
		#for pna
		self.lo_ro.set_power(self.lo_ro_pow)
		self.lo_ro.write("OUTP ON")
		self.lo_ro.write("SOUR1:POW1:MODE ON")
		self.lo_ro.write("SOUR1:POW2:MODE OFF")
		self.lo_ro.set_sweep_mode("CW")
	
	def set_awg_PXI(self):
		self.awg.stop()
		self.awg.set_offset(0.2, channel=1)
		self.awg.set_offset(0.2, channel=3)
		self.awg.set_output(1, channel=1)
		self.awg.set_output(1, channel=3)
		self.awg.set_nop(self.ex_clock/self.rep_rate) 
		self.awg.run()
		#?awg_tek = awg
		# channel 0 is master and triggers all others
		self.awg.trigger_source_types = [0, 6,6,6]
		self.awg.trigger_source_channels = [0, 4000,4000,4000] # pxi trigger 0 
		self.awg.trigger_delays = [40, 0,0,0] # master channel should wait 400 ns for others to start
		self.awg.trigger_behaviours = [0,4,4,4] #rising edge trigger
		for channel in range(0,4):
			self.awg.set_amplitude(.2, channel=channel)
			self.awg.set_offset(0, channel=channel)
			self.awg.set_output(1, channel=channel)
			self.awg.set_waveform(waveform=[0]*self.awg.get_nop(), channel=channel)
	
	def set_adc_PXI(self):
		self.adc.set_input(channel=1, input=1)
		self.adc.set_input(channel=2, input=1)
		self.adc.set_trigger_external(channel=1)
		self.adc.set_trigger_external(channel=2)
		
	def set_pulses_PXI(self):
		self.lo_ex.set_frequency(self.lo_freq)
		self.ex_if = self.lo_freq-self.qubits[self.params['qubit_id']]['q']['00-1-01']
		self.iq_ex = awg_iq_multi.awg_iq_multi(self.awg, self.awg, 0, 1, self.lo_ex)
		for tr,freq in self.qubits[self.params['qubit_id']]['q'].items():
			self.iq_ex.carriers[tr] = awg_iq_multi.carrier(self.iq_ex)
			self.iq_ex.carriers[tr].set_frequency(freq)
			self.awg_channels['iq_ex_'+tr]=self.iq_ex.carriers[tr]
			self.iq_ro = awg_iq.awg_iq(self.awg, self.awg, 2, 3, self.lo_ro)
		self.iq_ro.set_if(self.ro_if)
		self.iq_ro.set_sideband_id(-1)
		self.iq_ro.set_frequency(self.qubits[self.params['qubit_id']]['r']['Fr'])
		self.awg_channels['iq_ro'] = self.iq_ro
		self.pg = pulses.pulses(self.awg_channels)
		
	def set_trigger(self):
		self.awg.set_marker(length=marker_length, delay=0, channel=0)
		ro_trg = awg_digital.awg_digital(self.awg, 1)
		ro_trg.mode = 'set_delay'
		ro_trg.delay_setter = lambda x: self.adc.set_trigger_delay(int(x*self.adc.get_clock()/self.iq_ex.get_clock()-self.readout_trigger_delay))
		self.awg_channels['ro_trg'] = ro_trg
	
	def set_adc_filters(self):
		adc_reducer = data_reduce.data_reduce(self.adc)
		adc_reducer.filters['Mean Voltage (AC)'] = data_reduce.mean_reducer_noavg(self.adc, 'Voltage', 0)
		# #adc_reducer.filters['Std Voltage (AC)'] = data_reduce.mean_reducer_noavg(adc, 'Voltage std', 0)
		adc_reducer.filters['S21+'] = data_reduce.mean_reducer_freq(self.adc, 'Voltage', 0, self.iq_ro.get_if())
		adc_reducer.filters['S21-'] = data_reduce.mean_reducer_freq(self.adc, 'Voltage', 0, -self.iq_ro.get_if())#kill it
		adc_reducer.extra_opts['scatter'] = True
		self.adc = adc_reducer
		