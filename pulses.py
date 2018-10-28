from scipy.signal import gaussian
from scipy.signal import tukey
from scipy.signal import hann
import numpy as np

class pulses:
	def __init__(self, channels = {}):
		self.channels = channels
		self.settings = {}
	
	## generate waveform of a gaussian pulse with quadrature phase mixin
	def gauss_hd (self, channel, length, amp_x, sigma, alpha=0.):
		gauss = gaussian(int(round(length*self.channels[channel].get_clock())), sigma*self.channels[channel].get_clock())
		gauss -= gauss[0]
		gauss_der = np.gradient (gauss)*self.channels[channel].get_clock()
		return amp_x*(gauss + 1j*gauss_der*alpha)
		
	# def rect_cos (self, channel, length, amp, alpha=0.):
		# alfa = 0.5
		# impulse = tukey(int(round(length*self.channels[channel].get_clock())), alfa)
		# #print(alfa*self.channels[channel].get_clock())
		# #print(length)
		# #print(round(length*self.channels[channel].get_clock()))
		# impulse -= impulse[0]
		# impulse_der = np.gradient(impulse)*self.channels[channel].get_clock()
		# return amp*(impulse + 1j*impulse_der*alpha)
		
	def rect_cos (self, channel, length, amp, length_tail, alpha=0.):
		length_of_plato = length - length_tail*2
		length_of_one_tail = int(length_tail*self.channels[channel].get_clock())
		hann_function = hann(2*length_of_one_tail)
		first = hann_function[:length_of_one_tail]
		second = hann_function[length_of_one_tail:]
		plato = np.ones(int(round(length_of_plato*self.channels[channel].get_clock())))
		final = first.tolist() 
		final.extend(plato.tolist())
		final.extend(second.tolist())
		impulse = np.asarray(final)
		impulse -= impulse[0]
		impulse_der = np.gradient(impulse)*self.channels[channel].get_clock()
		#print(self.channels[channel].get_clock())
		#print(length_tail*self.channels[channel].get_clock())
		#print(first)
		#print(second)
		#print(plato)
		return amp*(impulse + 1j*impulse_der*alpha)
		
	## generate waveform of a rectangular pulse
	def rect(self, channel, length, amplitude):
		return amplitude*np.ones(int(round(length*self.channels[channel].get_clock())), dtype=np.complex)

	def pause(self, channel, length):
		return self.rect(channel, length, 0)
		
	def p(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
		
	def ps(self, channel, length, pulse_type=None, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		if channel:
			pulses[channel] = pulse_type(channel, length, *params)
		return pulses
		
	def pmulti(self, length, *params):
		pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
		for pulse in params:
			channel = pulse[0]
			pulses[channel] = pulse[1](channel, length, pulse[2:])
		return pulses
	
	def awg(self, channel, length, waveform):
		return waveform
	
	def set_seq(self, seq, force=True):
		initial_delay = 1e-6
		final_delay = 1e-6
		pulse_seq_padded = [self.p(None, initial_delay, None)]+seq+[self.p(None, final_delay, None)]
	
		try:
			for channel, channel_device in self.channels.items():
				channel_device.freeze()
	
			pulse_shape = {k:[] for k in self.channels.keys()}
			for channel, channel_device in self.channels.items():
				for pulse in pulse_seq_padded:
					pulse_shape[channel].extend(pulse[channel])
				pulse_shape[channel] = np.asarray(pulse_shape[channel])
		
				if len(pulse_shape[channel])>channel_device.get_nop():
					tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
					tmp = pulse_shape[channel][-channel_device.get_nop():]
					pulse_shape[channel] = tmp
					raise(ValueError('pulse sequence too long'))
				else:
					tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
					tmp[-len(pulse_shape[channel]):]=pulse_shape[channel]
					pulse_shape[channel] = tmp
				#print (channel, pulse_shape[channel], len(pulse_shape[channel]))
				#print ('Calling set_waveform on device '+channel)
				channel_device.set_waveform(pulse_shape[channel])

		finally:
			for channel, channel_device in self.channels.items():
				channel_device.unfreeze()
				
		devices = []
		for channel in self.channels.values():
			devices.extend(channel.get_physical_devices())
		for device in list(set(devices)):
			device.run()