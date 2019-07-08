import numpy as np
from . import pulses

class joined_readout:
	def __init__(self, drag_hds):
		self.ex_channels  = [drag_hds[key].tld.ex_channels[0] for key in drag_hds.keys()]
		self.qubit_number = len(self.ex_channels)
		self.pi_amplitudes = np.reshape([drag_hds[key].get_amplitude(np.pi) for key in drag_hds.keys()], (self.qubit_number))
		self.sigma = [drag_hds[key].sigma for key in drag_hds.keys()]
		self.alpha = [drag_hds[key].alpha for key in drag_hds.keys()]
		self.length = [drag_hds[key].length for key in drag_hds.keys()][0]
		self.ro_sequence = [drag_hds[key].tld.ro_sequence for key in drag_hds.keys()][0]
		self.readout_device = [drag_hds[key].tld.readout_device for key in drag_hds.keys()][0]
		self.pulse_sequencer = [drag_hds[key].tld.pulse_sequencer for key in drag_hds.keys()][0]
		self.drag_hds = drag_hds
		self.states = {}
		
	def generate_all_states(self):
		result = []
		for i in range(0, np.power(2, self.qubit_number)): result.append(str(bin(i))[2:])
		for i in range(0, self.qubit_number): 
			while len(result[i]) < self.qubit_number: result[i] = '0' + result[i]
		#print(result)
		return result

	def get_disp_shifts(self):
		readout_begin = self.length
		pg = self.pulse_sequencer
		sequence = []
		sequence_masks = self.generate_all_states()
		for seq_mask in sequence_masks:
			pre_pulse_seq = []
			for index in range(0, len(seq_mask)):
				if int(seq_mask[index]):
					pre_pulse_seq += self.drag_hds[str(index+1)].get_pulse_seq(np.pi, 0)
				else:
					pre_pulse_seq += self.drag_hds[str(index+1)].get_pulse_seq(0.0,   0)
			#channel_pulses = [(self.ex_channels[index], pg.gauss_hd, self.pi_amplitudes[index]*int(seq_mask[index]), self.sigma[index], self.alpha[index]) for index in range(0, len(seq_mask))]
			pg.set_seq(pre_pulse_seq + self.ro_sequence)
			self.states.update({seq_mask: self.readout_device.measure()})
		return 