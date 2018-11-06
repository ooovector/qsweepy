from . import sweep
import numpy as np
import matplotlib.pyplot as plt

class interleaved_benchmarking:
	def __init__(self, tomo, random_sequence_num=8):
		self.tomo = tomo
		self.interleavers = {}
		self.initial_state_vector = np.asarray([1, 0]).T
		self.random_sequence_num = random_sequence_num
		self.sequence_length = 20
		
		self.target_gate = []
		self.target_gate_unitary = np.asarray([[1,0],[0,1]], dtype=np.complex)
		self.target_gate_name = 'Identity (benchmarking)'
		
		self.reference_benchmark_result = None
		self.interleaving_sequences = None
		
		self.final_ground_state_rotation = True
	
	# used to transformed any of the |0>, |1>, |+>, |->, |i+>, |i-> states into the |0> state
	# low-budget function only appropiate for clifford benchmarking
	# higher budget functions require arbitrary rotation pulse generator
	# which we unfortunately don't have (yet)
	# maybe Chernogolovka control experiment will do this
	def state_to_zero_transformer(self, psi):
		good_interleavers_length = {}
		# try each of the interleavers available
		for name, interleaver in self.interleavers.items():
			result = np.dot(interleaver['unitary'], psi)
			if (1-np.abs(np.dot(result, self.initial_state_vector)))<1e-6:
				# if the gate does what we want than we append it to our pulse list
				good_interleavers_length[name] = len(interleaver['pulses'])
		# check our pulse list for the shortest pulse
		# return the name of the best interleaver
		# the whole 'pulse name' logic is stupid
		# if gates are better parameterized by numbers rather than names (which is true for non-Cliffords)
		name = min(good_interleavers_length, key=good_interleavers_length.get)
		return name, self.interleavers[name]
	
	def set_sequence_length(self, sequence_length):
		self.sequence_length = sequence_length
		
	def set_sequence_length_and_regenerate(self, sequence_length):
		self.sequence_length = sequence_length
		self.prepare_random_interleaving_sequences()
	
	def set_target_pulse(self, x):
		self.target_gate = x
	
	def prepare_random_interleaving_sequences(self):
		self.interleaving_sequences = [self.generate_random_interleaver_sequence(self.sequence_length) for i in range(self.random_sequence_num)]
	
	def add_intervealer(self, name, pulse_seq, unitary):
		self.interleavers[name] = {'pulses': pulse_seq, 'unitary': unitary}
		
	def generate_interleaver_sequence_from_names(self, names):
		sequence_pulses = [self.interleavers[k]['pulses'] for k in names]
		sequence_unitaries = [self.inbterleavers[k]['unitary'] for k in names]
		
		psi = self.initial_state_vector.copy()
		for U in sequence_unitaries:
			psi = np.dot(U, psi)
    
		rho = np.einsum('i,j->ij', np.conj(psi), psi)
		
		return {'Gate names':names,
				'Gate unitaries': sequence_unitaries,
				'Pulse sequence': sequence_pulses,
				'Final state vector': psi,
				'Final state matrix': rho}
	
	def generate_random_interleaver_sequence(self, n):
		ilk = [k for k in self.interleavers.keys()]
		ilv = [self.interleavers[k] for k in ilk]
		sequence = np.random.randint (len(ilv), size = n).tolist()
		sequence_pulses = [j for i in [ilv[i]['pulses'] for i in sequence] for j in i]
		sequence_unitaries = [ilv[i]['unitary'] for i in sequence]
		sequence_gate_names = [ilk[i] for i in sequence]
		
		psi = self.initial_state_vector.copy()
		for U in sequence_unitaries:
			psi = np.dot(U, psi)
    
		rho = np.einsum('i,j->ij', np.conj(psi), psi)
		
		return {'Gate names':sequence_gate_names,
				'Gate unitaries': sequence_unitaries,
				'Pulse sequence': sequence_pulses,
				'Final state vector': psi,
				'Final state matrix': rho}
		
	# TODO: density matrix evolution operator for non-hamiltonian mechnics
	def interleave(self, sequence_gate_names, pulse, unitary, gate_name):
		sequence_unitaries = [self.interleavers[i]['unitary'] for i in sequence_gate_names]
		
		sequence_pulses = []
		interleaved_sequence_gate_names = []
		psi = self.initial_state_vector.copy()
		for i in sequence_gate_names:
			psi = np.dot(unitary, np.dot(self.interleavers[i]['unitary'], psi))
			sequence_pulses.extend(self.interleavers[i]['pulses'])
			sequence_pulses.extend(pulse)
			interleaved_sequence_gate_names.append(i)
			interleaved_sequence_gate_names.append(gate_name)
			
		if self.final_ground_state_rotation:
			final_rotation_name, final_rotation = self.state_to_zero_transformer(psi)
		
		# we need something separate from the interleavers for the final gate
		# the logic is in principle OK but the final gate should be a function, n
		psi = np.dot(final_rotation['unitary'], psi)
		sequence_pulses.extend(final_rotation['pulses'])
		interleaved_sequence_gate_names.append(final_rotation_name)
			
		rho = np.einsum('i,j->ij', np.conj(psi), psi)
		
		return {'Gate names':interleaved_sequence_gate_names,
				'Gate unitaries': sequence_unitaries,
				'Pulse sequence': sequence_pulses,
				'Final state vector': psi,
				'Final state matrix': rho}
				
#	def set_target_gate_pulse(self, pulse):
#		self.target_gate = pulse

#	def set_interleaving_sequences(self, sequences):

	def reference_benchmark(self):
		old_target_gate = self.target_gate
		old_target_gate_unitary = self.target_gate_unitary
		old_target_gate_name = self.target_gate_name
		self.target_gate = []
		self.target_gate_unitary = np.asarray([[1,0],[0,1]], dtype=np.complex)
		self.target_gate_name = 'Identity (benchmarking)'
		
		self.reference_benchmark_result = self.measure()
		
		self.target_gate = old_target_gate
		self.target_gate_unitary = old_target_gate_unitary
		self.target_gate_name = old_target_gate_name
		
		return self.reference_benchmark_result

	def get_dtype(self):
		dtypes = self.tomo.get_dtype()
		dtypes['distance'] = float
		dtypes['Pulse sequence'] = object
		dtypes.update({'{0} fit'.format(a):float for a in self.tomo.reconstruction_basis.keys()})
		return dtypes
		
	def get_opts(self):
		dtypes = self.tomo.get_opts()
		dtypes['distance'] = {}
		dtypes['Pulse sequence'] = {}
		#dtypes.update({'{0} fit'.format(a):{} for a in self.tomo.reconstruction_basis.keys()})
		return dtypes
		
	def get_points(self):
		points = tuple([])
		_points = {a:points for a in self.tomo.get_points().keys()}
		_points['distance'] = tuple([])
		interleaved = [ self.interleave(i['Gate names'], self.target_gate, self.target_gate_unitary, self.target_gate_name) for i in self.interleaving_sequences]
		_points['Pulse sequence'] = tuple([])#[0], (('Pulse id'), np.arange(len(interleaved[0]['Gate names'])), ''),)
		#_points.update({'{0} fit'.format(a):points for a in self.tomo.reconstruction_basis.keys()})
		return _points
		
	def set_interleaved_sequence(self, seq_id):
		seq = self.interleave(self.interleaving_sequences[seq_id]['Gate names'], self.target_gate, self.target_gate_unitary, self.target_gate_name)
		self.tomo.set_prepare_seq(seq['Pulse sequence'])
		self.current_seq = seq
		
	def measure(self):
		#if self.interleaving_sequences:
		#il_seqs = self.interleaving_sequences
		#else:
		#	il_seqs = [self.generate_random_interleaver_sequence(self.sequence_length) for i in range(self.random_sequence_num)]
		#interleaved = [ self.interleave(i['Gate names'], self.target_gate, self.target_gate_unitary, self.target_gate_name) for i in il_seqs]

		measurement = self.tomo.measure()		
		#def set_interleaved_sequence(seq_id):
		#	self.tomo.set_prepare_seq(interleaved[seq_id]['Pulse sequence'])
		
		#results = sweep.sweep(self.tomo, (np.arange(len(interleaved)), set_interleaved_sequence, 'Interleaving sequence id'), 
		#						output=False, 
		#						plot=False, header='Interleaved benchmarking sequence sweep')
		
		#calculating euclidean distance metric
		axes = [a for a in self.tomo.reconstruction_basis.keys()]
		expected_projections = {a:np.trace(np.dot(self.tomo.reconstruction_basis[a]['operator'], self.current_seq['Final state matrix'])) for a in axes}
									
		#measurement.update({'{0} fit'.format(i):np.asarray([np.real(expected_projections[i])]) for i in axes})
		measurement['distance'] = np.asarray([np.real(np.sqrt(np.sum([(np.real(expected_projections[a] - measurement[a]))**2 for a in axes], axis=0)))])
		measurement['Pulse sequence'] = np.array([object()])
		measurement['Pulse sequence'][0] = self.current_seq['Gate names']
		
		return measurement
		#print (expected_projections, results)
		#print (results[axes[0]])
		#print (results[axes[0]][0])
		#print (results[axes[0]][1])
		#print (np.real(np.sqrt(np.sum([(expected_projections[a] - results[a][2])**2 for a in axes], axis=0))))
		#print (results[axes[0]][3])
		#results['Euclidean distance'] = (results[axes[0]][0],  
		#								results[axes[0]][1],
		#						np.real(np.sqrt(np.sum([(expected_projections[a] - results[a][2])**2 for a in axes], axis=0))),
		#								results[axes[0]][3])
		
		#results['Mean Euclidean distance'] = ([],  
		#								[],
		#						np.mean(np.real(np.sqrt(np.sum([(expected_projections[a] - results[a][2])**2 for a in axes], axis=0)))),
		#								[])
		
		#results['Pulse sequences']=((results[axes[0]][0], 'Gate names'), 
		#							(results[axes[0]][1], np.arange(len(interleaved[0]['Gate names']))),
		#							np.asarray([i['Gate names'] for i in interleaved]),
		#							results[axes[0]][3])
		#results['Pulse sequences']=((results[axes[0]][0]), 
		#							(results[axes[0]][1]),
		#							np.array([i['Gate names'] for i in interleaved]),
		#							results[axes[0]][3])
		
		#results.update({'{0} fit'.format(i):(results[i][0], results[i][1], np.real(expected_projections[i]), results[i][3]) for i in axes})
		
		# calculating fidelities of measured projections
		#psi = seq['Final state vector']
		#results['Fidelity'] = np.sqrt(np.sum([np.einsum('i,ij,jk,kl,j->', np.conj(psi), np.conj(Ut['unitary']), rho, Ut['unitary'], psi) \
		#							for rho, Ut in zip(, self.tomo.proj_seq)]))
		
		#for a in axes:
		#	plt.close(a)
		#for p in self.tomo.proj_seq.keys():
		#	plt.close(a)
		
		#return {k:v[2] for k,v in results.items()}