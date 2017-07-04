import sweep
import numpy as np
import matplotlib.pyplot as plt

class interleaved_benchmarking:
	def __init__(self, tomo, random_sequence_num=8):
		self.tomo = tomo
		self.interleavers = {}
		self.initial_state_vector = np.asarray([0, 1]).T
		self.random_sequence_num = random_sequence_num
		self.sequence_length = 20
		
		self.target_gate = []
		self.target_gate_unitary = np.asarray([[1,0],[0,1]], dtype=np.complex)
		self.target_gate_name = 'Identity (benchmarking)'
		
		self.reference_benchmark_result = None
		self.interleaving_sequences = None
	
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
		dtypes['Euclidean distance'] = float
		dtypes['Mean Euclidean distance'] = float
		dtypes['Pulse sequences'] = object
		dtypes.update({'{0} fit'.format(a):float for a in self.tomo.reconstruction_basis.keys()})
		return dtypes
		
	def get_opts(self):
		dtypes = self.tomo.get_opts()
		dtypes['Euclidean distance'] = {}
		dtypes['Mean Euclidean distance'] = {}
		dtypes['Pulse sequences'] = {}
		dtypes.update({'{0} fit'.format(a):{} for a in self.tomo.reconstruction_basis.keys()})
		return dtypes
		
	def get_points(self):
		points = (('Interleaving sequence id', np.arange(len(self.interleaving_sequences))),)
		_points = {a:points for a in self.tomo.get_points().keys()}
		_points['Euclidean distance'] = points
		_points['Mean Euclidean distance'] = tuple([])
		interleaved = [ self.interleave(i['Gate names'], self.target_gate, self.target_gate_unitary, self.target_gate_name) for i in self.interleaving_sequences]
		_points['Pulse sequences'] = (points[0], (('Pulse id'), np.arange(len(interleaved[0]['Gate names']))))
		_points.update({'{0} fit'.format(a):points for a in self.tomo.reconstruction_basis.keys()})
		return _points
		
	def measure(self):
		#if self.interleaving_sequences:
		il_seqs = self.interleaving_sequences
		#else:
		#	il_seqs = [self.generate_random_interleaver_sequence(self.sequence_length) for i in range(self.random_sequence_num)]
		interleaved = [ self.interleave(i['Gate names'], self.target_gate, self.target_gate_unitary, self.target_gate_name) for i in il_seqs]

		def set_interleaved_sequence(seq_id):
			self.tomo.set_prepare_seq(interleaved[seq_id]['Pulse sequence'])
		
		results = sweep.sweep(self.tomo, (np.arange(len(interleaved)), set_interleaved_sequence, 'Interleaving sequence id'), 
								output=False, 
								plot=False, header='Interleaved benchmarking sequence sweep')
		
		#calculating euclidean distance metric
		axes = [a for a in self.tomo.reconstruction_basis.keys()]
		expected_projections = {a:[np.trace(np.dot(self.tomo.reconstruction_basis[a]['operator'], seq['Final state matrix'])) \
									for seq in interleaved]	for a in axes}
		#print (expected_projections, results)
		results['Euclidean distance'] = (results[axes[0]][0],  
										results[axes[0]][1],
								np.real(np.sqrt(np.sum([(expected_projections[a] - results[a][2])**2 for a in axes], axis=0))),
										results[axes[0]][3])
		
		results['Mean Euclidean distance'] = ([],  
										[],
								np.mean(np.real(np.sqrt(np.sum([(expected_projections[a] - results[a][2])**2 for a in axes], axis=0)))),
										[])
		
		results['Pulse sequences']=(results[axes[0]][0],  
										results[axes[0]][1],
										np.asarray([i['Gate names'] for i in interleaved]),
										results[axes[0]][3])
		
		results.update({'{0} fit'.format(i):(results[i][0], results[i][1], np.real(expected_projections[i]), results[i][3]) for i in axes})
		
		# calculating fidelities of measured projections
		#psi = seq['Final state vector']
		#results['Fidelity'] = np.sqrt(np.sum([np.einsum('i,ij,jk,kl,j->', np.conj(psi), np.conj(Ut['unitary']), rho, Ut['unitary'], psi) \
		#							for rho, Ut in zip(, self.tomo.proj_seq)]))
		
		#for a in axes:
		#	plt.close(a)
		#for p in self.tomo.proj_seq.keys():
		#	plt.close(a)
		
		return {k:v[2] for k,v in results.items()}