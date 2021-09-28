import numpy as np
import copy

def two_qubit_clifford(generators_q1, generators_q2, plus_op_parallel, two_qubit_gate=None,
					   two_qubit_gate_name='fSim', error=1e-3, Clifford_group=False):
	# see https://arxiv.org/pdf/1210.7011.pdf
	c_q1 = generate_group(generators=generators_q1, Clifford_group=Clifford_group)
	c_q2 = generate_group(generators=generators_q2, Clifford_group=Clifford_group)
	s_q1 = {}
	s_q2 = {}
	pi2_q1 = {}
	pi2_q2 = {}
	# finding s-gates
	def find_s_gates_from_cliffords(cliffords):
		for name, clifford in cliffords.items():
			# has no 2-nd-order axis
			if not np.abs(np.abs(np.trace(clifford['unitary'] @ clifford['unitary'])) - clifford['unitary'].shape[0]) < error:
				# has a 3-rd order axis
				if np.abs(np.abs(np.trace(clifford['unitary'] @ clifford['unitary'] @ clifford['unitary'])) - clifford['unitary'].shape[0]) < error:
					generator = clifford
		s = {}
		s0 = generator['unitary'] @ generator['unitary'] @ generator['unitary']
		s1 = generator['unitary']
		s2 = generator['unitary'] @ generator['unitary']
		norm = np.abs(np.trace(s0))
		for name, clifford in cliffords.items():
			if np.abs(np.abs(np.sum(s0 * np.conj(clifford['unitary']))) - norm)<error:
				s[name] = clifford
			if np.abs(np.abs(np.sum(s1 * np.conj(clifford['unitary']))) - norm) < error:
				s[name] = clifford
			if np.abs(np.abs(np.sum(s2 * np.conj(clifford['unitary']))) - norm) < error:
				s[name] = clifford
		return s
	#s_q1 = find_s_gates_from_cliffords(c_q1)
	#s_q2 = find_s_gates_from_cliffords(c_q2)
	#print ('s_q1 length:', len(s_q1))
	# finding pi/2-gates
	for name, clifford in c_q1.items():
		square = clifford['unitary'] @ clifford['unitary']
		if np.abs(np.trace(square)) - clifford['unitary'].shape[0] > 1e-3: # not a pauli gate
			if np.abs(np.trace(square @ square)) - clifford['unitary'].shape[0] < 1e-3: # 4-fold symmetry
				pi2_q1[name] = clifford

	for name, clifford in c_q2.items():
		square = clifford['unitary'] @ clifford['unitary']
		if np.abs(np.trace(square)) - clifford['unitary'].shape[0] > 1e-3:  # not a pauli gate
			if np.abs(np.trace(square @ square)) - clifford['unitary'].shape[0] < 1e-3:  # 4-fold symmetry
				pi2_q2[name] = clifford

	group = {}
	#tensor product
	for name1, clifford1 in c_q1.items():
		for name2, clifford2 in c_q2.items():
			if name1==name2:
				group[name1] = {'unitary': clifford1['unitary']}
				#				'pulses': [plus_op_parallel(clifford1['pulses'], clifford2['pulses'])]}
				#group[name1]['unitary'] = clifford1['unitary']
				group[name1]['pulses'] = copy.copy(clifford1['pulses'])
				for j in range(len(group[name1]['pulses'])):
					group[name1]['pulses'][j] = plus_op_parallel(clifford1['pulses'][j], clifford2['pulses'][j])
				#group[name1+' '+name2] = {'unitary':clifford1['unitary']@clifford2['unitary'],
				#					  'pulses':plus_op_parallel(clifford1['pulses'],clifford2['pulses'])}
	if two_qubit_gate is None:
		return group
	else:
		group[two_qubit_gate_name] = {'unitary': two_qubit_gate['unitary'],
								'pulses': two_qubit_gate['pulses']}
		return group

	#return group

def generate_group(generators, error=1e-3, Clifford_group=True):

	group = dict(generators)
	if not Clifford_group:
		return group
	else:
		found = False
		while not found:
			#print (found)
			for name1, element1 in group.items():
				for name2, element2 in group.items():
					new_element = {'unitary': np.dot(element1['unitary'], element2['unitary']),
							   'price': element1['price']+element2['price']}
					new_element_normsqr  = np.sum(np.abs(new_element['unitary'])**2)
					found = False
				
					for name3, element3 in group.items():
						norm3sqr = np.sum(np.abs(element3['unitary']**2))
						scalar_product = np.abs(np.sum(element3['unitary']*np.conj(new_element['unitary'])))
						#print ('1:', name1, '2:', name2, 'new:', name3)
						if norm3sqr*new_element_normsqr-scalar_product**2<error:
						# this group element is already there
							if new_element['price']<element3['price']:
								del group[name3]
								found = False
							else:
								found = True
							#print (found)
							break

					if not found:
						break
				if not found:
					new_element['pulses'] = element2['pulses']+element1['pulses']
					group['{} {}'.format(name2, name1)] = new_element
					print(len(group), name2, name1)#, new_element['unitary'])
					#print ('{} {}'.format(name2, name1))
					break
			#print (found)
		return group