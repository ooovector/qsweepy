import numpy as np

def generate_group(generators, error=1e-3):
	group = dict(generators)
	found = False
	while not found:
		#print (found)
		for name1, element1 in group.items():
			for name2, element2 in group.items():
				new_element = {'pulses': element2['pulses']+element1['pulses'],
								'unitary': np.dot(element1['unitary'], element2['unitary'])}
				new_element_normsqr  = np.sum(np.abs(new_element['unitary'])**2)
				found = False
				
				for name3, element3 in group.items():
					norm3sqr = np.sum(np.abs(element3['unitary']**2))
					scalar_product = np.abs(np.sum(element3['unitary']*np.conj(new_element['unitary'])))
					#print ('1:', name1, '2:', name2, 'new:', name3)
					if norm3sqr*new_element_normsqr-scalar_product**2<error:
					# this group element is already there					
						if len (new_element['pulses'])<len(element3['pulses']):
							del group[name3]
							found = False
						else:
							found = True
						#print (found)
						break
					
				if not found:
					break
			if not found:
				group['{} {}'.format(name2, name1)] = new_element
				#print ('{} {}'.format(name2, name1))
				break
		#print (found)
		
	return group