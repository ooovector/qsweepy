from .. import exdir_db
from pony.orm import db_session, select, desc

### TODO: load from SQL datum with attributes
#def load_attribute(db, attributes):
#	with db_session:
#		return select(float(frequency_entry.value) 
#		  for frequency_entry in db.Metadata 
#		  for resonator_id_entry in db.Metadata
#		  for datum_entry in db.Data 
#		  if frequency_entry.name=='fr' 
#		  and frequency_entry.data_id==datum_entry
#		  and resonator_id_entry.data_id==datum_entry
#		  and resonator_id_entry.name=='resonator_id'
#		  and resonator_id_entry.value==str(resonator_id)).order_by(lambda: desc(datum_entry.id)).first()

def fit_nd(dataset, parameters_nonlinear, parameters_linear = []):
	'''
		Perform fitting over an n=d dataset. For example:
			1) we have a single-tone spectrum as a function of coil current and an additional dimension for real/imag. We want to fit a two-dimensional array (real and imag) at every current point.
			parameters_nonlinear would be frequency, 
			parameters_linear would scattering_parameter (0 for real part, 1 for imag part)
	'''
	parameters_nonlinear_axes = {p:-1 for p in parameters_nonlinear}
	parameters_linear_axes = {p:-1 for p in parameters_linear}
	
	for parameter_id, parameter in dataset.parameters:
		if parameter.name in parameters_nonlinear:
			parameters_nonlinear_axes[parameter.name] = parameter_id
		elif parameter.name in parameters_linear:
			parameters_linear_axes[parameter.name] = parameter_id
	
	parameters_free = [p.name for p in data.parameters if p.name not in parameters_nonlinear and p.name not in parameters_linear]
	for k,v in parameters_linear_axes.items():
		if v == -1:
			raise ValueError ('Axis {} not found in dataset'.format(k))
	for k,v in parameters_nonlinear_axes.items():
		if v == -1:
			raise ValueError ('Axis {} not found in dataset'.format(k))	
	
	# append an index to all linear axes to the end and set all others to singleton
	newshape = [d if axis_id not in parameters_linear_axes.values else 1 for axis_id,d in enumerate(data.shape)]+[np.prod(parameters_linear_axes.values)]
	data_with_one_linear_parameter_axis = np.reshape(dataset.data, newshape)
	
	spectrum = np.nditer([dataset.data], op_axes=[[i for i in range(len(spectrum.parameters)) if i != axis_id]], flags=["multi_index"])
		

class tunable_resonance:
	def __init__(self, db, name, sample):
		self.sample = sample
		self.min_scan_freq_global = load_attribute(db, entry = 'min_scan_freq', filter = {'name':self.name}, sample_name=self.sample.name)
		self.max_scan_freq_global = load_attribute(db, entry = 'max_scan_freq', filter = {'name':self.name}, sample_name=self.sample.name)
		self.step_scan_freq_global = load_attribute(db, entry = 'step_scan_freq', filter = {'name':self.name}, sample_name=self.sample.name)
		self.min_cw_scan_power_global = load_attribute(db, entry = 'min_cw_scan_power', filter = {'name': self.name}, sample_name = self.sample.name)
		self.max_cw_scan_power_global = load_attribute(db, entry = 'max_cw_scan_power', filter = {'name': self.name}, sample_name = self.sample.name)
		self.min_scan_freq_local = lambda **p: self.min_scan_freq_global
		self.max_scan_freq_local = lambda **p: self.max_scan_freq_global
		self.step_scan_freq_local = lambda **p: self.step_scan_freq_global
		self.min_cw_scan_power_local = lambda **p: self.min_cw_scan_power_global
		self.max_cw_scan_power_local = lambda **p: self.max_cw_scan_power_global
		
		self.ex_devices = []
		self.ro_devices = []
		
	def fit_single_tone_spectroscopy(self, spectrum): ## as 1D
		axis_id = -1
		for parameter_id, parameters in spectrum.parameters:
			if parameter.name == 'frequency': axis_id = parameter_id
		if axis_id == -1:
			raise TypeError('No "frequency" axis in dataset')
			spectrum = np.nditer([spectrum.data], op_axes=[[i for i in range(len(spectrum.parameters)) if i != axis_id]], flags=["multi_index"])
		
	def single_tone_spectroscopy(self):
		sweeper = self.sample.sweeper
		sweeper.sweep()
		
class resonator(tunable_resonance):
	def __init__(self, db, name, sample):
		super().__init__(db, name, sample)
	
		
class transmon(tunable_resonance):
	def __init__(self, db, name, sample, readout_resonator):
		super().__init__(db, name, sample)
		self.readout_resonator = readout_resonator
		
class input_wire_rf():
	def __init__(self):
		self.devices = []
	def set_device(device):
		self.devices.append(device)

class output_wire_rf():
	def set_device(device):
		self.device = device
		
class output_wire():
	def set_device(device):
		self.device = device
		
class two_transmon_chip(sample):
	def __init__(self, db, name, default_name='two_transmon_chip', **kwargs):
		self.db = db

		self.ro_in =  input_wire_rf()
		self.ro_out = output_wire_rf()
		self.ex1 =    output_wire_rf()
		self.ex2 =    output_wire_rf()
		self.c_bias = output_wire()
		self.c_rf =   output_wire_rf()

		self.r1 = resonator(self.db, 'r1', self)
		self.r2 = resonator(self.db, 'r2', self)
		self.rc = resonator(self.db, 'rc', self)
		self.q1 = transmon(self.db, 'q1', self, readout_resonator=self.r1)
		self.q2 = transmon(self.db, 'q2', self, readout_resonator=self.r2)
		self.c =  transmon(self.db, 'c', self, readout_resonator=self.rc)

# abstract class for parameters
class parameter:
	#def immediate_set():
	#def scheduled_set():
	def set():
	
class tunable_frequency:
	def __init__(self, exdir_db, parameters):
		self.parameters = parameters
		self.values = 

class tunable_coupling_transmons:
	def __init__(self, exdir_db, name_priority_list):
		self.name_priority_list = name_priority_list
		self.exdir_db = exdir_db
	
	def get_resonator_frequency(self, resonator_id):
		with db_session:
			return select(float(frequency_entry.value) 
					  for frequency_entry in self.exdir_db.db.Metadata 
					  for resonator_id_entry in self.exdir_db.db.Metadata
					  for datum_entry in self.exdir_db.db.Data 
					  if frequency_entry.name=='fr' 
					  and frequency_entry.data_id==datum_entry
					  and resonator_id_entry.data_id==datum_entry
					  and resonator_id_entry.name=='resonator_id'
					  and resonator_id_entry.value==str(resonator_id)).order_by(lambda: desc(datum_entry.id)).first()
									
	def set_resonator_frequency(self, resonator_id, value):
		self.exdir_db.save({'resonator_id':str(resonator_id), 'fr':str(value)})

	def get_resonator_frequency_min(self, resonator_id):
		with db_session:
			fr_min = select(float(frequency_entry.value) 
					  for frequency_entry in self.exdir_db.db.Metadata 
					  for resonator_id_entry in self.exdir_db.db.Metadata
					  for datum_entry in self.exdir_db.db.Data 
					  if frequency_entry.name=='fr_min' 
					  and frequency_entry.data_id==datum_entry
					  and resonator_id_entry.data_id==datum_entry
					  and resonator_id_entry.name=='resonator_id'
					  and resonator_id_entry.value==str(resonator_id)).order_by(lambda: desc(datum_entry.id)).first()
					  
			return fr_min
	
	def get_resonator_frequency_max(self, resonator_id_max)