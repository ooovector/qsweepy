from pony.orm import *
from time import gmtime, strftime
from .data_structures import *
from datetime import datetime

class database:
	def __init__(self, provider='postgres', user='qsweepy', password='qsweepy', host='localhost', database='qsweepy', port = 5432):
		db = Database()
		class Data(db.Entity):
			id = PrimaryKey(int, auto=True)
			comment = Optional(str)
			measurement_type = Required(str)
			sample_name = Required(str)
			measurement_time = Optional(float)
			start = Required(datetime, precision=6)
			stop = Optional(datetime, precision=6)
			filename = Optional(str)
			type_revision = Optional(str)
			incomplete = Optional(bool)
			invalid = Optional(bool)
			owner = Optional(str)
			metadata = Set('Metadata')
			reference_one = Set('Reference', reverse = 'this')
			reference_two = Set('Reference', reverse = 'that')
			linear_sweep = Set('Linear_sweep')
		self.Data = Data
			
		class Metadata(db.Entity):
			id = PrimaryKey(int, auto=True)
			data_id = Required(self.Data)
			name = Required(str)
			value = Required(str)
			#data = Required(Data)
		self.Metadata = Metadata
			
		class Reference(db.Entity):
			#id = PrimaryKey(int, auto=True)
			this = Required(self.Data)
			that = Required(self.Data)
			ref_type = Required(str)
			PrimaryKey(this, that)
		self.Reference = Reference
			
		class Linear_sweep(db.Entity):
			data_id = Required(self.Data)
			min_value = Optional(float)
			max_value = Optional(float)
			num_points = Optional(int)
			parameter_name = Required(str)
			parameter_units = Optional(str)
		self.Linear_sweep = Linear_sweep
			
		class Invalidations(db.Entity):
			this_type = Required(str)
			that_type = Required(str)
			ref_type = Required(str)
			id = PrimaryKey(int, auto=True)
		self.Invalidations = Invalidations
	
		db.bind(provider, user=user, password=password, host=host, database=database, port = port)
		db.generate_mapping(create_tables=True)
	
	def create_in_database(self, state):
		d = self.Data(comment = state.comment, measurement_type = state.measurement_type, sample_name = state.sample_name, start = state.start,   
			filename = state.filename, type_revision = state.type_revision, owner = state.owner, incomplete = True)

		for dataset in state.datasets.keys():
			for parameter in state.datasets[dataset].parameters:
				minv = np.min(parameter.values)
				maxv = np.max(parameter.values)
				number = len(parameter.values)
				name = parameter.name
				unit = parameter.unit
				self.Linear_sweep(data_id = d, min_value = minv, max_value = maxv, num_points = number, parameter_name = name, parameter_units = unit)
    
		for name, value in state.metadata.items():
			#print(name, value)
			self.Metadata(data_id = d, name = name, value = value)
		print(state.references)
		for ref_id, ref_type in state.references.items():
			self.Reference(this = d, that = ref_id, ref_type = ref_type)
        
		commit()   
		state.id = d.id 
		return d.id
		
	def update_in_database(self, state):
		d = self.Data[state.id]
		d.comment = state.comment
		d.measurement_type = state.measurement_type
		d.sample_name = state.sample_name
		d.type_revision = state.type_revision
		d.incomplete = state.total_sweeps == state.done_sweeps
		d.measurement_time = state.measurement_time
		d.start = state.start
		d.stop = state.stop
		d.filename = state.filename
		commit()    
		return d.id
		
	def get_from_database(self, filename = ''):
		#print(select(i for i in self.Data))
		id = get(i.id for i in self.Data if (i.filename == filename))
		#print(id)
		#d = self.Data[id]
		#state = read_exdir_new(d.filename)
		return id#tate
	
		
	# def put_to_database(self, sample_name, start, stop, incomplete, invalid, filename,
                    # names, values, parameter_names, parameter_units, measurement_type, owner = '', type_revision = '', min_values = '', max_values = '', 
                    # num_points = '', type_ref = '', ref_id = 0, comment = ''):
		# d = self.Data(comment = comment, sample_name = sample_name, time_start = start, time_stop = stop, 
			# filename = filename, type_revision = type_revision, incomplete = incomplete, invalid = invalid, owner = owner, measurement_type = measurement_type)
    
		# for minv, maxv, number, name, unit in zip(min_values, max_values, num_points, parameter_names, parameter_units):
			# self.Linear_sweep(data_id = d, min_value = minv, max_value = maxv, num_points = number, parameter_name = name, parameter_units = unit)
    
		# for name, value in zip(names, values):
			# self.Metadata(data_id = d, name = name, value = value)
        
		# if type_ref != '':
			# id_ref = self.Data[ref_id]
			# self.Reference(this = d, that = id_ref, ref_type = type_ref)
    
		# commit()    
		# return d.id