from pony.orm import*
from datetime import datetime
from time import gmtime, strftime
from .data_structures import *

def put_to_database(sample_name, start, stop, incomplete, invalid, filename,
                    names, values, parameter_names, parameter_units, owner = '', type_revision = '', min_values = '', max_values = '', 
                    num_points = '', type_ref = '', ref_id = 0, comment = ''):
					
					
	db = Database()
	class Data(db.Entity):
		id = PrimaryKey(int, auto=True)
		comment = Optional(str)
		sample_name = Required(str)
		time_start = Required(datetime, precision=6)
		time_stop = Optional(datetime, precision=6)
		filename = Required(str)
		type_revision = Optional(str)
		incomplete = Required(bool)
		invalid = Optional(bool)
		owner = Optional(str)
		metadata = Set('Metadata')
		reference_one = Set('Reference', reverse = 'this')
		reference_two = Set('Reference', reverse = 'that')
		linear_sweep = Set('Linear_sweep')

	class Metadata(db.Entity):
		id = PrimaryKey(int, auto=True)
		data_id = Required(Data)
		name = Required(str)
		value = Required(str)
		#data = Required(Data)
		
	class Reference(db.Entity):
		#id = PrimaryKey(int, auto=True)
		this = Required(Data)
		that = Required(Data)
		ref_type = Required(str)
		PrimaryKey(this, that)
		
	class Linear_sweep(db.Entity):
		data_id = Required(Data)
		min_value = Optional(float)
		max_value = Optional(float)
		num_points = Optional(int)
		parameter_name = Required(str)
		parameter_units = Optional(str)
		
	class Invalidations(db.Entity):
		this_type = Required(str)
		that_type = Required(str)
		ref_type = Required(str)
		id = PrimaryKey(int, auto=True)
		
	db.bind('postgres', user='qsweepy', password='qsweepy', host='localhost', database='qsweepy', port = 5432)
	db.generate_mapping(create_tables=True)
	
	d = Data(comment = comment, sample_name = sample_name, time_start = start, time_stop = stop, 
			filename = filename, type_revision = type_revision, incomplete = incomplete, invalid = invalid, owner = owner)
    
	for minv, maxv, number, name, unit in zip(min_values, max_values, num_points, parameter_names, parameter_units):
		Linear_sweep(data_id = d, min_value = minv, max_value = maxv, num_points = number, parameter_name = name, parameter_units = unit)
    
	for name, value in zip(names, values):
		Metadata(data_id = d, name = name, value = value)
        
	if type_ref != '':
		id_ref = Data[ref_id]
		Reference(this = d, that = id_ref, ref_type = type_ref)
    
	commit()    
	return d.id