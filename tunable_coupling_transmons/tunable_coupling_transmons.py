from .. import exdir_db
from pony.orm import db_session, select, desc

class tunable_coupling_transmons:
	def __init__(self, exdir_db, name_priority_list):
		self.name_priority_list = name_priority_list
		#self.db = db
		self.exdir_db = exdir_db
		
#	def get_single_with_priority_list(self, getter):
#		with db_session:
#			for name_priority in self.name_priority_list:
			#	try:
					#return self.exdir_db.db.select(getter).order_by(lambda: datum_entry.id)[-1]
#					return select(getter).order_by(desc(datum_entry.id)).first()
			#	except:
			#		pass
			#raise ValueError('Not found')
	
	def get_resonator_frequency(self, resonator_id):
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
		