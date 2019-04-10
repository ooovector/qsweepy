from . import save_exdir
from . import data_structures

class exdir_db:
	def __init__(self, db, sample_name = None):
		self.db = db
		if not sample_name:
			sample_name = 'anonymous-sample'
		self.sample_name = sample_name
		
	def save(self, values):
		data = data_structures.measurement_state()
		data.metadata = values
		self.db.create_in_database(data)
		save_exdir.save_exdir(data)
		self.db.update_in_database(data)
	