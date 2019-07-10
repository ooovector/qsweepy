from . import save_exdir
from . import data_structures
from pony.orm import desc, count
import datetime

class exdir_db:
	def __init__(self, db, sample_name = None):
		self.db = db
		if not sample_name:
			sample_name = 'anonymous-sample'
		self.sample_name = sample_name

	def save_measurement(self, data):
		self.db.create_in_database(data)
		save_exdir.save_exdir(data)
		self.db.update_in_database(data)

	def save(self, **values):
		if 'sample_name' not in values.keys():
			values['sample_name'] = self.sample_name
		data = data_structures.measurement_state(**values)
		self.save_measurement(data)
		return data

	def invalidate(self, data_id, reason='anonymous', chain=True):
		invalidation_time=str(datetime.datetime.now())
		invalidation_chain = {(None, self.db.Data[data_id])}
		invalidation_chain_processed = set()
		try:
			while (True):
				reason_data, current_data = invalidation_chain.pop()
				self.db.Metadata(data_id = current_data.id, name = 'invalidation', value = 'True')
				self.db.Metadata(data_id = current_data.id, name = 'invalidation_time', value = invalidation_time)
				self.db.Metadata(data_id = current_data.id, name = 'invalidation_reason', value = reason)
				if reason_data:
					self.db.Metadata(data_id = current_data.id, name = 'invalidation_reason_chain', value = str(reason_data.id))
				if chain:
					invalidation_chain.update(list((reference.that, reference.this)
												   for reference in self.db.Reference.select(lambda r: r.that.id==current_data.id)
												   if reference.ref_type in self.db.Invalidations.ref_type))
				invalidation_chain_processed.add(current_data)
				invalidation_chain = {i for i in invalidation_chain if i[1] not in invalidation_chain_processed}
		except KeyError: # everything has been invalidated
			pass

	def select_measurement(self, measurement_type, metadata={}, references_this={}, references_that={}, ignore_invalidation=False):
		measurement_db_list = self.select_measurements_db(measurement_type, metadata = metadata, references_this=references_this, references_that=references_that, ignore_invalidation=ignore_invalidation)
		return save_exdir.load_exdir(list(measurement_db_list.order_by(lambda d: desc(d.id)).limit(1))[0].filename, db=self.db)

	def select_measurement_by_id(self, id):
		print (self.db.Data[id].filename)
		return save_exdir.load_exdir(self.db.Data[id].filename, db=self.db)

	def select_measurements_db(self, measurement_type, metadata={}, references_this={}, references_that={}, ignore_invalidation=False):
		q = self.db.Data.select(lambda d: (d.measurement_type==measurement_type))
		if not ignore_invalidation:
			#q2 = q.where(lambda d: 'invalidation' not in d.metadata.name)
			q2 = q.where(lambda d: (not d.invalid) and (not d.incomplete))

		else:
			q2 = q
		for k,v in metadata.items():
			q2 = q2.where(lambda d: count(True for m in d.metadata if m.name == k and m.value == str(v))>0)
		for k,v in references_this.items():
			q2 = q2.where(lambda d: count(True for r in d.reference_two if r.ref_type == k and r.this.id == v)>0)
		for k,v in references_that.items():
			if type(k) is str:
				q2 = q2.where(lambda d: count(True for r in d.reference_one if r.ref_type == k and r.that.id == v)>0)
			elif type(k) is tuple:
				q2 = q2.where(lambda d: count(True for r in d.reference_one if (r.ref_type == k[0] and r.ref_comment == k[1]) and r.that.id == v)>0)

			print('reference:',  k,':', v)

		return q2

	#def select_measurements(self, )
#		q = exdir_db_inst.db.Data.select(lambda d: (d.measurement_type=='iq_dc_calib' and 'invalidation' not in d.metadata.name))
#q.where(lambda d: d.metadata)
#q1 = q.where(lambda d: d.measurement_type=='iq_dc_calib')
#q2 = q.where(lambda d: True in (True for m in d.metadata if m.name == 'ex_channel' and m.value == '1'))
#q2.order_by(lambda d: d.id)
