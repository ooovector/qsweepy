#глаза бы ничьи это не видели
#уже видели
import json
import json_tricks
import yaml
from . import save_pkl
import pathlib
import collections
def get_datetime_fmt():
	return '%Y%m%d%H%M%S' # ex. 20110104172008 -> Jan. 04, 2011 5:20:08pm 

def find_last(data_dir, name_parts, ignore_invalidation=False):
	from . import config
	import datetime
	if type(name_parts) is str:
		name_parts = [name_parts]
		
	print (data_dir, '*'.join(name_parts))
	last_time, last_file = max([(f.stat().st_mtime, f) for f in pathlib.Path(data_dir).glob('*'.join(name_parts)+'*')])
	last_time = datetime.datetime.fromtimestamp(last_time)
	if not last_file:
		return None
	if not ignore_invalidation:
		invalidation_time = datetime.datetime.strptime(config.get_config()['earliest_calibration_date'], get_datetime_fmt())
		if last_time<invalidation_time:
			return None
	return last_file
	
def dump(type,name,params, NEW = False):
	# if the same params already exists it shouldn't be dumped
	import os.path
	loc = save_pkl.get_location()
	data_dir = loc[0]+'/calibrations/'+type+"/"
	pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
	file_to_dump = data_dir+name+'.txt'
	with open(file_to_dump, 'w') as outfile:	 
		json_tricks.dump(params, outfile,indent=4,sort_keys=True)#, separators=(',', ': '))

def invalidate_calibrations():
	from . import config
	import datetime
	config = config.get_config()
	config['earliest_calibration_date'] = datetime.datetime.now().strftime(get_datetime_fmt())
	config.save()
		
def load(type,name):
	loc = save_pkl.get_location()
	data_dir = loc[0]+'/calibrations/'+type+'/'
	#find last data dir and lsat save file
	file_to_load = find_last(data_dir, name)
#	if not file_to_load:
#		raise
		#or raise?
	with open(file_to_load, "r") as read_file:
		data = json_tricks.load(read_file)
	return data