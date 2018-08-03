#глаза бы ничьи это не видели
#уже видели
import json
import json_tricks
import yaml
from . import save_pkl
import pathlib

def get_datetime_fmt():
	return '%Y%m%d%H%M%S' # ex. 20110104172008 -> Jan. 04, 2011 5:20:08pm 

# def calib_to_json(iq_ex,iq_ro):
	# def rf_tuple(items):
		# rf_calib = {}
		# k=0
		# for i,j in items:
			# d=dict((x,y) for x,y in i)
			# for num,v in j.items():
				# j[num] = str(v)
			# d.update(j)
			# rf_calib[k] = d
			# k+=1
		# return rf_calib
	# def dc_tuple(items):
		# dc_calib = {}
		# k=0
		# for i,j in items:
			# d={i[0]:i[1]}
			# for num,v in j.items():
				# j[num] = str(v)
			# d.update(j)
			# dc_calib[k] = d
	# rf_calib = rf_tuple(iq_ex.calibrations.items())
	# ro_calib = rf_tuple(iq_ro.calibrations.items())
	# dc_calib = dc_tuple(iq_ex.dc_calibrations.items())
	# calib_parameters = {'ex_dc':dc_calib,'ex_rf':rf_calib,'ro':ro_calib}
	# return calib_parameters

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
	import os.path
	loc = save_pkl.get_location()
	data_dir = loc[0]+'/calibrations/'+type+"/"
	#current_dir = data_dir+loc[1]
	pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
	# change calibrations format tuple to dict
	#if len(params)>1:
	#	if str(params[0].__module__).split('.')[-1] == 'awg_iq_multi':
	#		params = { name: calib_to_json(params[0],params[1])}
	#else:
	#	params = { name: params[0]}
	#if current_dir == find_last(current_dir):
	#	file_to_dump = current_dir+'/'+name+ '0'+'.txt'
	#else:
	#	num_of_last_file= int(find_last(current_dir).split('\\')[-1].split(name)[-1].split('.')[0])
	i=0
	while True:
		file_to_dump = data_dir+name+'-'+str(i)+'.txt'
		if not os.path.isfile(file_to_dump):
			break
		i+=1
	with open(file_to_dump, 'w') as outfile:	 
		json_tricks.dump(params, outfile,indent=4)#, separators=(',', ': '))

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