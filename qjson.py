#глаза бы ничьи это не видели
import json
from . import save_pkl
import pathlib

def calib_to_json(iq_ex,iq_ro):
	def rf_tuple(items):
		rf_calib = {}
		k=0
		for i,j in items:
			d=dict((x,y) for x,y in i)
			for num,v in j.items():
				j[num] = str(v)
			d.update(j)
			rf_calib[k] = d
			k+=1
		return rf_calib
	def dc_tuple(items):
		dc_calib = {}
		k=0
		for i,j in items:
			d={i[0]:i[1]}
			for num,v in j.items():
				j[num] = str(v)
			d.update(j)
			dc_calib[k] = d
	rf_calib = rf_tuple(iq_ex.calibrations.items())
	ro_calib = rf_tuple(iq_ro.calibrations.items())
	dc_calib = dc_tuple(iq_ex.dc_calibrations.items())
	calib_parameters = {'ex_dc':dc_calib,'ex_rf':rf_calib,'ro':ro_calib}
	return calib_parameters

def find_last(data_dir):
	names = list()
	for date in pathlib.Path(data_dir).iterdir():
		names.append( str(date).split('\\')[-1])
	if len(names) !=0:
		data_dir += '/'+ max(names)
		return data_dir
	else: return data_dir
	
def dump(name,params, NEW = True):
	loc = save_pkl.get_location()
	data_dir = loc[0]+'/setups/'+name+"/"
	current_dir = data_dir+loc[1]
	if not pathlib.Path(current_dir).exists():
		pathlib.Path(current_dir).mkdir(parents=True, exist_ok=True)
	# change calibrations format
	if len(params)>1:
		if str(params[0].__module__).split('.')[-1] == 'awg_iq_multi':
			params = { name: calib_to_json(params[0],params[1])}
	else:
		params = { name: params[0]}
	if current_dir == find_last(current_dir):
		file_to_dump = current_dir+'/'+name+ '0'+'.txt'
	else:
		num_of_last_file= find_last(current_dir).split('\\')[-1].split(name)[-1].split('.')[0]
		file_to_dump = current_dir+'/'+name+ str(num_of_last_file+1)+'.txt'
	with open(file_to_dump, 'w') as outfile:	 
		json.dump(params, outfile,indent=4, separators=(',', ': '))

def load(name):
	loc = save_pkl.get_location()
	data_dir = loc[0]+'/setups/'+name
	#find last data dir and lsat save file
	file_to_load = find_last(find_last(data_dir))
	with open(file_to_load, "r") as read_file:
		data = json.load(read_file)
	return data