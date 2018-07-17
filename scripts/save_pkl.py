import matplotlib.cm as cmap
import datetime
import os
from matplotlib import pyplot as plt
import pickle
from config import get_config
import numpy as np
import plotting
import scipy.io

def mk_dir(path = None, time=True, root = None, name=None, unfinished=False):
	if (path is None) or (path==''):
		(data_root, day_folder_name, time_folder_name) = get_location(unfinished=unfinished)
		if root is not None:
			data_root = root
		if not os.path.exists('{0}/{1}'.format(data_root, day_folder_name)):
			os.mkdir('{0}/{1}'.format(data_root, day_folder_name))
		if not os.path.isdir('{0}/{1}'.format(data_root, day_folder_name)):
			raise Exception('{0}/{1} is not a directory'.format(data_root, day_folder_name))
		if time:
			if not os.path.exists('{0}/{1}/{2}'.format(data_root, day_folder_name, time_folder_name)):
				os.mkdir('{0}/{1}/{2}'.format(data_root, day_folder_name, time_folder_name))
			if not os.path.isdir('{0}/{1}/{2}'.format(data_root, day_folder_name, time_folder_name)):
				raise Exception('{0}/{1}/{2} is not a directory'.format(data_root, day_folder_name, time_folder_name))
		
			return '{0}/{1}/{2}/'.format(data_root, day_folder_name, time_folder_name)
		else:
			return '{0}/{1}/'.format(data_root, day_folder_name)
	else:
		if not os.path.exists(path):
			os.mkdir(path)
		return path	

def get_location(Ndm=True, unfinished=False):
	config = get_config()
	if unfinished:
		data_root = config.get('datadir')+'/unfinished'
	else: 
		data_root = config.get('datadir')
	now = datetime.datetime.now()
	day_folder_name = now.strftime('%Y-%m-%d')
	time_folder_name = now.strftime('%H-%M-%S')
	return (data_root, day_folder_name, time_folder_name)

def load_pkl(filename, location=None):
	if not location:
		(data_root, day_folder_name, time_folder_name) = get_location()
		location = '{0}/{1}'.format(data_root, day_folder_name)
	print ('{0}/{1}.pkl'.format(location, filename))
	f = open('{0}/{1}.pkl'.format(location, filename), "rb")
	return pickle.load(f)
	
def save_pkl(header, data, plot = True, curve_fit=None, annotation=None, location=None, time=True, filename=None, matlab=False, gzip=False, plot_axes=None):
	import gzip
	location = mk_dir(path = location, time=time)

	if not filename:
		if 'type' in header:
			type = header['type']
		elif 'name' in header:
			#type = ' '.join(data.keys())
			filename = '{0}'.format(header['name'])
		else:
			filename = ' '.join(data.keys())

	f = open('{0}/{1}.pkl'.format(location, filename), 'wb')
	if header:
		data_pkl = (2, data, header)
	else:
		data_pkl = data
	pickle.dump(data_pkl, f)
	f.close()
	if plot_axes and plot:
		plotting.update_plot_measurement(data, plot_axes)
		plotting.plot_add_annotation(plot_axes, annotation)
		plotting.plot_save(plot_axes, location)
	elif plot:
		plotting.plot_measurement(data, filename, save=location, annotation=annotation, subplots=True)
	
	if matlab:
		matfilename = '{0}/{1}.mat'.format(location, filename)
		scipy.io.savemat(matfilename, mdict=data)
