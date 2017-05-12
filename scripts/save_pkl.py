import matplotlib.cm as cmap
import datetime
import os
from matplotlib import pyplot as plt
import pickle
from config import get_config
import numpy as np
import plotting

def mk_dir(path = None, time=True):
	if (path is None) or (path==''):
		(data_root, day_folder_name, time_folder_name) = get_location()
		
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
			return '{0}/{1}/{2}/'.format(data_root, day_folder_name)
	else:
		if not os.path.exists(path):
			os.mkdir(path)
		return path	

def get_location():
	config = get_config()
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
	
def save_pkl(header, data, plot = True, curve_fit=None, annotation=None, location=None, time=True, filename=None):
	location = mk_dir(path = location, time=time)

	if not filename:
		if 'name' in header:
			filename = '{0} {1}'.format(header['type'], header['name'])
		else:
			filename = '{0}'.format(header['type'])
		
	f = open('{0}/{1}.pkl'.format(location, filename), 'wb')
	if header:
		data_pkl = (2, data, header)
	else:
		data_pkl = data
	pickle.dump(data_pkl, f)
	f.close()
	if plot:
		plotting.plot_measurement(data, filename, save=location, annotation=annotation)
	
	'''
	if (np.iscomplexobj(data[-1][0])):
		data.append(np.angle(data[-1]))
		data[-2] = 20*np.log10(np.abs(data[-2]))
	if ( (len(data) == 4) and plot):
		if (len(data[2].shape) == 1):
			plt.figure(figsize=(8,6))
			plt.plot(data[0], data[2])
			if curve_fit != None:
				plt.plot(curve_fit[0], curve_fit[1])
			if annotation != None:
				plt.annotate(annotation, (0.3, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			plt.title('{0} {1} abs dB'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.abs dB.png'.format(location, filename))
			plt.figure(figsize=(8,6))
			plt.plot(data[0], 10**(data[2]/10))
			if curve_fit != None:
				plt.plot(curve_fit[0], 10**(curve_fit[1]/10))
			if annotation != None:
				plt.annotate(annotation, (0.3, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			plt.title('{0} {1} abs'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.abs.png'.format(location, filename))
			plt.figure(figsize=(8,6))
			plt.plot(data[0], data[3])
			if curve_fit != None:
				plt.plot(curve_fit[0], curve_fit[2])
			if annotation != None:
				plt.annotate(annotation, (0.3, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			plt.title('{0} {1} phase'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.phase.png'.format(location, filename))

			plt.figure(figsize=(8,6))
			plt.plot(data[0], np.unwrap(data[3]))
			if curve_fit != None:
				plt.plot(curve_fit[0], np.unwrap(curve_fit[2]))
			if annotation != None:
				plt.annotate(annotation, (0.3, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			plt.title('{0} {1} unwrapped phase'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.phase_unwrapped.png'.format(location, filename))

			plt.figure(figsize=(8,6))
			plt.plot((data[0][1:]+data[0][:1])/2, np.diff(np.unwrap(data[3])))
			if curve_fit != None:
				plt.plot(curve_fit[0], np.unwrap(curve_fit[2]))
			if annotation != None:
				plt.annotate(annotation, (0.3, 0.1), xycoords='axes fraction', bbox={'alpha':1.0, 'pad':3, 'edgecolor':'black', 'facecolor':'white'})
			plt.title('{0} {1} phase diff'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.phase_diff.png'.format(location, filename))

		if (len(data[2].shape) == 2):
			plt.figure(figsize=(8,6))
			plt.pcolor(data[1], data[0], data[2], cmap='RdBu_r')
			plt.colorbar()
			plt.title('{0} {1} abs'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.abs.png'.format(location, filename))
			plt.figure(figsize=(8,6))
			plt.pcolor(data[1], data[0], data[3], cmap='RdBu_r')
			plt.colorbar()
			plt.title('{0} {1} phase'.format(header['type'], header['name']))
			plt.grid()
			plt.savefig('{0}/{1}.phase.png'.format(location, filename))
			'''