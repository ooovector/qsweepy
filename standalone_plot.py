#!/usr/bin/python
# stuff to run always here such as class/def

import sys
from qsweepy import plotting
import pickle
import os
import pathlib
from matplotlib import pyplot as plt

def plot_filename(filename):
	with open(filename, 'rb') as f:
		measurement = pickle.load(f)[1]
	if hasattr(filename, 'parts'):
		window_name = filename.parts[-1]
	elif type(filename) is str:
		window_name = filename
	else:
		window_name = 'chinese noname'
	plotting.plot_measurement(measurement, window_name)
	plt.autoscale(True)
	plt.figure(num=window_name).tight_layout()
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
	if len(sys.argv)>1:
		plot_filename(filename = sys.argv[1])
	else:
		pickles = pathlib.Path('.').glob('*.pkl')
		for filename in pickles:
			plot_filename(filename)