import os
import sys
import logging
import runpy

#basedir = os.path.dirname(sys.argv[0]) #"D:\\qtlab\\qtlab-15a460b_with config and 3rdparty" 
basedir = "D:/qtlab_replacement"
_execdir = basedir
sys.path.append(os.path.abspath(os.path.join(basedir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, 'scripts')))
sys.path.append(os.path.abspath(os.path.join(basedir, 'instruments')))
sys.path.append(basedir)

import setup_logging
#from init_instruments import *
#def execfile(filename):
#	exec(compile(open(filename, "rb").read(), filename, 'exec'), globals(), locals())
#execfile('src/setup_logging.py')
#execfile('init_instruments.py')