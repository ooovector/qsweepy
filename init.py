import os
import sys
import logging
import runpy
basedir = 'C:/Users/User/Documents/qtlab_replacement'
_execdir = basedir
sys.path.append(os.path.abspath(os.path.join(basedir, 'src')))
sys.path.append(os.path.abspath(os.path.join(basedir, 'scripts')))
sys.path.append(os.path.abspath(os.path.join(basedir, 'instruments')))
sys.path.append(basedir)
#import setup_logging