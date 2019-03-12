import numpy as np
import time
import sys
from . import save_pkl
from . import plotting
from . import sweep
import shutil as sh
import pathlib
from .data_structures import *

'''
Interactive stuff:
- (matplotlib) UI &  & telegram bot,
- telegram bot
- plotly UI
- time_left UI
'''
'''
Hooks for sweep.
(1) state.id = db.create_in_database(state)
(2) db.update_in_database(state)
(3) save_exdir.save_exdir(state)
'''


class sweeper:
	def mkdir(self):
		pass
	def __init__(self, db):
		import save_exdir
		self.db = db
		self.default_save_path = ''
		self.on_start = [self.mkdir, db.create_in_database, save_exdir.save_exdir, db.update_in_database]
		self.on_update = [lambda x: save_exdir.update_exdir(x, db)]
		self.on_finish = [db.update_in_database, save_exdir.close_exdir]
	def sweep(self, *args, **kwargs):
		return sweep.sweep(*args, **kwargs)