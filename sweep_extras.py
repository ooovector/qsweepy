import numpy as np
import time
import sys
from . import save_pkl
from . import plotting
from . import sweep
import shutil as sh
import pathlib
from .data_structures import *
from . import plotly_plot
from . import sweep_fit

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
		from . import save_exdir
		self.db = db
		self.default_save_path = ''
		self.on_start = [(db.create_in_database,tuple()), 
						 (save_exdir.save_exdir,(True,)), 
						 (db.update_in_database,tuple()),
						 #(sweep_fit.fit_on_start, (db,))
						 ]
		self.on_update = [(save_exdir.update_exdir,tuple()),
						  #(sweep_fit.sweep_fit, (db, ))
						  ]
		self.on_finish = [#(sweep_fit.fit_on_finish, (db, )),
						  (db.update_in_database,tuple()), 
						  (save_exdir.close_exdir,tuple()),
						  (plotly_plot.save_default_plot,(self.db,))]
	
	def sweep(self, *args, **kwargs):
		return sweep.sweep(*args, on_start = self.on_start, on_finish = self.on_finish, on_update = self.on_update, **kwargs)