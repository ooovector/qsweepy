from qsweepy.instrument import Instrument
import types
import logging
import numpy as np

import requests
import re
import time

class nn_rf_switch(Instrument):
	def __init__(self, name, address):
		logging.info(__name__ + ' : Initializing instrument nn_rf_switch')
		Instrument.__init__(self, name, tags=['physical'])
		
		self._address = address
		self.num_channels = 2
	
		self.add_parameter('switch', type=int,
			flags=Instrument.FLAG_GETSET,
			minval=1, maxval=6,
			channels=(1, 2), channel_prefix='ch%d_') 
	
		for channel in range(1,self.num_channels+1):
			self.do_get_switch(channel=channel)
		
	def do_set_switch(self, switch, channel):
		if switch != self.do_get_switch(channel=channel):
			act = 1
		else :
			act = 0
		url = 'http://{0}/?pos={1}&ch={2}&act={3}'.format(self._address, switch, channel, act)
		#print (url)
		r = requests.get(url)
		time.sleep(1)
		
	def do_get_switch(self, channel):
		r = requests.get('http://{0}/'.format(self._address))
		time.sleep(1)
		answer = r.text
		matches = re.finditer('\\?pos=([0-9]+)&ch=([0-9]+)&act=([0-9]+)', answer)
		for match in matches:
			#print (match.groups())
			#print (match.group(1), match.group(2), match.group(3))
			if int(match.group(2)) == channel and int(match.group(3)) == 0:
				return int(match.group(1))
		else:
			return 0