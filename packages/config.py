import os
import sys
from yaml import load, dump
import logging

DEFAULT_CONFIG_FILENAME = 'qtlab.cfg'
def get_data_dir():
	if 'QTLAB_PATH' in os.environ:
		data_dir = os.environ['QTLAB_PATH']
	else:
		data_dir = os.path.expanduser('~/QtLab')
	return data_dir
	
def load_current_config(filename):
	Config._config_singleton = Config(filename)
	_setup_logging()
	return Config._config_singleton
	
def get_config():
	'''Get configuration object.'''
	if Config._config_singleton is None:
		Config._config_singleton = load_current_config(filename=None)
	return Config._config_singleton
	
def _setup_logging():
	logging.basicConfig(level=logging.INFO,
		format='%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)',
		datefmt='%Y-%m-%d %H:%M',
		filename=get_config()['logfile'],
		filemode='a+')
	console = logging.StreamHandler()
	console.setLevel(logging.WARNING)
	formatter = logging.Formatter('%(name)s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)

def set_debug(enable):
	logger = logging.getLogger()
	if enable:
		logger.setLevel(logging.DEBUG)
		logging.info('Set logging level to DEBUG')
	else:
		logger.setLevel(logging.INFO)
		logging.info('Set logging level to INFO')
	
class Config:
	_config_singleton = None
	def __init__(self, filename=None):
		global _config
		self._defaults = {}
		self.load_defaults()
		self._config = dict(self._defaults)
		self.load(filename)
	def _get_default_filename(self):
		return os.path.join(get_data_dir(), DEFAULT_CONFIG_FILENAME)
	def load_defaults(self):
		self._defaults['datadir'] = os.path.join(get_data_dir(), 'data')
		self._defaults['logfile'] = os.path.join(get_data_dir(), 'qtlab.log')
	def load(self, filename=None):
		if not filename:
			filename = self._get_default_filename()
		try:
			logging.debug('Loading settings from %s', filename)
			with open(filename, 'r') as f:
				self._config.update(load(f))
		except Exception as e:
			logging.warning('Unable to load config file %s', filename)
			logging.warning(str(e))
		self._filename = filename
	def save(self):
		try:
			filename = self._get_default_filename()
			logging.debug('Saving settings to %s', filename)
			with open(filename, 'w') as f:
				dump(self._config, f, indent=4)
		except Exception as e:
			logging.warning('Unable to save config file')
			logging.warning(str(e))
	def __getitem__(self, key):
		return self._config[key]
	def __setitem__(self, key, val):
		self._config[key] = val
	def __delitem__(self, key):
		del self._config[key]
	def __contains__(self, key):
		return key in self._config
	def __repr__(self):
		return repr(self._config)

