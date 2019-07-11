"""
Description:
	Loading all modules and subpackages from qsweepy.instrument_drivers package omitting module's
	members which name starts with "__"
"""

__all__ = []
import pkgutil
import inspect
import qsweepy.instrument_drivers

for loader, name, is_pkg in pkgutil.walk_packages(qsweepy.instrument_drivers.__path__):
	try:
		module = loader.find_module(name).load_module(name)

		for member_name, value in inspect.getmembers(module):
			if member_name.startswith('__'):
				continue
	
			globals()[member_name] = value
			__all__.append(member_name)
	except Exception as e:
		print('Failed loading module '+name+': ', e)