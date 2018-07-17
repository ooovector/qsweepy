# config.py, class for program configuration
# Reinier Heeres <reinier@heeres.eu>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import os
import sys

# for backward compatibility to python 2.5
import json
import logging

#TODO: 

# An ideal config module should be loadable (by config name or path)
# have default config ready
# Default config should be available in QTLAB_PATH

# Current functions: __init__ loads from filename (which filename?) fine
# Execdir is unnecessary. Remove it completely.


class Config():
    '''
    Class to manage settings for the QTLab environment.
    '''
    def __init__(self, filename):
        self._filename = filename
        self._config = {}
        self._defaults = {}

        self.load_defaults()
        self.load()

        # Override exec dir
        self['execdir'] = get_execdir()
# We don't have a tmpdir.
    def setup_tempdir(self):
        '''Get directory for temporary files.'''

        tdir = self.get('tempdir', None)
        if tdir == None or not os.path.exists(tdir):
            tdir = os.path.join(get_execdir(), 'tmp')
            self.set('tempdir', tdir)
        if not os.path.exists(tdir):
            os.makedirs(tdir)

        return tdir

    def _get_filename(self):
        return os.path.join(get_execdir(), self._filename)
		
    def load_defaults(self):
        self._defaults['datadir'] = os.path.join(get_execdir(), 'data')

    def save_defaults(self):
        return

    def load(self):
        '''
        Load settings.
        '''
        try:
            filename = self._get_filename()
            logging.debug('Loading settings from %s', filename)
            f = open(self._get_filename(), 'r')
            self._config = json.load(f)
            f.close()
        except Exception as e:
            logging.warning('Unable to load config file.')
            self._config = {}

    def remove(self, remove_list, save=True):
        '''
        Remove settings from config file

        Input:
            remove_list [string] : list of items to remove
        '''

        for item in remove_list:
            if item in self._config:
                del self._config[item]

        if save:
            self.save()

    def save(self, delay=5):
        '''
        Save settings.

        'delay' specifies the delay (in seconds) to use to avoid saving
        too often.
        '''

        if delay == 0:
            self._do_save()

    def _do_save(self):
        try:
            filename = self._get_filename()
            logging.debug('Saving settings to %s', filename)
            f = file(filename, 'w+')
            json.dump(self._config, f, indent=4, sort_keys=True)
            f.close()
        except Exception as e:
            logging.warning('Unable to save config file')

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, val):
        self.set(key, val)

    def get(self, key, default=None):
        '''
        Get configuration variable. If it is not defined, return the default
        value. In this case, the variable will be set to this default to
        ensure consistency.

        Input:
            key (string): variable name
            default (any type): default variable value

        Output:
            None
        '''

        if key in self._config:
            return self._config[key]
        elif default is not None:
            self._config[key] = default
            return default
        elif key in self._defaults:
            val = self._defaults[key]
            self._config[key] = val
            return val
        else:
            return None

    def set(self, key, val, save=True):
        '''
        Set configuration variable.

        Input:
            key (string): variable name
            val (any type): variable value

        Output:
            None
        '''

        self._config[key] = val
        if save:
            self.save()

    def get_all(self):
        return self._config

def get_config():
    '''Get configuration object.'''
    global _config
    if _config is None:
        pname = 'qtlab'#os.path.split(sys.argv[0])[-1]
        fname = pname + '.cfg'
        _config = create_config(fname)
    return _config

_config = None

def create_config(filename):
    global _config
    _config = Config(filename)
    return _config

_execdir = os.getcwd()

def get_execdir():
    '''Get work directory we started in.'''
    global _execdir
    return _execdir

