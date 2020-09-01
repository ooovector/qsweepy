from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module
import re

def readme():
    with open('README.md') as f:
        return f.read()


def license():
    with open('LICENSE') as f:
        return f.read()

setup(name='qsweepy',
      version=1,#fix it
      use_2to3=False,
      author='Ilia Besedin',
      author_email='ilia.besedin@gmail.com',
      maintainer='Ilia Besedin',
      maintainer_email='ilia.besedin@gmail.com',
      description='Python based Circuit QED data acquisition framework '
                  'developed by members of the supercoducting metamaterials laboratory at '
                   'NUST MISIS',
      long_description=readme(),
      url='https://github.com/ooovector/qsweepy',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering'
      ],
      license=license(),
      # if we want to install without tests:
      packages=find_packages(exclude=["*.tests", "tests"]),
      #packages=find_packages(),
      install_requires=[
          'numpy>=1.10',
          'pyvisa>=1.8',
          'IPython>=4.0',
          'ipywidgets>=4.1',
          'lmfit>=0.9.5',
          'scipy>=0.17',
          'h5py>=2.6',
      ],
      test_suite=None,
      zip_safe=False)
