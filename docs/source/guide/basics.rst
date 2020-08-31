Basic concepts
==============

QSweepy is a python-based piece of software designed for measurement automation and
management. It has several key components:

Basic data structures for measurements and datasets


.. automodule:: qsweepy.ponyfiles.data_structures
   :members:

Saving\loading meassurements and other data using the exdir format TODO: add link


.. automodule:: qsweepy.ponyfiles.save_exdir
   :members:

Managing and finding measurements in a database using PonyORM and a Postgresql
database
.. automodule:: qsweepy.ponyfiles.database

Instrument drivers
.. automodule:: qsweepy.instrument_drivers

Sweep functions & extras
.. automodule:: qsweepy.sweep

.. automodule:: qsweepy.sweep_extras

Qubit-chip related classes
.. autoclass:: qsweepy.qubit_device.QubitDevice

Qubit measurement instruments setup
.. automodule:: qsweepy.tunable_coupling_transmons
