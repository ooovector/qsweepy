Basic concepts
==============

QSweepy is a python-based piece of software designed for measurement automation and
management. It has several key components:

Basic data structures for measurements and datasets

.. automodule:: ponyfiles.data_structures
   :members:

Saving\loading meassurements and other data using the exdir format TODO: add link


.. automodule:: ponyfiles.save_exdir
   :members:

Managing and finding measurements in a database using PonyORM and a Postgresql
database

.. automodule:: ponyfiles.database
   :members:

Instrument drivers

Sweep functions & extras

.. automodule:: sweep
   :members:

.. automodule:: sweep_extras
   :members:

Qubit-chip related classes

.. autoclass:: qubit_device.QubitDevice
   :members:

Qubit measurement instruments setup
