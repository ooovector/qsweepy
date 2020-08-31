Basic concepts
==============

QSweepy is a python-based piece of software designed for measurement automation and
management. It has several key components:

Basic data structures for measurements and datasets

.. autoclass:: circuit.Variable
    :members:

.. automodule:: ponyfiles.data_structures
   :members:

Saving\loading meassurements and other data using the exdir format TODO: add link


.. automodule:: ponyfiles.save_exdir
   :members:

Managing and finding measurements in a database using PonyORM and a Postgresql
database
.. automodule:: ponyfiles.database

Instrument drivers
.. automodule:: instrument_drivers

Sweep functions & extras
.. automodule:: sweep.py
   :members:

.. automodule:: sweep_extras

Qubit-chip related classes
.. autoclass:: qubit_device.QubitDevice

Qubit measurement instruments setup
.. automodule:: tunable_coupling_transmons
