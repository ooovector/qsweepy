Single tone spectroscopy
==========================
Here you can find general information about how to do a single tone spectroscopy


The general examples of usage of written functions are:

0) First of all you should import the library:

.. code-block:: python

  from qsweepy.qubit_calibrations import spectroscopy

1) If you want to make a sweep over coil voltages you can use it:

.. code-block:: python

    spectroscopy.single_tone_spectroscopy(device,
                                   qubit_id,
                                   device.get_qubit_fr(qubit_id),
                                   (np.linspace(-2, 2, 401), hardware.coil.set_offset, 'Coil Voltage', 'V'))

2) or if you want to make a sweep over single-tone drive power you can use it:

.. code-block:: python

    spectroscopy.single_tone_spectroscopy(device,
                                   qubit_id,
                                   device.get_qubit_fr(qubit_id),
                                   (np.linspace(-45, 16, 124), hardware.pna.set_power, 'Power', 'dBm'))


The description of the implemented classes and functions:

.. automodule::  qsweepy.qubit_calibrations.spectroscopy
    :members:




.. autoclass:: qsweepy.fitters.spectroscopy_overview.SingleToneSpectroscopyOverviewFitter
   :members:
   :inherited-members:

.. automethod:: qsweepy.fitters.spectroscopy_overview.find_resonators
