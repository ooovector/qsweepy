Instrument driver guidelines
================================================

Closest to the actual devices in qsweepy are the "device drivers" that are located in the
instrument_drivers directory. When a new device is added to the system, it needs a device
driver. Devices can be roughly categorized into "measurement devices" and everything else.
"Measurement devices" are devices that yield measurement results, such as vector network
analyzers, oscilloscopes, voltmeters and so on. Measurement devices should implement
AbstractMeasurer abstract class:

.. autoclass:: qsweepy.instrument_drivers.abstract_measurer.AbstractMeasurer
   :members:

The AbstractMeasurer interface is what is expected from the sweep function. The common
inteface is specifically designed so that sweeping is performed in similar manner by all
devices.

All devices, both measurement and non-measurement devices, should have setters and getters
for experimentally relevant parameters, such as for a current source, the current value.
The setters and getters should use the common interface

.. code-block:: python

    device.get_parameter()
    device.set_parameter(value)

or, if it is a multi-channel device, such as a mutli-channel awg,

.. code-block:: python

    device.get_parameter(channel)
    device.set_parameter(value, channel)

where "parameter" could be "current", "bandwidth", "offset" or anything else. This common
interface is not utilized by any function, but nevertheless to be followed where possible.
Most important it is to follow the guideline for scalar values, because scalar value are
what is usually swept on.

