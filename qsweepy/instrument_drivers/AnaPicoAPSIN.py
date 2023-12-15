import pyvisa as visa  # -> We use the VISA standard for communicating with
from qsweepy.instrument_drivers.instrument import Instrument
# Actual Driver

import logging
# SCPI Commands can be found at https://www.anapico.com/downloads/manuals/


class AnaPicoAPSIN(Instrument):
    def __init__(self, name, address, channel=0):  # further variables needed?
        """
        :param visaName: Visa resource name ot the device
        :param singleChannel: If each channel of the device is initialized as own class object,
            set singleChannel True (and choose the channel index). Else (default): false
        :param channelIndex: If single Channel is set to True,
            set the channel index the class object describes.
        """
        # global constants
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])
        self.name = name
        self.address = address
        self.channel = channel
        self._sc = False
        # Connect to the device
        visa_instrument = visa.ResourceManager().open_resource(
            self.address, timeout=5000
        )
        self._visainstrument: visa.resources.MessageBasedResource = visa_instrument  # type:ignore
        self._output = False
        # default init values
        # (e.g. power, bandwidth, averages, sweep_mode, frequencies, fixed_frequency)

    def set(self, name, value) -> None:
        return getattr(self, "set_" + name)(value)

    def get(self, name) -> None:
        return getattr(self, "get_" + name)()

    def get_channel(self) -> int:
        return self.channel

    def reset(self) -> None:
        self.write("*RST")

    def close_all(self) -> None:
        self._visainstrument.close()

    def get_sc(self) -> bool:
        return self._sc

    def query(self, cmd: str) -> str:
        return self._visainstrument.query(cmd)

    def write(self, cmd: str) -> None:
        self._visainstrument.write(cmd)

    def wait_until_done(self) -> str:
        """
        "[...]stops any new commands from being processed until the current processing is complete"
        :return: 1 as soon as all pending operations are done
        """
        return self.query("*OPC?")  # OPC = Operation Complete

    def abort(self) -> None:
        """
        Aborts the current List or Step sweep
        """
        self.write(":ABORt")

    def set_rf_on(self, channel=None) -> None:
        """
        :param channel: If singleChannel is set to False: Number of the channel,
            on which the power should be turned on.
        Turn the output on
        """
        if channel is None:
            channel = self.channel
        self.write(f":OUTPut{channel}:STATe ON")
        self._output = True

    def set_rf_off(self, channel=None) -> None:
        """
        :param channel: If singleChannel is set to False: Number of the channel,
            on which the power should be turned off
        Turn the output off
        """
        if channel is None:
            channel = self.channel
        self.write(f":OUTPut{channel}:STATe OFF")
        self._output = False

    def set_status(self, status, channel=None) -> None:
        """
        Sets the status of chosen channel to on or off
        :param status: Boolean, True if device should be turned on or False
            if it should be turned off
        :param channel: If singleChannel is set to False: Number of the channel,
            whose status should be changed.
        """
        if channel is None:
            channel = self.channel
        if status is True or str(status) == str(1) or status == "on":
            self.set_rf_on(channel)
        elif status is False or str(status) == str(0) or status == "off":
            self.set_rf_off(channel)
        else:
            raise ValueError("Parameter status needs to be Boolean")

    def get_status(self, channel=None) -> str:
        """
        Returns output status (on or off) of the chosen channel
        :param channel: If singleChannel is set to False: Number of the channel,
            whose output status should be queried
        :return: 1 if (chosen channel of) device is turned on and 0 if (chosen channel)
            of device is turned off
        """
        if channel is None:
            channel = self.channel
        return self.query(f":OUTPut{channel}?")

    def set_blanking(self, blanking_status, channel=None) -> None:
        """
        Set the output blanking of the instrument (blanking means the ouput
            will be turned off when the frequency changes)
        :param channel: If singleChannel is set to False: Number of the channel,
            whose blanking status should be changed
        :param blanking_status: True if blanking should be turned on,
            False if blanking should be turned off
        """
        if channel is None:
            channel = self.channel
        if blanking_status is True:
            self.write(f":OUTPut{channel}:BLANking ON")
        elif blanking_status is False:
            self.write(f":OUTPut{channel}:BLANking OFF")
        else:
            raise ValueError("set_status(): can only set True or False")

    def get_blanking(self, channel=None) -> str:
        """
        Set the output blanking of the instrument (blanking means the ouput
        will be turned off when the frequency changes)
        :param channel: If singleChannel is set to False: Number of the channel,
        whose blanking status should be queried
        """
        if channel is None:
            channel = self.channel
        return self.query(f":OUTPut{channel}:BLANking?")

    def set_frequency(self, frequency, channel=None) -> None:
        """
        Set signal generator output frequency for CW frequency mode
        :param frequency: Frequency (in HZ, I think)
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be changed
        """
        if channel is None:
            channel = self.channel
        self.write(f":SOURce{channel}:FREQuency {frequency}")

    def get_frequency(self, channel=None) -> float:
        """
        Returns signal generator output frequency
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be queried
        :return: Frequency of the output signal (of the chosen channel)
        """
        if channel is None:
            channel = self.channel
        return float(self.query(f":SOURce{channel}:FREQuency?"))

    def set_center(self, frequency, channel=None) -> None:
        """
        Set signal generator output frequency for CW frequency mode
        :param frequency: Frequency (in HZ, I think)
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be changed
        """
        if channel is None:
            channel = self.channel
        self.write(f":SOURce{channel}:FREQuency:CENTer {frequency}")

    def ask_mode(self, channel=None) -> None:
        """
        Set signal generator output frequency for CW frequency mode
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be changed
        """
        if channel is None:
            channel = self.channel
        self.write(f":SOURce{channel}:FREQuency[:CW]?")

    def set_phase(self, phase, channel=None) -> None:
        """
        Set signal generator output phase for CW frequency mode
        :param phase: phase
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be changed
        """
        if channel is None:
            channel = self.channel
        self.write(f":SOURce{channel}:PHASe:REFerence")
        self.write(f":SOURce{channel}:PHASe {phase}")

    def get_phase(self, channel=None) -> float:
        """
        Returns signal generator output phase
        :param channel: If singleChannel is set to False: Number of the channel,
            whose frequency should be queried
        :return: phase
        """
        if channel is None:
            channel = self.channel
        return float(self.query(f":SOURce{channel}:PHASe?"))

    def set_power(self, power, channel=None) -> None:
        """
        Sets the power of the sigal (at the chosen channel)
        :param channel: If singleChannel is set to False: Number of the channel,
            whose power should be set
        :param power: Power of the signal (float)
        """
        if channel is None:
            channel = self.channel
        self.write(f":SOURce{channel}:POWer {power}")

    def get_power(self, channel=None) -> float:
        """
        Returns the power of the sigal (at the chosen channel)
        :param channel: If singleChannel is set to False: Number of the channel,
            whose power should be queried
        :return: Signal power (at chosen channel) (float)
        """
        if channel is None:
            channel = self.channel
        return float(self.query(f":SOURce{channel}:POWer?"))

    def set_ref_osc_external_freq(self, external_frequency, channel=None) -> None:
        # TODO: only 10MHz is implemented
        if channel is None:
            channel = self.channel
        if external_frequency == "10MHz":
            external_frequency = 10e6
        else:
            external_frequency = 10e6
        self.write(
            f":SOURce{channel}:ROSCillator:EXTernal:FREQuency {external_frequency}"
        )

    def get_ref_osc_source(self, channel=None) -> str:
        if channel is None:
            channel = self.channel
        return self.query(f":SOURce{channel}:ROSCillator:SOURce?")

    def set_ref_osc_source(self, source="INTernal", channel=None) -> None:
        if channel is None:
            channel = self.channel
        return self.write(f":SOURce{channel}:ROSCillator:SOURce {source}")
