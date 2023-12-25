import qsweepy.libraries.instruments as instruments
import qsweepy
from qsweepy.libraries_dorf.awg_channel import awg_channel
from qsweepy.libraries_dorf.awg_iq_multi import AWGIQMulti

import numpy as np


device_settings = {
    'vna_address': 'TCPIP0::10.1.0.75::inst0::INSTR',
    'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
    'rf_switch_address': '10.20.61.91',
    'use_rf_switch': False,
    'pxi_chassis_id': 0,
    'hdawg1_address': 'hdawg-dev8108',
    'hdawg2_address': 'hdawg-dev8250',
    'sa_address': 'TCPIP0::10.1.0.80::inst0::INSTR',
    'adc_address': '0009052001482f25',
    'adc_timeout': 30,
    'adc_trig_rep_period': 10,  # 12.5 MHz rate period
    'adc_trig_width': 2,  # 32 ns trigger length
    'awg_tek_address':'TCPIP0::10.1.0.76::inst0::INSTR',
    'anapico_address': 'USB0::0x03EB::0xAFFF::3B5-0B4M4001C-0780::INSTR',
}

cw_settings = {'mixer_thru': 0.5}

pulsed_settings = {'lo1_power': 15,
                   'vna_power': 12,
                   'ex_clock': 2400e6,  # 1 GHz - clocks of some devices
                   'ro_clock': 1000e6,
                   'rep_rate': 20e3,  # 10 kHz - pulse sequence repetition rate
                   # 500 ex_clocks - all waves is shorten by this amount of clock cycles
                   # to verify that M3202 will not miss next trigger
                   # (awgs are always missing trigger while they are still outputting waveform)
                   'global_num_points_delta': 400,
                   'lo1_freq':  5e9,
                   'pna_freq': 5.2e9,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_channel_name': 'ro_trg',
                   'trigger_readout_length': 200e-9,
                   'modem_dc_calibration_amplitude': 1.0,
                   'adc_nop': 1024,
                   'adc_nums': 10000,  ## Do we need control over this? Probably, but not now... WUT THE FUCK MAN
                   }

class prokladka():
    """
    Class for using many channels in Anapico
    """
    def __init__(self, lo1, name=None, channel=None):
        self.lo1 = lo1
        self.name = name
        self.channel = channel
    def set_status(self, status):
        self.lo1.set_status(status, channel=self.channel)
    def set_power(self, power):
        self.lo1.set_power(power, channel = self.channel)
    def set_frequency(self, frequency):
        self.lo1.set_frequency(frequency, channel=self.channel)
    def get_power(self):
        return self.lo1.get_power(channel = self.channel)
    def get_frequency(self):
        return self.lo1.get_frequency(channel=self.channel)
    def get_status(self):
        return self.lo1.get_status(channel=self.channel)


class hardware_setup():
    def __init__(self, device_settings, pulsed_settings):
        self.device_settings = device_settings
        self.pulsed_settings = pulsed_settings
        self.cw_settings = cw_settings
        self.hardware_state = 'undefined'
        self.sa = None

        self.pna = None
        self.lo_vna = None
        self.lo_q1 = None
        self.dorf = None
        self.rf_switch = None
        self.coil_device = None
        self.adc_device = None
        self.adc = None
        self.awg_tek = None

        self.ro_trg = None
        self.iq_devices = None

    def open_devices(self):
        self.lo_vna = instruments.AnaPicoAPSIN('lo1', address=self.device_settings['anapico_address'], channel=1)
        self.lo_q1 = prokladka(self.lo_vna, name='lo_q1', channel=2)
        self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])
        pass


    def set_cw_mode(self, channels_off=None):
        pass


    def set_pulsed_mode(self):
        if self.hardware_state == 'pulsed_mode':
            return
        self.hardware_state = 'undefined'

        self.lo_q1.set_status(1)  # turn on lo1 output
        self.lo_q1.set_power(self.pulsed_settings['lo1_power'])
        self.lo_q1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.lo_vna.set_status(1)  # turn on lo1 output
        self.lo_vna.set_power(self.pulsed_settings['vna_power'])
        self.lo_vna.set_frequency(self.pulsed_settings['pna_freq'])
        if self.pna:
            self.pna.write("SOUR1:POW1:MODE OFF")
            self.pna.write("SOUR1:POW2:MODE OFF")

        # TODO: setup for dorf device


    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch.do_get_switch(channel=channel) != value:
            self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        self.iq_devices = {'iq_ro':  AWGIQMulti(awg=self.dorf, channel_i=0, channel_q=1, lo=self.lo_vna, exdir_db=exdir_db),
                           'iq_ex7': AWGIQMulti(awg=self.dorf, channel_i=2, channel_q=3, lo=self.lo_q1, exdir_db=exdir_db),
                           }

        self.iq_devices['iq_ro'].name = 'ro'
        self.iq_devices['iq_ex7'].name = 'ex1'

        self.iq_devices['iq_ro'].sa = self.sa
        self.iq_devices['iq_ex7'].sa = self.sa

        self.fast_controls = {
                              'c6z':awg_channel(self.dorf, 3),
        }


