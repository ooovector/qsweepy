import qsweepy.libraries.instruments as instruments
import qsweepy
from qsweepy.libraries.awg_channel2 import awg_channel
from qsweepy.libraries.awg_digital2 import awg_digital




from qsweepy.instrument_drivers._QubitDAQ.reg_intf import *
# from qsweepy.instrument_drivers._QubitDAQ.usb_intf import *
import qsweepy.instrument_drivers._QubitDAQ.driver as drv
import numpy as np
from time import sleep
# import rpyc
import pyvisa as visa

# device_settings = {  # 'vna_address': 'TCPIP0::10.20.61.157::inst0::INSTR',
#     #'vna_address': 'TCPIP0::10.20.61.147::inst0::INSTR',
#     'vna_address': 'TCPIP0::10.20.61.94::inst0::INSTR',
#     'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
#     # 'lo1_timeout': 5000,
#     # 'lo1_address': 'TCPIP0::10.20.61.46::inst0::INSTR',
#     'rf_switch_address': '10.20.61.91',
#     'use_rf_switch': True,
#     'pxi_chassis_id': 0,
#     'hdawg1_address': 'hdawg-dev8108',
#     'hdawg2_address': 'hdawg-dev8250',
#     'sa_address': 'TCPIP0::10.20.61.56::inst0::INSTR',
#     'adc_timeout': 10,
#     'adc_trig_rep_period': 10,  # 12.5 MHz rate period
#     'adc_trig_width': 2,  # 32 ns trigger length
#     'awg_tek_address':'TCPIP0::10.20.61.67::inst0::INSTR',
#     # 'anapico_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
#     'anapico_address': 'USB0::0x03EB::0xAFFF::3B5-0B4M4001C-0780::INSTR',
# }

device_settings = {  # 'vna_address': 'TCPIP0::10.20.61.157::inst0::INSTR',
    #'vna_address': 'TCPIP0::10.20.61.147::inst0::INSTR',
    'vna_address': 'TCPIP0::10.1.0.75::inst0::INSTR',
    'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
    # 'lo1_timeout': 5000,
    # 'lo1_address': 'TCPIP0::10.20.61.46::inst0::INSTR',
    'rf_switch_address': '10.20.61.91',
    'use_rf_switch': True,
    'pxi_chassis_id': 0,
    'hdawg1_address': 'hdawg-dev8108',
    'hdawg2_address': 'hdawg-dev8250',
    'sa_address': 'TCPIP0::10.1.0.80::inst0::INSTR',
    'adc_address': '0009052001482f25',
    'adc_timeout': 30,
    'adc_trig_rep_period': 10,  # 12.5 MHz rate period
    'adc_trig_width': 2,  # 32 ns trigger length
    'awg_tek_address':'TCPIP0::10.1.0.76::inst0::INSTR',
    # 'anapico_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
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
                   'hdawg_ch0_amplitude': 0.8,
                   'hdawg_ch1_amplitude': 0.8,
                   'hdawg_ch2_amplitude': 0.8,
                   'hdawg_ch3_amplitude': 0.8,
                   'hdawg_ch4_amplitude': 0.8,
                   'hdawg_ch5_amplitude': 0.8,
                   'hdawg_ch6_amplitude': 0.8,
                   'hdawg_ch7_amplitude': 0.8,
                   'lo1_freq':  6.29e9, #6.35e9, #6.6e9, #6.45e9, для измерения 6-7 6.9e9,

                   # 'pna_freq': 7.088455e9, #7.1e9, #7.25e9,#   для измерения 6-7 7.4e9,#   для измерения 5-6 7.5e9,
                   # 'pna_freq': 7.219e9 - 20e6, #7.1e9, #7.25e9,#   для измерения 6-7 7.4e9,#   для измерения 5-6 7.5e9,
                   'pna_freq': 7.105e9 + 20e6, #7.5515e9 + 20e6, #7.4e9, #7.2338e9 + 20e6, #7.4e9,
                   # 'calibrate_delay_nop': 65536,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_channel_name': 'ro_trg',
                   'trigger_readout_length': 200e-9,
                   'modem_dc_calibration_amplitude': 1.0,
                   'adc_nop': 1024,
                   'adc_nums': 10000,  ## Do we need control over this? Probably, but not now... WUT THE FUCK MAN
                   # 'adc_default_delay': 550,
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
#
# class SC_PXI:
#     def __init__(self, address):
#         self.conn = rpyc.classic.connect(address)
#         self.conn.execute('import qsweepy.libraries.instruments as instruments')
#         self.conn.execute('lo = instruments.SignalCore_5502a()')
#
#     def search(self):
#         self.conn.execute('lo.search()')
#
#     def open(self):
#         self.conn.execute('lo.open()')
#
#     def set_status(self, status):
#         self.conn.execute('lo.set_status({})'.format(status))
#
#     def set_power(self, power):
#         self.conn.execute('lo.set_power({})'.format(power))
#
#     def set_frequency(self, frequency):
#         self.conn.execute('lo.set_frequency({})'.format(frequency))
#
#     def close(self):
#         self.conn.execute('lo.close()')
#
#     def get_frequency(self):
#         return self.conn.namespace['lo']._frequency



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
        self.lo_q2 = None
        self.lo1=None
        self.lo2=None
        self.rf_switch = None
        self.coil_device = None
        self.hdawg1 = None
        self.hdawg2 = None
        self.adc_device = None
        self.adc = None
        self.awg_tek = None

        self.ro_trg = None
        self.q1z = None
        self.c1z = None
        self.q2z = None
        self.c2z = None
        self.q3z = None
        self.c3z = None
        self.q4z = None
        self.c4z = None
        self.q5z = None
        self.c5z = None
        self.q6z = None
        self.c6z = None
        self.q7z = None
        self.c7z = None
        self.q8z = None
        self.c8z = None
        self.c9z = None
        self.c10z = None
        self.iq_devices = None

    def open_devices(self):

        # RF switch for making sure we know what sample we are measuring
        self.pna = instruments.Agilent_N5242A('pna', address=self.device_settings['vna_address'])
        # sleep(1)
        self.lo_vna = instruments.AnaPicoAPSIN('lo1', address=self.device_settings['anapico_address'], channel=1)
        self.lo_q1 = prokladka(self.lo_vna, name='lo_q1', channel=2)
        self.lo_q2 = prokladka(self.lo_vna, name='lo_q2', channel=3)
        # sleep(1)
        self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])


        self.awg_tek = instruments.Tektronix_AWG5014('awg_tek', address = self.device_settings['awg_tek_address'])

        # visa.log_to_screen()
        #
        # if self.device_settings['use_rf_switch']:
        #     self.rf_switch = instruments.nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])
        #
        self.hdawg1 = instruments.ZIDevice(self.device_settings['hdawg1_address'], devtype='HDAWG', delay_int=0)
        # self.hdawg2 = instruments.ZIDevice(self.device_settings['hdawg2_address'], devtype='HDAWG', delay_int=0)

        for channel_id in range(8):
            self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/range' % channel_id, 1)
        #     self.hdawg2.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/range' % channel_id, 1)
        #
        # for channel_id in range(8):
        #     self.hdawg2.daq.setDouble('/' + self.hdawg2.device + '/sigouts/%d/range' % channel_id, 1)
        # #     self.hdawg2.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/range' % channel_id, 1)

        # #It is necessary if you want to use DIOs control features during pulse sequence
        self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/dios/0/mode', 1)
        self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/dios/0/drive', 1)
        self.hdawg1.daq.sync()
        #
        # self.hdawg2.daq.setInt('/' + self.hdawg2.device + '/dios/0/mode', 1)
        # self.hdawg2.daq.setInt('/' + self.hdawg2.device + '/dios/0/drive', 2)
        # self.hdawg2.daq.sync()


        self.coil_device = self.awg_tek
        self.awg_tek.do_set_output(1, 1)
        self.awg_tek.do_set_output(1, 2)
        self.awg_tek.do_set_output(1, 3)
        self.awg_tek.do_set_output(1, 4)

        self.c7z = awg_channel(self.hdawg1, 6)
        self.c8z = awg_channel(self.hdawg1, 7)

        self.q7z = awg_channel(self.awg_tek, 2)
        self.q6z = awg_channel(self.awg_tek, 1)
        self.q8z = awg_channel(self.awg_tek, 3)
        self.q5z = awg_channel(self.awg_tek, 3)



        self.c7z.set_offset(0)
        self.c8z.set_offset(0)
        self.q7z.set_offset(0)
        self.q6z.set_offset(0)
        self.q8z.set_offset(0)
        self.q5z.set_offset(0)


        self.adc_device = drv.Device(self.device_settings['adc_address'])
        self.data_capture_timeout = self.device_settings['adc_timeout']
        self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        self.adc.output_raw = True
        self.adc.last_cov = False
        self.adc.avg_cov = False
        self.adc.resultnumber = False

        self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length

        ## OLD ADC DRIVER
        # self.adc_device = instruments.TSW14J56_evm()
        # self.adc_device.timeout = self.device_settings['adc_timeout']
        # self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        # self.adc.output_raw = True
        # self.adc.last_cov = False
        # self.adc.avg_cov = False
        # self.adc.resultnumber = False
        #
        # self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        # self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length
        #
        # self.hardware_state = 'undefined'
        self.awg_tek.set_nop(100)
        self.awg_tek.set_waveform([0] * 100, channel=1)
        self.awg_tek.set_waveform([0] * 100, channel=2)
        self.awg_tek.set_waveform([0] * 100, channel=3)
        self.awg_tek.set_waveform([0] * 100, channel=4)

    def set_cw_mode(self, channels_off=None):
        if self.hardware_state == 'cw_mode':
            return
        self.hardware_state = 'cw_mode'
        self.hdawg1.stop()
        # self.hdawg2.stop()

        global_num_points = int(np.round(
            self.pulsed_settings['ex_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
                'global_num_points_delta']))

        self.hdawg1.set_output(output=1, channel=0)
        self.hdawg1.set_output(output=1, channel=1)
        self.hdawg1.set_output(output=1, channel=2)
        self.hdawg1.set_output(output=1, channel=3)
        self.hdawg1.set_output(output=1, channel=4)
        self.hdawg1.set_output(output=1, channel=5)
        self.hdawg1.set_output(output=1, channel=6)
        self.hdawg1.set_output(output=1, channel=7)
        #
        # self.hdawg2.set_output(output=1, channel=0)
        # self.hdawg2.set_output(output=1, channel=1)
        # self.hdawg2.set_output(output=1, channel=2)
        # self.hdawg2.set_output(output=1, channel=3)
        # self.hdawg2.set_output(output=1, channel=4)
        # self.hdawg2.set_output(output=1, channel=5)
        # self.hdawg2.set_output(output=1, channel=6)
        # self.hdawg2.set_output(output=1, channel=7)
        #
        # # self.awg_tek.stop()
        #
        # for channel in range(0, 8):
        #     self.hdawg2.set_amplitude(amplitude=0.05, channel=channel)
        #     self.hdawg2.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
        #     self.hdawg2.set_output(output=1, channel=channel)
        #     # self.hdawg.set_waveform(waveform=[0] * global_num_points, channel=channel)
        #
        # for channel in [0,1,2,3,4,5]:
        for channel in [0, 1, 2, 3, 4, 5]:
            self.hdawg1.set_amplitude(amplitude=0.05, channel=channel)
            self.hdawg1.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            self.hdawg1.set_output(output=1, channel=channel)
        # #
        #
        # if channels_off is not None:
        #     for channel_off in channels_off:
        #         self.hdawg.set_output(output=0, channel=channel_off)

        self.pna.write("SOUR1:POW1:MODE ON")
        self.pna.write("SOUR1:POW2:MODE OFF")
        self.pna.set_sweep_mode("LIN")
        self.hardware_state = 'cw_mode'


    def set_pulsed_mode(self):
        if self.hardware_state == 'pulsed_mode':
            return
        self.hardware_state = 'undefined'

        self.lo_q1.set_status(1)  # turn on lo1 output
        self.lo_q1.set_power(self.pulsed_settings['lo1_power'])
        self.lo_q1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.lo_q2.set_status(1)  # turn on lo1 output
        self.lo_q2.set_power(self.pulsed_settings['lo1_power'])
        self.lo_q2.set_frequency(self.pulsed_settings['lo1_freq'])


        self.lo_vna.set_status(1)  # turn on lo1 output
        self.lo_vna.set_power(self.pulsed_settings['vna_power'])
        self.lo_vna.set_frequency(self.pulsed_settings['pna_freq'])
        if self.pna:
            self.pna.write("SOUR1:POW1:MODE OFF")
            self.pna.write("SOUR1:POW2:MODE OFF")

        #
        # self.pna.do_set_status(1)
        # self.pna.set_power(self.pulsed_settings['vna_power'])

        # # self.pna.write("OUTP ON")
        # # self.pna.write("SOUR1:POW1:MODE OFF")
        # # self.pna.write("SOUR1:POW2:MODE ON")
        # self.pna.set_sweep_mode("CW")  # privet RS ZVB20
        # # self.pna.set_trigger_source("ON")
        # self.pna.set_frequency(self.pulsed_settings['pna_freq'])

        self.hdawg1.stop()
        # self.hdawg2.stop()

        # self.hdawg1.set_clock(self.pulsed_settings['ex_clock'])
        self.hdawg1.set_clock_source(0)
        # self.hdawg2.set_clock(self.pulsed_settings['ex_clock'])
        # self.hdawg2.set_clock_source(0)

        self.hdawg1.set_trigger_impedance_1e3()
        self.hdawg1.set_dig_trig1_source([0, 2, 2, 2])
        self.hdawg1.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
        # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        self.hdawg1.set_dig_trig2_source([1, 3, 3, 3])
        self.hdawg1.set_dig_trig2_slope([1, 1, 1, 1])
        self.hdawg1.set_trig_level(0.3)
        self.hdawg1.set_trig_level(1.0, channels=[0])

        self.hdawg1.set_marker_out(0, 0)
        self.hdawg1.set_marker_out(1, 1)
        self.hdawg1.set_marker_out(2, 0)
        self.hdawg1.set_marker_out(3, 1)
        self.hdawg1.set_marker_out(4, 0)
        self.hdawg1.set_marker_out(5, 1)
        self.hdawg1.set_marker_out(6, 0)
        self.hdawg1.set_marker_out(7, 1)

        #self.hdawg1.set_marker_out(0, 4)
        #self.hdawg1.set_marker_out(1, 7)
        # self.hdawg1.set_marker_out(2, 0)
        # self.hdawg1.set_marker_out(3, 1)


        # self.hdawg2.set_trigger_impedance_1e3()
        # self.hdawg2.set_dig_trig1_source([1, 1, 1, 1])
        # self.hdawg2.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
        # # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        # self.hdawg2.set_dig_trig2_source([0, 0, 0, 0])
        # self.hdawg2.set_dig_trig2_slope([1, 1, 1, 1])
        # self.hdawg2.set_trig_level(0.3)





        self.hdawg1.set_offset(offset=0, channel=0)
        self.hdawg1.set_offset(offset=0, channel=1)
        self.hdawg1.set_offset(offset=0, channel=2)
        self.hdawg1.set_offset(offset=0, channel=3)
        self.hdawg1.set_offset(offset=0, channel=5)
        self.hdawg1.set_offset(offset=0, channel=6)

        # self.hdawg2.set_offset(offset=0, channel=0)
        # self.hdawg2.set_offset(offset=0, channel=1)
        # self.hdawg2.set_offset(offset=0, channel=2)
        # self.hdawg2.set_offset(offset=0, channel=3)
        # self.hdawg2.set_offset(offset=0, channel=4)
        # self.hdawg2.set_offset(offset=0, channel=5)
        # self.hdawg2.set_offset(offset=0, channel=6)
        # self.hdawg2.set_offset(offset=0, channel=7)

        self.ro_trg = awg_digital(self.hdawg1, 0, delay_tolerance=20e-9)  # triggers readout card
        self.ro_trg.adc = self.adc
        self.ro_trg.mode = 'waveform'
        self.hardware_state = 'pulsed_mode'

        # I don't know HOW but it works
        # For each excitation sequencers:
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        for ex_seq_id in range(4):
            self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/strobe/slope' % ex_seq_id, 1)
            # We need to set DIO valid polarity as High (0- none, 1 - low, 2 - high, 3 - both )
            self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/valid/polarity' % ex_seq_id, 2)
            self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/strobe/index' % ex_seq_id, 0)

        # for ex_seq_id in range(2):
        #     self.hdawg2.daq.setInt('/' + self.hdawg2.device + '/awgs/%d/dio/strobe/slope' % ex_seq_id, 1)
        #     # We need to set DIO valid polarity as High (0- none, 1 - low, 2 - high, 3 - both )
        #     self.hdawg2.daq.setInt('/' + self.hdawg2.device + '/awgs/%d/dio/valid/polarity' % ex_seq_id, 0)
        #     self.hdawg2.daq.setInt('/' + self.hdawg2.device + '/awgs/%d/dio/strobe/index' % ex_seq_id, 0)


        # for ex_seq_id in [3]:  # 6,7
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/timeconstant' % ex_seq_id, 25e-9)
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/timeconstant' % ex_seq_id, 400e-9)
        #
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/amplitude' % ex_seq_id,-0.030)
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/amplitude' % ex_seq_id, -0.010)
        #
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/enable' % ex_seq_id, 1)
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/enable' % ex_seq_id, 1)
        #
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/enable' % ex_seq_id, 1)

        # for ex_seq_id in [0,2,3]: #6,7
        # for ex_seq_id in [0, 1]:  # 6,7
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/timeconstant' % ex_seq_id, 25e-9)
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/timeconstant' % ex_seq_id, 400e-9)
        #
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/amplitude' % ex_seq_id,-0.030)
        #     self.hdawg1.daq.setDouble('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/amplitude' % ex_seq_id, -0.010)
        #
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/0/enable' % ex_seq_id, 1)
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/exponentials/1/enable' % ex_seq_id, 1)
        #
        #     self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/sigouts/%d/precompensation/enable' % ex_seq_id, 1)

        # For readout channels
        # For readout sequencer:
        read_seq_id = self.ro_trg.channel // 2
        # We need to set DIO slope as  Fall (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/strobe/slope' % read_seq_id, 1)
        # We need to set DIO valid polarity as  None (0- none, 1 - low, 2 - high, 3 - both )
        self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/valid/polarity' % read_seq_id, 0)
        self.hdawg1.daq.setInt('/' + self.hdawg1.device + '/awgs/%d/dio/strobe/index' % read_seq_id, 3)
        # self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/value' % read_seq_id, 2)
        # self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/shift' % read_seq_id, 1)
        # For readout channels


    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch.do_get_switch(channel=channel) != value:
            self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        self.iq_devices = {'iq_ro':  qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg1, sequencer_id=0,
                                                                                  lo=self.lo_vna, exdir_db=exdir_db),
                           # 'iq_ex1': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg = self.hdawg1, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex2': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg2, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex6': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg1, sequencer_id=1,
                           #                                                      lo=self.lo_q1, exdir_db=exdir_db),
                           'iq_ex7_12': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg1, sequencer_id=1,
                                                                                lo=self.lo_q1, exdir_db=exdir_db),
                           # 'iq_ex4': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg1, sequencer_id=1,
                           #                                                      lo=self.lo_q1, exdir_db=exdir_db),
                           'iq_ex7': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg1, sequencer_id=2,
                                                                                lo=self.lo_q2, exdir_db=exdir_db)
                           }

        self.iq_devices['iq_ro'].name = 'ro'
        # self.iq_devices['iq_ex1'].name = 'ex1'
        # self.iq_devices['iq_ex6'].name = 'ex6'
        # self.iq_devices['iq_ex5'].name = 'ex5'
        self.iq_devices['iq_ex7'].name = 'ex7'
        self.iq_devices['iq_ex7_12'].name = 'ex7_12'
        # self.iq_devices['iq_ex4'].name = 'ex4'
        # self.iq_devices['iq_ex2'].name = 'ex2'
        # self.iq_devices['iq_ex6'].name = 'ex6'
        self.iq_devices['iq_ro'].sa = self.sa
        # self.iq_devices['iq_ex1'].sa = self.sa
        # self.iq_devices['iq_ex6'].sa = self.sa
        self.iq_devices['iq_ex7'].sa = self.sa
        self.iq_devices['iq_ex7_12'].sa = self.sa
        # self.iq_devices['iq_ex6'].sa = self.sa
        # self.iq_devices['iq_ex2'].sa = self.sa
        # self.iq_devices['iq_ex4'].sa = self.sa
        # self.iq_devices['iq_ro'].calibration_switch_setter = lambda: self.set_switch_if_not_set(6, channel=1)
        # self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.set_switch_if_not_set(1, channel=1)
        # self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: self.set_switch_if_not_set(2, channel=1)
        # self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: self.set_switch_if_not_set(3, channel=1)
        # self.iq_devices['iq_ex4'].calibration_switch_setter = lambda: self.set_switch_if_not_set(4, channel=1)



        self.fast_controls = {#'q1z':awg_channel(self.awg_tek, 1),
                              #'q2z':awg_channel(self.awg_tek, 2),
                              #'q3z':awg_channel(self.awg_tek, 3),
                              #'q4z': awg_channel(self.awg_tek, 4),
                              # 'c5z':awg_channel(self.hdawg1, 6),
                              # 'c6z':awg_channel(self.hdawg1, 7),}
                              'c7z':awg_channel(self.hdawg1, 6),
                              'c8z':awg_channel(self.hdawg1, 7),
                              # 'c8z':awg_channel(self.hdawg1, 7),
        }


    def get_readout_trigger_pulse_length(self):
        return self.pulsed_settings['trigger_readout_length']

    def get_modem_dc_calibration_amplitude(self):
        return self.pulsed_settings['modem_dc_calibration_amplitude']

    def revert_setup(self, old_settings):
        if 'adc_nums' in old_settings:
            self.adc.set_adc_nums(old_settings['adc_nums'])
        if 'adc_nop' in old_settings:
            self.adc.set_adc_nop(old_settings['adc_nop'])
        if 'adc_posttrigger' in old_settings:
            self.adc.set_posttrigger(old_settings['adc_posttrigger'])

