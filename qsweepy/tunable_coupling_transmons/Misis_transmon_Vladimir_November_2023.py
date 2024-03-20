import qsweepy.libraries.instruments as instruments
import qsweepy
from qsweepy.libraries.awg_channel2 import awg_channel
from qsweepy.libraries.awg_digital2 import awg_digital
import numpy as np
from qsweepy import zi_scripts

# from qsweepy.instrument_drivers._QubitDAQ.reg_intf import *
# from qsweepy.instrument_drivers._QubitDAQ.usb_intf import *
import qsweepy.instrument_drivers._QubitDAQ.driver as drv

device_settings = {
                   'vna_address': 'TCPIP0::ZVB20-23-100170::inst0::INSTR',#'TCPIP0::10.20.61.66::inst0::INSTR', # TCPIP0::10.20.61.94::inst0::INSTR  TCPIP0::10.20.61.68::inst0::INSTR
                   # 'rf_switch_address': '10.20.61.91',
                   'use_rf_switch': False,
### закоменчено из-за сломанного Цуриха
                   'hdawg_address': 'hdawg-dev8108', #8250 8108
###
                   'sa_ro_address': 'TCPIP0::10.1.0.72::inst0::INSTR',
                   'sa_address': 'TCPIP0::10.20.61.37::inst0::INSTR',
                   'lo1_address': 'TCPIP0::10.1.0.54::inst0::INSTR',
                   'adc_timeout': 10,#25, 10
                   'adc_trig_rep_period': 20,  #10 -  12.5 MHz rate period
                   'adc_trig_width': 2,  # 32 ns trigger length
                   'anapico_address': 'USB0::0x03EB::0xAFFF::3B5-0B4M4001C-0781::INSTR'
                   }

cw_settings = { 'mixer_thru': 0.5 }

pulsed_settings = {'lo1_power': 14,
                   'vna_power': 16,
                   'ex_clock': 2400e6,  # 1 GHz - clocks of some devices
                   'ro_clock': 1000e6,
                   # закоменчено из-за сломанного Цуриха
                   'hdawg_ch0_amplitude': 0.8,
                   'hdawg_ch1_amplitude': 0.8,
                   'hdawg_ch2_amplitude': 0.8,
                   'hdawg_ch3_amplitude': 0.8,
                   'hdawg_ch4_amplitude': 0.8,
                   'hdawg_ch5_amplitude': 0.8,
                   'hdawg_ch6_amplitude': 0.8,
                   'hdawg_ch7_amplitude': 0.8,
                   ###
                   # 'lo1_freq': 3.2e9, #1.5e9,
                   'lo1_freq': 4.7e9, #4.6e9,5.15e9,#4.5e9, #1.5e9,5.15e9,  5.4e9, 4.5e9, 5.6e9
                   'pna_freq': 7.2e9, #6.7e9, #,7.9e9, 6.5e9, 7.25e9
                   #'calibrate_delay_nop': 65536,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_length': 200e-9,
                   'modem_dc_calibration_amplitude': 1.0,
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
        self.lo1 = None
        self.lo_ro = None
        self.lo_pump = None
        self.rf_switch = None
        self.coil_device = None
### закоменчено из-за сломанного Цуриха
        self.hdawg = None
###
        self.adc_device = None
        self.adc = None

        self.ro_trg = None
        self.q1z = None
        self.q2z = None
        self.q3z = None
        self.iq_devices = None
        self.fast_controls = None

    def open_devices(self):
        # RF switch for making sure we know what sample we are measuring
        self.pna = instruments.RS_ZVB20('pna', address=self.device_settings['vna_address'])
        # self.pna = instruments.Agilent_N5242A('pna', address=self.device_settings['vna_address'])
        self.lo_ro = instruments.AnaPicoAPSIN('lo1', address=self.device_settings['anapico_address'], channel=1)
        # self.lo_pump = instruments.Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])
        # self.lo1 = instruments.Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])
        self.lo1 = prokladka(self.lo_ro, name='lo2', channel=2)
        self.lo2 = prokladka(self.lo_ro, name='lo4', channel=4)

        # self.lo1 = instruments.Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])

        self.bias = instruments.Yokogawa_GS210(address='GPIB0::2::INSTR')
        #self.lo1._visainstrument.timeout = self.device_settings['lo1_timeout']
        #self.lo1 = instruments.SignalCore_5502a()
        #self.lo1.search()
        #self.lo1.open()

        if self.device_settings['use_rf_switch']:
            self.rf_switch = instruments.nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])
### закоменчено из-за сломанного Цуриха
        self.hdawg = instruments.ZIDevice(self.device_settings['hdawg_address'], devtype='HDAWG', clock=2.4e9, delay_int=0)

        for channel_id in range(8):
            self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/range' % channel_id, 1)
        # It is necessary if you want to use DIOs control features during pulse sequence
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/dios/0/mode', 1)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/dios/0/drive', 1)
        self.hdawg.daq.sync()
###
        #
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/0/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/1/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/2/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/3/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/4/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/5/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/6/range', 0.8)
        # self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/7/range', 0.8)
### закоменчено из-за сломанного Цуриха
        self.coil_device = self.hdawg
###
        # Qubit lines should be connected with even channels
        # self.q1z = awg_channel(self.hdawg, 2)  # coil control
### закоменчено из-за сломанного Цуриха
        self.q2z = awg_channel(self.hdawg, 4)  # coil control
###
        # self.q3z = awg_channel(self.hdawg, 6)  # coil control



        # self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])
        self.sa_ro = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_ro_address'])

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

        # self.adc_device = drv.Device("0009052001482f25")
        # НАДО ОТКОММЕНТИРОВАТЬ
        self.adc_device = drv.Device("0009052001481708")
        self.data_capture_timeout = self.device_settings['adc_timeout']
        self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        self.adc.output_raw = True
        self.adc.last_cov = False
        self.adc.avg_cov = False
        self.adc.resultnumber = False

        self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length

        self.hardware_state = 'undefined'
        # НАДО ОТКОММЕНТИРОВАТЬ

    def set_cw_mode(self, channels_off=None):
        if self.hardware_state == 'cw_mode':
            return
        self.hardware_state = 'cw_mode'
### закоменчено из-за сломанного Цуриха
        self.hdawg.stop()
###
        # self.cw_sequence = zi_scripts.CWSequence(awg=self.hdawg, sequencer_id=3)
        #self.hdawg.set_sequencer(self.cw_sequence)
        # self.hdawg.set_sequence(2, self.cw_sequence)
        # self.cw_sequence.set_amplitude_i(0)
        # self.cw_sequence.set_amplitude_q(0)
        # self.cw_sequence.set_phase_i(0)
        # self.cw_sequence.set_phase_q(0)
        # self.cw_sequence.set_offset_i(cw_settings['mixer_thru'])
        # self.cw_sequence.set_offset_q(0)

        # 0, 1 channels for readout, 6, 7 for ex 12 transition
        # for channel in [0, 1, 6, 7]:
### закоменчено из-за сломанного Цуриха
        self.lo_ro.set_status(0)

        self.lo1.set_status(0)
        try:
            self.lo2.set_status(0)
        except:
            pass
        ###
        self.pna.do_set_status(1)
### закоменчено из-за сломанного Цуриха
        for channel in [0, 1,2,3]:
            self.hdawg.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)

        self.hdawg.set_output(output=1, channel=0)
        self.hdawg.set_output(output=1, channel=1)
        self.hdawg.set_output(output=1, channel=2)
        self.hdawg.set_output(output=1, channel=3)
        self.hdawg.set_output(output=1, channel=4)
        self.hdawg.set_output(output=1, channel=5)
        self.hdawg.set_output(output=1, channel=6)
        self.hdawg.set_output(output=1, channel=7)
###
        self.pna.set_sweep_mode("LIN")
        self.hardware_state = 'cw_mode'


    def set_pulsed_mode(self):
        if self.hardware_state == 'pulsed_mode':
            return
        self.hardware_state = 'undefined'


        self.pna = instruments.RS_ZVB20('pna', address=self.device_settings['vna_address'])
        self.lo1.set_status(1)  # turn on lo1 output
        self.lo1.set_power(self.pulsed_settings['lo1_power'])
        self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

        try:
            self.lo2.set_status(1)  # turn on lo1 output
            self.lo2.set_power(self.pulsed_settings['lo1_power'])
            self.lo2.set_frequency(self.pulsed_settings['lo1_freq'])
        except:
            pass

        self.pna.do_set_status(1)
        self.pna.set_power(self.pulsed_settings['vna_power'])

        # # self.pna.write("OUTP ON")
        # # self.pna.write("SOUR1:POW1:MODE OFF")
        # # self.pna.write("SOUR1:POW2:MODE ON")
        # self.pna.set_sweep_mode("CW") # privet RS ZVB20
        # #self.pna.set_trigger_source("ON")
        # self.pna.set_frequency(self.pulsed_settings['pna_freq'])


        self.lo_ro.set_status(1)  # turn on lo1 output
        self.lo_ro.set_power(self.pulsed_settings['vna_power'])
        self.lo_ro.set_frequency(self.pulsed_settings['pna_freq'])
        self.pna.do_set_status(0)
        self.pna.write("SOUR1:POW1:MODE OFF")


        self.lo1.set_status(1)  # turn on lo1 output
        self.lo1.set_power(self.pulsed_settings['lo1_power'])
        self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

### закоменчено из-за сломанного Цуриха
        self.hdawg.stop()

        self.hdawg.set_clock(self.pulsed_settings['ex_clock'])
        self.hdawg.set_clock_source(0)

        self.hdawg.set_trigger_impedance_1e3()
        #self.hdawg.set_trigger_impedance_50()
        self.hdawg.set_dig_trig1_source([0, 2, 2, 2])#[6, 6, 4, 6]
        self.hdawg.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
                                                      # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        self.hdawg.set_dig_trig2_source([1, 2, 2, 2])#[0, 0, 5, 0]
        self.hdawg.set_dig_trig2_slope([1, 1, 1, 1])
        self.hdawg.set_trig_level(0.3)
        self.hdawg.set_trig_level(1.0, channels=[0])

        self.hdawg.set_marker_out(0, 0)
        self.hdawg.set_marker_out(1, 1)
        self.hdawg.set_marker_out(2, 2)
###
        # self.hdawg.set_marker_out(1, 1)
        # self.hdawg.set_marker_out(3, 1)
        # self.hdawg.set_marker_out(5, 1)
        # self.hdawg.set_marker_out(7, 1)


        self.ro_trg = awg_digital(self.hdawg, 0, delay_tolerance=20e-9)  # triggers readout card # Было 4
        self.ro_trg.adc = self.adc
        self.ro_trg.mode = 'waveform'
        self.hardware_state = 'pulsed_mode'

        # I don't know HOW but it works
        # For each exitation sequencers:
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        # for ex_seq_id in range(4):
### закоменчено из-за сломанного Цуриха
        for ex_seq_id in [1]:
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/slope' % ex_seq_id, 1)
            # We need to set DIO valid polarity as High (0- none, 1 - low, 2 - high, 3 - both )
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/valid/polarity' % ex_seq_id, 2)
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/index' % ex_seq_id, 0)

        # For readout channels
        # For readout sequencer:
        read_seq_id = self.ro_trg.channel //2
        # We need to set DIO slope as  Fall (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/slope' % read_seq_id, 1)
        # We need to set DIO valid polarity as  None (0- none, 1 - low, 2 - high, 3 - both )
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/valid/polarity' % read_seq_id, 0)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/index' % read_seq_id, 3)
###
        # for ex_seq_id in [0,2,3]: #6,7
        #     self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/timeconstant' % ex_seq_id, 25e-9)
        #     self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/timeconstant' % ex_seq_id, 400e-9)
        #
        #     self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/amplitude' % ex_seq_id,-0.030)
        #     self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/amplitude' % ex_seq_id, -0.010)
        #
        #     self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/enable' % ex_seq_id, 1)
        #     self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/enable' % ex_seq_id, 1)
        #
        #     self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/%d/precompensation/enable' % ex_seq_id, 1)


        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/value' % read_seq_id, 2)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/shift' % read_seq_id, 1)
        # For readout channels

    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch is not None:
            if self.rf_switch.do_get_switch(channel=channel) != value:
                self.rf_switch.do_set_switch(value, channel=channel)
### закоменчено из-за сломанного Цуриха
    def setup_iq_channel_connections(self, exdir_db):
        # промежуточные частоты для гетеродинной схемы new:

        # self.iq_devices = {'iq_ro':  qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=0,
        #                                                                           lo=self.pna, exdir_db=exdir_db),
        #                    # 'iq_ex': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=3,
        #                    #                                                       lo=self.lo1, exdir_db=exdir_db),
        #                    'iq_ex1_12': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=3,
        #                                                                            lo=self.lo1, exdir_db=exdir_db)
        #                    # 'iq_ex2_12': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=3,
        #                    #                                                        lo=self.lo1, exdir_db=exdir_db),
        #                    }

        self.iq_devices = {'iq_ro': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=0,
                                                                               lo=self.lo_ro, exdir_db=exdir_db),
                           'iq_ex1': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                                                                                   lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex2': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=2,
                           #                                                         lo=self.lo1, exdir_db=exdir_db),
                           'iq_ex1_2': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=3,
                                                                                  lo=self.lo2, exdir_db=exdir_db),
                           }

        self.iq_devices['iq_ro'].name = 'ro'
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: None
        self.iq_devices['iq_ro'].sa = self.sa_ro


        self.iq_devices['iq_ex1'].name = 'ex1'
        self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: None
        self.iq_devices['iq_ex1'].sa = self.sa_ro

        self.iq_devices['iq_ex1_2'].name = 'ex1_2'
        self.iq_devices['iq_ex1_2'].calibration_switch_setter = lambda: None
        self.iq_devices['iq_ex1_2'].sa = self.sa_ro

        # self.iq_devices['iq_ex2'].name = 'ex2'
        # self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex2'].sa = self.sa_ro
        #
        # self.iq_devices['iq_ex3'].name = 'ex3'
        # self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex3'].sa = self.sa_ro

        self.fast_controls = {
        #                       # 'q3z': awg_channel(self.hdawg, 6),
        #                       'q2z': awg_channel(self.hdawg, 4),
        #                       'q1z': awg_channel(self.hdawg, 2),
                              }  # coil control

    def get_modem_dc_calibration_amplitude(self):
        return self.pulsed_settings['modem_dc_calibration_amplitude']
##