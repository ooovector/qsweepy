import qsweepy.libraries.instruments as instruments
import qsweepy
from qsweepy.libraries.awg_channel2 import awg_channel
from qsweepy.libraries.awg_digital2 import awg_digital
import numpy as np


from qsweepy.instrument_drivers._QubitDAQ.reg_intf import *
from qsweepy.instrument_drivers._QubitDAQ.usb_intf import *
import qsweepy.instrument_drivers._QubitDAQ.driver as drv

from qsweepy import zi_scripts

# device_settings = {
#     # 'vna_address': 'TCPIP0::10.20.61.157::inst0::INSTR',
#     #'vna_address': 'TCPIP0::10.20.61.147::inst0::INSTR',
#     'vna_address': 'TCPIP0::ZVB20-23-100170::inst0::INSTR',
#     # 'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
#     # 'lo1_timeout': 5000,
#     'lo1_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
#     'rf_switch_address': '10.20.61.91',
#     'use_rf_switch': True,
#     'pxi_chassis_id': 0,
#     'hdawg1_address': 'hdawg-dev8108',
#     'hdawg2_address': 'hdawg-dev8250',
#     'sa_address_old': 'TCPIP0::10.20.61.56::inst0::INSTR',
#     'sa_address': 'TCPIP0::10.20.61.37::inst0::INSTR',
#     'adc_timeout': 20, #15, #10,
#     'adc_trig_rep_period': 20,  # 12.5 MHz rate period
#     'adc_trig_width': 2,  # 32 ns trigger length
#     'awg_tek_address':'TCPIP0::10.20.61.113::inst0::INSTR',
#     'anapico_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
#     'anapico_address1': 'TCPIP0::10.20.61.154::inst0::INSTR', # single port anapico
#     'lo_agilent': 'TCPIP0::10.20.61.59::inst0::INSTR'
# }

device_settings = {
    'vna_address': 'TCPIP0::ZVB20-23-100170::inst0::INSTR',
    'lo1_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
    'rf_switch_address': '10.20.61.91',
    'use_rf_switch': True,
    'pxi_chassis_id': 0,
    'hdawg1_address': 'hdawg-dev8108',
    'hdawg2_address': 'hdawg-dev8250',
    'sa_address_old': 'TCPIP0::10.20.61.56::inst0::INSTR',
    # 'sa_address': 'TCPIP0::10.20.61.37::inst0::INSTR',
    'sa_address': 'TCPIP0::10.1.0.72::inst0::INSTR',
    'adc_timeout': 20, #15, #10,
    'adc_trig_rep_period': 20,  # 12.5 MHz rate period
    'adc_trig_width': 2,  # 32 ns trigger length
    'awg_tek_address':'TCPIP0::10.20.61.113::inst0::INSTR',
    'anapico_address': 'TCPIP0::10.20.61.197::inst0::INSTR',
    'anapico_address1': 'TCPIP0::10.20.61.154::inst0::INSTR', # single port anapico
    # 'lo_agilent': 'TCPIP0::10.20.61.59::inst0::INSTR',
    'lo_agilent': 'TCPIP0::10.1.0.54::inst0::INSTR'
}

cw_settings = {'mixer_thru': 0.6}

pulsed_settings = {'lo1_power': 14, #15,
                   'vna_power': 16, #15,
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
                   'lo1_freq': 4.21e9,  # 4.35e9 frequency or dc excitation calibration
                   'pna_freq': 7.0e9, #7.2e9
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
        return(self.lo1.get_power(channel = self.channel))
    def get_frequency(self):
        return(self.lo1.get_frequency(channel=self.channel))
    def get_status(self):
        return(self.lo1.get_status(channel=self.channel))

class hardware_setup():
    def __init__(self, device_settings, pulsed_settings):
        self.device_settings = device_settings
        self.pulsed_settings = pulsed_settings
        self.cw_settings = cw_settings
        self.hardware_state = 'undefined'
        self.sa = None

        self.pna = None
        self.lo1 = None
        self.lo2 = None
        self.rf_switch = None
        self.coil_device = None
        self.hdawg = None
        self.adc_device = None
        self.adc = None
        self.awg_tek = None

        self.ro_trg = None
        self.q1z = None
        self.q2z = None
        self.q3z = None
        self.q4z = None
        self.q5z = None
        self.q6z = None
        self.iq_devices = None



    def open_devices(self):
        qubits_nu = 6
        # VNA
        self.pna = instruments.RS_ZVB20('pna', address=self.device_settings['vna_address'])
        # Anapico
        # self.lo1 = instruments.AnaPicoAPSIN('lo1', address=self.device_settings['anapico_address1'])
        self.lo1 = instruments.Agilent_E8257D('lo1', address=self.device_settings['lo_agilent'])

        # # Tektronix
        # # self.awg_tek = instruments.Tektronix_AWG5014('awg_tek', address = self.device_settings['awg_tek_address'])
        #
        # # sleep(1)
        # self.lo1 = instruments.AnaPicoAPSIN('lo1', address=self.device_settings['anapico_address'], channel=1)
        # self.lo2 = prokladka(self.lo1, name='lo2', channel=2)
        #
        #Spectral
        self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])
        #
        # # SWITCH
        # if self.device_settings['use_rf_switch']:
        #     self.rf_switch = instruments.nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])
        #
        # HDAWG Zurich
        self.hdawg = instruments.ZIDevice(self.device_settings['hdawg2_address'], devtype='HDAWG', delay_int=0)

        for channel_id in range(8):
            self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/%d/range' % channel_id, 1)

        #It is necessary if you want to use DIOs control features during pulse sequence
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/dios/0/mode', 1)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/dios/0/drive', 1)
        self.hdawg.daq.sync()


        # self.adc_device = drv.Device("0009052001482f25")
        self.adc_device = drv.Device("0009052001481708")
        self.data_capture_timeout = self.device_settings['adc_timeout']
        self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        self.adc.output_raw = True
        self.adc.last_cov = False
        self.adc.avg_cov = False
        self.adc.resultnumber = False

        self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length



        # """
        # OLd ADC driver
        # """
        # # self.adc_device.timeout = self.device_settings['adc_timeout']
        # # self.adc_device = instruments.TSW14J56_evm()
        # # self.adc_device.timeout = self.device_settings['adc_timeout']
        # # self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        # # self.adc.output_raw = True
        # # self.adc.last_cov = False
        # # self.adc.avg_cov = False
        # # self.adc.resultnumber = False
        # #
        # # self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        # # self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length
        #
        # self.coil_device = self.hdawg
        # self.q1z = awg_channel(self.hdawg, 7)
        # self.q1z.set_offset(0)
        # self.q2z = awg_channel(self.hdawg, 7)
        # self.q2z.set_offset(0)

        # self.q3z = awg_channel(self.hdawg, 7)
        # self.q3z.set_offset(0)
        #
        # self.q4z = awg_channel(self.hdawg, 7)
        # self.q4z.set_offset(0)

        # self.q5z = awg_channel(self.hdawg, 7)
        # self.q5z.set_offset(0)

        # self.q6z = awg_channel(self.hdawg, 7)
        # self.q6z.set_offset(0)
        #
        # self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])
        #
        #
        # # # self.awg_tek.set_nop(100)
        # # # self.awg_tek.set_waveform([0] * 100, channel=1)
        # # # self.awg_tek.set_waveform([0] * 100, channel=2)
        # # # self.awg_tek.set_waveform([0] * 100, channel=3)
        # # # self.awg_tek.set_waveform([0] * 100, channel=4)


    def set_cw_mode(self, channels_off=None):
        """
        Continuous wave mode for spectral measurements
        :param channels_off: channels which are note active
        """
        if self.hardware_state == 'cw_mode':
            return
        self.hardware_state = 'cw_mode'
        self.hdawg.stop()

        global_num_points = int(np.round(
            self.pulsed_settings['ex_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
                'global_num_points_delta']))

        # we use two hdawg channels for ro
        hdawg_channels = [0, 1]
        for channel in hdawg_channels:
            self.hdawg.set_output(output=1, channel=channel) # what is output???

        # For cw mode measurements we need to open I и Q for mixer ro1 to allow pass signal from с lo (vna) to a fridge.
        # For this reason we add quasi constant dc shift to I and Q quadratures with relatively high voltage values.

        for channel in [0, 1, 2, 3]:
            self.hdawg.set_amplitude(amplitude=0.05, channel=channel)
            self.hdawg.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            self.hdawg.set_output(output=1, channel=channel)
            # self.hdawg.set_waveform(waveform=[0] * global_num_points, channel=channel)

        if channels_off is not None:
            for channel_off in channels_off:
                self.hdawg.set_output(output=0, channel=channel_off)

        self.pna.write("SOUR1:POW1:MODE ON")
        self.pna.write("SOUR1:POW2:MODE OFF")
        self.pna.set_sweep_mode("LIN")
        self.hardware_state = 'cw_mode'
        # self.awg_tek.set_ch1_output(1)
        # self.awg_tek.set_ch2_output(1)
        # self.awg_tek.set_ch3_output(1)
        # self.awg_tek.set_ch4_output(1)

    def set_pulsed_mode(self):
        if self.hardware_state == 'pulsed_mode':
            return
        self.hardware_state = 'undefined'

        self.pna = instruments.RS_ZVB20('pna', address=self.device_settings['vna_address'])
        self.lo1.set_status(1)  # turn on lo1 output
        self.lo1.set_power(self.pulsed_settings['lo1_power'])
        self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.pna.do_set_status(1)
        self.pna.set_power(self.pulsed_settings['vna_power'])

        # self.pna.write("OUTP ON")
        # self.pna.write("SOUR1:POW1:MODE OFF")
        # self.pna.write("SOUR1:POW2:MODE ON")
        self.pna.set_sweep_mode("CW")  # privet RS ZVB20
        # self.pna.set_trigger_source("ON")
        self.pna.set_frequency(self.pulsed_settings['pna_freq'])

        # self.lo1.set_status(1)  # turn on lo1 output
        # self.lo1.set_power(self.pulsed_settings['lo1_power'])
        # self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])
        #
        #
        #
        # # self.pna.set_power(self.pulsed_settings['vna_power'])
        # # self.pna.write("SOUR1:POW1:MODE ON")
        # # self.pna.write("SOUR1:POW2:MODE OFF")
        # # self.pna.set_sweep_mode("CW")  # privet RS ZVB20
        # # self.pna.set_frequency(self.pulsed_settings['pna_freq'])
        #
        # # self.pna.write("SOUR1:POW1:MODE OFF")
        # # self.pna.write("SOUR1:POW2:MODE OFF")
        # # self.lo2.set_status(1)  # turn on lo1 output
        # # self.lo2.set_power(self.pulsed_settings['vna_power'])
        # # self.lo2.set_frequency(self.pulsed_settings['pna_freq'])

        self.hdawg.stop()

        self.hdawg.set_clock(self.pulsed_settings['ex_clock']) # probably it is important
        self.hdawg.set_clock_source(0)


        #TODO: something with triggers and markers


        # its fine
        # self.hdawg.set_trigger_impedance_1e3()
        # self.hdawg.set_dig_trig1_source([0, 0, 0, 0])
        # self.hdawg.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
        # # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        # self.hdawg.set_dig_trig2_source([1, 1, 1, 1])
        # self.hdawg.set_dig_trig2_slope([1, 1, 1, 1])
        # self.hdawg.set_trig_level(0.3)
        # self.hdawg.set_marker_out(2, 0)
        # self.hdawg.set_marker_out(3, 1)
        """
        set_dig_trig1_source и set_dig_trig2_source задают цифровые триггеры (Trigger In) для каждого сиквенсера (два на сиквенсер),
        причем триггером может являться любой физических выход триггера hdawg. Здесь они нумеруются с нуля, в LabOne с единицы.
        """

        self.hdawg.set_trigger_impedance_1e3()
        self.hdawg.set_dig_trig1_source([0, 2, 2, 2])
        self.hdawg.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
        # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        self.hdawg.set_dig_trig2_source([1, 2, 2, 2])
        self.hdawg.set_dig_trig2_slope([1, 1, 1, 1])
        self.hdawg.set_trig_level(0.3)
        """
        Нужно определить маркеры для подачи триггера, здесь мы опредеяем их через Trigger Out
        set_marker_out(физический канал маркера на hdawg от 0 до 7, запись четырех триггеров в бинарной форме)
        """

        self.hdawg.set_marker_out(0, 0)
        self.hdawg.set_marker_out(1, 1)
        self.hdawg.set_marker_out(2, 2)
        # self.hdawg.set_marker_out(3, 0)

        # self.hdawg.set_marker_out(5, 1)
        # self.hdawg.set_marker_out(1, 1)
        # self.hdawg.set_marker_out(3, 1)


        # self.hdawg.set_dig_trig1_source([0, 0, 0, 0])
        # self.hdawg.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
        # # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        # self.hdawg.set_dig_trig2_source([1, 1, 1, 1])
        # self.hdawg.set_dig_trig2_slope([1, 1, 1, 1])
        # self.hdawg.set_trig_level(0.3)
        # self.hdawg.set_marker_out(1, 6)
        # self.hdawg.set_marker_out(2, 4)


        """
        Ro trigger to ADC
        """
        self.ro_trg = awg_digital(self.hdawg, 0, delay_tolerance=20e-9)  # triggers readout card
        # self.ro_trg = awg_digital(self.hdawg, 0, delay_tolerance=20e-9)  # triggers readout card
        self.ro_trg.adc = self.adc
        self.ro_trg.mode = 'waveform'
        self.hardware_state = 'pulsed_mode'

        # I don't know HOW but it works
        # For each excitation sequencers:
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        # for ex_seq_id in [1]:
        for ex_seq_id in [1]:
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/slope' % ex_seq_id, 1)
            # We need to set DIO valid polarity as High (0- none, 1 - low, 2 - high, 3 - both )
            # self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/valid/polarity' % ex_seq_id, 2)
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/valid/polarity' % ex_seq_id, 2)
            #self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/index' % ex_seq_id, 1)
            self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/index' % ex_seq_id, 0)


        # for ex_seq_id in [0,2,3]: #6,7
        # for ex_seq_id in [0, 1]:  # 6,7
        #     self.hdawg.daq.setDouble(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/timeconstant' % ex_seq_id, 25e-9)
        #     self.hdawg.daq.setDouble(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/timeconstant' % ex_seq_id,
        #         400e-9)
        #
        #     self.hdawg.daq.setDouble(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/amplitude' % ex_seq_id, -0.030)
        #     self.hdawg.daq.setDouble(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/amplitude' % ex_seq_id, -0.010)
        #
        #     self.hdawg.daq.setInt(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/0/enable' % ex_seq_id, 1)
        #     self.hdawg.daq.setInt(
        #         '/' + self.hdawg.device + '/sigouts/%d/precompensation/exponentials/1/enable' % ex_seq_id, 1)
        #
        #     self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/%d/precompensation/enable' % ex_seq_id, 1)

        # For readout channels
        # For readout sequencer:
        read_seq_id = self.ro_trg.channel // 2
        # We need to set DIO slope as  Fall (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/slope' % read_seq_id, 1)
        # We need to set DIO valid polarity as  None (0- none, 1 - low, 2 - high, 3 - both )
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/valid/polarity' % read_seq_id, 0)
        self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/strobe/index' % read_seq_id, 3)
        # self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/value' % read_seq_id, 2)
        # self.hdawg.daq.setInt('/' + self.hdawg.device + '/awgs/%d/dio/mask/shift' % read_seq_id, 1)
        # For readout channels

    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch.do_get_switch(channel=channel) != value:
            self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        self.iq_devices = {'iq_ro':  qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=0,
                                                                                  lo=self.pna, exdir_db=exdir_db),
                           # 'iq_ex1': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex2': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg = self.hdawg, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex3': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex4': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           'iq_ex5': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                                                                                lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex6': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=1,
                           #                                                      lo=self.lo1, exdir_db=exdir_db),
                           # 'iq_ex6_12': qsweepy.libraries.awg_iq_multi2.AWGIQMulti(awg=self.hdawg, sequencer_id=2,
                           #                                                      lo=self.lo1, exdir_db=exdir_db)
                           }

        # self.iq_devices['iq_ro'].name = 'ro' # readout mixers
        # self.iq_devices['iq_ex4'].name = 'ex4'  # mixer for ext for qubit 4
        #
        # self.iq_devices['iq_ro'].sa = self.sa
        # self.iq_devices['iq_ex4'].sa = self.sa   # spectral analyzer for iq mixer 4

        self.iq_devices['iq_ro'].name = 'ro'
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: None
        self.iq_devices['iq_ro'].sa = self.sa  # self.sa_ro

        # self.iq_devices['iq_ex1'].name = 'ex'
        # self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex1'].sa = self.sa

        # self.iq_devices['iq_ex2'].name = 'ex'
        # self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex2'].sa = self.sa

        # self.iq_devices['iq_ex3'].name = 'ex'
        # self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex3'].sa = self.sa

        # self.iq_devices['iq_ex4'].name = 'ex'
        # self.iq_devices['iq_ex4'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex4'].sa = self.sa

        self.iq_devices['iq_ex5'].name = 'ex'
        self.iq_devices['iq_ex5'].calibration_switch_setter = lambda: None
        self.iq_devices['iq_ex5'].sa = self.sa

        # self.iq_devices['iq_ex6'].name = 'ex'
        # self.iq_devices['iq_ex6'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex6'].sa = self.sa
        #
        # self.iq_devices['iq_ex6_12'].name = 'ex'
        # self.iq_devices['iq_ex6_12'].calibration_switch_setter = lambda: None
        # self.iq_devices['iq_ex6_12'].sa = self.sa

        # self.fast_controls = {'q4z': awg_channel(self.hdawg, 7)}
        self.fast_controls = {}


        # self.iq_devices['iq_ro'].calibration_switch_setter = lambda: self.set_switch_if_not_set(6, channel=1)
        # self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.set_switch_if_not_set(1, channel=1)
        # self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: self.set_switch_if_not_set(2, channel=1)
        # self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: self.set_switch_if_not_set(3, channel=1)
        # self.iq_devices['iq_ex4'].calibration_switch_setter = lambda: self.set_switch_if_not_set(4, channel=1)
        # self.iq_devices['iq_ex5'].calibration_switch_setter = lambda: self.set_switch_if_not_set(5, channel=1)
        # self.iq_devices['iq_ex6'].calibration_switch_setter = lambda: self.set_switch_if_not_set(6, channel=1)

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