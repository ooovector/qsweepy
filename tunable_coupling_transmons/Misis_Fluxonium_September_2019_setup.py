from qsweepy.instruments import *
from qsweepy import *

from qsweepy import awg_iq_multi

import numpy as np

device_settings = {'vna_address': 'TCPIP0::10.20.61.48::inst0::INSTR',
                   #'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
                   #'lo1_timeout': 5000, 'rf_switch_address': '10.20.61.224',
                   'use_rf_switch': False,
                   'pxi_chassis_id': 0,
                   'hdawg_address': 'hdawg-dev8108',
                   'sa_address': 'TCPIP0::10.20.61.56::inst0::INSTR',
                   'adc_timeout': 10,
                   'adc_trig_rep_period': 100 * 125,  # 10 kHz rate period
                   'adc_trig_width': 2,  # 80 ns trigger length
                   }

cw_settings = {}
pulsed_settings = {#'lo1_power': 18,
                   'vna_power': 16,
                   'ex_clock': 2000e6,  # 1 GHz - clocks of some devices
                   'rep_rate': 10e3,  # 10 kHz - pulse sequence repetition rate
                   # 500 ex_clocks - all waves is shorten by this amount of clock cycles
                   # to verify that M3202 will not miss next trigger
                   # (awgs are always missing trigger while they are still outputting waveform)
                   'global_num_points_delta': 800,
                   'hdawg_ch0_amplitude': 1.0,
                   'hdawg_ch1_amplitude': 1.0,
                   'hdawg_ch2_amplitude': 1.0,
                   'hdawg_ch3_amplitude': 1.0,
                   'hdawg_ch4_amplitude': 1.0,
                   'hdawg_ch5_amplitude': 1.0,
                   'hdawg_ch6_amplitude': 1.0,
                   'hdawg_ch7_amplitude': 1.0,
                   #'lo1_freq': 3.3e9,#3.70e9,
                   'pna_freq': 5.1e9,
                   #'calibrate_delay_nop': 65536,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_channel_name': 'ro_trg',
                   'trigger_readout_length': 200e-9,
                   'modem_dc_calibration_amplitude': 1.0,
                   'adc_nop': 1024,
                   'adc_nums': 50000,  ## Do we need control over this? Probably, but not now... WUT THE FUCK MAN
                   }


class hardware_setup():
    def __init__(self, device_settings, pulsed_settings):
        self.device_settings = device_settings
        self.pulsed_settings = pulsed_settings
        self.cw_settings = cw_settings
        self.hardware_state = 'undefined'

        self.pna = None
        self.lo1 = None
        self.rf_switch = None
        self.sa = None
        self.coil_device = None
        self.hdawg = None
        self.adc_device = None
        self.adc = None

        self.ro_trg = None
        self.coil = None
        self.iq_devices = None

    def open_devices(self):
        # RF switch for making sure we know what sample we are measuring
        self.pna = Agilent_N5242A('pna', address=self.device_settings['vna_address'])
        #self.lo1 = Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])

        #self.lo1._visainstrument.timeout = self.device_settings['lo1_timeout']

        if self.device_settings['use_rf_switch']:
            self.rf_switch = nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])

        self.sa = Agilent_N9030A('pxa', address=self.device_settings['sa_address'])

        self.hdawg = Zurich_HDAWG1808(self.device_settings['hdawg_address'])
        self.dummy_awg = dummy_awg.DummyAWG(channels=1)
        self.coil_device = self.hdawg

        self.adc_device = TSW14J56_evm()
        self.adc_device.timeout = self.device_settings['adc_timeout']
        self.adc = TSW14J56_evm_reducer(self.adc_device)
        self.adc.output_raw = True
        self.adc.last_cov = False
        self.adc.avg_cov = False
        self.adc.resultnumber = False

        self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length

    # self.hardware_state = 'undefined'

    def set_pulsed_mode(self):
        #self.lo1.set_status(1)  # turn on lo1 output
        #self.lo1.set_power(self.pulsed_settings['lo1_power'])
        #self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.pna.set_power(self.pulsed_settings['vna_power'])
        self.pna.write("OUTP ON")
        self.pna.write("SOUR1:POW1:MODE ON")
        self.pna.write("SOUR1:POW2:MODE OFF")
        self.pna.set_sweep_mode("CW")
        self.pna.set_frequency(self.pulsed_settings['pna_freq'])

        self.hdawg.stop()

        self.hdawg.set_clock(self.pulsed_settings['ex_clock'])
        self.hdawg.set_clock_source(1)

        # setting repetition period for slave devices
        # 'global_num_points_delay' is needed to verify that M3202A and other slave devices will be free
        # when next trigger arrives.
        global_num_points = int(np.round(
            self.pulsed_settings['ex_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
                'global_num_points_delta']))

        # global_num_points = 20000

        self.hdawg.set_nop(global_num_points)
        self.hdawg.clear()

        # а вот длину сэмплов, которая очевидно то же самое, нужно задавать на всех авгшках.
        # хорошо, что сейчас она только одна.
        # this is zashkvar   WUT THE FUCK MAN

        self.hdawg.set_trigger_impedance_1e3()
        self.hdawg.set_dig_trig1_source([0, 0, 0, 0])
        self.hdawg.set_dig_trig1_slope([1, 1, 1, 1])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
                                                      # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        self.hdawg.set_dig_trig1_source([0, 0, 0, 0])
        self.hdawg.set_dig_trig2_slope([1, 1, 1, 1])
        self.hdawg.set_trig_level(0.6)

        for sequencer in range(4):
            self.hdawg.send_cur_prog(sequencer=sequencer)
            self.hdawg.set_marker_out(channel=np.int(2 * sequencer), source=4)  # set marker 1 to awg mark out 1 for sequencer
            self.hdawg.set_marker_out(channel=np.int(2 * sequencer + 1),
                                      source=7)  # set marker 2 to awg mark out 2 for sequencer
        for channel in range(8):
            self.hdawg.set_amplitude(channel=channel, amplitude=self.pulsed_settings['hdawg_ch%d_amplitude'%channel])
            self.hdawg.set_offset(channel=channel, offset=0 * 1.0)
            self.hdawg.set_digital(channel=channel, marker=[0]*(global_num_points))
            self.hdawg.daq.set([['/{}/sigouts/{}/range'.format(self.hdawg.device, channel), 0.4]])
        #self.hdawg.daq.set([['/{}/sigouts/4/range'.format(self.hdawg.device), 2]])
        self.hdawg.set_all_outs()
        self.hdawg.run()

        self.ro_trg = awg_digital.awg_digital(self.hdawg, 1, delay_tolerance=20e-9)  # triggers readout card
        self.coil = awg_channel.awg_channel(self.hdawg, 4)  # coil control !!!!!!!!!!!!!!!!!!!!!!!
        # ro_trg.mode = 'set_delay' #M3202A
        # ro_trg.delay_setter = lambda x: adc.set_trigger_delay(int(x*adc.get_clock()/iq_ex.get_clock()-readout_trigger_delay)) #M3202A
        self.ro_trg.mode = 'waveform'  # AWG5014C

        self.adc.set_nop(self.pulsed_settings['adc_nop'])
        self.adc.set_nums(self.pulsed_settings['adc_nums'])

    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch.do_get_switch(channel=channel) != value:
            self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        # промежуточные частоты для гетеродинной схемы new:
        self.iq_devices = {#'iq_ex1': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 2, 3, self.lo1, exdir_db=exdir_db),
                           # M3202A
                           #'iq_ex2': awg_iq_multi.Awg_iq_multi(self.hdawg, self.dummy_awg, 5, 0, self.lo1, exdir_db=exdir_db),
                           #'iq_ex3': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 6, 7, self.lo1, exdir_db=exdir_db),
                           # M3202A
                           'iq_ro': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 0, 1, self.pna, exdir_db=exdir_db)}  # M3202A
        # iq_pa = awg_iq_multi.Awg_iq_multi(awg_tek, awg_tek, 3, 4, lo_ro) #M3202A
        #self.iq_devices['iq_ex1'].name = 'ex1'
        #self.iq_devices['iq_ex2'].name = 'ex2'
        #self.iq_devices['iq_ex3'].name = 'ex3'
        # iq_pa.name='pa'
        self.iq_devices['iq_ro'].name = 'ro'

        #self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.set_switch_if_not_set(1, channel=1)
        #self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: self.set_switch_if_not_set(2, channel=1)
        #self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: self.set_switch_if_not_set(3, channel=1)
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: None #self.set_switch_if_not_set(4, channel=1)

        #self.iq_devices['iq_ex1'].sa = self.sa
        #self.iq_devices['iq_ex2'].sa = self.sa
        #self.iq_devices['iq_ex3'].sa = self.sa
        self.iq_devices['iq_ro'].sa = self.sa

        self.fast_controls = {'coil': awg_channel.awg_channel(self.hdawg, 4)}  # coil control

    def get_readout_trigger_pulse_length(self):
        return self.pulsed_settings['trigger_readout_length']

    def get_modem_dc_calibration_amplitude(self):
        return self.pulsed_settings['modem_dc_calibration_amplitude']

    def revert_setup(self, old_settings):
        if 'adc_nums' in old_settings:
            self.adc.set_nums(old_settings['adc_nums'])
        if 'adc_nop' in old_settings:
            self.adc.set_nop(old_settings['adc_nop'])
        if 'adc_posttrigger' in old_settings:
            self.adc.set_posttrigger(old_settings['adc_posttrigger'])
