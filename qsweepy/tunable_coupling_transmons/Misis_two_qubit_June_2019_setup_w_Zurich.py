from qsweepy.instruments import *
from qsweepy import *

from qsweepy import awg_iq_multi

import numpy as np

device_settings = {'vna_address': 'TCPIP0::10.20.61.48::inst0::INSTR',
                   'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
                   'lo1_timeout': 5000, 'rf_switch_address': '10.20.61.224',
                   'use_rf_switch': True,
                   'pxi_chassis_id': 0,
                   'hdawg_address': 'hdawg-dev8108',
                   'awg_tek_address': 'TCPIP0::10.20.61.186::inst0::INSTR',
                   'use_awg_tek': True,
                   'sa_address': 'TCPIP0::10.20.61.56::inst0::INSTR',
                   'adc_timeout': 10,
                   'adc_trig_rep_period': 50 * 125,  # 10 kHz rate period
                   'adc_trig_width': 2,  # 80 ns trigger length
                   }

cw_settings = {}
pulsed_settings = {'lo1_power': 18,
                   'vna_power': 16,
                   'ex_clock': 1000e6,  # 1 GHz - clocks of some devices
                   'rep_rate': 20e3,  # 10 kHz - pulse sequence repetition rate
                   # 500 ex_clocks - all waves is shorten by this amount of clock cycles
                   # to verify that M3202 will not miss next trigger
                   # (awgs are always missing trigger while they are still outputting waveform)
                   'global_num_points_delta': 500,
                   'hdawg_ch0_amplitude': 0.3,
                   'hdawg_ch1_amplitude': 0.3,
                   'hdawg_ch2_amplitude': 0.5,
                   'hdawg_ch3_amplitude': 0.5,
                   'hdawg_ch4_amplitude': 0.5,
                   'hdawg_ch5_amplitude': 0.5,
                   'hdawg_ch6_amplitude': 0.5,
                   'hdawg_ch7_amplitude': 0.5,
                   'awg_tek_ch1_amplitude': 1.0,
                   'awg_tek_ch2_amplitude': 1.0,
                   'awg_tek_ch3_amplitude': 1.0,
                   'awg_tek_ch4_amplitude': 1.0,
                   'awg_tek_ch1_offset': 0.0,
                   'awg_tek_ch2_offset': 0.0,
                   'awg_tek_ch3_offset': 0.0,
                   'awg_tek_ch4_offset': 0.0,
                   'lo1_freq': 3.41e9,
                   'pna_freq': 6.07e9,
                   'calibrate_delay_nop': 65536,
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
        self.awg_tek = None
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
        self.lo1 = Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])

        self.lo1._visainstrument.timeout = self.device_settings['lo1_timeout']

        if self.device_settings['use_rf_switch']:
            self.rf_switch = nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])

        if self.device_settings['use_awg_tek']:
            self.awg_tek = Tektronix_AWG5014('awg_tek', address=self.device_settings['awg_tek_address'])
        self.sa = Agilent_N9030A('pxa', address=self.device_settings['sa_address'])

        self.coil_device = self.awg_tek

        self.hdawg = Zurich_HDAWG1808(self.device_settings['hdawg_address'])

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
        self.lo1.set_status(1)  # turn on lo1 output
        self.lo1.set_power(self.pulsed_settings['lo1_power'])
        self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.pna.set_power(self.pulsed_settings['vna_power'])
        self.pna.write("OUTP ON")
        self.pna.write("SOUR1:POW1:MODE ON")
        self.pna.write("SOUR1:POW2:MODE OFF")
        self.pna.set_sweep_mode("CW")
        self.pna.set_frequency(self.pulsed_settings['pna_freq'])

        self.hdawg.stop()
        self.awg_tek.stop()

        self.awg_tek.set_clock(self.pulsed_settings['ex_clock'])  # клок всех авгшк
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
            self.hdawg.set_digital(channel=channel, marker=[1]*1000 + [0]*(global_num_points-1000))
        self.hdawg.set_all_outs()
        self.hdawg.run()

        self.awg_tek._visainstrument.write('AWGC:RMOD TRIG')
        self.awg_tek._visainstrument.write('TRIG:WVAL LAST')
        self.awg_tek._visainstrument.write('TRIG:IMP 1kohm')
        self.awg_tek._visainstrument.write('TRIG:SLOP POS')
        self.awg_tek._visainstrument.write('TRIG:LEV 0.5')
        self.awg_tek._visainstrument.write('SOUR1:ROSC:FREQ 10MHz')
        self.awg_tek._visainstrument.write('SOUR1:ROSC:SOUR EXT')
        # awg_tek.set_trigger_mode('CONT')
        self.awg_tek.set_nop(global_num_points)  # репрейт нужно задавать по=хорошему только на управляющей_t
        self.awg_tek.check_cached = True

        for channel in range(1, 5):
            self.awg_tek.set_amplitude(self.pulsed_settings['awg_tek_ch{}_amplitude'.format(channel)], channel=channel)
            self.awg_tek.set_offset(self.pulsed_settings['awg_tek_ch{}_offset'.format(channel)], channel=channel)
            self.awg_tek.set_output(1, channel=channel)
            self.awg_tek.set_waveform([0] * global_num_points, channel=channel)
        # awg_tek.set_amplitude(1.0,f channel=4)
        # awg_tek.set_amplitude(2.0, channel=3)
        self.awg_tek.run()

        self.awg_tek.set_digital([1] * 1000 + [0] * (global_num_points - 1000), channel=1)  # triggers PXI modules
        self.awg_tek.set_digital([1] * 1000 + [0] * (global_num_points - 1000), channel=2)  #
        self.awg_tek.set_digital([1] * 1000 + [0] * (global_num_points - 1000), channel=3)  #

        #	for other_channels in [3,4,5,6,7,8]:
        #		awg_tek.set_digital([1]*1000+[0]*(global_num_points+500-1210), channel=other_channels)

        self.awg_tek._visainstrument.write('SOUR1:DEL:ADJ 400 NS')
        self.awg_tek._visainstrument.write('SOUR2:DEL:ADJ 400 NS')
        self.awg_tek._visainstrument.write('SOUR3:DEL:ADJ 400 NS')
        self.awg_tek._visainstrument.write('SOUR4:DEL:ADJ 400 NS')
        # paramp pump power
        # awg_tek._visainstrument.write('SOUR4:MARK1:VOLT:HIGH 1.5')

        # self.hardware_state = 'pulsed'

        self.ro_trg = awg_digital.awg_digital(self.awg_tek, 3, delay_tolerance=20e-9)  # triggers readout card
        self.coil_multi = awg_channel.awg_channel(self.awg_tek, 4)  # coil control
        # ro_trg.mode = 'set_delay' #M3202A
        # ro_trg.delay_setter = lambda x: adc.set_trigger_delay(int(x*adc.get_clock()/iq_ex.get_clock()-readout_trigger_delay)) #M3202A
        self.ro_trg.mode = 'waveform'  # AWG5014C

        self.adc.set_nop(self.pulsed_settings['adc_nop'])
        self.adc.set_nums(self.pulsed_settings['adc_nums'])

    def setup_iq_channel_connections(self, exdir_db):
        # промежуточные частоты для гетеродинной схемы new:
        self.iq_devices = {'iq_ex1': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 2, 3, self.lo1, exdir_db=exdir_db),
                           # M3202A
                           # 'iq_ex2': hardware.iq_ex2 = awg_iq_multi.Awg_iq_multi(awg2, awg2, 2, 3, lo_ex), #M3202A
                           'iq_ex3': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 6, 7, self.lo1, exdir_db=exdir_db),
                           # M3202A
                           'iq_ro': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 0,            1, self.pna,
                                                              exdir_db=exdir_db)}  # M3202A
        # iq_pa = awg_iq_multi.Awg_iq_multi(awg_tek, awg_tek, 3, 4, lo_ro) #M3202A
        self.iq_devices['iq_ex1'].name = 'ex1'
        # iq_ex2.name='ex2'
        self.iq_devices['iq_ex3'].name = 'ex3'
        # iq_pa.name='pa'
        self.iq_devices['iq_ro'].name = 'ro'

        self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.rf_switch.do_set_switch(1,
                                                                                                   channel=1) if not self.rf_switch.do_get_switch(
            channel=1) == 1 else None
        # iq_ex2.calibration_switch_setter = lambda: self.rf_switch.do_set_switch(2, channel=1) if not self.rf_switch.do_get_switch(channel=1)==2 else None
        self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: self.rf_switch.do_set_switch(3,
                                                                                                   channel=1) if not self.rf_switch.do_get_switch(
            channel=1) == 3 else None
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: self.rf_switch.do_set_switch(4,
                                                                                                  channel=1) if not self.rf_switch.do_get_switch(
            channel=1) == 4 else None

        self.iq_devices['iq_ex1'].sa = self.sa
        self.iq_devices['iq_ex3'].sa = self.sa
        self.iq_devices['iq_ro'].sa = self.sa

        self.fast_controls = {'coil:': awg_channel.awg_channel(self.awg_tek, 4)}  # coil control

    def set_readout_delay_calibration_mode(self):
        old_settings = {'adc_nums': self.adc.get_nums(),
                        'adc_nop': self.adc.get_nop()}
        self.adc.set_nums(self.pulsed_settings['calibrate_delay_nums'])
        self.adc.set_nop(self.pulsed_settings['calibrate_delay_nop'])
        if hasattr(self.adc, 'set_posttrigger'):
            old_settings['adc_posttrigger'] = self.adc.get_posttrigger()
            self.adc.set_posttrigger(self.adc.get_nop() - 32)

        return old_settings

    def get_readout_trigger_pulse_length(self):
        return self.pulsed_settings['trigger_readout_length']

    def get_modem_dc_calibration_amplitude(self):
        return self.pulsed_settings['modem_dc_calibration_amplitude']

    # adc_reducers = {_qubit_id:data_reduce.data_reduce(device.modem) for  _qubit_id in qubits.keys()}
    # for _qubit_id in qubits.keys():
    #	adc_reducers[_qubit_id].filters = {'S21_r{}'.format(_qubit_id):data_reduce.mean_reducer(
    #		modem, 'iq_ro_q{}'.format(_qubit_id), axis=0)}
    #	adc_reducers[_qubit_id].extra_opts['scatter'] = True
    #	adc_reducers[_qubit_id].extra_opts['realimag'] = True

    def revert_setup(self, old_settings):
        if 'adc_nums' in old_settings:
            self.adc.set_nums(old_settings['adc_nums'])
        if 'adc_nop' in old_settings:
            self.adc.set_nop(old_settings['adc_nop'])
        if 'adc_posttrigger' in old_settings:
            self.adc.set_posttrigger(old_settings['adc_posttrigger'])
