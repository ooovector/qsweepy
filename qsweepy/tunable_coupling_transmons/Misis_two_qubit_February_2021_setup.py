import qsweepy.libraries.instruments as instruments
import qsweepy
from qsweepy.libraries.awg_channel import awg_channel
import numpy as np

device_settings = {#'vna_address': 'TCPIP0::10.20.61.157::inst0::INSTR',
                   'vna_address': 'TCPIP0::10.20.61.68::inst0::INSTR', #ZVB
                   #'vna_address': 'TCPIP0::10.20.61.48::inst0::INSTR',
                   #'lo1_address': 'TCPIP0::10.20.61.59::inst0::INSTR',
                   #'lo1_timeout': 5000,
                   'rf_switch_address': '10.20.61.91',
                   'use_rf_switch': False,
                   'pxi_chassis_id': 0,
                   'hdawg_address': 'hdawg-dev8108',
                   'uhfqa_address': 'uhf-dev2491',
                   'sa_address': 'TCPIP0::10.20.61.56::inst0::INSTR',
                   'adc_timeout': 10,
                   'adc_trig_rep_period': 200 * 125,  # 10 kHz rate period
                   'adc_trig_width': 2,  # 32 ns trigger length

                   }

cw_settings = {'mixer_thru':0.5}

pulsed_settings = {'lo1_power': 18,
                   'vna_power': 16,
                   'ex_clock': 2400e6,  # 1 GHz - clocks of some devices
                   'ro_clock': 1000e6,
                   'rep_rate': 5e3,  # 10 kHz - pulse sequence repetition rate
                   # 500 ex_clocks - all waves is shorten by this amount of clock cycles
                   # to verify that M3202 will not miss next trigger
                   # (awgs are always missing trigger while they are still outputting waveform)
                   'global_num_points_delta': 20000,
                   'hdawg_ch0_amplitude': 0.8,
                   'hdawg_ch1_amplitude': 0.8,
                   'hdawg_ch2_amplitude': 0.8,
                   'hdawg_ch3_amplitude': 0.8,
                   'hdawg_ch4_amplitude': 0.7,
                   'hdawg_ch5_amplitude': 0.7,
                   'hdawg_ch6_amplitude': 0.8,
                   'hdawg_ch7_amplitude': 0.8,
                   'uhfqa_ch0_amplitude': 0.7,
                   'uhfqa_ch1_amplitude': 0.7,
                   'lo1_freq': 3.70e9,
                   'pna_freq': 7.20e9, #
                   #'calibrate_delay_nop': 65536,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_channel_name': 'ro_trg',
                   'trigger_readout_length': 200e-9,
                   'modem_dc_calibration_amplitude': 1.0,
                   'adc_nop': 1024,
                   'adc_nums': 10000, ##10000 Do we need control over this? Probably, but not now... WUT THE FUCK MAN
                   #'adc_default_delay': 550,
                   }


class hardware_setup():
    def __init__(self, device_settings, pulsed_settings):
        self.device_settings = device_settings
        self.pulsed_settings = pulsed_settings
        self.cw_settings = cw_settings
        self.hardware_state = 'undefined'
        self.sa = None

        self.pna = None
        self.lo1 = None
        self.rf_switch = None
        self.coil_device = None
        self.hdawg = None
        self.adc_device = None
        self.adc = None

        self.ro_trg = None
        self.q1z = None
        self.cz = None
        self.q2z = None
        self.q3z = None
        self.iq_devices = None

    def open_devices(self):
        # RF switch for making sure we know what sample we are measuring
        self.pna = instruments.RS_ZVB20('pna', address=self.device_settings['vna_address'])
        #self.pna = instruments.Agilent_N5242A('pna', address=self.device_settings['vna_address'])
        #self.lo1 = Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])

        #self.lo1._visainstrument.timeout = self.device_settings['lo1_timeout']
        #self.lo1 = instruments.SignalCore_5502a()
        #self.lo1.search()
        #self.lo1.open()

        if self.device_settings['use_rf_switch']:
            self.rf_switch = instruments.nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])

        #self.hdawg = instruments.ZIDevice(self.device_settings['hdawg_address'], devtype='HDAWG', clock=2e9, delay_int=0)
        self.hdawg = instruments.zihdawg(self.device_settings['hdawg_address'], devtype='HDAWG', clock=2e9,
                                          delay_int=0)
        #self.uhfqa = instruments.ziUHF(2, delay_int=0)
        #self.uhfqa.daq.setInt('/' + self.uhfqa.device + '/sigins/0/ac', 1)
        #self.uhfqa.daq.setInt('/' + self.uhfqa.device + '/sigins/1/ac', 1)
        #self.uhfqa.daq.setInt('/' + self.uhfqa.device + '/sigins/0/imp50', 1)
        #self.uhfqa.daq.setInt('/' + self.uhfqa.device + '/sigins/1/imp50', 1)
        #self.uhfqa.daq.setDouble('/' + self.uhfqa.device + '/sigins/0/range', 0.2)
        #self.uhfqa.daq.setDouble('/' + self.uhfqa.device + '/sigins/1/range', 0.2)

        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/0/range', 0.8)
        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/1/range', 0.8)
        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/2/range', 0.8)
        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/3/range', 0.8)
        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/4/range', 0.8)
        self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/5/range', 0.8)
        #self.hdawg.daq.setDouble('/' + self.hdawg.device + '/sigouts/6/range', 0.2)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/0/direct', 1)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/1/direct', 1)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/2/direct', 1)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/3/direct', 1)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/4/direct', 0)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/5/direct', 0)
        #self.hdawg.daq.setInt('/' + self.hdawg.device + '/sigouts/6/direct', 0)

        #self.uhfqa = instruments.ZIDevice(self.device_settings['uhfqa_address'], devtype='UHF')
        self.coil_device = self.hdawg

        #self.q1x = awg_channel(self.hdawg, 4)
        #self.q2x = awg_channel(self.hdawg, 5)
        #self.q3x = awg_channel(self.hdawg, 6)

        self.q3z = awg_channel(self.hdawg, 6)  # coil control
        self.q2z = awg_channel(self.hdawg, 2)  # coil control
        self.cz = awg_channel(self.hdawg, 7)  # coil control
        self.q1z = awg_channel(self.hdawg, 0)  # coil control


        self.sa = instruments.Agilent_N9030A('pxa', address=self.device_settings['sa_address'])

        self.adc_device = instruments.TSW14J56_evm()
        self.adc_device.timeout = self.device_settings['adc_timeout']
        self.adc = instruments.TSW14J56_evm_reducer(self.adc_device)
        self.adc.output_raw = True
        self.adc.last_cov = False
        self.adc.avg_cov = False
        self.adc.resultnumber = False

        self.adc_device.set_trig_src_period(self.device_settings['adc_trig_rep_period'])  # 10 kHz period rate
        self.adc_device.set_trig_src_width(self.device_settings['adc_trig_width'])  # 80 ns trigger length

        #self.adc = self.uhfqa

        self.hardware_state = 'undefined'

    def set_cw_mode(self, channels_off=None):
        if self.hardware_state == 'cw_mode':
            return
        self.hardware_state = 'cw_mode'
        self.hdawg.stop()

        global_num_points = int(np.round(
            self.pulsed_settings['ex_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
                'global_num_points_delta']))

        for channel in range(4, 6):
            self.hdawg.set_amplitude(amplitude=0.05, channel=channel)
            self.hdawg.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            self.hdawg.set_output(output=1, channel=channel)
            #self.hdawg.set_waveform(waveform=[0] * global_num_points, channel=channel)

        #for channel in range(0, 2):
            #self.uhfqa.set_amplitude(amplitude=0.05, channel=channel)
            #self.uhfqa.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            #self.uhfqa.set_output(output=1, channel=channel)
            #self.uhfqa.set_waveform(waveform=[0] * global_num_points, channel=channel)

        self.hdawg.set_output(output=1, channel=0)
        self.hdawg.set_output(output=1, channel=1)
        self.hdawg.set_output(output=1, channel=2)
        self.hdawg.set_output(output=1, channel=3)
        self.hdawg.set_output(output=1, channel=4)
        self.hdawg.set_output(output=1, channel=5)
        self.hdawg.set_output(output=0, channel=6)
        self.hdawg.set_output(output=0, channel=7)
        if channels_off is not None:
            for channel_off in channels_off:
                self.hdawg.set_output(output=0, channel=channel_off)

        self.pna.set_sweep_mode("LIN")
        self.hardware_state = 'cw_mode'

    def set_pulsed_mode(self):
        if self.hardware_state == 'pulsed_mode':
            return
        self.hardware_state = 'undefined'

        #self.lo1.set_status(1)  # turn on lo1 output
        #self.lo1.set_power(self.pulsed_settings['lo1_power'])
        #self.lo1.set_frequency(self.pulsed_settings['lo1_freq'])

        self.pna.set_power(self.pulsed_settings['vna_power'])
        #self.pna.write("OUTP ON")
        self.pna.write("SOUR1:POW1:MODE ON")
        self.pna.write("SOUR1:POW2:MODE OFF")
        self.pna.set_sweep_mode("CW") # privet RS ZVB20
        #self.pna.set_trigger_source("ON")
        self.pna.set_frequency(self.pulsed_settings['pna_freq'])

        self.hdawg.stop()

        self.hdawg.set_clock(self.pulsed_settings['ex_clock'])
        self.hdawg.set_clock_source(0)
        #self.uhfqa.set_clock(self.pulsed_settings['ro_clock'])
        #self.uhfqa.set_clock_source(1)

        # setting repetition period for slave devices
        # 'global_num_points_delay' is needed to verify that M3202A and other slave devices will be free
        # when next trigger arrives.
        global_num_points = int(np.round(
            self.pulsed_settings['ex_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
                'global_num_points_delta']))
        #global_num_points_ro = int(np.round(
        #    self.pulsed_settings['ro_clock'] / self.pulsed_settings['rep_rate'] - self.pulsed_settings[
        #        'global_num_points_delta']))

        # global_num_points = 20000

        self.hdawg.set_nop(global_num_points)
        self.hdawg.clear()

        #self.uhfqa.set_nop(global_num_points_ro)
        #self.uhfqa.clear()

        # а вот длину сэмплов, которая очевидно то же самое, нужно задавать на всех авгшках.
        # хорошо, что сейчас она только одна.
        # this is zashkvar   WUT THE FUCK MAN

        self.hdawg.set_trigger_impedance_1e3()
        self.hdawg.set_dig_trig1_source([0, 0, 0, 0])
        self.hdawg.set_dig_trig1_slope([0, 0, 0, 0])  # 0 - Level sensitive trigger, 1 - Rising edge trigger,
                                                      # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
        self.hdawg.set_dig_trig2_source([0, 0, 0, 0])
        self.hdawg.set_dig_trig2_slope([0, 0, 0, 0])
        self.hdawg.set_trig_level(0.6)

        #self.uhfqa.trigger_channel0_dir = 1
        #self.uhfqa.trigger_channel1_dir = 0

        #self.uhfqa.set_dig_trig1_source([4])
        #self.uhfqa.set_dig_trig2_source([1])
        #self.uhfqa.default_delay = pulsed_settings['adc_default_delay']

        for sequencer in range(4):
            self.hdawg.send_cur_prog(sequencer=sequencer)
            self.hdawg.set_marker_out(channel=int(2 * sequencer), source=4)  # set marker 1 to awg mark out 1 for sequencer
            #self.hdawg.set_marker_out(channel=int(2 * sequencer + 1),
                                      #source=7)  # set marker 2 to awg mark out 2 for sequencer
        #self.uhfqa.send_cur_prog(sequencer=0)
        #self.uhfqa.set_marker_out(channel=0, source=32)  # set marker 1 to awg mark out 1 for sequencer
        #self.uhfqa.set_marker_out(channel=1, source=33)  # set marker 2 to awg mark out 2 for sequencer
        for channel in range(8):
            self.hdawg.set_amplitude(channel=channel, amplitude=self.pulsed_settings['hdawg_ch%d_amplitude'%channel])
            self.hdawg.set_offset(channel=channel, offset=0 * 1.0)
            self.hdawg.set_digital(channel=channel, marker=[0]*(global_num_points))
            #self.hdawg.set_output(channel=channel, output=1)
            self.hdawg.daq.set([['/{}/sigouts/{}/range'.format(self.hdawg.device, channel), 1]])
            #self.hdawg.daq.set([['/{}/awgs/{}/outputs/{}/gains/{}'.format(self.hdawg.device, channel//2,
                                                                         #channel%2, channel%2), 1]])
        self.hdawg.set_all_outs()
        self.hdawg.run()

        #for channel in range(2):
            #self.uhfqa.set_amplitude(channel=channel, amplitude=self.pulsed_settings['uhfqa_ch%d_amplitude'%channel])
            #self.uhfqa.set_offset(channel=channel, offset=0 * 1.0)
            #self.uhfqa.set_digital(channel=channel, marker=[0]*(global_num_points))
            #self.uhfqa.daq.set([['/{}/sigouts/{}/range'.format(self.uhfqa.device, channel), 1]])
        #self.uhfqa.set_all_outs()
        #self.uhfqa.run()

        #self.ro_trg = qsweepy.libraries.awg_digital.awg_digital(self.hdawg, 0, delay_tolerance=50e-9)  # triggers readout card
        #self.ro_trg.adc = self.adc
        #self.ro_trg.mode = 'internal_delay'
        # ro_trg.mode = 'set_delay' #M3202A
        # ro_trg.delay_setter = lambda x: adc.set_trigger_delay(int(x*adc.get_clock()/iq_ex.get_clock()-readout_trigger_delay)) #M3202A
        # self.ro_trg.mode = 'waveform'  # AWG5014C
        self.ro_trg = qsweepy.libraries.awg_digital.awg_digital(self.hdawg, 0, delay_tolerance=20e-9)  # triggers readout card
        self.ro_trg.adc = self.adc
        self.ro_trg.mode = 'waveform'
        #
        self.adc.set_adc_nop(self.pulsed_settings['adc_nop'])
        self.adc.set_adc_nums(self.pulsed_settings['adc_nums'])

        self.hardware_state = 'pulsed_mode'

    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch is not None:
            if self.rf_switch.do_get_switch(channel=channel) != value:
                self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        # промежуточные частоты для гетеродинной схемы new:
        self.iq_devices = {#'iq_ex1': qsweepy.libraries.awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 0, 1, self.lo1, exdir_db=exdir_db),
                           #'iq_ex2': qsweepy.libraries.awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 2, 3, self.lo1, exdir_db=exdir_db), #M3202A
                           #'iq_ex3': qsweepy.libraries.awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 4, 5, self.lo1, exdir_db=exdir_db),
                           # M3202A
                           'iq_ro':  qsweepy.libraries.awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 4, 5, self.pna, exdir_db=exdir_db)
                           }  # M3202A
        # iq_pa = awg_iq_multi.Awg_iq_multi(awg_tek, awg_tek, 3, 4, lo_ro) #M3202A
        #self.iq_devices['iq_ex1'].name = 'ex1'
        #self.iq_devices['iq_ex2'].name = 'ex2'
        #self.iq_devices['iq_ex3'].name = 'ex3'
        # iq_pa.name='pa'
        self.iq_devices['iq_ro'].name = 'ro'

        #self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.set_switch_if_not_set(1, channel=1)
        #self.iq_devices['iq_ex2'].calibration_switch_setter = lambda: self.set_switch_if_not_set(2, channel=1)
        # iq_ex2.calibration_switch_setter = lambda: self.rf_switch.do_set_switch(2, channel=1) if not self.rf_switch.do_get_switch(channel=1)==2 else None
        # self.iq_devices['iq_ex3'].calibration_switch_setter = lambda: self.set_switch_if_not_set(3, channel=1)
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: self.set_switch_if_not_set(4, channel=1)

        #self.iq_devices['iq_ex1'].sa = self.sa
        #self.iq_devices['iq_ex2'].sa = self.sa
        # self.iq_devices['iq_ex3'].sa = self.sa
        self.iq_devices['iq_ro'].sa = self.sa

        self.fast_controls = {
                              #'q1x': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 4),
                              #'q2x': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 5),
                              #'q3x': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 6),

                              'q3z': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 6),
                              'q2z': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 2),
                              'cz': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 7),
                              'q1z': qsweepy.libraries.awg_channel.awg_channel(self.hdawg, 0)}  # coil control

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
