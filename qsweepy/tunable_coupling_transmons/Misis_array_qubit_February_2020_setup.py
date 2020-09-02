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
                   'sa_address': 'TCPIP0::10.20.61.56::inst0::INSTR',
                   'adc_timeout': 15,
                   'adc_trig_rep_period': 50 * 125,  # 20 kHz rate period
                   'adc_trig_width': 2,  # 80 ns trigger length
                   }

cw_settings = {'mixer_thru':0.3}
pulsed_settings = {'lo1_power': 18,
                   'vna_power': 10,
                   'ex_clock': 1000e6,  # 1 GHz - clocks of some devices
                   'rep_rate': 20e3,  # 10 kHz - pulse sequence repetition rate
                   # 500 ex_clocks - all waves is shorten by this amount of clock cycles
                   # to verify that M3202 will not miss next trigger
                   # (awgs are always missing trigger while they are still outputting waveform)
                   'global_num_points_delta': 400,
                   'hdawg_ch0_amplitude': 1.0,
                   'hdawg_ch1_amplitude': 0.8,
                   'hdawg_ch2_amplitude': 0.8,
                   'hdawg_ch3_amplitude': 0.8,
                   'hdawg_ch4_amplitude': 0.9,
                   'hdawg_ch5_amplitude': 0.8,
                   'hdawg_ch6_amplitude': 0.5,
                   'hdawg_ch7_amplitude': 0.5,
                   'lo1_freq': 4.7e9,
                   'pna_freq': 5.83e9,
                   #'calibrate_delay_nop': 65536,
                   'calibrate_delay_nums': 200,
                   'trigger_readout_channel_name': 'ro_trg',
                   'trigger_readout_length': 20e-9,
                   'modem_dc_calibration_amplitude': 1.0,
                   'adc_nop': 1024,
                   'adc_nums': 25000,  ## Do we need control over this? Probably, but not now... WUT THE FUCK MAN
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
        self.set_zero = None

    def open_devices(self):
        # RF switch for making sure we know what sample we are measuring
        self.pna = RS_ZNB20('pna', address=self.device_settings['vna_address'])
        self.lo1 = Agilent_E8257D('lo1', address=self.device_settings['lo1_address'])

        self.lo1._visainstrument.timeout = self.device_settings['lo1_timeout']

        if self.device_settings['use_rf_switch']:
            self.rf_switch = nn_rf_switch('rf_switch', address=self.device_settings['rf_switch_address'])

        self.sa = Agilent_N9030A('pxa', address=self.device_settings['sa_address'])

        self.hdawg = Zurich_HDAWG1808(self.device_settings['hdawg_address'])
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

        self.dac = nndac('TCPIP0::10.20.61.12::1000::SOCKET')
        def set_zero():
            for nndac_channel in range(24):
                self.dac.set_voltage(0, channel=nndac_channel)
        self.set_zero = set_zero

        self.mcfq = MultiChannelFrequencyControl(hardware=self)
        self.controls = {}
        for qubit_id in ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                         '11', '12', '13', '14', '15', '16', '18', '19',
                         '20', '21', '22', '23', '24', '25']:
            self.controls['fq' + qubit_id] = FrequencyControl(qubit_id, self.mcfq)

        self.hardware_state = 'undefined'

    def set_cw_mode(self):
        self.hdawg.stop()

        for channel in range(2, 4):
            self.hdawg.set_amplitude(amplitude=0.05, channel=channel)
            self.hdawg.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            self.hdawg.set_output(output=1, channel=channel)
            self.hdawg.set_waveform(waveform=[0] * 99500, channel=channel)

        for channel in range(6, 8):
            self.hdawg.set_amplitude(amplitude=0.05, channel=channel)
            self.hdawg.set_offset(offset=self.cw_settings['mixer_thru'], channel=channel)
            self.hdawg.set_output(output=1, channel=channel)
            self.hdawg.set_waveform(waveform=[0] * 99500, channel=channel)
        self.pna.set_sweep_mode("LIN")
        self.hardware_state = 'cw_mode'

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
            self.hdawg.daq.set([['/{}/sigouts/{}/range'.format(self.hdawg.device, channel), 1]])
        self.hdawg.daq.set([['/{}/sigouts/4/range'.format(self.hdawg.device), 2]])
        self.hdawg.set_all_outs()
        self.hdawg.run()

        self.ro_trg = awg_digital.awg_digital(self.hdawg, 1, delay_tolerance=20e-9)  # triggers readout card
        self.coil = awg_channel.awg_channel(self.hdawg, 0)  # coil control
        # ro_trg.mode = 'set_delay' #M3202A
        # ro_trg.delay_setter = lambda x: adc.set_trigger_delay(int(x*adc.get_clock()/iq_ex.get_clock()-readout_trigger_delay)) #M3202A
        self.ro_trg.mode = 'waveform'  # AWG5014C

        self.adc.set_nop(self.pulsed_settings['adc_nop'])
        self.adc.set_nums(self.pulsed_settings['adc_nums'])

        self.hardware_state = 'pulsed_mode'

    def set_switch_if_not_set(self, value, channel):
        if self.rf_switch.do_get_switch(channel=channel) != value:
            self.rf_switch.do_set_switch(value, channel=channel)

    def setup_iq_channel_connections(self, exdir_db):
        # промежуточные частоты для гетеродинной схемы new:
        self.iq_devices = {'iq_ex1': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 2, 3, self.lo1, exdir_db=exdir_db),
                           'iq_ro': awg_iq_multi.Awg_iq_multi(self.hdawg, self.hdawg, 6, 7, self.pna, exdir_db=exdir_db)}  # M3202A

        self.iq_devices['iq_ex1'].name = 'ex1'
        self.iq_devices['iq_ro'].name = 'ro'

        self.iq_devices['iq_ex1'].calibration_switch_setter = lambda: self.set_switch_if_not_set(2, channel=1)
        self.iq_devices['iq_ro'].calibration_switch_setter = lambda: self.set_switch_if_not_set(4, channel=1)

        self.iq_devices['iq_ex1'].sa = self.sa
        self.iq_devices['iq_ro'].sa = self.sa

        self.fast_controls = {'coil': awg_channel.awg_channel(self.hdawg, 0)}  # coil control

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


class FrequencyControl:
    def __init__(self, qubit_id, multi_channel):
        self.qubit_id = qubit_id
        self.multi_channel = multi_channel

    def get_offset(self):
        return self.multi_channel.get_qubit_frequency(qubit_id=self.qubit_id)

    def set_offset(self, offset):
        self.multi_channel.set_qubit_frequency(offset, qubit_id=self.qubit_id)
        pass


class MultiChannelFrequencyControl:
    def __init__(self, hardware):
        self.state = {}
        self.hardware = hardware

    def set_exdir_db(self, exdir_db):
        self.exdir_db = exdir_db

    def set_all_frequencies(self, f):
        ind_mat_meas = self.exdir_db.select_measurement(measurement_type='Ind_mat_REC23')
        qubit_ids = [str(q) for q in ind_mat_meas.datasets['L_mat'].parameters[0].values]

        self.state = {qubit_id: f for qubit_id in qubit_ids}
        self.update_qubit_frequencies()

    def update_qubit_frequencies(self):
        # find voltages
        voltages = self.freqs2volts(freqs=self.state)
        # set voltages
        for coil_id, voltage in voltages.items():
            self.set_coil_voltage(voltage=voltage, coil_id=coil_id)

        for coil_id in range(24):
            if coil_id not in voltages.keys():
                self.set_coil_voltage(voltage=0, coil_id=coil_id)

    def set_qubit_frequency(self, frequency, qubit_id):
        self.state[qubit_id] = frequency
        self.update_qubit_frequencies()

    def get_qubit_frequency(self, qubit_id):
        return self.state[qubit_id]

    def set_coil_voltage(self, voltage, coil_id):
        if coil_id == 14:
            self.hardware.hdawg.set_offset(voltage, channel=0)
            self.hardware.hdawg.set_output(1, channel=0)
        else:
            self.hardware.dac.set_voltage(voltage, channel=coil_id)

    def get_coil_voltage(self, coil_id):
        if coil_id == 14:
            return self.hardware.awg.get_offset(channel=0)
        else:
            return self.hardware.dac.get_voltage(coil_id)

    # helper functions for freqs2volts
    def fr_coil(self, p, x):
        frb, Cc, EJ1, EJ2, EC, phi0 = p[:6]
        L = p[6:]
        return self.fR_1Q(x, frb, Cc, EJ1, EJ2, EC, phi0, L) / 1e9

    def fR_1Q(self, x, frb, Cc, EJ1, EJ2, EC, phi0, L):
        f_qubit = self.fQ(x, EJ1, EJ2, EC, phi0, L)
        f_QR = (f_qubit + frb) * 0.5 - (((f_qubit - frb) * 0.5) ** 2 + Cc ** 2 * f_qubit * frb) ** 0.5 * np.sign(
            -frb + f_qubit)
        return f_QR

    def fQ(self, x, EJ1, EJ2, EC, phi0, L):
        # f_qubit=fqb(x, EJ1, EJ2, EC, phi0, L)
        f_qubit = (8 * EC) ** 0.5 * ((EJ1 - EJ2) ** 2 * np.sin(np.pi * x * L + phi0 * np.pi) ** 2 +
                                     (EJ1 + EJ2) ** 2 * np.cos(np.pi * x * L + phi0 * np.pi) ** 2) ** 0.25
        return f_qubit

    def fq_1R(self, x, frb, Cc, EJ1, EJ2, EC, phi0, L):
        f_qubit = self.fQ(x, EJ1, EJ2, EC, phi0, L)
        f_QR = (f_qubit + frb) * 0.5 - (((f_qubit - frb) * 0.5) ** 2 + Cc ** 2 * f_qubit * frb) ** 0.5 * np.sign(
            frb - f_qubit)
        return f_QR

    def fQ_2R_coil(self, p, x):
        frb, Cc, EJ1, EJ2, EC, phi0 = p[:6]
        L = p[6]
        fr_com = p[7]
        Ccom = p[8]
        if frb < fr_com:
            fq1R = self.fq_1R(x, frb, Cc, EJ1, EJ2, EC, phi0, L)
            fq2R = (fq1R + fr_com) * 0.5 - (((fq1R - fr_com) * 0.5) ** 2 + Ccom ** 2 * fq1R * fr_com) ** 0.5 * np.sign(
                fr_com - fq1R)
        else:
            fq1R = self.fq_1R(x, fr_com, Ccom, EJ1, EJ2, EC, phi0, L)
            fq2R = (fq1R + frb) * 0.5 - (((fq1R - frb) * 0.5) ** 2 + Cc ** 2 * fq1R * frb) ** 0.5 * np.sign(frb - fq1R)

        return fq2R

    def freqs2volts(self, freqs, Podgon=True):
        '''
        :param freqs dict: qubit-frequency pairs
        :param Podgon bool:
        :return:
        '''
        from scipy.optimize import root

        ind_mat_meas = self.exdir_db.select_measurement(measurement_type='Ind_mat_REC23')
        qubit_ids = [str(q) for q in ind_mat_meas.datasets['L_mat'].parameters[0].values]
        coil_ids = ind_mat_meas.datasets['L_mat'].parameters[1].values
        ind_mat = ind_mat_meas.datasets['L_mat'].data

        if Podgon == True:
            linear_fit = self.exdir_db.select_measurement(measurement_type='Podgon_param', )
            b = 1e9* np.asarray(linear_fit.datasets['K'].parameters[1].values)  # TODO fix the data structure, it is invalid  !!!!!! 1e9 tak tak teperi vse v Hz a ne v GHz
            k = np.asarray(linear_fit.datasets['K'].data)
            qubit_ids_linear_fit = [str(q) for q in linear_fit.datasets['K'].parameters[0].values]
            freqs_corrected = {qubit_id: (freqs[qubit_id] - b[_id]) / k[_id]
                               for _id, qubit_id in enumerate(qubit_ids_linear_fit)}
        else:
            freqs_corrected = freqs

        phi_vector = np.zeros(len(qubit_ids))
        for _id, qubit_id in enumerate(qubit_ids):
            target_freq = freqs_corrected[qubit_id]

            phi_meas = self.exdir_db.select_measurement(measurement_type='Full_two_tone_fit_Recool',
                                                        metadata={'qubit_id': qubit_id, })
            p0 = [float(phi_meas.metadata['frb']), float(phi_meas.metadata['Cc']),
                  float(phi_meas.metadata['EJ1']), float(phi_meas.metadata['EJ2']),
                  float(phi_meas.metadata['EC']), float(phi_meas.metadata['phi0']), float(phi_meas.metadata['L']),
                  float(phi_meas.metadata['fr_com']), float(phi_meas.metadata['Ccom']), ]
            solutions = []
            for x0 in np.linspace(-0.5, 0.5, 51) / float(phi_meas.metadata['L']):
                v = root(lambda x: self.fQ_2R_coil(p0, (x)) /1e9- target_freq/1e9, x0)  # delim na 1e9 chtoby emy normalinye chisla vychitati a ne milliardy
                solutions.append(v.x * float(phi_meas.metadata['L']))  # /(np.pi)
            phi_vector[_id] = solutions[np.argmin(np.abs(solutions))]

        voltage_vector = np.linalg.solve(ind_mat, phi_vector)
        voltage_dict = {coil: voltage for coil, voltage in zip(coil_ids, voltage_vector)}
        return voltage_dict

    def get_qubit_freq_min(self, qubit_id):
        source = self.exdir_db.select_measurement(measurement_type='Full_two_tone_fit_Recool',
                                                      metadata={'qubit_id': qubit_id, })
        p0 = [float(source.metadata['frb']), float(source.metadata['Cc']), \
                  float(source.metadata['EJ1']), float(source.metadata['EJ2']), \
                  float(source.metadata['EC']), float(source.metadata['phi0']), float(source.metadata['L']), \
                  float(source.metadata['fr_com']), float(source.metadata['Ccom']), ]
        x0 = (0.5 - float(source.metadata['phi0'])) / float(source.metadata['L'])
        fmin = self.fQ_2R_coil(p0, (x0))
        return fmin