
from . import sweep
# from . import save_pkl
from . import fitting
import numpy as np
from qsweepy import zi_scripts
from qsweepy.libraries import data_reduce

import matplotlib.pyplot as plt


# Several carriers for readout??
# data_reduce.data_reduce does all the stuff but still...
# This class doesn't know about the excitation tone and can be used for any systems
class modem_readout(data_reduce.data_reduce):
    ### Frequency multiplexing:
    ### Implements measure(), get_points(), get_opts() and  so on.
    ### 1) Calibrates f*** S+ and S- channels into I and Q by measureing how how in-phase and out-phase signals on DAC look on ADC and zero-level signal
    ### 2) Uses data_reduce.data_reduce for I, Q on carrier frequencies.
    ### 3) Uses data_reduce.data_reduce for downsampling.
    ### 4) Doesn't perform classification.
    ### 5)
    # trigger sequence:
    # pg.p('ro_trg', trg_length, pg.rect, 1),

    def __init__(self, pulse_sequencer, hardware, trigger_channel, src_meas='Voltage', axis_mean=0, exdir_db=None):
        self.pulse_sequencer = pulse_sequencer
        self.hardware = hardware
        self.adc = hardware.adc
        self.src_meas = src_meas
        self.axis_mean = axis_mean
        self.trigger_channel = trigger_channel
        self.exdir_db = exdir_db
        self.awg = hardware.iq_devices['iq_ro'].awg #self.awg = hardware.hdawg

        self.readout_channels = {}
        self.delay_calibrations = {}

        self.calibrations = {}
        self.iq_readout_calibrations = {}
        self.calibrated_filters = {}
        self.calibration_measurements = {}
        super().__init__(hardware.adc)  # modem_readout is a

    # random pulse sequence for wideband calibration of everything
    # shot noise with characterisitic timescale 30 time slots should be like 640K of RAM from microsoft
    def random_alignment_sequence(self, ex_channel, shot_noise_time=30):
        from scipy.signal import resample
        if not hasattr(self, 'sequence_length'):
            sequence_length = int(np.ceil(ex_channel.get_nop() / 2))
        else:
            sequence_length = self.sequence_length
        dac_sequence = []
        while len(dac_sequence) < sequence_length:
            dac_sequence.extend([1] * np.random.randint(shot_noise_time) + [-1] * np.random.randint(shot_noise_time))
        dac_sequence = np.asarray(dac_sequence[:sequence_length])
        dac_sequence_adc_time = resample(dac_sequence,
                                         int(np.ceil(sequence_length * self.adc.get_clock() / ex_channel.get_clock())))

        return dac_sequence, dac_sequence_adc_time

    def demodulation(self, ex_channel, sign=True, use_carrier=True):
        readout_time_axis = self.adc.get_points()[self.src_meas][1 - self.axis_mean][1]
        if hasattr(ex_channel, 'get_if') and use_carrier:  # has get_if method => works on carrier, need to demodulate
            demodulation = np.exp(
                1j * np.asarray(readout_time_axis) * 2 * np.pi * ex_channel.get_if() * (1 if sign else -1))
        else:
            demodulation = np.ones(len(readout_time_axis))  # otherwise demodulation with unity (multiply by one)
        return demodulation

    def calibrate_delay(self, ex_channel_name, save=True):
        from scipy.signal import correlate
        self.hardware.set_pulsed_mode()
        # delay is calibrated on all lines (we probably don't really need that, but whatever)
        # readout_delays = {}
        # for ex_channel_name, ex_channel in self.readout_channels.items():
        # delay calibration pulse sequence

        ex_channel = self.readout_channels[ex_channel_name]
        dac_sequence, dac_sequence_adc_time = self.random_alignment_sequence(ex_channel)

        sequence = zi_scripts.DMSequence(ex_channel.parent.sequencer_id, self.awg, len(dac_sequence))
        self.awg.set_sequence(ex_channel.parent.sequencer_id, sequence)
        sequence.stop()
        sequence.set_waveform(dac_sequence)
        if self.trigger_channel.delay>0:
            self.trigger_channel.delay=0
        sequence.set_delay(self.trigger_channel.delay)
        sequence.start()

        self.adc.output_raw = True
        meas_result = self.adc.measure()[self.src_meas]
        if meas_result.ndim == 1:  # in case of UHFQA data is already averaged on hardware
            adc_sequence = meas_result
        else:
            adc_sequence = np.mean(meas_result, axis=self.axis_mean)
        # adc_sequence = np.mean(self.adc.measure()[self.src_meas], axis=self.axis_mean)
        # demodulate
        demodulation = self.demodulation(ex_channel, sign=True, use_carrier=False)
        # depending on how the cables are plugged in, measure
        xc1 = correlate(np.real(adc_sequence) * demodulation, dac_sequence_adc_time, mode='full')  # HUYAGIC is here
        xc2 = correlate(np.imag(adc_sequence) * demodulation, dac_sequence_adc_time, mode='full')  # magic is here
        abs_xc = np.abs(xc1) + np.abs(xc2)
        # maximum correlation:
        readout_delay = -(np.argmax(abs_xc) - len(
            dac_sequence_adc_time) - 50) / self.adc.get_clock()  # get delay time in absolute units
        if readout_delay>0:
            readout_delay=0
        # plt.plot(xc1)
        # plt.plot(xc2)
        # plt.plot(xc3)
        # plt.plot(xc4)
        plt.plot(abs_xc)
        plt.figure()
        plt.plot(np.real(adc_sequence))
        plt.plot(np.imag(adc_sequence))
        plt.figure()
        if save:
            self.delay_calibrations[ex_channel_name] = readout_delay
            self.abs_xc = abs_xc
            self.xc_points = np.linspace(-len(dac_sequence_adc_time) + 1, len(adc_sequence) - 1,
                                         len(abs_xc)) / self.adc.get_clock()
            self.dac_sequence = dac_sequence
            self.adc_sequence = adc_sequence
            self.dac_sequence_adc_time = dac_sequence_adc_time
            self.ex_channel_clock = ex_channel.get_clock()
        return readout_delay

    def create_filters(self, ex_channel_name):
        ex_channel = self.readout_channels[ex_channel_name]
        # for ex_channel_name, ex_channel in self.readout_channels.items():
        demodulation = self.demodulation(ex_channel, sign=True)
        feature = demodulation * self.calibrations[ex_channel_name + '+'] + np.conj(demodulation) * self.calibrations[
            ex_channel_name + '-']
        self.iq_readout_calibrations[ex_channel_name] = {
            'iq_calibration': [self.calibrations[ex_channel_name + '+'], self.calibrations[ex_channel_name + '-']],
            'feature': feature}
        self.calibrated_filters[ex_channel_name] = data_reduce.feature_reducer(self.adc, self.src_meas,

                                                                               1 - self.axis_mean, self.bg, feature)

    def get_dc_bg_calibration(self):
        try:
            dc_bg_calibration_measurement = self.exdir_db.select_measurement(
                references_that={'delay_measurement': self.delay_measurement.id},
                measurement_type='modem_dc_bg_calibration')
            assert (len(dc_bg_calibration_measurement.datasets['bg'].data) == self.adc.get_nop())
            self.bg = dc_bg_calibration_measurement.datasets['bg'].data
        except Exception as e:
            print(str(e), type(e))
            self.calibrate_dc_bg()

    def get_dc_calibrations(self, amplitude=1.0, shot_noise_time=30):
        calibrations = {}
        for ex_channel_name, ex_channel in self.readout_channels.items():
            try:
                metadata = {'ex_channel': ex_channel_name}
                calibrations_measurement = self.exdir_db.select_measurement(measurement_type='modem_dc_iq_calibration',
                                                                            references_that={
                                                                                'delay_measurement': self.delay_measurement.id},
                                                                            metadata=metadata)
                self.calibrations[ex_channel_name + '+'] = np.complex(
                    calibrations_measurement.metadata[ex_channel_name + '+'])
                self.calibrations[ex_channel_name + '-'] = np.complex(
                    calibrations_measurement.metadata[ex_channel_name + '-'])
                self.calibration_measurements[ex_channel_name] = calibrations_measurement
                self.create_filters(ex_channel_name)
            except Exception as e:
                print(str(e), type(e))
                calibration = self.calibrate_dc(ex_channel_name=ex_channel_name, amplitude=amplitude,
                                                shot_noise_time=shot_noise_time)
                calibration.update(metadata)
                self.calibration_measurements[ex_channel_name] = self.exdir_db.save(
                    measurement_type='modem_dc_iq_calibration', metadata=calibration,
                    references={'delay_measurement': self.delay_measurement.id})

        self.calibrations.update(calibrations)

    def calibrate_dc_bg(self):
        # send nothing
        self.hardware.set_pulsed_mode()

        #self.pulse_sequencer.set_seq(self.trigger_daq_seq)
        ex_channel_name = [channel_name for channel_name in self.readout_channels.keys()][0]
        ex_channel = self.readout_channels[ex_channel_name]
        sequence = zi_scripts.DCSequence(ex_channel.parent.sequencer_id, self.awg)
        self.awg.set_sequence(ex_channel.parent.sequencer_id, sequence) #kostyl podumoti
        sequence.stop()
        sequence.start()

        measurer = data_reduce.data_reduce(self.adc)
        measurer.filters['bg'] = data_reduce.mean_reducer(self.adc, self.src_meas, self.axis_mean)

        dc_bg_calibration_measurement = sweep.sweep(measurer,
                                                    references={'delay_measurement': self.delay_measurement.id},
                                                    measurement_type='modem_dc_bg_calibration')
        self.exdir_db.save_measurement(dc_bg_calibration_measurement)
        self.bg = dc_bg_calibration_measurement.datasets['bg'].data

    def calibrate_dc(self, ex_channel_name, amplitude=1.0, shot_noise_time=10, save=True):
        self.hardware.set_pulsed_mode()

        calibrations = {}
        calibrated_filters = {}
        # send I and Q pulses
        ex_channel = self.readout_channels[ex_channel_name]
        # for ex_channel_name, ex_channel in self.readout_channels.items():
        # delay calibration pulse sequence
        dac_sequence_I, dac_sequence_adc_time_I = self.random_alignment_sequence(ex_channel,
                                                                                 shot_noise_time=shot_noise_time)
        dac_sequence_Q, dac_sequence_adc_time_Q = self.random_alignment_sequence(ex_channel,
                                                                                 shot_noise_time=shot_noise_time)

        dac_sequence = np.asarray(dac_sequence_I + 1j * dac_sequence_Q, dtype=np.complex) * amplitude
        dac_sequence_adc_time = dac_sequence_adc_time_I + 1j * dac_sequence_adc_time_Q
        demodulation = self.demodulation(ex_channel, sign=True)
        # set sequence in sequencer with amplitude & phase

        sequence = zi_scripts.DMSequence(ex_channel.parent.sequencer_id, self.awg, len(dac_sequence), use_modulation = True)
        self.awg.set_sequence(ex_channel.parent.sequencer_id, sequence)
        sequence.stop()
        #ex_channel.parent.get_dc_calibration()
        #ex_channel.parent.get_rf_calibration()
        calib_dc = ex_channel.parent.calib_dc()
        calib_rf = ex_channel.parent.calib_rf(ex_channel)
        # set ampl? set phase? set ofset from calibrations возможно не надо делать setSinePhase(0, 0); а то калибровка миксеров слетит(
        sequence.set_amplitude_i(np.abs(calib_rf['I']))
        sequence.set_amplitude_q(np.abs(calib_rf['Q']))
        sequence.set_phase_i(np.angle(calib_rf['I'])*360/np.pi)
        sequence.set_phase_q(np.angle(calib_rf['Q'])*360/np.pi)
        sequence.set_offset_i(np.real(calib_dc['dc']))
        sequence.set_offset_q(np.imag(calib_dc['dc']))
        sequence.set_frequency(np.abs(ex_channel.get_if())) # ex_channel == carrier
        #sequence.set_waveform(np.asarray(dac_sequence_I)* amplitude, np.asarray(dac_sequence_Q)* amplitude)
        sequence.set_waveform(np.real(dac_sequence), np.imag(dac_sequence))
        sequence.set_delay(self.trigger_channel.delay)
        sequence.start()

        calibration_measurement = self.adc.measure()
        meas_I = (np.mean(calibration_measurement[self.src_meas], axis=self.axis_mean))[
                 :len(dac_sequence_adc_time)]

        A_I = (meas_I * np.asarray(
            [demodulation[:len(dac_sequence_adc_time)], np.conj(demodulation[:len(dac_sequence_adc_time)])])).T
        # A_Q = meas_Q*np.asarray([demodulation, np.conj(demodulation)])
        b_I = dac_sequence_adc_time
        # print (A_I.shape, b_I.shape)
        print(A_I.shape, b_I.shape)
        iq_calibration = np.linalg.lstsq(A_I, b_I)[0]

        calibrations[ex_channel_name + '+'] = iq_calibration[0]
        calibrations[ex_channel_name + '-'] = iq_calibration[1]

        if save:
            self.calibrations.update(calibrations)
            # self.calibrated_filters = calibrated_filters
            self.create_filters(ex_channel_name)
        return calibrations
