import numpy as np
import logging
#from .save_pkl import *
from qsweepy.libraries.config import get_config
from qsweepy.ponyfiles.data_structures import MeasurementState, MeasurementDataset, MeasurementParameter

import traceback

class awg_digital:
    def __init__(self, awg, channel, delay_tolerance=20e-9):#, mixer):
        self.awg = awg
        self.channel = channel
        self.frozen = False
        self.delay = 0  # this value is supposed to be an integer amount of clock cycles!!!
        self.measured_delay = None
        self.delay_tolerance = delay_tolerance
        self.adc = None
        self._last_waveform = None

    def get_nop(self):
        return self.awg.get_nop()

    def get_clock(self):
        return self.awg.get_clock()

    def set_nop(self, nop):
        return self.awg.set_nop(nop)

    def set_clock(self, clock):
        return self.awg.set_clock(clock)

    def set_delay(self, delay):
        self.delay = delay

    def get_physical_devices(self):
        return [self.awg]

    def validate_modem_delay_calibration(self, modem, ex_channel_name):
        measured_delay = modem.calibrate_delay(ex_channel_name)
        # if this delay validation has been already been obtained with this calibration, don't save
        calibration_references, calibration_metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
        calibration_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_calibration', references_that=calibration_references, metadata=calibration_metadata)
        validation_metadata = {'measured_delay':measured_delay}
        try:
            validation_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_validation',
                                                                                references_that={'calibration':calibration_measurement.id}, metadata=validation_metadata)
        except Exception as e:
            print(traceback.format_exc())
            xc_dataset = MeasurementDataset([MeasurementParameter(
                name='Time delta',
                values=modem.xc_points,
                units='s',
                setter=False)],
                modem.abs_xc)
            adc_dataset = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.adc_sequence)-1)/modem.adc.get_clock(), len(modem.adc_sequence)),
                units='s',
                setter='False')],
                modem.adc_sequence)
            dac_dataset = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.dac_sequence)-1)/modem.ex_channel_clock, len(modem.dac_sequence)),
                units='s',
                setter='False')],
                modem.dac_sequence)
            dac_dataset_adc_time = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.dac_sequence_adc_time)-1)/modem.adc.get_clock(), len(modem.dac_sequence_adc_time)),
                units='s',
                setter='False')],
                modem.dac_sequence_adc_time)
            validation_measurement = modem.exdir_db.save(measurement_type='modem_readout_delay_validation',
                                                         references={'calibration':calibration_measurement.id},
                                                         metadata=validation_metadata,
                                                         datasets={'xc':xc_dataset,
                                                                   'adc':adc_dataset,
                                                                   'dac':dac_dataset,
                                                                   'dac_adc_time':dac_dataset_adc_time})
        print ('Validation measurement (delay):', measured_delay, self.delay_tolerance)
        assert (abs(measured_delay)<self.delay_tolerance)
        modem.delay_measurement = validation_measurement
        return measured_delay

    def get_modem_delay_calibration_references(self, modem, ex_channel_name):
        references = {}
        ex_channel = modem.pulse_sequencer.channels[ex_channel_name]
        if (hasattr(ex_channel, 'get_calibration_measurement')):
            references[('channel_calibration', ex_channel_name)] = ex_channel.get_calibration_measurement()
        metadata = {'ex_channel':ex_channel_name}
        return references, metadata

    def modem_delay_calibrate(self, modem, ex_channel_name):
        old_delay = self.delay
        self.delay = 0
        self.measured_delay = modem.calibrate_delay(ex_channel_name)
        if modem.exdir_db:
            references,metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
            metadata['measured_delay'] = self.measured_delay
            xc_dataset = MeasurementDataset([MeasurementParameter(
                name='Time delta',
                values=modem.xc_points,
                units='s',
                setter=False)],
                modem.abs_xc)
            adc_dataset = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.adc_sequence) - 1) / modem.adc.get_clock(), len(modem.adc_sequence)),
                units='s',
                setter='False')],
                modem.adc_sequence)
            dac_dataset = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.dac_sequence) - 1) / modem.ex_channel_clock, len(modem.dac_sequence)),
                units='s',
                setter='False')],
                modem.dac_sequence)
            dac_dataset_adc_time = MeasurementDataset([MeasurementParameter(
                name='Time',
                values=np.linspace(0, (len(modem.dac_sequence_adc_time) - 1) / modem.adc.get_clock(),
                                   len(modem.dac_sequence_adc_time)),
                units='s',
                setter='False')],
                modem.dac_sequence_adc_time)
            calibration_measurement = modem.exdir_db.save(measurement_type='modem_readout_delay_calibration',
                                                          references=references,
                                                          metadata = metadata,
                                                          datasets={'xc': xc_dataset,
                                                                    'adc': adc_dataset,
                                                                    'dac': dac_dataset,
                                                                    'dac_adc_time': dac_dataset_adc_time}
                                                          )
            modem.delay_calibration = calibration_measurement
        print ('Calibration measurement (delay):', self.measured_delay)
        self.delay = self.measured_delay

    def get_modem_delay_calibration(self, modem, ex_channel_name):
        try:
            references, metadata = self.get_modem_delay_calibration_references(modem, ex_channel_name)
            calibration_measurement = modem.exdir_db.select_measurement(measurement_type='modem_readout_delay_calibration', references_that=references, metadata=metadata)
            #modem.delay_calibration = calibration_measurement
            self.measured_delay = float(calibration_measurement.metadata['measured_delay'])
            result = self.measured_delay
            self.delay = self.measured_delay
        except Exception as e:
            print(str(e), type(e))
            self.modem_delay_calibrate(modem, ex_channel_name)
                # save calibration to database

        return self.measured_delay
