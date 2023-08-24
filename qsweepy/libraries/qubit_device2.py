from qsweepy.libraries import pulses2 as pulses
from qsweepy.libraries import awg_iq_multi2 as awg_iq_multi
from qsweepy.libraries import awg_channel2 as awg_channel
from qsweepy.libraries import modem_readout2 as modem_readout


import copy
from qsweepy.ponyfiles.exdir_db import Exdir_db
from typing import Mapping, List


class QubitDevice:
    # explicit variables type declaration
    exdir_db: Exdir_db
    pg: pulses.Pulses

    def __init__(self, exdir_db, sweeper, hardware = None, controls=()):
        self.exdir_db = exdir_db
        self.ftol = 100
        self.sweeper = sweeper
        self.controls = controls
        self.hardware = hardware
        self.pre_pulses = []
        self.all_cov = False

        self.pg = None

    def set_qubits_from_dict(self, _dict: Mapping):
        try:
            assert set(_dict.keys()) == set(self.get_qubit_list())
        except Exception as e:
            print(str(e), type(e))
            self.set_qubit_list(_dict)
        for qubit_id, qubit in _dict.items():
            if 'r' in qubit:
                if 'Fr' in qubit['r']:
                    try:
                        assert(abs(self.get_qubit_fr(qubit_id)-qubit['r']['Fr'])<self.ftol)
                    except Exception as e:
                        print(str(e), type(e))
                        self.set_qubit_fr(qubit_id=qubit_id, fr=qubit['r']['Fr'])
                if 'iq_devices' in qubit['r']:
                    self.set_qubit_readout_channel_list(qubit_id, qubit['r']['iq_devices'])
            if 'q' in qubit:
                if 'F01_min' in qubit['q']['F']:

                    try:
                        assert(abs(self.get_qubit_fq(qubit_id, transition_name='01')-qubit['q']['F']['F01_min'])<self.ftol)
                    except Exception as e:
                        print(str(e), type(e))
                        self.set_qubit_fq(qubit_id=qubit_id, fq=qubit['q']['F']['F01_min'], transition_name='01')
                if 'F12_min' in qubit['q']['F']:
                    try:
                        assert(abs(self.get_qubit_fq(qubit_id, transition_name='12')-qubit['q']['F']['F12_min']))<self.ftol
                    except Exception as e:
                        print(str(e), type(e))
                        self.set_qubit_fq(qubit_id=qubit_id, fq=qubit['q']['F']['F12_min'], transition_name='12')
                if 'iq_devices' in qubit['q']:
                    self.set_qubit_excitation_channel_list(qubit_id, qubit['q']['iq_devices'])
                if 'iq_devices_transitions' in qubit['q']:
                    self.set_qubit_excitation_transition_list(qubit_id, qubit['q']['iq_devices_transitions'])

            for key, value in qubit.items():
                if key != 'r' and key != 'q':
                    try:
                        last_measurement = self.exdir_db.select_measurement(measurement_type=key, metadata={'qubit_id':qubit_id})
                        assert(last_measurement.metadata[key] == str(value))
                    except:
                        self.exdir_db.save(measurement_type=key, metadata={key: str(value), 'qubit_id': qubit_id})

    def set_sample_globals(self, globals):
        for global_name, global_value in globals.items():
            try:
                last_measurement = self.exdir_db.select_measurement(measurement_type=global_name, metadata={'scope': 'sample'})
                assert(last_measurement.metadata[global_name] == str(global_value))
            except:
                # if there is no measurement associated
                self.exdir_db.save(measurement_type=global_name, metadata={global_name: str(global_value), 'scope':'sample'})

    def get_sample_global(self, name):
        return self.exdir_db.select_measurement(measurement_type=name, metadata={'scope':'sample'}).metadata[name]

    def set_qubit_constant(self, qubit_id, name, value):
        self.exdir_db.save(measurement_type=name, metadata={'qubit_id': qubit_id, name: value})

    def get_qubit_constant(self, name, qubit_id):
        try:
            return self.exdir_db.select_measurement(measurement_type=name, metadata={'qubit_id': qubit_id}).metadata[name]
        except:
            return self.exdir_db.select_measurement(measurement_type=name, metadata={'scope': 'sample'}).metadata[name]

    def get_frequency_control_measurement_id(self, qubit_id, control_values={}):
        frequency_control_values = {}
        for frequency_control in self.get_frequency_controls(qubit_id=qubit_id):
            if frequency_control not in control_values:
                frequency_control_values[frequency_control] = self.controls[frequency_control].get_offset()
            else:
                frequency_control_values[frequency_control] = control_values[frequency_control]
        metadata = {'qubit_id':qubit_id}
        metadata.update(frequency_control_values)
        try:
            frequency_controls = self.exdir_db.select_measurement(measurement_type='frequency_control', metadata=metadata)
        except:
            frequency_controls = self.exdir_db.save(measurement_type='frequency_control', metadata=metadata)
        return frequency_controls.id

    def set_frequency_controls(self, qubit_id, controls):
        try:
            assert(sorted(self.get_frequency_controls(qubit_id))==sorted(controls))
        except:
            metadata = {control:'-' for control in controls}
            metadata['qubit_id'] = qubit_id
            self.exdir_db.save(measurement_type='qubit_frequency_controls', metadata=metadata)

    def get_frequency_controls(self, qubit_id):
        frequency_controls = self.exdir_db.select_measurement(measurement_type='qubit_frequency_controls', metadata={'qubit_id':qubit_id})
        return [k for k,v in frequency_controls.metadata.items() if k != 'qubit_id']

    def get_qubit_fq(self, qubit_id, transition_name='01', control_values={}):
        """
        Reads qubit's transition frequency from exdir_db.

        Parameters
        ----------
        qubit_id : str
            Qubit identifier

        Returns
        -------
        fq : float
            qubit transition frequency
        """
        metadata = {'qubit_id':qubit_id, 'transition_name':transition_name}
        try:
            fq_measurement = self.exdir_db.select_measurement(
                measurement_type='qubit_fq',
                metadata=metadata,
                references_that={'frequency_controls': self.get_frequency_control_measurement_id(qubit_id=qubit_id,
                                                                                            control_values=control_values)}
            )
        except:
            fq_measurement = self.exdir_db.select_measurement(measurement_type='qubit_fq', metadata=metadata) ##TODO: try to pick closest control
        return float(fq_measurement.metadata['fq'])

    def set_qubit_fq(self, fq, qubit_id, transition_name='01', control_values={}):
        metadata={'qubit_id':qubit_id, 'transition_name':transition_name, 'fq':str(fq)}

        self.exdir_db.save(measurement_type='qubit_fq',
                           metadata=metadata,
                           references={'frequency_controls': self.get_frequency_control_measurement_id(qubit_id=qubit_id, control_values=control_values)})

    def get_qubit_fr(self, qubit_id, control_values={}, ignore_control_values=False, recalibrate=True):
        from qsweepy.qubit_calibrations import spectroscopy
        """
        Reads qubit's readout frequency from exdir_db for given control_values. If the

        Parameters
        ----------
        qubit_id : str
            Qubit identifier

        Returns
        -------
        fr : float
            qubit readout frequency
        """
        frequency_control_measurement_id = self.get_frequency_control_measurement_id(qubit_id=qubit_id,
                                                                                     control_values=control_values)
        try:
            fr_measurement = self.exdir_db.select_measurement(measurement_type='qubit_fr',
                                                          metadata={'qubit_id':qubit_id},
                    references_that={'frequency_controls': frequency_control_measurement_id})
        except IndexError as e:
            print ('Could not find fr measurement for qubit {} with frequency controls id {}'.format(qubit_id,
                                                                                frequency_control_measurement_id))
            assert not ignore_control_values
            assert recalibrate
            assert not len(control_values)
            fr_measurement = self.exdir_db.select_measurement(measurement_type='qubit_fr',
                                                              metadata={'qubit_id': qubit_id})
            fr_guess = float(fr_measurement.metadata['fr'])
            spectrum_fit = spectroscopy.measure_fr(self, qubit_id, fr_guess)
            fr = float(spectrum_fit.metadata['fr'])

            self.set_qubit_fr(fr=fr, qubit_id=qubit_id, measurement_reference={'single_tone_spectrum_fit': spectrum_fit.id})

        return float(fr_measurement.metadata['fr'])

    def set_qubit_fr(self, fr, qubit_id, control_values={}, measurement_reference={}):
        references = {'frequency_controls': self.get_frequency_control_measurement_id(qubit_id=qubit_id,
                                                                                    control_values=control_values)}
        references.update(measurement_reference)
        print(references)
        self.exdir_db.save(measurement_type='qubit_fr',
                                                  metadata={'qubit_id':qubit_id,
                                                            'fr': str(fr)},
                                                  references=references)


    def set_qubit_list(self, qubit_list):
        self.exdir_db.save(measurement_type='qubit_list', metadata={key: key for key in qubit_list})

    def get_qubit_list(self):
        """

        Returns
        -------
        list[str]
            List of separate qubits ids. Preferably represented by string number e.g. ["1","3"].
        """
        return [i for i in self.exdir_db.select_measurement(measurement_type='qubit_list').metadata.keys()]

    def set_qubit_excitation_channel_list(self, qubit_id, device_list):
        metadata = {k:v for k,v in device_list.items()}
        metadata['qubit_id'] = qubit_id
        self.exdir_db.save(measurement_type='qubit_excitation_channel_list', metadata=metadata)

    def set_qubit_excitation_transition_list(self, qubit_id, transition_list):
        metadata = {k:v for k,v in transition_list.items()}
        metadata['qubit_id'] = qubit_id
        self.exdir_db.save(measurement_type='qubit_excitation_transition_list', metadata=metadata)

    def get_qubit_excitation_channel_list(self, qubit_id, transition='01'):
        excitations_db_metadata = self.exdir_db.select_measurement(measurement_type='qubit_excitation_channel_list',
                                                                   metadata={'qubit_id': qubit_id}).metadata
        if transition is not None:
            excitation_transition_types = self.exdir_db.select_measurement(measurement_type='qubit_excitation_transition_list',
                                                                       metadata={'qubit_id': qubit_id}).metadata

            excitations = {k:v for k,v in excitations_db_metadata.items() if k != 'qubit_id' and excitation_transition_types[k] == transition}
        else:
            excitations = {k:v for k,v in excitations_db_metadata.items() if k != 'qubit_id'}
        #excitations_db_metadata = copy.deepcopy(excitations_db_metadata)

        return excitations

    def get_qubit_excitation_transition_list(self, qubit_id):
        excitation_transition_types = self.exdir_db.select_measurement(
            measurement_type='qubit_excitation_transition_list',
            metadata={'qubit_id': qubit_id}).metadata
        return {k: v for k, v in excitation_transition_types.items() if k != 'qubit_id'}

    def set_qubit_readout_channel_list(self, qubit_id, device_list):
        metadata = {k:v for k,v in device_list.items()}
        metadata['qubit_id'] = qubit_id
        self.exdir_db.save(measurement_type='qubit_readout_channel_list', metadata=metadata)

    def get_qubit_readout_channel_list(self, qubit_id):
        """
        Loads channel_name: device_name map from exdir_db system.

        channel_name
            name of the channel in the sample to be measured.
        device_name
            name of the device that sends EM energy into this channel.

        Parameters
        ----------
        qubit_id : str
            qubit identifier

        Returns
        -------
        excitations_map : dict[str,str]
            channel_name: channel_device_name dictionary

        Notes
        ------
        As a consequence: only one device can be connected to the specific channel.

        See Also
        -------
        QubitDevice.set_qubits_from_dict : the way to load this parameters to the exdir_db system.
        """
        readout_db_metadata = self.exdir_db.select_measurement(measurement_type='qubit_readout_channel_list',
                                                 metadata={'qubit_id': qubit_id}).metadata
        readout_db_metadata = copy.deepcopy(readout_db_metadata)

        # everything except qubit_id represents mappinig from channels to devices
        del readout_db_metadata ["qubit_id"]
        readout_map = readout_db_metadata

        return readout_map

    def get_two_qubit_gates(self):
        gates = {}
        gate_list = [gate for gate in self.exdir_db.select_measurement(measurement_type='two_qubit_gate_list').metadata.values()]
        for gate_id in gate_list:
            gates[gate_id] = self.exdir_db.select_measurement(measurement_type='two_qubit_gate', metadata={'gate_id': gate_id})
        return gates

    def set_two_qubit_gates_from_dict(self, two_qubit_gates):
        self.exdir_db.save(measurement_type='two_qubit_gate_list', metadata={key: key for key in two_qubit_gates})
        for gate_id, gate in two_qubit_gates.items():
            gate_entry = {k: v for k, v in gate.items()}
            gate_entry['gate_id'] = gate_id
            self.exdir_db.save(measurement_type='two_qubit_gate', metadata=gate_entry)

    def get_zgates(self):
        gates = {}
        gate_list = [gate for gate in self.exdir_db.select_measurement(measurement_type='zgate_list').metadata.values()]
        for gate_id in gate_list:
            gates[gate_id] = self.exdir_db.select_measurement(measurement_type='zgate', metadata={'gate_id': gate_id})
        return gates

    def set_zgates_from_dict(self, zgates):
        self.exdir_db.save(measurement_type='zgate_list', metadata={key: key for key in zgates})
        for gate_id, gate in zgates.items():
            gate_entry = {k: v for k, v in gate.items()}
            gate_entry['gate_id'] = gate_id
            self.exdir_db.save(measurement_type='zgate', metadata=gate_entry)

    def create_pulsed_interfaces(self, iq_devices, fast_controls, extra_channels={}):
        self.awg_channels = {}
        self.readout_channels = {}
        for _qubit_id in self.get_qubit_list():
            for channel_name, device_name in self.get_qubit_excitation_channel_list(_qubit_id, None).items():
                if device_name in iq_devices:
                    device = iq_devices[device_name]
                    carrier = awg_iq_multi.Carrier(device)
                    device.carriers[channel_name] = carrier
                else:
                    carrier = awg_channel.awg_channel_carrier(fast_controls[device_name], frequency = None)
                assert (channel_name not in self.awg_channels)
                self.awg_channels[channel_name] = carrier

            for channel_name, device_name in self.get_qubit_readout_channel_list(_qubit_id).items():
                if device_name in iq_devices:
                    device = iq_devices[device_name]
                    carrier = awg_iq_multi.Carrier(device)
                    device.carriers[channel_name] = carrier
                else:
                    carrier = awg_channel.awg_channel_carrier(fast_controls[device_name], frequency=None)
                assert (channel_name not in self.awg_channels)
                self.awg_channels[channel_name] = carrier
                self.readout_channels[channel_name] = carrier

        for fast_control_name, fast_control in fast_controls.items():
            self.awg_channels[fast_control_name] = awg_channel.awg_channel_carrier(fast_control, frequency=0.0)

        for gate_id, gate in self.get_two_qubit_gates().items():
            if gate.metadata['pulse_type'] == 'parametric':
                control_name = gate.metadata['control']
                control = fast_controls[control_name]
                carrier_name = gate.metadata['carrier_name']
                assert (carrier_name not in self.awg_channels)
                self.awg_channels[carrier_name] = awg_channel.awg_channel_carrier(control,frequency = None)
            #pass

        for gate_id, gate in self.get_zgates().items():
            if 'pulse_type' in gate.metadata:
                if gate.metadata['pulse_type'] == 'parametric':
                    control_name = gate.metadata['control']
                    control = fast_controls[control_name]
                    carrier_name = gate.metadata['carrier_name']
                    assert (carrier_name not in self.awg_channels)
                    frequency = float(gate.metadata['frequency'])
                    self.awg_channels[carrier_name] = awg_channel.awg_channel_carrier(control, frequency = frequency)

        self.awg_channels.update(extra_channels)
        self.pg = pulses.Pulses(self.awg_channels)
        self.update_pulsed_frequencies()

    def update_pulsed_frequencies(self):
        iq_devices = []
        for _qubit_id in self.get_qubit_list():
            fr = self.get_qubit_fr(_qubit_id)
            fq = self.get_qubit_fq(_qubit_id)
            transitions = self.get_qubit_excitation_transition_list(_qubit_id)
            for channel_name, device_name in self.get_qubit_excitation_channel_list(_qubit_id, None).items():
                if transitions[channel_name] == '01':
                    self.awg_channels[channel_name].set_frequency(fq)
                elif transitions[channel_name] == 'q01r10':
                    self.awg_channels[channel_name].set_frequency(fr-fq)
                elif transitions[channel_name] == '12':
                    fq12 = self.get_qubit_fq(_qubit_id, transition_name='12')
                    self.awg_channels[channel_name].set_frequency(fq12)
                iq_devices.append(self.awg_channels[channel_name].parent)
            for channel_name, device_name in self.get_qubit_readout_channel_list(_qubit_id).items():
                self.awg_channels[channel_name].set_frequency(fr)
                iq_devices.append(self.awg_channels[channel_name].parent)
        for gate_id, gate in self.get_two_qubit_gates().items():
            if gate.metadata['pulse_type'] == 'parametric':
                carrier_name = gate.metadata['carrier_name']
                carrier = self.awg_channels[carrier_name]
                f1 = self.get_qubit_fq(qubit_id=gate.metadata['q1'], transition_name=gate.metadata['transition_q1'])
                f2 = self.get_qubit_fq(qubit_id=gate.metadata['q2'], transition_name=gate.metadata['transition_q2'])
                if gate.metadata['logic_type'] == 'iSWAP':
                    frequency = (f1 - f2)*float(gate.metadata['carrier_harmonic'])
                elif gate.metadata['logic_type'] == 'bSWAP':
                    frequency = (f1 + f2) * float(gate.metadata['carrier_harmonic'])
                carrier.set_frequency(frequency)

        for d in list(set(iq_devices)):
            if hasattr(d, 'do_calibration'):
                d.do_calibration(d.sa)

    def setup_modem_readout(self, hardware):
        hardware.adc.set_adc_nums(int(self.get_sample_global('delay_calibration_nums')))
        hardware.adc.set_adc_nop(int(self.get_sample_global('delay_calibration_nop')))

        #self.trigger_readout_seq = [self.pg.p('ro_trg', hardware.get_readout_trigger_pulse_length(), self.pg.rect, 1)]
        self.trigger_readout_seq = []

        self.modem = modem_readout.modem_readout(self.pg, hardware, hardware.ro_trg, axis_mean=0, exdir_db=self.exdir_db)
        self.modem.save=True
        self.modem.sequence_length = int(self.get_sample_global('delay_calibration_sequence_length'))
        self.modem.readout_channels = self.readout_channels

        self.modem.adc_device = hardware.adc

        if hasattr(self.modem.adc_device, 'sync_mode'):
            if not self.modem.adc_device.sync_mode:
                print('ADC device is turned to sync mode')
                self.modem.adc_device.sync_mode = True
                self.modem.adc_device.set_cur_prog(self.modem.adc_device.initial_param_values, 0)
                self.modem.adc_device.send_cur_prog(0)
        else:
            pass

        self.modem_delay_calibration_channel = [channel_name for channel_name in self.modem.readout_channels.keys()][0]
        self.ro_trg = hardware.ro_trg

        readout_trigger_delay = self.ro_trg.get_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
        print('Got delay calibration:', readout_trigger_delay)
        try:
            self.ro_trg.validate_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
        except Exception as e:
            print(type(e), str(e))
            self.ro_trg.modem_delay_calibrate(self.modem, self.modem_delay_calibration_channel)
            self.ro_trg.validate_modem_delay_calibration(self.modem, self.modem_delay_calibration_channel)
        self.modem.get_dc_bg_calibration()
        self.modem.get_dc_calibrations(amplitude=hardware.get_modem_dc_calibration_amplitude())
        return self.modem

    def setup_adc_reducer_iq(self, qubits, raw=False, internal_avg=True): ### pimp this code to make it more universal. All the hardware belongs to the hardware
        # file, but how do we do that here without too much boilerplate???
        # params: qubits: str or list #
        feature_id = 0

        self.hardware.set_pulsed_mode()
        if self.hardware.adc.devtype == 'SK':
            from qsweepy.instrument_drivers.TSW14J56driver import TSW14J56_evm_reducer
            adc_reducer = TSW14J56_evm_reducer(self.modem.adc_device.adc)
            #adc_reducer = self.hardware.adc
            adc_reducer.devtype = 'SK'
            adc_reducer.averaging = internal_avg
            adc_reducer.samples = raw
            adc_reducer.last_cov = False
            adc_reducer.avg_cov = True
            adc_reducer.resultnumber = False
            adc_reducer.avg_cov_mode = 'iq'

        elif self.hardware.adc.devtype == 'UHF':
            adc_reducer = self.hardware.adc
            adc_reducer.devtype = 'UHF'
            adc_reducer.averaging = internal_avg
            adc_reducer.config_iterations(adc_reducer.nsegm, adc_reducer.nres)
            adc_reducer.output_result = True
            adc_reducer.output_resnum = False
            adc_reducer.output_raw = raw

        qubit_measurement_dict = {}

        if type(qubits) is str:
            qubits = [qubits]
        for qubit_id in qubits:
            if feature_id > 1:
                raise ValueError('Cannot setup hardware adc_reducer for more that 2 qubits')
            readout_channel_name = [i for i in self.get_qubit_readout_channel_list(qubit_id=qubit_id).keys()][0]
            calibration = self.modem.iq_readout_calibrations[readout_channel_name]

            print(qubit_id, len(calibration['feature']))
            qubit_measurement_dict[qubit_id] = adc_reducer.result_source + str(feature_id)

            adc_reducer.set_feature_iq(feature_id=feature_id, feature=calibration['feature'])
            feature_id += 1
        return adc_reducer, qubit_measurement_dict

    # def set_adc_features_and_thresholds(self, features, thresholds, disable_rest=True, raw=False, internal_avg=False):
    #     self.hardware.set_pulsed_mode()
    #     if self.hardware.adc.devtype == 'SK':
    #         from qsweepy.instrument_drivers.TSW14J56driver import TSW14J56_evm_reducer
    #         adc_reducer = TSW14J56_evm_reducer(self.modem.adc_device.adc)
    #         adc_reducer.output_raw = raw
    #         adc_reducer.last_cov = False
    #         adc_reducer.avg_cov = False
    #         adc_reducer.resultnumber = True
    #         adc_reducer.all_cov = self.all_cov
    #
    #         adc_reducer.avg_cov_mode = 'real'
    #     elif self.hardware.adc.devtype == 'UHF':
    #         adc_reducer = self.hardware.adc
    #         adc_reducer.internal_avg = internal_avg
    #         adc_reducer.config_iterations(adc_reducer.nsegm, adc_reducer.nres)
    #
    #
    #     feature_id = 0
    #     for feature, threshold in zip(features, thresholds):
    #         adc_reducer.set_feature_real(feature_id=feature_id, feature=feature, threshold=threshold)
    #         feature_id += 1
    #
    #     adc_reducer.resultnumbers_dimension = 2**feature_id
    #
    #     if disable_rest:
    #         while (feature_id < self.modem.adc_device.adc.num_covariances):
    #             adc_reducer.disable_feature(feature_id=feature_id)
    #             feature_id += 1
    #     return adc_reducer

    def set_adc_features_and_thresholds(self, features, thresholds, disable_rest=True, raw=False, internal_avg=False):
        self.hardware.set_pulsed_mode()
        if self.hardware.adc.devtype == 'SK':
            from qsweepy.instrument_drivers.TSW14J56driver import TSW14J56_evm_reducer
            adc_reducer = TSW14J56_evm_reducer(self.modem.adc_device.adc)
            adc_reducer.samples = raw
            adc_reducer.last_cov = False
            adc_reducer.avg_cov = False
            adc_reducer.resultnumber = True
            adc_reducer.all_cov = self.all_cov

            adc_reducer.avg_cov_mode = 'real'
        elif self.hardware.adc.devtype == 'UHF':
            adc_reducer = self.hardware.adc
            adc_reducer.internal_avg = internal_avg
            adc_reducer.config_iterations(adc_reducer.nsegm, adc_reducer.nres)


        feature_id = 0
        for feature, threshold in zip(features, thresholds):
            adc_reducer.set_feature_real(feature_id=feature_id, feature=feature, threshold=threshold)
            feature_id += 1

        adc_reducer.resultnumbers_dimension = 2**feature_id

        if disable_rest:
            while (feature_id < self.modem.adc_device.adc.num_covariances):
                adc_reducer.disable_feature(feature_id=feature_id)
                feature_id += 1
        return adc_reducer

    def invalid_calib(self, invalid: bool, calib_type: str, awg_ch: str):
        """
        Change validation of calibration type on channel from self.awg_channels

        invalid: True or False
        calib_type: 'iq_rf_calibration' or 'iq_dc_calibration'
        awg_ch: 'iq_ex1_q1', 'iq_ro_q1', 'iq_ex2_q2', 'iq_ex2_q2_12', 'iq_ro_q2', 'iq_ex3_q3', 'iq_ex3_q3_12', 'iq_ro_q3', etc

        """
        if calib_type == 'iq_rf_calibration':
            calib = self.exdir_db.select_measurement(measurement_type=calib_type,
                                                     metadata=self.awg_channels[awg_ch].parent.rf_calibration_identifier(self.awg_channels[awg_ch]),\
                                                     ignore_invalidation = True)
        elif calib_type =='iq_dc_calibration':
            calib = self.exdir_db.select_measurement(measurement_type=calib_type,
                                                     metadata=self.awg_channels[awg_ch].parent.dc_calibration_identifier(),\
                                                     ignore_invalidation = True)
        calib.invalid = invalid
        self.exdir_db.db.update_in_database(calib)
        return calib.id