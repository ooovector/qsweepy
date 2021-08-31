# awg_iq class
# two channels of awgs and a local oscillator used to feed a single iq mixer

# maximum settings of the current mixer that should not be exceeded

import numpy as np
import logging
from qsweepy import zi_scripts
import time

def get_config_float_fmt():
    return '6.4g'


def build_param_names(params):
    """

    Parameters
    ----------
    params : dict
        dictionary that

    Returns
    -------

    """
    list_of_strs = []
    for param_name, param_value in sorted(params.items()):
        if type(param_value) != str:  # if value is numeric (not str)
            list_of_strs.append('{0}-{1:6.4g}'.format(param_name, param_value))
        else:
            list_of_strs.append(param_name + '-' + param_value)
    return '-'.join(list_of_strs)


class Carrier:
    def __init__(self, parent):  # , mixer):
        self._if = 0
        self.frequency = parent.lo.get_frequency()
        self.parent = parent
        self.status = 1
        self.waveform = None

    def is_iq(self):
        return True

    def get_nop(self):
        return self.parent.get_nop()

    def get_clock(self):
        return self.parent.get_clock()

    def set_nop(self, nop):
        self.parent.set_nop(nop)

    def set_clock(self, clock):
        self.parent.set_clock(clock)

    def set_status(self, status):
        self.status = status
        self.parent.assemble_waveform()

    def set_if(self, _if):
        self._if = _if

    def get_if(self):
        return self._if

    def set_frequency(self, frequency):
        self._if = frequency - self.parent.lo.get_frequency()
        self.frequency = frequency

    def set_uncal_frequency(self, frequency):
        self.parent.lo.set_frequency(frequency-self.get_if())

    def get_frequency(self):
        return self.frequency

    def freeze(self):
        self.parent.freeze()

    def unfreeze(self):
        self.parent.unfreeze()

    def get_physical_devices(self):
        return self.parent.get_physical_devices()

    def get_ignore_calibration_drift(self):
        return self.parent.ignore_calibration_drift

    def set_ignore_calibration_drift(self, v):
        self.parent.ignore_calibration_drift = v

    def get_calibration_measurement(self):
        return self.parent.exdir_db.select_measurement(measurement_type='iq_rf_calibration', metadata=self.parent.rf_calibration_identifier(self)).id


class AWGIQMulti:
    """Interface for IQ modulation of RF signals wth two AWG channels.

    IQ modulation requires two low (intermediate) frequency arbitrary waveform generators for the I and Q
    connectors of the mixer and one for the LO input connector.
    Modulated signal is output at RF (radio frequency).

    Attributes:
        awg (:obj:`awg`): Instance of an AWG class (for example, hdawg)
            that is connected to the I input of the mixer. Should implement the methods get_clock,
            set_clock, set_status, run and stop.
        sequencer_id (int): sequencer id that mixer is connected to
        lo (:obj:`psg`): Instance of a sinusoidal signal generator. Should implement the methods get_frequency and set_frequency.

    """

    def __init__(self, awg, sequencer_id, lo, exdir_db):
        """
        """
        self.awg = awg
        self.sequencer_id = sequencer_id

        self.carriers = {}
        self.name = 'default'
        self.lo = lo
        self.exdir_db = exdir_db

        self.dc_calibrations = {}
        self.dc_calibrations_offset = {}
        self.rf_calibrations = {}

        self.ignore_calibration_drift = False
        self.calibration_switch_setter = lambda: None

    def get_physical_devices(self):
        return self.awg

    def get_clock(self):
        """int: Sample rate of I and Q channels (complex amplitude envelope)."""
        return self.awg.get_clock()

    def set_clock(self, clock):
        """Sets sampling rate."""
        self.awg.set_clock(clock)

    def set_status(self, status):
        """Turns on and off the lo and awg channels."""
        self.lo.set_status(status)
        self.awg.set_status(status, sequencer_id=self.sequencer_id)

    # clip DC to prevent mixer damage
    def clip_dc(self, x):
        """Clips the dc component of the output at both channels of the AWG to prevent mixer damage."""
        x = [np.real(x), np.imag(x)]
        for c in (0,1):
            if x[c] < -0.5:
                x[c] = -0.5
            if x[c] > 0.5:
                x[c] = 0.5
        x = x[0] + 1j * x[1]
        return x

    def _set_dc(self, x):
        """Clips the dc component of the output at both channels of the AWG to prevent mixer damage."""
        dc = self.clip_dc(x)
        if self.use_offset_I:
            self.awg_I.set_offset(np.real(dc), channel=self.awg_ch_I)
            dc-=np.real(dc)
        if self.use_offset_Q:
            self.awg_Q.set_offset(np.imag(dc), channel=self.awg_ch_Q)
            dc-=1j*np.imag(dc)
        self.__set_waveform_IQ_cmplx([dc]*self.get_nop())

    def dc_cname(self):
        return build_param_names(self.dc_calibration_identifier())

    def dc_calibration_identifier(self):
        return {'mixer': self.name, 'lo_freq': self.lo.get_frequency()}

    def do_calibration(self, sa=None):
        """User-level function to sort out mixer calibration matters. Checks if there is a saved calibration for the given
        LO and IF frequencies and loads it.
        When invoked with a spectrum analyzer instance as an argument it perform and save the calibration with the current
        frequencies.
        """
        print('Calibrate DC for device {} sequencer {} \n'.format(self.awg.device, self.sequencer_id))
        self.get_dc_calibration(sa)
        for carrier_name, carrier in self.carriers.items():
            print('Calibrate RF for device {} sequencer {}, if={} \n'.format(self.awg.device, self.sequencer_id,
                                                                             carrier._if))
            self.get_rf_calibration(carrier=carrier, sa=sa)

    def get_dc_calibration(self, sa=None):
        """


        Parameters
        ----------
        sa

        Returns
        -------

        """
        try:
            calibration = self.exdir_db.select_measurement(measurement_type='iq_dc_calibration', metadata=self.dc_calibration_identifier()).metadata
            self.dc_calibrations[self.dc_cname()] = {}
            for k,v in calibration.items():
                if k not in self.dc_calibration_identifier():
                    try:
                        self.dc_calibrations[self.dc_cname()][k] = float(v)
                    except Exception as e:
                        self.dc_calibrations[self.dc_cname()][k] = complex(v)

        except Exception as e:
            if not sa:
                logging.error('No ready calibration found and no spectrum analyzer to calibrate')
            else:
                self._calibrate_zero_sa(sa)
                self.save_dc_calibration()
        return self.dc_calibrations[self.dc_cname()]

    def save_dc_calibration(self):
        calibration = self.dc_calibration_identifier()
        calibration.update({k: str(v) for k,v in self.dc_calibrations[self.dc_cname()].items()})
        self.exdir_db.save(measurement_type='iq_dc_calibration', metadata=calibration, type_revision='1')

    def rf_calibration_identifier(self, carrier):
        return {'cname':self.name, 'if':carrier.get_if(), 'frequency':carrier.get_frequency()}

    def rf_cname(self, carrier):
        return build_param_names(self.rf_calibration_identifier(carrier))

    def get_rf_calibration(self, carrier, sa=None):
        """User-level function to sort out mxer calibration matters. Checks if there is a saved calibration for the given
        LO and IF frequencies and loads it.
        When invoked with a spectrum analyzer instance as an argument it perform and save the calibration with the current
        frequencies.
        """
        try:
            calibration = self.exdir_db.select_measurement(measurement_type='iq_rf_calibration', metadata=self.rf_calibration_identifier(carrier)).metadata
            self.rf_calibrations[self.rf_cname(carrier)] = {}
            for k,v in calibration.items():
                if k not in self.rf_calibration_identifier(carrier):
                    try:
                        self.rf_calibrations[self.rf_cname(carrier)][k] = float(v)
                    except:
                        self.rf_calibrations[self.rf_cname(carrier)][k] = complex(v)

        except Exception as e:
            if not sa:
                logging.error('No ready calibration found and no spectrum analyzer to calibrate')
            else:
                self._calibrate_cw_sa(sa, carrier)
                self.save_rf_calibration(carrier)
        return self.rf_calibrations[self.rf_cname(carrier)]

    def save_rf_calibration(self, carrier):
        calibration = self.rf_calibration_identifier(carrier)
        calibration.update({k:str(v) for k,v in self.rf_calibrations[self.rf_cname(carrier)].items()})
        self.exdir_db.save(measurement_type='iq_rf_calibration', metadata=calibration, type_revision='1')

    def calib_dc(self):
        cname = self.dc_cname()
        if self.ignore_calibration_drift:
            if cname not in self.dc_calibrations:
                dc_c = [calib for calib in self.dc_calibrations.values()]
                return dc_c[0]
        if cname not in self.dc_calibrations:
            print ('Calibration not loaded. Use ignore_calibration_drift to use any calibration.')
        return self.dc_calibrations[cname]

    def calib_rf(self, carrier):
        cname = self.rf_cname(carrier)
        if self.ignore_calibration_drift:
            if cname not in self.rf_calibrations:
                rf_c = [calib for calib in self.rf_calibrations.values()]
                rf_c.sort(key=lambda x: np.abs(x['if']-carrier._if)) # get calibration with nearest intermediate frequency
                return rf_c[0]
        if cname not in self.rf_calibrations:
            print ('Calibration not loaded. Use ignore_calibration_drift to use any calibration.')
        return self.rf_calibrations[cname]

    def _calibrate_cw_sa(self, sa, carrier, num_sidebands = 3, use_central = False, num_sidebands_final = 9, half_length = True, use_single_sweep=False):
        """Performs IQ mixer calibration with the spectrum analyzer sa with the intermediate frequency."""
        from scipy.optimize import fmin
        #import time
        res_bw = 4e6
        video_bw = 1e3

        sequence = zi_scripts.CWSequence(self.sequencer_id, self.awg)
        self.awg.set_sequence(self.sequencer_id, sequence)
        sequence.stop()
        sequence.set_frequency(np.abs(carrier.get_if()))
        sequence.set_amplitude_i(0)
        sequence.set_amplitude_q(0)
        sequence.set_phase_i(0)
        sequence.set_phase_q(0)
        sequence.set_offset_i(0)
        sequence.set_offset_q(0)
        sequence.start()
        self.calibration_switch_setter()

        if hasattr(sa, 'set_nop') and use_single_sweep:
            sa.set_centerfreq(self.lo.get_frequency())
            sa.set_span((num_sidebands-1)*np.abs(carrier.get_if()))
            sa.set_nop(num_sidebands)
            sa.set_detector('POS')
            sa.set_res_bw(res_bw)
            sa.set_video_bw(video_bw)
            #sa.set_trigger_mode('CONT')
            sa.set_sweep_time_auto(1)
        else:
            sa.set_detector('rms')
            sa.set_res_bw(res_bw)
            sa.set_video_bw(video_bw)
            sa.set_span(res_bw)
            if hasattr(sa, 'set_nop'):
                res_bw = 1e6
                video_bw = 2e2
                sa.set_sweep_time(50e-3)
                sa.set_nop(1)

        self.lo.set_status(True)

        sign = 1 if carrier.get_if() > 0 else -1
        solution = [-0.1, 0.1]

        def tfunc(x):
            # dc = x[0] + x[1]*1j
            target_sideband_id = 1 if carrier.get_if() > 0 else -1
            sideband_ids = np.asarray(np.linspace(-(num_sidebands - 1) / 2, (num_sidebands - 1) / 2, num_sidebands),
                                      dtype=int)
            I = 0.3
            Q = x[0] + x[1] * 1j

            if np.abs(Q) >= 0.45:
                Q = Q/np.abs(Q)*0.45

            sequence.set_amplitude_i(np.abs(I))
            sequence.set_amplitude_q(np.abs(Q))
            sequence.set_phase_i(np.angle(I)*360/np.pi)
            sequence.set_phase_q(np.angle(Q)*360/np.pi)
            sequence.set_offset_i(np.real(self.calib_dc()['dc']))
            sequence.set_offset_q(np.imag(self.calib_dc()['dc']))

            max_amplitude = np.max([np.abs(I)+np.real(self.calib_dc()['dc']), np.abs(Q)+np.imag(self.calib_dc()['dc'])])
            if max_amplitude < 1:
                clipping = 0
            else:
                clipping = (max_amplitude - 1)
            # if we can measure all sidebands in a single sweep, do it
            if hasattr(sa, 'set_nop') and use_single_sweep:
                time.sleep(0.1)
                result = sa.measure()['Power'].ravel()
            else:
                # otherwise, sweep through each sideband
                result = []
                for sideband_id in range(num_sidebands):
                    sa.set_centerfreq(
                        self.lo.get_frequency() + (sideband_id - (num_sidebands - 1) / 2.) * np.abs(carrier.get_if()))
                    result.append(np.log10(np.sum(10 ** (sa.measure()['Power'] / 10))) * 10)
                result = np.asarray(result)
            if use_central:
                bad_sidebands = sideband_ids != target_sideband_id
            else:
                bad_sidebands = np.logical_and(sideband_ids != target_sideband_id, sideband_ids != 0)
            bad_power = np.sum(10 ** ((result[bad_sidebands]) / 20))
            good_power = np.sum(10 ** ((result[sideband_ids == target_sideband_id]) / 20))
            bad_power_dbm = np.log10(bad_power) * 20
            good_power_dbm = np.log10(good_power) * 20
            print('\rdc: {0: 4.2e}\tI: {1: 4.2e}\tQ:{2: 4.2e}\t'
                  'B: {3:4.2f} G: {4:4.2f}, C:{5:4.2f}'.format(self.calib_dc()['dc'], I, Q,
                                                               bad_power_dbm, good_power_dbm, clipping) + str(result),
                  end="")
            return -good_power / bad_power + np.abs(good_power / bad_power) * 10 * clipping

        if tfunc(solution) > tfunc(-np.asarray(solution)):
            solution = -np.asarray(solution)
        print (carrier.get_if(), carrier.frequency)
        for iter_id in range(1):
            #solution = fmin(tfunc, solution, maxiter=75, xtol=2**(-13))
            solution = fmin(tfunc, solution, maxiter=30, xtol=2**(-12))
            num_sidebands = num_sidebands_final
            use_central = True

            score = tfunc(solution)
        Q_save = solution[0]+solution[1]*1j
        if np.abs(Q_save) >= 0.45:
            Q_save = Q_save / np.abs(Q_save) * 0.45

        self.rf_calibrations[self.rf_cname(carrier)] = {'I': 0.3,
                                                        'Q': Q_save,
                                                        'score': score,
                                                        'num_sidebands': num_sidebands,
                                                        'if': carrier._if,
                                                        'lo_freq': self.lo.get_frequency()}

        return self.rf_calibrations[self.rf_cname(carrier)]

    def _calibrate_zero_sa(self, sa):
        """Performs IQ mixer calibration for DC signals at the I and Q inputs."""
        import time
        from scipy.optimize import fmin
        print(self.lo.get_frequency())

        self.calibration_switch_setter()
        sequence = zi_scripts.CWSequence(self.sequencer_id, self.awg)
        self.awg.set_sequence(self.sequencer_id, sequence)
        sequence.stop()
        sequence.set_amplitude_i(0)
        sequence.set_amplitude_q(0)
        sequence.set_phase_i(0)
        sequence.set_phase_q(0)
        sequence.set_offset_i(0)
        sequence.set_offset_q(0)
        sequence.start()

        res_bw = 4e6
        video_bw = 2e2
        sa.set_res_bw(res_bw)
        sa.set_video_bw(video_bw)
        sa.set_detector('rms')
        sa.set_centerfreq(self.lo.get_frequency())
        sa.set_sweep_time(50e-3)
        #time.sleep(0.1)
        if hasattr(sa, 'set_nop'):
            sa.set_span(0)
            sa.set_nop(1)
        #self.set_trigger_mode('CONT')
        else:
            sa.set_span(res_bw)
        self.lo.set_status(True)
        def tfunc(x):
            sequence.set_offset_i(x[0])
            sequence.set_offset_q(x[1])
            if hasattr(sa, 'set_nop'):
                result = sa.measure()['Power'].ravel()[0]
            else:
                #result = np.log10(np.sum(10**(sa.measure()['Power']/10)))*10
                result = np.log10(np.sum(sa.measure()['Power']))*10
            print (x, result)
            return result

        #solution = fmin(tfunc, [0.3,0.3], maxiter=30, xtol=2**(-14))
        solution = fmin(tfunc, [0.1,0.1], maxiter=50, xtol=2**(-13))
        x = self.clip_dc(solution[0]+1j*solution[1])
        self.zero = x

        self.dc_calibrations[self.dc_cname()] = {'dc': solution[0]+solution[1]*1j}

        return self.dc_calibrations[self.dc_cname()]
