import numpy as np
import time


class READSequence:
    def __init__(self, sequencer_id, awg, use_modulation=True, awg_amp=1, trig_delay_reg=0):
        """
        Sequence for readout delay calibration.
        Send trigger for adc start.
        Produces random waveform.
        Parameters

        ----------
        awg
        n_samples - number of samples in calibration waveform.
        """

        self.awg = awg
        self.control_frequency = 0
        self.clock = float(self.awg._clock)
        self.params = dict(sequencer_id=sequencer_id, use_modulation=use_modulation,
                           awg_amp=awg_amp, i_amp = 0, q_amp = 0,
                           trig_delay_reg=trig_delay_reg,
                           ic=2 * sequencer_id, qc=sequencer_id * 2 + 1, nco_id=sequencer_id*4)

        # self.awg.
        #self.awg.set_dig_trig1_source([4, 4, 4, 4])
        #self.awg.set_dig_trig1_slope([1, 1, 1, 1])

        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/strobe/slope' % self.params['sequencer_id'], 1)
        # We need to set DIO valid polarity as  None (0- none, 1 - low, 2 - high, 3 - both )
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/valid/polarity' % self.params['sequencer_id'], 0)
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/strobe/index' % self.params['sequencer_id'], 3)
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/mask/value' % self.params['sequencer_id'], 2)
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/mask/shift' % self.params['sequencer_id'], 1)
        #self.set_awg_amp(self.params['awg_amp'])
        self.definition_fragment = '''
// Readout sequence definition fragment
//setInt('sines/{ic}/oscselect', {nco_id});
//setInt('sines/{qc}/oscselect', {nco_id});
var delay = getUserReg({trig_delay_reg});'''.format(**self.params)
        # In this fragment we define work sequence for play
        self.play_fragment = '''
    //'''


    def zicode(self):
        definition_fragment = self.definition_fragment
        play_fragment = self.play_fragment

        play_fragment1 = '''

while (true) {{
    // Wait DIO trigger from qubit control sequencer.
    setDIO(0);
    wait(3000);
    waitDigTrigger(1);
    setDIO(1);
    wait(10);
    setDIO(0);
    waitDIOTrigger();
    //setDIO(0);
    waitDigTrigger(1);
    //resetOscPhase();
    //waitSineOscPhase(1);
    //wait(10);
'''
        play_fragment2 ='''
    wait(delay);
    setTrigger(1);
    wait(10);
    setTrigger(0);
    waitWave();
}}
'''
        code = ''.join(definition_fragment + play_fragment1 + play_fragment + play_fragment2)
        #print(code)
        return code

    def add_readout_pulse(self, definition_fragment, play_fragment):
        # self.pre_pulses.append(pre_pulse)
        self.definition_fragment += definition_fragment
        self.play_fragment += play_fragment
    def clear_readout_pulse(self):
        self.definition_fragment = '''
// Readout sequence definition fragment
//setInt('sines/{ic}/oscselect', {nco_id});
//setInt('sines/{qc}/oscselect', {nco_id});
var delay = getUserReg({trig_delay_reg});'''.format(**self.params)
        # In this fragment we define work sequence for play
        self.play_fragment = '''
    //'''


    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['nco_id'], frequency)
        self.awg.set_frequency(self.params['nco_id']+1, self.control_frequency)

    def set_awg_amp(self, awg_amp):
        self.awg.set_amplitude(self.params['ic'], awg_amp)
        self.awg.set_amplitude(self.params['qc'], awg_amp)

    def set_amplitude_i(self, amplitude_i):
        assert (np.abs(amplitude_i) <= 0.5)
        # self.awg.set_register(self.params['sequencer_id'], self.params['ia'], np.asarray(int(amplitude_i*0xffffffff), dtype=np.uint32))
        self.awg.set_sin_amplitude(self.params['ic'], 0, amplitude_i)

    def set_amplitude_q(self, amplitude_q):
        assert (np.abs(amplitude_q) <= 0.5)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qa'], np.asarray(int(amplitude_q*0xffffffff), dtype=np.uint32))
        self.awg.set_sin_amplitude(self.params['qc'], 1, amplitude_q)

    def set_phase_i(self, phase_i):
        # self.awg.set_register(self.params['sequencer_id'], self.params['ip'], np.asarray(int(phase_i * 0xffffffff / 360.0), dtype=np.uint32))
        self.awg.set_sin_phase(self.params['ic'], phase_i)

    def set_phase_q(self, phase_q):
        # self.awg.set_register(self.params['sequencer_id'], self.params['qp'], np.asarray(int(phase_q * 0xffffffff / 360.0), dtype=np.uint32))
        self.awg.set_sin_phase(self.params['qc'], phase_q)

    def set_offset_i(self, offset_i):
        assert (np.abs(offset_i) <= 0.5)
        self.awg.set_offset(self.params['ic'], offset_i)
        # self.awg.set_register(self.params['sequencer_id'], self.params['io'], np.asarray(int(offset_i*0xffffffff), dtype=np.uint32))

    def set_offset_q(self, offset_q):
        assert (np.abs(offset_q) <= 0.5)
        self.awg.set_offset(self.params['qc'], offset_q)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qo'], np.asarray(int(offset_q*0xffffffff), dtype=np.uint32))

    def set_waveform(self, waveform1=None, waveform2=None, waveform_id=0):
        if (waveform1 is None) and (waveform2 is None):
            self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)
        elif (waveform2 is None):
            if len(waveform1) != self.params['n_samples']:
                print(
                    'Could not set waveform1 with length {}. Waveform length should be equal {}'.format(len(waveform1),
                                                                                                        self.params[
                                                                                                            'n_samples']))
            else:
                self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)
        elif (waveform1 is None):
            if len(waveform2) != self.params['n_samples']:
                print(
                    'Could not set waveform2 with length {}. Waveform length should be equal {}'.format(len(waveform2),
                                                                                                        self.params[
                                                                                                            'n_samples']))
            else:
                self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)

    def set_delay(self, delay):
        # self.trig_delay = delay
        # self.params['trig_delay'] = delay
        print('setting delay to ', delay)
        self.awg.set_register(self.params['sequencer_id'], self.params['trig_delay_reg'], int(-np.floor(delay * 300e6)))

    def start(self):
        #self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
        # Мне не нравится. Заставить это работать так как ты хочешь будет очень сложно.
        #if self.params['use_modulation']:
        #    self.awg.set_modulation(self.params['ic'], 1)
        #    self.awg.set_modulation(self.params['qc'], 2)
        #else:
        self.awg.set_modulation(self.params['ic'], 0)
        self.awg.set_modulation(self.params['qc'], 0)
        self.awg.set_output(self.params['ic'], 1)
        self.awg.set_output(self.params['qc'], 1)
        self.awg.start_seq(self.params['sequencer_id'])

    def stop(self):
        self.awg.stop_seq(self.params['sequencer_id'])
        self.awg.set_output(self.params['ic'], 0)
        self.awg.set_output(self.params['qc'], 0)
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
