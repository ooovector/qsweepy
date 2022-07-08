import numpy as np
import time

class DMSequence:
    def __init__(self, sequencer_id, awg, n_samples, use_modulation=False, awg_amp = 1, trig_delay_reg=0):
        """
        Sequence for readout delay calibration.
        Send trigger for adc start.
        Produces random waveform.
        Parameters

        ----------
        awg
        trigger_channel - Physical AWG RF "trig" input connected for trigger from ADC. It should be the same which would
        be used for further measurements.
        dig_trig - Digital trigger for sequencer. Can be 1 or 2 for each sequencer. It should be the same which would
        be used for further measurements.
        n_samples - number of samples in calibration waveform.
        """
        self.awg = awg
        self.clock = self.awg._clock
        self.params = dict(sequencer_id=sequencer_id, use_modulation = use_modulation,
                           n_samples=n_samples, awg_amp = awg_amp, trig_delay_reg = trig_delay_reg,
                           ic=2*sequencer_id, qc=sequencer_id*2+1, nco_id=sequencer_id*4)

        self.set_awg_amp(self.params['awg_amp'])
        #self.awg.set_dig_trig1_source([4, 4, 4, 4])
        #self.awg.set_dig_trig1_slope([1, 1, 1, 1])

    def zicode(self):
        code = '''
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id});
const n_samp = {n_samples};
wave w_zeros = rect(n_samp, 1);
var delay = getUserReg({trig_delay_reg});

while (true) {{
   // Wait trigger from adc. The same trigger channel as for qubit control sequence.
   wait(3000);
   waitDigTrigger(1);
   //resetOscPhase();
   //wait(10);
   //setSinePhase(0, 270);
   playWave(w_zeros, w_zeros);
   wait(delay);
   setTrigger(1);
   wait(10);
   setTrigger(0);
   waitWave();
}}
'''.format(**self.params)
        return code

    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['sequencer_id']*4, frequency)

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
        self.awg.set_offset(channel=self.params['ic'], offset=offset_i)
        # self.awg.set_register(self.params['sequencer_id'], self.params['io'], np.asarray(int(offset_i*0xffffffff), dtype=np.uint32))

    def set_offset_q(self, offset_q):
        assert (np.abs(offset_q) <= 0.5)
        self.awg.set_offset(channel=self.params['qc'], offset=offset_q)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qo'], np.asarray(int(offset_q*0xffffffff), dtype=np.uint32))

    def set_waveform(self, waveform1 = None, waveform2 = None, waveform_id = 0):
        if (waveform1 is None) and (waveform2 is None):
            self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)
        elif (waveform2 is None):
            if len(waveform1)!= self.params['n_samples']:
                print('Could not set waveform1 with length {}. Waveform length should be equal {}'.format(len(waveform1),
                                                                                            self.params['n_samples']))
            else:
                self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)
        elif (waveform1 is None):
            if len(waveform2)!= self.params['n_samples']:
                print('Could not set waveform2 with length {}. Waveform length should be equal {}'.format(len(waveform2),
                                                                                            self.params['n_samples']))
            else:
                self.awg.set_waveform_indexed(self.params['sequencer_id'], waveform_id, waveform1, waveform2)

    def set_delay(self, delay):
        #self.trig_delay = delay
        #self.params['trig_delay'] = delay
        print('setting delay to ',  delay)
        self.awg.set_register(self.params['sequencer_id'], self.params['trig_delay_reg'], int(-np.floor(delay*300e6)))

    def start(self):
        self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
        # Мне не нравится. Заставить это работать так как ты хочешь будет очень сложно.
        if self.params['use_modulation']:
            self.awg.set_modulation(self.params['ic'], 1)
            self.awg.set_modulation(self.params['qc'], 2)
        else:
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
