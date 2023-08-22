import numpy as np
import time


class DCSequence:
    def __init__(self, sequencer_id, awg, trigger_channel=4, dig_trig=1):
        """
        Sequence for modem readout dc calibration.
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
        self.params = dict(sequencer_id=sequencer_id, trigger_channel=trigger_channel, dig_trig=dig_trig)
        self.awg = awg
        #if dig_trig == 1:
        #    self.awg.set_dig_trig1_source(4 * [self.params['trigger_channel']])
        #else:
        #    self.awg.set_dig_trig2_source(4 * [self.params['trigger_channel']])

    def zicode(self):
        code = '''
//wave marker = marker(50,1);

while (true) {{
   // Send trigger to adc. The same trigger channel as for qubit control sequence.
   setTrigger(1);
   wait(10);
   setTrigger(0);
   //playWave(marker);
   waitWave();

}}
'''.format(**self.params)
        return code


    def start(self):
        self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_output(2 * self.params['sequencer_id'], 1)
        self.awg.set_output(2 * self.params['sequencer_id'] + 1, 1)
        self.awg.set_sin_enable(2 * self.params['sequencer_id'], 0, 1)
        self.awg.set_sin_enable(2 * self.params['sequencer_id'] + 1, 1, 1)

    def stop(self):
        self.awg.stop_seq(self.params['sequencer_id'])
        self.awg.set_output(2 * self.params['sequencer_id'], 0)
        self.awg.set_output(2 * self.params['sequencer_id'] + 1, 0)
        self.awg.set_sin_enable(2 * self.params['sequencer_id'], 0, 0)
        self.awg.set_sin_enable(2 * self.params['sequencer_id'] + 1, 1, 0)
