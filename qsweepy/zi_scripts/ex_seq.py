import numpy as np
import time


class EXSequence:
    def __init__(self, sequencer_id, awg, n_samp=16, use_modulation=False, awg_amp=1, trigger_channel=0,
                 dig_trig=1, trig_delay_reg=0):
        """
        Sequence for qubit exitation and manipulation.
        Send trigger for readout sequence start.
        Parameters

        ----------
        awg
        trigger_channel - Physical AWG RF "trig" input connected for trigger from ADC. It should be the same which would
        be used for further measurements.
        dig_trig - Digital trigger for sequencer. Can be 1 or 2 for each sequencer. It should be the same which would
        be used for further measurements.
        n_samples - number of samples in calibration waveform.
        """

        self.params = dict(sequencer_id=sequencer_id, trigger_channel=trigger_channel, use_modulation=use_modulation,
                           n_samples=n_samp, awg_amp=awg_amp, dig_trig=dig_trig, trig_delay_reg=trig_delay_reg,
                           ic=2 * sequencer_id, qc=sequencer_id * 2 + 1)
        self.awg = awg
        # self.awg.
        if dig_trig == 1:
            self.awg.set_dig_trig1_source(4 * [self.params['trigger_channel']])
        else:
            self.awg.set_dig_trig2_source(4 * [self.params['trigger_channel']])
        self.set_awg_amp(self.params['awg_amp'])

    def zicode(self):

        definition_fragments = []
        play_fragment = []
        if pre_pulses is not None:
            definition_fragments.append(self.pre_pulses.get_definition_fragment(self.params('sequencer_id')))

            play_fragment.append(self.pre_pulses.get_play_fragment(self.params('sequencer_id')))
        play_fragment1 = []
        play_fragment2 = []

        for wave_length in range(1, self.params('n_samp') + 1):
            if wave_length != self.params('n_samp'):
                definition_fragments.append(textwrap.dedent('''
                wave w_{wave_length} = join(zeros({n_samp}-{wave_length}), ones({wave_length}));
                '''.format(wave_length=wave_length, n_samp=self.params('n_samp'))))
            else:
                definition_fragments.append(textwrap.dedent('''
                wave w_{wave_length} = ones({wave_length});
                '''.format(wave_length=wave_length)))

        for wave_length in range(1, self.params('n_samp') + 1):
            switch_fragment.append(textwrap.dedent('''
                case {wave_length}:
                    playWawe(w_{wave_length})
                '''.format(wave_length=wave_length)))

        play_fragment1 = '''
while (true) {{
    waitDigTrigger(1);
    setDIO(1);
    wait(50);
    waitDIOTrigger();
    wait(100);
    resetOscPhase();
    switch (getUserReg({wave_reg})) {  
        '''

        play_fragment2 = '''

    wait(getUserReg({wait_reg}))
    setDIO(8);
    playWave(zeros(1000);
    waitWave();
    wait(100);
}}
'''

        code = ''.join(definition_fragments + play_fragment1 + play_fragment + play_fragment2)
        return code

    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['sequencer_id'] * 4, frequency)

    def set_offset_i(self, offset_i):
        assert (np.abs(offset_i) <= 0.5)
        self.awg.set_offset(self.params['ic'], offset_i)
        # self.awg.set_register(self.params['sequencer_id'], self.params['io'], np.asarray(int(offset_i*0xffffffff), dtype=np.uint32))

    def set_offset_q(self, offset_q):
        assert (np.abs(offset_q) <= 0.5)
        self.awg.set_offset(self.params['qc'], offset_q)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qo'], np.asarray(int(offset_q*0xffffffff), dtype=np.uint32))