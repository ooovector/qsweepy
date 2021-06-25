import numpy as np
import textwrap
import time


class IQ_RABISequence:
    def __init__(self, sequencer_id, awg, tail_length, readout_delay=0, awg_amp=1, pre_pulses=[],
                 use_modulation=True, length__reg=0, resudual__reg =1):
        """
        Sequence for rabi measurement.
        Send trigger for readout sequencer start.
        Produces random waveform.
        Parameters

        ----------
        awg
        tail_length[sec] - rising and falling edges length
        readout_delay[sec] - additional delay between Rabi falling edge and readout waveform. There is initial delay around 140ns exists
        due to DIO protocol.
        awg_amp - output amplitude of awg generator. Befor generation it will be multiplied by Range value!!!!!!
        pre_pulses - waveform you want to set befor worl sequence. Really you want to place here reset pulses.
        use_modulation - if you want to use awg's sine generators.
        length__reg - User register for fast length setting.
        resudual__reg - User register for fast length setting.

        """
        self.registors = {} #Dictionary for reserved User registors
        self.registors['length__reg'] = length__reg
        self.registors['resudual__reg'] = resudual__reg
        self.nsamp = 16
        self.awg = awg
        self.clock = float(self.awg._clock)
        self.params = dict(sequencer_id=sequencer_id, use_modulation=use_modulation,
                           tail_samp=int(tail_length*self.clock), n_samp=self.nsamp, awg_amp=awg_amp,
                           readout_delay=int(readout_delay*self.clock), length__reg=length__reg,
                           resudual__reg=resudual__reg, nco_id=sequencer_id*4,
                           ic=2 * sequencer_id, qc=sequencer_id* 2 + 1)

        self.pre_pulses = pre_pulses
        self.set_awg_amp(self.params['awg_amp'])
        self.set_phase_offset(90)
        # For exitation channels
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        # self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/strobe/slope' % self.params['sequencer_id'], 1)
        # We need to set DIO valid polarity as  Rise (0- none, 1 - low, 2 - high, 3 - both )
        # self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/valid/polarity' % self.params['sequencer_id'], 2)


    def zicode(self):
        # In this fragment we define all waveforms
        definition_fragments = []
        # In this fragment we define work sequence for play
        play_fragment = []
        # In play_fragment1 we wait for trigger from readout sequencer
        play_fragment1 = []
        # In play_fragment2 we set trigger for readout sequencer start to play readout waveform
        play_fragment2 = []

        if self.pre_pulses is not []:
            for _i in range(len(self.pre_pulses)):
                definition_fragments.append(self.pre_pulses[_i].get_definition_fragment(self.params['sequencer_id']))
                play_fragment.append(self.pre_pulses[_i].get_play_fragment(self.params['sequencer_id']))

        definition_fragments.append(textwrap.dedent('''
// Device settings
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id});

// Constant's and variables definition
const readout_delay = {readout_delay};
const tail_samp = {tail_samp};
const readout_samp = {n_samp_read};
const length__reg = {length__reg};
const resudual__reg = {resudual__reg};
var pause_length = getUserReg(length__reg);

// Waveform definition
wave tail_wave_0 = hann(2*tail_samp, 1);
        '''.format(**self.params)))

        # It's a trick how to set Rabi pulse length with precision corresponded to awg.clock (2.4GHz)
        for wave_length in range(1, self.params['n_samp'] + 1):
            if wave_length != self.params['n_samp']:
                if self.params['tail_samp'] > 2:
                    definition_fragments.append(textwrap.dedent('''
wave tail_rise = join(zeros(31-tail_samp%32), cut(tail_wave_0, 0, tail_samp));
wave tail_fall = cut(tail_wave_0, tail_samp, 2*tail_samp-1);
wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise, rect({wave_length}, 1));
'''.format(wave_length=wave_length, n_samp=self.params['n_samp'])))
                else:
                    definition_fragments.append(textwrap.dedent('''
wave tail_rise = zeros(32);
wave tail_fall = zeros(32);
wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise, rect({wave_length}, 1));
'''.format(wave_length=wave_length, n_samp=self.params['n_samp'])))
            else:
                definition_fragments.append(textwrap.dedent('''
wave w_{wave_length} = join(zeros(32-{wave_length}), ones({wave_length}));
'''.format(wave_length=wave_length, n_samp=self.params['n_samp'])))

        play_fragment.append(textwrap.dedent('''
    waitSineOscPhase(1);
    switch (getUserReg(resudual__reg)) {'''))
        for wave_length in range(1, self.params['n_samp'] + 1):
            play_fragment.append(textwrap.dedent('''
case {wave_length}:
    playWave({qubit_channel}, w_{wave_length})
    wait(pause_length)
    playWave({qubit_channel}, tail_fall);
    '''.format(qubit_channel=self.params['qubit_channel'] + 1, wave_length=wave_length)))

        # Sequencer for qubit's control are triggered by DIO protocol.
        # This DIO signal is controlled by readout sequencer.
        play_fragment1.append(textwrap.dedent('''
while (true) {{
    waitDIOTrigger();
    resetOscPhase();
    '''))
        # Then work sequence has done you need to send trigger for readout sequencer to start playWave.
        # There is initial delay between readout trigger and and readout waveform generation around 140 ns.
        play_fragment2.append(textwrap.dedent('''
    waitWave();
    playWave(zeros(readout_delay));
    setDIO(8);
    wait(50);
    setDIO(0);
}}
'''))
        code = ''.join(definition_fragments + play_fragment1 + play_fragment + play_fragment2)
        return code

    def add_register(self, reg_name, value):
        # reg_name is a str
        if reg_name in self.registors.keys():
            raise Exception('Error! Can not do this! Reg name ='+reg_name+' is already exist')
            #print('Error! Can not do this! Reg name ='+reg_name+' is already exist')
        elif value in self.registors.values():
            raise Exception('Error! Can not do this! User Register =' + str(value) + ' is already reserved')
            #print('Error! Can not do this! User Register =' + str(value) + ' is already reserved')
        else:
            self.registors[reg_name] = value

    def add_pre_pulse(self, pre_pulses):
        self.pre_pulses = pre_pulses

    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['nco_id'], frequency)

    def set_amplitude_i(self, amplitude_i):
        assert (np.abs(amplitude_i) <= 0.5)
        # self.awg.set_register(self.params['sequencer_id'], self.params['ia'], np.asarray(int(amplitude_i*0xffffffff), dtype=np.uint32))
        self.awg.set_sin_amplitude(self.params['ic'], 0, amplitude_i)
        #self.awg.set_wave_amplitude(self.params['ic'], self.params['ic'] % 2, amplitude_i)

    def set_amplitude_q(self, amplitude_q):
        assert (np.abs(amplitude_q) <= 0.5)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qa'], np.asarray(int(amplitude_q*0xffffffff), dtype=np.uint32))
        self.awg.set_sin_amplitude(self.params['qc'], 1, amplitude_q)
        #self.awg.set_wave_amplitude(self.params['qc'], self.params['ic'] % 2, amplitude_q)

    def set_awg_amp(self, awg_amp):
        # self.awg.set_amplitude(self.params['ic'], awg_amp)
        # self.awg.set_amplitude(self.params['qc'], awg_amp)
        self.awg.set_wave_amplitude(self.params['ic'], self.params['ic'] % 2, awg_amp)
        self.awg.set_wave_amplitude(self.params['qc'], self.params['ic'] % 2, awg_amp)

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

    def set_length(self, length):
        # length in seconds
        # self.trig_delay = delay
        # self.params['trig_delay'] = delay
        # print('setting delay to ',  length)
        pause_cycles = int(np.round(length * self.awg._clock))
        self.awg.set_register(self.params['sequencer_id'], self.params['length__reg'], pause_cycles // 8)
        self.awg.set_register(self.params['sequencer_id'], self.params['resudual__reg'], pause_cycles % 8)

    def start(self):
        self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
        # Мне не нравится. Заставить это работать так как ты хочешь будет очень сложно.
        if self.params['use_modulation']:
            self.awg.set_modulation(self.params['ic'], 1)
            self.awg.set_modulation(self.params['qc'], 1)
        else:
            self.awg.set_modulation(self.params['ic'], 0)
            self.awg.set_modulation(self.params['qc'], 0)
        self.awg.set_holder(self.params['ic'], 1)
        self.awg.set_holder(self.params['qc'], 1)
        self.awg.set_output(self.params['ic'], 1)
        self.awg.set_output(self.params['qc'], 1)
        self.awg.start_seq(self.params['sequencer_id'])

    def stop(self):
        self.awg.stop_seq(self.params['sequencer_id'])
        self.awg.set_output(self.params['ic'], 0)
        self.awg.set_output(self.params['qc'], 0)
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)