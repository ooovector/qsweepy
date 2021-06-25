import numpy as np
import textwrap
import time


class RABISequence:
    def __init__(self, sequencer_id, awg, tail_length, readout_delay=0, awg_amp=1, pre_pulses = [],
                 use_modulation=True, length_reg=0, residual_reg =1):
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
        self.control_frequency = 0
        self.registors = {} #Dictionary for reserved User registors
        self.registors['length_reg'] = length_reg
        self.registors['resudual_reg'] = residual_reg
        self.nsamp = 16
        self.awg = awg
        self.clock = float(self.awg._clock)
        self.params = dict(sequencer_id=sequencer_id, qubit_channel=0,
                           offset_channel=1, use_modulation=use_modulation,
                           tail_samp=int(tail_length*self.clock), n_samp=self.nsamp,
                           awg_amp=awg_amp, readout_delay=int(readout_delay*self.clock),
                           length_reg=length_reg, residual_reg=residual_reg, nco_id=sequencer_id * 4,
                           ic=2 * sequencer_id, qc=sequencer_id*2 + 1)

        self.pre_pulses = pre_pulses
        self.definition_pre_pulses = '''
// Pre pulses definition'''
        self.play_pre_pulses = '''
// Pre pulses play'''
        # Initial settings
        self.set_awg_amp(self.params['awg_amp'])
        # You need to set phase in offset channel equal to 90 degrees
        # In other case it doesn't work
        self.set_phase_control(90)
        #self.set_frequency_control(0)

        # For exitation channels
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/strobe/slope' % self.params['sequencer_id'], 1)
        # We need to set DIO valid polarity as  Rise (0- none, 1 - low, 2 - high, 3 - both )
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/valid/polarity' % self.params['sequencer_id'], 2)

    def zicode(self):
        # In this fragment we collect pre_pulses from in device.pre_pulses
        definition_pre_pulses = self.definition_pre_pulses
        # In this fragment we define all waveforms
        definition_fragments = '''
//'''
        # In this fragment we define work sequence for play
        play_fragment = '''
//'''
        # In play_pre_pulses we play pre_pulses defined in definition_pre_pulses
        play_pre_pulses = self.play_pre_pulses
        # In play_fragment1 we wait for trigger from readout sequencer
        play_fragment1 = '''
//'''
        # In play_fragment2 we set trigger for readout sequencer start to play readout waveform
        play_fragment2 = '''
//'''
        if self.pre_pulses is not []:
            for _i in range(len(self.pre_pulses)):
                definition_fragments += self.pre_pulses[_i].get_definition_fragment(self.clock, _i)
                play_fragment += self.pre_pulses[_i].get_play_fragment(self.params['qubit_channel'], _i)

        definition_fragments += textwrap.dedent('''
// Device settings
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id}+1);

// Constant's and variables definition
const readout_delay = {readout_delay};
const tail_samp = {tail_samp};
const length__reg = {length_reg};
const resudual__reg = {resudual_reg};
var pause_length = getUserReg(length_reg);


// Waveform definition
wave tail_wave_0 = hann(2*tail_samp, 1);
'''.format(**self.params))

        #It's a trick how to set Rabi pulse length with precision corresponded to awg.clock (2.4GHz)
        for wave_length in range(1, self.params['n_samp']+1):
            if wave_length != self.params['n_samp']:
                if self.params['tail_samp'] > 2:
                    definition_fragments += textwrap.dedent('''
wave tail_rise_{wave_length} = join(zeros(31-tail_samp%32), cut(tail_wave_0, 0, tail_samp));
wave tail_fall_{wave_length} = cut(tail_wave_0, tail_samp, 2*tail_samp-1);
wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));
                '''.format(wave_length=wave_length, n_samp=self.params['n_samp']))
                else:
                    definition_fragments += textwrap.dedent('''
wave tail_rise_{wave_length} = zeros(32);
wave tail_fall_{wave_length} = zeros(32);
//wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));
wave w_{wave_length} = join(zeros(32-{wave_length}), rect({wave_length}, 1));
'''.format(wave_length=wave_length, n_samp=self.params['n_samp']))
            else:
                definition_fragments += textwrap.dedent('''
wave w_{wave_length} = join(zeros(32-{wave_length}), ones({wave_length}));
'''.format(wave_length=wave_length, n_samp=self.params['n_samp']))

        play_fragment += textwrap.dedent('''
//
    //waitSineOscPhase(1);
    resetOscPhase();
    wait(10);
    switch (getUserReg(resudual_reg)) {''')
        for wave_length in range(1, self.params['n_samp']+1):
            if wave_length != self.params['n_samp']:
                play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave({qubit_channel}, w_{wave_length});
            wait(pause_length);
            playWave({qubit_channel}, tail_fall_{wave_length});
            '''.format(qubit_channel=self.params['qubit_channel']+1, wave_length = wave_length))
            else:
                play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave({qubit_channel}, w_{wave_length});
            wait(pause_length);
            '''.format(qubit_channel=self.params['qubit_channel'] + 1, wave_length=wave_length))


        # Sequencer for qubit's control are triggered by DIO protocol.
        # This DIO signal is controlled by readout sequencer.
        play_fragment1 += textwrap.dedent( '''
while (true) {{
    // Wait trigger and reset
    pause_length = getUserReg(length_reg);
    //switch_value = getUserReg(resudual_reg); 
    waitDIOTrigger();
    //setDIO(0);
    //resetOscPhase();
    ''')
        # Then work sequence has done you need to send trigger for readout sequencer to start playWave.
        # There is initial delay between readout trigger and and readout waveform generation around 140 ns.
        play_fragment2 += textwrap.dedent( '''
//    
    }
    // Send trigger for readout_channel to start waveform generation
    waitWave();
    playWave(zeros(readout_delay));
    setDIO(8);
    wait(10);
    setDIO(0);
}}
''')
        code = ''.join(definition_pre_pulses + definition_fragments + play_fragment1 + play_pre_pulses + play_fragment + play_fragment2)
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

    def add_pre_pulse(self, definition, play_fragment):
        #self.pre_pulses.append(pre_pulse)
        self.definition_pre_pulses += definition
        self.play_pre_pulses += play_fragment

    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['nco_id'], frequency)
        self.awg.set_frequency(self.params['nco_id']+1, self.control_frequency)

    def set_awg_amp(self, awg_amp):
        #self.awg.set_amplitude(self.params['qubit_channel'], awg_amp)
        #self.awg.set_amplitude(self.params['offset_channel'], awg_amp)
        self.awg.set_wave_amplitude(self.params['ic'], 0, awg_amp)
        #self.awg.set_wave_amplitude(self.params['qubit_channel'], 1, awg_amp)

    def set_phase_qubit(self, phase_qubit):
        # phase_qubit in degrees
        # self.awg.set_register(self.params['sequencer_id'], self.params['ip'], np.asarray(int(phase_i * 0xffffffff / 360.0), dtype=np.uint32))
        self.awg.set_sin_phase(self.params['ic'], phase_qubit)

    def set_phase_control(self, phase_offset):
        # phase_offset in degrees
        # self.awg.set_register(self.params['sequencer_id'], self.params['qp'], np.asarray(int(phase_q * 0xffffffff / 360.0), dtype=np.uint32))
        self.awg.set_sin_phase(self.params['qc'], phase_offset)


    def set_offset(self, channel, offset):
        #assert (np.abs(offset) <= 0.5)
        self.awg.set_offset(channel, offset)

    def set_length(self, length):
        # length in seconds
        #self.trig_delay = delay
        #self.params['trig_delay'] = delay
        #print('setting delay to ',  length)
        #self.awg.stop_seq(self.params['sequencer_id'])
        pause_cycles = int(np.round(length*self.awg._clock))
        self.awg.set_register(self.params['sequencer_id'], self.params['length_reg'], pause_cycles//8)
        self.awg.set_register(self.params['sequencer_id'], self.params['resudual_reg'], 1+pause_cycles % 8)
        #self.awg.start_seq(self.params['sequencer_id'])

    def start(self):
        #self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
        # Мне не нравится. Заставить это работать так как ты хочешь будет очень сложно.
        if self.params['use_modulation']:
            self.awg.set_modulation(self.params['ic'], 3)
            self.awg.set_modulation(self.params['qc'], 0)
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