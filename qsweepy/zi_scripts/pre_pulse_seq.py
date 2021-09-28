import numpy as np
import textwrap
import time

class Offset:
    def __init__(self, channel, offset, IQ_modulation = False):
        self.channel = channel
        self.offset = offset
        self.IQ = IQ_modulation
    def is_offset(self):
        return True

class Prepulse:
    def __init__(self, channel, length, amp, length_tail, wait_after = 0):
        self.channel = channel
        self.length = length
        self.amp = amp
        self.length_tail = length_tail
        self.wait_after = wait_after
    def is_prepulse(self):
        return True

    def get_definition_fragment(self, clock, index =0):
        self.definition_fragments = ''''''
        if self.length_tail * clock > 2:
            self.definition_fragments += textwrap.dedent('''
const pre_samp{index} = {pre_samp_0};
setUserReg(15, pre_samp{index});
var variable_register15;
const pre_amp{index} = {pre_amp_0};
const pre_tail{index} = {pre_tail_0};
wave plato_wawe_{index} = rect(pre_samp{index} - 2*pre_tail{index}, pre_amp{index});
wave pre_tail_wave_{index} = hann(2*pre_tail{index}, pre_amp{index});
//wave pre_wawe_{index}=join(cut(pre_tail_wave_{index}, 0, pre_tail{index}), plato_wawe_{index}, cut(pre_tail_wave_{index}, pre_tail{index}, 2*pre_tail{index}-1));
wave pre_wawe_{index}=join(zeros(31-pre_tail{index}%32), cut(pre_tail_wave_{index}, 0, pre_tail{index}));
'''.format(index= index, pre_samp_0=int(self.length * clock/8), pre_tail_0=int(self.length_tail * clock), pre_amp_0=self.amp))
        else:
            self.definition_fragments += textwrap.dedent('''
const pre_samp{index} = {pre_samp_0};
const pre_amp{index} = {pre_amp_0};
wave plato_wawe_{index} = rect(pre_samp{index}, pre_amp{index});
wave pre_wawe_{index} = plato_wawe_{index};
'''.format(index = index, pre_samp_0=self.length * clock, pre_amp_0=self.amp))

        return self.definition_fragments

    def get_play_fragment(self, channel, clock, index = 0):
        self.play_fragment = ''''''
        if channel == 0:
            self.play_fragment += textwrap.dedent('''
//
    playWave(2, 0*pre_wawe_{index}, 1, pre_wawe_{index});
    wait(variable_register15);
    playWave(2, 0*flip(pre_wawe_{index}), 1, flip(pre_wawe_{index}));
    playWave(zeros({wait_after}));
'''.format(index = index, wait_after=int(self.wait_after * clock)))
        elif channel == 1:
            self.play_fragment += textwrap.dedent('''
//
    playWave(2, pre_wawe_{index}, 1, 0*pre_wawe_{index});
    wait(variable_register15);
    playWave(2, flip(pre_wawe_{index}), 1, flip(0*pre_wawe_{index}));
    playWave(zeros({wait_after}));
'''.format(index=index, wait_after=int(self.wait_after * clock)))

        return self.play_fragment

class PrepulseSetter:
    def __init__(self, device, offsets, pre_pulses):
        """
        Pre pulses settings.
        Parameters

        ----------
        offsets
        pre_pulses

        """

        self.device = device
        self.seq_in_use = []
        self.offsets = offsets
        self.pre_pulses = pre_pulses
        for offset in self.offsets:
            if hasattr(self.device.awg_channels[offset.channel].parent, 'sequencer_id'):
                seq_id = device.awg_channels[offset.channel].parent.sequencer_id
            else:
                seq_id = self.device.awg_channels[offset.channel].parent.channel//2
            if seq_id not in self.seq_in_use:
                self.seq_in_use.append(seq_id)
        for pre_pulse in self.pre_pulses:
            if hasattr(self.device.awg_channels[pre_pulse.channel].parent, 'sequencer_id'):
                seq_id = self.device.awg_channels[pre_pulse.channel].parent.sequencer_id
            else:
                seq_id = self.device.awg_channels[pre_pulse.channel].parent.channel//2
            if seq_id not in self.seq_in_use:
                self.seq_in_use.append(seq_id)

    def set_seq_offsets(self, sequencer):
        for offset in self.offsets:
            seq_id = self.device.awg_channels[offset.channel].parent.channel // 2
            if seq_id == sequencer.params['sequencer_id']:
                sequencer.set_offset(self.device.awg_channels[offset.channel].parent.channel, offset.offset)

    def set_seq_prepulses(self, sequencer):

        definition, play_fragment = self.get_seq_prepulses(sequencer.params['sequencer_id'], sequencer.awg)
        sequencer.add_pre_pulse(definition, play_fragment)
        return



    def get_seq_prepulses(self, sequencer_id, awg):
        definition_fragment = ''''''
        play_fragment = ''''''
        index = 0
        pre_pulses = []
        for pre_pulse in self.pre_pulses:
            if hasattr(self.device.awg_channels[pre_pulse.channel].parent, 'sequencer_id'):
                seq_id = self.device.awg_channels[pre_pulse.channel].parent.sequencer_id
                raise Exception('Error! Can not set pre_pulse in sequencer'+str(seq_id)+' because it is IQ!')
            else:
                seq_id = self.device.awg_channels[pre_pulse.channel].parent.channel//2
            if seq_id == sequencer_id:
                definition_fragment += pre_pulse.get_definition_fragment(awg._clock, index)
                index += 1
                pre_pulses.append(pre_pulse)
        play_fragment += self.get_parallel_play(pre_pulses, awg)

        return definition_fragment, play_fragment


    def get_parallel_play(self, pre_pulses, awg):
        play_fragment = ''''''
        if len(pre_pulses) == 1:
            channel_id = self.device.awg_channels[pre_pulses[0].channel].parent.channel % 2
            play_fragment += pre_pulses[0].get_play_fragment(channel_id, awg._clock)
        elif len(pre_pulses) == 2:
            wait_after = max(pre_pulses[0].wait_after, pre_pulses[1].wait_after)
            channel_id = self.device.awg_channels[pre_pulses[0].channel].parent.channel % 2
            if channel_id == 0:
                index1 = 1
                index2 = 0
            else:
                index2 = 1
                index1 = 0
            play_fragment += textwrap.dedent('''
//
    playWave(2, pre_wawe_{index1}, 1, pre_wawe_{index2});
    playWave(zeros({wait_after}));
    resetOscPhase();
    setSinePhase(0, 0);
    setSinePhase(1, 90);
    wait(10);
'''.format(index1=index1, index2=index2, wait_after=int(wait_after * awg._clock)))
        return play_fragment



class SIMPLESequence:
    def __init__(self, sequencer_id, awg, readout_delay=0, awg_amp=1, pre_pulses = [],
                 use_modulation=True, var_reg0=0, var_reg1 =1, var_reg2 =2, var_reg3 =3, control=False, is_iq = False):
        """
        Pre pulses settings.
        Parameters

        ----------
        offsets
        pre_pulses

        """
        self.control = control
        self.registors = {}  # Dictionary for reserved User registors
        self.registors['var_reg0'] = var_reg0
        self.registors['var_reg1'] = var_reg1
        self.awg = awg
        self.clock = self.awg._clock
        frequency = self.awg.get_frequency(sequencer_id * 4)
        self.params = dict(sequencer_id=int(sequencer_id), qubit_channel=0,
                           offset_channel=1, use_modulation=use_modulation, is_iq=is_iq,
                           awg_amp=awg_amp, readout_delay=int(readout_delay * self.clock),
                           var_reg0=int(var_reg0), var_reg1=int(var_reg1), var_reg2=int(var_reg2),
                           var_reg3=int(var_reg3), var_reg15=int(15),
                           nco_id=sequencer_id * 4, nco_control_id=sequencer_id * 4 + 1,
                           frequency=frequency, control_frequency=0,
                           ic=2 * sequencer_id, qc=sequencer_id * 2 + 1)
        self.pre_pulses = pre_pulses
        self.definition_pre_pulses = '''
// Pre pulses definition'''
        self.play_pre_pulses = '''
// Pre pulses play'''
        # Initial settings
        self.set_awg_amp(self.params['awg_amp'])
        # You need to set phase in offset channel equal to 90 degrees
        # In other case it doesn't work
        self.set_phase_q(90)
        self.definition_fragments = textwrap.dedent('''
// Device settings
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id}+1);
cvar control_frequency = {control_frequency};
setDouble('oscs/{nco_id}/freq', {frequency});
setDouble('oscs/{nco_control_id}/freq', control_frequency);

// Constant's and variables definition
const readout_delay = {readout_delay};
const var_reg0 = {var_reg0};
const var_reg1 = {var_reg1};
const var_reg2 = {var_reg2};
const var_reg3 = {var_reg3};
var wave_ind=0;
var variable_register0 = getUserReg(var_reg0);
var variable_register1 = getUserReg(var_reg1);
var variable_register2 = getUserReg(var_reg2);
var variable_register3 = getUserReg(var_reg3);
'''.format(**self.params))
        self.play_fragment = '''
    '''



    def zicode(self):
        # In this fragment we collect pre_pulses from in device.pre_pulses
        definition_pre_pulses = self.definition_pre_pulses
        # In this fragment we define all waveforms
        definition_fragments = self.definition_fragments
        # In this fragment we define work sequence for play
        play_fragment = self.play_fragment
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
                definition_pre_pulses += self.pre_pulses[_i].get_definition_fragment(self.clock, _i)
                play_pre_pulses += self.pre_pulses[_i].get_play_fragment(self.params['qubit_channel'], _i)
        play_pre_pulses+=textwrap.dedent('''
//    
    setDouble('oscs/{nco_control_id}/freq', control_frequency);
    resetOscPhase();
    setSinePhase(0, 0);
    setSinePhase(1, 90);
    wait(10);
    '''.format(**self.params))

        play_fragment1 += textwrap.dedent('''
setDIO(0); 
while (true) {{''')
        play_fragment1 += textwrap.dedent('''
//    
    // Wait trigger and reset
    //
    waitDIOTrigger();
    setDouble('oscs/{nco_control_id}/freq', {control_frequency});
    variable_register0 = getUserReg(var_reg0);
    variable_register1 = getUserReg(var_reg1);
    variable_register2 = getUserReg(var_reg2);
    variable_register3 = getUserReg(var_reg3);
    variable_register15 = getUserReg(15);
    setPRNGSeed(variable_register1);
    resetOscPhase();
    setSinePhase(0, 0);
    setSinePhase(1, 90);
    '''.format(**self.params))
        # Then work sequence has done you need to send trigger for readout sequencer to start playWave.
        # There is initial delay between readout trigger and and readout waveform generation around 140 ns.
        if self.control:
            play_fragment2 += textwrap.dedent('''
//
    // Send trigger for readout_channel to start waveform generation
    waitWave();
    playZero(readout_delay);
    setDIO(8);
    wait(10);
    setDIO(0);
}}
''')
        else:
            play_fragment2 += textwrap.dedent('''
//
    // Send trigger for readout_channel to start waveform generation
    //waitWave();
    //playZero(readout_delay);
    //setDIO(8);
    //wait(50);
    //setDIO(0);
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

    def clear_pulse_sequence(self):
        self.definition_fragments = textwrap.dedent('''
// Device settings
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id}+1);
cvar control_frequency = {control_frequency};
setDouble('oscs/{nco_id}/freq', {frequency});
setDouble('oscs/{nco_control_id}/freq', control_frequency);


// Constant's and variables definition
const readout_delay = {readout_delay};
const var_reg0 = {var_reg0};
const var_reg1 = {var_reg1};
const var_reg2 = {var_reg2};
const var_reg3 = {var_reg3};
var wave_ind=0;
var variable_register0 = getUserReg(var_reg0);
var variable_register1 = getUserReg(var_reg1);
var variable_register2 = getUserReg(var_reg2);
var variable_register3 = getUserReg(var_reg3);
'''.format(**self.params))
        self.play_fragment = '''
    '''

    def add_pre_pulse(self, definition, play_fragment):
        # self.pre_pulses.append(pre_pulse)
        self.definition_pre_pulses += definition
        self.play_pre_pulses += play_fragment

    def add_definition_fragment(self, definition_fragment):
        if definition_fragment in self.definition_fragments:
            Warning('The same definition fragment has already been added to the program')
        else:
            self.definition_fragments +=definition_fragment

    def add_play_fragment(self, play_fragment):
        self.play_fragment += play_fragment

    def set_frequency(self, frequency):
        self.params['frequency'] = frequency

        self.awg.set_frequency(self.params['nco_id'], frequency)
        self.awg.set_frequency(self.params['nco_id']+1, self.params['control_frequency'])

    def set_awg_amp(self, awg_amp):
        if self.params['is_iq']:
            self.awg.set_amplitude(self.params['ic'], awg_amp)
            self.awg.set_amplitude(self.params['qc'], awg_amp)
        else:
            self.awg.set_wave_amplitude(self.params['ic'], 0, awg_amp)
            self.awg.set_wave_amplitude(self.params['ic'], 1, 1)
            self.awg.set_wave_amplitude(self.params['qc'], 0, 1)
            self.awg.set_wave_amplitude(self.params['qc'], 1, 1)

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

    def set_offset(self, channel, offset):
        #assert (np.abs(offset) <= 0.5)
        self.awg.set_offset(channel, offset)

    def set_prepulse_delay(self, delay):
        delay_cycles = int(np.round(delay * self.awg._clock/8))
        self.awg.set_register(self.params['sequencer_id'], self.params['var_reg15'], delay_cycles)

    def set_length(self, length):
        # length in seconds
        #self.trig_delay = delay
        #self.params['trig_delay'] = delay
        #print('setting delay to ',  length)
        pause_cycles = int(np.round(length*self.awg._clock))
        if (pause_cycles//8)!=0:
            if (pause_cycles % 8)!=0:
                self.awg.set_register(self.params['sequencer_id'], self.params['var_reg0'], pause_cycles // 8)
                self.awg.set_register(self.params['sequencer_id'], self.params['var_reg1'], pause_cycles % 8)
            else:
                self.awg.set_register(self.params['sequencer_id'], self.params['var_reg0'], pause_cycles // 8-1)
                self.awg.set_register(self.params['sequencer_id'], self.params['var_reg1'], 8)

        else:
            self.awg.set_register(self.params['sequencer_id'], self.params['var_reg0'], pause_cycles//8)
            self.awg.set_register(self.params['sequencer_id'], self.params['var_reg1'], pause_cycles % 8)

    def set_phase(self, phase):

        self.awg.set_register(self.params['sequencer_id'], self.params['var_reg2'], phase)

    def start(self):
        #self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)
        # Мне не нравится. Заставить это работать так как ты хочешь будет очень сложно.
        if self.params['use_modulation']:
            if self.params['is_iq']:
                self.awg.set_modulation(self.params['ic'], 1)
                self.awg.set_modulation(self.params['qc'], 2)
            else:
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




