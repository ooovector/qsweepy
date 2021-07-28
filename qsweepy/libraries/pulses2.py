#from scipy.signal import gaussian
# from scipy.signal import tukey
# from scipy.signal import hann
import numpy as np
import textwrap
import copy


class offset:
    def __init__(self, channel, length, offset):
        self.offset = offset

    def is_offset(self):
        return True


class prepulse:
    def __init__(self, channel, length, amp, length_tail):
        self.length = length
        self.amp = amp
        self.length_tail = length_tail

    def is_prepulse(self):
        return True


class Pulses:
    def __init__(self, channels={}):
        self.channels = channels
        self.settings = {}

    def p(self, channel, length, pulse_type=None, *params):
        pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        if channel:
            pulses[channel] = pulse_type(channel, length, *params)
        return pulses

    def gauss_hd(self, channel, length, amp_x, sigma, alpha=0., phase = 0, fast_control = False):
        #gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
        #                 sigma * self.channels[channel].get_clock())
        #gauss -= gauss[0]
        #gauss /= np.max(gauss)
        #gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        definition_fragment = ''''''
        play_fragment = ''''''
        ex_channel = self.channels[channel]
        length_samp = round((length) * ex_channel.get_clock()/16)*16
        sigma_samp = int((sigma) * ex_channel.get_clock())
        #position_samp = round((length)*ex_channel.get_clock())/2
        position_samp = (length_samp-1)/ 2
        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_{length_samp}
wave gauss_hd_{length_samp}_{sigma_samp} = gauss({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
gauss_hd_{length_samp}_{sigma_samp} = gauss_hd_{length_samp}_{sigma_samp} - gauss_hd_{length_samp}_{sigma_samp}[0];
wave drag_hd_{length_samp}_{sigma_samp} = {alpha}*drag({length_samp}, {amp_x}, {position_samp}, {sigma_samp});'''.format(length_samp=length_samp, sigma_samp=sigma_samp, position_samp=position_samp, amp_x=1, alpha = alpha))
        if fast_control == 'phase_correction':

            play_fragment += '''
//
    //for (i = 0; i < variable_register0; i = i + 1) {
    //for (j = 0; j < variable_register1; j = j + 1) {
    repeat(variable_register0){
        repeat(variable_register1){'''

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();
            playWave(1, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=phase*360/2/np.pi))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();
            playWave(1, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=phase*360/2/np.pi))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
            playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();
            playWave(2, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=phase*360/2/np.pi))
            play_fragment += '''
//
    }}'''

        elif fast_control == 'Benchmarking':
            if ex_channel.is_iq():
                definition_fragment += textwrap.dedent('''
        assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, wave_ind);
        wave_ind=wave_ind+1;'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                       ampQ=np.imag(amp_x), alpha=alpha, phase=phase * 360 / 2 / np.pi))
                _ct_entry = {'index': 1,
                             'waveform': {'index': wave_ind},
                             'amplitude0': {'value': 1},
                             'phase0': {'value': phase * 360 / 2 / np.pi, 'increment': True},
                             'amplitude1': {'value': 1},
                             'phase1': {'value': phase * 360 / 2 / np.pi, 'increment': True}
                             }

            else:
                if control_channel_id == 0:
                    definition_fragment += textwrap.dedent('''
        assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, wave_ind);
        wave_ind=wave_ind+1;'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                       ampQ=np.imag(amp_x), alpha=alpha, phase=phase * 360 / 2 / np.pi))

                elif control_channel_id == 1:
                    definition_fragment += textwrap.dedent('''
        assignWaveIndex(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, wave_ind);
        wave_ind=wave_ind+1;'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                       ampQ=np.imag(amp_x), alpha=alpha, phase=phase * 360 / 2 / np.pi))

        elif fast_control:
            definition_fragment +='''
//var i = 0;
//var j = 0;'''
            play_fragment += '''
//
    //for (i = 0; i < variable_register0; i = i + 1) {
        //for (j = 0; j < variable_register1; j = j + 1) {
    repeat(variable_register0){
        repeat(variable_register1){'''

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha = alpha, phase=phase*360/2/np.pi))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha=alpha, phase=phase*360/2/np.pi))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
            playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha=alpha, phase=phase*360/2/np.pi))
            play_fragment +='''
//
    }}'''

        else:
            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(0,{phase});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha = alpha, phase=phase*360/2/np.pi))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(0,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha = alpha, phase=phase*360/2/np.pi))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha = alpha, phase=phase*360/2/np.pi))

        return definition_fragment, play_fragment

    def rect_cos(self, channel, length, amp, length_tail, fast_control=False):
        # this part is necessary for accurate length setter and fast control
        # for example: rabi sequence
        # for fast control you can use "variable_register" defined in the beginning of the sequencer play program
        n_samp = 8
        definition_fragment = ''''''
        play_fragment = ''''''
        var_reg0 = 0
        var_reg1 = 1
        ex_channel = self.channels[channel]
        pause_cycles = int(np.round((length - 2*length_tail) * ex_channel.get_clock()))
        tail_samp = int(length_tail * ex_channel.get_clock())
        length_samp = int((length - 2*length_tail) * ex_channel.get_clock())
        if ex_channel.is_iq():
            control_seq_id = ex_channel.parent.sequencer_id
            control_channel_id = None
            if fast_control:
                ex_channel.parent.awg.set_amplitude(2*control_seq_id, 1)
                ex_channel.parent.awg.set_amplitude(2*control_seq_id+1, 1)
        else:
            control_seq_id = ex_channel.channel // 2
            control_channel_id = ex_channel.channel % 2
            if fast_control:
                ex_channel.parent.awg.set_wave_amplitude(2*control_seq_id, 0, 1)
                ex_channel.parent.awg.set_wave_amplitude(2 * control_seq_id, 1, 1)
                ex_channel.parent.awg.set_wave_amplitude(2 * control_seq_id + 1, 0, 1)
                ex_channel.parent.awg.set_wave_amplitude(2 * control_seq_id + 1, 1, 1)

        if fast_control:
            ex_channel.parent.awg.set_register(control_seq_id, var_reg0, pause_cycles // 8)
            ex_channel.parent.awg.set_register(control_seq_id, var_reg1, pause_cycles % 8)
            definition_fragment += textwrap.dedent('''
const tail_samp = {tail_samp};

// Waveform definition
wave tail_wave_0 = hann(2*tail_samp, 1);'''.format(tail_samp=tail_samp))
            # It's a trick how to set Rabi pulse length with precision corresponded to awg.clock (2.4GHz)
            for wave_length in range(0, n_samp+1):
                if tail_samp > 2:
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length} = join(zeros(31-tail_samp%32), cut(tail_wave_0, 0, tail_samp));
wave tail_fall_{wave_length} = cut(tail_wave_0, tail_samp, 2*tail_samp-1);
wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));'''.format(wave_length=wave_length, n_samp=n_samp))
                else:
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length} = zeros(32);
wave tail_fall_{wave_length} = zeros(32);
//wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));
wave w_{wave_length} = join(zeros(32-{wave_length}), rect({wave_length}, 1));'''.format(wave_length=wave_length, n_samp=n_samp))

            play_fragment += textwrap.dedent('''
//
    //waitSineOscPhase(1);
    //resetOscPhase();
    //wait(10);
    switch (variable_register1) {''')
            for wave_length in range(0, n_samp+1):
                if ex_channel.is_iq():
                    play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(1, {ampI}*w_{wave_length}, 2, {ampQ}*w_{wave_length});
            wait(variable_register0);
            playWave(1, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length});'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
                    if wave_length==n_samp:
                        play_fragment += '''
//
        }'''
                else:
                    if control_channel_id == 0:
                        play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(1, {ampI}*w_{wave_length}, 1, {ampQ}*w_{wave_length});
            wait(variable_register0);
            playWave(1, {ampI}*tail_fall_{wave_length}, 1, {ampQ}*tail_fall_{wave_length});'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
                        if wave_length == n_samp:
                            play_fragment +='''
//
        }'''
                    elif control_channel_id == 1:
                        play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(2, {ampI}*w_{wave_length}, 2, {ampQ}*w_{wave_length});
            wait(variable_register0);
            playWave(2, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length});'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
                        if wave_length == n_samp:
                            play_fragment += '''
//
        }'''

        else:
            if tail_samp > 2:
                definition_fragment += textwrap.dedent('''
// Waveform definition rect_cos_{length_samp}
wave tail_rect_cos_{length_samp} = hann(2*{tail_samp}, {amp});
wave rise_rect_cos_{length_samp} = cut(tail_rect_cos_{length_samp}, 0, tail_samp);
wave fall_rect_cos_{length_samp} = cut(tail_rect_cos_{length_samp}, {tail_samp}, 2*{tail_samp}-1);
wave rect_cos_{length_samp} = join(rise_rect_cos_{length_samp}, rect(length_samp, {amp})), fall_rect_cos_{length_samp});
'''.format(tail_samp=tail_samp, length_samp=length_samp, amp=1))
            else:
                definition_fragment += textwrap.dedent('''
// Waveform definition rect_cos_{length_samp}
wave rect_cos_{length_samp} = rect({length_samp}, {amp});'''.format(tail_samp=tail_samp, length_samp=length_samp, amp=1))

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
            else:
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 1, {ampQ}*rect_cos_{length_samp});'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))


        return definition_fragment, play_fragment

    def pause(self, channel, length, fast_control = False):
        if length==0 and fast_control is False:
            definition_fragment, play_fragment = self.pause2(channel, length)
        else:
            definition_fragment, play_fragment = self.rect_cos(channel, length, amp=0, length_tail=0, fast_control=fast_control)
        return definition_fragment, play_fragment

    def pause2(self, channel, length):
        ex_channel = self.channels[channel]
        length_samp = int(np.round(length * ex_channel.get_clock()))
        definition_fragment = ''''''
        play_fragment = ''''''
        play_fragment += textwrap.dedent('''
//
    playZero({length_samp});
    waitWave();'''.format(length_samp=length_samp))
        return definition_fragment, play_fragment

    def virtual_z(self, channel, length, phase, fast_control = False, resolution = 6):
        '''
        Parameters
        :param channel: channel name
        :param phase: virtual phase to be added to qubit pulses [phase] = grad
        :param fast_control: if true you can set phase from register
        :return: definition_fragment, play_fragment for zurich sequencer
        '''
        definition_fragment = ''''''
        play_fragment = ''''''
        ex_channel = self.channels[channel]
        if fast_control == 'quasi-binary':
            definition_fragment = '''
cvar resolution = {resolution};
'''.format(resolution=resolution)
            for bit in range(resolution):
                bitval = 1 << bit
                play_fragment += textwrap.dedent('''
//
    if (phase_variable & {bitval}) {{
        incrementSinePhase(0, {increment});
        //waitWave();'''.format(bitval=bitval, increment=bitval / (1 << resolution) * 360.0))
                if ex_channel.is_iq():
                    play_fragment += textwrap.dedent('''
//
        incrementSinePhase(1, {increment});
        //waitWave();'''.format(bitval=bitval, increment=bitval / (1 << resolution) * 360.0))
                play_fragment += '''
        wait(1);
//
    } else {
        wait(4);
    }'''

        elif fast_control:
            definition_fragment += '''
//
var i;
var j;'''
            if ex_channel.is_iq():
                play_fragment += '''
//
    i=0;
    for (i=0; i < variable_register0; i = i +1) {'''
                play_fragment += textwrap.dedent('''
//
        incrementSinePhase(0,{phase1});
        incrementSinePhase(1,{phase1});
        //playZero(0);
        waitWave();'''.format(phase1=8 * phase))
                play_fragment += '''
//
    }'''
                play_fragment += textwrap.dedent('''
//    
    incrementSinePhase(0,{phase2});
    incrementSinePhase(1,{phase2});
    //playZero(0);
    waitWave();'''.format(phase2=64 * phase))
            else:
                play_fragment +='''
//
    i=0;
    for (i=0; i < variable_register0; i = i +1) {'''
                play_fragment += textwrap.dedent('''
//
        incrementSinePhase(0,{phase1});
        //playZero(0);
        waitWave();'''.format(phase1=8*phase))
                play_fragment += '''
//
    }'''
                play_fragment += textwrap.dedent('''
//    
    incrementSinePhase(0,{phase2});
    waitWave();'''.format(phase2=64*phase))
        else:
            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    incrementSinePhase(0,{phase});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(phase=phase))
            else:
                play_fragment += textwrap.dedent('''
//
    incrementSinePhase({channel}, {phase});
    waitWave();'''.format(channel=ex_channel.channel % 2, phase=phase))

        return definition_fragment, play_fragment

    def pmulti(self, device, length, *params):
        try:
            length_set = length[0]
            fast_control = True
        except:
            length_set = length
            fast_control = False

        pulses = {}
        for seq_id in device.pre_pulses.seq_in_use:
            for channel_name, channel in self.channels.items():
                ex_channel = self.channels[channel_name]
                if ex_channel.is_iq():
                    if seq_id ==ex_channel.parent.sequencer_id:
                        pulses[seq_id] = self.pause(channel_name, length_set, fast_control)
                        break
                else:
                    if seq_id == ex_channel.channel // 2:
                        pulses[seq_id] = self.pause(channel_name, length_set, fast_control)
                        break
        #pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        #pulses = {seq_id: self.pause(seq_id, length) for seq_id in device.pre_pulses.seq_in_use}
        control_seq_id = None
        for pulse in params:
            channel = pulse[0]
            ex_channel = self.channels[channel]
            if ex_channel.is_iq():
                control_seq_id = ex_channel.parent.sequencer_id
            else:
                control_seq_id = ex_channel.channel // 2
            # print ('Setting multipulse: \npulse:', pulse[1], 'channel:', channel, 'length:', length, 'other args:', pulse[2:])
            pulses[control_seq_id] = pulse[1](channel, length_set, *pulse[2:])
        return [pulses, [control_seq_id]]

    def sin(self, channel, length, amp, freq, fast_control=False):
        definition_fragment=''''''
        play_fragment=''''''
        ex_channel = self.channels[channel]
        nrOfsampl = int(length * ex_channel.get_clock())
        nrOfPeriods = int(length * np.abs(freq))
        definition_fragment += textwrap.dedent('''
wave sine_wawe_{samples} = sine({samples}, 1, 0, {nrOfPeriods});'''.format(name=channel, samples=nrOfsampl, nrOfPeriods=nrOfPeriods))

        if ex_channel.is_iq():
            calib_dc = ex_channel.parent.calib_dc()
            calib_rf = ex_channel.parent.calib_rf(ex_channel)
            awg_channel = ex_channel.parent.sequencer_id
            ex_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
            ex_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
            ex_channel.parent.awg.set_offset(2 * awg_channel, np.real(calib_dc['dc']))
            ex_channel.parent.awg.set_offset(2 * awg_channel + 1, np.imag(calib_dc['dc']))
            play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*sine_wawe_{samples}, 2, {ampQ}*sine_wawe_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp)*np.abs(calib_rf['I']), ampQ=np.imag(amp)*np.abs(calib_rf['Q'])))
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*sine_wawe_{samples}, 1, {ampQ}*sine_wawe_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp), ampQ=np.imag(amp)))
            elif control_channel_id == 1:
                play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*sine_wawe_{samples}, 2, {ampQ}*sine_wawe_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp), ampQ=np.imag(amp)))

        return definition_fragment, play_fragment

    def parallel(self, *pulse_sequences):
        # pulse_sequences = [pulses,seq_id] -
        # list of lists where first element is pulses (from pg.pmulty) and second element is control_channel
        # pulses - dict
        # seq_id - control sequencer id for pulse_sequence
        # return new list merged_pulses with pulses for each channel

        merged_pulses = {}
        merged_seq_id = []
        merged_pulses = copy.copy(pulse_sequences[0][0])

        for _sequences in pulse_sequences:
            merged_pulses[_sequences[1][0]] = _sequences[0][_sequences[1][0]]
            merged_seq_id.append(_sequences[1][0])

        return [merged_pulses, merged_seq_id]


    def readout_rect(self, channel, length, amplitude):
        # re_channel = device.awg_channels[channel]
        re_channel = self.channels[channel]
        calib_dc = re_channel.parent.calib_dc()
        calib_rf = re_channel.parent.calib_rf(re_channel)
        nrOfsampl = int(length * re_channel.get_clock())
        nrOfPeriods = int(length * np.abs(re_channel.get_if()))
        awg_channel = re_channel.parent.sequencer_id
        re_channel.parent.awg.set_amplitude(2 * awg_channel, amplitude)
        re_channel.parent.awg.set_amplitude(2*awg_channel + 1, amplitude)
        re_channel.parent.awg.set_offset(2 * awg_channel, np.real(calib_dc['dc']))
        re_channel.parent.awg.set_offset(2 * awg_channel + 1, np.imag(calib_dc['dc']))
        definition_fragment = textwrap.dedent('''
wave {name}_wave_i = join(sine({samples}, {amplitude_i}, {phaseOffset_i}, {nrOfPeriods}), zeros(16));
wave {name}_wave_q = join(sine({samples}, {amplitude_q}, {phaseOffset_q}, {nrOfPeriods}), zeros(16));
'''.format(name=channel, samples=nrOfsampl, amplitude_i=np.abs(calib_rf['I']), amplitude_q=np.abs(calib_rf['Q']),
           phaseOffset_i=np.angle(calib_rf['I']) * 2, phaseOffset_q=np.angle(calib_rf['Q']) * 2,
           nrOfPeriods=nrOfPeriods))
        play_fragment = textwrap.dedent('''
//
    playWave({name}_wave_i, {name}_wave_q);
'''.format(name=channel))
        return definition_fragment, play_fragment

    def readout_rect_multi(self, length, *params):
        definition_fragment = ''''''
        play_fragment = ''''''
        if len(params)==1:
            return self.readout_rect(params[0][0], length, params[0][1])
        else:
            for param in params:
                channel = param[0]
                amplitude = param[1]
                # re_channel = device.awg_channels[channel]
                re_channel = self.channels[channel]
                calib_dc = re_channel.parent.calib_dc()
                calib_rf = re_channel.parent.calib_rf(re_channel)
                nrOfsampl = int(length * re_channel.get_clock())
                nrOfPeriods = int(length * np.abs(re_channel.get_if()))
                awg_channel = re_channel.parent.sequencer_id
                re_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
                re_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
                re_channel.parent.awg.set_offset(2 * awg_channel, np.real(calib_dc['dc']))
                re_channel.parent.awg.set_offset(2 * awg_channel + 1, np.imag(calib_dc['dc']))
                definition_fragment += textwrap.dedent('''
wave {name}_wave_i = join(sine({samples}, {amplitude_i}, {phaseOffset_i}, {nrOfPeriods}), zeros(16));
wave {name}_wave_q = join(sine({samples}, {amplitude_q}, {phaseOffset_q}, {nrOfPeriods}), zeros(16));
'''.format(name=channel, samples=nrOfsampl, amplitude_i=amplitude * np.abs(calib_rf['I']),
        amplitude_q=amplitude * np.abs(calib_rf['Q']),
           phaseOffset_i=np.angle(calib_rf['I']) * 2, phaseOffset_q=np.angle(calib_rf['Q']) * 2,
           nrOfPeriods=nrOfPeriods))
            add_wave_i = ''''''
            add_wave_q = ''''''
            for param in params:
                channel = param[0]
                add_wave_i += textwrap.dedent('''{name}_wave_i,'''.format(name=channel))
                add_wave_q += textwrap.dedent('''{name}_wave_q,'''.format(name=channel))

            definition_fragment += textwrap.dedent('''
wave ro_wave_i = add({add_wave_i});
wave ro_wave_q = add({add_wave_q});
'''.format(add_wave_i=add_wave_i[:-1], add_wave_q=add_wave_q[:-1]))

            play_fragment += textwrap.dedent('''
//
    playWave(ro_wave_i, ro_wave_q);
'''.format(name=channel))

        return definition_fragment, play_fragment
