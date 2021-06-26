#from scipy.signal import gaussian
# from scipy.signal import tukey
# from scipy.signal import hann
import numpy as np
import textwrap


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

    def gauss_hd(self, channel, length, amp_x, sigma, alpha=0., position = 0):
        #gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
        #                 sigma * self.channels[channel].get_clock())
        #gauss -= gauss[0]
        #gauss /= np.max(gauss)
        #gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        definition_fragment = ''''''
        play_fragment = ''''''
        var_reg0 = 0
        var_reg1 = 1
        ex_channel = self.channels[channel]
        length_samp = int((length) * ex_channel.get_clock())
        sigma_samp = int((sigma) * ex_channel.get_clock())
        position_samp = int((position) * ex_channel.get_clock())
        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_{length_samp}
wave gauss_hd_{length_samp} = gauss({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
wave drag_hd_{length_samp} = {alpha}*drag({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
'''.format(length_samp=length_samp, sigma_samp=sigma_samp, position_samp=position_samp, amp_x=amp_x, alpha = alpha))

        play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*gauss_hd_{length_samp} , 2, {ampI}*drag_hd_{length_samp});
    '''.format(length_samp=length_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x)))

        if ex_channel.is_iq():
            play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});
    '''.format(length_samp=length_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x)))
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 1, {ampQ}*rect_cos_{length_samp});
    '''.format(length_samp=length_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x)))
            elif control_channel_id == 1:
                play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});
    '''.format(length_samp=length_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x)))

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
wave tail_wave_0 = hann(2*tail_samp, 1);
'''.format(tail_samp=tail_samp))
            # It's a trick how to set Rabi pulse length with precision corresponded to awg.clock (2.4GHz)
            for wave_length in range(0, n_samp+1):
                if tail_samp > 2:
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length} = join(zeros(31-tail_samp%32), cut(tail_wave_0, 0, tail_samp));
wave tail_fall_{wave_length} = cut(tail_wave_0, tail_samp, 2*tail_samp-1);
wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));
'''.format(wave_length=wave_length, n_samp=n_samp))
                else:
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length} = zeros(32);
wave tail_fall_{wave_length} = zeros(32);
//wave w_{wave_length} = join(zeros(32-{wave_length}), tail_rise_{wave_length}, rect({wave_length}, 1));
wave w_{wave_length} = join(zeros(32-{wave_length}), rect({wave_length}, 1));
'''.format(wave_length=wave_length, n_samp=n_samp))

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
            playWave(1, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length});
'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
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
            playWave(1, {ampI}*tail_fall_{wave_length}, 1, {ampQ}*tail_fall_{wave_length});
'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
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
            playWave(2, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length});
'''.format(wave_length=wave_length, ampI=np.real(amp), ampQ=np.imag(amp)))
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
wave rect_cos_{length_samp} = rect({length_samp}, {amp});
'''.format(tail_samp=tail_samp, length_samp=length_samp, amp=1))

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});
'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
            else:
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}, 1, {ampQ}*rect_cos_{length_samp});
'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*rect_cos_{length_samp}, 2, {ampQ}*rect_cos_{length_samp});
'''.format(length_samp=length_samp, ampI=np.real(amp), ampQ=np.imag(amp)))


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
    waitWave();
    '''.format(length_samp=length_samp))
        return definition_fragment, play_fragment

    def virtual_z(self, channel, length, phase, fast_control = False, resolution = 8):
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
            definition_fragment = '''cvar resolution = {resolution};'''.format(resolution=resolution)
            for bit in range(resolution):
                bitval = 1 << resolution
                play_fragment += textwrap.dedent('''
                    if (phase_variable & {bitval}) {
                        incrementSinePhase(0, {increment});''')
                if ex_channel.is_iq():
                    play_fragment += textwrap.dedent('''
                    incrementSinePhase(1, {increment});''')
                play_fragment += '''
                }'''
                play_fragment = play_fragment.format(bitval=bitval, increment=bitval / (2 << resolution) * 360.0)
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
        incrementSinePhase(1,{phase1});'''.format(phase1=8 * phase))
                play_fragment += '''
//
    }'''
                play_fragment += textwrap.dedent('''
//    
    incrementSinePhase(0,{phase2});
    incrementSinePhase(1,{phase2});
    waitWave();
'''.format(phase2=64 * phase))
            else:
                play_fragment +='''
//
    i=0;
    for (i=0; i < variable_register0; i = i +1) {'''
                play_fragment += textwrap.dedent('''
//
        incrementSinePhase(0,{phase1});
        waitWave();'''.format(phase1=8*phase))
                play_fragment += '''
//
    }'''
                play_fragment += textwrap.dedent('''
//    
    incrementSinePhase(0,{phase2});
    waitWave();
    '''.format(phase2=64*phase))
        else:
            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    incrementSinePhase(0,{phase});
    incrementSinePhase(1,{phase});
'''.format(phase=phase))
            else:
                play_fragment += textwrap.dedent('''
//
    incrementSinePhase({channel}, {phase});
'''.format(channel=ex_channel.channel % 2, phase=phase))

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
                ex_channel =  self.channels[channel_name]
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
        for pulse in params:
            channel = pulse[0]
            ex_channel = self.channels[channel]
            if ex_channel.is_iq():
                seq_id = ex_channel.parent.sequencer_id
            else:
                seq_id = ex_channel.channel // 2
            # print ('Setting multipulse: \npulse:', pulse[1], 'channel:', channel, 'length:', length, 'other args:', pulse[2:])
            pulses[seq_id] = pulse[1](channel, length_set, *pulse[2:])
        return pulses

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
wave {name}_wawe_i = sine({samples}, {amplitude_i}, {phaseOffset_i}, {nrOfPeriods});
wave {name}_wawe_q = sine({samples}, {amplitude_q}, {phaseOffset_q}, {nrOfPeriods});
'''.format(name=channel, samples=nrOfsampl, amplitude_i=np.abs(calib_rf['I']), amplitude_q=np.abs(calib_rf['Q']),
           phaseOffset_i=np.angle(calib_rf['I']) * 2, phaseOffset_q=np.angle(calib_rf['Q']) * 2,
           nrOfPeriods=nrOfPeriods))
        play_fragment = textwrap.dedent('''
//
    playWave({name}_wawe_i, {name}_wawe_q);
'''.format(name=channel))
        return definition_fragment, play_fragment

    def readout_rect_multi(self, length, *params):
        definition_fragment = ''''''
        play_fragment = ''''''
        if len(params)==1:
            return readout_rect(param[0][0], length, param[0][1])
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
wave {name}_wawe_i = sine({samples}, {amplitude_i}, {phaseOffset_i}, {nrOfPeriods});
wave {name}_wawe_q = sine({samples}, {amplitude_q}, {phaseOffset_q}, {nrOfPeriods});
'''.format(name=channel, samples=nrOfsampl, amplitude_i=amplitude * np.abs(calib_rf['I']),
            amplitude_q=amplitude * np.abs(calib_rf['Q']),
           phaseOffset_i=np.angle(calib_rf['I']) * 2, phaseOffset_q=np.angle(calib_rf['Q']) * 2,
           nrOfPeriods=nrOfPeriods))
            add_wave_i = ''''''
            add_wave_q = ''''''
            for param in params:
                channel = param[0]
                add_wave_i += textwrap.dedent('''{name}_wawe_i,'''.format(name=channel))
                add_wave_q += textwrap.dedent('''{name}_wawe_q,'''.format(name=channel))

            definition_fragment += textwrap.dedent('''
wave ro_wawe_i = add({add_wave_i});
wave ro_wawe_q = add({add_wave_q});
'''.format(add_wave_i=add_wave_i[:-1], add_wave_q=add_wave_q[:-1]))

            play_fragment += textwrap.dedent('''
//
    playWave(ro_wawe_i, ro_wawe_q);
'''.format(name=channel))

        return definition_fragment, play_fragment
