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

    def hadamar_cz_sequence_qubit_1(self, channel, length, length1, length_fsim, amp_x, sigma, phase_1=0,
                                    phase_x1=0, phase_x2=0, phase_3=0, virtual_phase = 0, amp_middle=0):

        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int(round((length) * ex_channel.get_clock()/16)*16)
        sigma_samp = int((sigma) * ex_channel.get_clock())
        # nrOfPeriods = int(np.round((length_samp) * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()))
        nrOfPeriods = length_samp * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()
        position_samp = (length_samp - 1) / 2
        length_samp_fsim = round((length_fsim) * ex_channel.get_clock() / 16) * 16

        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_mod_hadamar_cz
wave gauss_hd_mod_hadamar_cz = gauss({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
gauss_hd_mod_hadamar_cz = gauss_hd_mod_hadamar_cz - gauss_hd_mod_hadamar_cz[0];

wave sine_wave_initial_phase1 = sine({length_samp}, 1, {initial_phase1}, {nrOfPeriods});
wave w_1_qubit1_I = multiply({ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase1);
wave w_1_qubit1_Q = multiply({ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase1);

wave w_2_qubit1_I = zeros({length_samp_fsim});
wave w_2_qubit1_Q = zeros({length_samp_fsim});

wave sine_wave_initial_phase_x1 = sine({length_samp}, 1, {initial_phase_x1}, {nrOfPeriods});
wave w_x1_qubit1_I = multiply({amp_middle}*{ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x1);
wave w_x1_qubit1_Q = multiply({amp_middle}*{ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x1);
wave sine_wave_initial_phase_x2 = sine({length_samp}, 1, {initial_phase_x2}, {nrOfPeriods});
wave w_x2_qubit1_I = multiply({amp_middle}*{ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x2);
wave w_x2_qubit1_Q = multiply({amp_middle}*{ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x2);

wave sine_wave_initial_phase3 = sine({length_samp}, 1, {initial_phase3}, {nrOfPeriods});
wave w_3_qubit1_I = multiply({ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase3);
wave w_3_qubit1_Q = multiply({ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase3);

wave wave_I=join(w_1_qubit1_I, w_2_qubit1_I, w_x1_qubit1_I, w_x2_qubit1_I, w_2_qubit1_I, w_3_qubit1_I);
wave wave_Q=join(w_1_qubit1_Q, w_2_qubit1_Q, w_x1_qubit1_Q, w_x2_qubit1_Q, w_2_qubit1_Q, w_3_qubit1_Q);
'''.format(length_samp=length_samp, sigma_samp=sigma_samp, position_samp=position_samp, amp_x=1,
               ampI=np.real(amp_x), ampQ=np.imag(amp_x), nrOfPeriods=nrOfPeriods, initial_phase1=phase_1,
               initial_phase_x1=phase_x1, initial_phase_x2=phase_x2, initial_phase3=phase_3, length_samp_fsim=length_samp_fsim,
           amp_middle=amp_middle))

        play_fragment += '''
//
    playWave(1, wave_I, 1, wave_Q,);
    waitWave();'''
                    # TODO
        entry_table_index_constants.append('etic_wave_I_wave_Q_qubit1')
        assign_fragment += 'assignWaveIndex(1, wave_I, 1, wave_Q, etic_wave_I_wave_Q_qubit1);'

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}
            table_entry['phase1'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def hadamar_cz_sequence_qubit_2(self, channel, length, length1, length_fsim, amp_x, sigma, freq, amp_sin, phase_2=0,
                                    phase_x1=0, phase_x2=0, phase_4=0, virtual_phase=0, amp_middle=1):

        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int(round((length) * ex_channel.get_clock()/16)*16)
        sigma_samp = int((sigma) * ex_channel.get_clock())
        # nrOfPeriods = int(np.round((length_samp) * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()))
        nrOfPeriods = length_samp * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()
        position_samp = (length_samp - 1) / 2

        nrOfsampl_fsim = round((length_fsim) * ex_channel.get_clock() / 16) * 16
        nrOfPeriods_fsim = int(np.round((nrOfsampl_fsim) * np.abs(freq) / ex_channel.get_clock()))

        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_mod_hadamar_cz
wave gauss_hd_mod_hadamar_cz = gauss({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
gauss_hd_mod_hadamar_cz = gauss_hd_mod_hadamar_cz - gauss_hd_mod_hadamar_cz[0];

wave sine_wave_initial_phase2 = sine({length_samp}, 1, {initial_phase2}, {nrOfPeriods});
wave w_1_qubit2_I = multiply({ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase2);
wave w_1_qubit2_Q = multiply({ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase2);

wave sine_wave_initial_phase_x = sine({length_samp_fsim}, 1, 0, {nrOfPeriods_fsim});

wave w_2_qubit2_I = {amp_fsim_I}*sine_wave_initial_phase_x;
wave w_2_qubit2_Q = {amp_fsim_Q}*sine_wave_initial_phase_x;

wave sine_wave_initial_phase_x1 = sine({length_samp}, 1, {initial_phase_x1}, {nrOfPeriods});
wave w_x1_qubit2_I = multiply({amp_middle}*{ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x1);
wave w_x1_qubit2_Q = multiply({amp_middle}*{ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x1);
wave sine_wave_initial_phase_x2 = sine({length_samp}, 1, {initial_phase_x2}, {nrOfPeriods});
wave w_x2_qubit2_I = multiply({amp_middle}*{ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x2);
wave w_x2_qubit2_Q = multiply({amp_middle}*{ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase_x2);


wave sine_wave_initial_phase4 = sine({length_samp}, 1, {initial_phase4}, {nrOfPeriods});
wave w_3_qubit2_I = multiply({ampI}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase4);
wave w_3_qubit2_Q = multiply({ampQ}*gauss_hd_mod_hadamar_cz, sine_wave_initial_phase4);

wave wave_I=join(w_1_qubit2_I, w_2_qubit2_I, w_x1_qubit2_I, w_x2_qubit2_I, w_2_qubit2_I, w_3_qubit2_I);
wave wave_Q=join(w_1_qubit2_Q, w_2_qubit2_Q, w_x1_qubit2_Q, w_x2_qubit2_Q, w_2_qubit2_Q, w_3_qubit2_Q);
    '''.format(length_samp=length_samp, sigma_samp=sigma_samp, position_samp=position_samp, amp_x=1,
               ampI=np.real(amp_x), ampQ=np.imag(amp_x),
               nrOfPeriods=nrOfPeriods, initial_phase2=phase_2, initial_phase4=phase_4,
               initial_phase_x1=phase_x1, initial_phase_x2=phase_x2,
               amp_fsim_I=np.real(amp_sin), amp_fsim_Q=np.imag(amp_sin),
               length_samp_fsim=nrOfsampl_fsim, nrOfPeriods_fsim=nrOfPeriods_fsim, amp_middle=amp_middle))

        play_fragment += '''
//
    playWave(1, wave_I, 1, wave_Q,);
    waitWave();'''
        # TODO
        entry_table_index_constants.append('etic_wave_I_wave_Q_qubit2')
        assign_fragment += 'assignWaveIndex(1, wave_I, 1, wave_Q, etic_wave_I_wave_Q_qubit2);'

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}
            table_entry['phase1'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3), 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3),
                                             'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': np.round(virtual_phase * 360 / 2 / np.pi, 3),
                                             'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def hadamar_cz_sequence_coupler(self, channel, length, length1, length_fsim, amp, length_tail,sign=1):

        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int(round((length) * ex_channel.get_clock()/16)*16)
        # nrOfPeriods = int(np.round((length_samp) * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()))

        tail_samp = int(np.round(length_tail * ex_channel.get_clock()))
        #length_samp = int((length - 2*length_tail) * ex_channel.get_clock())
        length_samp_fsim = int(np.round((length_fsim) * ex_channel.get_clock()))-2*tail_samp

        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_mod_hadamar_cz

wave w_1_coupler_I = zeros({length_samp});
wave w_1_coupler_Q = zeros({length_samp});

wave tail_rect_cos_{length_samp_fsim} = (hann(4*{tail_samp}+1, {amp})-0.5)*2.0;
wave rise_rect_cos_{length_samp_fsim} = cut(tail_rect_cos_{length_samp_fsim}, {tail_samp}, 2*{tail_samp}-1);
wave fall_rect_cos_{length_samp_fsim} = cut(tail_rect_cos_{length_samp_fsim}, 2*{tail_samp}+1, 3*{tail_samp});

wave w_2_coupler_I = {amp_fsim_I}*join(rise_rect_cos_{length_samp_fsim}, rect({length_samp_fsim}, {amp}), fall_rect_cos_{length_samp_fsim});
wave w_2_coupler_Q = {amp_fsim_Q}*join(rise_rect_cos_{length_samp_fsim}, rect({length_samp_fsim}, {amp}), fall_rect_cos_{length_samp_fsim});

wave w_x_coupler_I = zeros(2*{length_samp});
wave w_x_coupler_Q = zeros(2*{length_samp});

wave w_3_coupler_I = ({sign})*({amp_fsim_I})*join(rise_rect_cos_{length_samp_fsim}, rect({length_samp_fsim}, {amp}), fall_rect_cos_{length_samp_fsim});
wave w_3_coupler_Q = ({sign})*({amp_fsim_Q})*join(rise_rect_cos_{length_samp_fsim}, rect({length_samp_fsim}, {amp}), fall_rect_cos_{length_samp_fsim});

wave wave_I=join(w_1_coupler_I, w_2_coupler_I, w_x_coupler_I, w_3_coupler_I, w_1_coupler_I);
wave wave_Q=join(w_1_coupler_Q, w_2_coupler_Q, w_x_coupler_Q, w_3_coupler_Q, w_1_coupler_Q);
            '''.format(amp=1, amp_fsim_I=np.real(amp), amp_fsim_Q=np.imag(amp), tail_samp=tail_samp, length_samp=length_samp,
                       length_samp_fsim=length_samp_fsim, sign=sign))

        play_fragment += '''
//
    playWave(2, wave_I, 2, wave_Q,);
    waitWave();'''
        # TODO
        entry_table_index_constants.append('etic_wave_I_wave_Q_coupler')
        assign_fragment += 'assignWaveIndex(2, wave_I, 2, wave_Q, etic_wave_I_wave_Q_coupler);'

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': 0, 'increment': True}
            table_entry['phase1'] = {'value': 0, 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': 0 ,
                                         'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': 0 ,
                                         'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry


    def gauss_hd_modulation(self, channel, length, amp_x, sigma, alpha=0., phase=0, fast_control=False):
        # gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
        #                 sigma * self.channels[channel].get_clock())
        # gauss -= gauss[0]
        # gauss /= np.max(gauss)
        # gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int(round((length) * ex_channel.get_clock()/16)*16)
        sigma_samp = int((sigma) * ex_channel.get_clock())
        # nrOfPeriods = int(np.round((length_samp) * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()))
        nrOfPeriods = length_samp * np.abs(ex_channel.get_frequency()) / ex_channel.get_clock()

        position_samp = (length_samp - 1) / 2
        definition_fragment += textwrap.dedent('''
// Waveform definition gauss_hd_{length_samp}
wave gauss_hd_mod_{length_samp}_{sigma_samp}_{phase_round} = gauss({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
gauss_hd_mod_{length_samp}_{sigma_samp}_{phase_round} = gauss_hd_mod_{length_samp}_{sigma_samp}_{phase_round} - gauss_hd_mod_{length_samp}_{sigma_samp}_{phase_round}[0];
wave drag_hd_mod_{length_samp}_{sigma_samp}_{phase_round} = {alpha}*drag({length_samp}, {amp_x}, {position_samp}, {sigma_samp});
wave sine_wave_{length_samp}_{nrOfPeriods_round}_{phase_round} = sine({length_samp}, 1, {initial_phase}, {nrOfPeriods});

wave gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} = multiply(gauss_hd_mod_{length_samp}_{sigma_samp}_{phase_round}, sine_wave_{length_samp}_{nrOfPeriods_round}_{phase_round});
wave drag_modulation_{length_samp}_{sigma_samp}_{phase_round} = multiply(drag_hd_mod_{length_samp}_{sigma_samp}_{phase_round}, sine_wave_{length_samp}_{nrOfPeriods_round}_{phase_round});
'''.format(length_samp=length_samp, sigma_samp=sigma_samp, position_samp=position_samp, amp_x=1, alpha=alpha,
            nrOfPeriods=nrOfPeriods, nrOfPeriods_round=int(np.round(nrOfPeriods)), initial_phase=phase, phase_round=int(1000*phase)))

        if ex_channel.is_iq():
            play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 2, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x), alpha=alpha, phase_round=(1000*phase)))
            # TODO
            entry_table_index_constants.append('etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x)) // 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x)) // 0.01),
                                                      phase_round=(1000*phase)))

            assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
2, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x)) // 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01),
                                                      phase_round=int(1000*phase)))
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                play_fragment += textwrap.dedent('''
        //
            playWave(1, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 1, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                  ampQ=np.imag(amp_x), alpha=alpha, phase=np.round(phase*360/2/np.pi, 3), phase_round=int(1000*phase)))
                # TODO
                entry_table_index_constants.append('etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01),
                                                      phase_round=int(1000*phase)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
1, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01),
                                                      phase_round=int(1000*phase)))

            elif control_channel_id == 1:
                play_fragment += textwrap.dedent('''
        //
            playWave(2, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 2, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                  ampQ=np.imag(amp_x), alpha=alpha, phase=np.round(phase*360/2/np.pi, 3),
                                  phase_round=int(1000*phase)))
                # TODO
                entry_table_index_constants.append('etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01),
                                                      phase_round=int(1000*phase)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} - {ampQ}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
2, {ampQ}*gauss_modulation_{length_samp}_{sigma_samp}_{phase_round} + {ampI}*drag_modulation_{length_samp}_{sigma_samp}_{phase_round}, 
etic_gauss_modulation_{length_samp}_{signI}{realI}_{signQ}{imagQ}_{phase_round});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01),
                                                      phase_round=int(1000*phase)))

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}
            table_entry['phase1'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': 0*np.round(phase*360/2/np.pi, 3), 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': 0*np.round(phase*360/2/np.pi, 3), 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def gauss_hd(self, channel, length, amp_x, sigma, alpha=0., phase = 0, fast_control = False):
        #gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
        #                 sigma * self.channels[channel].get_clock())
        #gauss -= gauss[0]
        #gauss /= np.max(gauss)
        #gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        #if phase<0:
            #phase=2*np.pi +phase
        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int(round((length) * ex_channel.get_clock()/16)*16)
        sigma_samp = int((sigma) * ex_channel.get_clock())
        #position_samp = round((length)*ex_channel.get_clock())/2
        position_samp = (length_samp-1)/2
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
            //incrementSinePhase(0,{phase});
            //incrementSinePhase(1,{phase});
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();
            //incrementSinePhase(0,{phase});
            //incrementSinePhase(1,{phase});
            playWave(1, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=np.round(phase*360/2/np.pi, 3)))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
            //incrementSinePhase(0,{phase});
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();
            //incrementSinePhase(0,{phase});
            playWave(1, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=np.round(phase*360/2/np.pi, 3)))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
            //incrementSinePhase(1,{phase});
            playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();
            //incrementSinePhase(1,{phase});
            playWave(2, -{ampI}*gauss_hd_{length_samp}_{sigma_samp} + {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, -{ampQ}*gauss_hd_{length_samp}_{sigma_samp} - {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x), ampQ=np.imag(amp_x),
                       alpha=alpha, phase=np.round(phase*360/2/np.pi, 3)))
            play_fragment += '''
//
    }}'''

        elif fast_control:
            definition_fragment +='''
//var i = 0;
//var j = 0;'''
            play_fragment += '''
//
    repeat(variable_register0){
        repeat(variable_register1){'''

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
            //incrementSinePhase(0,{phase});
            //incrementSinePhase(1,{phase});
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                  ampQ=np.imag(amp_x), alpha = alpha, phase=np.round(phase*360/2/np.pi, 3)))

                # TODO
                entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))

                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
            //incrementSinePhase(0,{phase});
            playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(0,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                  ampQ=np.imag(amp_x), alpha=alpha, phase=np.round(phase*360/2/np.pi, 3)))

                    # TODO
                    entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
            //incrementSinePhase(1,{phase});
            playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
            incrementSinePhase(1,{phase});
            waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                                  ampQ=np.imag(amp_x), alpha=alpha, phase=np.round(phase*360/2/np.pi, 3)))
                    # TODO
                    entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
            play_fragment +='''
//
    }}'''

        else:
            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    //incrementSinePhase(0,{phase});
    //incrementSinePhase(1,{phase});
    playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(0,{phase});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                          ampQ=np.imag(amp_x), alpha = alpha, phase=np.round(phase*360/2/np.pi, 3)))

                # TODO
                entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
            else:
                control_channel_id = ex_channel.channel % 2
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
    //incrementSinePhase(0,{phase});
    playWave(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(0,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                          ampQ=np.imag(amp_x), alpha = alpha, phase=np.round(phase*360/2/np.pi, 3)))
                    # TODO
                    entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
1, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))

                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
    //incrementSinePhase(1,{phase});
    playWave(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(length_samp=length_samp, sigma_samp=sigma_samp, ampI=np.real(amp_x),
                          ampQ=np.imag(amp_x), alpha = alpha, phase=np.round(phase*360/2/np.pi, 3)))
                    #TODO
                    entry_table_index_constants.append('etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*gauss_hd_{length_samp}_{sigma_samp} - {ampQ}*drag_hd_{length_samp}_{sigma_samp}, 
2, {ampQ}*gauss_hd_{length_samp}_{sigma_samp} + {ampI}*drag_hd_{length_samp}_{sigma_samp}, 
etic_gauss_hd_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(length_samp=length_samp, sigma_samp=sigma_samp,
                                                      signI=(np.sign(np.real(amp_x)) == 1), ampI=np.real(amp_x), realI=int(np.abs(np.real(amp_x))// 0.01),
                                                      signQ=(np.sign(np.imag(amp_x)) == 1), ampQ=np.imag(amp_x), imagQ=int(np.abs(np.imag(amp_x))// 0.01)))
        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            calib_dc = ex_channel.parent.calib_dc()
            calib_rf = ex_channel.parent.calib_rf(ex_channel)
            awg_channel = ex_channel.parent.sequencer_id
            ex_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
            ex_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
            ex_channel.parent.awg.set_offset(channel=2 * awg_channel, offset=np.real(calib_dc['dc']))
            ex_channel.parent.awg.set_offset(channel=2 * awg_channel + 1, offset=np.imag(calib_dc['dc']))
            ex_channel.parent.awg.set_sin_phase(2 * awg_channel, np.angle(calib_rf['I']) * 360 / np.pi)
            ex_channel.parent.awg.set_sin_phase(2 * awg_channel + 1, np.angle(calib_rf['Q']) * 360 / np.pi)
            ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel, 0, np.abs(calib_rf['I']))
            ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel + 1, 1, np.abs(calib_rf['Q']))


            table_entry['phase0'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}
            table_entry['phase1'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}

        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': np.round(phase*360/2/np.pi, 3), 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def rect_cos(self, channel, length, amp, length_tail, fast_control=False, control_frequency=0):
        # this part is necessary for accurate length setter and fast control
        # for example: rabi sequence
        # for fast control you can use "variable_register" defined in the beginning of the sequencer play program
        n_samp = 8
        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        var_reg0 = 0
        var_reg1 = 1
        ex_channel = self.channels[channel]
        pause_cycles = int(np.round((length - 2*length_tail) * ex_channel.get_clock()))
        tail_samp = int(np.round(length_tail * ex_channel.get_clock()))
        #length_samp = int((length - 2*length_tail) * ex_channel.get_clock())
        length_samp = int(np.round((length) * ex_channel.get_clock()))-2*tail_samp
        entry_table_index_constants = []
        if ex_channel.is_iq():
            control_seq_id = ex_channel.parent.sequencer_id
            control_channel_id = None
            calib_dc = ex_channel.parent.calib_dc()
            calib_rf = ex_channel.parent.calib_rf(ex_channel)
            awg_channel = ex_channel.parent.sequencer_id
            ex_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
            ex_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
            ex_channel.parent.awg.set_offset(channel=2 * awg_channel, offset=np.real(calib_dc['dc']))
            ex_channel.parent.awg.set_offset(channel=2 * awg_channel + 1, offset=np.imag(calib_dc['dc']))
            ex_channel.parent.awg.set_sin_phase(2 * awg_channel, np.angle(calib_rf['I']) * 360 / np.pi)
            ex_channel.parent.awg.set_sin_phase(2 * awg_channel+1, np.angle(calib_rf['Q']) * 360 / np.pi)
            ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel, 0, np.abs(calib_rf['I']))
            ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel + 1, 1, np.abs(calib_rf['Q']))
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
const tail_samp_{amp} = {tail_samp};
control_frequency = {control_frequency};

// Waveform definition
wave tail_wave_0_{amp} = hann(2*tail_samp_{amp}, 1);'''.format(tail_samp=tail_samp, control_frequency=control_frequency,
                                                   amp=int(np.abs(amp)// 0.01)))
            # It's a trick how to set Rabi pulse length with precision corresponded to awg.clock (2.4GHz)
            for wave_length in range(0, n_samp+1):
                if tail_samp > 2:
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length}_{amp} = join(zeros(31-tail_samp_{amp}%32), cut(tail_wave_0_{amp}, 0, tail_samp_{amp}));
wave tail_fall_{wave_length}_{amp} = cut(tail_wave_0_{amp}, tail_samp_{amp}, 2*tail_samp_{amp}-1);
wave w_{wave_length}_{amp} = join(zeros(32-{wave_length}), tail_rise_{wave_length}_{amp}, rect({wave_length}, 1));'''.format(wave_length=wave_length,
                                                                                                                       n_samp=n_samp,
                                                                                                                       amp=int(np.abs(amp)// 0.01)))
                else:
#                     definition_fragment += textwrap.dedent('''
# wave tail_rise_{wave_length}_{amp} = join(zeros(31-tail_samp_{amp}%32), cut(tail_wave_0_{amp}, 0, tail_samp_{amp}));
# wave tail_fall_{wave_length}_{amp} = cut(tail_wave_0_{amp}, tail_samp_{amp}, 2*tail_samp_{amp}-1);
# wave w_{wave_length}_{amp} = join(zeros(32-{wave_length}), tail_rise_{wave_length}_{amp}, rect({wave_length}, 1));'''.format(wave_length=wave_length,
#                                                                                                                              n_samp=n_samp,
#                                                                                                                              amp=int(np.abs(amp) // 0.01)))
                    definition_fragment += textwrap.dedent('''
wave tail_rise_{wave_length}_{amp} = zeros(32);
wave tail_fall_{wave_length}_{amp}  = zeros(32);
wave w_{wave_length}_{amp} = join(zeros(32-{wave_length}), rect({wave_length}, 1));'''.format(wave_length=wave_length,
                                                                                              n_samp=n_samp, amp=int(np.abs(amp)// 0.01)))


            play_fragment += textwrap.dedent('''
//

    switch (variable_register1) {''')
            for wave_length in range(0, n_samp+1):
                if ex_channel.is_iq():
                    play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(1, {ampI}*w_{wave_length}_{amp}, 2, {ampQ}*w_{wave_length}_{amp});
            wait(variable_register0);
            playWave(1, {ampI}*tail_fall_{wave_length}_{amp}, 2, {ampQ}*tail_fall_{wave_length}_{amp});'''.format(wave_length=wave_length,
                                                                                        ampI=np.real(amp), ampQ=np.imag(amp),
                                                                                                amp=int(np.abs(amp)// 0.01)))
                    # TODO
                    entry_table_index_constants.append('etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*w_{wave_length}, 2, {ampQ}*w_{wave_length}, etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                    entry_table_index_constants.append('etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length}, etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

                    if wave_length==n_samp:
                        play_fragment += '''
//
        }'''
                else:
                    if control_channel_id == 0:
                        play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(1, {ampI}*w_{wave_length}_{amp}, 1, {ampQ}*w_{wave_length}_{amp});
            wait(variable_register0);
            playWave(1, {ampI}*tail_fall_{wave_length}_{amp}, 1, {ampQ}*tail_fall_{wave_length}_{amp});'''.format(wave_length=wave_length,
                                                                                        ampI=np.real(amp), ampQ=np.imag(amp),
                                                                                                      amp=int(np.abs(amp)// 0.01)))
                        # TODO
                        entry_table_index_constants.append('etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*w_{wave_length}, 1, {ampQ}*w_{wave_length}, etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        entry_table_index_constants.append('etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*tail_fall_{wave_length}, 1, {ampQ}*tail_fall_{wave_length}, etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

                        if wave_length == n_samp:
                            play_fragment +='''
//
        }'''
                    elif control_channel_id == 1:
                        play_fragment += textwrap.dedent('''
//
        case {wave_length}:
            playWave(2, {ampI}*w_{wave_length}_{amp}, 2, {ampQ}*w_{wave_length}_{amp});
            wait(variable_register0);
            playWave(2, {ampI}*tail_fall_{wave_length}_{amp}, 2, {ampQ}*tail_fall_{wave_length}_{amp});'''.format(wave_length=wave_length,
                                                                                        ampI=np.real(amp), ampQ=np.imag(amp),
                                                                                                      amp=int(np.abs(amp)// 0.01)))
                        # TODO
                        entry_table_index_constants.append('etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*w_{wave_length}, 2, {ampQ}*w_{wave_length}, etic_w_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        entry_table_index_constants.append('etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ}'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                        assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*tail_fall_{wave_length}, 2, {ampQ}*tail_fall_{wave_length}, etic_tail_fall_{wave_length}_{signI}{realI}_{signQ}{imagQ});'''.format(
                            wave_length=wave_length, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                            signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

                        if wave_length == n_samp:
                            play_fragment += '''
//
        }'''

        else:
            if tail_samp > 2:
                definition_fragment += textwrap.dedent('''
// Waveform definition rect_cos_{length_samp}
wave tail_rect_cos_{length_samp} = (hann(4*{tail_samp}+1, {amp})-0.5)*2.0;
wave rise_rect_cos_{length_samp} = cut(tail_rect_cos_{length_samp}, {tail_samp}, 2*{tail_samp}-1);
wave fall_rect_cos_{length_samp} = cut(tail_rect_cos_{length_samp}, 2*{tail_samp}+1, 3*{tail_samp});
wave rect_cos_{length_samp}_{tail_samp} = join(rise_rect_cos_{length_samp}, rect({length_samp}, {amp}), fall_rect_cos_{length_samp});
'''.format(tail_samp=tail_samp, length_samp=length_samp, amp=1))
            else:
                definition_fragment += textwrap.dedent('''
// Waveform definition rect_cos_{length_samp}
wave rect_cos_{length_samp}_{tail_samp} = rect({length_samp}, {amp});'''.format(tail_samp=tail_samp, length_samp=length_samp, amp=1))

            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}_{tail_samp}, 2, {ampQ}*rect_cos_{length_samp}_{tail_samp});'''.format(length_samp=length_samp, tail_samp=tail_samp,
                                                                                ampI=np.real(amp), ampQ=np.imag(amp)))
                # TODO
                entry_table_index_constants.append('etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*rect_cos_{length_samp}_{tail_samp}, 2, {ampQ}*rect_cos_{length_samp}_{tail_samp}, etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
            else:
                if control_channel_id == 0:
                    play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*rect_cos_{length_samp}_{tail_samp}, 1, {ampQ}*rect_cos_{length_samp}_{tail_samp});'''.format(length_samp=length_samp,
                                                                                tail_samp=tail_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
                    # TODO
                    entry_table_index_constants.append('etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*rect_cos_{length_samp}_{tail_samp}, 1, {ampQ}*rect_cos_{length_samp}_{tail_samp}, etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                elif control_channel_id == 1:
                    play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*rect_cos_{length_samp}_{tail_samp}, 2, {ampQ}*rect_cos_{length_samp}_{tail_samp});'''.format(length_samp=length_samp,
                                                                                tail_samp=tail_samp, ampI=np.real(amp), ampQ=np.imag(amp)))
                    # TODO
                    entry_table_index_constants.append('etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                    assign_fragment += textwrap.dedent('''
assignWaveIndex(2, {ampI}*rect_cos_{length_samp}_{tail_samp}, 2, {ampQ}*rect_cos_{length_samp}_{tail_samp}, etic_rect_cos_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
                        length_samp=length_samp, tail_samp=tail_samp, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                        signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': np.round(np.angle(calib_rf['I']) * 360 / np.pi, 3), 'increment': False}
            table_entry['phase1'] = {'value': np.round(np.angle(calib_rf['Q']) * 360 / np.pi, 3), 'increment': False}
            #table_entry['phase0'] = {'value': 0, 'increment': True}
            #table_entry['phase1'] = {'value': 0, 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def pause(self, channel, length, fast_control = False):
        if length<=0 and fast_control is False:
            definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry  = self.pause2(channel, -length)
        else:
            definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry = self.rect_cos(channel, length, amp=0, length_tail=0, fast_control=fast_control)
        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry


    def pause2(self, channel, length):
        ex_channel = self.channels[channel]
        length_samp = int(np.round(length * ex_channel.get_clock()))
        definition_fragment = ''''''
        play_fragment = ''''''
        entry_table_index_constants =[]
        assign_fragment = ''''''
        definition_fragment+= textwrap.dedent('''
wave wave_zeros_{length_samp}= zeros({length_samp});'''.format(length_samp=length_samp))

        if length_samp==0:
            play_fragment += textwrap.dedent('''
//
        waitWave();'''.format(length_samp=length_samp))

        else:
            play_fragment += textwrap.dedent('''
//
    playZero({length_samp});
    waitWave();'''.format(length_samp=length_samp))
        entry_table_index_constants.append('''etic_zeros_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp,
                signI=(np.sign(0) == 1), realI=0, signQ=(np.sign(0) == 1), imagQ=0))
        assign_fragment += textwrap.dedent('''
assignWaveIndex(wave_zeros_{length_samp}, etic_zeros_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
            length_samp=length_samp, signI=(np.sign(0) == 1), realI=0, signQ=(np.sign(0) == 1), imagQ=0))

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            table_entry['phase0'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}
            table_entry['phase1'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': 0.0 * 360 / 2 / np.pi, 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}
        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

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
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        length_samp = int((length) * ex_channel.get_clock())
        if fast_control == 'quasi-binary':
            definition_fragment = '''
cvar resolution = {resolution};
'''.format(resolution=resolution)
            play_fragment +=textwrap.dedent('''
//
//    playWave(zeros({length_samp}));'''.format(length_samp=length_samp))
            for bit in range(resolution):
                bitval = 1 << bit
                play_fragment += textwrap.dedent('''
//
    if (variable_register2 & {bitval}) {{
        incrementSinePhase(0, {increment});'''.format(bitval=bitval, increment=bitval / (1 << resolution) * 360.0))
                if ex_channel.is_iq():
                    play_fragment += textwrap.dedent('''
//
        incrementSinePhase(1, {increment});'''.format(bitval=bitval, increment=bitval / (1 << resolution) * 360.0))
                play_fragment += '''
//
        wait(1);
    } else {
        incrementSinePhase(0, 0.00000001);
'''
                if ex_channel.is_iq():
                    play_fragment += '''
//
        incrementSinePhase(1, 0.00000001);'''
                play_fragment += '''
//
    wait(1);
    }'''

        elif fast_control:
            definition_fragment += '''
//
var i;
var j;'''
            play_fragment += textwrap.dedent('''
//
//    playWave(zeros({length_samp}));'''.format(length_samp=length_samp))
            play_fragment += '''
//
    i=0;
    for (i=0; i < variable_register0; i = i +1) {'''
            if ex_channel.is_iq():
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
            # TODO
            definition_fragment+= textwrap.dedent('''
wave wave_zeros_{length_samp}= zeros({length_samp});'''.format(length_samp=length_samp,
                                                 signP=(np.sign(phase)==1), phase = int(np.abs(phase))))
            entry_table_index_constants.append('''etic_zeros_{length_samp}_{signI}{realI}_{signQ}{imagQ}'''.format(length_samp=length_samp,
                                                signI = (np.sign(0)==1), realI=0, signQ = (np.sign(0)==1), imagQ=0, signP=(np.sign(phase)==1), phase = int(np.abs(phase))))


            if ex_channel.is_iq():
                play_fragment += textwrap.dedent('''
//
//    playWave(zeros({length_samp}));
    incrementSinePhase(0,{phase});
    incrementSinePhase(1,{phase});
    waitWave();'''.format(phase=phase, length_samp=length_samp))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, wave_zeros_{length_samp}, 2, wave_zeros_{length_samp}, etic_zeros_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
                    length_samp=length_samp,
                    signI=(np.sign(0) == 1), realI=0, signQ=(np.sign(0) == 1), imagQ=0, signP=(np.sign(phase) == 1),
                    phase=int(np.abs(phase))))
            else:
                play_fragment += textwrap.dedent('''
//
//    playWave(zeros({length_samp}));
    incrementSinePhase({channel}, {phase});
    waitWave();'''.format(channel=ex_channel.channel % 2, phase=phase, length_samp=length_samp))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, wave_zeros_{length_samp}, 1, wave_zeros_{length_samp}, etic_zeros_{length_samp}_{signI}{realI}_{signQ}{imagQ});'''.format(
                    length_samp=length_samp,
                    signI=(np.sign(0) == 1), realI=0, signQ=(np.sign(0) == 1), imagQ=0, signP=(np.sign(phase) == 1),
                    phase=int(np.abs(phase))))

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}

        if ex_channel.is_iq():
            # calib_dc = ex_channel.parent.calib_dc()
            # calib_rf = ex_channel.parent.calib_rf(ex_channel)
            # awg_channel = ex_channel.parent.sequencer_id
            # ex_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
            # ex_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
            # ex_channel.parent.awg.set_offset(channel=2 * awg_channel, offset=np.real(calib_dc['dc']))
            # ex_channel.parent.awg.set_offset(channel=2 * awg_channel + 1, offset=np.imag(calib_dc['dc']))
            # ex_channel.parent.awg.set_sin_phase(2 * awg_channel, np.angle(calib_rf['I']) * 360 / np.pi)
            # ex_channel.parent.awg.set_sin_phase(2 * awg_channel + 1, np.angle(calib_rf['Q']) * 360 / np.pi)
            # ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel, 0, np.abs(calib_rf['I']))
            # ex_channel.parent.awg.set_sin_amplitude(2 * awg_channel + 1, 1, np.abs(calib_rf['Q']))
            #
            # table_entry['phase0'] = {'value': np.round(np.angle(calib_rf['I']) * 360 / np.pi, 3),
            #                          'increment': False}
            # table_entry['phase1'] = {'value': np.round(np.angle(calib_rf['Q']) * 360 / np.pi, 3),
            #                          'increment': False}

            table_entry['phase0'] = {'value': phase, 'increment': True}
            table_entry['phase1'] = {'value': phase, 'increment': True}
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                table_entry['phase0'] = {'value': phase, 'increment': True}
                table_entry['phase1'] = {'value': 90.0, 'increment': False}

            elif control_channel_id == 1:
                table_entry['phase0'] = {'value': 90.0, 'increment': False}
                table_entry['phase1'] = {'value': phase, 'increment': True}

        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def pmulti(self, device, length, *params):
        try:
            length_set = length[0]
            fast_control = True
        except:
            length_set = length
            fast_control = False
        if fast_control in params:
            fast_control = fast_control

        pulses = {}
        for awg, seq_id in device.pre_pulses.seq_in_use:
            pulses.update({awg.device_id: {}})

        for awg, seq_id in device.pre_pulses.seq_in_use:
            for channel_name, channel in self.channels.items():
                ex_channel = self.channels[channel_name]
                if ex_channel.is_iq():
                    if [awg, seq_id] == [ex_channel.parent.awg, ex_channel.parent.sequencer_id]:
                    #if seq_id ==ex_channel.parent.sequencer_id:
                        pulses[awg.device_id][seq_id] = self.pause(channel_name, length_set, fast_control)
                        break
                else:
                    if [awg, seq_id] == [ex_channel.parent.awg, ex_channel.channel // 2]:
                    #if seq_id == ex_channel.channel // 2:
                        pulses[awg.device_id][seq_id] = self.pause(channel_name, length_set, fast_control)
                        break
        #pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        #pulses = {seq_id: self.pause(seq_id, length) for seq_id in device.pre_pulses.seq_in_use}
        control_seq_id = None
        control_awg_id = None
        for pulse in params:
            channel = pulse[0]
            ex_channel = self.channels[channel]
            if ex_channel.is_iq():
                control_seq_id = ex_channel.parent.sequencer_id
                control_awg_id = ex_channel.parent.awg.device_id
            else:
                control_seq_id = ex_channel.channel // 2
                control_awg_id = ex_channel.parent.awg.device_id
            # print ('Setting multipulse: \npulse:', pulse[1], 'channel:', channel, 'length:', length, 'other args:', pulse[2:])
            pulses[ex_channel.parent.awg.device_id][control_seq_id] = pulse[1](channel, length_set, *pulse[2:])
        return [pulses, [control_seq_id], [control_awg_id]]

    def sin(self, channel, length, amp, freq, initial_phase, fast_control=False):
        definition_fragment = ''''''
        play_fragment = ''''''
        assign_fragment = ''''''
        entry_table_index_constants = []
        ex_channel = self.channels[channel]
        nrOfsampl = int(length * ex_channel.get_clock())-1
        nrOfPeriods = int(np.round((nrOfsampl+1)*np.abs(freq)/ex_channel.get_clock()))
        definition_fragment += textwrap.dedent('''
wave sine_wave_{samples} = sine({samples}, 1, {initial_phase}, {nrOfPeriods});'''.format(name=channel,
                                                initial_phase=initial_phase, samples=nrOfsampl, nrOfPeriods=nrOfPeriods))

        if ex_channel.is_iq():
            calib_dc = ex_channel.parent.calib_dc()
            calib_rf = ex_channel.parent.calib_rf(ex_channel)
            awg_channel = ex_channel.parent.sequencer_id
            ex_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
            ex_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
            ex_channel.parent.awg.set_offset(channel = 2 * awg_channel, offset = np.real(calib_dc['dc']))
            ex_channel.parent.awg.set_offset(channel = 2 * awg_channel + 1, offset =  np.imag(calib_dc['dc']))
            play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*sine_wave_{samples}, 2, {ampQ}*sine_wave_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp)*np.abs(calib_rf['I']), ampQ=np.imag(amp)*np.abs(calib_rf['Q'])))
            # TODO
            entry_table_index_constants.append('etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ}'''.format(
                samples=nrOfsampl, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
            assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*sine_wave_{samples}, 2, {ampQ}*sine_wave_{samples}, etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ});'''.format(
                samples=nrOfsampl, signI=(np.sign(np.real(amp))==1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                signQ=(np.sign(np.imag(amp))==1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
        else:
            control_channel_id = ex_channel.channel % 2
            if control_channel_id == 0:
                play_fragment += textwrap.dedent('''
//
    playWave(1, {ampI}*sine_wave_{samples}, 1, {ampQ}*sine_wave_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp), ampQ=np.imag(amp)))
                # TODO
                entry_table_index_constants.append('etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ}'''.format(
                    samples=nrOfsampl, signI=(np.sign(np.real(amp)) == 1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                    signQ=(np.sign(np.imag(amp)) == 1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*sine_wave_{samples}, 1, {ampQ}*sine_wave_{samples}, etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ});'''.format(
                    samples=nrOfsampl, signI=(np.sign(np.real(amp)) == 1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                    signQ=(np.sign(np.imag(amp)) == 1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

            else:
                play_fragment += textwrap.dedent('''
//
    playWave(2, {ampI}*sine_wave_{samples}, 2, {ampQ}*sine_wave_{samples});'''.format(samples=nrOfsampl, ampI=np.real(amp), ampQ=np.imag(amp)))
                # TODO
                entry_table_index_constants.append('etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ}'''.format(
                    samples=nrOfsampl, signI=(np.sign(np.real(amp)) == 1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                    signQ=(np.sign(np.imag(amp)) == 1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))
                assign_fragment += textwrap.dedent('''
assignWaveIndex(1, {ampI}*sine_wave_{samples}, 1, {ampQ}*sine_wave_{samples}, etic_sine_wave_{samples}_{signI}{realI}_{signQ}{imagQ});'''.format(
                    samples=nrOfsampl, signI=(np.sign(np.real(amp)) == 1), ampI=np.real(amp), realI=int(np.abs(np.real(amp))// 0.01),
                    signQ=(np.sign(np.imag(amp)) == 1), ampQ=np.imag(amp), imagQ=int(np.abs(np.imag(amp))// 0.01)))

        table_entry = {'index': 0}
        table_entry['waveform'] = {'index': 0}
        table_entry['phase0'] = {'value': 0.0, 'increment': True}
        table_entry['phase1'] = {'value': 90.0, 'increment': False}
        table_entry['amplitude0'] = {'value': 1.0}
        table_entry['amplitude1'] = {'value': 1.0}

        return definition_fragment, play_fragment, entry_table_index_constants, assign_fragment, table_entry

    def parallel(self, *pulse_sequences):
        # pulse_sequences = [pulses,seq_id] -
        # list of lists where first element is pulses (from pg.pmulty) and second element is control_channel
        # pulses - dict
        # seq_id - control sequencer id for pulse_sequence
        # return new list merged_pulses with pulses for each channel

        merged_pulses = {}
        merged_seq_id = []
        merged_awg_id = []
        merged_pulses = copy.copy(pulse_sequences[0][0])

        for _sequences in pulse_sequences:
            merged_pulses[_sequences[2][0]][_sequences[1][0]] = _sequences[0][_sequences[2][0]][_sequences[1][0]]
            merged_seq_id.append(_sequences[1][0])
            merged_awg_id.append(_sequences[1][0])

        return [merged_pulses, merged_seq_id, merged_awg_id]


    def readout_rect(self, channel, length, amplitude):
        # re_channel = device.awg_channels[channel]
        re_channel = self.channels[channel]
        calib_dc = re_channel.parent.calib_dc()
        calib_rf = re_channel.parent.calib_rf(re_channel)

        #nrOfsampl = int(length * re_channel.get_clock()) - 1
        #nrOfPeriods = int(np.round((nrOfsampl + 1) * np.abs(re_channel.get_if()) / re_channel.get_clock()))
        nrOfsampl = round((length) * re_channel.get_clock() / 16) * 16
        nrOfPeriods = int(np.round((nrOfsampl) * np.abs(re_channel.get_if()) / re_channel.get_clock()))
        #nrOfsampl = int(length * re_channel.get_clock())
        #nrOfPeriods = int(length * np.abs(re_channel.get_if()))
        awg_channel = re_channel.parent.sequencer_id
        re_channel.parent.awg.set_amplitude(2 * awg_channel, amplitude)
        re_channel.parent.awg.set_amplitude(2*awg_channel + 1, amplitude)
        re_channel.parent.awg.set_offset(channel=2 * awg_channel, offset=np.real(calib_dc['dc']))
        re_channel.parent.awg.set_offset(channel=2 * awg_channel + 1, offset=np.imag(calib_dc['dc']))
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
                #nrOfsampl = int(length * re_channel.get_clock())
                #nrOfPeriods = int(length * np.abs(re_channel.get_if()))

                #nrOfsampl = int(length * re_channel.get_clock()) - 1
                #nrOfPeriods = int(np.round((nrOfsampl + 1) * np.abs(re_channel.get_if()) / re_channel.get_clock()))
                nrOfsampl = round((length) * re_channel.get_clock() / 16) * 16
                nrOfPeriods = int(np.round((nrOfsampl) * np.abs(re_channel.get_if()) / re_channel.get_clock()))
                awg_channel = re_channel.parent.sequencer_id
                re_channel.parent.awg.set_amplitude(2 * awg_channel, 1)
                re_channel.parent.awg.set_amplitude(2 * awg_channel + 1, 1)
                re_channel.parent.awg.set_offset(channel=2 * awg_channel, offset=np.real(calib_dc['dc']))
                re_channel.parent.awg.set_offset(channel=2 * awg_channel + 1, offset=np.imag(calib_dc['dc']))
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
