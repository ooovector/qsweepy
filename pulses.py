from scipy.signal import gaussian
from scipy.signal import tukey
from scipy.signal import hann
import numpy as np


class vz:
    def __init__(self, channel, length, phi):
        self.phi = phi
    def is_vz(self):
        return True

class vf:
    def __init__(self, channel, length, freq):
        self.freq = freq
    def is_vf(self):
        return True


class offset:
    def __init__(self, channel, length, offset):
        self.offset = offset
    def is_offset(self):
        return True


class pulses:
    def __init__(self, channels={}):
        self.channels = channels
        self.settings = {}

        self.initial_delay = 1e-6
        self.final_delay = 1e-6
        self.global_pre = [self.p(None, self.initial_delay, None)]
        self.global_post = [self.p(None, self.final_delay, None)]

    ## generate waveform of a gaussian pulse with quadrature phase mixin
    def gauss_hd(self, channel, length, amp_x, sigma, alpha=0.):
        gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
                         sigma * self.channels[channel].get_clock())
        gauss -= gauss[0]
        gauss /= np.max(gauss)
        gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        return amp_x * (gauss + 1j * gauss_der * alpha)

    # def rect_cos (self, channel, length, amp, alpha=0.):
    # alfa = 0.5
    # impulse = tukey(int(round(length*self.channels[channel].get_clock())), alfa)
    # #print(alfa*self.channels[channel].get_clock())
    # #print(length)
    # #print(round(length*self.channels[channel].get_clock()))
    # impulse -= impulse[0]
    # impulse_der = np.gradient(impulse)*self.channels[channel].get_clock()
    # return amp*(impulse + 1j*impulse_der*alpha)

    def envelope(self, channel, length, function, impulse):
        t = np.linspace(0, self.channels[channel].get_nop() / self.channels[channel].get_clock(),
                        self.channels[channel].get_nop(), endpoint=False)
        # time_arr = [self.channels[channel].get_clock()*i for i in range(int(round(length*self.channels[channel].get_clock())))]
        # print(function(time_arr[0]))
        return np.asarray(impulse * function(
            t))  # (1/self.channels[channel].get_clock()*np.arange(len(impulse))))# for i in range(len(impulse))], dtype = complex)

    def rect_cos(self, channel, length, amp, length_tail, function_for_envelope=lambda x: 1, alpha=0.):
        length_of_plato = length - length_tail * 2
        length_of_one_tail = int(length_tail * self.channels[channel].get_clock())
        hann_function = hann(2 * length_of_one_tail)
        first = hann_function[:length_of_one_tail]
        second = hann_function[length_of_one_tail:]
        plato = np.ones(int(round(length_of_plato * self.channels[channel].get_clock())))
        final = first.tolist()
        final.extend(plato.tolist())
        final.extend(second.tolist())
        impulse = np.asarray(final)
        impulse -= impulse[0]
        impulse = self.envelope(channel, length, function_for_envelope, impulse)
        # print(np.real(impulse)[50:])
        impulse_der = np.gradient(impulse) * self.channels[channel].get_clock()
        # print(self.channels[channel].get_clock())
        # print(length_tail*self.channels[channel].get_clock())
        # print(first)
        # print(second)
        # print(plato)
        return amp * (impulse + 1j * impulse_der * alpha)

    def sin(self, channel, length, amplitude, frequency):
        return amplitude * np.sin(2*np.pi*frequency*np.arange(0, length, 1/self.channels[channel].get_clock()))

    ## generate waveform of a rectangular pulse
    def rect(self, channel, length, amplitude):
        return amplitude * np.ones(int(round(length * self.channels[channel].get_clock())), dtype=np.complex)

    def pause(self, channel, length):
        return self.rect(channel, length, 0)

    def p(self, channel, length, pulse_type=None, *params):
        pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        if channel:
            pulses[channel] = pulse_type(channel, length, *params)
        return pulses

    def ps(self, channel, length, pulse_type=None, *params):
        pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        if channel:
            pulses[channel] = pulse_type(channel, length, *params)
        return pulses

    def pmulti(self, length, *params):
        pulses = {channel_name: self.pause(channel_name, length) for channel_name, channel in self.channels.items()}
        for pulse in params:
            channel = pulse[0]
            # print ('Setting multipulse: \npulse:', pulse[1], 'channel:', channel, 'length:', length, 'other args:', pulse[2:])
            pulses[channel] = pulse[1](channel, length, *pulse[2:])
        return pulses

    def parallel(self, *pulse_sequences):
        current_pulse = [0]*len(pulse_sequences)
        sequence_lengths = [len(sequence) for sequence in pulse_sequences]

        merged_sequence = []
        depleted = [False] * len(pulse_sequences)
        while not np.all(depleted):
            physical_sequence = [False]*len(pulse_sequences)
            max_physical_sequence_length = 0
            for sequence_id, sequence in enumerate(pulse_sequences):
                if len(sequence) <= current_pulse[sequence_id]:
                    depleted[sequence_id] = True # skip this sequence of there are no pulses in it left
                    continue
                for channel_name, channel_pulse in sequence[current_pulse[sequence_id]].items():
                    if hasattr(channel_pulse, 'shape'):
                        if channel_pulse.shape[0] > 0:
                            physical_sequence[sequence_id] = True
                            if max_physical_sequence_length < channel_pulse.shape[0]/self.channels[channel_name].get_clock():
                                max_physical_sequence_length = channel_pulse.shape[0]/self.channels[channel_name].get_clock()
            # if there are virtual gates pending, do them
            if not np.all(np.logical_or(physical_sequence,depleted)):
                # take the first virtual pulse from the sequences
                sequence_id = np.arange(len(pulse_sequences))[np.logical_not(np.logical_or(physical_sequence, depleted))][0]
                merged_sequence.append(pulse_sequences[sequence_id][current_pulse[sequence_id]])
                current_pulse[sequence_id] += 1
            else:
                pulse = self.pmulti(max_physical_sequence_length)
                for sequence_id, sequence in enumerate(pulse_sequences):
                    if depleted[sequence_id]:
                        continue
                    for channel_name, channel_pulse in sequence[current_pulse[sequence_id]].items():
                        pulse[channel_name][-len(channel_pulse):] += channel_pulse
                    current_pulse[sequence_id] += 1
                merged_sequence.append(pulse)
        return merged_sequence

    def awg(self, channel, length, waveform):
        return waveform

    def set_seq(self, seq, force=True):
        from time import time
        pulse_seq_padded = self.global_pre + seq + self.global_post
        try:
            for channel, channel_device in self.channels.items():
                channel_device.freeze()
            virtual_phase = {k: 0 for k in self.channels.keys()}
            df = {k: 0 for k in self.channels.keys()}
            offsets = {k: 0 for k in self.channels.keys()}
            pulse_shape = {k: [] for k in self.channels.keys()}
            for channel, channel_device in self.channels.items():
                for pulse in pulse_seq_padded:
                    if hasattr(pulse[channel], 'is_vz'):
                        virtual_phase[channel] += pulse[channel].phi
                        continue
                    if hasattr(pulse[channel], 'is_vf'):
                        df[channel] = pulse[channel].freq
                        continue
                    if hasattr(pulse[channel], 'is_offset'):
                        offsets[channel] = pulse[channel].offset
                        continue
                    # print (channel, df[channel])
                    pulse_shape[channel].extend(pulse[channel] * np.exp(1j * (
                                virtual_phase[channel] + 2 * np.pi * df[channel] / self.channels[
                            channel].get_clock() * np.arange(len(pulse[channel])))) + offsets[channel])
                    virtual_phase[channel] += 2 * np.pi * df[channel] / self.channels[channel].get_clock() * len(
                        pulse[channel])
                pulse_shape[channel] = np.asarray(pulse_shape[channel])

                if len(pulse_shape[channel]) > channel_device.get_nop():
                    tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
                    tmp = pulse_shape[channel][-channel_device.get_nop():]
                    pulse_shape[channel] = tmp
                    raise (ValueError('pulse sequence too long'))
                else:
                    tmp = np.zeros(channel_device.get_nop(), dtype=pulse_shape[channel].dtype)
                    tmp[-len(pulse_shape[channel]):] = pulse_shape[channel]
                    pulse_shape[channel] = tmp
                # print (channel, pulse_shape[channel], len(pulse_shape[channel]))
                # print ('Calling set_waveform on device '+channel)
                # setter_start = time()
                channel_device.set_waveform(pulse_shape[channel])
        # print ('channel {} time: {}'.format(channel, time() - setter_start))
        finally:
            for channel, channel_device in self.channels.items():
                # setter_start = time()
                channel_device.unfreeze()
        # print ('channel {} unfreeze time: {}'.format(channel, time() - setter_start))

        self.last_seq = seq
        devices = []
        for channel in self.channels.values():
            devices.extend(channel.get_physical_devices())
        for device in list(set(devices)):
            device.run()
