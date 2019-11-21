from __future__ import print_function
import time
import zhinst.utils
import textwrap
import timeit
from qsweepy.instrument import Instrument

from scipy.signal import gaussian
import numpy as np


class Zurich_HDAWG1808():
    def __init__(self, device_id, config=0, clock=1e9, nop=1000):
        """
        Parameters
        ----------
        device_id
        config
        clock
        nop

        Initializes
        """

        # Settings
        apilevel_example = 6  # The API level supported by this example.
        (self.daq, self.device, _) = zhinst.utils.create_api_session(device_id, apilevel_example,
                                                                     required_devtype='HDAWG')
        zhinst.utils.api_server_version_check(self.daq)
        self.device_id = device_id
        zhinst.utils.disable_everything(self.daq, self.device)

        # self.awg_module = self.daq.awgModule()
        # self.awg_module.set('awgModule/device', self.device)
        # self.awg_module.execute()
        # self.awg_module = self.daq.awgModule()

        # 'system/awg/channelgrouping' : Configure how many independent sequencers
        #	should run on the AWG and how the outputs are grouped by sequencer.
        #	0 : 4x2 with HDAWG8; 2x2 with HDAWG4.
        #	1 : 2x4 with HDAWG8; 1x4 with HDAWG4.
        #	2 : 1x8 with HDAWG8.
        # Configure the HDAWG to use one sequencer with the same waveform on all output channels.
        self.daq.setInt('/{}/system/awg/channelgrouping'.format(self.device), config)
        self.awg_config = config
        if (self.awg_config == 0): self.num_seq = 4
        if (self.awg_config == 1): self.num_seq = 2
        if (self.awg_config == 2): self.num_seq = 1

        self._clock = clock
        self._nop = nop
        # self.nop=1000
        self.rep_rate = 10e3
        self.predelay = 0
        self.postdelay = 0
        self.wave_lengths_default = [200, 600, 2000, 6000, 20000]
        self.wave_lengths = tuple(self.wave_lengths_default)
        # self.repetition_period=

        self.amplitudes = np.ones((8))
        self.modulation = np.zeros((8), dtype=int)
        self.carrier_osc = np.zeros((8, 4), dtype=int)
        self.carrier_harm = np.zeros((8, 4), dtype=int)
        self.carrier_freq = np.zeros((8, 4), dtype=float)
        self.carrier_phase = np.zeros((8, 4), dtype=float)

        self.filter = np.zeros((8))
        self.offset = np.zeros((8))
        self.range = np.zeros((8))
        self.harm_fucktor = np.zeros((16,), dtype=int)
        self.osc_freq = np.zeros((16,), dtype=float)

        # self._waveforms=np.zeros((8,self.nop))
        # self._markers=np.zeros((8))
        self._waveforms = [None] * 8
        self.Predelay = np.zeros((4), dtype=int)
        self.Postdelay = np.zeros((4), dtype=int)
        self._markers = [None] * 8
        self._values = {}
        self._values['files'] = {}
        # self.marker_delay_I=np.zeros((8,))

        self.marker_out = np.zeros((8,), dtype=int)
        self.Marker_Out_Allowed_Values = {
            '0': "Trigger output is assigned to AWG Trigger 1, controlled by AWG sequencer commands.",
            '1': "Trigger output is assigned to AWG Trigger 2, controlled by AWG sequencer commands.",
            '2': "Trigger output is assigned to AWG Trigger 3, controlled by AWG sequencer commands.",
            '3': "Trigger output is assigned to AWG Trigger 4, controlled by AWG sequencer commands.",
            '4': "Output is assigned to Output 1 Marker 1.",
            '5': "Output is assigned to Output 1 Marker 2.",
            '6': "Output is assigned to Output 2 Marker 1.",
            '7': "Output is assigned to Output 2 Marker 2.",
            '8': "Output is assigned to Trigger Input 1.",
            '9': "Output is assigned to Trigger Input 2.",
            '10': "Output is assigned to Trigger Input 3.",
            '11': "Output is assigned to Trigger Input 4.",
            '12': "Output is assigned to Trigger Input 5.",
            '13': "Output is assigned to Trigger Input 6.",
            '14': "Output is assigned to Trigger Input 7.",
            '15': "Output is assigned to Trigger Input 8.",
            '17': "Output is set to high.",
            '18': "Output is set to low",
            }
        for channel in range(8):
            self.marker_out[channel] = 0
            self.set_marker_out(channel, 0)

        self.sampling_rate = np.zeros((8,))
        self.sin_osc_num = np.zeros((8,), dtype=int)  # maybe (8,16)
        self.sin_phase = np.zeros((8,), dtype=float)
        self.sin_amplitude = np.zeros((8, 2), dtype=float)
        self.sin_enable = np.zeros((8, 2), dtype=int)

        # Triggers
        self.trig_input_level = np.zeros((8,), dtype=float)
        self.source_dig_trig_1 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Trigger In1
        self.source_dig_trig_2 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Trigger In1
        self.slope_dig_trig_1 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Level sensitive trigger
        self.slope_dig_trig_2 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Level sensitive trigger
        self.set_dig_trig1_source(sources=self.source_dig_trig_1)
        self.set_dig_trig2_source(sources=self.source_dig_trig_2)
        self.set_dig_trig1_slope(slope=self.slope_dig_trig_1)
        self.set_dig_trig2_slope(slope=self.slope_dig_trig_2)
        self.known_programs = {}  # dictionary of AWG pograms for each sequencer

        self.initial_param_values = {'predelay': self.predelay,
                                     'postdelay': self.postdelay,
                                     'nop': self._nop,
                                     'nsupersamp': self._nop//8,
                                     'wave_lengths':self.wave_lengths,
                                     'marker_delay_I': 0,
                                     'marker_length_I': 1000,
                                     'marker_delay_Q': 0,
                                     'marker_length_Q': 1000,
                                     'rep_rate': int(self.rep_rate),
                                     'pre_delay_reg': 0,
                                     'wave_length_reg': 1,
                                     }

        self.awgModule = self.daq.awgModule()
        self.awgModule.set('awgModule/device', self.device)
        self.awgModule.execute()

        self.sequencers_updated = [False]*4
        self.frozen = False

        self.current_programs = ['' for i in range(4)]
        for sequencer_idx in range(4):
            self.set_cur_prog(self.initial_param_values, sequencer_idx)
            self.send_cur_prog(sequencer_idx)


    def clear(self):
        for sequencer_id in range(0, 4):
            self.send_cur_prog(sequencer_id)

        for waveform_id in range(8):
            self._waveforms[waveform_id] = np.zeros(self._nop)

        self.stop()

    def set_cur_prog(self, parameters, sequencer_idx):
        definition_fragments = []
        play_fragments = []

        for wave_length_id, wave_length in enumerate(self.wave_lengths):
            definition_fragments.append(textwrap.dedent('''
            wave w_marker_I_{wave_length} = join(marker(1, 1), marker({wave_length} - 1, 0));
            wave w_marker_Q_{wave_length} = join(marker(1, 2), marker({wave_length} - 1, 0));
            wave w_I_{wave_length} = zeros({wave_length}) + w_marker_I_{wave_length};
            wave w_Q_{wave_length} = zeros({wave_length}) + w_marker_Q_{wave_length};
            '''.format(wave_length = wave_length)))
            play_fragments.append(textwrap.dedent('''
            if (getUserReg({wave_length_reg}) == {wave_length_supersamp}) {{
                while(true) {{
                    waitDigTrigger(1);
                    wait(getUserReg({pre_delay_reg}));
                    playWave(w_I_{wave_length},w_Q_{wave_length});
                    waitWave();
                    wait({nsupersamp}-getUserReg({pre_delay_reg})-getUserReg({wave_length_reg}));
                    waitWave();
                }}
            }}
            ''').format(wave_length = wave_length, wave_length_supersamp = wave_length//8, **parameters))
        zero_length_program = textwrap.dedent('''
        if (getUserReg({wave_length_reg}) == 0) {{
            while(true) {{
                waitDigTrigger(1);
                wait({nsupersamp});
                waitWave();
            }}
        }}
        ''').format(**parameters)
        self.current_programs[sequencer_idx] = ''.join(definition_fragments+play_fragments)+zero_length_program

    def send_cur_prog(self, sequencer):
        awg_program = self.current_programs[sequencer]

        self._waveforms[sequencer * 2 + 0] = np.zeros(self.get_nop())
        self._waveforms[sequencer * 2 + 1] = np.zeros(self.get_nop())
        self._markers[sequencer * 2 + 0] = np.zeros(self.get_nop())
        self._markers[sequencer * 2 + 1] = np.zeros(self.get_nop())
        self._markers[sequencer * 2 + 0][0] = 1
        self._markers[sequencer * 2 + 1][0] = 1

        if (sequencer > (self.num_seq - 1)):
            print('awg_config={}. Max sequencer number ={}'.format(self.awg_config, (self.num_seq - 1)))
        self.awgModule.set('awgModule/index', sequencer)
        self.awgModule.set('awgModule/compiler/sourcestring', awg_program)
        start = timeit.default_timer()
        while self.awgModule.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
        if self.awgModule.getInt('awgModule/compiler/status') == 1:
            # compilation failed, raise an exceptionawg
            raise Exception(self.awgModule.getString('awgModule/compiler/statusstring'))
        if self.awgModule.getInt('awgModule/compiler/status') == 0:
            print("Compilation successful with no warnings, will upload the program to the instrument.")
        if self.awgModule.getInt('awgModule/compiler/status') == 2:
            print("Compilation successful with warnings, will upload the program to the instrument.")
            print("Compiler warning: ", self.awgModule.getString('awgModule/compiler/statusstring'))
        # Wait for the waveform upload to finish
        time.sleep(0.1)
        i = 0
        while (self.awgModule.getDouble('awgModule/progress') < 1.0) and (
                self.awgModule.getInt('awgModule/elf/status') != 1):
            print("{} awgModule/progress: {:.2f}".format(i, self.awgModule.getDouble('awgModule/progress')))
            time.sleep(0.1)
            i += 1
        print("{} awgModule/progress: {:.2f}".format(i, self.awgModule.getDouble('awgModule/progress')))
        if self.awgModule.getInt('awgModule/elf/status') == 0:
            print("Upload to the instrument successful.")
        if self.awgModule.getInt('awgModule/elf/status') == 1:
            raise Exception("Upload to the instrument failed.")
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        # self.daq.setInt('/' + self.device + '/awgs/%d/enable'%index, 1)
        return True

    def programs(self):
        return set(self.known_programs.keys())

    # clear all AWG programs in dict known_programs
    def _clear_all_programs(self):
        self.known_programs.clear()

    # remove select AWG programs from dict known_programs
    def remove_awg_program(self, name: str):
        # Call removal of program waveforms on WaveManger.
        self.known_programs.pop(name)

    #
    def set_clock(self, clock):
        self._clock = clock
        self.daq.set([['/' + self.device + '/SYSTEM/CLOCKS/SAMPLECLOCK/FREQ', clock]])
        self.daq.sync()

    def set_clock_source(self, source):
        self.daq.set([['/' + self.device + '/SYSTEM/CLOCKS/REFERENCECLOCK/SOURCE', source]])
        self.daq.sync()

    def get_clock(self):
        return self._clock

    def set_nop(self, numpts):
        if self._nop != numpts:
            self._nop = numpts
            self._waveforms = [None] * 8
        self.initial_param_values['nop'] = numpts
        self.initial_param_values['nsupersamp'] = numpts//8
        self.wave_lengths = tuple(self.wave_lengths_default+[numpts])
        for sequencer in range(0, 4):
            self.set_cur_prog(self.initial_param_values, sequencer)
            self.send_cur_prog(sequencer)

    def get_nop(self):
        return self._nop

    def set_repetition_period(self, repetition_period):
        self.repetition_period = repetition_period
        self.set_nop(int(repetition_period * self.get_clock()))

    def get_repetition_period(self, repetition_period):
        return self.get_nop() / self.get_clock()

    #
    # Out settings
    # out channels = physical channels on the front pannel of  the device
    # In this part of the code out_channel is called 'channel'
    def set_all_outs(self):
        for channel in range(8):
            self.daq.set([['/' + self.device + '/SIGOUTS/%s/ON' % channel, 1]])
        #self.daq.sync()

    def run(self):
        for out_seq in range(self.num_seq):
            self.daq.setInt('/' + self.device + '/awgs/%d/enable' % out_seq, 1)
        #self.set_all_outs()


    def stop(self):
        #for channel in range(8):
            #self.daq.set([['/' + self.device + '/SIGOUTS/%s/ON' % channel, 0]])
        #self.daq.sync()
        for out_seq in range(self.num_seq):
            self.daq.setInt('/' + self.device + '/awgs/%d/enable' % out_seq, 0)
        #self.daq.sync()

    def set_output(self, output, channel):
        self.daq.set([['/' + self.device + '/SIGOUTS/%d/ON' % channel, output]])
        #self.daq.sync()

    def get_output(self, channel):
        self.daq.get([['/' + self.device + '/SIGOUTS/%d/ON' % channel]])

    def stop_seq(self, sequencer):
        self.daq.setInt('/' + self.device + '/awgs/%d/enable' % sequencer, 0)
        #self.daq.sync()

    def start_seq(self, sequencer):
        self.daq.setInt('/' + self.device + '/awgs/%d/enable' % sequencer, 1)
        #self.daq.sync()

    def set_offset(self, offset, channel):
        self.offset[channel] = offset
        self.daq.set([['/' + self.device + '/SIGOUTS/%d/OFFSET' % channel, offset]])
        #self.daq.sync()

    def get_offset(self, channel):
        return self.offset[channel]

    def set_range(self, channel, range):
        self.range[channel] = range
        self.daq.set([['/' + self.device + '/SIGOUTS/%d/RANGE' % channel, range]])
        self.daq.sync()

    def get_range(self, channel):
        return self.range[channel]

    def set_filter(self, filter, channel):
        self.filter[channel] = filter
        self.daq.set([['/' + self.device + '/SIGOUTS/%d/FILTER' % channel, range]])
        self.daq.sync()

    def get_filter(self, channel):
        return filter[channel]

    # Triger settings
    # Trigger impedance
    def set_trigger_impedance(self, trigger_in, impedance):
        # Sets the trigger impedance 0 - 1 kOhm, 1 - 50 Ohm
        self.daq.set([['/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, impedance]])
        self.daq.sync()

    def set_trigger_impedance_1e3(self):
        # Sets the trigger impedance to 1 kOhm
        for trigger_in in range(8):
            self.daq.set([['/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, 0]])
            self.daq.sync()

    def set_trigger_impedance_50(self):
        # Sets the trigger impedance to 50 Ohm
        for trigger_in in range(8):
            self.daq.set([['/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, 1]])
            self.daq.sync()

    def set_trig_level(self, input_level):
        for trigger_in in range(8):
            self.trig_input_level[trigger_in] = input_level
            self.daq.set([['/' + self.device + '/triggers/in/%d/LEVEL' % trigger_in, input_level]])
            self.daq.sync()

    # Set external trigger channels

    # Allowed Values: 0 - Trigger In 1, 1 - Trigger In 2, 2 - Trigger In 3, 3 - Trigger In 4, 4 - Trigger In 5
    # 5 - Trigger In 6, 6 - Trigger In 7, 7 - Trigger In 8
    def set_dig_trig1_source(self, sources):  # sourses=self.source_dig_trig_1):
        if len(sources) != len(self.source_dig_trig_1):
            print('Could not set source for digital trigger 1. Length for sourses array should be equal', self.num_seq)
        else:
            self.source_dig_trig_1 = sources
            for num_seq in range(len(sources)):
                exp_settig = [['/' + self.device + '/AWGS/%d/auxtriggers/%d/channel' % (num_seq, 0), sources[num_seq]]]
                self.daq.set(exp_settig)
            self.daq.sync()

    def set_dig_trig2_source(self, sources):  # =self.source_dig_trig_2):
        if len(sources) != len(self.source_dig_trig_1):
            print('Could not set source for digital trigger 2. Length for sourses array should be equal', self.num_seq)
        else:
            self.source_dig_trig_2 = sources
            for num_seq in range(len(sources)):
                exp_settig = [['/' + self.device + '/AWGS/%d/auxtriggers/%d/channel' % (num_seq, 1), sources[num_seq]]]
                self.daq.set(exp_settig)
            self.daq.sync()

    # Set external trigger slope
    # Allowed Values: 0 - Level sensitive trigger, 1 - Rising edge trigger
    # 2 - Falling edge trigger, 3 - Rising or falling edge trigger
    def set_dig_trig1_slope(self, slope):
        if len(slope) != len(self.slope_dig_trig_1):
            print('Could not set slope for digital trigger 1. Length for slope array should be equal', self.num_seq)
        else:
            self.slope_dig_trig_1 = slope
            for num_seq in range(len(slope)):
                exp_settig = [['/' + self.device + '/AWGS/%d/auxtriggers/%d/slope' % (num_seq, 0), slope[num_seq]]]
                self.daq.set(exp_settig)
            self.daq.sync()

    def set_dig_trig2_slope(self, slope):
        if len(slope) != len(self.slope_dig_trig_1):
            print(
                'Could not set slope for digital trigger 2. Length for slope array should be equal number of sequencers',
                self.num_seq)
        else:
            self.slope_dig_trig_1 = slope
            for num_seq in range(len(slope)):
                exp_settig = [['/' + self.device + '/AWGS/%d/auxtriggers/%d/slope' % (num_seq, 1), slope[num_seq]]]
                self.daq.set(exp_settig)
            self.daq.sync()

    # AWG settings
    # In this part of the code channel means the number of awgs
    # for config=0

    def set_sampling_rate(self, rate, channel):
        """	0 2.4 GHz 1 1.2 GHz 2 600 MHz 3 300 MHz 4 150 MHz """
        self.sampling_rate[channel] = rate
        self.daq.set(['/' + self.device + '/AWGS/%d/TIME' % channel, rate])
        self.daq.sync()

    def get_sampling_rate(self, channel):
        return self.sampling_rate[channel]

    def set_amplitude(self, amplitude, channel):
        self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.amplitudes[channel] = amplitude
        exp_setting = [['/%s/awgs/%d/outputs/%d/amplitude' % (self.device, awg_channel, awg_out), amplitude]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_amplitude(self, channel):
        return self.amplitudes[channel]

    # Marker out settings

    def set_marker_out(self, source, channel):
        if source < 19:
            exp_setting = [['/%s/triggers/out/%d/source' % (self.device, channel), source]]
            self.daq.set(exp_setting)
            self.daq.sync()
            print(self.Marker_Out_Allowed_Values.get(str(source)))
        else:
            print('source=', source, ' is not allowed')
            print(self.Marker_Out_Allowed_Values.items())

    # Set markers
    # def set_digital(self, marker, channel):

    # If you whant to use modulation for AWG channel
    # 0 Modulation Off: AWG Output goes directly to Signal Output.
    # 1 Sine 1: AWG Output is multiplied with Sine Generator signal 0
    # 2 Sine 2: AWG Output is multiplied with Sine Generator signal 1
    # 5 Advanced: Output modulates corresponding sines from modulation carriers.

    def set_modulation(self,  mode, channel):
        self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.modulation[channel] = mode
        exp_setting = [['/%s/awgs/%d/outputs/%d/modulation/mode' % (self.device, awg_channel, awg_out), mode]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_amplitude(self, channel):
        return self.modulation[channel]
        print('0 Modulation Off: AWG Output goes directly to Signal Output.')
        print('1 Sine 1: AWG Output is multiplied with Sine Generator signal 0')
        print('2 Sine 2: AWG Output is multiplied with Sine Generator signal 1')
        print('5 Advanced: Output modulates corresponding sines from modulation carriers')

    # If you want to use multifrequency modelation, max 4 oscillators
    # def set_awg_multifreq(self, channel, osc_num):
    # return self.set_modulation(channel, mode=5)
    # self.amplitudes[channel] = amplitude
    # exp_setting = ['/%s/awgs/%d/outputs/%d/amplitude' % (self.device, awg_channel, awg_out), amplitude]
    # self.daq.set(exp_setting)
    # self.daq.sync()

    # osc_num=integer
    def set_awg_multi_osc(self, carrier, osc_num, channel):
        self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        if (osc_num > 15):
            self.carrier_osc[channel, carrier] = 15
            exp_setting = [['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/oscselect' % (
            self.device, awg_channel, awg_out, carrier), 15]]
            self.daq.sync()
            print('Warning. Out of range. Max oscillator number=15 (from 0 to 15). osc_num=15 has been installed')
        else:
            self.carrier_osc[channel, carrier] = carrier
            exp_setting = [['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/oscselect' % (
            self.device, awg_channel, awg_out, carrier), osc_num]]
            self.daq.set(exp_setting)
            self.daq.sync()

    def get_awg_multi_osc(self, carrier, channel):
        return self.carrier_osc[channel, carrier]

    # harm=integer
    def set_awg_multi_harm(self, carrier, harm, channel):
        self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.carrier_harm[channel, carrier] = harm
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/harmonic' % (self.device, awg_channel, awg_out, carrier),
             harm]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_harm(self, carrier, channel):
        return self.carrier_harm[channel, carrier]

    # freq units=Hz
    def set_awg_multi_freq(self, channel, carrier, freq):
        self.stop()
        awg_channel = channel // 4
        awg_out = channel % 4
        self.carrier_freq[channel, carrier] = freq
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/freq' % (self.device, awg_channel, awg_out, carrier), freq]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_freq(self, carrier, channel):
        return self.carrier_freq[channel, carrier]

    # phase units=deg
    def set_awg_multi_phase(self, carrier, phase, channel):
        self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.carrier_phase[channel, carrier] = phase
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/phaseshift' % (self.device, awg_channel, awg_out, carrier),
             phase]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_phase(self, carrier, channel):
        return self.carrier_phase[channel, carrier]

    # we should add to waveform(16 bits) 2 bits with information about markers(they are simply set
    # to zero by default) We use the marker function to assign the desired non-zero marker bits to the
    # wave. The marker function takes two arguments, the first is the length of the wave, the second
    # is the marker configuration in binary encoding: the value 0 stands for a both marker bits low, the
    # values 1, 2, and 3 stand for the first, the second, and both marker bits high, respectively
    def generate_waveform(self, index, waveform):
        # page 148
        wave = "wave w{0} = vect(_w_);".format(index)
        wave = wave.replace('_w_', ','.join([str(x) for x in waveform]))
        return wave

    def generate_play_waveform(self, index):
        # page 148
        wave = "playWave(w{0});".format(index)
        return wave

    def check_waveform(self):
        pass

    def write_waveform(self,  waveform, channel):
        if len(waveform) != self.get_nop():
            print('Error!!! Length of waveform should be equal', self.get_nop())
        else:
            self._waveforms[channel] = waveform

    def write_digital(self, marker, channel):
        if len(marker) != self.get_nop():
            print('Error!!! Length of marker should be equal', self.get_nop())
        else:
            self._marker[channel] = marker

    def set_waveform(self, waveform, channel):
        sequencer = channel // 2

        waveform_nop = np.zeros(self.get_nop(), dtype=float)
        waveform = waveform[:self.get_nop()]
        waveform_nop[:len(waveform)] = waveform

        if self._waveforms[channel] is not None:
            if len(waveform) == len(self._waveforms[channel]):
                if np.sum(np.abs(waveform-self._waveforms[channel]))<1e-5:
                    return

        self._waveforms[channel] = waveform
        self.set_sequencer(sequencer)

    def set_digital(self, marker, channel):
        sequencer = channel // 2

        marker_nop = np.zeros(self.get_nop(), dtype=int)
        marker = marker[:self.get_nop()]
        marker_nop[:len(marker)] = marker

        if self._markers[channel] is not None:
            if len(marker) == len(self._markers[channel]):
                if np.sum(np.abs(marker-self._markers[channel])) < 1e-5:
                    return

        self._markers[channel] = marker
        self.set_sequencer(sequencer)

    def set_sequencer(self, sequencer):
        if self.frozen:
            self.sequencers_updated[sequencer] = True
            return

        self.stop_seq(sequencer=sequencer)
        waveformI = self._waveforms[sequencer * 2 + 0]
        waveformQ = self._waveforms[sequencer * 2 + 1]
        markerI = self._markers[sequencer * 2 + 0]
        markerQ = self._markers[sequencer * 2 + 1]

        nonzero_sig = np.abs(waveformI)+np.abs(waveformQ)+np.abs(markerI)+np.abs(markerQ) > 1e-5
        nonzero = np.nonzero(nonzero_sig)[0]

        if len(nonzero) == 0:
            wave_length = 0
        else:
            first_point = int(np.floor(nonzero[0]/8)*8)
            last_point = int(np.ceil((nonzero[-1]+1)/8)*8)

            post_delay = self.get_nop() - last_point

            wave_length_nonzero = last_point - first_point
            wave_length_found = False
            for wave_length_available in self.wave_lengths:
                if wave_length_available >= wave_length_nonzero:
                    wave_length = wave_length_available
                    wave_length_found = True
                    break
            if not wave_length_found:
                raise ValueError('Waveform too long on sequencer {}'.format(sequencer))

            self.Predelay[sequencer] = self.get_nop()-post_delay-wave_length
            if self.Predelay[sequencer] < 0:
                post_delay += -self.Predelay[sequencer]
                self.Predelay[sequencer] = 0

            waveformI_truncated = waveformI[self.Predelay[sequencer]:self.Predelay[sequencer]+wave_length+1]
            waveformQ_truncated = waveformQ[self.Predelay[sequencer]:self.Predelay[sequencer]+wave_length+1]
            markerI_truncated = markerI[self.Predelay[sequencer]:self.Predelay[sequencer]+wave_length+1]
            markerQ_truncated = markerQ[self.Predelay[sequencer]:self.Predelay[sequencer]+wave_length+1]

            #self.Postdelay[sequencer] = post_delay

            ch1 = np.asarray(waveformI_truncated * (2 ** 13 - 1), dtype=np.int16)  # TODO check if this sets to 0 or to 0.001
            ch2 = np.asarray(waveformQ_truncated * (2 ** 13 - 1), dtype=np.int16)
            m1 = np.asarray(markerI_truncated, dtype=np.uint16)
            m2 = 2 * np.asarray(markerQ_truncated, dtype=np.uint16)
            vector = np.asarray(np.transpose([ch1, ch2]).ravel())
            markers = np.asarray(np.transpose([m1, m2]).ravel())
            vector = (vector << 2 | markers).astype('int16')

            self.daq.setInt('/' + self.device + '/awgs/%d/waveform/index' % sequencer, self.wave_lengths.index(wave_length))
            # self.daq.sync()
            self.daq.vectorWrite('/' + self.device + '/awgs/%d/waveform/data' % sequencer, vector)
            # self.daq.sync()

            self.daq.setInt('/' + self.device + '/awgs/%d/single' % sequencer, 0)
            # self.daq.setInt('/' + self.device + '/awgs/%d/enable'%sequencer, 1)
            self.daq.sync()

            print ('first_point: ', first_point,
               'last_point: ', last_point,
               'waveformI length', waveformI.shape,
               'waveformQ length', waveformQ.shape,
               'markerI length', len(markerI),
               'markerQ length', len(markerQ),
               'wave_length', wave_length,
               'nop', self.get_nop(),
               'post_delay', post_delay,
               'pre_delay', self.Predelay[sequencer],
               'pre_delay_reg', int(self.Predelay[sequencer]//8),
            )

        self.daq.setInt('/{device}/awgs/{sequencer}/userregs/{pre_delay_reg}'.format(device = self.device,
                sequencer=sequencer, pre_delay_reg = self.initial_param_values['pre_delay_reg']),
                        int(self.Predelay[sequencer]//8))
        self.daq.setInt('/{device}/awgs/{sequencer}/userregs/{wave_length_reg}'.format(device = self.device,
                sequencer=sequencer, wave_length_reg = self.initial_param_values['wave_length_reg']),
                        wave_length//8)
        time.sleep(0.3)

    def get_waveform(self, channel):
        return self._waveforms[channel]



    # self.set_output(channel=channel1, output=1)
    # self.set_output(channel=channel2, output=1)
    # exp_setting = [['/%s/sigouts/%d/on'%(self.device, channel), 1]]
    # self.daq.set(exp_setting)
    # self.daq.sync()

    def get_digital(self, channel):
        return self._markers[channel]

    # Send waveform to the device

    # Osccillator settings
    # Oscillator frequency
    def set_frequency(self, osc_num, freq):
        self.osc_freq[osc_num] = freq
        self.daq.set([['/' + self.device + '/OSCS/%d/FREQ' % osc_num, freq]])
        self.daq.sync()

    def get_frequency(self, osc_num):
        return self.osc_freq[osc_num]

    # Sines settings
    #
    # Harmonic choose for sin signal from oscillator
    def set_harmonic(self, sin_num, harmonic):
        self.harm_fucktor[sin_num] = harmonic
        self.daq.set([['/' + self.device + '/SINES/%d/HARMONIC' % sin_num, harmonic]])
        self.daq.sync()

    def get_harmonic(self, sin_num):
        return self.harm_fucktor[sin_num]

    # Oscillator number choosing for sine signal generation type sin_num=integer, type osc_num=integer
    def set_sin_osc(self, sin_num, osc_num):
        if (osc_num > 15):
            self.sin_osc_num[sin_num] = 15
            self.daq.set([['/' + self.device + '/SINES/%d/OSCSELECT' % sin_num, 15]])
            self.daq.sync()
            print('Warning. Out of range. Max oscillator number=15 (from 0 to 15). osc_num=15 has been installed')

        else:
            self.sin_osc_num[sin_num] = osc_num
            self.daq.set([['/' + self.device + '/SINES/%d/OSCSELECT' % sin_num, osc_num]])
            self.daq.sync()

    def get_sin_osc(self, sin_num):
        return self.sin_osc_num[sin_num]

    # Phaseshift for oscillator
    def set_sin_phase(self, sin_num, phase):
        self.sin_phase[sin_num] = pohase
        self.daq.set([['/' + self.device + '/SINES/%d/PHASESHIFT' % sin_num, phase]])
        self.daq.sync()

    def get_sin_phase(self, sin_num):
        return self.sin_phase[sin_num]

    # Amplitudes
    def set_sin_amplitudes(self, sin_num, wave_output, amplitude):
        self.sin_amplitude[sin_num, wave_output] = amplitude
        self.daq.set([['/' + self.device + '/SINES/%d/AMPLITUDES/%d' % (sin_num, wave_output), amplitude]])
        self.daq.sync()

    def get_sin_amplitudes(self, sin_num):
        return self.sin_amplitude[sin_num, :]

    # Enables

    def set_sin_amplitudes(self, sin_num, wave_output, enable):
        self.sin_enable[sin_num, wave_output] = enable
        self.daq.set([['/' + self.device + '/SINES/%d/ENABLES/%d' % (sin_num, wave_output), enable]])
        self.daq.sync()

    def get_sin_amplitudes(self, sin_num):
        return self.sin_enable[sin_num, :]

    def freeze(self):
        if not self.frozen:
            self.sequencers_updated = [False]*4
            self.frozen = True

    def unfreeze(self):
        if self.frozen:
            self.frozen = False
            for sequencer, updated in enumerate(self.sequencers_updated):
                if updated:
                    self.set_sequencer(sequencer)

class Zurich_sequencer():

    def __init__(self, channels={}):
        self.channels = channels
        self.settings = {}

    ## generate waveform of a gaussian pulse with quadrature phase mixin
    def gauss_hd(self, channel, length, amp_x, sigma, alpha=0.):
        gauss = gaussian(int(round(length * self.channels[channel].get_clock())),
                         sigma * self.channels[channel].get_clock())
        gauss -= gauss[0]
        gauss /= np.max(gauss)
        gauss_der = np.gradient(gauss) * self.channels[channel].get_clock()
        return amp_x * (gauss + 1j * gauss_der * alpha)
