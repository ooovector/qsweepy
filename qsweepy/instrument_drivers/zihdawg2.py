from __future__ import print_function
import time
import zhinst.utils
import textwrap
import timeit
import traceback
import logging
#from qsweepy.libraries.instrument import Instrument
import numpy as np


class ZIDevice():
    def __init__(self, device_id, devtype, config=0, clock=2.40e9, nop=1000, delay_int=4e-6):
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
        (self.daq, self.device, _) = zhinst.utils.create_api_session(device_id, apilevel_example)
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
        self.set_oscillator_control(1)
        self.awg_config = config
        self.devtype = devtype
        if devtype == 'HDAWG':
            if (self.awg_config == 0): self.num_seq = 4
            if (self.awg_config == 1): self.num_seq = 2
            if (self.awg_config == 2): self.num_seq = 1
            self.num_channels = 8
        elif devtype == 'UHF':
            self.num_seq = 1
            self.num_channels = 2
        else:
            self.num_channels = None
            raise ValueError('devtype not recognized')

        self._clock = clock

        self.marker_out = np.zeros((self.num_channels,), dtype=int)
        self.modulation = np.zeros((self.num_channels,), dtype=int)
        if devtype == 'HDAWG':
            self.mk_out_allowed = {
                # Select the signal assigned to the marker output
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
        elif devtype == 'UHF':
            self.mk_out_allowed = {
                # Select the signal assigned to the trigger output
                '0': 'The output trigger is disabled',
                '1': 'Oscillator phase of demod 4 (trigger output channel 1) or demod 8 (trigger output channel 2). \
                Trigger event is output for each zero crossing of the oscillator phase.',
                '2': 'Scope Trigger',
                '3': 'Scope/Trigger',
                '4': 'Scope Armed',
                '5': 'Scope/Armed',
                '6': 'Scope Active',
                '7': 'Scope/Active',
                '8': 'AWG Marker 1',
                '9': 'AWG Marker 2',
                '10': 'AWG Marker 3',
                '11': 'AWG Marker 4',
                '20': 'AWG Active',
                '21': 'AWG Waiting',
                '22': 'AWG Fetching',
                '23': 'AWG Playing',
                '32': 'AWG Trigger 1',
                '33': 'AWG Trigger 2',
                '34': 'AWG Trigger 3',
                '35': 'AWG Trigger 4',
                '51': 'MDS Clock Out',
                '52': 'MDS Sync Out',
                }
        else:
            raise ValueError('devtype not recognized')
        for channel in range(self.num_channels):
            self.marker_out[channel] = 0
            self.set_marker_out(channel, 0)

        # Triggers
        self.trig_input_level = np.zeros((self.num_channels, ), dtype=float)
        self.source_dig_trig_1 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Trigger In1
        self.source_dig_trig_2 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Trigger In1
        self.slope_dig_trig_1 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Level sensitive trigger
        self.slope_dig_trig_2 = np.zeros((self.num_seq,), dtype=int)  # defolt values 0 = Level sensitive trigger
        self.set_dig_trig1_source(sources=self.source_dig_trig_1)
        self.set_dig_trig2_source(sources=self.source_dig_trig_2)
        self.set_dig_trig1_slope(slope=self.slope_dig_trig_1)
        self.set_dig_trig2_slope(slope=self.slope_dig_trig_2)
        self.known_programs = {}  # dictionary of AWG pograms for each sequencer


    def set_sequence(self, sequencer_id, sequence):
        #awg_program = self.current_programs[sequencer_id]
        awg_program = sequence.zicode()
        #print(sequence.zicode())

        if (sequencer_id > (self.num_seq - 1)):
            print('Sequencer #{}: awg_config={}. Max sequencer number ={}'.format(sequencer_id, self.awg_config, (self.num_seq - 1)))
        self.awgModule = self.daq.awgModule()
        self.awgModule.set('device', self.device)
        self.awgModule.set('index', sequencer_id)
        self.awgModule.execute()
        self.awgModule.set('awg/enable', 1);
        self.awgModule.set('awg/enable', 0);
        self.awgModule.set('compiler/sourcestring', awg_program)

        start = timeit.default_timer()
        while self.awgModule.getInt('compiler/status') == -1:
            time.sleep(0.1)
        if self.awgModule.getInt('compiler/status') == 1:
            # compilation failed, raise an exceptionawg
            raise Exception(self.awgModule.getString('compiler/statusstring'))
        if self.awgModule.getInt('compiler/status') == 0:
            print("Sequencer #{}: Compilation successful with no warnings, will upload the program to the instrument.".format(sequencer_id))
        if self.awgModule.getInt('compiler/status') == 2:
            print("Sequencer #{}: Compilation successful with warnings, will upload the program to the instrument.")
            print("Sequencer #{}: Compiler warning: ", self.awgModule.getString('compiler/statusstring'))
        # Wait for the waveform upload to finish
        time.sleep(0.1)
        i = 0
        while (self.awgModule.getDouble('awgModule/progress') < 1.0) and (
                self.awgModule.getInt('awgModule/elf/status') != 1):
            #print("{} awgModule/progress: {:.2f}".format(i, self.awgModule.getDouble('awgModule/progress')))
            time.sleep(0.1)
            i += 1
        print("Sequencer #{}: {} awgModule/progress: {:.2f}".format(sequencer_id, i, self.awgModule.getDouble('awgModule/progress')))
        if self.awgModule.getInt('awgModule/elf/status') == 0:
            print("Sequencer #{}: Upload to the instrument successful.".format(sequencer_id))
        if self.awgModule.getInt('awgModule/elf/status') == 1:
            raise Exception("Upload to the instrument failed.")
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        # self.daq.setInt('/' + self.device + '/awgs/%d/enable'%index, 1)
        self.daq.sync()
        return True

    def set_waveform_indexed(self, sequencer_id, waveform_index, waveform1 = None, waveform2 = None):
        factor = 2 ** 15-1
        if waveform1 is None and waveform2 is None:
            print('Warning. Zero waveforms.There are no waveforms for loading to AWG. Nothing happens!!!!')
        elif waveform1 is None:
            ch1 = np.zeros(len(waveform2), dtype=np.int16)
            ch2 = np.asarray(waveform2 * factor, dtype=np.int16)
        elif waveform2 is None:
            ch1 = np.asarray(waveform1 * factor, dtype=np.int16)
            ch2 = np.zeros(len(waveform1), dtype=np.int16)
        else:
            ch1 = np.asarray(waveform1 * factor, dtype=np.int16)
            ch2 = np.asarray(waveform2 * factor, dtype=np.int16)

        vector = zhinst.utils.convert_awg_waveform(wave1=ch1, wave2=ch2)
        self.daq.setVector('/' + self.device + '/awgs/{}/waveform/waves/{}'.format(sequencer_id, waveform_index), vector)
        self.daq.sync()

    def load_instructions(self, sequencer_id, json_str):
        if not len(json_str) % 2:
            json_str += '\n'
        self.daq.setVector('/' + self.device + '/awgs/{}/commandtable/data'.format(sequencer_id), json_str)
        self.daq.sync()
        return

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
        self.daq.setDouble('/' + self.device + '/SYSTEM/CLOCKS/SAMPLECLOCK/FREQ', clock)
        self.daq.sync()

    def set_clock_source(self, source):
        self.daq.setDouble('/' + self.device + '/SYSTEM/CLOCKS/REFERENCECLOCK/SOURCE', source)
        self.daq.sync()

    def get_clock(self):
        return self.daq.getDouble('/' + self.device + '/SYSTEM/CLOCKS/SAMPLECLOCK/FREQ')

    #
    # Out settings
    # out channels = physical channels on the front pannel of  the device
    # In this part of the code out_channel is called 'channel'
    def set_all_outs(self):
        for channel in range(self.num_channels):
            self.daq.setDouble('/' + self.device + '/SIGOUTS/%s/ON' % channel, 1)
        #self.daq.sync()

    def run(self):
        for out_seq in range(self.num_seq):
            self.daq.setInt('/' + self.device + '/awgs/%d/enable' % out_seq, 1)
        #self.set_all_outs()
        self.daq.sync()

    def stop(self):
        for out_seq in range(self.num_seq):
            self.daq.setInt('/' + self.device + '/awgs/%d/enable' % out_seq, 0)
        self.daq.sync()

    def set_output(self, channel, output):
        self.daq.setDouble('/' + self.device + '/SIGOUTS/%d/ON' % channel, output)
        self.daq.sync()

    def get_output(self, channel):
        return self.daq.getDouble('/' + self.device + '/SIGOUTS/%d/ON' % channel)

    def stop_seq(self, sequencer):
        self.daq.setInt('/' + self.device + '/awgs/%d/enable' % sequencer, 0)
        self.daq.sync()

    def start_seq(self, sequencer):
        self.daq.setInt('/' + self.device + '/awgs/%d/enable' % sequencer, 1)
        self.daq.sync()

    def set_offset(self, channel, offset):
        self.daq.setDouble('/' + self.device + '/SIGOUTS/%d/OFFSET' % channel, offset)
        self.daq.sync()

    def get_offset(self, channel):
        return self.daq.getDouble('/' + self.device + '/SIGOUTS/%d/OFFSET' % channel)

    def set_range(self, channel, range):
        self.daq.setDouble('/' + self.device + '/SIGOUTS/%d/RANGE' % channel, range)
        self.daq.sync()

    def get_range(self, channel):
        return self.daq.getDouble('/' + self.device + '/SIGOUTS/%d/RANGE' % channel)

    def set_filter(self, channel, filter):
        self.daq.set('/' + self.device + '/SIGOUTS/%d/FILTER' % channel, filter)
        self.daq.sync()

    def get_filter(self, channel):
        return self.daq.get('/' + self.device + '/SIGOUTS/%d/FILTER' % channel)

    # Triger settings
    # Trigger impedance
    def set_trigger_impedance(self, trigger_in, impedance):
        # Sets the trigger impedance 0 - 1 kOhm, 1 - 50 Ohm
        self.daq.setInt('/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, impedance)
        self.daq.sync()

    def set_trigger_impedance_1e3(self):
        # Sets the trigger impedance to 1 kOhm
        for trigger_in in range(self.num_channels):
            self.daq.setInt('/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, 0)
            self.daq.sync()

    def set_trigger_impedance_50(self):
        # Sets the trigger impedance to 50 Ohm
        for trigger_in in range(self.num_channels):
            self.daq.setInt('/' + self.device + '/triggers/in/%d/IMP50' % trigger_in, 1)
            self.daq.sync()

    def set_trig_level(self, input_level):
        for trigger_in in range(self.num_channels):
            self.trig_input_level[trigger_in] = input_level
            self.daq.setDouble('/' + self.device + '/triggers/in/%d/LEVEL' % trigger_in, input_level)
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

    def set_sampling_rate(self, channel, rate):
        """	0 2.4 GHz 1 1.2 GHz 2 600 MHz 3 300 MHz 4 150 MHz """
        self.daq.setDouble('/' + self.device + '/AWGS/%d/TIME' % channel, rate)
        self.daq.sync()

    def get_sampling_rate(self, channel):
        return self.daq.get('/' + self.device + '/AWGS/%d/TIME' % channel)

    def set_amplitude(self, channel, amplitude):
        # self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.daq.setDouble('/%s/awgs/%d/outputs/%d/amplitude' % (self.device, awg_channel, awg_out), amplitude)
        self.daq.sync()

    def set_wave_amplitude(self, channel, awg_wave, amplitude):
        # self.stop()
        awg_channel = channel // 2
        awg_out = channel % 2
        self.daq.setDouble('/%s/awgs/%d/outputs/%d/gains/%d' % (self.device, awg_channel, awg_out, awg_wave), amplitude)
        self.daq.sync()


    def get_amplitude(self, channel):
        awg_channel = channel // 2
        awg_out = channel % 2
        return self.daq.getDouble('/%s/awgs/%d/outputs/%d/amplitude' % (self.device, awg_channel, awg_out))

    # Marker out settings

    def set_marker_out(self, channel, source):
        if str(source) in self.mk_out_allowed.keys():
            exp_setting = [['/%s/triggers/out/%d/source' % (self.device, channel), source]]
            self.daq.set(exp_setting)
            self.daq.sync()
            print(self.mk_out_allowed.get(str(source)))
        else:
            print('source=', source, ' is not allowed')
            print(self.mk_out_allowed.items())

    # Set markers
    # def set_digital(self, marker, channel):

    # If you want to use modulation for AWG channel

    # for HDAWG:
    # 0 Modulation Off: AWG Output goes directly to Signal Output.
    # 1 Sine 11: AWG Outputs 0 and 1 are both multiplied with Sine Generator signal 0
    # 2 Sine 22: AWG Output 0 and 1 are both multiplied with Sine Generator signal 1
    # 3 Sine 21: AWG Outputs 0 and 1 are multiplied with Sine Generator signal 1 and 0, respectively
    # 5 Advanced: Output modulates corresponding sines from modulation carriers.

    # for UHFQA:
    # 0 Plain: AWG Output goes directly to Signal Output
    # 1 Modulation: AWG Output 1 (2) is multiplied with oscillator signal of demodulator 4(8)
    def set_holder(self, channel, holder=0):
        # holder 0 - there is no holder for last point
        # holder 1 - the device will keep the last point
        awg_channel = channel // 2
        awg_out = channel % 2
        self.daq.setInt('/' + self.device + '/awgs/%d/outputs/%d/hold'%(awg_channel, awg_out), holder)

    def set_oscillator_control(self, control):
        self.daq.setInt('/' + self.device + '/system/awg/oscillatorcontrol', control)
        self.daq.sync()


    def set_modulation(self, channel, mode):

        awg_channel = channel // 2
        awg_out = channel % 2
        self.modulation[channel] = mode
        self.stop_seq(awg_channel)
        if self.devtype == 'HDAWG':
            exp_setting = [['/%s/awgs/%d/outputs/%d/modulation/mode' % (self.device, awg_channel, awg_out), mode]]
        elif self.devtype == 'UHF':
            exp_setting = [['/%s/awgs/%d/outputs/%d/mode' % (self.device, awg_channel, awg_out), mode]]
        else:
            exp_setting = None
            raise ValueError('devtype not recognized')
        self.daq.set(exp_setting)
        self.daq.sync()

    def set_register(self, sequencer_id: int, register_id: int, value: int):
        self.daq.setInt('/' + self.device + '/awgs/%d/userregs/%d' % (sequencer_id, register_id), value)
        self.daq.sync()

    def get_modulation(self, channel):
        if self.devtype == 'HDAWG':
            print('0 Modulation Off: AWG Output goes directly to Signal Output.')
            print('1 Sine 11: AWG Outputs 0 and 1 are both multiplied with Sine Generator signal 0')
            print('2 Sine 22: AWG Output 0 and 1 are both multiplied with Sine Generator signal 1')
            print('3 Sine 21: AWG Outputs 0 and 1 are multiplied with Sine Generator signal 1 and 0, respectively')
            print('5 Advanced: Output modulates corresponding sines from modulation carriers')
        elif self.devtype == 'UHF':
            print('0 Plain: AWG Output goes directly to Signal Output.')
            print('1 Modulation: AWG Output 1 (2) is multiplied with oscillator signal of demodulator 4(8)')
        else:
            raise ValueError('devtype not recognized')
        if self.devtype == 'HDAWG':
            return self.daq.getInt('/%s/awgs/%d/outputs/%d/modulation/mode' % (self.device, channel // 2, channel % 2))
        elif self.devtype == 'UHF':
            return self.daq.getInt('/%s/awgs/0/outputs/%d/mode' % (self.device, channel % 2))
        else:
            raise ValueError('devtype not recognized')

    # If you want to use multifrequency modelation, max 4 oscillators
    # def set_awg_multifreq(self, channel, osc_num):
    # return self.set_modulation(channel, mode=5)
    # self.amplitudes[channel] = amplitude
    # exp_setting = ['/%s/awgs/%d/outputs/%d/amplitude' % (self.device, awg_channel, awg_out), amplitude]
    # self.daq.set(exp_setting)
    # self.daq.sync()

    def set_awg_multi_osc(self, channel, carrier, osc_num):
        awg_channel = channel // 2
        awg_out = channel % 2

        exp_setting = [['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/oscselect' % (
        self.device, awg_channel, awg_out, carrier), osc_num]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_osc(self, channel, carrier):
        awg_channel = channel // 2
        awg_out = channel % 2
        return self.daq.getInt('/%s/awgs/%d/outputs/%d/modulation/carriers/%d/oscselect' % (self.device, awg_channel,
                                                                                         awg_out, carrier))

    # harm=integer
    def set_awg_multi_harm(self, channel, carrier, harm):
        awg_channel = channel // 2
        awg_out = channel % 2
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/harmonic' % (self.device, awg_channel, awg_out, carrier),
             harm]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_harm(self, channel, carrier):
        awg_channel = channel // 2
        awg_out = channel % 2
        return self.daq.getInt('/%s/awgs/%d/outputs/%d/modulation/carriers/%d/harmonic' % (self.device, awg_channel,
                                                                                        awg_out, carrier))

    # freq units=Hz
    def set_awg_multi_freq(self, channel, carrier, freq):
        awg_channel = channel // 4
        awg_out = channel % 4
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/freq' % (self.device, awg_channel, awg_out, carrier), freq]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_freq(self, channel, carrier):
        awg_channel = channel // 4
        awg_out = channel % 4
        return self.daq.getDouble('/%s/awgs/%d/outputs/%d/modulation/carriers/%d/freq' % (self.device, awg_channel,
                                                                                    awg_out, carrier))

    # phase units=deg
    def set_awg_multi_phase(self, channel, carrier, phase):
        awg_channel = channel // 2
        awg_out = channel % 2
        exp_setting = [
            ['/%s/awgs/%d/outputs/%d/modulation/carriers/%d/phaseshift' % (self.device, awg_channel, awg_out, carrier),
             phase]]
        self.daq.set(exp_setting)
        self.daq.sync()

    def get_awg_multi_phase(self, channel, carrier):
        awg_channel = channel // 2
        awg_out = channel % 2
        return self.daq.getDouble('/%s/awgs/%d/outputs/%d/modulation/carriers/%d/phaseshift' % (self.device, awg_channel,
                                                                                          awg_out, carrier))

    # Osccillator settings
    # Oscillator frequency
    def set_frequency(self, osc_num, freq):
        self.daq.set([['/' + self.device + '/OSCS/%d/FREQ' % osc_num, freq]])
        self.daq.sync()

    def get_frequency(self, osc_num):
        return self.daq.getDouble('/' + self.device + '/OSCS/%d/FREQ' % osc_num)

    # Sines settings
    #
    # Harmonic choose for sin signal from oscillator
    def set_harmonic(self, sin_num, harmonic):
        self.daq.set([['/' + self.device + '/SINES/%d/HARMONIC' % sin_num, harmonic]])
        self.daq.sync()

    def get_harmonic(self, sin_num):
        return self.daq.getInt('/' + self.device + '/SINES/%d/HARMONIC' % sin_num)

    # Oscillator number choosing for sine signal generation type sin_num=integer, type osc_num=integer
    def set_sin_osc(self, sin_num, osc_num):
        self.daq.set([['/' + self.device + '/SINES/%d/OSCSELECT' % sin_num, osc_num]])
        self.daq.sync()

    def get_sin_osc(self, sin_num):
        return self.daq.getInt('/' + self.device + '/SINES/%d/OSCSELECT' % sin_num)

    # Phaseshift for oscillator
    def set_sin_phase(self, sin_num, phase):
        self.daq.set([['/' + self.device + '/SINES/%d/PHASESHIFT' % sin_num, phase]])
        self.daq.sync()

    def get_sin_phase(self, sin_num):
        return self.daq.getDouble('/' + self.device + '/SINES/%d/PHASESHIFT' % sin_num)

    # Amplitudes
    def set_sin_amplitude(self, sin_num, wave_output, amplitude):
        self.daq.set([['/' + self.device + '/SINES/%d/AMPLITUDES/%d' % (sin_num, wave_output), amplitude]])
        self.daq.sync()

    def get_sin_amplitude(self, sin_num, wave_output):
        return self.daq.getDouble('/' + self.device + '/SINES/%d/AMPLITUDES/%d' % (sin_num, wave_output))

    # Enables
    def set_sin_enable(self, sin_num, wave_output, enable):
        self.daq.set([['/' + self.device + '/SINES/%d/ENABLES/%d' % (sin_num, wave_output), enable]])
        self.daq.sync()

    def get_sin_enable(self, sin_num, wave_output):
        return self.daq.getInt('/' + self.device + '/SINES/%d/ENABLES/%d' % (sin_num, wave_output))

    # we should add to waveform(16 bits) 2 bits with information about markers(they are simply set
    # to zero by default) We use the marker function to assign the desired non-zero marker bits to the
    # wave. The marker function takes two arguments, the first is the length of the wave, the second
    # is the marker configuration in binary encoding: the value 0 stands for a both marker bits low, the
    # values 1, 2, and 3 stand for the first, the second, and both marker bits high, respectively

