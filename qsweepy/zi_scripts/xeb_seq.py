
import numpy as np
import textwrap
import time


class XEBSequence:
    def __init__(self, sequencer_id, awg, readout_delay=0, pre_pulses = [],
                 control=False, is_iq = False, num_gates_reg = 0, random_seed_reg = 1, interleaver_repeats_reg = 2, random_repeats_reg = 3):
        """
        Sequence for XEB pulse generation on a single channel.
        Send trigger for readout sequencer start.
        Produces random waveform.
        Parameters

        ----------
        awg
        tail_length[sec] - rising and falling edges length
        readout_delay[sec] - additional delay between Rabi falling edge and readout waveform. There is initial delay around 140ns exists
        due to DIO protocol.
        awg_amp - output amplitude of awg generator. Before generation it will be multiplied by Range value!!!!!!
        pre_pulses - waveform you want to set before worl sequence. Really you want to place here reset pulses.

        """
        self.registers = {} #Dictionary for reserved User registers
        self.nsamp = 16
        self.awg = awg
        self.clock = float(self.awg._clock)
        self.params = dict(sequencer_id=sequencer_id, qubit_channel=0,
                           offset_channel=1, n_samp=self.nsamp,
                           readout_delay=int(readout_delay*self.clock),
                           nco_id=sequencer_id*4, ic=2 * sequencer_id, qc=sequencer_id*2 + 1,
                           num_gates_reg=num_gates_reg, random_seed_reg=random_seed_reg,
                           interleaver_repeats_reg=interleaver_repeats_reg, random_repeats_reg=random_repeats_reg)

        self.pre_pulses = pre_pulses
        self.definition_pre_pulses = '''
// Pre pulses definition'''
        self.play_pre_pulses = '''
// Pre pulses play'''
        self.definition_interleaver = '\n//Interleaver pulse definition'
        self.play_interleaver = '\n//Interleaver play'
#        self.definition_random_group = '\n//Random pulses definition'
#        self.play_random_group = '\n//Random '
        self.random_pulses = []
        # Initial settings
        self.set_awg_amp(self.params['awg_amp'])
        # You need to set phase in offset channel equal to 90 degrees
        # In other case it doesn't work
        self.set_phase_control(90)
        self.set_frequency_control(0)

        # For exitation channels
        # We need to set DIO slope as  Rise (0- off, 1 - rising edge, 2 - falling edge, 3 - both edges)
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/strobe/slope' % self.params['sequencer_id'], 1)
        # We need to set DIO valid polarity as  Rise (0- none, 1 - low, 2 - high, 3 - both )
        #self.awg.daq.setInt('/' + self.awg.device + '/awgs/%d/dio/valid/polarity' % self.params['sequencer_id'], 2)

    def zicode(self):
        # In this fragment we collect pre_pulses from in device.pre_pulses
        definition_pre_pulses = self.definition_pre_pulses
        # In this fragment we define constants
        self.params['random_gate_num'] = len(self.random_pulses)
        definition_fragments = textwrap.dedent('''
        // Definition fragment start
        // Device settings
        setInt('sines/{ic}/oscselect', {nco_id});
        setInt('sines/{qc}/oscselect', {nco_id}+1);
        
        // Constant's and variables definition
        const readout_delay = {readout_delay};
        const num_gates_reg = {num_gates_reg};
        const random_seed_reg = {random_seed_reg};
        const interleaver_repeats_reg = {interleaver_repeats_reg};
        const random_repeats_reg = {random_repeats_reg};
        const random_gate_num = {random_gate_num};
        
        // Pre-pulse definition fragment
        '''.format(**self.params))
        # In play_fragment1 we wait for trigger from readout sequencer
        # Sequencer for qubit's control are triggered by DIO protocol.
        # This DIO signal is controlled by readout sequencer.
        wait_trigger_fragment = textwrap.dedent('''
        // Wait fragment
        while (true) {{
            var num_gates = getUserReg(num_gates_reg);
            var random_seed = getUserReg(random_seed_reg);
            var interleaver_repeats = getUserReg(interleaver_repeats_reg);
            var random_repeats = getUserReg(random_repeats_reg);
        
            setPRNGSeed(random_seed);
            setPRNGRange(0, random_gate_num - 1);
            // Wait trigger and reset
            waitDIOTrigger();
            //resetOscPhase();
            ''')


        # In this fragment we define work sequence for play
        play_fragment = '''// Play pre-pulses fragment
        //'''

        for _i in range(len(self.pre_pulses)):
            definition_fragments += self.pre_pulses[_i].get_definition_fragment(self.clock, _i)
            play_fragment += self.pre_pulses[_i].get_play_fragment(self.params['qubit_channel'], _i)

        for random_pulse_id, random_pulse in enumerate(self.random_pulses):
            if 'waveform' in random_pulse:
                definition_fragments += "wave random_pulse_{name}_ch0 = placeholder({wave_len:d});\n".format(
                    name=random_pulse['name'], wave_len=len(random_pulse['waveform_ch0']))
                definition_fragments += "wave random_pulse_{name}_ch1 = placeholder({wave_len:d});\n".format(
                    name=random_pulse['name'], wave_len=len(random_pulse['waveform_ch1']))
                definition_fragments += "assignWaveIndex(random_pulse_{name}_ch0, random_pulse_{name}_ch1, {id});\n".format(
                    name=random_pulse['name'], id=random_pulse_id)

        play_fragment += textwrap.dedent('''
    // Main play sequence
        resetOscPhase();
        wait(10);
        repeat (num_gates) {{
            repeat (random_repeats) {{
                // (Pseudo) Random sequence of command table entries
                executeTableEntry(getPRNGValue());
            }}
            repeat (interleaver_repeats) {{
                {play_interleaver}
            }}
        }})
        '''.format(play_interleaver=self.play_interleaver))


        # Then work sequence has done you need to send trigger for readout sequencer to start playWave.
        # There is initial delay between readout trigger and and readout waveform generation around 140 ns.
        if self.control:
            end_sequence_fragment = textwrap.dedent('''
            // This is the control sequencer. It sends a trigger pulse via DIO to the readout channels.
            // Send trigger for readout_channel to start waveform generation
            waitWave();
            playZero(readout_delay);
            setDIO(8);
            wait(10);
            setDIO(0);
        }}
        ''')
        else:
            end_sequence_fragment = textwrap.dedent('''
            // This is not the control sequencer. It doesn't send a trigger pulse via DIO to the readout controls.
        }}
        ''')
        code = ''.join(definition_pre_pulses + definition_fragments + wait_trigger_fragment + play_fragment + end_sequence_fragment)
        return code

    def set_commandtable(self):
        import json
        # command_table = {"$schema": "http://docs.zhinst.com/hdawg/commandtable/v2/schema",
        #                  "header": { "version": "0.2" },
        #                  "table": [] }

        command_table = {'$schema': 'http://docs.zhinst.com/hdawg/commandtable/v2/schema',
                         'header': {'version': '1.2'},
                         'table': []}

        for random_pulse_id, random_pulse in enumerate(self.random_pulses):
            table_entry = {'index': random_pulse_id}
            if 'waveform' in random_pulse:
                table_entry['waveform'] = {'index': random_pulse_id}
            if 'phase0' in random_pulse:
                table_entry['phase0'] = {'phase0': {'value': random_pulse['phase0'], 'increment': True}}
            if 'phase1' in random_pulse:
                table_entry['phase1'] = {'phase0': {'value': random_pulse['phase1'], 'increment': True}}
            command_table['table'].append(table_entry)

        json_str = json.dumps(command_table)

        self.awg.load_instructions(self.sequencer_id, json_str)

    def load_waves(self):
        """uploads a set of waveforms to the waveform memory

        Parameters
        ----------
        waves: list
            list of two-channel waveforms, given as [[I], [Q]]
        """
        # load AWG waveforms
        for random_pulse_id, random_pulse in enumerate(self.random_pulses):
            if 'waveform' in random_pulse:
                self.awg.set_waveform_indexed(np.real(random_pulse['waveform']), np.imag(random_pulse['waveform']))

    def add_register(self, reg_name, value):
        # reg_name is a str
        if reg_name in self.registers.keys():
            raise Exception('Error! Can not do this! Reg name ='+reg_name+' is already exist')
            #print('Error! Can not do this! Reg name ='+reg_name+' is already exist')
        elif value in self.registers.values():
            raise Exception('Error! Can not do this! User Register =' + str(value) + ' is already reserved')
            #print('Error! Can not do this! User Register =' + str(value) + ' is already reserved')
        else:
            self.registers[reg_name] = value

    def add_pre_pulse(self, definition, play_fragment):
        #self.pre_pulses.append(pre_pulse)
        self.definition_pre_pulses += definition
        self.play_pre_pulses += play_fragment

    def set_frequency_qubit(self, frequency):
        self.awg.set_frequency(self.params['nco_id'], frequency)
        #self.awg.set_frequency(self.params['nco_id']+1, 0)

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

    def set_frequency_control(self, frequency):
        self.awg.set_frequency(self.params['nco_id'] + 1, frequency)

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
        self.awg.set_register(self.params['sequencer_id'], self.params['length__reg'], pause_cycles//8)
        self.awg.set_register(self.params['sequencer_id'], self.params['resudual__reg'], 1+pause_cycles % 8)
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
        self.awg.set_output(self.params['ic'], 1)
        self.awg.set_output(self.params['qc'], 1)
        self.awg.start_seq(self.params['sequencer_id'])

    def stop(self):
        self.awg.stop_seq(self.params['sequencer_id'])
        self.awg.set_output(self.params['ic'], 0)
        self.awg.set_output(self.params['qc'], 0)
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)