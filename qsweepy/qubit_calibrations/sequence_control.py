import numpy as np
from qsweepy import zi_scripts
import json



def define_qubit_control_seq(device, ex_sequence, ex_channel, exitation_amplitude, readout_delay):
    if hasattr(ex_channel.parent, 'sequencer_id'):
        ex_sequence.stop()
        # Mixer calibrations result
        calib_dc_ex = ex_channel.parent.calib_dc()
        calib_rf_ex = ex_channel.parent.calib_rf(ex_channel)
        ex_sequence.set_amplitude_i(np.abs(calib_rf_ex['I']))
        ex_sequence.set_amplitude_q(np.abs(calib_rf_ex['Q']))
        ex_sequence.set_awg_amp(exitation_amplitude)
        ex_sequence.set_phase_i(np.angle(calib_rf_ex['I']) * 360 / np.pi)
        ex_sequence.set_phase_q(np.angle(calib_rf_ex['Q']) * 360 / np.pi)
        ex_sequence.set_offset_i(np.real(calib_dc_ex['dc']))
        ex_sequence.set_offset_q(np.imag(calib_dc_ex['dc']))
        ex_sequence.set_frequency(np.abs(ex_channel.get_if()))
    else:
        ex_sequence.stop()
        ex_sequence.set_awg_amp(exitation_amplitude)
        ex_sequence.set_frequency_qubit(np.abs(ex_channel.get_frequency()))
    ex_sequencers = []
    control_seq_id = ex_sequence.params['sequencer_id']
    for seq_id in device.pre_pulses.seq_in_use:
        if seq_id != control_seq_id:
            ex_seq = zi_scripts.SIMPLESequence(device=device, sequencer_id=seq_id, awg=device.modem.awg, readout_delay=readout_delay,
                                               awg_amp=1, use_modulation=True, pre_pulses = [])
            ex_seq.stop()
            device.pre_pulses.set_seq_offsets(ex_seq)
            device.pre_pulses.set_seq_prepulses(ex_seq)
            ex_sequencers.append(ex_seq)
            device.modem.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
            #ex_seq.start()
        else:
            device.pre_pulses.set_seq_offsets(ex_sequence)
            device.pre_pulses.set_seq_prepulses(ex_sequence)
            ex_sequencers.append(ex_sequence)
            device.modem.awg.set_sequence(ex_sequence.params['sequencer_id'], ex_sequence)
            #ex_sequence.start()

    return ex_sequencers
def set_preparation_sequence(device, ex_sequencers, prepare_seq, control_sequence = None, instructions=None):
    for ex_channel in device.awg_channels.keys():
        awg_channel = device.awg_channels[ex_channel]
        if hasattr(awg_channel.parent, 'sequencer_id'):
            calib_dc = awg_channel.parent.calib_dc()
            calib_rf = awg_channel.parent.calib_rf(awg_channel)
            awg_channel_id = awg_channel.parent.sequencer_id
            awg_channel.parent.awg.set_offset(channel=2 * awg_channel_id, offset=np.real(calib_dc['dc']))
            awg_channel.parent.awg.set_offset(channel=2 * awg_channel_id + 1, offset=np.imag(calib_dc['dc']))

    if control_sequence is None:
        #for ex_seq in ex_sequencers:
        for _id, ex_seq in enumerate(ex_sequencers):
            ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
            ex_seq.clear_pulse_sequence()
            for prep_seq in prepare_seq:
                for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
                    if seq_id == ex_seq.params['sequencer_id']:
                        ex_seq.add_definition_fragment(single_sequence[0])
                        ex_seq.add_play_fragment(single_sequence[1])
            ex_seq.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
            if instructions is not None:
                json_str = json.dumps(instructions[_id])
                print(_id, 'string length: ', len(json_str) , json_str)
                #raise ValueError('fallos')
                ex_seq.awg.load_instructions(ex_seq.params['sequencer_id'], json_str)
                print ('commandtable status: ', ex_seq.awg.daq.get('/' + ex_seq.awg.device +'/AWGS/{}/COMMANDTABLE/STATUS'.format(ex_seq.params['sequencer_id'])))
            ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])
    else:
        #for ex_seq in ex_sequencers:
        for _id, ex_seq in enumerate(ex_sequencers):
            if ex_seq.params['sequencer_id'] == control_sequence.params['sequencer_id']:
                ex_seq.awg.stop_seq(ex_seq.params['sequencer_id'])
                ex_seq.clear_pulse_sequence()
                for prep_seq in prepare_seq:
                    for seq_id, single_sequence in prep_seq[0][ex_seq.awg.device_id].items():
                        if seq_id == ex_seq.params['sequencer_id']:
                            ex_seq.add_definition_fragment(single_sequence[0])
                            ex_seq.add_play_fragment(single_sequence[1])

                ex_seq.awg.set_sequence(ex_seq.params['sequencer_id'], ex_seq)
                if instructions is not None:
                    json_str = json.dumps(instructions[_id])
                    print(_id, 'string length: ', len(json_str) , json_str)
                    #raise ValueError('fallos')
                    ex_seq.awg.load_instructions(ex_seq.params['sequencer_id'], json_str)
                    print ('commandtable status: ', ex_seq.awg.daq.get('/' + ex_seq.awg.device +'/AWGS/{}/COMMANDTABLE/STATUS'.format(ex_seq.params['sequencer_id'])))
                ex_seq.awg.start_seq(ex_seq.params['sequencer_id'])

def define_readout_control_seq(device, readout_pulse):
    try:
        re_channel = device.awg_channels[readout_pulse.metadata['channel']]
    except:
        qubit_id = readout_pulse.metadata['qubit_ids'][0]
        readout_channel = [i for i in device.get_qubit_readout_channel_list(qubit_id).keys()][0]
        re_channel = device.awg_channels[readout_channel]
    sequence = zi_scripts.READSequence(re_channel.parent.sequencer_id, device.modem.awg)

    #def_frag, play_frag = device.pg.readout_rect(channel=readout_pulse.metadata['channel'],
    #                                             length=float(readout_pulse.metadata['length']),
    #                                             amplitude=float(readout_pulse.metadata['amplitude']))
    #sequence.add_readout_pulse(def_frag, play_frag)
    sequence.add_readout_pulse(readout_pulse.definition_fragment, readout_pulse.play_fragment)
    sequence.stop()
    device.modem.awg.set_sequence(re_channel.parent.sequencer_id, sequence)
    sequence.set_delay(device.modem.trigger_channel.delay)
    sequence.start()
    sequence.awg.stop_seq(sequence.params['sequencer_id'])

    return sequence