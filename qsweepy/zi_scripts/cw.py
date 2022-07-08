import numpy as np
import time

class CWSequence:
    def __init__(self, sequencer_id, awg):
        """
        Sequence for mixer calibration.
        Produces cw waves from both channels with given complex amplitudes.
        Complex amplitudes can be set into registers corresponding
        Parameters
        ----------
        awg
        """

        self.params = dict(sequencer_id=sequencer_id, nco_id=sequencer_id*4, ia=0, qa=1, ip=2, qp=3,
                           ic=2*sequencer_id, qc=sequencer_id*2+1)
        self.awg = awg

    def zicode(self):
        code = '''
const phase_mul = 360.0/0xffffffff;
const amplitude_mul = 1.0/0xffffffff;

// turn off outputs to protect mixers&qubits 
// setInt('sigouts/{ic}/on', 0);
// setInt('sigouts/{qc}/on', 0);
// use first nco of sequencer
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id});
// set sine wave amplitudes
// setDouble('sines/{ic}/amplitudes/0', getUserReg({ia}), amplitude_mul);
// setDouble('sines/{qc}/amplitudes/1', getUserReg({qa}), amplitude_mul);
// set sine wave phases
// setDouble('sines/{ic}/phaseshift', getUserReg({ip}), phase_mul);
// setDouble('sines/{qc}/phaseshift', getUserReg({qp}), phase_mul);
// setSinePhase(0, 0);
// setSinePhase(1, 0);
// cvar b;
// for (b =0; b<32; b++) {{
//     if ( getUserReg({ip}) & (1 << b) ) {{
//         incrementSinePhase(0, phase_mul*(1 << b));
//     }}
//     if ( getUserReg({qp}) & (1 << b) ) {{
//         incrementSinePhase(1, phase_mul*(1 << b));
//     }}
// }}
// set output offset dc

// sine waves -> output
// setInt('sines/{ic}/enables/0', 1);
// setInt('sines/{qc}/enables/1', 1);
// turn outputs back on
// setInt('sigouts/{ic}/on', 1);
// setInt('sigouts/{qc}/on', 1);
'''.format(**self.params)
        return code

    def set_frequency(self, frequency):
        self.awg.set_frequency(self.params['sequencer_id']*4, frequency)

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
        assert (np.abs(offset_i) <= 0.6)
        self.awg.set_offset(channel=self.params['ic'], offset=offset_i)
        # self.awg.set_register(self.params['sequencer_id'], self.params['io'], np.asarray(int(offset_i*0xffffffff), dtype=np.uint32))

    def set_offset_q(self, offset_q):
        assert (np.abs(offset_q) <= 0.6)
        self.awg.set_offset(channel=self.params['qc'], offset=offset_q)
        # self.awg.set_register(self.params['sequencer_id'], self.params['qo'], np.asarray(int(offset_q*0xffffffff), dtype=np.uint32))

    def start(self):
        self.awg.start_seq(self.params['sequencer_id'])
        self.awg.set_output(self.params['ic'], 1)
        self.awg.set_output(self.params['qc'], 1)
        self.awg.set_sin_enable(self.params['ic'], 0, 1)
        self.awg.set_sin_enable(self.params['qc'], 1, 1)

    def stop(self):
        self.awg.stop_seq(self.params['sequencer_id'])
        self.awg.set_output(self.params['ic'], 0)
        self.awg.set_output(self.params['qc'], 0)
        self.awg.set_sin_enable(self.params['ic'], 0, 0)
        self.awg.set_sin_enable(self.params['qc'], 1, 0)