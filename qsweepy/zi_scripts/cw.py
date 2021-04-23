import numpy as np


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

        self.params = dict(sequencer_id=sequencer_id, nco_id=sequencer_id*4, ia=0, qa=1, ip=2, qp=3, io=4, qo=5)
        self.awg = awg

    def zicode(self):
        code = '''
const phase_mul = 360.0/0xffffffff
const amplitude_mul = 1.0/0xffffffff

// turn off outputs to protect mixers&qubits 
setInt('sigouts/{ic}/on', 0)
setInt('sigouts/{qc}/on', 0)
// use first nco of sequencer
setInt('sines/{ic}/oscselect', {nco_id});
setInt('sines/{qc}/oscselect', {nco_id});
// set sine wave amplitudes
setDouble('sines/{ic}/amplitudes/{ic}', getUserReg({ia})*amplitude_mul)
setDouble('sines/{qc}/amplitudes/{qc}', getUserReg({qa})*amplitude_mul)
// set sine wave phases
setDouble('sines/{ic}/phase', getUserReg({ip})*phase_mul);
setDouble('sines/{qc}/phase', getUserReg({qp})*phase_mul);
// set output offset dc
setDouble('sigouts/{ic}/offsets', getUserReg({io})*amplitude_mul)
setDouble('sigouts/{qc}/offsets', getUserReg({qo})*amplitude_mul)

// sine waves -> output
setInt('sines/{ic}/enables/{ic}', 1)
setInt('sines/{qc}/enables/{qc}', 1)
// turn outputs back on
// setInt('sigouts/{ic}/on', 1)
// setInt('sigouts/{qc}/on', 1)
'''.format(**self.params)

    def set_amplitude_i(self, amplitude_i):
        assert (np.abs(amplitude_i) <= 0.5)
        self.awg.set_register(self.params['sequencer_id'], self.params['ia'], amplitude_i*0xffffffff)

    def set_amplitude_q(self, amplitude_q):
        assert (np.abs(amplitude_q) <= 0.5)
        self.awg.set_register(self.params['sequencer_id'], self.params['qa'], amplitude_q*0xffffffff)

    def set_phase_i(self, phase_i):
        self.awg.set_register(self.params['sequencer_id'], self.params['ip'], phase_i * 0xffffffff / 360.0)

    def set_phase_q(self, phase_q):
        self.awg.set_register(self.params['sequencer_id'], self.params['qp'], phase_q * 0xffffffff / 360.0)

    def set_offset_i(self, offset_i):
        assert (np.abs(offset_i) <= 0.5)
        self.awg.set_register(self.params['sequencer_id'], self.params['io'], offset_i*0xffffffff)

    def set_offset_q(self, offset_q):
        assert (np.abs(offset_q) <= 0.5)
        self.awg.set_register(self.params['sequencer_id'], self.params['qo'], offset_q*0xffffffff)
