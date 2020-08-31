How to readout and analyze measured data
========================================


Measured data can be uploaded from the saved format by the following way

from qsweepy.ponyfiles import *

sample_name = '11qubits_021119_2'
| db = database.MyDatabase(host='10.20.61.31')
| exdir_db_inst = exdir_db.Exdir_db(db=db, sample_name=sample_name)

m=exdir_db_inst.select_measurement(measurement_type='two_tone',metadata={'num_resonant_qubits': num_qubits,'resonator_id':'1','ex_qubit': '1','pump_power': '-20.0'})

for coil, voltage in m.metadata.items():

| m_ref = exdir_db_inst.select_measurement(measurement_type = 'resonator', references_this={'resonator':m.id})
| resonator_scan_f = m_ref.datasets['S-parameter'].parameters[1].values
| resonator_scan_S21 = np.mean(m_ref.datasets['S-parameter'].data, axis=0)
| f_extracted = m.datasets['S-parameter'].parameters[2].values[0]
