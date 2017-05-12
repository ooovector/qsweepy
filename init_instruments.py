#Create Instruments

#######################
# PHYSICAL INSTRUMENTS#
#######################
from instruments import *

pna = Agilent_N5242A('pna', address = 'PNA')
pxa = Agilent_N9030A('pxa', address = 'PXA')
lo1 = Agilent_E8257D('lo1', address = 'PSG1')
awg_tek = Tektronix_AWG5014('awg_tek', address = 'TCPIP0::10.20.61.186::inst0::INSTR')

#awg = qt.instruments.create('awg', 'AWG500', address=0)
#daq0 = qt.instruments.create('daq0', 'Spectrum_M3i2132')

#ni_daq = qt.instruments.create('ni_daq','NI_DAQ', address = 'GPIB::0::INSTR')

#lo2 = qt.instruments.create('local_oscillator_2','Agilent_E8257D', address = 'TCPIP0::10.20.61.247::inst0::INSTR')
#current = qt.instruments.create('current','Keithley_6221', address = 'GPIB0::10::INSTR', range = (-100e-3, 100e-3))

#dso = qt.instruments.create('dso', 'Agilent_DSO', address = 'TCPIP0::10.20.61.43::inst0::INSTR')



