import ftd2xx
import ctypes
import struct
import array
import time
from qsweepy.instrument import Instrument
import types

BASE_ID      = 64                               # base ID selector for the AWG board

ID_PROG      = BASE_ID+0                        # reserved for FPGA programing
ID_CTRL      = BASE_ID+1                        # 32 soft switches
ID_T_PROG    = BASE_ID+2                        # Tiny DAC CPU
ID_PWL       = BASE_ID+3
ID_TIMER     = BASE_ID+3
ID_PULSE     = BASE_ID+4
ID_RAM       = BASE_ID+5
ID_TRIGGER   = BASE_ID+5
ID_READ_A    = BASE_ID+6
ID_TINY      = BASE_ID+7

ID_DISABLE   = BASE_ID+12                       # 

ID_CPU       = BASE_ID+15                       # Nano-Blaze

ID_REQUEST     = 128                          # bit-7 set
ID_REQ_BIT     = 129                          # +1
ID_REQ_POS     = 130                          # +2
ID_REQ_ARM     = 131                          # +3
ID_POL_DAT_BIT = 132                          # +4
ID_POL_MUX     = 133                          # +5
ID_EDGE        = 134                          # +6
ID_DATA_MUX    = 135                          # +7

ADDR_RES  = 0                                   # two bit (0 and 1) mow are used to
ADDR_DATA = 1                                   # select Program address selector
ADDR_LED0 = 2                                   # top 6-bits may be used
ADDR_LED1 = 6                                   # for different selections 
ADDR_LED2 = 10                                  # outside the prog_blcok.vhd
ADDR_LED3 = 14
ADDR_NU   = 3                                   # not used so far

FT_FLOW_NONE		=0x0000
FT_FLOW_RTS_CTS		=0x0100
FT_FLOW_DTR_DSR		=0x0200
FT_FLOW_XON_XOFF	=0x0400

class AWG500(Instrument):
	def get_clock(self):
		return 500e6
		
	def SendPacket(self, id, addr, data):
		packet = bytes([id, addr])+data
		#print (packet)
		bytesWritten = self.h.write(packet)
		#print ('Sending packet id {0}, addr {1}, bytes written: {2}'.format(id, addr, bytesWritten))
		time.sleep(0.015)
#        if bytesWritten < 1000:
#            print ('Sent packet: {0}'.format(':'.join(hex(ord(x))[2:] for x in packet)))
		return bytesWritten
	def ReadPacket(self, cnt):
		#print ('Reading {0} bytes.'.format(cnt))
		buf = self.h.read(cnt)
		#print ('Read {0} bytes'. format(len(buf)))
		return buf
	
	def SelectPolMux(self, sel):
		self.SendPacket(ID_POL_MUX, 0, self.intToBytes([sel]))
	def SetPolBit(self, bit):
		self.SendPacket(ID_POL_DAT_BIT, 0, self.intToBytes([bit]))
	def RequestData(self, cnt):
		self.SendPacket(ID_REQUEST, 0, self.intToBytes([cnt]))
	def WaitForData(self, cnt):
		# not implemented. This driver doesn't use async io anyway =(
		return self.ReadPacket(cnt)
	
	def IsVirtexProgrammed(self):
		self.SelectPolMux(1)
		self.SetPolBit(1)
		self.h.purge(ftd2xx.defines.PURGE_RX)
		num = 8 # read N+1 bytes; last byte is still wrong (bug must be fixed)
		self.RequestData(num)
		cc1=self.WaitForData(num) # this should yield FF FF FF FF
		self.SetPolBit(0)
		self.h.purge(ftd2xx.defines.PURGE_RX)
		self.RequestData(num)
		cc0=self.WaitForData(num) # this should yield 00 00 00 00
		self.SelectPolMux(0)
		if cc1 == bytes([0xFF, 0xFF, 0xFF, 0xFF]) and cc0 == bytes([0x00, 0x00, 0x00, 0x00]):
			check = 'passed'
		else:
			check = 'failed'
		#print ('Virtex check {0}: ', check, cc1, cc0)
		return True if check=='passed' else False

	def ProgVirtex6(self, filename=None):
		from qsweepy.config import get_config
		if not filename:
			filename = get_config().get('AWG500_default_firmware')
		self.SendPacket(ID_PROG, ADDR_RES, self.intToBytes([0]))
		with open(filename, 'rb') as f:
			firmware = f.read()
		self.SendPacket(ID_PROG, 1, bytes(firmware))
		time.sleep(1.0)
		self.SendPacket(ID_PROG, ADDR_LED0, self.intToBytes([0]))
		
	def InitPulseDacs(self, value):
		value_hi = (value//256)&255
		value_lo = value&255
		dac_prog = b''.join([b'\x07\x01\x00\x3f\x07\x0f\x00\x30\x07\x0f\x00\x27\x07', chr(value_lo), chr(value_hi), b'\x00'])#b'\x1f'])
		return self.SendPacket(64, 1, dac_prog)

	def SendControlWord(self, value):
		controlWord = self.intToBytes([value])
		return self.SendPacket(65, 0, controlWord)
	
	def SetPulseAmplitude(self, channel, value):
		value_hi = (value//256)&255
		value_lo = value&255
		
		dac_prog = bytes([0x07, value_lo, value_hi, 0x18+(channel&0x3)])
		#dac_prog = bytes('').join([chr(0x07), chr(value_lo), chr(value_hi), chr(0x18+(channel&0x3))])
	#print dac_prog
		return self.SendPacket(64, 1, dac_prog)
	
	def ProgPulse(self, channel, pulse):
		return self.SendPacket(64+4, channel, pulse)
	
	def SetDAC16Gain(self, dac, value):
		dac_prog = self.intToBytes([0x33C01])
		return self.SendPacket(64, 1, dac_prog)

	def SetDAC16zero(self, dac, channel, value):
		value = value & 0xFFFF
		dac_prog = 0x8000000+0x1000000*(channel&3) + value*0x100
		dac_prog = self.intToBytes([1+dac+dac_prog])
		return self.SendPacket(64, 1, dac_prog)

	def SetDAC16(self, dac, channel, value):
		value = value & 0xFFFF
		dac_prog = bytes([dac+1, value&0xFF, (value//256)&0xFF, 0x4+0x1*(channel&0x3)])
		return self.SendPacket(64, 1, bytes(dac_prog))
	
	def intToBytes(self, ints):
		result = b''
		for i in range(0, len(ints)):
			b1 = (ints[i]//16777216)&0xFF
			b2 = (ints[i]//65536)&0xFF
			b3 = (ints[i]//256)&0xFF
			b4 = (ints[i])&0xFF
			result = result+bytes([b4, b3, b2, b1])
		return bytes(result)

	def LoadPWL(self, channel, data):
		return self.SendPacket(64+3, channel, data)

	def ProgRepeater(self, repeater):
		return self.SendPacket(64+3, 9, repeater)

	def ProgTimer(self, timer):
		return self.SendPacket(64+3, 8, timer)

	def ProgAutoTrigger(self, trigger):
		return self.SendPacket(64+3, 10, trigger)

	def InitDAC8(self, value):
		dac_prog = b'\x13\x00\x80\x90'+join([(value&0xf0)+3, (value//256)&0xff])+b'\x3f\x90\x03\x00\x60\x90'
		return self.SendPacket(64, 1, dac_prog)

	def SetDAC8(self, channel, value):
		value = value&0xfff0;                     #   // mask 12-bit data and clear 4 low bits
		channel = channel&0xF;                           #// 1 -:- 7 anf 0xF only allowed
		channel = channel*65536;                  #     // move into 3rd byte

		dac_prog = self.intToBytes([0x90000003+channel+value, 0x00000000])
		return self.SendPacket(64, 1, dac_prog)

	def RamAddrCtrl(self, channel, mode, delta, slow, limit, delay):
		if (slow > 2):                                # must be < 0x3fffffff (30 bits only) 
			slow = slow - 2                           # -2 => adjust for DSP counter lattency
			slow = slow * 4                           # shif the rest by 2 buts left, 2 msbs are gone
			slow = slow + 3                           # add USE_DSP_COUNTER mx selector
		dat = [mode, delta, slow, limit, delay];      # mode_selector : Zero, RAM, CPU, PWL
		return self.SendPacket(64+6, channel, self.intToBytes(dat)) # delta : run read address faster, slow  : run it slower : 0x100000==slow switch 00000- actual data, limit : use for shorter then 32K waveforms, ID=64+6, addr=channel => write RAM

	def ProgLimit(self, value, limitNo):
		value = (value << 16) + 10

		if (limitNo):
			value = value + (limitNo & 3);             # Jump Table entry 11-13 - set hi, start, ????
			self.SendPacket(64, 1, self.intToBytes([value]));          # ID=64, Addr=1(prog FIFO), send 1 dwords

		return value

	def Send2CPU(self, data, addr): #          // stop CPU with dummy call 0 or new function
		return self.SendPacket(64, 128+(addr&127), self.intToBytes([data])) #// ID=64, Addr=128(prog FIFO)+addr, send 1 dwords

	def LoadRAM(self, channel, data):
		return self.SendPacket(64+5, channel, self.intToBytes(data))
	
	def load_ram_offset(self, chan, data):
		#Load signal with offset
		data += self.channel_settings[chan]['offset']
		return self.SendPacket(64+5, channel, self.intToBytes(data))
		
	def send_control_word(self):
		USE_INTERNAL_TRIGGER = 0x4000000
		USE_10MHZ_FILTER_CHANNEL_0 = 0x100
		USE_CHANNEL_0 = 0x10001
		USE_CHANNEL_1 = 0x20002
		USE_CHANNEL_2 = 0x40004
		USE_CHANNEL_3 = 0x80008
		USE_CHANNEL_4 = 0x100010
		USE_CHANNEL_5 = 0x200020
		USE_CHANNEL_6 = 0x400040 

		control_word = 0x0
		if self.use_internal_trigger:
			control_word |= USE_INTERNAL_TRIGGER
		for channel_id in self.channel_settings.keys():
			control_word |= USE_CHANNEL_0*int(2**channel_id)*self.channel_settings[channel_id]['on']
		if self.channel_settings[0]['filter']:
			control_word |= USE_10MHZ_FILTER_CHANNEL_0
			
		self.SendControlWord(control_word);
	
	# sets the amount of (digital) trigger pulses per single (analog) AWG pulse sequence.
	# Usually this is 1.
	def set_trigger_repeats(self, trigger_repeats):
		self.trigger_repeats =  trigger_repeats
		repeater = self.intToBytes([trigger_repeats%(2**32), trigger_repeats//(2**32)])
		self.ProgRepeater(repeater)

	def get_trigger_repeats(self):
		return self.trigger_repeats
	# sets the time between several (digital) trigger pulses per single (analog) AWG pulse sequence.
	# Usually this is unused because there is only a single trigger repeat anyway.
	# input: period, in seconds
	def set_trigger_period(self, trigger_period):
		self.trigger_period = trigger_period
		trigger_period = int(trigger_period/2e-9)
		timer = self.intToBytes([trigger_period%(2**32), trigger_period//(2**32)])
		self.ProgTimer(timer)

	def get_trigger_period(self):
		return self.trigger_period
	# set automatic trigger period between AWG pulse sequences.
	# This function should be used on a regular basis.
	def do_set_repetition_period(self, repetition_period):
		self.repetition_period = repetition_period
		repetition_period = int(repetition_period/2e-9)
		auto_trigger = self.intToBytes([int(repetition_period%(2**32)), int(repetition_period/(2**32))])
		self.ProgAutoTrigger(auto_trigger)

	def get_repetition_period(self):
		return self.repetition_period*2e9
		
	def get_nop(self):
		return int(self.repetition_period/2e-9)
		
	def set_nop(self, repetition_period):
		self.repetition_period = repetition_period*2e-9
		auto_trigger = self.intToBytes([int(repetition_period%(2**32)), int(repetition_period/(2**32))])
		self.ProgAutoTrigger(auto_trigger)
		
	# set trigger (digital output) amplitude.
	def set_trigger_amplitude(self, channel_name, amplitude):
		channel_names = {'PLS5':0, 'PLS2':1, 'PLS1':2}
		self.trigger_amplitudes[channel_name] = amplitude
		self.SetPulseAmplitude(channel_names[channel_name], 0xFFF0)

	def get_trigger_amplitude(self, channel_name):
		return self.trigger_amplitudes[channel_name]
	# set trigger (digital output) pulse form. 
	# setting trigger pulse form. This is sick shit and it doesn't work as expected.
	# see documentation for explanation how it should work in principle.
	def set_trigger_pulse(self, channel_name, pulse):
		channel_names = {'SYNC0':0, 'PLS1':1, 'PLS2':2, 'SYNC3':3, 'SYNC4':4, 'PLS5':5}
		self.trigger_pulses[channel_name] = pulse
		self.ProgPulse(channel_names[channel_name], self.intToBytes(pulse))

	def get_trigger_pulse(self, channel_name):
		return self.trigger_pulses[channel_name]
	# function to create a delayed trigger pulse sequence.
	# I have no idea why it works.
	def create_delayed_trigger_pulse(self, delay):
		if self.ver == 2:
			# AWG-500 ver 2 probably works in a concievable manner...
			return [0, int(delay/1e-9), 3, 2, 0, 4]
		else:
			return [0, int(delay/1e-9), 3, 100, 3, 1, 0, 4]
	# But it does	
	def set_delayed_trigger_pulse(self, chan, delay):
		pulse = self.create_delayed_trigger_pulse(delay)
		channel_names = {'SYNC0':0, 'PLS1':1, 'PLS2':2, 'SYNC3':3, 'SYNC4':4, 'PLS5':5}
		self.trigger_pulses[chan] = pulse
		self.ProgPulse(channel_names[chan], self.intToBytes(pulse))
	
	# setting AWG channel settings
	def set_channel_settings(self, channel_id, settings):
		channel_modes = {'zero':0, 'tiny':64, 'ram':128, 'pwl':196}
		self.channel_settings[channel_id] = settings
		self.send_control_word() # update control word to turn on/off channel and filters
		slow = settings['slow']
		if (slow > 2):                                # must be < 0x3fffffff (30 bits only) 
			slow = slow - 2                           # -2 => adjust for DSP counter lattency
			slow = slow * 4                           # shif the rest by 2 buts left, 2 msbs are gone
			slow = slow + 3                           # add USE_DSP_COUNTER mx selector
		self.RamAddrCtrl(channel_id, channel_modes[settings['mode']], settings['delta_ram'], settings['slow'], settings['ram_pulse_length'], settings['delay'])
		if channel_id != 0:
			if settings['filter'] != False:
				print ('Turn on 10 MHz filter for channels 1-6 not implemented in hardware')
				settings['filter'] = False
			if settings['offset'] != 0:
				print ('Offset for channels 1-6 not implemented in hardware')
				settings['offset'] = 0

	def get_channel_settings(self, channel_id):
		return self.channel_settings[channel_id]
	# setting AWG pulse
	def set_waveform(self, waveform, channel):
		import numpy as np
		self.pulses[channel] = np.asarray(waveform*8127.5, dtype=np.int)
		self.LoadRAM(channel, self.pulses[channel])

	def get_waveform(self, channel_id):
		return self.pulses[channel_id]

	def get_trigger_pulse(self, channel_id):
		return self.trigger_pulses[channel_id]

	def get_min_value(self, channel_id):
		if self.channel_settings[channel_id]['signed']: return -8192
		else: return 0

	def get_max_value(self, channel_id):
		if self.channel_settings[channel_id]['signed']: return 8191
		else: return 16383

	def get_sample_period(self, channel_id):
		return (1+self.channel_settings[channel_id]['slow'])*2e-9

	def get_sample_dtype(self):
		return int

	def get_max_sample_size(self):
		return 32768
		
	#Application specific functions	
	def period_ram_pls_clk_out(self, chans, period ,offsets):
		ch_off = [ch_id for ch_id in range(7) if ch_id not in chans]
		
		for ch_id in ch_off:
			ch_settings = self.get_channel_settings(ch_id)
			ch_settings['on'] = False
			self.set_channel_settings(ch_id, ch_settings)
		
		for ch_id, offset in zip(chans, offsets):
			ch_settings = self.get_channel_settings(ch_id)
			ch_settings['on'] = True
			ch_settings['slow'] = 0
			ch_settings['mode'] = 'ram'
			ch_settings['delay'] = 0
			ch_settings['sw_offset'] = offset
			self.set_channel_settings(ch_id, ch_settings)
		#Something....
		self.ProgTimer(self.intToBytes([0xfc, 0]))
		
		self.set_trigger_repeats(1)
		self.set_repetition_period(period)
		

	#Constructor
	def __init__(self, name, address=0, ver=0, use_internal_trigger=True, mode='master'):
		from qsweepy.config import get_config
		import numbers
		#qtlab stuff
		Instrument.__init__(self, name, tags=['measure'])
		# Opening ftd2xx connection (USB-COM dongle)
		
		if not isinstance(address, numbers.Integral):
			address = ftd2xx.listDevices().index(address)
		print ('opening device')
		self.h = ftd2xx.open(address)
		print ('device open')

		# QTLab stuff
		self.add_parameter('repetition_period', type=float,
				flags=Instrument.FLAG_SET | Instrument.FLAG_SOFTGET,
				minval=2e-9)
				
		self.add_function('LoadRAM')
		self.add_function('Send2CPU')
		self.add_function('ProgLimit')
		self.add_function('RamAddrCtrl')
		self.add_function('SetDAC8')
		self.add_function('InitDAC8')
		self.add_function('ProgAutoTrigger')
		self.add_function('ProgTimer')
		self.add_function('ProgRepeater')
		self.add_function('LoadPWL')
		self.add_function('intToBytes')
		#self.add_function('SetDAC16Gain')
		self.add_function('SetDAC16zero')
		self.add_function('SetDAC16')
		self.add_function('ProgPulse')
		self.add_function('SetPulseAmplitude')
		self.add_function('SendControlWord')
		self.add_function('InitPulseDacs')
		self.add_function('SendPacket')

		self.add_function('set_trigger_repeats')
		self.add_function('get_trigger_repeats')
		self.add_function('set_trigger_period')
		self.add_function('get_trigger_period')
		self.add_function('set_repetition_period')
		self.add_function('get_repetition_period')
		self.add_function('set_trigger_amplitude')
		self.add_function('get_trigger_amplitude')
		self.add_function('set_trigger_pulse')
		self.add_function('get_trigger_pulse')
		self.add_function('set_waveform')
		self.add_function('get_waveform')
		self.add_function('create_delayed_trigger_pulse')
		self.add_function('get_max_value')
		self.add_function('get_min_value')
		self.add_function('get_sample_period')
		self.add_function('get_sample_dtype')
		self.add_function('get_max_sample_size')

		# default driver settings 
		channel_settings = {channel_id:{'on':True,                # turn on channel power
											'mode': 'ram',            # AWG mode: 'ram' - predefined point, 'PWL' - piecewise linear
											'filter':False,            # turno/off 10 MHz filter (only available for channel 0)
											'offset':0,               # slow hardware offset (only available for channel 0)
											'sw_offset':0,				#software offset
											'signed':True,            # 0 for -8192 to 8191, 1 for 0 to 16383
											'continuous_run':False,   # enables repetition inside a single AWG pulse sequence for this channel
											'reset_enable':False,     # enables repetition rate inside a single AWG pulse sequence 
																	  # for this channel different from 32768 pulse points
											'clock_phase':0,          # clock phase inverter (1 ns shift)
											'delta_ram':0,            # points per clock period
											'slow':0,                 # 2e-9 s clock period per data point
											'ram_pulse_length':32768, # repetition rate inside a single AWG pulse sequence if 'continuous run' and 'reset enable' is on
											'delay':0             # some strange delay 
											} for channel_id in range(7)}
		self.pulses = {channel_id: [] for channel_id in range(7)}
		self.channel_settings = channel_settings
		self.use_internal_trigger = use_internal_trigger
		# turning on AWG channels (control word)
		self.ver = ver
		if ver == 2:
			self.h.setBitMode(0xff, 0x40)
			self.h.setUSBParameters(0x10000, 0x10000)
			self.h.setLatencyTimer(2)
			self.h.setFlowControl(FT_FLOW_RTS_CTS, 0, 0)
			is_programmed = self.IsVirtexProgrammed()
			print ('Virtex is programmed? ', is_programmed)
			if not is_programmed:
				if mode=='master':
					self.ProgVirtex6(filename=get_config()['AWG500_default_firmware'])
				else:
					self.ProgVirtex6(filename=get_config()['AWG500_slave_firmware'])
		
		self.send_control_word()
		# setting a single trigger per AWG repeat
		self.set_trigger_repeats(1)
		self.set_trigger_period(0)
		self.set_repetition_period(self.get_max_sample_size()*2e-9)
		# setting trigger amplitudes for PLS channels
		#self.trigger_amplitudes = {'PLS5':0xfff0, 'PLS2':0xfff0, 'PLS1':0xfff0}
		self.trigger_amplitudes = {'PLS5':0x0000, 'PLS2':0x0000, 'PLS1':0x0000}
		for channel_name, amplitude in zip(self.trigger_amplitudes.keys(), self.trigger_amplitudes.values()):
			self.set_trigger_amplitude(channel_name, amplitude)
		# setting trigger pulse form. This is sick shit and it doesn't work as expected.
		# see documentation for explanation how it should work in principle.
		#default_trigger_pulse = [0, 4]
		default_trigger_pulse = self.create_delayed_trigger_pulse(0)
		self.trigger_pulses = {channel_name:default_trigger_pulse for channel_name in ['SYNC0']}#, 'PLS1', 'PLS2', 'SYNC3', 'SYNC4', 'PLS5']}
		for channel_name, pulse in zip(self.trigger_pulses.keys(), self.trigger_pulses.values()):
			self.set_trigger_pulse(channel_name, pulse)
		# setting default AWG channel settings
		for channel_id, settings in zip(self.channel_settings.keys(), self.channel_settings.values()):
			self.set_channel_settings(channel_id, settings)

