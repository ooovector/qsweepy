from numpy import *

CAP_BASE = 0x0

CAP_CTRL = 0x0
#Bit masks
CAP_CTRL_START = 0x0
CAP_CTRL_BUSY = 0x1
CAP_CTRL_ABORT = 0x2
CAP_CTRL_EXT_TRIG = 0x3

CAP_LEN = 0x4
CAP_SEGM_NUM = 0xC
CAP_ADDR = 0x10
CAP_FIFO = 0x8 
#############################################
#FX3 usb chip interface module
FX3_BASE= 	0x10000

FX3_CTRL = 0x0
FX3_CTRL_START =1<<0
FX3_CTRL_WR =	1<<1 	#Memory write.
FX3_CTRL_ABORT =1<<2
FX3_CTRL_BUSY = 1<<3

FX3_RST = 12 	#Soft reset
FX3_LEN	= 0x4	#Number of samples to read from memory
FX3_CRC32 = 20	#CRC32 of FPGA firmware

##############################################
JESD_BASE = 0x20000
JESD_CTRL  = 0x0

##############################################
RAM_BASE = 0x50000
COV_ST = 0x4
FIFO_ST = 0x5
COV_LEN = 60
COV_THRESH_BASE = 300
COV_THRESH_SUBBASE = 16
COV_RES_BASE = 100
COV_RES_SUBBASE = 16
COV_RESAVG_BASE = 200
COV_RESAVG_SUBBASE = 16
COV_NUMB_BASE = 400


