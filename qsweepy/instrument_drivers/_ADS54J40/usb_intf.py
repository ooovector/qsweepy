from numpy import *

#USB device id's
class id:
	VENDOR = 1204
	PRODUCT = 241

#USB vendor request direction
class vend_req_dir:
	WR = 0x40
	RD = 0xC0
	
#USB vendor requests
class vend_req:
	REG_READ = 0xbb
	REG_WRITE = 0xba
	FPGA_CONF_INIT = 0xb2
	FPGA_CONF_FIN = 0xb1
class endpoints:
	IN = 0x81
	OUT = 0x1
	
def to_bytes(number, len, order = '>'):
	if order=='>':
		return [(number>>(8*i))&0xff for i in range(len-1,-1,-1)]
	elif order =='<' :
		return [(number>>(8*i))&0xff for i in range(len)]
	else:
		raise ValueError('Order must be "<" or ">"!')
		
def mk_val_ind( val32):
		Value = uint16(uint32(val32)>>16)
		Index = uint16(uint32(val32)&0xffff)
		return (Value, Index)		

uint32odd = dtype(uint32).newbyteorder('>')