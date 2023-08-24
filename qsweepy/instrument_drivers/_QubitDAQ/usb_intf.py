from numpy import *

"""USB device id's"""
ID_VENDOR 	= 0x4b4 #1204
ID_PRODUCT 	= 0xF1 #0xd6 #241
ID_SERIAL 	= 0x0

"""USB bmRequestType components. Use bitwice OR to construct bmRequestType.
For example: bmRequestType = RQ_TYPE_RECIPIENT_DEVICE | RQ_TYPE_TYPE_VENDOR | RQ_TYPE_DIRECTION_FROM_DEV"""

RQ_TYPE_RECIPIENT_DEVICE 	= 0
RQ_TYPE_RECIPIENT_INTERFACE = 1
RQ_TYPE_RECIPIENT_ENDPOINT 	= 2
RQ_TYPE_RECIPIENT_OTHER 	= 3

RQ_TYPE_TYPE_STANDARD	= 0
RQ_TYPE_TYPE_CLASS		= 1<<5
RQ_TYPE_TYPE_VENDOR		= 2<<5
RQ_TYPE_TYPE_RESERVED	= 3<<5

RQ_TYPE_DIRECTION_TO_DEV	= 0
RQ_TYPE_DIRECTION_FROM_DEV	= 1<<7

RQ_TYPE_DEV_VEND_RD = RQ_TYPE_RECIPIENT_DEVICE | RQ_TYPE_TYPE_VENDOR | RQ_TYPE_DIRECTION_FROM_DEV
RQ_TYPE_DEV_VEND_WR = RQ_TYPE_RECIPIENT_DEVICE | RQ_TYPE_TYPE_VENDOR | RQ_TYPE_DIRECTION_TO_DEV

"""Vendor requests"""
VEND_RQ_REG_READ		= 0xBB
VEND_RQ_REG_WRITE		= 0xBA
VEND_RQ_USB_RESET		= 0xE0
VEND_RQ_FPGA_CONF_INIT	= 0xBC
VEND_RQ_FPGA_CONF_FIN 	= 0xBD
VEND_RQ_FPGA_CONF_CHK 	= 0xBE
VEND_RQ_RESET_I2C 		= 0xBF

"""Endpoits addresses for heavy data transfer"""
EP_IN	=	0x81
EP_OUT	=	0x01
	
def to_bytes(number, len, order = '<'):
	"""Converts number into n bytes for sending via USB control transfers as a payload."""
	if order=='>':
		return [(number>>(8*i))&0xff for i in range(len-1,-1,-1)]
	elif order =='<' :
		return [(number>>(8*i))&0xff for i in range(len)]
	else:
		raise ValueError('Order must be "<" or ">"!')

def mk_val_ind(val32):
	"""Converts 32-bit address into Value and Index for USB vendor requests."""
	Value = uint16(uint32(val32)>>16)
	Index = uint16(uint32(val32)&0xffff)
	return (Value, Index)

"""Data type to represent bytes received FPGA registers reading via I2C"""
uint32odd = dtype(uint32).newbyteorder('>')