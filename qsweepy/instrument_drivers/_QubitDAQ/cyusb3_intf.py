from ctypes import *
import weakref
import numpy as np
from typing import Optional, Tuple
import os
import sys

module_dir = os.path.dirname(os.path.abspath(__file__))

if sys.maxsize > 2**32:
    dll_path = module_dir + r"\x64\cyusb_intf.dll"
else:
    dll_path = module_dir + r"\Win32\cyusb_intf.dll"

def mk_val_ind( val32:int )->Tuple[int, int]:
    """Converts 32-bit address into Value and Index for USB vendor requests."""
    Value = int( np.uint32( val32 ) >> 16 )
    Index = int( np.uint32( val32 ) & 0xffff )
    return (Value, Index) 

BULK_WRITE_BURST_SIZE = 1
BULK_READ_BURST_SIZE = 16
MAX_BULK_XFER_SIZE = 4000000

#USB bulk endpoints addresses
class Ep:
    IN = 0x81
    OUT = 0x01

class DeviceList(Structure):
    _fields_ = [('serials', c_wchar_p ),
                ('count', c_int)]
DeviceList_p = POINTER( DeviceList )

def FindDevices()->str:
    lib = CDLL(dll_path)
    lib.USBList.restype = DeviceList_p
    res = lib.USBList()
    print( "Devices found: ", res.contents.count )
    print( "Devices serial numbers: ", res.contents.serials )
    return res.contents.serials

class USBDevice(object):
    def __init__(self, serial):
        self.serial = serial
        self.lib = CDLL(dll_path)

        self.lib.CtrlRead.restype = c_long
        self.lib.CtrlWrite.restype = c_long
        self.lib.OpenDevice.restype = c_void_p
        self.lib.MaxPacketSize.restype = c_int
        self.lib.BulkXfer.restype = c_bool
        
        self.device_handle = None

        self._max_bulk_read_size = MAX_BULK_XFER_SIZE
        self._max_bulk_write_size = MAX_BULK_XFER_SIZE

        self._finalizer = weakref.finalize(self, self._cleanup, self.lib, self.device_handle)
        self.connect()
        if self.device_handle is None:
            raise Exception("Qubit DAQ device s/n {:s} not found".format(serial) )
        self._get_max_xfer_sizes()

    @staticmethod
    def _cleanup( lib, device_handle ):
        if device_handle is not None:
            lib.CloseDevice( device_handle )

    def connect(self):
        self.device_handle = self.lib.OpenDevice( self.serial)

    def close(self):
        self.lib.CloseDevice( self.device_handle )
        self.device_handle = None

    def max_packet_size(self)->int:
        return int( self.lib.MaxPacketSize(self.device_handle) )

    def _get_max_xfer_sizes(self):
        pkt_size = self.max_packet_size()
        self._max_bulk_read_size = pkt_size * BULK_READ_BURST_SIZE* ( MAX_BULK_XFER_SIZE // (pkt_size * BULK_READ_BURST_SIZE) )
        self._max_bulk_write_size = pkt_size * BULK_WRITE_BURST_SIZE* ( MAX_BULK_XFER_SIZE // (pkt_size * BULK_WRITE_BURST_SIZE) )

    def ctrl_read(self, req_code: int, value: int, index: int, length: int) -> bytes:
        buf = create_string_buffer( length )
        bytes_read = self.lib.CtrlRead( self.device_handle, req_code, value, index, buf, length )
        return bytes( buf )[:bytes_read]

    def ctrl_write(self, req_code: int, value: int, index: int, data: Optional[bytes] = None) -> int:
        if data is not None:
            length = len( data )
            buf = create_string_buffer( data )
        else:
            length = 0
            buf = create_string_buffer( 0 )
        bytes_written = self.lib.CtrlWrite( self.device_handle, req_code, value, index, buf, length )
        return int( bytes_written )

    def gpif_read(self, length)->bytes:
        buf = create_string_buffer( length )
        success = self.lib.BulkXfer( self.device_handle, c_ubyte( Ep.IN ), buf, length, self._max_bulk_read_size)
        if not success:
            raise Exception("Read from bulk endpoint {:x} failed".format(Ep.IN) )
        return bytes(buf)
        
    def gpif_write(self, data: bytes)->None:
        length = len( data )
        buf = create_string_buffer( data )
        success = self.lib.BulkXfer( self.device_handle, c_ubyte( Ep.OUT ), buf, length, self._max_bulk_write_size)
        if not success:
            raise Exception("Write to bulk endpoint {:x} failed".format(Ep.OUT) )