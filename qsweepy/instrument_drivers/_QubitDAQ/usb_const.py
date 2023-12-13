#USB specific constans

class VendorRq:    
    """Vendor requests codes"""
    REG_READ		= 0xBB
    REG_WRITE		= 0xBA
    USB_RESET		= 0xE0
    FPGA_CONF_INIT	= 0xBC
    FPGA_CONF_FIN 	= 0xBD
    FPGA_CONF_CHK 	= 0xBE
    RESET_I2C 		= 0xBF
    FW_ID           = 0xB0    