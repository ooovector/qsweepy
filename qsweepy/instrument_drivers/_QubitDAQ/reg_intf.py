#ADS regs to check program status
ADS_CTRL_ST_ADDR = 0x156
#ADS_CTRL_ST_VL = 0x64
ADS_CTRL_ST_VL = 0xe8
###############################
#Pulse processing module
PULSE_PROC_BASE = 0x0

#Control register
PULSE_PROC_CTRL = 0x0
#Control register bit masks
#++++++++++++++++++++++++++++++++++++
PULSE_PROC_CTRL_START           = 1<<0
PULSE_PROC_CTRL_BUSY            = 1<<1
PULSE_PROC_CTRL_ABORT           = 1<<2
PULSE_PROC_CTRL_READY           = 1<<8
PULSE_PROC_CTRL_EXT_TRIG_EN     = 1<<3
PULSE_PROC_CTRL_TRIG_EDGE       = 1<<6
PULSE_PROC_CTRL_SOFT_TRIG       = 1<<7
PULSE_PROC_CTRL_RES_DMA_BUSY    = 1<<4
PULSE_PROC_CTRL_RES_DMA_READY   = 1<<5
#++++++++++++++++++++++++++++++++++++

PULSE_PROC_CAP_LEN      = 4
PULSE_PROC_CAP_FIFO_LVL = 8
PULSE_PROC_NSEGM        = 12
PULSE_PROC_RES_DMA_FIFO_LVL  = 16
PULSE_PROC_TRIG_DLY = 20
PULSE_PROC_THRESHOLD    = 24
PULSE_PROC_DOT_AVE      = 68
#############################################
#FX3 usb chip interface module
FX3_BASE = 0x10000

#Control register
FX3_CTRL = 0x0
#Control register bit masks
#++++++++++++++++++++++++++++++++++++
FX3_CTRL_START       = 1<<0   #Read/write process start
FX3_CTRL_WR          = 1<<1   #Direction: 1-write; 0-read
FX3_CTRL_ABORT       = 1<<2   #Read/write process abort
FX3_CTRL_DDR_WR_BUSY = 1<<3   #DDR write status
FX3_CTRL_DDR_RD_BUSY = 1<<4   #DDR read status
FX3_CTRL_RESET       = 1<<5   #Global reset
FX3_CTRL_PATH        = 1<<6   #Data path: 0-DDR3; 1-onchip memory
FX3_CTRL_ONCHIP_WR_BUSY = 1<<7
FX3_CTRL_ONCHIP_RD_BUSY = 1<<8
#++++++++++++++++++++++++++++++++++++

#Other registers
FX3_LEN	            = 4	    #Read/write, data length
FX3_ADDR            = 8 	#Read/write, start address
FX3_CRC32           = 12    #Read/write, CRC32 of FPGA firmware
FX3_FIFO_LVL_DDR_WR = 16    #Read only, DDR write fifo max level
FX3_FIFO_LVL_DDR_RD = 20    #Read only, DDR read fifo max level
##############################################
#JESD204 link layer
JESD_LINK_BASE  = 0x50000

#Control register
JESD_LINK_CTRL  = 0x0
#Control register bit masks
#++++++++++++++++++++++++++++++++++++
JESD_LINK_CTRL_SYNC_REQ     = 1
JESD_LINK_CTRL_ERR_CLR      = 1<<1
JESD_LINK_CTRL_SCR_EN       = 1<<2
JESD_LINK_CTRL_CHAR_REP_EN  = 1<<3
JESD_LINK_CTRL_ERR_MASK     = 0xF<<4
#++++++++++++++++++++++++++++++++++++

JESD_LINK_RDB_OFFSET = 0x4

#Lane status registers
JESD_L = 8 #Number of lanes
JESD_LINK_STATUS = 0x8 #L registers starting from here with address increment 2
#Lane status register bit masks
#++++++++++++++++++++++++++++++++++++
JESD_LINK_STATUS_CGS_DONE       = 1
JESD_LINK_STATUS_FRAME_ALIGN    = 1<<1
JESD_LINK_STATUS_ERR            = 0xF<<2
#++++++++++++++++++++++++++++++++++++

##############################################
#JESD PLL reconfiguration
JESD_PLL_BASE = 0x20000

#Control interface mode register
JESD_PLL_MODE = 0
#+++++++++++++++++++++++
JESD_PLL_MODE_POOL   = 1
JESD_PLL_MODE_WAITRQ = 0
#++++++++++++++++++++++

JESD_PLL_STATUS = 0x1
JESD_PLL_START  = 0x2

#Dynamic phase shift register
JESD_PLL_PHASE  = 0x6
#Offsets
#++++++++++++++++++++++++++++++++
JESD_PLL_PHASE_SHIFT    = 0 #A 16-bit phase shift
JESD_PLL_PHASE_CNT_SEL  = 16 #A 5-bit counter select field
JESD_PLL_PHASE_DIR      = 21# A 1-bit phase shift direction: 1-up, 0-dn.
#++++++++++++++++++++++++++++++++

#The counter of the PLL which is responsible for clocking of the trigger
#capture circuit.
JESD_PLL_TRIG_CLK_CNT = 2
#VCO operation frequency
JESD_PLL_F_VCO = 375e6
#Output frequency of clocking of the trigger
#capture circuit.
JESD_PLL_F_TRIG_CLK = 125e6
##############################################
#Triggrt source
TRIG_SRC_BASE = 0x60000
TRIG_SRC_CTRL = 0
#Control register bit masks
#++++++++++++++++++++++++++++++++++++
TRIG_SRC_CTRL_UPDATE = 1<<0 #Set this bit after loading of new values
TRIG_SRC_CTRL_MODE =   1<<1 #Trigger source mode 0 - free run; 1 - when ready
#++++++++++++++++++++++++++++++++++++
TRIG_SRC_PERIOD_LO = 4	# 64 bit pulse period in clock cycles
TRIG_SRC_PERIOD_HI =8
TRIG_SRC_WIDTH_LO = 12	# 64 bit pulse width in clock cycles
TRIG_SRC_WIDTH_HI = 16

"""Number of state discrimination channels"""
NUM_DESCR_CH = 4
ADC_RESOLUTION = 14
# MAX_NSEGM = 32768 #DDR_DATA_BASE*512/(64*NUM_DESCR_CH)
MAX_NSEGM = 500000
MAX_NSAMP_NSEGM = 1099511365616 #(DDR_MEM_DEPTH-DDR_DATA_BASE)*512/32
PP_CLK_FREQ = 125e6

""" Memory map for the QubitDAQ board"""

"""DDR3 memory"""
DDR_RESULT_BASE =   0x0 #Dot product storage
# DDR_DATA_BASE   =   0x4000 #Data storage
DDR_DATA_BASE = 0x3D090
DDR_MEM_DEPTH   =   0xFFFFFFFFF

"""Feature memory (uses byte addressing)"""
FEATURE_MEM_BASE_ADDR = 0
FEATURE_MEM_DEPTH = 8192*4  #per channel in bytes

"""Averaged data memory (uses byte addressing)"""
AVER_MEM_BASE_ADDR = 0x20000
AVER_MEM_DEPTH = 8192*2*4