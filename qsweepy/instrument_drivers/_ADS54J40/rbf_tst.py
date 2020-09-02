from usb_intf import *
import usb.core
import time
from numpy import *
from matplotlib.pyplot import *

def to_bytes(number, len, order = '>'):
	if order=='>':
		return [(number>>(8*i))&0xff for i in range(len-1,-1,-1)]
	elif order =='<' :
		return [(number>>(8*i))&0xff for i in range(len)]
	else:
		raise ValueError('order must be "<" or ">"')

dev=usb.core.find(idVendor=1204, idProduct=241)
'''
dev.reset()
n=0
while((dev is None) and n<50):
	try:
		dev=usb.core.find(idVendor=1204, idProduct=241)
	except:
		time.sleep(0.02)
		pass
	n+=1
	
'''	
n=0
while(n<50):
	try:
		dev.set_configuration(1)
		print (n)
		break
	except:
		time.sleep(0.02)
		pass
	n+=1

#print (dev)	

rbf = open("qubit_daq.rbf", 'rb')
data = rbf.read()

#print(data[packet_size*n_packets:])
print(len(data))
print(to_bytes(len(data)+2,4), to_bytes(len(data)+2,4, order = '<'))

dev.ctrl_transfer(vend_req_dir.WR, vend_req.FPGA_CONF_INIT, 0, 0, to_bytes(len(data)+2,4, order = '>') )

print("Sending data")

dev.write( endpoints.OUT, data+bytes([0,0]))

print(dev.ctrl_transfer(vend_req_dir.RD, vend_req.FPGA_CONF_FIN, 0, 0, 2 ))