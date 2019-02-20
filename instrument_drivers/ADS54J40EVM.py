import ftd2xx 

class ADS54J40EVM:
	def __init__(self, device_id):
		self.device_id = device_id
		self.device = ftd2xx.open(device_id)
		
	def decode_addr(self, device, address, value):
		if device == "LMK04828":
			return [[address, value]]
		elif device == "ADS_ANALOG":
			return [[0x11, address//0x100], [address&0xFF, value]]
		elif device == "ADS_DIGITAL":
			return [[0x4003,(address//0x100)&0xff],[0x4004, address//0x10000],[(address&0xFF)+0x6000,value],[(address&0xFF)+0x7000,value]]    
		elif device == "ADS_LOWLVL":
			return [[address, value]]
			
	def encode_bits (self, device, decoded):
		b=15
		if device == "LMK04828": 
			byte_mask = 0xa8
		else:
			byte_mask = 0xb0
		sequence = []
		for q in decoded:
			address=[(q[0]&(2**(i))>0) for i in range(b,-1,-1)]
			data=[(q[1]&(2**(i))>0) for i in range(7,-1,-1)]
			sequence.append(address+data)
		output=[0xb8]
		for packet in sequence:
			for bit in packet:
				output.extend([byte_mask|(bit*2), (byte_mask+1)|(bit*2)])
			output.append(0xb8)
				
		return(output)
	
	def load_lmk_config(self, filename=None):
		if filename == None:
			filename = r'C:\qtlab_replacement\qsweepy\instrument_drivers\_ADS54J40\Config_ADC\LMK_100MHz_osc_10MHz_ref_Dpll.cfg'
		with open(filename, 'rb') as file:
			config = [[int (i, 16) for i in row.strip().split()[:2]] for row in file if len(row.strip().split())>1]
		r = []
		for l in config[0:108]:
			t	=	self.decode_addr("LMK04828",l[0],l[1])
			r.append(bytes(self.encode_bits("LMK04828",t)))
		for l in config[108:115]:
			t	=	self.decode_addr("ADS_ANALOG",l[0],l[1])
			r.append(bytes(self.encode_bits("ADS_ANALOG",t)))
		for l in config[115:127]:
			t	=	self.decode_addr("ADS_DIGITAL",l[0],l[1])
			r.append(bytes(self.encode_bits("ADS_DIGITAL",t)))
		for l in config[127:129]:
			t	=	self.decode_addr("ADS_LOWLVL",l[0],l[1])
			r.append(bytes(self.encode_bits("ADS_LOWLVL",t)))
		for i in r:
			ftd2xx.ftd2xx.FTD2XX.write(self.device,i)
	
	def load_ads_config(self, filename=None):
		if filename == None:
			filename = r'C:\qtlab_replacement\qsweepy\instrument_drivers\_ADS54J40\Config_ADC\ADS54J40_LMF_8224.cfg'
		with open(filename, 'rb') as file:
			config = [[int (i, 16) for i in row.strip().split()[:2]] for row in file if len(row.strip().split())>1]
		x = []
		for q in config[0:1]:
			y=self.decode_addr("LMK04828",q[0],q[1])
			x.append(bytes(self.encode_bits("LMK04828",y)))
		for q in config[1:4]:
			y=self.decode_addr("ADS_ANALOG",q[0],q[1])
			x.append(bytes(self.encode_bits("ADS_ANALOG",y)))
		for q in config[4:(len(config)-1)]:
			y=self.decode_addr("ADS_DIGITAL",q[0],q[1])
			x.append(bytes(self.encode_bits("ADS_DIGITAL",y)))
		q=config[-1]
		y=self.decode_addr("LMK04828",q[0],q[1])
		x.append(bytes(self.encode_bits("LMK04828",y)))
		for i in x:
			ftd2xx.ftd2xx.FTD2XX.write(self.device,i)