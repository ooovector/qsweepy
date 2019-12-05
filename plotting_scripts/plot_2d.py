import tables
from numpy import *
from matplotlib.pyplot import *

#Open HDF5 data file
f = tables.open_file('data.h5', mode='r')

db = lambda x: 20*log10(x)
data_2d = array(f.root.data)
c_coord = array(f.root.column_coordinate)
r_cord = array(f.root.row_coordinate)

pcolormesh( c_coord, r_cord, db(abs(data_2d)) )
colorbar()
show()

f.close()