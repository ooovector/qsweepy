import tables
from numpy import *
from matplotlib.pyplot import *

#Open HDF5 data file
f = tables.open_file('data.h5', mode='r')

db = lambda x: 20*log10(x)
data_2d = array(f.root.data)
c_coord = array(f.root.column_coordinate)
r_coord = array(f.root.row_coordinate)

ind = int(len(c_coord)/2)+2

ref = array(f.root.ref)

data_2d = data_2d/ref

data_2d[::,ind]

subplot(2,2,1)
pcolormesh( c_coord, r_coord, db(abs(data_2d)) )
colorbar()

subplot(2,2,2)
plot( c_coord, db(abs(data_2d[0])) )
grid()

subplot(2,2,3)
plot( r_coord, db(abs(data_2d[::,ind] )))
grid()

show()

f.close()