import tables
from numpy import *
from matplotlib.pyplot import *
from matplotlib.widgets import Button

#Open HDF5 data file


db = lambda x: 20*log10(x)

ax1 = subplot(2,2,1)
ax2 = subplot(2,2,2)
ax3 = subplot(2,2,3)

def update(event):
	f = tables.open_file('data.h5', mode='r')
	data_2d = array(f.root.data)
	c_coord = array(f.root.column_coordinate)
	r_coord = array(f.root.row_coordinate)
	ind = int(len(c_coord)/2)+2
	ref = array(f.root.ref)
	data_2d = data_2d/ref
	
	ax1.clear()
	m1=ax1.pcolormesh( c_coord, r_coord, db(abs(data_2d)) )
	ax2.clear()
	m2 = ax2.plot( c_coord, db(abs(data_2d[0])) )
	ax2.grid()
	ax3.clear()
	m3 = ax3.plot( r_coord, db(abs(data_2d[::,ind] )))
	ax3.grid()
	gcf().canvas.draw()
	f.close()
	return m1, m2, m3

m1,m2,m3 = update(0)
colorbar(m1, ax=ax1)

ax_upd = axes([0.81, 0.05, 0.1, 0.075])
b_upd = Button(ax_upd, 'Update')
b_upd.on_clicked(update)
show()