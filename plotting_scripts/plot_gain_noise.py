from matplotlib.pyplot import *
from numpy import *
import pickle
from qsweepy import gain_noise
import scipy.constants as sc

data = pickle.load(open('raw_data.pkl', 'rb'))

a = 1./( 10.**(6./10.) )

G, Tn = gain_noise.gain_noise(data['F_S'], [data['S']['cold']['P'], data['S']['hot']['P']],
	[[data['S']['cold']['T'], 1]], 
	[[data['S']['hot']['T'], a], [data['S']['cold']['T'], 1-a] ] )

subplot(2,3,1)
plot(data['F_S'], 10*log10(data['S']['cold']['P']))
plot(data['F_S'], 10*log10(data['S']['hot']['P']))
plot(data['F_S'], 10*log10(data['S']['signal']['P']))
grid()

subplot(2,3,2)

plot(data['F_S21'], 20*log10( abs(data['S21_off']) ))
plot(data['F_S21'], 20*log10( abs(data['S21_on']) ))
grid()

subplot(2,3,3)
plot( data['F_S21'], 20*log10( abs(data['S21_on'])) - 20*log10( abs(data['S21_off']) ))
grid()

subplot(2,3,4)
plot(data['F_S'], Tn)
plot(data['F_S'], 2*pi*data['F_S']*sc.hbar/(2*sc.k), '--')
ylim([0,1])
grid()

subplot(2,3,5)
plot(data['F_S'], 10*log10(G) )
grid()

show()
print (a)