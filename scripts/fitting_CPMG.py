import numpy as np
import matplotlib.pyplot as plt

def gaussian_CPMG_predefined(x, y, gamma1, gammap, tayp, gammaphi):
	def model(x, p):
		A = p[0]
		B = p[1]
		return A*np.exp(-x*gamma1/2)*np.exp(-gammap*tayp)*np.exp(-(gammaphi*x)**2)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	p0 = []
	p0.append(y_first)
	p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
def gaussian_CPMG_exp(x, y):
	def model(x, p):
		gamma = p[0]
		A = p[1]
		B = p[2]
		return A*np.exp(-x*gamma)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	gamma0 = 10e06/2
	p0 = []
	p0.append(gamma0)
	p0.append(y_first)
	p0.append(y_last)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
def lorenzian_CPMG(x, y):
	def model(x, p):
		kappa = p[0]
		#kappa = 1e06
		A = p[1]
		B = p[2]
		return A/(kappa*kappa+4*np.pi**2*x*x)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()

	y = np.asarray(y)
	y_last = np.min(y)
	y_first = np.max(y)
	print(y_first, y_last)
	kappa0 = 1e06
	p0 = []
	p0.append(kappa0)
	p0.append((y_first-y_last)*kappa0*kappa0)
	p0.append(y_last)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0, ftol = 1.5e-015, xtol = 1.5e-015)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
	
def gaussian_CPMG(x, y, gamma1, gammap, tayp):
	def model(x, p):
		gammaphi = p[0]
		A = p[1]
		B = p[2]
		return A*np.exp(-x*gamma1/2)*np.exp(-gammap*tayp)*np.exp(-(gammaphi*x)**2)+B
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	gammaphi_0 = 1e05#np.sqrt(np.abs(np.log((y_first-y_last)/y_first)))/x[0]
	#print (x[0], y)
	p0 = []
	p0.append(gammaphi_0)
	p0.append(y_first - y_last)
	p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0, ftol = 1.5e-015, xtol = 1.5e-015)
	fitted_curve = model(x, fitresults[0])
	return fitresults[0], fitted_curve
	
def sum_of_lorenzians(x, y, omega_Lamb, khi, kappa, order, gamma2):
	#according to the article Qubit-photon interactions in a cavity: Measurement-induced dephasing and number splitting from Phys Rev 2006
	#from paragraph V
	#dephasing due to photon number fluctuations
	def model(x, p):
		sum = 0
		tetta_zero = 2*kappa/khi
		gamma_m = 2*kappa*p*tetta_zero**2
		gamma = []
		for j in range (order):
			gamma.append(2*(gamma2 + gamma_m[j]) + j*kappa)
			sum += ((-2*gamma_m[j]/kappa)**j)*0.5*gamma[j]/((x - omega_Lamb - 2*p[j]*khi)**2 + (gamma[j]/2)**2)
		print((np.abs(sum/(2*np.pi) - y)**2))
		return sum/(2*np.pi)
	error_function = lambda p: (np.abs(model(x,p) - y)**2).ravel()
	
	y = np.asarray(y)
	y_last = y[-1]
	y_first = y[0]
	p0 = []
	for i in range(order):
		p0.append(0.01)
	#p0.append(y_first)
	#p0.append(y_last)
	#p0 = np.asarray([gammaphi_0] + y_first.tolist() + y_last.tolist())
	#print(p0)
	
	#print(p0)
	from scipy.optimize import leastsq
	fitresults = leastsq(error_function, p0)
	fitted_curve = model(x, fitresults[0])
	#print(fitresults)
	return fitresults[0], fitted_curve
	
	
def fit_CPMG_new(N, pi2_pulse, gamma1, gammap, tayp, t, data, fitter):
	fitdata = data
	parameters, fitted_curve = fitter(t, fitdata, gamma1, gammap, tayp)
	return fitted_curve, parameters
	
def fit_CPMG(N, pi2_pulse, gamma1, gammap, tayp, data, fitter):
	t = (np.asarray(data[1]['z'][1])[0])# + N*pi2_pulse)[0]
	fitdata = data[1]['z'][2]
	parameters, fitted_curve = fitter(t, fitdata, gamma1, gammap, tayp)
	#parameters = {'gamma': parameters[0], 'amplitudes':parameters[1:]}
	
	#measurement['fit'] = [t for t in measurement[1]['z'][1]]
	#measurement['fit'] = fitted_curve[:]
	
	#if len(measurement[1]['z'])>3:
	#	measurement['fit'][3] = dict(measurement['fit'][3])
	#	if 'scatter' in measurement['S21+ fit'][3]:
	#		del measurement['S21+ fit'][3]['scatter']
	#if len(measurement['S21- fit'])>3:
	#	measurement['S21- fit'][3] = dict(measurement['S21- fit'][3])
	#	if 'scatter' in measurement['S21- fit'][3]:
	#		del measurement['S21- fit'][3]['scatter']
			
	#measurement['S21+ fit'] = tuple(measurement['S21+ fit'])
	#measurement['S21- fit'] = tuple(measurement['S21- fit'])
	#measurement['fit'] = tuple(measurement['fit'])
	return fitted_curve, parameters
	
	
	
def find_dw(a, f):
    max = np.max(a)
    eps = 0.1*max
    half = max/2
    i = 0
    if ((max<1e-04) or (max>1)):
        return 0
    #print(f)
    #print(a[0], max, half, eps)
    if (a[i] > half+eps):
        first = f[i]
    else:
        while ((abs(a[i] - half)>eps) and (i<len(a)-1)):
            i +=1
        first = f[i]
        while ((abs(a[i] - max)>eps) and (i<len(a)-1)):
            i +=1
    while ((abs(a[i] - half)>eps) and (i<len(a)-1)):
        i +=1
    #print(first, max)
    if (i==len(a)):
           return 0
    second = f[i]
    return abs(first - second)

def get_g(tay, N):
    g_array = []
    freq_for_g = []
    if (tay==0):
        return [], []
    freq_for_g = np.linspace(max(N/(2*tay) - 15e06, 1e-08), N/(2*tay) + 25e06, 50000)
    #freq_for_g = np.linspace(max(np.pi*N/(2*tay) - 25e06, 1e-08), np.pi*N/(2*tay) + 25e06, 100000)
    #else:
        #if (tay<3.5e-06):
            #freq_for_g = np.linspace(1e-06, 25e06, 5000)#(1e-06, 0.5, 20000)
        #else:
            #freq_for_g = np.linspace(1e-06, 6e06, 5000)
    if (N==0):
        freq_for_g = np.linspace(1e-07, 10e06, 10000)
   # if (N<=10):
   #     freq_for_g = np.linspace(1e-06, 8e06, 20000)#(max(freq[len(freq)-1]-1e06, 0), freq[0]+1e06, 1000) 
   # else:
   # freq_for_g = np.linspace(8e-06, 17e06, 20000)
    for i in range (len(freq_for_g)):
        g_array.append(g(freq_for_g[i], tay, N))
    return g_array, freq_for_g

def get_PSD(array, N, tays):
    PSD = []
    dw01_dl = 2e05
    index = 0
    freq = N/(2*tays)
    new = []
    g_array = []
    #freq_for_g = np.linspace(max(freq[len(freq)-1]-1e06, 0), freq[0]+1e06, 1000) 
    #for i in range (len(freq_for_g)):
    #    g_array.append(g(freq_for_g[i], tays[index], N))
    #plt.plot(freq_for_g/1e06, g_array)
    #dw = find_dw(g_array[1:], freq_for_g[1:])
    #print(dw)
    g_add = []
    freq_add =[]
    while (index < len(tays)):    
        #plt.plot(freq_for_g/1e06, g_array)
        #if ((tays[index]>3.5e-06) and (N>10)) or ((tays[index]>3.1e-06) and (N<=10)):
        g_array, freq_for_g = get_g(tays[index], N)
        if (tays[index]<3e-08):#(tays[index]==0):
            dw = 0
        else:
            dw = 2*np.pi*find_dw(g_array, freq_for_g) #check it
        if ((dw>0) and (tays[index]>0)):
            PSD.append(array[index]/(((dw01_dl*tays[index])**2)*g(freq[index], tays[index], N)*dw))    
            #print(index, tays[index], freq[index], dw, g(freq[index], tays[index], N), PSD[index])
            new.append(g(freq[index], tays[index], N)*dw*(tays[index])**2)
            index +=1
            g_add.append(g_array)
            freq_add.append(freq_for_g)
        else:
            index += 1
            PSD.append(0)
        print(index-1, N, dw, PSD[index-1], tays[index-1])
    return PSD, freq, new, g_add, freq_add
	