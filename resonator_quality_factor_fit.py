import pandas as pd
import numpy as np
from resonator_tools import circuit
#resonator_tools = imp.load_source('circuit', 'C:/python27/lib/site-packages/resonator_tools/circuit.py').load_module()
	
def resonator_quality_factor_fit(measurement, sweep_parameter_values, sweep_parameter_name='power', resonator_type='notch_port', delay=None, use_calibrate=False):
	fitresults = []
	sweep_parameter = measurement[1][0]
	f_data = measurement[1][1]
	z_data = measurement[2]

	#
	max_power_id = np.argmax(sweep_parameter)
	if resonator_type == 'notch_port':
		fitter = circuit.notch_port(f_data = f_data, z_data_raw=z_data[max_power_id,:])
	else:
		fitter = circuit.reflection_port(f_data = f_data, z_data_raw=z_data[max_power_id,:])
	delay, amp_norm, alpha, fr, Ql, A2, frcal = \
		fitter.do_calibration(f_data, z_data[max_power_id,:],ignoreslope=True,guessdelay=False)

	for power_id, power in enumerate(sweep_parameter_values):
		try:
			if use_calibrate:
				fitter.z_data = fitter.do_normalization(fitter.f_data,z_data[power_id,:],delay,amp_norm,alpha,A2,frcal)
				fitter.fitresults = fitter.circlefit(fitter.f_data,fitter.z_data,fr,Ql,refine_results=True,calc_errors=True)
			else:
				if resonator_type == 'notch_port':
					fitter = circuit.notch_port(f_data = f_data, z_data_raw=z_data[power_id,:])
				else:
					fitter = circuit.reflection_port(f_data = f_data, z_data_raw=z_data[power_id,:])
				fitter.autofit(electric_delay=delay)
			fitter.fitresults[sweep_parameter_name] = power
			fitter.fitresults['single_photon_limit'] = fitter.get_single_photon_limit()
			fitresults.append(fitter.fitresults.copy())
		except:
			pass
		#plt.figure(power_id)
		#fitter.plotall()
		#print(fitter.fitresults)
	return pd.DataFrame(fitresults)