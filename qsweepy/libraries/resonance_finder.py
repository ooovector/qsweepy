import numpy as np

def diff_avg(measurement, diff_axis=0, mnames=None):
    measurement = measurement.copy()
    if not mnames:
        mnames = list(measurement.keys())
    if type(mnames) is not list:
        mnames = [mnames]
    for mname in mnames:
        new_measurement = [i for i in measurement[mname]]
        new_measurement[2] = new_measurement[2] - np.mean(new_measurement[2], axis=diff_axis)
        measurement[mname+' diff avg'] = tuple(new_measurement)
    return measurement
def diff_resonance_finder(measurement, threshold, diff_axis=0, reduce_axis=0, diff_type='mean', min_separation=1, max_noise=0.1):
    if type(min_separation) is float:
        min_separation = min_separation/(measurement[1][diff_axis][1]-measurement[1][diff_axis][0])
    data = measurement[2]
    if diff_type=='mean':
        new_shape = [i for i in data.shape]
        new_shape[diff_axis] = 1
        data_diff = np.abs(data - np.reshape(np.mean(data, axis=diff_axis), new_shape))
    else:
        data_diff = np.abs(np.gradient(data)[diff_axis])
        if diff_axis == 0:
            data_diff[0,:]=0
            data_diff[-1,:]=0
        else:
            data_diff[:,0]=0
            data_diff[:,-1]=0
    resonance_freq_id = np.argmax(data_diff, axis=reduce_axis)
    invalid_mask = np.abs(resonance_freq_id[1:]-resonance_freq_id[:-1])>max_noise*data_diff.shape[reduce_axis]
    invalid_mask = np.logical_or(np.hstack((invalid_mask, 0)), np.hstack((0, invalid_mask)))
    resonance_freqs = measurement[1][reduce_axis][resonance_freq_id]
    resonance_freqs[invalid_mask] = None
    return resonance_freqs