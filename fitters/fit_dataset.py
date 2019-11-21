import numpy as np
import pandas as pd
from ..ponyfiles import data_structures


def fit_dataset_1d(source_measurement, dataset_name, fitter, time_parameter_id=-1, sweep_parameter_ids=[], mode=None) -> data_structures.MeasurementState:
    ''' Fits an n-d array of measurements with 1d curve, for example exp-sin or exp (theoretical curve for Rabi, Ramsey, delay in Markov approximation).
        This function is a frontend that uses data_structures, specifically, measurement_parameter.

        :param measurement_dataset dataset: source data
        :param int time_parameter_id: id of the parameter that is the "t" in the sine and exponent
        :param iterable_of_ints linear_parameter_ids: ids of the parameters that enumerate different measurements. For example, if you
        measure the V(t_ro) of dispersive readout after a Rabi pulse (of length t_ex), each t_r point contains signal
        that performs Rabi oscillations. t_ro would be a linear_parameter (see example)
        :param iterable_of_ints sweep_parameter_ids: ids of the parameters that are

        :returns measurement: fit result
    '''

    ## turn the data into a 3d array: [sweep_parameters, linear_parameters, time]
    dataset = source_measurement.datasets[dataset_name]

    data = dataset.data
    if time_parameter_id<0: time_parameter_id = len(data.shape)+time_parameter_id
    sweep_parameter_ids_positive = [p if p>=0 else len(data.shape)+p for p in sweep_parameter_ids]

    linear_parameter_ids=list(set(np.arange(len(data.shape)))-set([time_parameter_id])-set(sweep_parameter_ids_positive))
    initial_axes_order = [(0, p) for p in sweep_parameter_ids_positive]+[(1, p) for p in linear_parameter_ids]+[(2, time_parameter_id)]
    initial_axes_order = [(a[0],a[1]) for i, a in enumerate(initial_axes_order)]
    print('initial_axes_order', initial_axes_order)
    sorted_axes_order = sorted(initial_axes_order)

    transposition = [a[1] for a in sorted_axes_order]
    #inverse_transposition = [a[2] for a in sorted_axes_order]
    inverse_transposition = [transposition.index(i) for i in range(len(transposition))]

    t = dataset.parameters[time_parameter_id].values
    t_fit = resample_x_fit(t)

    sweep_parameter_shape = np.asarray(data.shape)[sweep_parameter_ids_positive]
    linear_parameter_shape = np.asarray(data.shape)[linear_parameter_ids]

    # make a function for update so that we can replace old data with new data
    def fit_data(source_measurement_updated, indeces_updated):
        data = source_measurement_updated.datasets[dataset_name].data
        data_sorted = np.transpose(data, transposition)
        data_3d = np.reshape(data_sorted, (np.prod(sweep_parameter_shape), np.prod(linear_parameter_shape), len(t)))

        ## TODO: this is a shitty way of checking if something is complex, as it can fail on other complex datatypes.
        unpack_complex = np.iscomplexobj(data_3d)

        # load fit data from last measurement
        if hasattr(source_measurement_updated, 'fit'):
            order_amplitudes = np.asarray([a for i, a in enumerate(transposition) if i < len(linear_parameter_shape)+len(sweep_parameter_shape)])
            if len(order_amplitudes):
                removals = [i for i in range(max(order_amplitudes)) if i not in order_amplitudes]
                for removal in removals:
                    order_amplitudes[order_amplitudes>removal] -= 1
                order_amplitudes = order_amplitudes.tolist()

            order_fit_parameters = [a for i, a in enumerate(order_amplitudes) if i < len(sweep_parameter_shape)]
            if len(order_fit_parameters):
                removals = [i for i in range(max(order_fit_parameters)) if i not in order_fit_parameters ]
                for removal in removals:
                    order_fit_parameters[order_fit_parameters > removal] -= 1
                order_fit_parameters = order_fit_parameters.tolist()

            if len(linear_parameter_shape)+len(sweep_parameter_shape):
                #A_sorted = np.transpose(source_measurement_updated.fit.datasets['amplitudes'].data, order_amplitudes)
                amplitude_parameters_sorted = {k: np.transpose(v.data, order_amplitudes) for k,v in source_measurement_updated.fit.datasets.items()
                                                            if np.prod(v.data.shape)==np.prod(sweep_parameter_shape)*np.prod(linear_parameter_shape)}
            else:
                #A_sorted = source_measurement_updated.fit.datasets['amplitudes'].data
                amplitude_parameters_sorted = {k: v.data for k,v in source_measurement_updated.fit.datasets.items()
                                                            if np.prod(v.data.shape)==np.prod(sweep_parameter_shape)*np.prod(linear_parameter_shape)}
            if len(sweep_parameter_shape):
                fit_parameters_sorted = {k: np.transpose(v.data, order_fit_parameters) for k,v in source_measurement_updated.fit.datasets.items()
                                                            if np.prod(v.data.shape)==np.prod(sweep_parameter_shape)}
            else:
                fit_parameters_sorted = {k: v.data for k,v in source_measurement_updated.fit.datasets.items()
                                                            if np.prod(v.data.shape)==np.prod(sweep_parameter_shape)}

            #old_A_2d = np.reshape(A_sorted, [np.prod(sweep_parameter_shape), np.prod(linear_parameter_shape)])
            old_amplitudes_2d = {k: np.reshape(v, [np.prod(sweep_parameter_shape), np.prod(linear_parameter_shape)]) for k, v in amplitude_parameters_sorted.items()}
            if unpack_complex:
                #print('unpacking old_A_2d complex, old shape: ', old_A_2d.shape)
                #old_A_2d = np.vstack([np.real(A_sorted).T, np.imag(A_sorted).T]).T
                old_amplitudes_2d = {k: np.vstack([np.real(v).T, np.imag(v).T]) for k, v in old_amplitudes_2d.items()}
                #print('new shape: ', old_A_2d.shape)
            old_fit_parameters_1d = {k: np.reshape(v, [np.prod(sweep_parameter_shape)]) for k, v in fit_parameters_sorted.items()}

        ## initializing 3d fit array
        fit_3d_shape = [i for i in data_3d.shape]
        fit_3d_shape[2] = len(t_fit)
        fit_3d = np.zeros(fit_3d_shape, data_3d.dtype)
        fit_parameters = []


        ## initializing amplitude array
        #A = np.zeros((data_3d.shape[0], data_3d.shape[1]), data_3d.dtype)
        amplitudes = {}

        for sweep_parameter_id in range(data_3d.shape[0]):
            y = data_3d[sweep_parameter_id, :, :]
            if unpack_complex:  y_real = np.vstack((np.real(y), np.imag(y)))
            else:               y_real = y

            if hasattr(source_measurement_updated, 'fit'):
                old_parameters = {k:v[sweep_parameter_id] for k,v in old_fit_parameters_1d.items()}
                #old_parameters['A'] = old_A_2d[sweep_parameter_id,:]
                old_parameters.update({k:v[sweep_parameter_id, :] for k,v in old_amplitudes_2d.items()})
            else:
                old_parameters = None

            #print ('old_parameters:', old_parameters)
            x_fit, y_fit, fitresults = fitter.fit(t, y_real, old_parameters)

            #print ('x fit shape: ', x_fit.shape, ' x shape: ', t.shape)
            #print ('y fit shape: ', y_fit.shape, ' y real: ', y_real.shape)

            #print ('num_amplitudes: ', num_amplitudes)
            #print ('fitresults', fitresults)
            if unpack_complex:
                num_amplitudes = y_fit.shape[0] // 2
                fit_3d[sweep_parameter_id, :, :] = y_fit[:num_amplitudes,:]+1j*y_fit[num_amplitudes:,:]
                #fitresults['A'] = fitresults['A'][:num_amplitudes]+1j*fitresults['A'][num_amplitudes:]
                # if fit result is twice the length of the amplitude, build a complex out of it
                for fitresult in fitresults.keys():
                    #print ('fitresult_name', fitresult, 'value', np.asarray(fitresults[fitresult]).ravel())
                    if len(np.asarray(fitresults[fitresult]).ravel()) == num_amplitudes*2:
                        fitresults[fitresult] = fitresults[fitresult][:num_amplitudes]+1j*fitresults[fitresult][num_amplitudes:]
            else:
                fit_3d[sweep_parameter_id, :, :] = y_fit
            fit_parameters.append({k: v for k, v in fitresults.items() if not hasattr(v, '__iter__')})
            #A[sweep_parameter_id, :] = fitresults['A']
            for fitresult in fitresults.keys():
                if hasattr(fitresults[fitresult], '__iter__'):
                    if not fitresult in amplitudes:
                        amplitudes[fitresult] = np.zeros((data_3d.shape[0], data_3d.shape[1]), data_3d.dtype)
                    if unpack_complex:
                        fitresults[fitresult] = fitresults[fitresult][:num_amplitudes]+1j*fitresults[fitresult][num_amplitudes:]
                    amplitudes[fitresult][sweep_parameter_id] = fitresults[fitresult]

        fit_parameters_pd = pd.DataFrame(fit_parameters)

        ## turning fit back into original shape of data
        fit_sorted = np.reshape(fit_3d, [i for i in data_sorted.shape][:-1]+list(t_fit.shape))
        #A_sorted = np.reshape(A, [i for i in data_sorted.shape][:-1])

        amplitudes_sorted = {k: np.reshape(v, [i for i in data_sorted.shape][:-1]) for k,v in amplitudes.items()}
        if len(sweep_parameter_ids):
            fit_parameters_sorted = {fit_parameter: np.reshape(np.asarray(fit_parameters_pd[fit_parameter]), [i for i in data_sorted.shape][:len(sweep_parameter_ids)]) for fit_parameter in fit_parameters_pd.columns}
        else:
            fit_parameters_sorted = {fit_parameter: np.asarray(fit_parameters_pd[fit_parameter]) for fit_parameter in fit_parameters_pd.columns}

        ## turn fit parameters back to original order
        fit_unsorted = np.transpose(fit_sorted, inverse_transposition)

        order_amplitudes = [a for a in inverse_transposition if a < len(linear_parameter_shape)+len(sweep_parameter_shape)]
        order_fit_parameters = [a for a in order_amplitudes if a < len(sweep_parameter_shape)]

        if len(linear_parameter_shape)+len(sweep_parameter_shape):
            #A_unsorted = np.transpose(A_sorted, order_amplitudes)
            amplitudes_unsorted = {k: np.transpose(v, order_amplitudes) for k, v in amplitudes_sorted.items()}
        else:
            #A_unsorted = A_sorted
            amplitudes_unsorted = amplitudes_sorted

        if len(sweep_parameter_shape):								fit_parameters_unsorted = {k: np.transpose(v, order_fit_parameters) for k,v in fit_parameters_sorted.items()}
        else:														fit_parameters_unsorted = fit_parameters_sorted
        #print('fit_parameters_pd', fit_parameters_pd)
        #print ('fit_parameters_unsorted', fit_parameters_unsorted)

        metadata = {'fitter_name': fitter.name, 'fitted_dataset': dataset_name}
        references = {'fit_source': source_measurement.id}
        if not len(sweep_parameter_shape):
            metadata.update({k: str(v.ravel()[0]) for k, v in fit_parameters_unsorted.items()})
        if not len(linear_parameter_shape)+len(sweep_parameter_shape):
            #metadata['A'] = str(A_unsorted.ravel()[0])
            metadata.update({k: str(v.ravel()[0]) for k, v in amplitudes_unsorted.items()})
            # copy fit parameters to metadata if singleton

        #return fit_unsorted, A_unsorted, fit_parameters_unsorted, metadata, references, x_fit
        return fit_unsorted, amplitudes_unsorted, fit_parameters_unsorted, metadata, references, x_fit

    # fit_unsorted, A_unsorted, fit_parameters_unsorted, metadata, references, x_fit = fit_data(source_measurement, None)
    fit_unsorted, amplitudes_unsorted, fit_parameters_unsorted, metadata, references, x_fit = fit_data(source_measurement, None)

    # create fit dataset
    fit_dataset = data_structures.MeasurementDataset(data=fit_unsorted, parameters=[
        data_structures.MeasurementParameter(**p.__dict__) for p in dataset.parameters])
    fit_dataset.parameters[time_parameter_id].values = x_fit
    #amplitudes_dataset = data_structures.MeasurementDataset(data=A_unsorted, parameters=[dataset.parameters[i] for i in sorted(linear_parameter_ids + sweep_parameter_ids)])
    amplitudes_datasets = {k: data_structures.MeasurementDataset(data=v,
                parameters=[dataset.parameters[i] for i in sorted(linear_parameter_ids + sweep_parameter_ids)]) for k, v in amplitudes_unsorted.items()}
    fit_parameter_datasets = {k: data_structures.MeasurementDataset(data=v, parameters=[dataset.parameters[i] for i in sorted(sweep_parameter_ids)])
                              for k, v in fit_parameters_unsorted.items()}

    fit_measurement = data_structures.MeasurementState(measurement_type='fit_dataset_1d', sample_name=source_measurement.sample_name, metadata = metadata, references=references)
    fit_measurement.datasets = {dataset_name: fit_dataset}
    fit_measurement.datasets.update(fit_parameter_datasets)
    fit_measurement.datasets.update(amplitudes_datasets)

    def updater(source_measurement_updated, updated_indeces):
        #fit_unsorted, A_unsorted, fit_parameters_unsorted, metadata, references, x_fit  = fit_data(source_measurement, None)
        fit_unsorted, amplitudes_unsorted, fit_parameters_unsorted, metadata, references, x_fit = fit_data(source_measurement,
                                                                                                  None)
        print (fit_parameters_unsorted)
        if not len(sweep_parameter_shape) or not np.prod(sweep_parameter_shape):
            metadata.update({k: str(v.ravel()[0]) for k, v in fit_parameters_unsorted.items() if k not in amplitudes_unsorted})
        if not len(linear_parameter_shape)+len(sweep_parameter_shape) or not np.prod(linear_parameter_shape)*np.prod(sweep_parameter_shape):
            #metadata['A'] = str(A_unsorted.ravel()[0])
            metadata.update({k: str(v.ravel()[0]) for k, v in amplitudes_unsorted.items()})

        fit_measurement.metadata.update(metadata)
        fit_measurement.datasets[dataset_name].data[...] = fit_unsorted[...]
        #if A_unsorted.shape:
        #    fit_measurement.datasets['amplitudes'].data[...] = A_unsorted[...]
        #else:
        #    fit_measurement.datasets['amplitudes'].data = A_unsorted
        for name, fit_parameter in fit_parameters_unsorted.items():
            if name not in amplitudes_unsorted:
                if len(fit_measurement.datasets[name].data.shape):
                    fit_measurement.datasets[name].data[...] = fit_parameter[...]
                else:
                    fit_measurement.datasets[name].data = fit_parameter

        for name, amplitude in amplitudes_unsorted.items():
            if len(fit_measurement.datasets[name].data.shape):
                fit_measurement.datasets[name].data[...] = amplitude[...]
            else:
                fit_measurement.datasets[name].data = amplitude

        #raise Exception('debug')
        #print('updater called')

    source_measurement.update_fit = updater
    source_measurement.fit = fit_measurement

    return fit_measurement



def resample_x_fit(x):
    if len(x) < 500:
        return np.linspace(np.min(x), np.max(x), 501)
    else:
        return x
