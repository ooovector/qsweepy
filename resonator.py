def measure_resonance(exdir_db_inst, pna, criterion='min_abs'):
    #    print ('resonator_id: ', resonator_id)
    # turn of all second tones
    lo1_status = lo1.get_status()
    lo1.set_status(0)
    # prior bandwidth
    bandwidth = pna.get_bandwidth()
    pna.set_bandwidth(100)
    average = pna.get_average()
    pna.set_average(0)
    # pna.set_power(-60)
    # pna.set_nop(501)
    nop = pna.get_nop()
    xlim = pna.get_xlim()
    pna.set_nop((qubits[resonator_id]['r']['Fr_max'] -
                 qubits[resonator_id]['r']['Fr_min']) / qubits[resonator_id]['r']['dFr'] + 1)
    # pna.set_nop((30e6)/qubits[resonator_id]['r']['dFr']+1)
    pna.set_xlim(qubits[resonator_id]['r']['Fr_min'], qubits[resonator_id]['r']['Fr_max'])
    # pna.set_xlim(qubits[qubit_id]['r']['Fr_min']-10e6, qubits[qubit_id]['r']['Fr_max']+20e6)
    # current_src1.set_current(current)

    # measure S21
    results = []
    S21 = []
    S21_results = []
    for iteration_id in range(check + 1):
        S21 = pna.measure()['S-parameter']
        freqs = pna.get_points()['S-parameter'][0][1]

        if (criterion == 'max_dev_complex'):
            measurement = S21 * np.exp(2 * np.pi * 1j * freqs * delay)
            # plt.figure()
            # plt.plot(freqs, np.real(measurement-np.mean(measurement)))
            # plt.plot(freqs, np.imag(measurement-np.mean(measurement)))
            # plt.plot(freqs, np.abs(measurement-np.mean(measurement)))
            results.append(freqs[np.argmax(np.abs(measurement - np.mean(measurement)))])
        if (criterion == 'min_abs'):
            measurement = np.abs(S21)
            results.append(freqs[np.argmin(measurement)])
            S21_results.append(S21[np.argmin(measurement)])
    # if resonator is off by more 3 times tolerance, raise if raise_on_error, otherwise warn
    if (np.std(results) > qubits[resonator_id]['r']['dFr'] * 10):
        if raise_on_error:
            raise Exception('Could not find resonator')
        else:
            print('Could not find resonator')
    #     else:
    #         print('resonator found at '+str(np.mean(results)),end="\r")
    lo1.set_status(lo1_status)
    pna.set_bandwidth(bandwidth)
    pna.set_average(average)
    pna.set_nop(nop)
    pna.set_xlim(*xlim)

    if S21_r is True:
        return np.mean(results), np.mean(np.asarray(S21_results))
    else:
        return np.mean(results)