def resonator_tools_notch_port(f, S):
    '''
    :param iterable_of_float f: frequencies at which the S-parameter has been sampled
    :param iterable_of_complex S: measured S-parameters
    :return x,z,parameters: fit results
    '''
    from resonator_tools import circuit

    fitter = circuit.notch_port(f, S.ravel())
    fitter.autofit()

    return f, fitter.z_data_sim, fitter.fitresults


def resonator_tools_reflection_port(f, S):
    '''
    :param iterable_of_float f: frequencies at which the S-parameter has been sampled
    :param iterable_of_complex S: measured S-parameters
    :return x,z,parameters: fit results
    '''
    from resonator_tools import circuit

    fitter = circuit.reflection_port(f, S.ravel())
    fitter.autofit()

    return f, fitter.z_data_sim, fitter.fitresults


class ResonatorToolsFitter:
    def __init__(self, mode='notch_port'):
        self.name = 'resonator_tools_fitter'
        self.mode = mode

    def fit(self,x,y, parameters_old=None):
        if self.mode == 'resonator_tools_notch_port':
            return resonator_tools_notch_port(x, y)
        elif self.mode == 'resonator_tools_reflection_port':
            return resonator_tools_reflection_port(x, y)