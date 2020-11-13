from qsweepy.fitters.single_period_sin import sin_model

class adaptive_coil_vna_tool:
    '''
    Parallel coil current and vna frequency setter for adaptive two-tone spectroscopy
    '''
    def __init__(self, vna_freq_setter, cur_setter, model_params_tuple):
        self.vna_freq_setter = vna_freq_setter
        self.cur_setter = cur_setter
        self.model_params_tuple = model_params_tuple


    def set_coil_current_vna_freq(self, cur):
        self.cur_setter(cur)
        vna_freq = sin_model(cur, *self._model_params_tuple)
        self.vna_freq_setter(vna_freq)

