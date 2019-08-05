import numpy as np
from datetime import datetime


class MeasurementParameter:
    '''
    Sweep parameter data structure.
    Data structure has a function (setter), which makes it
    impractical for serialization.
    '''
    def __init__(self, *param, **kwargs):
        self.values = param[0] if len(param) > 0 else kwargs['values']
        self.setter = param[1] if len(param) > 1 else kwargs['setter']
        self.name = param[2] if len(param) > 2 else kwargs['name']
        self.unit = param[3] if len(param) > 3 else ''
        self.pre_setter = param[4] if len(param) > 4 else None
        self.setter_time = 0

        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'unit' in kwargs:
            self.unit = kwargs['unit']
        if 'pre_setter' in kwargs:
            self.pre_setter = kwargs['pre_setter']

    def __str__(self):
        return '{name} ({units}),:[{min}, {max}] ({num_points} points) {setter_str}'.format(#'{name} ({units}): [{min}, {max}] ({num_points} points) {setter_str}'.format(
            name=self.name,
            units=self.unit,
            min=np.min(self.values),
            max=np.max(self.values),
            num_points=len(self.values),
            setter_str='with setter' if self.setter else 'without setter')

    def __repr__(self):
        return str(self)


def measurer_point_parameters(measurer):
    """Constructs dictionary of lists containing MeasurementParameter for every parameter

    Parameters
    ----------
    measurer
        an object that supports get_points(), measure(), get_dtype() and get_opts() methods. e.g. Spectrum_M3i2132

    Returns
    -------
    dict[str, list[MeasurementParameter]]
        dictionary containing lists of MeasurementParameter instances

    See Also
    --------
    sweepy.instrument_drivers.Spectrum_M3i2132 : Example for measurer that supports get_points()
    """
    point_parameters = {}
    for dataset_name, points in measurer.get_points().items():
        point_parameters[dataset_name] = []
        for dimension in points:
            name, values, unit = dimension
            point_parameters[dataset_name].append(MeasurementParameter(values, None, name, unit))
    return point_parameters


class MeasurementState:
    def __init__(self, *args, **kwargs):
        # copy constructor?
        if len(args) and not len(kwargs):
            if isinstance(args[0], MeasurementState):
                kwargs = args[0].__dict__
        # if not copy constructor, leave blank
        self.datasets = {}  # here you have datasets
        self.parameter_values = []
        self.start = datetime.now()  # time.time()
        self.stop = datetime.now()
        self.measurement_time = 0
        self.started_sweeps = 0
        self.done_sweeps = 0
        self.filename = ''
        self.id = None
        self.owner = 'qtlab'
        self.sample_name = 'anonymous_sample'
        self.comment = ''
        self.references = {}
        self.measurement_type = 'anonymous_measurement'
        self.type_revision = '0'
        # TODO: invalidation synchronization with db!!!
        self.metadata = {}
        self.total_sweeps = 0
        self.request_stop_acq = False
        self.sweep_error = None
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.metadata = {k:str(v) for k,v in self.metadata.items()}

    def __str__(self):
        format_str = '''start: {start}, started/done/total sweeps: {started}/{done}/{total}, 
Measured data: \n{datasets}'''
        datasets_str = '\n'.join(['\'{}\': {}'.format(dataset_name, dataset.__str__()) for dataset_name, dataset in self.datasets.items()])
        return format_str.format(start=self.start, started=self.started_sweeps, done=self.done_sweeps, total=self.total_sweeps, datasets=datasets_str)

    def __repr__(self):
        return str(self)


class MeasurementDataset:
    def __init__(self, parameters, data):
        self.parameters = parameters
        # TODO: rename to parameters_squeezed
        self.nonunity_parameters = [parameter for parameter in self.parameters if len(parameter.values) > 1]
        self.indices_updated = []
        self.data = data
        try:
            self.data_squeezed = np.squeeze(self.data)
        except RuntimeError:
                self.data_squeezed = self.data.ravel()

    def __getattr__(self, attr_name):
        if attr_name != 'data':
            return self.parameters[attr_name]
        else:
            return self.data

    def __str__(self):
        format_str = '''parameters: {}
data: {}'''
        return format_str.format('\n'.join(parameter.__str__() for parameter in self.parameters), self.data)

    def __repr__(self):
        return str(self)
