import pathlib
import exdir
import datetime

from .data_structures import *

import os.path
from pony.orm import get, select
from ..config import get_config
from collections import OrderedDict

from .database import MyDatabase
from .data_structures import MeasurementState


def default_measurement_save_path(state):
    identifiers = OrderedDict()
    identifiers.update({'id': '{:06d}'.format(state.id)})
    if state.measurement_type:
        identifiers.update({'measurement_type': state.measurement_type})
    if state.sample_name:
        identifiers.update({'sample_name': state.sample_name})
    if state.comment:
        identifiers.update({'comment': state.comment})

    config = get_config()
    data_root = config['datadir']
    now = datetime.now()
    day_folder_name = now.strftime('%Y-%m-%d')

    parent = os.path.join(data_root, day_folder_name)
    # print (parent, identifiers)
    fullpath = os.path.join(parent, '-'.join(identifiers.values()))

    return fullpath


def save_exdir(state, keep_open=False):
    # parameters = []
    if not state.filename:
        state.filename = default_measurement_save_path(state)
    pathlib.Path(os.path.abspath(os.path.join(state.filename, os.pardir))).mkdir(parents=True, exist_ok=True)

    f = exdir.File(state.filename, 'w', allow_remove=True)
    f.attrs = {k: v for k, v in state.metadata.items()}
    if keep_open:
        if hasattr(state, 'exdir'):
            close_exdir(state)
        state.exdir = f
    try:
        for dataset in state.datasets.keys():
            dataset_exdir = f.create_group(str(dataset))
            parameters_exdir = dataset_exdir.create_group('parameters')
            for index in range(len(state.datasets[dataset].parameters)):
                parameter_values = state.datasets[dataset].parameters[index].values
                parameter_name = state.datasets[dataset].parameters[index].name
                parameter_unit = state.datasets[dataset].parameters[index].unit
                has_setter = True if state.datasets[dataset].parameters[index].setter else False
                d = parameters_exdir.create_dataset(str(index), dtype=np.asarray(parameter_values).dtype,
                                                    shape=np.asarray(parameter_values).shape)
                d.attrs = {'name': parameter_name, 'unit': parameter_unit, 'has_setter': has_setter}
                d.data[:] = np.asarray(parameter_values)
            data_exdir = dataset_exdir.create_dataset('data', dtype=state.datasets[dataset].data.dtype,
                                                      data=state.datasets[dataset].data)
            if keep_open:
                state.datasets[dataset].data_exdir = data_exdir
    except:
        raise
    finally:
        if not keep_open:
            f.close()


def update_exdir(state, indeces):
    for dataset in state.datasets.keys():
        state.exdir.attrs.update(state.metadata)
        try:
            state.datasets[dataset].data_exdir[tuple(indeces)] = state.datasets[dataset].data[tuple(indeces)]
        except Exception as e:
            state.datasets[dataset].data_exdir[...] = state.datasets[dataset].data[...]


def close_exdir(state):
    if hasattr(state, 'exdir'):
        for dataset in state.datasets.keys():
            try:
                del state.datasets[dataset].data_exdir
            except AttributeError:
                continue
            try:
                del state.references['current ref']
            except KeyError:
                continue
        state.exdir.close()
        del state.exdir


class LazyMeasParFromExdir:
    """
    Sweep parameter data structure.
    Data structure has a function (setter), which makes it
    impractical for serialization.
    """
    def __init__(self, exdir_parameter):
        self.exdir_parameter = exdir_parameter

    @property
    def name(self):
        return self.exdir_parameter.attrs['name']

    @property
    def setter(self):
        return self.exdir_parameter.attrs['has_setter']

    @property
    def unit(self):
        return self.exdir_parameter.attrs['unit']

    @property
    def values(self):
        return self.exdir_parameter.data

    def __str__(self):
        return '{name} lazy-loaded ({units}),:[{min}, {max}] ({num_points} points) {setter_str}'.format(#'{name} ({units}): [{min}, {max}] ({num_points} points) {setter_str}'.format(
            name=self.name,
            units=self.unit,
            min=np.min(self.values),
            max=np.max(self.values),
            num_points=len(self.values),
            setter_str='with setter' if self.setter else 'without setter')

    def __repr__(self):
        return str(self)


def load_exdir(filename, db=None, lazy=False):
    """
    Loads measurement state from ExDir file system and database if the latter is provided.

    Parameters
    ----------
    filename : str
        Absolute path to the exdir file.
    db : MyDatabase
        Binded pony database instance.
    lazy : bool
        If True, function leaves ExDir file open and sets
        retval.exdir to this file.


    Returns
    -------
    MeasurementState : retval
        Measurement state that is obtained from combining data from ExDir by filename and from
        PostSQL by finding record with the same filename
    """
    from time import time
    from sys import stdout

    # load_start = time()
    f = exdir.File(filename, 'r')
    # file_open_time = time()
    # stdout.flush()
    # print('load_exdir: file open time: ',  file_open_time - load_start)
    # stdout.flush()

    try:
        state = MeasurementState()
        if not lazy:
            state.metadata.update(f.attrs)
        else:
            state.metadata = f.attrs

        # metadata_time = time()
        # print ('load_exdir: metadata_time', metadata_time-file_open_time)
        # stdout.flush()

        for dataset_name in f.keys():
            # dataset_start_time = time()
            parameters = [None for key in f[dataset_name]['parameters'].keys()]
            for parameter_id, parameter in f[dataset_name]['parameters'].items():
                if not lazy:
                    #print (parameter.attrs)
                    parameter_name = parameter.attrs['name']
                    parameter_setter = parameter.attrs['has_setter']
                    parameter_unit = parameter.attrs['unit']
                    parameter_values = parameter.data[:].copy()
                    parameters[int(parameter_id)] = MeasurementParameter(parameter_values, parameter_setter,
                                                                         parameter_name, parameter_unit)
                else:
                    parameters[int(parameter_id)] = LazyMeasParFromExdir(parameter)
            # parameter_time = time()
            # print ('load_exdir: dataset_parameter_time: ', parameter_time - dataset_start_time)
            # stdout.flush()
            if not lazy:
                try:
                    data = f[dataset_name]['data'].data[:].copy()
                except:
                    data = f[dataset_name]['data'].data
            else:
                data = f[dataset_name]['data'].data
            state.datasets[dataset_name] = MeasurementDataset(parameters, data)
            # dataset_end_time = time()
        # print ('load_exdir: dataset_data_time: ', dataset_end_time - parameter_time)
        # stdout.flush()

        if db:
            # get db record and add info to the returned measurement state
            db_record = get(i for i in db.Data if (i.filename == filename))
            # print (filename)
            state.id = db_record.id
            state.start = db_record.start
            state.stop = db_record.stop
            state.measurement_type = db_record.measurement_type
            query = select(i for i in db.Reference if (i.this.id == state.id))
            references = {}
            for q in query:
                references.update({q.ref_type: q.that.id})
            # print(references)
            state.references = references
            state.filename = filename
        # print ('load_exdir: dataset_db_time: ', time() - dataset_end_time )
        # stdout.flush()
    except Exception as e:
        raise e
    finally:
        if not lazy:
            f.close()
        else:
            state.exdir = f
    return state
