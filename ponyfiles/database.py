from pony.orm import *
from .data_structures import *
from.data_structures import  MeasurementState
from datetime import datetime
from decimal import Decimal


class MyDatabase:
    def __init__(self, provider='postgres', user='qsweepy', password='qsweepy',
                 host='localhost', database='qsweepy', port=5432):
        db = Database()

        class Data(db.Entity):
            id = PrimaryKey(int, auto=True)
            comment = Optional(str)
            measurement_type = Required(str)
            sample_name = Required(str)
            measurement_time = Optional(Decimal, 8, 2, volatile=True)
            start = Required(datetime, precision=3)
            stop = Optional(datetime, precision=3)
            filename = Optional(str)
            type_revision = Optional(str)
            incomplete = Optional(bool)
            invalid = Optional(bool)
            owner = Optional(str)

            # one-to-many references to the other tables (entities)
            metadata = Set('Metadata')
            reference_one = Set('Reference', reverse='this')
            reference_two = Set('Reference', reverse='that')
            linear_sweep = Set('Linear_sweep')  # 1d sweep parameter uniform grid description
        self.Data = Data

        class Metadata(db.Entity):
            # id = PrimaryKey(int, auto=True)
            data_id = Required(self.Data)
            name = Required(str)
            value = Required(str)
            # data = Required(Data)
            PrimaryKey(data_id, name)
        self.Metadata = Metadata

        class Reference(db.Entity):
            # id = PrimaryKey(int, auto=True)
            this = Required(self.Data)
            that = Required(self.Data)
            ref_type = Required(str)
            ref_comment = Required(str)
            PrimaryKey(this, ref_type, ref_comment)
        self.Reference = Reference

        class Linear_sweep(db.Entity):
            data_id = Required(self.Data)
            min_value = Optional(float)
            max_value = Optional(float)
            num_points = Optional(int)
            parameter_name = Required(str)
            parameter_units = Optional(str)
        self.Linear_sweep = Linear_sweep

        class Invalidations(db.Entity):
            # this_type = Required(str)
            # that_type = Required(str)
            ref_type = Required(str)
            # id = PrimaryKey(int, auto=True)
        self.Invalidations = Invalidations

        db.bind(provider, user=user, password=password, host=host, database=database, port=port)
        db.generate_mapping(create_tables=True)
        self.db = db

    def create_in_database(self, state):
        """
        Creates entity instance in program memory space
        and commits it into the database.

        Parameters
        ----------
        state : MeasurementState

        Returns
        -------
        id : int
            state database id
        """
        d = self.Data(comment=state.comment,
                      measurement_type=state.measurement_type,
                      sample_name=state.sample_name,
                      start=state.start,
                      filename=state.filename,
                      type_revision=state.type_revision,
                      owner=state.owner,
                      incomplete=True)

        for dataset in state.datasets.keys():
            for parameter in state.datasets[dataset].parameters:
                minv = np.min(parameter.values)
                maxv = np.max(parameter.values)
                number = len(parameter.values)
                name = parameter.name
                unit = parameter.unit
                self.Linear_sweep(data_id=d,
                                  min_value=minv,
                                  max_value=maxv,
                                  num_points=number,
                                  parameter_name=name,
                                  parameter_units=unit)

        for name, value in state.metadata.items():
            # print(name, value)
            self.Metadata(data_id=d, name=name, value=value)
        # print('Inserting references:', state.references)
        for ref_description, ref_that in state.references.items():
            if type(ref_description) is tuple:
                self.Reference(this=d, that=ref_that, ref_type=ref_description[0], ref_comment=ref_description[1])
            else:
                self.Reference(this=d, that=ref_that, ref_type=ref_description, ref_comment='-')

        commit()
        state.id = d.id
        return d.id

    def update_in_database(self, state):
        d = self.Data[state.id]

        d.comment = state.comment
        d.measurement_type = state.measurement_type
        d.sample_name = state.sample_name
        d.type_revision = state.type_revision
        d.incomplete = state.total_sweeps != state.done_sweeps
        d.measurement_time = state.measurement_time
        d.start = state.start
        d.stop = state.stop
        d.filename = state.filename

        for k, v in state.metadata.items():
            if k not in d.metadata.name:
                self.Metadata(data_id=d, name=k, value=str(v))
            else:
                self.Metadata[d, k].value = str(v)
        # d.metadata.update(state.metadata)
        commit()
        return d.id

    def get_from_database(self, filename = ''):
        # print(select(i for i in self.Data))
        id = get(i.id for i in self.Data if (i.filename == filename))
        # print(id)
        # d = self.Data[id]
        # state = read_exdir_new(d.filename)
        return id  # tate