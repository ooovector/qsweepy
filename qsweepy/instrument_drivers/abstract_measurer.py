import typing
from abc import *
from typing import Mapping, Tuple, Iterable, Any
import numpy as np


class AbstractMeasurer(ABC):
    """
    Abstract interface for a measurement device, such as an ADC, vector network analyzer or spectrum analyzer.
    A measurement consists of a number of datasets, which are identified by names. All functions in this interface
    return dicts, where the key is the dataset name.
    Actual measurement is performed by the measure() function, which returns an ndarray for each dataset. The shape of
    the ndarray depends of the swept parameter of the measurement device: for a VNA, for example, this could be the
    frequency.
    Swept parameters of the measurement device are returned by get_points(). For each dataset, it yields a tuple,
    containing the name of the swept parameter, an iterable with its value, and a string, containing the unit for the
    swept parameter.
    """
    def get_points(self) -> Mapping[str, Tuple[str, Iterable, str]]:
        """
        Returns a dict containing the swept parameter names, values and units.

        Returns
        -------
        Mapping[str, Tuple[str, Iterable, str]]
        """
        pass

    def get_dtype(self) -> Mapping[str, type(None)]:
        """
        Returns a dict containing the data type of the measurement result, for example float, complex or int.

        Returns
        -------
        Mapping[str, type(None)]
        """
        pass

    def get_opts(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Returns an empty dict.

        Returns
        -------
        Mapping[str, Mapping[str, Any]]
        """
        pass

    def measure(self) -> Mapping[str, np.ndarray]:
        """
        Perform a measurement and returns the results.
        Each measurement can return several ``datasets'', which correspond to different physical quantities. Dataset
        names are given by the keys.

        Returns
        -------
        Mapping[str, np.ndarray]
        """
        pass

