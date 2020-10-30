"""
Module with reading functionalities for data from individual objects.
"""

import os
import configparser

from typing import Optional, Union, Tuple

import h5py
import numpy as np

from typeguard import typechecked

from species.util import phot_util


class ReadObject:
    """
    Class for reading data from an individual object from the database.
    """

    @typechecked
    def __init__(self,
                 object_name: str) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database (e.g. 'beta Pic b', 'PZ Tel B').

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        with h5py.File(self.database, 'r') as h5_file:
            if f'objects/{self.object_name}' not in h5_file:
                raise ValueError(f'The object \'{self.object_name}\' is not present in the '
                                 f'database.')

    @typechecked
    def get_photometry(self,
                       filter_name: str) -> np.ndarray:
        """
        Function for reading the photometry of the object.

        Parameters
        ----------
        filter_name : str
            Filter ID.

        Returns
        -------
        np.ndarray
            Apparent magnitude, magnitude error (error), flux (W m-2 um-1),
            flux error (W m-2 um-1).
        """

        with h5py.File(self.database, 'r') as h5_file:
            if filter_name in h5_file[f'objects/{self.object_name}']:
                obj_phot = np.asarray(h5_file[f'objects/{self.object_name}/{filter_name}'])

            else:
                raise ValueError(f'There is no photometric data of {self.object_name} '
                                 f'available with the {filter_name} filter.')

        return obj_phot

    @typechecked
    def get_spectrum(self) -> dict:
        """
        Function for reading the spectra and covariance matrices of the object.

        Returns
        -------
        dict
            Dictionary with spectra and covariance matrices.
        """

        with h5py.File(self.database, 'r') as h5_file:
            if f'objects/{self.object_name}/spectrum' in h5_file:
                spectrum = {}

                for item in h5_file[f'objects/{self.object_name}/spectrum']:
                    data_group = f'objects/{self.object_name}/spectrum/{item}'

                    if f'{data_group}/covariance' not in h5_file:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          None,
                                          None,
                                          h5_file[f'{data_group}'].attrs['specres'])

                    else:
                        spectrum[item] = (np.asarray(h5_file[f'{data_group}/spectrum']),
                                          np.asarray(h5_file[f'{data_group}/covariance']),
                                          np.asarray(h5_file[f'{data_group}/inv_covariance']),
                                          h5_file[f'{data_group}'].attrs['specres'])

            else:
                spectrum = None

        return spectrum

    @typechecked
    def get_distance(self) -> Tuple[float, float]:
        """
        Function for reading the distance to the object.

        Returns
        -------
        float
            Distance (pc).
        float
            Uncertainty (pc).
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_distance = np.asarray(h5_file[f'objects/{self.object_name}/distance'])

        return obj_distance[0], obj_distance[1]

    @typechecked
    def get_absmag(self,
                   filter_name: str) -> Union[Tuple[float, Optional[float]],
                                              Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Function for calculating the absolute magnitudes of the object from the apparent
        magnitudes and distance. The errors on the apparent magnitude and distance are propagated
        into an error on the absolute magnitude.

        Parameters
        ----------
        filter_name : str
            Filter name.

        Returns
        -------
        float, np.ndarray
            Absolute magnitude.
        float, np.ndarray
            Error on the absolute magnitude.
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_distance = np.asarray(h5_file[f'objects/{self.object_name}/distance'])

            if filter_name in h5_file[f'objects/{self.object_name}']:
                obj_phot = np.asarray(h5_file[f'objects/{self.object_name}/{filter_name}'])

            else:
                raise ValueError(f'There is no photometric data of \'{self.object_name}\' '
                                 f'available with the {filter_name}.')

        if obj_phot.ndim == 1:
            abs_mag = phot_util.apparent_to_absolute((obj_phot[0], obj_phot[1]),
                                                     (obj_distance[0], obj_distance[1]))

        elif obj_phot.ndim == 2:
            abs_mag = phot_util.apparent_to_absolute((obj_phot[0, :], obj_phot[1, :]),
                                                     (obj_distance[0], obj_distance[1]))

        return abs_mag
