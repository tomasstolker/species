"""
Module with reading functionalities for data from individual objects.
"""

import os
import math
import configparser

import h5py
import numpy as np

from species.util import phot_util


class ReadObject:
    """
    Class for reading data from an individual object from the database.
    """

    def __init__(self,
                 object_name):
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
            if 'objects/'+self.object_name not in h5_file:
                raise ValueError(f'The object \'{self.object_name}\' is not present in the '
                                 f'database.')

    def get_photometry(self,
                       filter_name):
        """
        Function for extracting the photometry of the object.

        Parameters
        ----------
        filter_name : str
            Filter ID.

        Returns
        -------
        numpy.ndarray
            Apparent magnitude (mag), magnitude error (error), flux (W m-2 micron-1),
            flux error (W m-2 micron-1).
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])

        return obj_phot

    def get_spectrum(self):
        """
        Function for extracting the spectrum of the object.

        Returns
        -------
        numpy.ndarray
            Wavelength (micron), flux (W m-2 micron-1), and flux error (W m-2 micron-1).
        """

        with h5py.File(self.database, 'r') as h5_file:
            spectrum = np.asarray(h5_file['objects/'+self.object_name+'/spectrum'])

        return spectrum

    def get_instrument(self):
        """
        Function for extracting the instrument name of the spectrum.

        Returns
        -------
        str
            Instrument that was used for the spectrum.
        """

        with h5py.File(self.database, 'r') as h5_file:
            if 'objects/'+self.object_name+'/spectrum' in h5_file:
                dset = h5_file['objects/'+self.object_name+'/spectrum']
                instrument = dset.attrs['instrument']

            else:
                instrument = None

        return instrument

    def get_distance(self):
        """
        Function for extracting the distance to the object.

        Returns
        -------
        float
            Distance (pc).
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])[0]

        return float(obj_distance)

    def get_absmag(self,
                   filter_name):
        """
        Function for calculating the absolute magnitudes of the object from the apparent
        magnitudes and distance. The errors on the apparent magnitude and distance are propagated
        into an error on the absolute magnitude.

        Parameters
        ----------
        filter_name : str
            Filter ID.

        Returns
        -------
        float, float
            Absolute magnitude (mag), uncertainty (mag).
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
            obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])

        abs_mag = phot_util.apparent_to_absolute(obj_phot[0], obj_distance[0])

        dist_err = obj_distance[1] * (5./(obj_distance[0]*math.log(10.)))
        abs_err = math.sqrt(obj_phot[1]**2 + dist_err**2)

        return abs_mag, abs_err
