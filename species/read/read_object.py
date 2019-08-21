"""
Module for reading object data.
"""

import os
import math
import configparser

import h5py
import numpy as np

from species.util import phot_util


class ReadObject:
    """
    Reading object data from the database.
    """

    def __init__(self,
                 object_name):
        """
        Parameters
        ----------
        object_name : str
            Object name.

        Returns
        -------
        None
        """

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        with h5py.File(self.database, 'r') as h5_file:
            if 'objects/'+self.object_name not in h5_file:
                raise ValueError(f'{self.object_name} is not present in the database')

    def get_photometry(self,
                       filter_name):
        """
        Parameters
        ----------
        filter_name : str
            Filter name.

        Returns
        -------
        numpy.ndarray
            Apparent magnitude (mag), magnitude error (error), apparent flux (W m-2 micron-1),
            flux error (W m-2 micron-1).
        """

        with h5py.File(self.database, 'r') as h5_file:
            obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])

        return obj_phot

    def get_spectrum(self):
        """
        Returns
        -------
        numpy.ndarray
            Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
        """

        with h5py.File(self.database, 'r') as h5_file:
            spectrum = np.asarray(h5_file['objects/'+self.object_name+'/spectrum'])

        return spectrum

    def get_instrument(self):
        """
        Returns
        -------
        str
            Instrument that was used for the spectrum.
        """

        with h5py.File(self.database, 'r') as h5_file:
            dset = h5_file['objects/'+self.object_name+'/spectrum']
            instrument = dset.attrs['instrument']

        return instrument

    def get_distance(self):
        """
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
        Computes the absolute magnitude from the apparent magnitude and distance. The error
        on the distance is propagated into the error on the absolute magnitude.

        Parameters
        ----------
        filter_name : str
            Filter name.

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
