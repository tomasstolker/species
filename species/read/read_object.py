"""
Module for reading object data.
"""

import os
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

        h5_file = h5py.File(self.database, 'r')
        obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])
        h5_file.close()

        return obj_phot

    def get_spectrum(self):
        """
        Returns
        -------
        numpy.ndarray
            Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
        """

        h5_file = h5py.File(self.database, 'r')
        spectrum = np.asarray(h5_file['objects/'+self.object_name+'/spectrum'])
        h5_file.close()

        return spectrum

    def get_instrument(self):
        """
        Returns
        -------
        str
            Instrument that was used for the spectrum.
        """

        h5_file = h5py.File(self.database, 'r')
        dset = h5_file['objects/'+self.object_name+'/spectrum']
        instrument = dset.attrs['instrument']
        h5_file.close()

        return instrument

    def get_distance(self):
        """
        Returns
        -------
        float
            Distance (pc).
        """

        h5_file = h5py.File(self.database, 'r')
        obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        h5_file.close()

        return float(obj_distance)

    def get_absmag(self,
                   filter_name):
        """
        Parameters
        ----------
        filter_name : str
            Filter name.

        Returns
        -------
        float, float
            Absolute magnitude (mag), magnitude error (error).
        """

        h5_file = h5py.File(self.database, 'r')
        obj_distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        obj_phot = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])
        h5_file.close()

        abs_mag = phot_util.apparent_to_absolute(obj_phot[0], obj_distance)

        return abs_mag, obj_phot[1]
