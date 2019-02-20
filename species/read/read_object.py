"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np


class ReadObject:
    """
    Text
    """

    def __init__(self, object_name):
        """
        :param object_name: Object name.
        :type object_name: str

        :return: None
        """

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_photometry(self, filter_name):
        """
        :param filter_name: Filter name.
        :type filter_name: str

        :return: Wavelength (micron), apparent magnitude (mag), magnitude error (error),
                 apparent flux (W m-2 micron-1), flux error (W m-2 micron-1).
        :rtype: numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')
        photometry = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])
        h5_file.close()

        return photometry

    def get_distance(self):
        """
        :return: Distance (pc).
        :rtype: float
        """

        h5_file = h5py.File(self.database, 'r')
        distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        h5_file.close()

        return float(distance)
