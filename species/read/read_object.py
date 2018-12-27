"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from .. data import database


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

    def get_magnitude(self, filter_name):
        """
        :param filter_name: Filter name.
        :type filter_name: str

        :return: Apparent magnitude (mag).
        :rtype: float
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['objects/'+self.object_name]

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_filter(self.object_name)
            h5_file = h5py.File(self.database, 'r')

        mag = np.asarray(h5_file['objects/'+self.object_name+'/'+filter_name])

        h5_file.close()

        return mag

    def get_distance(self):
        """
        :return: Distance (pc).
        :rtype: float
        """

        h5_file = h5py.File(self.database, 'r')
        distance = np.asarray(h5_file['objects/'+self.object_name+'/distance'])
        h5_file.close()

        return float(distance)
