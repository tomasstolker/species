"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from .. data import database
from .. analysis import photometry


class ReadColorMagnitude:
    """
    Text
    """

    def __init__(self, filters_color, filter_mag):
        """
        :param filters_color:
        :type filters_color: tuple(str, str)
        :param filter_mag:
        :type filter_mag: str

        :return: None
        """

        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_color_magnitude(self, object_type):
        """
        :param object_type:
        :type object_type:

        :return:
        :rtype:
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['photometry/vlm-plx']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_photometry('vlm-plx')
            h5_file = h5py.File(self.database, 'r')

        sptype = np.asarray(h5_file['photometry/vlm-plx/sptype'])
        flag = np.asarray(h5_file['photometry/vlm-plx/flag'])
        distance = np.asarray(h5_file['photometry/vlm-plx/distance']) # [pc]

        if object_type == 'field':
            indices = np.where(flag == b'null')[0]

        mag1 = np.asarray(h5_file['photometry/vlm-plx/'+self.filters_color[0]])
        mag2 = np.asarray(h5_file['photometry/vlm-plx/'+self.filters_color[1]])

        color = mag1 - mag2

        if self.filter_mag == self.filters_color[0]:
            mag = photometry.apparent_to_absolute(mag1, distance)

        elif self.filter_mag == self.filters_color[1]:
            mag = photometry.apparent_to_absolute(mag2, distance)

        h5_file.close()

        return color[indices], mag[indices], sptype[indices]
