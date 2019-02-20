"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from species.analysis import photometry
from species.data import database


class ReadColorMagnitude:
    """
    Text
    """

    def __init__(self,
                 library,
                 filters_color,
                 filter_mag):
        """
        :param library: Photometric libraries.
        :type library: tuple(str, )
        :param filters_color:
        :type filters_color: tuple(str, str)
        :param filter_mag:
        :type filter_mag: str

        :return: None
        """

        self.library = library
        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def get_color_magnitude(self,
                            object_type):
        """
        :param object_type:
        :type object_type: str

        :return:
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')

        for i, item in enumerate(self.library):
            try:
                h5_file['photometry/'+item]

            except KeyError:
                h5_file.close()
                species_db = database.Database()
                species_db.add_photometry(item)
                h5_file = h5py.File(self.database, 'r')

            sptype_tmp = np.asarray(h5_file['photometry/'+item+'/sptype'])
            distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance']) # [pc]
            flag_tmp = np.asarray(h5_file['photometry/'+item+'/flag'])

            if object_type == 'field':
                indices_tmp = np.where(flag_tmp == b'null')[0]
            else:
                indices_tmp = np.arange(0, np.size(sptype_tmp), 1)

            if i == 0:
                sptype = sptype_tmp
                distance = distance_tmp
                flag = flag_tmp
                indices = indices_tmp

                mag1 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[0]])
                mag2 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[1]])

            else:
                distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance']) # [pc]
                distance = np.concatenate((distance, distance_tmp), axis=0)

                sptype_tmp = np.asarray(h5_file['photometry/'+item+'/sptype'])
                sptype = np.concatenate((sptype, sptype_tmp), axis=0)

                flag = np.concatenate((flag, flag_tmp), axis=0)
                indices = np.concatenate((indices, indices_tmp), axis=0)

                mag1_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[0]])
                mag2_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[1]])

                mag1 = np.concatenate((mag1, mag1_tmp), axis=0)
                mag2 = np.concatenate((mag2, mag2_tmp), axis=0)

        color = mag1 - mag2

        if self.filter_mag == self.filters_color[0]:
            mag = photometry.apparent_to_absolute(mag1, distance)

        elif self.filter_mag == self.filters_color[1]:
            mag = photometry.apparent_to_absolute(mag2, distance)

        h5_file.close()

        return color[indices], mag[indices], sptype[indices]
