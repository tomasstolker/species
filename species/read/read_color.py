"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from species.core import box
from species.data import database
from species.util import phot_util


class ReadColorMagnitude:
    """
    Text
    """

    def __init__(self,
                 library,
                 filters_color,
                 filter_mag):
        """
        Parameters
        ----------
        library : tuple(str, )
            Photometric libraries.
        filters_color : tuple(str, str)
        filter_mag : str

        Returns
        -------
        NoneType
            None
        """

        self.library = library
        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        if isinstance(self.library, str):
            self.library = (self.library, )

    def get_color_magnitude(self,
                            object_type):
        """
        Parameters
        ----------
        object_type : str

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
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
            distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance'])  # [pc]
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
                distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance'])  # [pc]
                distance = np.concatenate((distance, distance_tmp), axis=0)

                sptype_tmp = np.asarray(h5_file['photometry/'+item+'/sptype'])
                sptype = np.concatenate((sptype, sptype_tmp), axis=0)

                flag = np.concatenate((flag, flag_tmp), axis=0)
                indices = np.concatenate((indices, indices.shape+indices_tmp), axis=0)

                mag1_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[0]])
                mag2_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_color[1]])

                mag1 = np.concatenate((mag1, mag1_tmp), axis=0)
                mag2 = np.concatenate((mag2, mag2_tmp), axis=0)

        color = mag1 - mag2

        if self.filter_mag == self.filters_color[0]:
            mag = phot_util.apparent_to_absolute(mag1, distance)

        elif self.filter_mag == self.filters_color[1]:
            mag = phot_util.apparent_to_absolute(mag2, distance)

        h5_file.close()

        return box.create_box(boxtype='colormag',
                              library=self.library,
                              object_type=object_type,
                              filters_color=self.filters_color,
                              filter_mag=self.filter_mag,
                              color=color[indices],
                              magnitude=mag[indices],
                              sptype=sptype[indices])


class ReadColorColor:
    """
    Text
    """

    def __init__(self,
                 library,
                 filters):
        """
        Parameters
        ----------
        library : tuple(str, )
            Photometric libraries.
        filters : tuple(tuple(str, str), tuple(str, str))
            Filter IDs for the two colors.

        Returns
        -------
        NoneType
            None
        """

        self.library = library
        self.filters = filters

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        if isinstance(self.library, str):
            self.library = (self.library, )

    def get_color_color(self,
                        object_type):
        """
        Parameters
        ----------
        object_type : str
            Object type (currently only 'field' possible). All objects are selected if set to None.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the colors.
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
            distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance'])  # [pc]
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

                mag1 = np.asarray(h5_file['photometry/'+item+'/'+self.filters[0][0]])
                mag2 = np.asarray(h5_file['photometry/'+item+'/'+self.filters[0][1]])
                mag3 = np.asarray(h5_file['photometry/'+item+'/'+self.filters[1][0]])
                mag4 = np.asarray(h5_file['photometry/'+item+'/'+self.filters[1][1]])

            else:
                distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance'])  # [pc]
                distance = np.concatenate((distance, distance_tmp), axis=0)

                sptype_tmp = np.asarray(h5_file['photometry/'+item+'/sptype'])
                sptype = np.concatenate((sptype, sptype_tmp), axis=0)

                flag = np.concatenate((flag, flag_tmp), axis=0)
                indices = np.concatenate((indices, indices.shape+indices_tmp), axis=0)

                mag1_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters[0][0]])
                mag2_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters[0][1]])
                mag3_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters[1][0]])
                mag4_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters[1][1]])

                mag1 = np.concatenate((mag1, mag1_tmp), axis=0)
                mag2 = np.concatenate((mag2, mag2_tmp), axis=0)
                mag3 = np.concatenate((mag3, mag3_tmp), axis=0)
                mag4 = np.concatenate((mag4, mag4_tmp), axis=0)

        color1 = mag1 - mag2
        color2 = mag3 - mag4

        h5_file.close()

        return box.create_box(boxtype='colorcolor',
                              library=self.library,
                              object_type=object_type,
                              filters=self.filters,
                              color1=color1[indices],
                              color2=color2[indices],
                              sptype=sptype[indices])
