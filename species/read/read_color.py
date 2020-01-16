"""
Module with reading functionalities for photometric libraries.
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
    Class for reading color-magnitude data from the database.
    """

    def __init__(self,
                 phot_library,
                 filters_color,
                 filter_mag):
        """
        Parameters
        ----------
        phot_library : list(str, )
            Photometric libraries ('vlm-plx', 'leggett', or 'mamajek').
        filters_color : tuple(str, str)
            Filter IDs for the color (typically in the MKO, 2MASS, or WISE system).
        filter_mag : str
            Filter ID for the absolute magnitudes (typically in the MKO, 2MASS, or WISE system).

        Returns
        -------
        NoneType
            None
        """

        self.phot_library = phot_library
        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        if isinstance(self.phot_library, str):
            self.phot_library = (self.phot_library, )

    def get_color_magnitude(self,
                            object_type=None):
        """
        Function for extracting color-magnitude data from the selected photometric libraries.

        Parameters
        ----------
        object_type : str, None
            Object type to select either field dwarfs ('field'), or young and/or low-gravity
            objects ('young'). All objects are selected if set to None.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
        """

        h5_file = h5py.File(self.database, 'r')

        indices = None

        for item in self.phot_library:
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

            if object_type is None:
                indices_tmp = np.arange(0, np.size(sptype_tmp), 1)

            elif object_type == 'field':
                indices_tmp = np.where(flag_tmp == b'null')[0]

            elif object_type == 'young':
                indices_tmp = []

                for j, object_flag in enumerate(flag_tmp):
                    if b'young' in object_flag:
                        indices_tmp.append(j)

                    elif b'lowg' in object_flag:
                        indices_tmp.append(j)

                indices_tmp = np.array(indices_tmp)

            if indices_tmp.size > 0:
                if indices is None:
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
                              library=self.phot_library,
                              object_type=object_type,
                              filters_color=self.filters_color,
                              filter_mag=self.filter_mag,
                              color=color[indices],
                              magnitude=mag[indices],
                              sptype=sptype[indices])


class ReadColorColor:
    """
    Class for reading color-color data from the database.
    """

    def __init__(self,
                 phot_library,
                 filters_colors):
        """
        Parameters
        ----------
        phot_library : list(str, )
            Photometric libraries ('vlm-plx', 'leggett', or 'mamajek').
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter IDs for the two color (typically in the MKO, 2MASS, or WISE system).

        Returns
        -------
        NoneType
            None
        """

        self.phot_library = phot_library
        self.filters_colors = filters_colors

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        if isinstance(self.phot_library, str):
            self.phot_library = (self.phot_library, )

    def get_color_color(self,
                        object_type):
        """
        Function for extracting color-color data from the selected photometric libraries.

        Parameters
        ----------
        object_type : str, None
            Object type to select either field dwarfs ('field'), or young and/or low-gravity
            objects ('young'). All objects are selected if set to None.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the colors.
        """

        h5_file = h5py.File(self.database, 'r')

        indices = None

        for item in self.phot_library:
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

            if object_type is None:
                indices_tmp = np.arange(0, np.size(sptype_tmp), 1)

            elif object_type == 'field':
                indices_tmp = np.where(flag_tmp == b'null')[0]

            elif object_type == 'young':
                indices_tmp = []

                for j, object_flag in enumerate(flag_tmp):
                    if b'young' in object_flag:
                        indices_tmp.append(j)

                    elif b'lowg' in object_flag:
                        indices_tmp.append(j)

                indices_tmp = np.array(indices_tmp)

            if indices_tmp.size > 0:
                if indices is None:
                    sptype = sptype_tmp
                    distance = distance_tmp
                    flag = flag_tmp
                    indices = indices_tmp

                    mag1 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[0][0]])
                    mag2 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[0][1]])
                    mag3 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[1][0]])
                    mag4 = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[1][1]])

                else:
                    distance_tmp = np.asarray(h5_file['photometry/'+item+'/distance'])  # [pc]
                    distance = np.concatenate((distance, distance_tmp), axis=0)

                    sptype_tmp = np.asarray(h5_file['photometry/'+item+'/sptype'])
                    sptype = np.concatenate((sptype, sptype_tmp), axis=0)

                    flag = np.concatenate((flag, flag_tmp), axis=0)
                    indices = np.concatenate((indices, indices.shape+indices_tmp), axis=0)

                    mag1_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[0][0]])
                    mag2_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[0][1]])
                    mag3_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[1][0]])
                    mag4_tmp = np.asarray(h5_file['photometry/'+item+'/'+self.filters_colors[1][1]])

                    mag1 = np.concatenate((mag1, mag1_tmp), axis=0)
                    mag2 = np.concatenate((mag2, mag2_tmp), axis=0)
                    mag3 = np.concatenate((mag3, mag3_tmp), axis=0)
                    mag4 = np.concatenate((mag4, mag4_tmp), axis=0)

        color1 = mag1 - mag2
        color2 = mag3 - mag4

        h5_file.close()

        return box.create_box(boxtype='colorcolor',
                              library=self.phot_library,
                              object_type=object_type,
                              filters=self.filters_colors,
                              color1=color1[indices],
                              color2=color2[indices],
                              sptype=sptype[indices])
