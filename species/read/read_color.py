"""
Module with reading functionalities of color and magnitude data from photometric and
spectral libraries.
"""

import os
import configparser

import h5py
import numpy as np

from species.core import box
from species.read import read_spectrum
from species.util import phot_util


class ReadColorMagnitude:
    """
    Class for reading color-magnitude data from the database.
    """

    def __init__(self,
                 library,
                 filters_color,
                 filter_mag):
        """
        Parameters
        ----------
        library : str
            Photometric ('vlm-plx' or 'leggett') or spectral ('irtf' or 'spex') library.
        filters_color : tuple(str, str)
            Filter names for the color. For a photometric library, these have to be present in
            the database (typically in the MKO, 2MASS, or WISE system). For a spectral library,
            any filter names can be provided as long as they overlap with the wavelength range
            of the spectra.
        filter_mag : str
            Filter name for the absolute magnitudes (see also description of ``filters_color``).

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

        with h5py.File(self.database, 'r') as hdf_file:
            if 'photometry' in hdf_file and self.library in hdf_file['photometry']:
                self.lib_type = 'phot_lib'

            elif 'spectra' in hdf_file and self.library in hdf_file['spectra']:
                self.lib_type = 'spec_lib'

            else:
                raise ValueError(f'The \'{self.library}\' library is not present in the database.')

    def get_color_magnitude(self,
                            object_type=None):
        """
        Function for extracting color-magnitude data from the selected library.

        Parameters
        ----------
        object_type : str, None
            Object type for which the colors and magnitudes are extracted. Either field dwarfs
            ('field') or young/low-gravity objects ('young'). All objects are selected if set
            to None.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
        """

        h5_file = h5py.File(self.database, 'r')

        if self.lib_type == 'phot_lib':
            sptype = np.asarray(h5_file[f'photometry/{self.library}/sptype'])
            distance = np.asarray(h5_file[f'photometry/{self.library}/distance'])  # [pc]
            flag = np.asarray(h5_file[f'photometry/{self.library}/flag'])

            if object_type is None:
                indices = np.arange(0, np.size(sptype), 1)

            elif object_type == 'field':
                indices = np.where(flag == b'null')[0]

            elif object_type == 'young':
                indices = []

                for j, object_flag in enumerate(flag):
                    if b'young' in object_flag:
                        indices.append(j)

                    elif b'lowg' in object_flag:
                        indices.append(j)

                indices = np.array(indices)

            if indices.size > 0:
                mag1 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_color[0]}'])
                mag2 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_color[1]}'])

            color = mag1 - mag2

            if self.filter_mag == self.filters_color[0]:
                mag = phot_util.apparent_to_absolute(mag1, distance)

            elif self.filter_mag == self.filters_color[1]:
                mag = phot_util.apparent_to_absolute(mag2, distance)

            colormag_box = box.create_box(boxtype='colormag',
                                          library=self.library,
                                          object_type=object_type,
                                          filters_color=self.filters_color,
                                          filter_mag=self.filter_mag,
                                          color=color[indices],
                                          magnitude=mag[indices],
                                          sptype=sptype[indices])

        elif self.lib_type == 'spec_lib':
            read_spec_0 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_color[0])

            read_spec_1 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_color[1])

            read_spec_2 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filter_mag)

            phot_box_0 = read_spec_0.get_magnitude(sptypes=None)
            phot_box_1 = read_spec_1.get_magnitude(sptypes=None)
            phot_box_2 = read_spec_2.get_magnitude(sptypes=None)

            colormag_box = box.create_box(boxtype='colormag',
                                          library=self.library,
                                          object_type=object_type,
                                          filters_color=self.filters_color,
                                          filter_mag=self.filter_mag,
                                          color=phot_box_0.app_mag-phot_box_1.app_mag,
                                          magnitude=phot_box_2.abs_mag,
                                          sptype=phot_box_0.sptype)

        h5_file.close()

        return colormag_box


class ReadColorColor:
    """
    Class for reading color-color data from the database.
    """

    def __init__(self,
                 library,
                 filters_colors):
        """
        Parameters
        ----------
        library : str
            Photometric ('vlm-plx' or 'leggett') or spectral ('irtf' or 'spex') library.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names for the colors. For a photometric library, these have to be present in
            the database (typically in the MKO, 2MASS, or WISE system). For a spectral library,
            any filter names can be provided as long as they overlap with the wavelength range
            of the spectra.

        Returns
        -------
        NoneType
            None
        """

        self.library = library
        self.filters_colors = filters_colors

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

        with h5py.File(self.database, 'r') as hdf_file:
            if 'photometry' in hdf_file and self.library in hdf_file['photometry']:
                self.lib_type = 'phot_lib'

            elif 'spectra' in hdf_file and self.library in hdf_file['spectra']:
                self.lib_type = 'spec_lib'

            else:
                raise ValueError(f'The \'{self.library}\' library is not present in the database.')

    def get_color_color(self,
                        object_type):
        """
        Function for extracting color-color data from the selected library.

        Parameters
        ----------
        object_type : str, None
            Object type for which the colors and magnitudes are extracted. Either field dwarfs
            ('field') or young/low-gravity objects ('young'). All objects are selected if set
            to None.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the colors.
        """

        h5_file = h5py.File(self.database, 'r')

        if self.lib_type == 'phot_lib':
            sptype = np.asarray(h5_file[f'photometry/{self.library}/sptype'])
            flag = np.asarray(h5_file[f'photometry/{self.library}/flag'])

            if object_type is None:
                indices = np.arange(0, np.size(sptype), 1)

            elif object_type == 'field':
                indices = np.where(flag == b'null')[0]

            elif object_type == 'young':
                indices = []

                for j, object_flag in enumerate(flag):
                    if b'young' in object_flag:
                        indices.append(j)

                    elif b'lowg' in object_flag:
                        indices.append(j)

                indices = np.array(indices)

            mag1 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_colors[0][0]}'])
            mag2 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_colors[0][1]}'])
            mag3 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_colors[1][0]}'])
            mag4 = np.asarray(h5_file[f'photometry/{self.library}/{self.filters_colors[1][1]}'])

            color1 = mag1 - mag2
            color2 = mag3 - mag4

            colorbox = box.create_box(boxtype='colorcolor',
                                      library=self.library,
                                      object_type=object_type,
                                      filters=self.filters_colors,
                                      color1=color1[indices],
                                      color2=color2[indices],
                                      sptype=sptype[indices])

        elif self.lib_type == 'spec_lib':
            read_spec_0 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_colors[0][0])

            read_spec_1 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_colors[0][1])

            read_spec_2 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_colors[1][0])

            read_spec_3 = read_spectrum.ReadSpectrum(spec_library=self.library,
                                                     filter_name=self.filters_colors[1][1])

            phot_box_0 = read_spec_0.get_magnitude(sptypes=None)
            phot_box_1 = read_spec_1.get_magnitude(sptypes=None)
            phot_box_2 = read_spec_2.get_magnitude(sptypes=None)
            phot_box_3 = read_spec_3.get_magnitude(sptypes=None)

            colorbox = box.create_box(boxtype='colorcolor',
                                      library=self.library,
                                      object_type=object_type,
                                      filters=self.filters_colors,
                                      color1=phot_box_0.app_mag-phot_box_1.app_mag,
                                      color2=phot_box_2.app_mag-phot_box_3.app_mag,
                                      sptype=phot_box_0.sptype)

        h5_file.close()

        return colorbox
