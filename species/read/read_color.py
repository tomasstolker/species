"""
Module with reading functionalities of color and magnitude data from
photometric and libraries.
"""

import os
import configparser

from typing import Optional, Tuple

import h5py
import numpy as np

from typeguard import typechecked

from species.core import box
from species.read import read_spectrum
from species.util import phot_util


class ReadColorMagnitude:
    """
    Class for reading color-magnitude data from the database.
    """

    @typechecked
    def __init__(
        self, library: str, filters_color: Tuple[str, str], filter_mag: str
    ) -> None:
        """
        Parameters
        ----------
        library : str
            Photometric ('vlm-plx' or 'leggett') or spectral ('irtf' or
            'spex') library.
        filters_color : tuple(str, str)
            Filter names for the color. For a photometric library,
            these have to be present in the database (typically in the
            MKO, 2MASS, or WISE system). For a spectral library, any
            filter names can be provided as long as they overlap with
            the wavelength range of the spectra.
        filter_mag : str
            Filter name for the absolute magnitudes (see also
            description of ``filters_color``).

        Returns
        -------
        NoneType
            None
        """

        self.library = library
        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        with h5py.File(self.database, "r") as hdf_file:
            if "photometry" in hdf_file and self.library in hdf_file["photometry"]:
                self.lib_type = "phot_lib"

            elif "spectra" in hdf_file and self.library in hdf_file["spectra"]:
                self.lib_type = "spec_lib"

            else:
                raise ValueError(
                    f"The '{self.library}' library is not present in the database."
                )

    @typechecked
    def get_color_magnitude(self, object_type: Optional[str] = None) -> box.ColorMagBox:
        """
        Function for extracting color-magnitude data from the selected
        library.

        Parameters
        ----------
        object_type : str, None
            Object type for which the colors and magnitudes are
            extracted. Either field dwarfs ('field') or
            young/low-gravity objects ('young'). All objects are
            selected if set to ``None``.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
        """

        if self.lib_type == "phot_lib":
            with h5py.File(self.database, "r") as h5_file:
                sptype = np.asarray(h5_file[f"photometry/{self.library}/sptype"])
                parallax = np.asarray(h5_file[f"photometry/{self.library}/parallax"])
                parallax_error = np.asarray(
                    h5_file[f"photometry/{self.library}/parallax_error"]
                )
                flag = np.asarray(h5_file[f"photometry/{self.library}/flag"])
                obj_names = np.asarray(h5_file[f"photometry/{self.library}/name"])

                for i in range(sptype.shape[0]):
                    if isinstance(sptype[i], bytes):
                        sptype[i] = sptype[i].decode("utf-8")

                    if isinstance(flag[i], bytes):
                        flag[i] = flag[i].decode("utf-8")

                    if isinstance(obj_names[i], bytes):
                        obj_names[i] = obj_names[i].decode("utf-8")

            if object_type is None:
                indices = np.arange(0, np.size(sptype), 1)

            elif object_type == "field":
                indices = np.where(flag == "null")[0]

            elif object_type == "young":
                indices = []

                for j, object_flag in enumerate(flag):
                    if "young" in object_flag:
                        indices.append(j)

                    elif "lowg" in object_flag:
                        indices.append(j)

                indices = np.array(indices)

            if indices.size > 0:
                with h5py.File(self.database, "r") as h5_file:
                    filter_list = []

                    for item in list(h5_file[f"photometry/{self.library}/MKO"]):
                        filter_list.append(f"MKO/{item}")

                    for item in list(h5_file[f"photometry/{self.library}/2MASS"]):
                        filter_list.append(f"2MASS/{item}")

                    for item in list(h5_file[f"photometry/{self.library}/Keck"]):
                        filter_list.append(f"Keck/{item}")

                    if f"photometry/{self.library}/{self.filters_color[0]}" in h5_file:
                        mag1 = np.asarray(
                            h5_file[f"photometry/{self.library}/{self.filters_color[0]}"]
                        )

                    else:
                        raise ValueError(f"The \'{self.filters_color[0]}\' "
                                         "filter is not available in the "
                                         f"\'{self.library}\' library. Please "
                                         "select any of the following "
                                         f"filters: {filter_list}")

                    if f"photometry/{self.library}/{self.filters_color[1]}" in h5_file:
                        mag2 = np.asarray(
                            h5_file[f"photometry/{self.library}/{self.filters_color[1]}"]
                        )

                    else:
                        raise ValueError(f"The \'{self.filters_color[1]}\' "
                                         "filter is not available in the "
                                         f"\'{self.library}\' library. Please "
                                         "select any of the following "
                                         f"filters: {filter_list}")

            else:
                raise ValueError(
                    f"There is not data available from '{self.library}' for "
                    f"'{object_type}' type objects with the chosen filters."
                )

            color = mag1 - mag2

            distance = phot_util.parallax_to_distance((parallax, parallax_error))

            if self.filter_mag == self.filters_color[0]:
                mag, _ = phot_util.apparent_to_absolute(
                    (mag1, None), (distance[0], distance[1])
                )

            elif self.filter_mag == self.filters_color[1]:
                mag, _ = phot_util.apparent_to_absolute(
                    (mag2, None), (distance[0], distance[1])
                )

            else:
                raise ValueError("The argument of filter_mag is set "
                                 f"to {self.filter_mag} but should be "
                                 "the same as one of the two values of "
                                 "filters_color.")

            color = color[indices]
            mag = mag[indices]
            sptype = sptype[indices]
            obj_names = obj_names[indices]

            indices = []
            for i in range(color.size):
                if not np.isnan(color[i]) and not np.isnan(mag[i]):
                    indices.append(i)

            colormag_box = box.create_box(
                boxtype="colormag",
                library=self.library,
                object_type=object_type,
                filters_color=self.filters_color,
                filter_mag=self.filter_mag,
                color=color[indices],
                magnitude=mag[indices],
                sptype=sptype[indices],
                names=obj_names[indices],
            )

        elif self.lib_type == "spec_lib":
            read_spec_0 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_color[0]
            )

            read_spec_1 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_color[1]
            )

            read_spec_2 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filter_mag
            )

            phot_box_0 = read_spec_0.get_magnitude(sptypes=None)
            phot_box_1 = read_spec_1.get_magnitude(sptypes=None)
            phot_box_2 = read_spec_2.get_magnitude(sptypes=None)

            colormag_box = box.create_box(
                boxtype="colormag",
                library=self.library,
                object_type=object_type,
                filters_color=self.filters_color,
                filter_mag=self.filter_mag,
                color=phot_box_0.app_mag[:, 0] - phot_box_1.app_mag[:, 0],
                magnitude=phot_box_2.abs_mag[:, 0],
                sptype=phot_box_0.sptype,
                names=None,
            )

        return colormag_box


class ReadColorColor:
    """
    Class for reading color-color data from the database.
    """

    @typechecked
    def __init__(
        self, library: str, filters_colors: Tuple[Tuple[str, str], Tuple[str, str]]
    ) -> None:
        """
        Parameters
        ----------
        library : str
            Photometric ('vlm-plx' or 'leggett') or spectral ('irtf' or
            'spex') library.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names for the colors. For a photometric library,
            these have to be present in the database (typically in the
            MKO, 2MASS, or WISE system). For a spectral library, any
            filter names can be provided as long as they overlap with
            the wavelength range of the spectra.

        Returns
        -------
        NoneType
            None
        """

        self.library = library
        self.filters_colors = filters_colors

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        with h5py.File(self.database, "r") as hdf_file:
            if "photometry" in hdf_file and self.library in hdf_file["photometry"]:
                self.lib_type = "phot_lib"

            elif "spectra" in hdf_file and self.library in hdf_file["spectra"]:
                self.lib_type = "spec_lib"

            else:
                raise ValueError(
                    f"The '{self.library}' library is not present in the database."
                )

    @typechecked
    def get_color_color(self, object_type: Optional[str] = None) -> box.ColorColorBox:
        """
        Function for extracting color-color data from the selected
        library.

        Parameters
        ----------
        object_type : str, None
            Object type for which the colors and magnitudes are
            extracted. Either field dwarfs ('field') or
            young/low-gravity objects ('young'). All objects are
            selected if set to ``None``.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the colors.
        """

        if self.lib_type == "phot_lib":
            h5_file = h5py.File(self.database, "r")

            sptype = np.asarray(h5_file[f"photometry/{self.library}/sptype"])
            flag = np.asarray(h5_file[f"photometry/{self.library}/flag"])
            obj_names = np.asarray(h5_file[f"photometry/{self.library}/name"])

            if object_type is None:
                indices = np.arange(0, np.size(sptype), 1)

            elif object_type == "field":
                indices = np.where(flag == "null")[0]

            elif object_type == "young":
                indices = []

                for j, object_flag in enumerate(flag):
                    if "young" in object_flag:
                        indices.append(j)

                    elif "lowg" in object_flag:
                        indices.append(j)

                indices = np.array(indices)

            mag1 = np.asarray(
                h5_file[f"photometry/{self.library}/{self.filters_colors[0][0]}"]
            )
            mag2 = np.asarray(
                h5_file[f"photometry/{self.library}/{self.filters_colors[0][1]}"]
            )
            mag3 = np.asarray(
                h5_file[f"photometry/{self.library}/{self.filters_colors[1][0]}"]
            )
            mag4 = np.asarray(
                h5_file[f"photometry/{self.library}/{self.filters_colors[1][1]}"]
            )

            color1 = mag1 - mag2
            color2 = mag3 - mag4

            color1 = color1[indices]
            color2 = color2[indices]
            sptype = sptype[indices]
            obj_names = obj_names[indices]

            indices = []
            for i in range(color1.size):
                if not np.isnan(color1[i]) and not np.isnan(color2[i]):
                    indices.append(i)

            colorbox = box.create_box(
                boxtype="colorcolor",
                library=self.library,
                object_type=object_type,
                filters=self.filters_colors,
                color1=color1[indices],
                color2=color2[indices],
                sptype=sptype[indices],
                names=obj_names[indices],
            )

            h5_file.close()

        elif self.lib_type == "spec_lib":
            read_spec_0 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_colors[0][0]
            )

            read_spec_1 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_colors[0][1]
            )

            read_spec_2 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_colors[1][0]
            )

            read_spec_3 = read_spectrum.ReadSpectrum(
                spec_library=self.library, filter_name=self.filters_colors[1][1]
            )

            phot_box_0 = read_spec_0.get_magnitude(sptypes=None)
            phot_box_1 = read_spec_1.get_magnitude(sptypes=None)
            phot_box_2 = read_spec_2.get_magnitude(sptypes=None)
            phot_box_3 = read_spec_3.get_magnitude(sptypes=None)

            colorbox = box.create_box(
                boxtype="colorcolor",
                library=self.library,
                object_type=object_type,
                filters=self.filters_colors,
                color1=phot_box_0.app_mag[:, 0] - phot_box_1.app_mag[:, 0],
                color2=phot_box_2.app_mag[:, 0] - phot_box_3.app_mag[:, 0],
                sptype=phot_box_0.sptype,
                names=None,
            )

        return colorbox
