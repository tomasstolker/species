"""
Module with reading functionalities for data from individual objects.
"""

import os

from configparser import ConfigParser
from typing import List, Optional, Union, Tuple

import h5py
import numpy as np

from typeguard import typechecked

from species.util.convert_util import apparent_to_absolute


class ReadObject:
    """
    Class for reading data from an individual object from the database.
    """

    @typechecked
    def __init__(self, object_name: str) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name as stored in the database (e.g. 'beta Pic b',
            'PZ Tel B').

        Returns
        -------
        NoneType
            None
        """

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        with h5py.File(self.database, "r") as h5_file:
            if f"objects/{self.object_name}" not in h5_file:
                raise ValueError(
                    f"The object '{self.object_name}' is not "
                    f"present in the database."
                )

    @typechecked
    def list_filters(self) -> List[str]:
        """
        Function for listing and returning the filter profile names for
        which there is photometric data stored in the database.

        Returns
        -------
        list(str)
            List with names of the filter profiles.
        """

        filter_list = []

        print(f"Available photometric data for {self.object_name}:")

        with h5py.File(self.database, "r") as h5_file:
            for tel_item in h5_file[f"objects/{self.object_name}"]:
                if tel_item not in ["parallax", "distance", "spectrum"]:
                    for filt_item in h5_file[f"objects/{self.object_name}/{tel_item}"]:
                        print(f"   - {tel_item}/{filt_item}")
                        filter_list.append(f"{tel_item}/{filt_item}")

        return filter_list

    @typechecked
    def get_photometry(self, filter_name: str) -> np.ndarray:
        """
        Function for reading photometric data of the object
        for a specified filter name.

        Parameters
        ----------
        filter_name : str
            Filter name as stored in the database. The
            :func:`~species.read.read_object.ReadObject.list_filters`
            method can be used for listing the filter names for
            which photometric data of the object is available.

        Returns
        -------
        np.ndarray
            Apparent magnitude, magnitude error (error),
            flux (W m-2 um-1), flux error (W m-2 um-1).
        """

        with h5py.File(self.database, "r") as h5_file:
            if filter_name in h5_file[f"objects/{self.object_name}"]:
                obj_phot = np.asarray(
                    h5_file[f"objects/{self.object_name}/{filter_name}"]
                )

            else:
                raise ValueError(
                    f"There is no photometric data of {self.object_name} "
                    f"available with the {filter_name} filter."
                )

        return obj_phot

    @typechecked
    def get_spectrum(self) -> dict:
        """
        Function for reading the spectra and covariance matrices of the
        object.

        Returns
        -------
        dict
            Dictionary with spectra and covariance matrices.
        """

        with h5py.File(self.database, "r") as h5_file:
            if f"objects/{self.object_name}/spectrum" in h5_file:
                spectrum = {}

                for item in h5_file[f"objects/{self.object_name}/spectrum"]:
                    data_group = f"objects/{self.object_name}/spectrum/{item}"

                    if f"{data_group}/covariance" not in h5_file:
                        spectrum[item] = (
                            np.asarray(h5_file[f"{data_group}/spectrum"]),
                            None,
                            None,
                            h5_file[f"{data_group}"].attrs["specres"],
                        )

                    else:
                        spectrum[item] = (
                            np.asarray(h5_file[f"{data_group}/spectrum"]),
                            np.asarray(h5_file[f"{data_group}/covariance"]),
                            np.asarray(h5_file[f"{data_group}/inv_covariance"]),
                            h5_file[f"{data_group}"].attrs["specres"],
                        )

            else:
                spectrum = None

        return spectrum

    @typechecked
    def get_distance(self) -> Tuple[float, float]:
        """
        Function for reading the distance to the object.

        Returns
        -------
        float
            Distance (pc).
        float
            Uncertainty (pc).
        """

        with h5py.File(self.database, "r") as h5_file:
            if f"objects/{self.object_name}/parallax" in h5_file:
                parallax = np.asarray(h5_file[f"objects/{self.object_name}/parallax"])
                calc_dist = 1.0 / (parallax[0] * 1e-3)  # (pc)
                dist_plus = 1.0 / ((parallax[0] - parallax[1]) * 1e-3) - calc_dist
                dist_minus = calc_dist - 1.0 / ((parallax[0] + parallax[1]) * 1e-3)
                distance = (calc_dist, (dist_plus + dist_minus) / 2.0)

            elif f"objects/{self.object_name}/distance" in h5_file:
                distance = np.asarray(h5_file[f"objects/{self.object_name}/distance"])

            else:
                raise RuntimeError(
                    f"Could not read the distance of "
                    f"{self.object_name}. Please add "
                    f"the parallax with the add_object "
                    f"method of the Database class."
                )

        return distance[0], distance[1]

    @typechecked
    def get_parallax(self) -> Tuple[float, float]:
        """
        Function for reading the parallax of the object.

        Returns
        -------
        float
            Parallax (mas).
        float
            Uncertainty (mas).
        """

        with h5py.File(self.database, "r") as h5_file:
            if f"objects/{self.object_name}/parallax" in h5_file:
                obj_parallax = np.asarray(
                    h5_file[f"objects/{self.object_name}/parallax"]
                )

            else:
                raise RuntimeError(
                    f"Could not read the parallax of "
                    f"{self.object_name}. Please add "
                    f"the parallax with the add_object "
                    f"method of the Database class."
                )

        return obj_parallax[0], obj_parallax[1]

    @typechecked
    def get_absmag(
        self, filter_name: str
    ) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Function for calculating the absolute magnitudes of the object
        from the apparent magnitudes and distance. The errors on the
        apparent magnitude and distance are propagated into an error
        on the absolute magnitude.

        Parameters
        ----------
        filter_name : str
            Filter name as stored in the database.

        Returns
        -------
        float, np.ndarray
            Absolute magnitude.
        float, np.ndarray
            Error on the absolute magnitude.
        """

        with h5py.File(self.database, "r") as h5_file:
            obj_distance = self.get_distance()

            if filter_name in h5_file[f"objects/{self.object_name}"]:
                obj_phot = np.asarray(
                    h5_file[f"objects/{self.object_name}/{filter_name}"]
                )

            else:
                raise ValueError(
                    f"There is no photometric data of '{self.object_name}' "
                    f"available with the {filter_name}."
                )

        if obj_phot.ndim == 1:
            abs_mag = apparent_to_absolute(
                (obj_phot[0], obj_phot[1]), (obj_distance[0], obj_distance[1])
            )

        elif obj_phot.ndim == 2:
            abs_mag = apparent_to_absolute(
                (obj_phot[0, :], obj_phot[1, :]), (obj_distance[0], obj_distance[1])
            )

        return abs_mag
