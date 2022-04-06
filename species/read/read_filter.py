"""
Module with reading functionalities for filter profiles.
"""

import os
import warnings
import configparser

from typing import Union, Tuple

import h5py
import numpy as np

from typeguard import typechecked
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, interpolate

from species.data import database


class ReadFilter:
    """
    Class for reading a filter profile from the database.
    """

    @typechecked
    def __init__(self, filter_name: str) -> None:
        """
        Parameters
        ----------
        filter_name : str
            Filter name as stored in the database. Filter names from
            the SVO Filter Profile Service will be automatically
            downloaded, stored in the database, and read from the
            database.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        h5_file = h5py.File(self.database, "r")

        if "filters" not in h5_file or self.filter_name not in h5_file["filters"]:
            h5_file.close()
            species_db = database.Database()
            species_db.add_filter(self.filter_name)
            h5_file = h5py.File(self.database, "r")

        h5_file.close()

    @typechecked
    def get_filter(self) -> np.ndarray:
        """
        Function for selecting a filter profile from the database.

        Returns
        -------
        np.ndarray
            Array with the wavelengths and filter transmission.
        """

        h5_file = h5py.File(self.database, "r")

        data = np.asarray(h5_file[f"filters/{self.filter_name}"])

        if data.shape[0] == 2 and data.shape[1] > data.shape[0]:
            # Required for backward compatibility
            data = np.transpose(data)

        h5_file.close()

        return data

    @typechecked
    def interpolate_filter(self) -> interp1d:
        """
        Function for linearly interpolating a filter profile.

        Returns
        -------
        scipy.interpolate.interp1d
            Linearly interpolated filter.
        """

        data = self.get_filter()

        return interp1d(
            data[:, 0],
            data[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value=float("nan"),
        )

    @typechecked
    def wavelength_range(
        self,
    ) -> Tuple[Union[np.float32, np.float64], Union[np.float32, np.float64]]:
        """
        Extract the wavelength range of the filter profile.

        Returns
        -------
        float
            Minimum wavelength (um).
        float
            Maximum wavelength (um).
        """

        data = self.get_filter()

        return data[0, 0], data[-1, 0]

    @typechecked
    def mean_wavelength(self) -> Union[np.float32, np.float64]:
        """
        Calculate the weighted mean wavelength of the filter profile.

        Returns
        -------
        float
            Mean wavelength (um).
        """

        data = self.get_filter()

        return np.trapz(data[:, 0] * data[:, 1], data[:, 0]) / np.trapz(
            data[:, 1], data[:, 0]
        )

    @typechecked
    def filter_fwhm(self) -> float:
        """
        Calculate the full width at half maximum (FWHM) of the filter
        profile.

        Returns
        -------
        float
            Full width at half maximum (um).
        """

        data = self.get_filter()

        spline = InterpolatedUnivariateSpline(
            data[:, 0], data[:, 1] - np.max(data[:, 1]) / 2.0
        )
        root = spline.roots()

        diff = root - self.mean_wavelength()

        root1 = np.amax(diff[diff < 0.0])
        root2 = np.amin(diff[diff > 0.0])

        return root2 - root1

    @typechecked
    def effective_width(self) -> np.float32:
        """
        Calculate the effective width of the filter profile. The
        effective width is equivalent to the horizontal size of a
        rectangle with height equal to the maximum transmission and
        with the same area as the one covered by the filter profile.

        Returns
        -------
        float
            Effective width (um).
        """

        data = self.get_filter()

        return np.trapz(data[:, 1], data[:, 0]) / np.amax(data[:, 1])

    @typechecked
    def detector_type(self) -> str:
        """
        Return the detector type.

        Returns
        -------
        str
            Detector type ('energy' or 'photon').
        """

        with h5py.File(self.database, "r") as h5_file:
            dset = h5_file[f"filters/{self.filter_name}"]

            if "det_type" in dset.attrs:
                det_type = dset.attrs["det_type"]

            else:
                warnings.warn(
                    f"Detector type not found for {self.filter_name}. The database "
                    f"was probably created before the detector type was introduced "
                    f"in species (v0.3.1). Assuming an energy-counting detector."
                )

                det_type = "energy"

        return det_type
