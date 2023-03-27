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
from scipy import interpolate

from species.data import database
from species.read import read_calibration


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

        else:
            h5_file.close()

    @typechecked
    def get_filter(self) -> np.ndarray:
        """
        Select a filter profile from the database.

        Returns
        -------
        np.ndarray
            Array with the wavelengths and filter transmission. The
            array has 2 dimensions with the shape (n_wavelengths, 2).
        """

        with h5py.File(self.database, "r") as h5_file:
            data = np.asarray(h5_file[f"filters/{self.filter_name}"])

            if data.shape[0] == 2 and data.shape[1] > data.shape[0]:
                # Required for backward compatibility
                data = np.transpose(data)

        return data

    @typechecked
    def interpolate_filter(self) -> interpolate.interp1d:
        """
        Interpolate a filter profile with the `interp1d <https://
        docs.scipy.org/doc/scipy/reference/generated/
        scipy.interpolate.interp1d.html>`_ function from
        ``scipy.interpolate`` and linear kind of interpolation.

        Returns
        -------
        scipy.interpolate.interp1d
            Linearly interpolated filter profile.
        """

        data = self.get_filter()

        return interpolate.interp1d(
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
            Minimum wavelength (:math:`\\mu\\mathrm{m}`).
        float
            Maximum wavelength (:math:`\\mu\\mathrm{m}`).
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
            Mean wavelength (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        return np.trapz(data[:, 0] * data[:, 1], x=data[:, 0]) / np.trapz(
            data[:, 1], x=data[:, 0]
        )

    @typechecked
    def effective_wavelength(self) -> Union[np.float32, np.float64]:
        """
        Calculate the effective wavelength of the filter profile.
        The effective wavelength is calculated as the weighted
        average based on the filter profile and the spectrum of Vega.

        Returns
        -------
        float
            Effective wavelength (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        h5_file = h5py.File(self.database, "r")

        if "spectra/calibration/vega" not in h5_file:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra("vega")

        else:
            h5_file.close()

        read_calib = read_calibration.ReadCalibration("vega")
        calib_box = read_calib.resample_spectrum(data[:, 0])

        return np.trapz(
            data[:, 0] * data[:, 1] * calib_box.flux, x=data[:, 0]
        ) / np.trapz(data[:, 1] * calib_box.flux, x=data[:, 0])

    @typechecked
    def filter_fwhm(self) -> float:
        """
        Calculate the full width at half maximum (FWHM) of the filter
        profile.

        Returns
        -------
        float
            Full width at half maximum (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        spline = interpolate.InterpolatedUnivariateSpline(
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
            Effective width (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        return np.trapz(data[:, 1], x=data[:, 0]) / np.amax(data[:, 1])

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
                    f"Detector type not found for {self.filter_name}. "
                    "The database was probably created before the "
                    "detector type was introduced in species v0.3.1. "
                    "Assuming a photon-counting detector."
                )

                det_type = "photon"

        return det_type
