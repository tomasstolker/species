"""
Module with reading functionalities for filter profiles.
"""

import os
import warnings

from configparser import ConfigParser
from typing import Optional, Tuple, Union

import h5py
import numpy as np

from typeguard import typechecked
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from species.data.filter_data.filter_data import add_filter_profile
from species.data.spec_data.spec_vega import add_vega


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

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.data_folder = config["species"]["data_folder"]

        with h5py.File(self.database, "r") as hdf5_file:
            # Check if the filter is found in 'r' mode
            # because the 'a' mode is not possible when
            # using multiprocessing
            if f"filters/{self.filter_name}" in hdf5_file:
                filter_found = True
            else:
                filter_found = False

        if not filter_found:
            with h5py.File(self.database, "a") as hdf5_file:
                add_filter_profile(self.data_folder, hdf5_file, self.filter_name)

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

        with h5py.File(self.database, "r") as hdf5_file:
            data = np.asarray(hdf5_file[f"filters/{self.filter_name}"])

            if data.shape[0] == 2 and data.shape[1] > data.shape[0]:
                # Required for backward compatibility
                data = np.transpose(data)

        return data

    @typechecked
    def interpolate_filter(self) -> interp1d:
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

        return np.trapezoid(data[:, 0] * data[:, 1], x=data[:, 0]) / np.trapezoid(
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

        filter_profile = self.get_filter()

        with h5py.File(self.database, "r") as hdf5_file:
            # Check if the Vega spectrum is found in 'r'
            # mode because the 'a' mode is not possible
            # when using multiprocessing
            if "spectra/calibration/vega" in hdf5_file:
                vega_found = True
            else:
                vega_found = False

        if not vega_found:
            with h5py.File(self.database, "a") as hdf5_file:
                add_vega(self.data_folder, hdf5_file)

        with h5py.File(self.database, "r") as hdf5_file:
            vega_spec = np.array(hdf5_file["spectra/calibration/vega"])

        flux_interp = interp1d(
            vega_spec[0,],
            vega_spec[1,],
            bounds_error=False,
            fill_value="extrapolate",
        )

        flux_filter = flux_interp(filter_profile[:, 0])

        return np.trapezoid(
            filter_profile[:, 0] * filter_profile[:, 1] * flux_filter,
            x=filter_profile[:, 0],
        ) / np.trapezoid(filter_profile[:, 1] * flux_filter, x=filter_profile[:, 0])

    @typechecked
    def filter_fwhm(self) -> Optional[float]:
        """
        Calculate the full width at half maximum (FWHM)
        of the filter profile.

        Returns
        -------
        float, None
            Full width at half maximum (:math:`\\mu\\mathrm{m}`).
            Returns ``None`` if the filter has only one wavelength.
            with a non-zero transmission.
        """

        data = self.get_filter()

        if len(np.nonzero(data[:, 1])[0]) > 1:
            spline = InterpolatedUnivariateSpline(
                data[:, 0], data[:, 1] - np.max(data[:, 1]) / 2.0
            )
            root = spline.roots()

            diff = root - self.mean_wavelength()

            root1 = np.amax(diff[diff < 0.0])
            root2 = np.amin(diff[diff > 0.0])

            filt_fwhm = root2 - root1

        else:
            filt_fwhm = None

        return filt_fwhm

    @typechecked
    def effective_width(self) -> Union[np.float32, np.float64]:
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

        return np.trapezoid(data[:, 1], x=data[:, 0]) / np.amax(data[:, 1])

    @typechecked
    def detector_type(self) -> str:
        """
        Return the detector type.

        Returns
        -------
        str
            Detector type ('energy' or 'photon').
        """

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"filters/{self.filter_name}"]

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
