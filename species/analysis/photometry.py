"""
Module with functionalities for calculating synthetic photometry.
"""

import os
import math
import warnings
import configparser

from typing import List, Optional, Union, Tuple

import h5py
import numpy as np

from typeguard import typechecked

from species.data import database
from species.read import read_filter, read_calibration
from species.util import phot_util


class SyntheticPhotometry:
    """
    Class for calculating synthetic photometry from a spectrum and also
    for converting between magnitudes and fluxes. Any filter from the
    `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/svo/
    theory/fps/>`_ will be automatically downloaded and added to the
    database. Also the detector type (energy- or photon-counting) will
    be fetched. For a photon-counting detector, an additional
    wavelength factor is included in the integral for calculating the
    synthetic photometry, although typically the impact of the factor
    on the calculated flux is negligible. It is also important to note
    that by default the magnitude of Vega is set to 0.03 for all
    filters. The value can be adjusted in the `configuration file
    <https://species.readthedocs.io/en/latest/configuration.html>`_.
    """

    @typechecked
    def __init__(self, filter_name: str, zero_point: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        filter_name : str
            Filter name by which the profile is stored in database.
            Any filter from the `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_ will be
            automatically downloaded and added to the database.
        zero_point : float, None
            Zero-point flux (:math:`\\mathrm{W}`
            :math:`\\mathrm{m}^{-2}` :math:`\\mu\\mathrm{m}^{-1}`) for
            ``filter_name``. This flux is equalized to the magnitude of
            Vega, which by default is set to 0.03 for all filters. The
            value can be adjusted in the `configuration file <https://
            species.readthedocs.io/en/latest/configuration.html>`_.
            By default, the argument of ``zero_point`` is set to
            ``None``, in which case the zero point is calculated
            internally. The zero point can be accessed through
            ``zero_point`` attribute from instance of
            :class:`~species.analysis.photometry.SyntheticPhotometry`.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name
        self.zero_point = zero_point
        self.filter_interp = None
        self.wavel_range = None

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.vega_mag = float(config["species"]["vega_mag"])

        read_filt = read_filter.ReadFilter(self.filter_name)
        self.det_type = read_filt.detector_type()

        if self.zero_point is None:
            self.zero_point = self.calc_zero_point()

        else:
            warnings.warn(
                "Please note that a manually provided zero-point flux "
                "is by default equalized to a magnitude of 0.03 for "
                "all filters. The magnitude of Vega can be adjusted "
                "in the configuration file (see https://species."
                "readthedocs.io/en/latest/configuration.html) by "
                "setting the 'vega_mag' parameter. Currently the "
                f"parameter is set to {self.vega_mag}."
            )

    @typechecked
    def calc_zero_point(self) -> np.float64:
        """
        Internal function for calculating the zero point of the
        provided ``filter_name``. The zero point is here defined
        as the flux of Vega, which by default is set to a
        magnitude of 0.03 for all filters.

        Returns
        -------
        float
            Zero-point flux (:math:`\\mathrm{W}`
            :math:`\\mathrm{m}^{-2}` :math:`\\mu\\mathrm{m}^{-1}`).
        """

        if self.wavel_range is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        h5_file = h5py.File(self.database, "r")

        if "spectra/calibration/vega" not in h5_file:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectra("vega")
            h5_file = h5py.File(self.database, "r")

        read_calib = read_calibration.ReadCalibration("vega", None)
        calib_box = read_calib.get_spectrum()

        wavelength = calib_box.wavelength
        flux = calib_box.flux

        wavelength_crop = wavelength[
            (wavelength > self.wavel_range[0]) & (wavelength < self.wavel_range[1])
        ]

        flux_crop = flux[
            (wavelength > self.wavel_range[0]) & (wavelength < self.wavel_range[1])
        ]

        h5_file.close()

        return self.spectrum_to_flux(wavelength_crop, flux_crop)[0]

    @typechecked
    def spectrum_to_flux(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[np.ndarray] = None,
        threshold: Optional[float] = 0.05,
    ) -> Tuple[
        Union[float, np.float32, np.float64],
        Union[Optional[float], Optional[np.float32], Optional[np.float64]],
    ]:
        """
        Function for calculating the average flux from a spectrum and
        a filter profile. The uncertainty is propagated by sampling
        200 random values from the error distributions.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength points (um).
        flux : np.ndarray
            Flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        error : np.ndarray, None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`). Not used if set to ``None``.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the
            minimum transmission value is larger than the threshold,
            a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter
            profile. The parameter is not used if set to ``None``.

        Returns
        -------
        float
            Average flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        float, None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        """

        # Remove fluxes that are a NaN

        nan_idx = np.isnan(flux)

        if np.sum(nan_idx) > 0:
            warnings.warn(
                f"Found {np.sum(nan_idx)} fluxes with NaN. Removing "
                "these spectral fluxes from the input data before "
                "calculating synthetic photometry."
            )

            wavelength = wavelength[~nan_idx]
            flux = flux[~nan_idx]

            if error is not None:
                error = error[~nan_idx]

        if error is not None:
            # The error calculation requires the original
            # spectrum because spectrum_to_flux is used
            wavel_error = wavelength.copy()
            flux_error = flux.copy()

        if self.filter_interp is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.filter_interp = transmission.interpolate_filter()

            if self.wavel_range is None:
                self.wavel_range = transmission.wavelength_range()

        if wavelength.size == 0:
            syn_flux = np.nan

            if error is not None:
                error_flux = np.nan
            else:
                error_flux = None

            indices = None

            warnings.warn(
                f"Calculation of the mean flux for {self.filter_name} "
                "is not possible because the wavelength array is "
                "empty. Returning a NaN for the flux."
            )

        else:
            indices = np.where(
                (self.wavel_range[0] <= wavelength) & (wavelength <= self.wavel_range[1])
            )[0]

        if indices is not None and indices.size < 2:
            syn_flux = np.nan

            if error is not None:
                error_flux = np.nan
            else:
                error_flux = None

            warnings.warn(
                "Calculating a synthetic flux requires more than "
                "one wavelength point. Photometry is set to NaN."
            )

        else:
            if threshold is None and (
                wavelength[0] > self.wavel_range[0]
                or wavelength[-1] < self.wavel_range[1]
            ):
                warnings.warn(
                    f"The filter profile of {self.filter_name} "
                    f"({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) extends "
                    f"beyond the wavelength range of the spectrum ({wavelength[0]:.4f} "
                    f"-{wavelength[-1]:.4f}). The flux is set to NaN. Setting the "
                    f"'threshold' parameter will loosen the wavelength constraints."
                )

                syn_flux = np.nan

            else:
                wavelength = wavelength[indices]
                flux = flux[indices]

                transmission = self.filter_interp(wavelength)

                if (
                    threshold is not None
                    and (transmission[0] > threshold or transmission[-1] > threshold)
                    and (
                        wavelength[0] < self.wavel_range[0]
                        or wavelength[-1] > self.wavel_range[-1]
                    )
                ):
                    warnings.warn(
                        f"The filter profile of {self.filter_name} "
                        f"({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) "
                        f"extends beyond the wavelength range of the spectrum "
                        f"({wavelength[0]:.4f}-{wavelength[-1]:.4f}). The flux "
                        f"is set to NaN. Increasing the 'threshold' parameter "
                        f"({threshold}) will loosen the wavelength constraint."
                    )

                    syn_flux = np.nan

                else:
                    indices = np.isnan(transmission)
                    indices = np.logical_not(indices)

                    if self.det_type == "energy":
                        # Energy counting detector
                        integrand1 = transmission[indices] * flux[indices]
                        integrand2 = transmission[indices]

                    elif self.det_type == "photon":
                        # Photon counting detector
                        integrand1 = (
                            wavelength[indices] * transmission[indices] * flux[indices]
                        )
                        integrand2 = wavelength[indices] * transmission[indices]

                    integral1 = np.trapz(integrand1, x=wavelength[indices])
                    integral2 = np.trapz(integrand2, x=wavelength[indices])

                    syn_flux = integral1 / integral2

        if error is not None and not np.any(np.isnan(error)) and not np.isnan(syn_flux):
            phot_random = np.zeros(200)

            for i in range(200):
                # Use the original spectrum size (i.e. wavel_error and flux_error)
                spec_random = (
                    flux_error
                    + np.random.normal(loc=0.0, scale=1.0, size=wavel_error.shape[0])
                    * error
                )

                phot_random[i] = self.spectrum_to_flux(
                    wavel_error, spec_random, error=None, threshold=threshold
                )[0]

            nan_idx = np.isnan(phot_random)

            if np.sum(nan_idx) > 0:
                warnings.warn(
                    f"{np.sum(nan_idx)} out of 200 samples "
                    "that are used for estimating the "
                    "uncertainty on the synthetic flux "
                    "are NaN so removing these samples."
                )

                phot_random = phot_random[~nan_idx]

            error_flux = np.std(phot_random)

        elif error is not None and np.any(np.isnan(error)):
            warnings.warn("Spectum contains NaN so can not calculate the error.")
            error_flux = None

        else:
            error_flux = None

        return syn_flux, error_flux

    @typechecked
    def spectrum_to_magnitude(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        parallax: Optional[Tuple[float, Optional[float]]] = None,
        distance: Optional[Tuple[float, Optional[float]]] = None,
        threshold: Optional[float] = 0.05,
    ) -> Tuple[
        Tuple[float, Optional[float]], Optional[Tuple[Optional[float], Optional[float]]]
    ]:
        """
        Function for calculating the apparent and absolute magnitude
        from a spectrum and a filter profile. The uncertainty is
        propagated by sampling 200 random values from the error
        distributions.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength points (um).
        flux : np.ndarray
            Flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        error : np.ndarray, list(np.ndarray), None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        parallax : tuple(float, float), None
            Parallax and uncertainty (mas). No absolute magnitude is
            calculated if set to ``None``. No error on the absolute
            magnitude is calculated if the ``error`` parameter is
            set to ``None``.
        distance : tuple(float, float), None
            Distance and uncertainty (pc). No absolute magnitude is
            calculated if set to ``None``. No error on the absolute
            magnitude is calculated if the ``error`` parameter is
            set to ``None``. This parameter is ignored if the
            ``parallax`` parameter is used.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the
            minimum transmission value is larger than the threshold,
            a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter
            profile. The parameter is not used if set to ``None``.

        Returns
        -------
        tuple(float, float)
            Apparent magnitude and uncertainty.
        tuple(float, float)
            Absolute magnitude and uncertainty.
        """

        # Remove fluxes that are a NaN

        nan_idx = np.isnan(flux)

        if np.sum(nan_idx) > 0:
            warnings.warn(
                f"Found {np.sum(nan_idx)} fluxes with NaN. Removing "
                "these spectral fluxes from the input data before "
                "calculating synthetic photometry."
            )

            wavelength = wavelength[~nan_idx]
            flux = flux[~nan_idx]

            if error is not None:
                error = error[~nan_idx]

        if parallax is not None:
            distance = phot_util.parallax_to_distance(parallax)

        syn_flux = self.spectrum_to_flux(
            wavelength, flux, error=error, threshold=threshold
        )

        app_mag = self.vega_mag - 2.5 * math.log10(syn_flux[0] / self.zero_point)

        if error is not None and not np.any(np.isnan(error)):
            mag_random = np.zeros(200)

            for i in range(200):
                spec_random = (
                    flux
                    + np.random.normal(loc=0.0, scale=1.0, size=wavelength.shape[0])
                    * error
                )

                flux_random = self.spectrum_to_flux(
                    wavelength, spec_random, error=None, threshold=threshold
                )

                mag_random[i] = self.vega_mag - 2.5 * np.log10(
                    flux_random[0] / self.zero_point
                )

            nan_idx = np.isnan(mag_random)

            if np.sum(nan_idx) > 0:
                warnings.warn(
                    f"{np.sum(nan_idx)} out of 200 samples "
                    "that are used for estimating the "
                    "uncertainty on the synthetic magnitude "
                    "are NaN so removing these samples."
                )

                mag_random = mag_random[~nan_idx]

            error_app_mag = np.std(mag_random)

        elif error is not None and np.any(np.isnan(error)):
            warnings.warn("Spectum contains NaN so can not calculate the error.")
            error_app_mag = None

        else:
            error_app_mag = None

        if distance is None:
            abs_mag = None
            error_abs_mag = None

        else:
            abs_mag = app_mag - 5.0 * np.log10(distance[0]) + 5.0

            if error_app_mag is not None and distance[1] is not None:
                error_dist = distance[1] * (5.0 / (distance[0] * math.log(10.0)))
                error_abs_mag = math.sqrt(error_app_mag**2 + error_dist**2)

            else:
                error_abs_mag = None

        return (app_mag, error_app_mag), (abs_mag, error_abs_mag)

    @typechecked
    def magnitude_to_flux(
        self,
        magnitude: float,
        error: Optional[float] = None,
        zp_flux: Optional[float] = None,
    ) -> Tuple[
        Union[float, np.float32, np.float64],
        Optional[Union[float, np.float32, np.float64]],
    ]:
        """
        Function for converting a magnitude to a flux.

        Parameters
        ----------
        magnitude : float
            Magnitude.
        error : float, None
            Error on the magnitude. Not used if set to ``None``.
        zp_flux : float, None
            DEPRECATED: Zero-point flux (:math:`\\mathrm{W}`
            :math:`\\mathrm{m}^{-2}` :math:`\\mu\\mathrm{m}^{-1}`).
            This parameter is deprecated and will be removed in a
            future release. Please use the zero_point parameter
            of the constructor of 
            :class:`~species.analysis.photometry.SyntheticPhotometry`
            instead. By default, the zero point is calculated
            internally and stored as the ``zero_point`` attribute
            of an instance from
            :class:`~species.analysis.photometry.SyntheticPhotometry`.

        Returns
        -------
        float
            Flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        float, None
            Error (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`). The returned value is
            ``None`` if the argument of ``error`` is ``None``.
        """

        if zp_flux is None:
            flux = 10.0 ** (-0.4 * (magnitude - self.vega_mag)) * self.zero_point

        else:
            flux = 10.0 ** (-0.4 * (magnitude - self.vega_mag)) * zp_flux

            warnings.warn(
                "The 'zp_flux' parameter is deprecated "
                "and will be removed in a future release. "
                "Please use the 'zero_point' parameter "
                "of the SyntheticPhotometry constructor "
                "instead.",
                DeprecationWarning,
            )

        if error is None:
            error_flux = None

        else:
            error_upper = flux * (10.0 ** (0.4 * error) - 1.0)
            error_lower = flux * (1.0 - 10.0 ** (-0.4 * error))
            error_flux = (error_lower + error_upper) / 2.0

        return flux, error_flux

    @typechecked
    def flux_to_magnitude(
        self,
        flux: float,
        error: Optional[Union[float, np.ndarray]] = None,
        parallax: Optional[
            Union[
                Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
            ]
        ] = None,
        distance: Optional[
            Union[
                Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
            ]
        ] = None,
    ) -> Tuple[
        Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]],
        Union[
            Tuple[Optional[float], Optional[float]],
            Tuple[Optional[np.ndarray], Optional[np.ndarray]],
        ],
    ]:
        """
        Function for converting a flux into a magnitude.

        Parameters
        ----------
        flux : float, np.ndarray
            Flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        error : float, np.ndarray, None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`). Not used if set to None.
        parallax : tuple(float, float), , tuple(np.ndarray, np.ndarray), None
            Parallax and uncertainty (mas). The returned absolute
            magnitude is set to ``None`` in case ``parallax`` and
            ``distance`` are set to ``None``. The error is not
            propagated into the error on the absolute magnitude
            in case the parallax uncertainty is set to ``None``,
            for example ``parallax=(10., None)``.
        distance : tuple(float, float), tuple(np.ndarray, np.ndarray), None
            Distance and uncertainty (pc). The returned absolute
            magnitude is set to ``None`` in case ``distance`` and
            ``parallax`` are set to ``None``. The error is not
            propagated into the error on the absolute magnitude in
            case the distance uncertainty is set to ``None``, for
            example ``distance=(20., None)``. This parameter is
            ignored if the ``parallax`` parameter is used.

        Returns
        -------
        tuple(float, float), tuple(np.ndarray, np.ndarray)
            Apparent magnitude and uncertainty.
        tuple(float, float), tuple(np.ndarray, np.ndarray)
            Absolute magnitude and uncertainty.
        """

        if parallax is not None:
            distance = phot_util.parallax_to_distance(parallax)

        if flux <= 0.0:
            raise ValueError(
                "Converting a flux into a magnitude "
                "is only possible if the argument of "
                "'flux' has a positive value."
            )

        app_mag = self.vega_mag - 2.5 * np.log10(flux / self.zero_point)

        if error is None:
            error_app_mag = None
            error_abs_mag = None

        else:
            if flux + error > 0.0:
                error_app_lower = app_mag - (
                    self.vega_mag - 2.5 * np.log10((flux + error) / self.zero_point)
                )

            else:
                error_app_lower = np.nan

            if flux - error > 0.0:
                error_app_upper = (
                    self.vega_mag - 2.5 * np.log10((flux - error) / self.zero_point)
                ) - app_mag

            else:
                error_app_upper = np.nan

            error_app_mag = np.nanmean([error_app_lower, error_app_upper])

            if np.isnan(error_app_mag):
                error_app_mag = None

                warnings.warn(
                    "This warning should not have occurred "
                    "since either error_app_lower and/or "
                    "error_app_upper should not be NaN."
                )

        if distance is None:
            abs_mag = None
            error_abs_mag = None

        else:
            abs_mag, error_abs_mag = phot_util.apparent_to_absolute(
                (app_mag, error_app_mag), distance
            )

        return (app_mag, error_app_mag), (abs_mag, error_abs_mag)
