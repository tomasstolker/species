"""
Module with reading functionalities for calibration spectra.
"""

import configparser
import os

from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import spectres

from scipy import interpolate, optimize

from typeguard import typechecked

from species.core.box import SpectrumBox, create_box
from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_filter import ReadFilter
from species.util.spec_util import create_wavelengths, smooth_spectrum


class ReadCalibration:
    """
    Class for reading a calibration spectrum from the database.
    """

    @typechecked
    def __init__(self, tag: str, filter_name: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag of the calibration spectrum.
        filter_name : str, None
            Filter that is used for the wavelength range. The full
            spectrum is read if the argument is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag
        self.filter_name = filter_name

        if filter_name is None:
            self.wavel_range = None

        else:
            transmission = ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

    @typechecked
    def resample_spectrum(
        self,
        wavel_points: np.ndarray,
        model_param: Optional[Dict[str, float]] = None,
        spec_res: Optional[float] = None,
        apply_mask: bool = False,
        interp_highres: bool = False,
    ) -> SpectrumBox:
        """
        Function for resampling the spectrum and optional
        uncertainties onto a new wavelength grid.

        Parameters
        ----------
        wavel_points : np.ndarray
            Wavelengths (um).
        model_param : dict, None
            Dictionary with the model parameters, which should only
            contain the ``'scaling'`` keyword. No scaling is applied if
            the argument of ``model_param`` is set to ``None``.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum
            before resampling the wavelengths. No smoothing is applied
            if the argument is set to ``None``. The smoothing can only
            be applied to spectra with a constant spectral resolution
            (which is the case for all model spectra that are
            supported by ``species``) or a constant wavelength
            spacing. The first smoothing approach is fastest.
        apply_mask : bool
            Exclude negative values and NaNs.
        interp_highres : bool
            Oversample the spectrum to $R = 10000$, such that the
            ``spec_res`` parameter can be applied on a spectrum with
            constant $\\lambda/\\Delta\\lambda$. The uncertainties
            are crudely propagated with an interpolation as well
            and should only be considered as estimate.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the resampled spectrum.
        """

        calib_box = self.get_spectrum(apply_mask=apply_mask)

        if interp_highres:
            flux_interp = interpolate.interp1d(
                calib_box.wavelength,
                calib_box.flux,
                bounds_error=False,
                fill_value="extrapolate",
            )

            sigma_interp = interpolate.interp1d(
                calib_box.wavelength,
                calib_box.error,
                bounds_error=False,
                fill_value="extrapolate",
            )

            wavel_range = (calib_box.wavelength[0], calib_box.wavelength[-1])
            wavel_highres = create_wavelengths(wavel_range, 10000.0)

            calib_box.wavelength = wavel_highres
            calib_box.flux = flux_interp(wavel_highres)
            calib_box.error = sigma_interp(wavel_highres)

            if spec_res is not None:
                calib_box.flux = smooth_spectrum(
                    wavelength=calib_box.wavelength,
                    flux=calib_box.flux,
                    spec_res=spec_res,
                )

            flux_interp = interpolate.interp1d(
                calib_box.wavelength,
                calib_box.flux,
                bounds_error=False,
                fill_value="extrapolate",
            )

            sigma_interp = interpolate.interp1d(
                calib_box.wavelength,
                calib_box.error,
                bounds_error=False,
                fill_value="extrapolate",
            )

            flux_new = flux_interp(wavel_points)
            error_new = sigma_interp(wavel_points)

        else:
            if spec_res is not None:
                calib_box.flux = smooth_spectrum(
                    wavelength=calib_box.wavelength,
                    flux=calib_box.flux,
                    spec_res=spec_res,
                )

            flux_new, error_new = spectres.spectres(
                wavel_points,
                calib_box.wavelength,
                calib_box.flux,
                spec_errs=calib_box.error,
                fill=0.0,
                verbose=False,
            )

        if model_param is not None:
            flux_new = model_param["scaling"] * flux_new
            error_new = model_param["scaling"] * error_new

        return create_box(
            boxtype="spectrum",
            spectrum="calibration",
            wavelength=wavel_points,
            flux=flux_new,
            error=error_new,
            name=self.tag,
        )

    @typechecked
    def get_spectrum(
        self,
        model_param: Optional[Dict[str, float]] = None,
        apply_mask: bool = False,
        wavel_sampling: Optional[float] = None,
        extrapolate: bool = False,
        min_wavelength: Optional[float] = None,
    ) -> SpectrumBox:
        """
        Function for selecting the calibration spectrum.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not
            used if set to ``None``.
        apply_mask : bool
            Exclude negative values and NaN values.
        wavel_sampling : float, None
            Wavelength sampling :math:`\\lambda/\\Delta\\lambda`. The
            original wavelength points are used if set to ``None``.
        extrapolate : bool
            Extrapolate to 6 um by fitting a power law function.
        min_wavelength : float, None
            Minimum wavelength used for fitting the power law function.
            All data is used if set to ``None``.

        Returns
        -------
        species.core.box.SpectrumBox
            Box with the spectrum.
        """

        with h5py.File(self.database, "r") as h5_file:
            cal_spec = np.array(h5_file[f"spectra/calibration/{self.tag}"])

            wavelength = cal_spec[0,]
            flux = cal_spec[1,]
            error = cal_spec[2,]

        if apply_mask:
            indices = np.where(flux > 0.0)[0]

            wavelength = wavelength[indices]
            flux = flux[indices]
            error = error[indices]

        if model_param is not None:
            flux = model_param["scaling"] * flux
            error = model_param["scaling"] * error

        if self.wavel_range is None:
            wl_index = np.ones(wavelength.size, dtype=bool)
        else:
            wl_index = (
                (flux > 0.0)
                & (wavelength > self.wavel_range[0])
                & (wavelength < self.wavel_range[1])
            )

        count = np.count_nonzero(wl_index)

        if count > 0:
            index = np.where(wl_index)[0]

            if index[0] > 0:
                wl_index[index[0] - 1] = True

            if index[-1] < len(wl_index) - 1:
                wl_index[index[-1] + 1] = True

            wavelength = wavelength[wl_index]
            flux = flux[wl_index]
            error = error[wl_index]

        if extrapolate:

            def _power_law(wavelength, offset, scaling, power_index):
                return offset + scaling * wavelength**power_index

            if min_wavelength:
                indices = np.where(wavelength > min_wavelength)[0]
            else:
                indices = np.arange(0, wavelength.size, 1)

            popt, pcov = optimize.curve_fit(
                f=_power_law,
                xdata=wavelength[indices],
                ydata=flux[indices],
                p0=(0.0, np.mean(flux[indices]), -1.0),
                sigma=error[indices],
            )

            sigma = np.sqrt(np.diag(pcov))

            print("Fit result for f(x) = a + b*x^c:")
            print(f"a = {popt[0]} +/- {sigma[0]}")
            print(f"b = {popt[1]} +/- {sigma[1]}")
            print(f"c = {popt[2]} +/- {sigma[2]}")

            while wavelength[-1] <= 6.0:
                wl_add = wavelength[-1] + wavelength[-1] / 1000.0

                wavelength = np.append(wavelength, wl_add)
                flux = np.append(flux, _power_law(wl_add, popt[0], popt[1], popt[2]))
                error = np.append(error, 0.0)

        if wavel_sampling is not None:
            wavelength_new = create_wavelengths(
                (wavelength[0], wavelength[-1]), wavel_sampling
            )

            flux_new, error_new = spectres.spectres(
                wavelength_new,
                wavelength,
                flux,
                spec_errs=error,
                fill=0.0,
                verbose=True,
            )

            wavelength = wavelength_new
            flux = flux_new
            error = error_new

        return create_box(
            boxtype="spectrum",
            spectrum="calibration",
            wavelength=wavelength,
            flux=flux,
            error=error,
            name=self.tag,
        )

    @typechecked
    def get_flux(
        self, model_param: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Function for calculating the average flux for the
        ``filter_name``.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not
            used if set to ``None``.

        Returns
        -------
        Returns
        -------
        float
            Average flux (W m-2 um-1).
        float
            Uncertainty (W m-2 um-1).
        """

        specbox = self.get_spectrum(model_param=model_param, apply_mask=True)

        synphot = SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(
            specbox.wavelength, specbox.flux, error=specbox.flux
        )

    @typechecked
    def get_magnitude(
        self,
        model_param: Optional[Dict[str, float]] = None,
        distance: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, Optional[float]], Tuple[Optional[float], Optional[float]]]:
        """
        Function for calculating the apparent magnitude for the
        ``filter_name``.

        Parameters
        ----------
        model_param : dict, None
            Model parameters. Should contain the 'scaling' value. Not
            used if set to ``None``.
        distance : tuple(float, float), None
            Distance and uncertainty to the calibration object (pc).
            Not used if set to ``None``, in which case the returned
            absolute magnitude is ``(None, None)``.

        Returns
        -------
        tuple(float, float)
            Apparent magnitude and uncertainty.
        tuple(float, float), tuple(None, None)
            Absolute magnitude and uncertainty.
        """

        specbox = self.get_spectrum(model_param=model_param, apply_mask=True)

        if np.count_nonzero(specbox.error) == 0:
            error = None
        else:
            error = specbox.error

        synphot = SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_magnitude(
            specbox.wavelength, specbox.flux, error=error, distance=distance
        )
