"""
Utility functions for manipulating spectra.
"""

import warnings

from math import ceil
from typing import Tuple, Union

import numpy as np

from scipy.ndimage import gaussian_filter
from typeguard import typechecked


@typechecked
def create_wavelengths(
    wavel_range: Tuple[Union[float, np.float32], Union[float, np.float32]],
    wavel_sampling: float,
) -> np.ndarray:
    """
    Function for creating logarithmically-spaced wavelengths,
    so with a constant :math:`\\lambda/\\Delta\\lambda`.

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Wavelength range (:math:`\\mu\\mathrm{m}`). Tuple with the
        minimum and maximum wavelength.
    wavel_sampling : float
        Wavelength sampling :math:`\\lambda/\\Delta\\lambda`.

    Returns
    -------
    np.ndarray
        Array with the wavelengths (:math:`\\mu\\mathrm{m}`). Since
        the wavelength boundaries are fixed, the output sampling
        is slightly different from the value provided as
        argument of ``wavel_sampling``.
    """

    n_test = 100

    wavel_test = np.logspace(np.log10(wavel_range[0]), np.log10(wavel_range[1]), n_test)
    sampling_test = 0.5 * (wavel_test[1:] + wavel_test[:-1]) / np.diff(wavel_test)

    # math.ceil returns int, but np.ceil returns float
    wavel_array = np.logspace(
        np.log10(wavel_range[0]),
        np.log10(wavel_range[1]),
        ceil(n_test * wavel_sampling / np.mean(sampling_test)) + 1,
    )

    # res_out = np.mean(0.5*(wavel_array[1:]+wavel_array[:-1])/np.diff(wavel_array))

    return wavel_array


@typechecked
def smooth_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    spec_res: float,
    size: int = 11,
    force_smooth: bool = False,
) -> np.ndarray:
    """
    Function for smoothing a spectrum with a Gaussian kernel to a
    fixed spectral resolution. The kernel size is set to 5 times the
    FWHM of the Gaussian. The FWHM of the Gaussian is equal to the
    ratio of the wavelength and the spectral resolution. If the
    kernel does not fit within the available wavelength grid (i.e.
    at the edge of the array) then the flux values are set to NaN.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength points (um). Should be sampled with a uniform
        spectral resolution or a uniform wavelength spacing (slow).
    flux : np.ndarray
        Flux (W m-2 um-1).
    spec_res : float
        Spectral resolution.
    size : int
        Kernel size (odd integer).
    force_smooth : bool
        Force smoothing for constant spectral resolution

    Returns
    -------
    np.ndarray
        Smoothed spectrum (W m-2 um-1).
    """

    def _gaussian(size, sigma):
        pos = range(-(size - 1) // 2, (size - 1) // 2 + 1)
        kernel = [
            np.exp(-float(x) ** 2 / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))
            for x in pos
        ]

        return np.asarray(kernel / sum(kernel))

    spacing = np.mean(2.0 * np.diff(wavelength) / (wavelength[1:] + wavelength[:-1]))
    spacing_std = np.std(2.0 * np.diff(wavelength) / (wavelength[1:] + wavelength[:-1]))

    if spacing_std / spacing < 1e-2 or force_smooth:
        # see retrieval_util.convolve_spectrum
        sigma_lsf = 1.0 / spec_res / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        flux_smooth = gaussian_filter(flux, sigma=sigma_lsf / spacing, mode="nearest")

    else:
        if size % 2 == 0:
            raise ValueError("The kernel size should be an odd number.")

        flux_smooth = np.zeros(flux.shape)  # (W m-2 um-1)

        spacing = np.mean(np.diff(wavelength))  # (um)
        spacing_std = np.std(np.diff(wavelength))  # (um)

        if spacing_std / spacing > 1e-2:
            warnings.warn(
                f"The wavelength spacing is not uniform ({spacing} +/- {spacing_std}). "
                f"The smoothing with the Gaussian kernel requires either the spectral "
                f"resolution or the wavelength spacing to be uniformly sampled."
            )

        for i, item in enumerate(wavelength):
            fwhm = item / spec_res  # (um)
            sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # (um)

            size = int(
                5.0 * sigma / spacing
            )  # Kernel size 5 times the width of the LSF
            if size % 2 == 0:
                size += 1

            gaussian = _gaussian(size, sigma / spacing)

            try:
                flux_smooth[i] = np.sum(
                    gaussian * flux[i - (size - 1) // 2 : i + (size - 1) // 2 + 1]
                )

            except ValueError:
                flux_smooth[i] = np.nan

    return flux_smooth
