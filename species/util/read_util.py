"""
Utility functions for reading data.
"""

import math
import warnings

from typing import Tuple, Union

import numpy as np

from typeguard import typechecked
from scipy.integrate import simps
from scipy.ndimage.filters import gaussian_filter

from species.core import constants
from species.read import read_model, read_planck


def get_mass(model_param):
    """
    Parameters
    ----------
    model_param : dict
        Model parameter values. Should contain the surface gravity and radius.

    Returns
    -------
    float
        Mass (Mjup).
    """

    logg = 1e-2 * 10.**model_param['logg']  # (m s-1)

    radius = model_param['radius']  # (Rjup)
    radius *= constants.R_JUP  # (m)

    mass = logg*radius**2/constants.GRAVITY  # (kg)
    mass /= constants.M_JUP  # (Mjup)

    return mass


def add_luminosity(modelbox):
    """
    Function to add the luminosity of a model spectrum to the parameter dictionary of the box.

    Parameters
    ----------
    modelbox : species.core.box.ModelBox
        Box with the model spectrum. Should also contain the dictionary with the model parameters,
        the radius in particular.

    Returns
    -------
    species.core.box.ModelBox
        The input box with the luminosity added in the parameter dictionary.
    """

    print('Calculating the luminosity...', end='', flush=True)

    if modelbox.model == 'planck':
        readmodel = read_planck.ReadPlanck(wavel_range=(1e-1, 1e3))
        fullspec = readmodel.get_spectrum(model_param=modelbox.parameters, spec_res=1000.)

    else:
        readmodel = read_model.ReadModel(modelbox.model)
        fullspec = readmodel.get_model(modelbox.parameters)

    flux = simps(fullspec.flux, fullspec.wavelength)

    if 'distance' in modelbox.parameters:
        luminosity = 4.*np.pi*(fullspec.parameters['distance']*constants.PARSEC)**2*flux  # (W)

        # Analytical solution for a single-component Planck function
        # luminosity = 4.*np.pi*(modelbox.parameters['radius']*constants.R_JUP)**2* \
        #     constants.SIGMA_SB*modelbox.parameters['teff']**4.

    else:
        luminosity = 4.*np.pi*(fullspec.parameters['radius']*constants.R_JUP)**2*flux  # (W)

    modelbox.parameters['luminosity'] = luminosity/constants.L_SUN  # (Lsun)

    print(' [DONE]')

    print(f'Wavelength range (um): {fullspec.wavelength[0]:.2e} - '
          f'{fullspec.wavelength[-1]:.2e}')

    print(f'Luminosity (Lsun): {luminosity/constants.L_SUN:.2e}')

    return modelbox


def update_spectra(objectbox,
                   model_param):
    """
    Function for applying a best-fit scaling and/or error inflation to the spectra of an object.

    Parameters
    ----------
    objectbox : species.core.box.ObjectBox
        Box with the object's data, including the spectra.
    model_param : dict
        Model parameter values. Should contain the scaling and/or error inflation values.

    Returns
    -------
    species.core.box.ObjectBox
        The input box with the scaled and/or error inflated spectra.
    """

    for key, value in objectbox.spectrum.items():
        spec_tmp = value[0]

        if f'scaling_{key}' in model_param:
            scaling = model_param[f'scaling_{key}']

            print(f'Scaling the flux of {key}: {scaling:.2f}...', end='', flush=True)
            spec_tmp[:, 1] *= model_param[f'scaling_{key}']
            # spec_tmp[:, 2] *= model_param[f'scaling_{key}']
            print(' [DONE]')

        if f'error_{key}' in model_param:
            error = 10.**model_param[f'error_{key}']

            print(f'Inflating the error of {key} (W m-2 um-1): {error:.2e}...', end='', flush=True)
            spec_tmp[:, 2] += error
            print(' [DONE]')

        objectbox.spectrum[key] = (spec_tmp, value[1], value[2], value[3])

    return objectbox


@typechecked
def create_wavelengths(wavel_range: Tuple[Union[float, np.float32], Union[float, np.float32]],
                       spec_res: float) -> np.ndarray:
    """
    Function for creating logarithmically-spaced wavelengths at a constant spectral resolution.

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Wavelength range (um). Tuple with the minimum and maximum wavelength.
    spec_res : float
        Spectral resolution at which the wavelengths are sampled.

    Returns
    -------
    np.ndarray
        Array with the wavelength points and a fixed spectral resolution. Since the wavelength
        boundaries are fixed, the output spectral resolution is slightly different from the
        ``spec_res`` value.
    """

    n_test = 100

    wavel_test = np.logspace(np.log10(wavel_range[0]), np.log10(wavel_range[1]), n_test)

    res_test = 0.5*(wavel_test[1:]+wavel_test[:-1])/np.diff(wavel_test)

    # R = lambda / delta_lambda / 2, because twice as many points as R are required to resolve
    # two features that are lambda / R apart

    wavelength = np.logspace(np.log10(wavel_range[0]),
                             np.log10(wavel_range[1]),
                             math.ceil(2.*n_test*spec_res/np.mean(res_test))+1)

    res_out = np.mean(0.5*(wavelength[1:]+wavelength[:-1])/np.diff(wavelength)/2.)

    return wavelength


@typechecked
def smooth_spectrum(wavelength: np.ndarray,
                    flux: np.ndarray,
                    spec_res: float,
                    size: int = 11) -> np.ndarray:
    """
    Function for smoothing a spectrum with a Gaussian kernel to a fixed spectral resolution. The
    kernel size is set to 5 times the FWHM of the Gaussian. The FWHM of the Gaussian is equal
    to the ratio of the wavelength and the spectral resolution. If the kernel does not fit within
    the available wavelength grid (i.e. at the edge of the array) then the flux values are set
    to NaN.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength points (um). Should be sampled with a uniform spectral resolution or a uniform
        wavelength spacing (slow).
    flux : np.ndarray
        Flux (W m-2 um-1).
    spec_res : float
        Spectral resolution.
    size : int
        Kernel size (odd integer).

    Returns
    -------
    np.ndarray
        Smoothed spectrum (W m-2 um-1).
    """

    def _gaussian(size, sigma):
        pos = range(-(size-1)//2, (size-1)//2+1)
        kernel = [np.exp(-float(x)**2/(2.*sigma**2))/(sigma*np.sqrt(2.*np.pi)) for x in pos]

        return np.asarray(kernel/sum(kernel))

    spacing = np.mean(2.*np.diff(wavelength)/(wavelength[1:]+wavelength[:-1]))
    spacing_std = np.std(2.*np.diff(wavelength)/(wavelength[1:]+wavelength[:-1]))

    if spacing_std/spacing < 1e-2:
        # see retrieval_util.convolve
        sigma_lsf = 1./spec_res/(2.*np.sqrt(2.*np.log(2.)))
        flux_smooth = gaussian_filter(flux, sigma=sigma_lsf/spacing, mode='nearest')

    else:
        if size % 2 == 0:
            raise ValueError('The kernel size should be an odd number.')

        flux_smooth = np.zeros(flux.shape)  # (W m-2 um-1)

        spacing = np.mean(np.diff(wavelength))  # (um)
        spacing_std = np.std(np.diff(wavelength))  # (um)

        if spacing_std/spacing > 1e-2:
            warnings.warn(f'The wavelength spacing is not uniform ({spacing} +/- {spacing_std}). '
                          f'The smoothing with the Gaussian kernel requires either the spectral '
                          f'resolution or the wavelength spacing to be uniformly sampled.')

        for i, item in enumerate(wavelength):
            fwhm = item/spec_res  # (um)
            sigma = fwhm/(2.*np.sqrt(2.*np.log(2.)))  # (um)

            size = int(5.*sigma/spacing)  # Kernel size 5 times the width of the LSF
            if size % 2 == 0:
                size += 1

            gaussian = _gaussian(size, sigma/spacing)

            try:
                flux_smooth[i] = np.sum(gaussian * flux[i-(size-1)//2:i+(size-1)//2+1])

            except ValueError:
                flux_smooth[i] = np.nan

    return flux_smooth
