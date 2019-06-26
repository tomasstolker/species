"""
Utility functions for reading data.
"""

import math

import numpy as np

from scipy.integrate import simps

from species.core import constants
from species.read import read_model


def get_mass(model_par):
    """
    Parameters
    ----------
    model_par : dict
        Model parameter values. Should contain the surface gravity and radius.

    Returns
    -------
    float
        Mass (Mjup).
    """

    logg = 1e-2 * 10.**model_par['logg']  # [m s-1]

    radius = model_par['radius']  # [Rjup]
    radius *= constants.R_JUP  # [m]

    mass = logg*radius**2/constants.GRAVITY  # [kg]
    mass /= constants.M_JUP  # [Mjup]

    return mass


def add_luminosity(modelbox):
    """
    Function to add the luminosity of a model spectrum to the parameter dictionary of the box. The
    luminosity is by default calculated at a spectral resolution of 1000.

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

    readmodel = read_model.ReadModel(model=modelbox.model, wavelength=None, teff=None)
    fullspec = readmodel.get_model(model_par=modelbox.parameters)

    flux = simps(fullspec.flux, fullspec.wavelength)

    if 'distance' in modelbox.parameters:
        luminosity = 4.*math.pi*(fullspec.parameters['distance']*constants.PARSEC)**2*flux  # [W]
    else:
        luminosity = 4.*math.pi*(fullspec.parameters['radius']*constants.R_JUP)**2*flux  # [W]

    modelbox.parameters['luminosity'] = luminosity/constants.L_SUN  # [Lsun]

    return modelbox


def smooth_spectrum(wavelength,
                    flux,
                    specres,
                    size=11):
    """
    Function to convolve a spectrum with a Gaussian kernel to a fixed spectral resolution. The
    kernel size is set to 5 times the FWHM of the Gaussian. The FWHM of the Gaussian is equal
    to the the wavelength divided by the spectral resolution. If the kernel does not fit within
    the available wavelength grid (i.e., at the edge of the array) then the flux values are set
    to NaN.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Wavelength points (micron). Should be equally-spaced.
    flux : numpy.ndarray
        Flux density (W m-2 micron-1).
    specres : float
        Spectral resolution

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum (W m-2 micron-1) at the same wavelength points as the
    """

    def _gaussian(size, sigma):
        pos = range(-(size-1)//2, (size-1)//2+1)
        kernel = [math.exp(-float(x)**2/(2.*sigma**2))/(sigma*math.sqrt(2.*math.pi)) for x in pos]
        return np.asarray(kernel/sum(kernel))

    if size % 2 == 0:
        raise ValueError('The kernel size should be an odd number.')

    spacing = np.mean(np.diff(wavelength))  # [micron]
    flux_smooth = np.zeros(flux.shape)  # [W m-2 micron-1]

    for i, item in enumerate(wavelength):
        fwhm = item/specres  # [micron]
        sigma = fwhm/(2.*math.sqrt(2.*math.log(2.)))  # [micron]

        size = int(5.*sigma/spacing)  # Kernel size 5 times the FWHM
        if size % 2 == 0:
            size += 1

        gaussian = _gaussian(size, sigma/spacing)

        try:
            flux_smooth[i] = np.sum(gaussian * flux[i-(size-1)//2:i+(size-1)//2+1])

        except ValueError:
            flux_smooth[i] = np.nan

    return flux_smooth
