"""
Text
"""

import math

import numpy as np

from species.core import box
from species.core import constants as con


def planck(wl_points,
           temperature,
           scaling):
    """
    Parameters
    ----------
    temperature : float
        Temperature (K).
    scaling : float
        Scaling parameter.
    wl_points : numpy.ndarray
        Wavelength points (micron).

    Returns
    -------
    numpy.ndarray
        Flux density (W m-2 micron-1).
    """

    planck1 = 2.*con.PLANCK*con.LIGHT**2/(1e-6*wl_points)**5
    planck2 = np.exp(con.PLANCK*con.LIGHT/(1e-6*wl_points*con.BOLTZMANN*temperature)) - 1.

    flux = 4.*math.pi * scaling * planck1/planck2  # [W m-2 m-1]
    flux *= 1e-6  # [W m-2 micron-1]

    return flux


def get_planck(temperature,
               radius,
               distance,
               wavelength,
               specres):
    """
    Parameters
    ----------
    temperature : float
        Temperature (K).
    radius : float
        Radius (Rjup).
    distance : float
        Distance (pc).
    wavelength : tuple(float, float)
        Wavelength range (micron).
    specres : float
        Spectral resolution

    Returns
    -------
    species.core.box.SpectrumBox
        Box with the Planck spectrum.
    """

    wl_points = [wavelength[0]]
    while wl_points[-1] <= wavelength[1]:
        wl_points.append(wl_points[-1] + wl_points[-1]/specres)

    wl_points = np.asarray(wl_points)  # [micron]

    scaling = (radius*con.R_JUP/(distance*con.PARSEC))**2
    flux = planck(np.copy(wl_points), temperature, scaling)  # [W m-2 micron-1]

    return box.create_box(boxtype='spectrum',
                          spectrum='planck',
                          wavelength=wl_points,
                          flux=flux,
                          error=None,
                          name=None,
                          simbad=None,
                          sptype=None,
                          distance=None)
