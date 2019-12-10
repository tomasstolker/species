"""
Text
"""

import os
import math
import configparser

import numpy as np

from species.analysis import photometry
from species.core import box, constants
from species.read import read_filter


class ReadPlanck:
    """
    Read a Planck spectrum.
    """

    def __init__(self,
                 wavelength):
        """
        Parameters
        ----------
        wavelength : tuple(float, float) or str
            Wavelength range (micron) or filter name. Full spectrum is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        if isinstance(wavelength, str):
            self.filter_name = wavelength
            transmission = read_filter.ReadFilter(wavelength)
            self.wavelength = transmission.wavelength_range()

        else:
            self.filter_name = None
            self.wavelength = wavelength

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @staticmethod
    def planck(wl_points,
               temperature,
               scaling):
        """
        Parameters
        ----------
        wl_points : numpy.ndarray
            Wavelength points (micron).
        temperature : float
            Temperature (K).
        scaling : float
            Scaling parameter.

        Returns
        -------
        numpy.ndarray
            Flux density (W m-2 micron-1).
        """

        planck1 = 2.*constants.PLANCK*constants.LIGHT**2/(1e-6*wl_points)**5
        planck2 = np.exp(constants.PLANCK*constants.LIGHT /
                         (1e-6*wl_points*constants.BOLTZMANN*temperature)) - 1.

        flux = 4.*math.pi * scaling * planck1/planck2  # [W m-2 m-1]
        flux *= 1e-6  # [W m-2 micron-1]

        return flux

    def get_spectrum(self,
                     model_par,
                     specres):
        """
        Parameters
        ----------
        model_par : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc).
        specres : float
            Spectral resolution.

        Returns
        -------
        species.core.box.ModelBox
            Box with the Planck spectrum.
        """

        wl_points = [self.wavelength[0]]
        while wl_points[-1] <= self.wavelength[1]:
            wl_points.append(wl_points[-1] + wl_points[-1]/specres)

        wl_points = np.asarray(wl_points)  # [micron]

        scaling = (model_par['radius']*constants.R_JUP/(model_par['distance']*constants.PARSEC))**2
        flux = self.planck(np.copy(wl_points), model_par['teff'], scaling)  # [W m-2 micron-1]

        return box.create_box(boxtype='model',
                              model='planck',
                              wavelength=wl_points,
                              flux=flux,
                              parameters=model_par,
                              quantity='flux')

    def get_photometry(self,
                       model_par,
                       synphot=None):
        """
        Parameters
        ----------
        model_par : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc).
        synphot : species.analysis.photometry.SyntheticPhotometry
            Synthetic photometry object.

        Returns
        -------
        float
            Average flux density (W m-2 micron-1).
        """

        spectrum = self.get_spectrum(model_par, 100.)

        if not synphot:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(spectrum.wavelength, spectrum.flux)
