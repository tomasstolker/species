"""
Module with reading functionalities for Planck spectra.
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
    Class for reading a Planck spectrum.
    """

    def __init__(self,
                 filter_name):
        """
        Parameters
        ----------
        filter_name : str, None
            Filter ID that is used for the wavelength range. Full spectrum is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        if isinstance(filter_name, str):
            self.filter_name = filter_name
            transmission = read_filter.ReadFilter(filter_name)
            self.wavel_range = transmission.wavelength_range()

        else:
            self.filter_name = None
            self.wavel_range = filter_name

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    @staticmethod
    def planck(wavel_points,
               temperature,
               scaling):
        """
        Internal function for calculating a Planck function.

        Parameters
        ----------
        wavel_points : numpy.ndarray
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

        planck_1 = 2.*constants.PLANCK*constants.LIGHT**2/(1e-6*wavel_points)**5

        planck_2 = np.exp(constants.PLANCK*constants.LIGHT /
                          (1e-6*wavel_points*constants.BOLTZMANN*temperature)) - 1.

        return 1e-6 * 4.*math.pi * scaling * planck_1/planck_2  # [W m-2 micron-1]

    def get_spectrum(self,
                     model_param,
                     spec_res):
        """
        Function for calculating a Planck spectrum.

        Parameters
        ----------
        model_param : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc).
        spec_res : float
            Spectral resolution.

        Returns
        -------
        species.core.box.ModelBox
            Box with the Planck spectrum.
        """

        wavel_points = [self.wavel_range[0]]

        while wavel_points[-1] <= self.wavel_range[1]:
            wavel_points.append(wavel_points[-1] + wavel_points[-1]/spec_res)

        wavel_points = np.asarray(wavel_points)  # [micron]

        scaling = ((model_param['radius']*constants.R_JUP) /
                   (model_param['distance']*constants.PARSEC))**2

        flux = self.planck(np.copy(wavel_points), model_param['teff'], scaling)  # [W m-2 micron-1]

        return box.create_box(boxtype='model',
                              model='planck',
                              wavelength=wavel_points,
                              flux=flux,
                              parameters=model_param,
                              quantity='flux')

    def get_flux(self,
                 model_param,
                 synphot=None):
        """
        Function for calculating the average flux density for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc).
        synphot : species.analysis.photometry.SyntheticPhotometry, None
            Synthetic photometry object. The object is created if set to None.

        Returns
        -------
        float
            Average flux density (W m-2 micron-1).
        """

        spectrum = self.get_spectrum(model_param, 100.)

        if synphot is None:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(spectrum.wavelength, spectrum.flux)
