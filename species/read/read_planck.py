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
                 wavel_range=None,
                 filter_name=None):
        """
        Parameters
        ----------
        wavel_range : tuple(float, float), None
            Wavelength range (um). A wavelength range of 0.1-1000 um is used if set to
            None. Not used if ``filter_name`` is not None.
        filter_name : str, None
            Filter name that is used for the wavelength range. The ``wavel_range`` is used if set
            to None.

        Returns
        -------
        NoneType
            None
        """

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        self.filter_name = filter_name
        self.wavel_range = wavel_range

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        elif self.wavel_range is None:
            self.wavel_range = (0.1, 1000.)

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
            Wavelength points (um).
        temperature : float
            Temperature (K).
        scaling : float
            Scaling parameter.

        Returns
        -------
        numpy.ndarray
            Flux density (W m-2 um-1).
        """

        planck_1 = 2.*constants.PLANCK*constants.LIGHT**2/(1e-6*wavel_points)**5

        planck_2 = np.exp(constants.PLANCK*constants.LIGHT /
                          (1e-6*wavel_points*constants.BOLTZMANN*temperature)) - 1.

        return 1e-6 * math.pi * scaling * planck_1/planck_2  # (W m-2 um-1)

    @staticmethod
    def update_parameters(model_param):
        """
        Internal function for updating the dictionary with model parameters.

        Parameters
        ----------
        model_param : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc). The values
            of 'teff' and 'radius' can be a single float, or a list with floats for a combination
            of multiple Planck functions, e.g.
            {'teff': [1500., 1000.], 'radius': [1., 2.], 'distance': 10.}.

        Returns
        -------
        dict
            Updated dictionary with model parameters.
        """

        updated_param = {}

        for i, _ in enumerate(model_param['teff']):
            updated_param[f'teff_{i}'] = model_param['teff'][i]
            updated_param[f'radius_{i}'] = model_param['radius'][i]

        updated_param['distance'] = model_param['distance']

        return updated_param

    def get_spectrum(self,
                     model_param,
                     spec_res):
        """
        Function for calculating a Planck spectrum or a combination of multiple Planck spectra.

        Parameters
        ----------
        model_param : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc). The values
            of 'teff' and 'radius' can be a single float, or a list with floats for a combination
            of multiple Planck functions, e.g.
            {'teff': [1500., 1000.], 'radius': [1., 2.], 'distance': 10.}.
        spec_res : float
            Spectral resolution.

        Returns
        -------
        species.core.box.ModelBox
            Box with the Planck spectrum.
        """

        if 'teff' in model_param and isinstance(model_param['teff'], list):
            model_param = self.update_parameters(model_param)

        wavel_points = [self.wavel_range[0]]

        while wavel_points[-1] <= self.wavel_range[1]:
            wavel_points.append(wavel_points[-1] + wavel_points[-1]/spec_res)

        wavel_points = np.asarray(wavel_points)  # (um)

        n_planck = (len(model_param)-1) // 2

        if n_planck == 1:
            scaling = ((model_param['radius']*constants.R_JUP) /
                       (model_param['distance']*constants.PARSEC))**2

            flux = self.planck(np.copy(wavel_points),
                               model_param['teff'],
                               scaling)  # (W m-2 um-1)

        else:
            flux = np.zeros(wavel_points.shape)

            for i in range(n_planck):
                scaling = ((model_param[f'radius_{i}']*constants.R_JUP) /
                           (model_param['distance']*constants.PARSEC))**2

                flux += self.planck(np.copy(wavel_points),
                                    model_param[f'teff_{i}'],
                                    scaling)  # (W m-2 um-1)

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
            Average flux density (W m-2 um-1).
        """

        if 'teff' in model_param and isinstance(model_param['teff'], list):
            model_param = self.update_parameters(model_param)

        spectrum = self.get_spectrum(model_param, 100.)

        if synphot is None:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)

    def get_magnitude(self,
                      model_param,
                      synphot=None):
        """
        Function for calculating the magnitude for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the 'teff' (K), 'radius' (Rjup), and 'distance' (pc).
        synphot : species.analysis.photometry.SyntheticPhotometry, None
            Synthetic photometry object. The object is created if set to None.

        Returns
        -------
        float
            Apparent magnitude (mag).
        float
            Absolute magnitude (mag)
        """

        if 'teff' in model_param and isinstance(model_param['teff'], list):
            model_param = self.update_parameters(model_param)

        spectrum = self.get_spectrum(model_param, 100.)

        if synphot is None:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_magnitude(spectrum.wavelength,
                                             spectrum.flux,
                                             distance=(model_param['distance'], None))

    def get_color_magnitude(self,
                            temperatures,
                            radius,
                            filters_color,
                            filter_mag):
        """
        Function for calculating the colors and magnitudes in the range of 100-10000 K.

        Parameters
        ----------
        temperatures : numpy.ndarray
            Temperatures (K) for which the colors and magnitude are calculated.
        radius : float
            Radius t
        filters_color : tuple(str, str)
            Filter names for the color.
        filter_mag : str
            Filter name for the absolute magnitudes.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
        """

        list_color = []
        list_mag = []

        for item in temperatures:
            model_param = {'teff': item, 'radius': radius, 'distance': 10.}

            read_planck_0 = ReadPlanck(filter_name=filters_color[0])
            read_planck_1 = ReadPlanck(filter_name=filters_color[1])
            read_planck_2 = ReadPlanck(filter_name=filter_mag)

            app_mag_0, _ = read_planck_0.get_magnitude(model_param)
            app_mag_1, _ = read_planck_1.get_magnitude(model_param)
            app_mag_2, _ = read_planck_2.get_magnitude(model_param)

            list_color.append(app_mag_0[0]-app_mag_1[0])
            list_mag.append(app_mag_2[0])

        return box.create_box(boxtype='colormag',
                              library='planck',
                              object_type=None,
                              filters_color=filters_color,
                              filter_mag=filter_mag,
                              color=list_color,
                              magnitude=list_mag,
                              sptype=temperatures)
