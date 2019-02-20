"""
Photometry module.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.integrate import simps

from species.data import database
from species.read import read_filter, read_spectrum


def apparent_to_absolute(app_mag, distance):
    """
    :param app_mag: Apparent magnitude (mag).
    :type app_mag: float or numpy.ndarray
    :param distance: Distance (pc).
    :type distance: float or numpy.ndarray

    :return: Absolute magnitude (mag).
    :rtype: float or numpy.ndarray
    """

    return app_mag - 5.*np.log10(distance) + 5.


class SyntheticPhotometry:
    """
    Text
    """

    def __init__(self, filter_name):
        """
        :param filter_name: Filter name.
        :type filter_name: str

        :return: None
        """

        self.filter_name = filter_name
        self.filter_interp = None

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def zero_point(self, wl_range):
        """
        :param wl_range: Wavelength range (micron). The range from the filter transmission
                         curve is used if set to None.
        :type wl_range: float

        :return: tuple(float, float)
        :rtype:
        """

        if wl_range is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            wl_range = transmission.wavelength_range()

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['spectra/calibration/vega']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectrum('vega')
            h5_file = h5py.File(self.database, 'r')

        specdata = read_spectrum.ReadSpectrum('calibration', None)
        specbox = specdata.get_spectrum()

        wavelength = specbox.wavelength[0]
        flux = specbox.flux[0]

        wavelength_crop = wavelength[(wavelength > wl_range[0]) & (wavelength < wl_range[1])]
        flux_crop = flux[(wavelength > wl_range[0]) & (wavelength < wl_range[1])]

        return self.spectrum_to_photometry(wavelength_crop, flux_crop)

    def spectrum_to_photometry(self, wavelength, flux_density):
        """
        :param wavelength: Wavelength (micron).
        :type wavelength: numpy.ndarray
        :param flux_density: Flux density (W m-2 micron-1).
        :type flux_density: numpy.ndarray

        :return: Average flux density (W m-2 micron-1).
        :rtype: float or numpy.ndarray
        """

        if self.filter_interp is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.filter_interp = transmission.interpolate()

        if isinstance(wavelength[0], (np.float32, np.float64)):
            transmission = self.filter_interp(wavelength)

            indices = np.isnan(transmission)
            indices = np.logical_not(indices)

            integrand1 = transmission[indices]*flux_density[indices]
            integrand2 = transmission[indices]

            integral1 = simps(integrand1, wavelength[indices])
            integral2 = simps(integrand2, wavelength[indices])

            photometry = integral1/integral2

        else:
            photometry = []
            for i, _ in enumerate(wavelength):
                transmission = self.filter_interp(wavelength[i])

                indices = np.isnan(transmission)
                indices = np.logical_not(indices)

                integrand1 = transmission[indices]*flux_density[i][indices]
                integrand2 = transmission[indices]

                integral1 = simps(integrand1, wavelength[i][indices])
                integral2 = simps(integrand2, wavelength[i][indices])

                photometry.append(integral1/integral2)

            photometry = np.asarray(photometry)

        return photometry

    def spectrum_to_magnitude(self, wavelength, flux_density, distance):
        """
        :param wavelength: Wavelength (micron).
        :type wavelength: numpy.ndarray
        :param flux_density: Flux density (W m-2 micron-1).
        :type flux_density: numpy.ndarray
        :param distance: Distance (pc). No absolute magnitude is calculated if set to None.
        :type distance: float

        :return: Flux (W m-2).
        :rtype: float or numpy.ndarray
        """

        vega_mag = 0.03 # [mag]

        flux = self.spectrum_to_photometry(wavelength, flux_density)
        zp_flux = self.zero_point((wavelength[0], wavelength[-1]))

        # indices = np.isnan(distance)
        # indices = np.logical_not(indices)
        # indices = np.where(indices)[0]
        #
        # flux = flux[indices]
        # distance = distance[indices]

        app_mag = vega_mag - 2.5*np.log10(flux/zp_flux)

        if distance is None:
            abs_mag = None
        else:
            abs_mag = app_mag - 5.*np.log10(distance) + 5.

        return app_mag, abs_mag

    def magnitude_to_flux(self, magnitude, error, zp_flux=None):
        """
        :param magnitude: Magnitude (mag).
        :type magnitude: float
        :param error: Error (mag).
        :type error: float
        :param zp_flux: Zero point flux (W m-2 micron-1). Calculated if set to None.
        :type zp_flux: float

        :return: Flux (W m-2 micron-1), lower error, upper error
        :rtype: float, tuple(float, float)
        """

        vega_mag = 0.03 # [mag]

        if zp_flux is None:
            zp_flux = self.zero_point(None)

        flux = 10.**(-0.4*(magnitude-vega_mag))*zp_flux

        if error is None:
            error_upper = None
            error_lower = None

        else:
            error_upper = flux * (10.**(0.4*error) - 1.)
            error_lower = flux * (1. - 10.**(-0.4*error))

        return flux, (error_lower+error_upper)/2.

    def flux_to_magnitude(self, flux, distance):
        """
        :param flux: Flux density (W m-2 micron-1).
        :type flux: float
        :param error: Distance (pc).
        :type error: float

        :return: Apparent magnitude (mag), absolute magnitude (mag).
        :rtype: float, float
        """

        vega_mag = 0.03 # [mag]

        zp_flux = self.zero_point(None)

        app_mag = vega_mag - 2.5*np.log10(flux/zp_flux)

        if distance is None:
            abs_mag = None
        else:
            abs_mag = app_mag - 5.*np.log10(distance) + 5.

        return app_mag, abs_mag
