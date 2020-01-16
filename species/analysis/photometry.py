"""
Module with functionalities for calculating synthetic photometry.
"""

import os
import warnings
import configparser

import h5py
import numpy as np

from species.data import database
from species.read import read_filter, read_calibration


class SyntheticPhotometry:
    """
    Class for calculating synthetic photometry from a spectrum.
    """

    def __init__(self,
                 filter_name):
        """
        Parameters
        ----------
        filter_name : str
            Filter ID as listed in the database. Filters from the SVO Filter Profile Service are
            automatically downloaded and added to the database.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name
        self.filter_interp = None
        self.wavel_range = None

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def zero_point(self):
        """
        Internal function for calculating the zero-point of the provided `filter_name`.

        Returns
        -------
        float
            Zero-point flux density (W m-2 micron-1).
        """

        if self.wavel_range is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['spectra/calibration/vega']

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_spectrum('vega')
            h5_file = h5py.File(self.database, 'r')

        readcalib = read_calibration.ReadCalibration('vega', None)
        calibbox = readcalib.get_spectrum()

        wavelength = calibbox.wavelength
        flux = calibbox.flux

        wavelength_crop = wavelength[(wavelength > self.wavel_range[0]) &
                                     (wavelength < self.wavel_range[1])]

        flux_crop = flux[(wavelength > self.wavel_range[0]) &
                         (wavelength < self.wavel_range[1])]

        h5_file.close()

        return self.spectrum_to_flux(wavelength_crop, flux_crop)

    def spectrum_to_flux(self,
                         wavelength,
                         flux,
                         threshold=None):
        """
        Function for calculating the average flux density from a spectrum and a filter profile.

        Parameters
        ----------
        wavelength : numpy.ndarray, list(numpy.ndarray)
            Wavelength points (micron) provided as a 1D array or a list of 1D arrays.
        flux : numpy.ndarray, list(numpy.ndarray)
            Flux density (W m-2 micron-1) provided as a 1D array or a list of 1D arrays.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the minimum transmission value is
            larger than the threshold, a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter profile. Not used if set to
            None.

        Returns
        -------
        float, numpy.ndarray
            Average flux density (W m-2 micron-1).
        """

        if self.filter_interp is None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.filter_interp = transmission.interpolate_filter()

            if self.wavel_range is None:
                self.wavel_range = transmission.wavelength_range()

        if wavelength.size == 0:
            raise ValueError('Calculation of the mean flux is not possible because the '
                             'wavelength array is empty.')

        if isinstance(wavelength[0], (np.float32, np.float64)):
            indices = np.where((self.wavel_range[0] < wavelength) &
                               (wavelength < self.wavel_range[1]))[0]

            if indices.size == 1:
                raise ValueError("Calculating synthetic photometry requires more than one "
                                 "wavelength point.")

            wavelength = wavelength[indices]
            flux = flux[indices]

            transmission = self.filter_interp(wavelength)

            indices = np.isnan(transmission)
            indices = np.logical_not(indices)

            integrand1 = transmission[indices]*flux[indices]
            integrand2 = transmission[indices]

            integral1 = np.trapz(integrand1, wavelength[indices])
            integral2 = np.trapz(integrand2, wavelength[indices])

            photometry = integral1/integral2

        else:
            photometry = []

            for i, wl_item in enumerate(wavelength):
                indices = np.where((self.wavel_range[0] <= wl_item) &
                                   (wl_item <= self.wavel_range[1]))[0]

                if indices.size < 2:
                    photometry.append(np.nan)

                    warnings.warn('Calculating synthetic photometry requires more than one '
                                  'wavelength point. Photometry is set to NaN.', RuntimeWarning)

                else:
                    if threshold is None and (wl_item[0] > self.wavel_range[0] or
                                              wl_item[-1] < self.wavel_range[1]):

                        warnings.warn('Filter profile of '+self.filter_name+' extends beyond the '
                                      'spectrum ('+str(wl_item[0])+'-'+str(wl_item[-1])+'). The '
                                      'magnitude is set to NaN.', RuntimeWarning)

                        photometry.append(np.nan)

                    else:
                        wl_item = wl_item[indices]
                        flux_item = flux[i][indices]

                        transmission = self.filter_interp(wl_item)

                        if threshold is not None and (transmission[0] > threshold or
                                                      transmission[-1] > threshold):

                            warnings.warn(f'Filter profile of {self.filter_name} extends beyond '
                                          f'the spectrum ({wl_item[0]} - {wl_item[-1]}). The '
                                          f'magnitude is set to NaN.', RuntimeWarning)

                            photometry.append(np.nan)

                        else:
                            indices = np.isnan(transmission)
                            indices = np.logical_not(indices)

                            integrand1 = transmission[indices]*flux_item[indices]
                            integrand2 = transmission[indices]

                            integral1 = np.trapz(integrand1, wl_item[indices])
                            integral2 = np.trapz(integrand2, wl_item[indices])

                            photometry.append(integral1/integral2)

            photometry = np.asarray(photometry)

        return photometry

    def spectrum_to_magnitude(self,
                              wavelength,
                              flux,
                              distance=None,
                              threshold=None):
        """
        Function for calculating the apparent and absolute magnitude from a spectrum and a
        filter profile.

        Parameters
        ----------
        wavelength : numpy.ndarray, list(numpy.ndarray)
            Wavelength points (micron) provided as a 1D array or a list of 1D arrays.
        flux : numpy.ndarray, list(numpy.ndarray)
            Flux density (W m-2 micron-1) provided as a 1D array or a list of 1D arrays.
        distance : float, None
            Distance (pc). No absolute magnitude is calculated if set to None.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the minimum transmission value is
            larger than the threshold, a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter profile. Not used if set to
            None.

        Returns
        -------
        float, numpy.ndarray
            Apparent magnitude (mag).
        float, numpy.ndarray
            Absolute magnitude (mag).
        """

        vega_mag = 0.03  # [mag]

        zp_flux = self.zero_point()
        syn_flux = self.spectrum_to_flux(wavelength, flux, threshold)

        app_mag = vega_mag - 2.5*np.log10(syn_flux/zp_flux)

        if distance is None:
            abs_mag = None
        else:
            abs_mag = app_mag - 5.*np.log10(distance) + 5.

        return app_mag, abs_mag

    def magnitude_to_flux(self,
                          magnitude,
                          error=None,
                          zp_flux=None):
        """
        Function for converting a magnitude to a flux density.

        Parameters
        ----------
        magnitude : float
            Magnitude (mag).
        error : float, None
            Error (mag). Not used if set to None.
        zp_flux : float
            Zero-point flux density (W m-2 micron-1). The value is calculated if set to None.

        Returns
        -------
        float
            Flux density (W m-2 micron-1).
        float
            Error (W m-2 micron-1).
        """

        vega_mag = 0.03  # [mag]

        if zp_flux is None:
            zp_flux = self.zero_point()

        flux = 10.**(-0.4*(magnitude-vega_mag))*zp_flux

        if error is None:
            error_flux = None

        else:
            error_upper = flux * (10.**(0.4*error) - 1.)
            error_lower = flux * (1. - 10.**(-0.4*error))
            error_flux = (error_lower+error_upper)/2.

        return flux, error_flux

    def flux_to_magnitude(self,
                          flux,
                          distance):
        """
        Function for converting a flux density to a magnitude.

        Parameters
        ----------
        flux : float
            Flux density (W m-2 micron-1).
        distance : float
            Distance (pc).

        Returns
        -------
        float
            Apparent magnitude (mag).
        float
            Absolute magnitude (mag).
        """

        vega_mag = 0.03  # [mag]

        zp_flux = self.zero_point()

        app_mag = vega_mag - 2.5*np.log10(flux/zp_flux)

        if distance is None:
            abs_mag = None
        else:
            abs_mag = app_mag - 5.*np.log10(distance) + 5.

        return app_mag, abs_mag
