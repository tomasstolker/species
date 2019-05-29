"""
Read module.
"""

import os
import sys
import configparser

import h5py
import spectres
import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from species.analysis import photometry
from species.core import box
from species.read import read_filter


class ReadCalibration:
    """
    Text
    """

    def __init__(self,
                 spectrum,
                 filter_name):
        """
        Parameters
        ----------
        spectrum : str
            Database tag of the calibration spectrum.
        filter_name : str
            Filter ID. Full spectrum is used if set to None.

        Returns
        -------
        NoneType
            None
        """

        self.spectrum = spectrum
        self.filter_name = filter_name

        if filter_name:
            transmission = read_filter.ReadFilter(filter_name)
            self.wl_range = transmission.wavelength_range()

        else:
            self.wl_range = None

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def interpolate(self):
        """
        :return: Linearly interpolated spectrum.
        :rtype: scipy.interpolate.interpolate.interp1d
        """

        calibbox = self.get_spectrum()

        return interp1d(calibbox.wavelength,
                        calibbox.flux,
                        kind='linear',
                        bounds_error=False,
                        fill_value=float('nan'))

    def get_spectrum(self,
                     parameters=None,
                     negative=False,
                     specres=None,
                     extrapolate=False,
                     min_wavelength=None):
        """
        Parameters
        ----------

        parameters : dict
            Model parameter values. Not used if set to None.
        negative : bool
            Include negative values.
        specres : float
            Spectral resolution. Original wavelength points are used if set to None.
        extrapolate : bool
            Extrapolate to 6 micron by fitting a power law function.
        min_wavelength : float
            Minimum wavelength used for fitting the power law function. All data is used if set
            to None.

        Returns
        -------
        numpy.ndarray
            Spectrum data.
        """

        h5_file = h5py.File(self.database, 'r')

        data = h5_file['spectra/calibration/'+self.spectrum]
        data = np.asarray(data)

        wavelength = np.asarray(data[0, ])
        flux = np.asarray(data[1, ])
        error = np.asarray(data[2, ])

        h5_file.close()

        if not negative:
            indices = np.where(flux > 0.)[0]
            wavelength = wavelength[indices]
            flux = flux[indices]
            error = error[indices]

        if parameters:
            flux = parameters['scaling']*flux

        if self.wl_range:
            wl_index = (flux > 0.) & (wavelength > self.wl_range[0]) & \
                       (wavelength < self.wl_range[1])

        else:
            wl_index = np.arange(0, wavelength.size, 1)

        count = np.count_nonzero(wl_index)

        if count > 0:
            index = np.where(wl_index)[0]

            if index[0] > 0:
                wl_index[index[0] - 1] = True

            if index[-1] < len(wl_index)-1:
                wl_index[index[-1] + 1] = True

            wavelength = wavelength[wl_index]
            flux = flux[wl_index]
            error = error[wl_index]

        if extrapolate:
            def _power_law(wavelength, offset, scaling, power_index):
                return offset + scaling*wavelength**power_index

            if min_wavelength:
                indices = np.where(wavelength > min_wavelength)[0]
            else:
                indices = np.arange(0, wavelength.size, 1)

            popt, pcov = curve_fit(f=_power_law,
                                   xdata=wavelength[indices],
                                   ydata=flux[indices],
                                   p0=(0., np.mean(flux[indices]), -1.),
                                   sigma=error[indices])

            sigma = np.sqrt(np.diag(pcov))

            sys.stdout.write('Fit result for f(x) = a + b*x^c:\n')
            sys.stdout.write('a = '+str(popt[0])+' +/- '+str(sigma[0])+'\n')
            sys.stdout.write('b = '+str(popt[1])+' +/- '+str(sigma[1])+'\n')
            sys.stdout.write('c = '+str(popt[2])+' +/- '+str(sigma[2])+'\n')
            sys.stdout.flush()

            while wavelength[-1] <= 6.:
                wl_add = wavelength[-1] + wavelength[-1]/1000.
                wavelength = np.append(wavelength, wl_add)
                flux = np.append(flux, _power_law(wl_add, popt[0], popt[1], popt[2]))
                error = np.append(error, 0.)

        if specres:
            wavelength_new = [wavelength[0]]
            while wavelength_new[-1] < wavelength[-1]:
                wavelength_new.append(wavelength_new[-1] + wavelength_new[-1]/specres)

            wavelength_new = np.asarray(wavelength_new[:-1])

            value_error = True

            while value_error:
                try:
                    flux_new, error_new = spectres.spectres(new_spec_wavs=wavelength_new,
                                                            old_spec_wavs=wavelength,
                                                            spec_fluxes=flux,
                                                            spec_errs=error)

                    value_error = False

                except ValueError:
                    wavelength_new = wavelength_new[1:-1]
                    value_error = True

            wavelength = wavelength_new
            flux = flux_new
            error = error_new

        return box.create_box(boxtype='spectrum',
                              spectrum='calibration',
                              wavelength=wavelength,
                              flux=flux,
                              error=error,
                              name=self.spectrum,
                              simbad=None,
                              sptype=None,
                              distance=None)

    def get_photometry(self,
                       parameters=None,
                       synphot=None):
        """
        Parameters
        ----------
        parameters : dict, None
            Model parameter values. Not used if set to None.
        synphot

        Returns
        -------
        float
            Average flux density (W m-2 micron-1).
        """

        specbox = self.get_spectrum(parameters, )

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(specbox.wavelength, specbox.flux)

    def get_magnitude(self,
                      parameters=None,
                      synphot=None):
        """
        Parameters
        ----------
        parameters : dict, None
            Model parameter values. Not used if set to None.
        synphot

        Returns
        -------
        float
            Apparent magnitude (mag).
        """

        specbox = self.get_spectrum(parameters, )

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_magnitude(specbox.wavelength, specbox.flux, distance=None)
