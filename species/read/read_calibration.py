'''
Read module.
'''

import os
import sys
import configparser

import h5py
import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from species.analysis import photometry
from species.core import box
from species.read import read_filter


class ReadCalibration:
    '''
    Text
    '''

    def __init__(self,
                 spectrum,
                 filter_name):
        '''
        :param spectrum: Database tag of the calibration spectrum.
        :type spectrum: str
        :param filter_name: Filter ID. Full spectrum is used if set to None.
        :type filter_name: str

        :return: None
        '''

        self.spectrum = spectrum
        self.filter_name = filter_name

        if filter_name is None:
            self.wl_range = None

        else:
            transmission = read_filter.ReadFilter(filter_name)
            self.wl_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def interpolate(self):
        '''
        :return: Interpolated spectrum.
        :rtype: scipy.interpolate.interpolate.interp1d
        '''

        calibbox = self.get_spectrum()

        return interp1d(calibbox.wavelength,
                        calibbox.flux,
                        kind='cubic',
                        bounds_error=False,
                        fill_value=float('nan'))

    def get_spectrum(self,
                     parameters=None,
                     negative=False,
                     extrapolate=False,
                     min_wavelength=None):
        '''
        :param parameters: Model parameter values. Not used if set to None.
        :type parameters: dict
        :param negative: Include negative values.
        :type negative: bool
        :param extrapolate: Extrapolate to 6 micron by fitting a power law function.
        :type extrapolate: bool
        :param min_wavelength: Minimum wavelength used for fitting the power law function. All data
                               is used if set to None.
        :type min_wavelength: float

        :return: Spectrum data.
        :rtype: numpy.ndarray
        '''

        h5_file = h5py.File(self.database, 'r')

        data = h5_file['spectra/calibration/'+self.spectrum]
        data = np.asarray(data)

        wavelength = np.asarray(data[0, ])
        flux = np.asarray(data[1, ])
        error = np.asarray(data[2, ])

        if not negative:
            indices = np.where(flux > 0.)[0]

            wavelength = wavelength[indices]
            flux = flux[indices]
            error = error[indices]

        h5_file.close()

        if parameters:
            flux = parameters['offset'] + parameters['scaling']*flux

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
                                   p0=(0., np.mean(flux), -1.),
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
                       parameters,
                       synphot=None):
        '''
        :param parameters: Model parameter values.
        :type parameters: dict

        :return: Average flux density (W m-2 micron-1).
        :rtype: float
        '''

        specbox = self.get_spectrum(parameters)
        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(specbox.wavelength, specbox.flux)
