'''
Read module.
'''

import os
import configparser

import h5py
import numpy as np

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
                     model_par=None,
                     negative=False):
        '''
        :param model_par: Model parameter values. Not used if set to None.
        :type model_par: dict
        :param negative: Include negative values.
        :type negative: bool

        :return: Spectrum data.
        :rtype: numpy.ndarray
        '''

        h5_file = h5py.File(self.database, 'r')

        data = h5_file['spectra/calibration/'+self.spectrum]
        data = np.asarray(data)

        if not negative:
            indices = np.where(data[1, ] > 0.)[0]
            data = data[:, indices]

        h5_file.close()

        if model_par:
            data[1, ] = model_par['offset'] + model_par['scaling']*data[1, ]

        return box.create_box(boxtype='spectrum',
                              spectrum='calibration',
                              wavelength=data[0, ],
                              flux=data[1, ],
                              name=self.spectrum,
                              simbad=None,
                              sptype=None,
                              distance=None)

    def get_photometry(self,
                       model_par,
                       synphot=None):
        '''
        :param model_par: Model parameter values.
        :type model_par: dict

        :return: Average flux density (W m-2 micron-1).
        :rtype: float
        '''

        specbox = self.get_spectrum(model_par)
        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(specbox.wavelength, specbox.flux)
