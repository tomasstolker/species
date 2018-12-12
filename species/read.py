"""
Read module.
"""

import os
import sys
import configparser

import h5py
import numpy as np

from scipy.optimize import fsolve
from scipy.interpolate import interp1d, RegularGridInterpolator

import species.database
import species.photometry


class ReadModel(object):
    """
    Text
    """

    def __init__(self, model, wavelength):
        """
        :param model: Model name.
        :type model: str
        :param wavelength: Wavelength range (micron) or filter name.
        :type wavelength: tuple(float, float) or str

        :return: None
        """

        self.model = model
        self.spectrum_interp = None

        if isinstance(wavelength, str):
            self.filter_name = wavelength
            transmission = ReadFilter(wavelength)
            self.wavelength = transmission.wavelength_range()

        else:
            self.filter_name = None
            self.wavelength = wavelength

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']

    def get_data(self, model_par):
        """
        :param model_par: Model parameter values. Only discrete values from the original grid
                          are possible. Else, the nearest grid values are selected.
        :type model_par: dict

        :return: Spectrum (micron, W m-2 micron-1).
        :rtype: numpy.ndarray
        """

        parsec = 3.08567758147e16 # [m]
        r_jup = 71492000. # [m]

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["models/"+self.model]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_model(self.model)
            h5_file = h5py.File(self.database, 'r')

        wavelength = np.asarray(h5_file["models/"+self.model+"/wavelength"])
        flux = np.asarray(h5_file["models/"+self.model+"/flux"])
        teff = np.asarray(h5_file["models/"+self.model+"/teff"])
        logg = np.asarray(h5_file["models/"+self.model+"/logg"])

        teff_index = np.abs(teff - model_par['teff']).argmin()
        logg_index = np.abs(logg - model_par['logg']).argmin()

        wl_index = (wavelength > self.wavelength[0]) & (wavelength < self.wavelength[1])
        index = np.where(wl_index)[0]

        wl_index[index[0] - 1] = True
        wl_index[index[-1] + 1] = True

        if self.model == "drift-phoenix":
            feh = np.asarray(h5_file["models/drift-phoenix/feh"])
            feh_index = np.abs(feh - model_par['feh']).argmin()

            wavelength = wavelength[wl_index]
            spectrum = flux[teff_index, logg_index, feh_index, wl_index]

        scaling = (model_par['radius']*r_jup)**2 / (model_par['distance']*parsec)**2
        spectrum *= scaling

        return np.vstack((wavelength, spectrum))

    def interpolate(self):
        """
        :return: None
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["models/"+self.model]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_model(self.model)
            h5_file = h5py.File(self.database, 'r')

        wavelength = np.asarray(h5_file["models/"+self.model+"/wavelength"])
        flux = np.asarray(h5_file["models/"+self.model+"/flux"])
        teff = np.asarray(h5_file["models/"+self.model+"/teff"])
        logg = np.asarray(h5_file["models/"+self.model+"/logg"])

        wl_index = (wavelength > self.wavelength[0]) & (wavelength < self.wavelength[1])

        index = np.where(wl_index)[0]

        wl_index[index[0] - 1] = True
        wl_index[index[-1] + 1] = True

        if self.model == "drift-phoenix":
            feh = np.asarray(h5_file["models/drift-phoenix/feh"])
            flux = flux[:, :, :, wl_index]
            points = np.asarray((teff, logg, feh, wavelength[wl_index]))

        elif self.model == "petitcode_warm_clear":
            feh = np.asarray(h5_file["models/petitcode_warm_clear/feh"])
            flux = flux[:, :, :, wl_index]
            points = np.asarray((teff, logg, feh, wavelength[wl_index]))

        elif self.model == "petitcode_warm_cloudy":
            feh = np.asarray(h5_file["models/petitcode_warm_cloudy/feh"])
            fsed = np.asarray(h5_file["models/petitcode_warm_cloudy/fsed"])
            flux = flux[:, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, fsed, wavelength[wl_index]))

        elif self.model == "petitcode_hot_clear":
            feh = np.asarray(h5_file["models/petitcode_hot_clear/feh"])
            co_ratio = np.asarray(h5_file["models/petitcode_hot_clear/co"])
            flux = flux[:, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, co_ratio, wavelength[wl_index]))

        elif self.model == "petitcode_hot_cloudy":
            feh = np.asarray(h5_file["models/petitcode_hot_cloudy/feh"])
            co_ratio = np.asarray(h5_file["models/petitcode_hot_cloudy/co"])
            fsed = np.asarray(h5_file["models/petitcode_hot_cloudy/fsed"])
            flux = flux[:, :, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, co_ratio, fsed, wavelength[wl_index]))

        h5_file.close()

        self.spectrum_interp = RegularGridInterpolator(points, flux)

    def get_model(self, model_par, resolution):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param resolution: Spectral resolution.
        :type resolution: float

        :return: Spectrum (micron, W m-2 micron-1).
        :rtype: numpy.ndarray
        """

        if self.spectrum_interp is None:
            self.interpolate()

        wavelength = [self.wavelength[0]]

        while wavelength[-1] <= self.wavelength[1]:
            wavelength.append(wavelength[-1] + wavelength[-1]/resolution)

        wavelength = np.asarray(wavelength[:-1])
        flux = np.zeros(wavelength.shape)

        for i, item in enumerate(wavelength):

            if self.model == "petitcode_warm_clear":
                parameters = [model_par["teff"],
                              model_par["logg"],
                              model_par["feh"],
                              item]

            elif self.model == "petitcode_warm_cloudy":
                parameters = [model_par["teff"],
                              model_par["logg"],
                              model_par["feh"],
                              model_par["fsed"],
                              item]

            elif self.model == "petitcode_hot_clear":
                parameters = [model_par["teff"],
                              model_par["logg"],
                              model_par["feh"],
                              model_par["co"],
                              item]

            elif self.model == "petitcode_hot_cloudy":
                parameters = [model_par["teff"],
                              model_par["logg"],
                              model_par["feh"],
                              model_par["co"],
                              model_par["fsed"],
                              item]

            elif self.model == "drift-phoenix":
                parameters = [model_par["teff"],
                              model_par["logg"],
                              model_par["feh"],
                              item]

            flux[i] = self.spectrum_interp(np.asarray(parameters))

        return np.vstack((wavelength, flux))

    def get_magnitude(self, model_par, resolution):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param resolution: Spectral resolution. The original grid is used (nearest model parameter
                           values) if set to none.
        :type resolution: float

        :return: Apparent magnitude (mag), absolute magnitude (mag).
        :rtype: float, float
        """

        if resolution is None:
            spectrum = self.get_data(model_par)

        else:
            if self.spectrum_interp is None:
                self.interpolate()

            spectrum = self.get_model(model_par, resolution)

        transmission = ReadFilter(self.filter_name)
        filter_interp = transmission.interpolate()

        synphot = species.photometry.SyntheticPhotometry(filter_interp)
        mag = synphot.spectrum_to_magnitude(spectrum[0, ], spectrum[1, ], model_par['distance'])

        return mag[0], mag[1]


class ReadSpectrum(object):
    """
    Text
    """

    def __init__(self, spectrum, filter_name):
        """
        :param spectrum: Spectral library.
        :type spectrum: str
        :param filter_name: Filter name. Full spectrum is read if filter_name is set to None.
        :type filter_name: str

        :return: None
        """

        self.spectrum = spectrum
        self.filter_name = filter_name

        if filter_name is None:
            self.wl_range = None

        else:
            read_filter = ReadFilter(filter_name)
            self.wl_range = read_filter.wavelength_range()

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']

    def get_spectrum(self, ignore_nan=True):
        """
        :return: Wavelength, flux.
        :rtype: numpy.ndarray, numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["spectra/"+self.spectrum]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_spectrum(self.spectrum)
            h5_file = h5py.File(self.database, 'r')

        list_wavelength = []
        list_flux = []

        # list_name = []
        # list_sptype = []
        # list_distance = []

        for item in h5_file["spectra/"+self.spectrum]:
            data = h5_file["spectra/"+self.spectrum+"/"+item]

            wavelength = data[0, :] # [micron]
            flux = data[1, :] # [W m-2 micron-1]
            error = data[2, :] # [W m-2 micron-1]

            if ignore_nan:
                indices = np.isnan(flux)
                indices = np.logical_not(indices)
                indices = np.where(indices)[0]

                wavelength = wavelength[indices]
                flux = flux[indices]
                error = error[indices]

            if self.wl_range is None:
                wl_index = np.arange(0, len(wavelength), 1)

            else:
                wl_index = (flux > 0.) & (wavelength > self.wl_range[0]) & \
                           (wavelength < self.wl_range[1])

            count = np.count_nonzero(wl_index)

            if count > 0:
                index = np.where(wl_index)[0]

                if index[0] > 0:
                    wl_index[index[0] - 1] = True

                if index[-1] < len(wl_index)-1:
                    wl_index[index[-1] + 1] = True

                list_wavelength.append(wavelength[wl_index])
                list_flux.append(flux[wl_index])

                # list_name.append(data.attrs['name'])
                # list_sptype.append(data.attrs['sptype'])
                # list_distance.append(data.attrs['distance'])

        wavelength = np.asarray(list_wavelength)
        flux = np.asarray(list_flux)

        # name = np.asarray(list_name)
        # sptype = np.asarray(list_sptype)
        # distance = np.asarray(list_distance)

        if wavelength.shape[0] == 1:
            wavelength = np.squeeze(wavelength)
            flux = np.squeeze(flux)

        return wavelength, flux

class ReadFilter(object):
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

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']

    def get_filter(self):
        """
        :return: Filter data.
        :rtype: numpy.ndarray
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["filters/"+self.filter_name]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_filter(self.filter_name)
            h5_file = h5py.File(self.database, 'r')

        data = h5_file["filters/"+self.filter_name]
        data = np.asarray(data)

        h5_file.close()

        return data

    def wavelength_range(self):
        """
        :return: Minimum and maximum wavelength (micron).
        :rtype: float, float
        """

        data = self.get_filter()

        return np.amin(data[0, ]), np.amax(data[0, ])

    def mean_wavelength(self):
        """
        :return: Mean wavelength (micron).
        :rtype: float
        """

        data = self.get_filter()

        return np.trapz(data[0, ]*data[1, ], data[0, ]) / np.trapz(data[1, ], data[0, ])

    def interpolate(self):
        """
        :return: Interpolated filter.
        :rtype: scipy.interpolate.interpolate.interp1d
        """

        data = self.get_filter()

        filter_interp = interp1d(data[0, ],
                                 data[1, ],
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=float('nan'))

        return filter_interp

    def filter_width(self):
        """
        :return: Filter width (micron).
        :rtype: float
        """

        data = self.get_filter()

        wl_min, _ = self.wavelength_range()
        wl_mean = self.mean_wavelength()

        filter_interp = self.interpolate()

        interp_shift = lambda x: filter_interp(x) - np.amax(data[1, ])/2.

        return 2.*abs(wl_mean - fsolve(interp_shift, (wl_mean+wl_min)/2.)[0])


class ReadColorMagnitude(object):
    """
    Text
    """

    def __init__(self, filters_color, filter_mag):
        """
        :param filters_color:
        :type filters_color: tuple(str, str)
        :param filter_mag:
        :type filter_mag: str

        :return: None
        """

        self.filters_color = filters_color
        self.filter_mag = filter_mag

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']

    def get_color_magnitude(self, object_type):
        """
        :param object_type:
        :type object_type:

        :return:
        :rtype:
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["photometry/vlm-plx"]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_photometry("vlm-plx")
            h5_file = h5py.File(self.database, 'r')

        sptype = np.asarray(h5_file["photometry/vlm-plx/sptype"])
        flag = np.asarray(h5_file["photometry/vlm-plx/flag"])
        distance = np.asarray(h5_file["photometry/vlm-plx/distance"]) # [pc]

        if object_type == "field":
            indices = np.where(flag == "null")[0]

        mag1 = np.asarray(h5_file["photometry/vlm-plx/"+self.filters_color[0]])
        mag2 = np.asarray(h5_file["photometry/vlm-plx/"+self.filters_color[1]])

        color = mag1 - mag2

        if self.filter_mag == self.filters_color[0]:
            mag = species.photometry.apparent_to_absolute(mag1, distance)

        elif self.filter_mag == self.filters_color[1]:
            mag = species.photometry.apparent_to_absolute(mag2, distance)

        h5_file.close()

        return color[indices], mag[indices], sptype[indices]


class ReadObject(object):
    """
    Text
    """

    def __init__(self, object_name):
        """
        :param object_name: Object name.
        :type object_name: str

        :return: None
        """

        self.object_name = object_name

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']

    def get_magnitude(self, filter_name):
        """
        :param filter_name: Filter name.
        :type filter_name: str

        :return: Apparent magnitude (mag).
        :rtype: float
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file["objects/"+self.object_name]

        except KeyError:
            h5_file.close()
            database = species.database.Database()
            database.add_filter(self.object_name)
            h5_file = h5py.File(self.database, 'r')

        return np.asarray(h5_file["objects/"+self.object_name+"/"+filter_name])
