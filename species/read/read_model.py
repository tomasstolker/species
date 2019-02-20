"""
Text
"""

import os
import math
import configparser

import h5py
import numpy as np

from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from PyAstronomy.pyasl import instrBroadGaussFast

from . import read_filter
from .. analysis import photometry
from .. core import box
from .. data import database


def multi_photometry(model,
                     filters,
                     parameters):
    """
    :param model: Model name.
    :type model: str
    :param filters: Filter IDs.
    :type filters: tuple(str, )
    :param parameters: Model parameter values.
    :type parameters: dict

    :return: Box with synthetic photometry.
    :rtype: species.core.box.SynphotBox
    """

    flux = {}
    for item in filters:
        readmodel = ReadModel(model, item)
        flux[item] = readmodel.get_photometry(parameters, ('specres', 100.))

    return box.create_box('synphot', name='synphot', flux=flux)


class ReadModel:
    """
    Text
    """

    def __init__(self,
                 model,
                 wavelength,
                 teff=None):
        """
        :param model: Model name.
        :type model: str
        :param wavelength: Wavelength range (micron) or filter name. Full spectrum if set to None.
        :type wavelength: tuple(float, float) or str
        :param teff: Effective temperature (K) range. Restricting the temperature range will speed
                     up the computation.
        :type teff: tuple(float, float)

        :return: None
        """

        self.gravity = 6.67408e-11 # [m3 kg−1 s−2]
        self.parsec = 3.08567758147e16 # [m]

        self.r_jup = 71492000. # [m]
        self.m_jup = 1.89813e27 # [kg]
        self.l_sun = 3.828e26 # [W]

        self.model = model
        self.teff = teff
        self.spectrum_interp = None

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

    def open_database(self):
        """
        :return: Database.
        :rtype: h5py._hl.files.File
        """

        h5_file = h5py.File(self.database, 'r')

        try:
            h5_file['models/'+self.model]

        except KeyError:
            h5_file.close()
            species_db = database.Database()
            species_db.add_model(self.model, self.wavelength, self.teff)
            h5_file = h5py.File(self.database, 'r')

        return h5_file

    def interpolate(self):
        """
        :return: None
        """

        h5_file = self.open_database()

        wavelength = np.asarray(h5_file['models/'+self.model+'/wavelength'])
        flux = np.asarray(h5_file['models/'+self.model+'/flux'])
        teff = np.asarray(h5_file['models/'+self.model+'/teff'])
        logg = np.asarray(h5_file['models/'+self.model+'/logg'])

        wl_index = (wavelength > self.wavelength[0]) & (wavelength < self.wavelength[1])

        index = np.where(wl_index)[0]

        wl_index[index[0] - 1] = True
        wl_index[index[-1] + 1] = True

        if self.model in ('drift-phoenix', 'bt-nextgen'):
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])
            flux = flux[:, :, :, wl_index]
            points = (teff, logg, feh, wavelength[wl_index])

        elif self.model == 'petitcode_warm_clear':
            feh = np.asarray(h5_file['models/petitcode_warm_clear/feh'])
            flux = flux[:, :, :, wl_index]
            points = np.asarray((teff, logg, feh, wavelength[wl_index]))

        elif self.model == 'petitcode_warm_cloudy':
            feh = np.asarray(h5_file['models/petitcode_warm_cloudy/feh'])
            fsed = np.asarray(h5_file['models/petitcode_warm_cloudy/fsed'])
            flux = flux[:, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, fsed, wavelength[wl_index]))

        elif self.model == 'petitcode_hot_clear':
            feh = np.asarray(h5_file['models/petitcode_hot_clear/feh'])
            co_ratio = np.asarray(h5_file['models/petitcode_hot_clear/co'])
            flux = flux[:, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, co_ratio, wavelength[wl_index]))

        elif self.model == 'petitcode_hot_cloudy':
            feh = np.asarray(h5_file['models/petitcode_hot_cloudy/feh'])
            co_ratio = np.asarray(h5_file['models/petitcode_hot_cloudy/co'])
            fsed = np.asarray(h5_file['models/petitcode_hot_cloudy/fsed'])
            flux = flux[:, :, :, :, :, wl_index]
            points = np.asarray((teff, logg, feh, co_ratio, fsed, wavelength[wl_index]))

        h5_file.close()

        self.spectrum_interp = RegularGridInterpolator(points=points,
                                                       values=flux,
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=np.nan)

    def get_data(self,
                 model_par):
        """
        :param model_par: Model parameter values. Only discrete values from the original grid
                          are possible. Else, the nearest grid values are selected.
        :type model_par: dict

        :return: Spectrum (micron, W m-2 micron-1).
        :rtype: species.core.box.ModelBox
        """

        h5_file = self.open_database()

        wavelength = np.asarray(h5_file['models/'+self.model+'/wavelength'])
        flux = np.asarray(h5_file['models/'+self.model+'/flux'])
        teff = np.asarray(h5_file['models/'+self.model+'/teff'])
        logg = np.asarray(h5_file['models/'+self.model+'/logg'])

        modelbox = box.ModelBox()

        if self.wavelength is None:
            wl_index = np.ones(wavelength.shape[0], dtype=bool)

        else:
            wl_index = (wavelength > self.wavelength[0]) & (wavelength < self.wavelength[1])
            index = np.where(wl_index)[0]

            wl_index[index[0] - 1] = True
            wl_index[index[-1] + 1] = True

        if self.model in ('drift-phoenix', 'bt-nextgen'):
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])

            teff_index = np.argwhere(teff == model_par['teff'])[0]

            if not teff_index:
                raise ValueError('Temperature value not found.')
            else:
                teff_index = teff_index[0]

            logg_index = np.argwhere(logg == model_par['logg'])[0]

            if not logg_index:
                raise ValueError('Surface gravity value not found.')
            else:
                logg_index = logg_index[0]

            feh_index = np.argwhere(feh == model_par['feh'])[0]

            if not feh_index:
                raise ValueError('Metallicity value not found.')
            else:
                feh_index = feh_index[0]

            wavelength = wavelength[wl_index]
            flux = flux[teff_index, logg_index, feh_index, wl_index]

            modelbox.model = self.model
            modelbox.feh = model_par['feh']

        if 'radius' in model_par and 'distance' in model_par:
            scaling = (model_par['radius']*self.r_jup)**2 / (model_par['distance']*self.parsec)**2
            flux *= scaling

        modelbox.wavelength = wavelength
        modelbox.flux = flux
        modelbox.teff = model_par['teff']
        modelbox.logg = model_par['logg']

        return modelbox

    def get_model(self,
                  model_par,
                  sampling):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param sampling: Type of wavelength sampling.
        :type sampling: tuple

        :return: Spectrum (micron, W m-2 micron-1).
        :rtype: species.core.box.ModelBox
        """

        if not self.wavelength:
            wl_points = self.get_wavelength()
            self.wavelength = (wl_points[0], wl_points[-1])

        if self.spectrum_interp is None:
            self.interpolate()

        wavelength = [self.wavelength[0]]

        if sampling[0] == 'specres':
            while wavelength[-1] <= self.wavelength[1]:
                wavelength.append(wavelength[-1] + wavelength[-1]/sampling[1])
            wavelength = np.asarray(wavelength[:-1])

        elif sampling[0] == 'gaussian':
            wavelength = np.linspace(self.wavelength[0], self.wavelength[1], sampling[1][0])

        flux = np.zeros(wavelength.shape)

        for i, item in enumerate(wavelength):

            if self.model in ('drift-phoenix', 'bt-nextgen'):
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              item]

            # elif self.model == 'petitcode_warm_clear':
            #     parameters = [model_par['teff'],
            #                   model_par['logg'],
            #                   model_par['feh'],
            #                   item]
            #
            # elif self.model == 'petitcode_warm_cloudy':
            #     parameters = [model_par['teff'],
            #                   model_par['logg'],
            #                   model_par['feh'],
            #                   model_par['fsed'],
            #                   item]
            #
            # elif self.model == 'petitcode_hot_clear':
            #     parameters = [model_par['teff'],
            #                   model_par['logg'],
            #                   model_par['feh'],
            #                   model_par['co'],
            #                   item]
            #
            # elif self.model == 'petitcode_hot_cloudy':
            #     parameters = [model_par['teff'],
            #                   model_par['logg'],
            #                   model_par['feh'],
            #                   model_par['co'],
            #                   model_par['fsed'],
            #                   item]

            flux[i] = self.spectrum_interp(np.asarray(parameters))

        if 'radius' in model_par and 'distance' in model_par:
            scaling = (model_par['radius']*self.r_jup)**2 / (model_par['distance']*self.parsec)**2
            flux *= scaling

            model_par['mass'], model_par['luminosity'] = \
                self.mass_luminosity(wavelength, flux, model_par)

        if sampling[0] == 'gaussian':
            index = np.where(np.isnan(flux))[0]

            wavelength = np.delete(wavelength, index)
            flux = np.delete(flux, index)

            flux = instrBroadGaussFast(wavelength, flux, sampling[1][1])

        return box.create_box(boxtype='model',
                              model=self.model,
                              wavelength=wavelength,
                              flux=flux,
                              parameters=model_par)

    def get_photometry(self,
                       model_par,
                       sampling,
                       synphot=None):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param sampling: Spectral sampling. The original grid is used (nearest model parameter
                         values) if set to none.
        :type sampling: float
        :param synphot: Synthetic photometry object.
        :type synphot: species.analysis.photometry.SyntheticPhotometry

        :return: Apparent magnitude (mag), absolute magnitude (mag).
        :rtype: float, float
        """

        if sampling is None:
            spectrum = self.get_data(model_par)

        else:
            if self.spectrum_interp is None:
                self.interpolate()

            spectrum = self.get_model(model_par, sampling)

        if not synphot:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(spectrum.wavelength, spectrum.flux)

    # def get_magnitude(self, model_par, sampling):
    #     """
    #     :param model_par: Model parameter values.
    #     :type model_par: dict
    #     :param sampling: Spectral sampling. The original grid is used (nearest model parameter
    #                     values) if set to none.
    #     :type sampling: float
    #
    #     :return: Apparent magnitude (mag), absolute magnitude (mag).
    #     :rtype: float, float
    #     """
    #
    #     if sampling is None:
    #         spectrum = self.get_data(model_par)
    #
    #     else:
    #         if self.spectrum_interp is None:
    #             self.interpolate()
    #
    #         spectrum = self.get_model(model_par, sampling)
    #
    #     transmission = read_filter.ReadFilter(self.filter_name)
    #     filter_interp = transmission.interpolate()
    #
    #     synphot = photometry.SyntheticPhotometry(filter_interp)
    #     mag = synphot.spectrum_to_magnitude(spectrum.wavelength,
    #                                         spectrum.flux,
    #                                         model_par['distance'])
    #
    #     return mag[0], mag[1]

    def get_bounds(self):
        """
        :return: Parameter boundaries of the model grid.
        :rtype: dict
        """

        h5_file = self.open_database()

        teff = h5_file['models/'+self.model+'/teff']
        logg = h5_file['models/'+self.model+'/logg']

        if self.model in ('drift-phoenix', 'bt-nextgen'):
            feh = h5_file['models/'+self.model+'/feh']
            bounds = {'teff':(teff[0], teff[-1]),
                      'logg':(logg[0], logg[-1]),
                      'feh':(feh[0], feh[-1])}

        h5_file.close()

        return bounds

    def get_wavelength(self):
        """
        :return: Wavelength points (micron).
        :rtype: numpy.ndarray
        """

        h5_file = self.open_database()
        wavelength = np.asarray(h5_file['models/'+self.model+'/wavelength'])
        h5_file.close()

        return wavelength

    def get_points(self):
        """
        :return: Parameter points of the model grid.
        :rtype: dict
        """

        points = {}

        h5_file = self.open_database()

        teff = h5_file['models/'+self.model+'/teff']
        logg = h5_file['models/'+self.model+'/logg']

        points['teff'] = np.asarray(teff)
        points['logg'] = np.asarray(logg)

        if self.model == 'drift-phoenix':
            feh = h5_file['models/'+self.model+'/feh']
            points['feh'] = np.asarray(feh)

        h5_file.close()

        return points

    def get_parameters(self):
        """
        :return: Model parameters.
        :rtype: list(str, )
        """

        h5_file = self.open_database()

        dset = h5_file['models/'+self.model]
        nparam = dset.attrs['nparam']

        param = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i)])

        return param

    def mass_luminosity(self,
                        wavelength,
                        flux,
                        model_par):
        """
        :param wavelength: Wavelength points (micro).
        :type wavelength: numpy.ndarray
        :param flux: Flux density (W m-2 micron-1).
        :type flux: numpy.ndarray
        :param model_par: Model parameter values. Should contain the surface gravity, radius,
                          and distance.
        :type model_par: dict

        :return: Mass (Mjup) and Luminosity (Lsun).
        :rtype: float, float
        """

        logg = 1e-2 * 10.**model_par['logg'] # [m s-1]

        radius = model_par['radius'] # [Rjup]
        radius *= self.r_jup # [m]

        distance = model_par['distance'] # [pc]
        distance *= self.parsec # [m]

        mass = logg*radius**2/self.gravity # [kg]
        mass /= self.m_jup # [Mjup]

        luminosity = 4.*math.pi*distance**2*simps(flux, wavelength) # [W]
        luminosity /= self.l_sun # [Lsun]

        return mass, luminosity
