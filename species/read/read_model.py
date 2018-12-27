"""
Read module.
"""

import os
import configparser

import h5py
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from .. analysis import photometry
from .. core import box
from .. data import database
from . import read_filter


class ReadModel:
    """
    Text
    """

    def __init__(self, model, wavelength):
        """
        :param model: Model name.
        :type model: str
        :param wavelength: Wavelength range (micron) or filter name. Full spectrum if set to None.
        :type wavelength: tuple(float, float) or str

        :return: None
        """

        self.parsec = 3.08567758147e16 # [m]
        self.r_jup = 71492000. # [m]

        self.model = model
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
            species_db.add_model(self.model)
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

        if self.model == 'drift-phoenix':
            feh = np.asarray(h5_file['models/drift-phoenix/feh'])
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
                                                       bounds_error=True,
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

        if self.model == 'drift-phoenix':
            feh = np.asarray(h5_file['models/drift-phoenix/feh'])

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

            modelbox.model = 'drift-phoenix'
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
                  coverage):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param coverage: Wavelength coverage.
        :type coverage: tuple

        :return: Spectrum (micron, W m-2 micron-1).
        :rtype: species.core.box.ModelBox
        """

        if self.spectrum_interp is None:
            self.interpolate()

        wavelength = [self.wavelength[0]]

        if coverage[0] == 'specres':
            while wavelength[-1] <= self.wavelength[1]:
                wavelength.append(wavelength[-1] + wavelength[-1]/coverage[1])
            wavelength = np.asarray(wavelength[:-1])

        elif coverage[0] == 'random':
            rannum = np.random.rand(int(coverage[1]))
            wavelength = self.wavelength[0] + rannum*(self.wavelength[1]-self.wavelength[0])
            wavelength = np.sort(wavelength)

        elif coverage[0] == 'wavelength':
            wavelength = coverage[1]

        flux = np.zeros(wavelength.shape)

        for i, item in enumerate(wavelength):

            if self.model == 'drift-phoenix':
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              item]

            elif self.model == 'petitcode_warm_clear':
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              item]

            elif self.model == 'petitcode_warm_cloudy':
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              model_par['fsed'],
                              item]

            elif self.model == 'petitcode_hot_clear':
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              model_par['co'],
                              item]

            elif self.model == 'petitcode_hot_cloudy':
                parameters = [model_par['teff'],
                              model_par['logg'],
                              model_par['feh'],
                              model_par['co'],
                              model_par['fsed'],
                              item]

            flux[i] = self.spectrum_interp(np.asarray(parameters))

        if 'radius' in model_par and 'distance' in model_par:
            scaling = (model_par['radius']*self.r_jup)**2 / (model_par['distance']*self.parsec)**2
            flux *= scaling

        modelbox = box.ModelBox()

        modelbox.model = 'drift-phoenix'
        modelbox.wavelength = wavelength
        modelbox.flux = flux
        modelbox.teff = model_par['teff']
        modelbox.logg = model_par['logg']
        modelbox.feh = model_par['feh']

        return modelbox

    def get_photometry(self,
                       model_par,
                       coverage,
                       synphot=None):
        """
        :param model_par: Model parameter values.
        :type model_par: dict
        :param coverage: Spectral coverage. The original grid is used (nearest model parameter
                         values) if set to none.
        :type coverage: float
        :param synphot: Synthetic photometry object.
        :type synphot: species.analysis.photometry.SyntheticPhotometry

        :return: Apparent magnitude (mag), absolute magnitude (mag).
        :rtype: float, float
        """

        if coverage is None:
            spectrum = self.get_data(model_par)

        else:
            if self.spectrum_interp is None:
                self.interpolate()

            spectrum = self.get_model(model_par, coverage)

        if not synphot:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_photometry(spectrum.wavelength, spectrum.flux)

    # def get_magnitude(self, model_par, coverage):
    #     """
    #     :param model_par: Model parameter values.
    #     :type model_par: dict
    #     :param coverage: Spectral coverage. The original grid is used (nearest model parameter
    #                     values) if set to none.
    #     :type coverage: float
    #
    #     :return: Apparent magnitude (mag), absolute magnitude (mag).
    #     :rtype: float, float
    #     """
    #
    #     if coverage is None:
    #         spectrum = self.get_data(model_par)
    #
    #     else:
    #         if self.spectrum_interp is None:
    #             self.interpolate()
    #
    #         spectrum = self.get_model(model_par, coverage)
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
        :return:
        :rtype: dict
        """

        h5_file = self.open_database()

        teff = h5_file['models/'+self.model+'/teff']
        logg = h5_file['models/'+self.model+'/logg']

        if self.model == 'drift-phoenix':
            feh = h5_file['models/'+self.model+'/feh']
            bounds = {'teff':(teff[0], teff[-1]),
                      'logg':(logg[0], logg[-1]),
                      'feh':(feh[0], feh[-1])}

        h5_file.close()

        return bounds

    def get_points(self):
        """
        :return:
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
