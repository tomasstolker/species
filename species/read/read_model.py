"""
Module with reading functionalities for atmospheric model spectra.
"""

import os
import math
import warnings
import configparser

import h5py
import spectres
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from species.analysis import photometry
from species.core import box, constants
from species.data import database
from species.read import read_filter, read_calibration
from species.util import read_util


class ReadModel:
    """
    Class for reading a model spectrum from the database.
    """

    def __init__(self,
                 model,
                 wavel_range=None,
                 filter_name=None):
        """
        Parameters
        ----------
        model : str
            Name of the atmospheric model.
        wavel_range : tuple(float, float), None
            Wavelength range (micron). Full spectrum is selected if set to None. Not used if
            ``filter_name`` is not None.
        filter_name : str, None
            Filter ID that is used for the wavelength range. The ``wavel_range`` is used if set
            to None.

        Returns
        -------
        NoneType
            None
        """

        self.model = model

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        self.filter_name = filter_name
        self.wavel_range = wavel_range

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        self.database = config['species']['database']

    def open_database(self):
        """
        Internal function for opening the `species` database.

        Returns
        -------
        h5py._hl.files.File
            The HDF5 database.
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database_path = config['species']['database']

        h5_file = h5py.File(database_path, 'r')

        try:
            h5_file['models/'+self.model]
        except KeyError:
            raise ValueError(f'The \'{self.model}\' model spectra are not present in the '
                             f'database.')

        return h5_file

    def wavelength_points(self,
                          hdf5_file):
        """
        Internal function for extracting the wavelength points and indices that are used.

        Parameters
        ----------
        hdf5_file : h5py._hl.files.File
            The HDF5 database.

        Returns
        -------
        numpy.ndarray
            Wavelength points (micron).
        numpy.ndarray
            Array with the size of the original wavelength grid. The booleans indicate if a
            wavelength point was used.
        """

        wl_points = np.asarray(hdf5_file[f'models/{self.model}/wavelength'])

        if self.wavel_range is None:
            wl_index = np.ones(wl_points.shape[0], dtype=bool)

        else:
            wl_index = (wl_points > self.wavel_range[0]) & (wl_points < self.wavel_range[1])
            index = np.where(wl_index)[0]

            if index[0]-1 >= 0:
                wl_index[index[0] - 1] = True

            if index[-1]+1 < wl_index.size:
                wl_index[index[-1] + 1] = True

        return wl_points[wl_index], wl_index

    def interpolate_model(self):
        """
        Internal function for linearly interpolating the grid of model spectra.

        Returns
        -------
        NoneType
            None
        """

        h5_file = self.open_database()

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])
        teff = np.asarray(h5_file[f'models/{self.model}/teff'])
        logg = np.asarray(h5_file[f'models/{self.model}/logg'])

        if self.wl_points is None:
            self.wl_points, self.wl_index = self.wavelength_points(h5_file)

        if self.model in ['ames-cond', 'ames-dusty', 'bt-settl']:
            points = (teff, logg)
            flux = flux[:, :, self.wl_index]

        if self.model in ['bt-nextgen', 'drift-phoenix', 'petitcode-cool-clear']:
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])

            points = (teff, logg, feh)
            flux = flux[:, :, :, self.wl_index]

        elif self.model == 'petitcode-cool-cloudy':
            feh = np.asarray(h5_file['models/petitcode-cool-cloudy/feh'])
            fsed = np.asarray(h5_file['models/petitcode-cool-cloudy/fsed'])

            points = (teff, logg, feh, fsed)
            flux = flux[:, :, :, :, self.wl_index]

        elif self.model == 'petitcode-hot-clear':
            feh = np.asarray(h5_file['models/petitcode-hot-clear/feh'])
            co_ratio = np.asarray(h5_file['models/petitcode-hot-clear/co'])

            points = (teff, logg, feh, co_ratio)
            flux = flux[:, :, :, :, self.wl_index]

        elif self.model == 'petitcode-hot-cloudy':
            feh = np.asarray(h5_file['models/petitcode-hot-cloudy/feh'])
            co_ratio = np.asarray(h5_file['models/petitcode-hot-cloudy/co'])
            fsed = np.asarray(h5_file['models/petitcode-hot-cloudy/fsed'])

            flux = flux[:, :, :, :, :, self.wl_index]
            points = (teff, logg, feh, co_ratio, fsed)

        h5_file.close()

        self.spectrum_interp = RegularGridInterpolator(points=points,
                                                       values=flux,
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=np.nan)

    def get_model(self,
                  model_param,
                  spec_res=None,
                  wavel_resample=None,
                  magnitude=False,
                  smooth=False):
        """
        Function for extracting a model spectrum by linearly interpolating the model grid. The
        parameters values should lie within the boundaries of the grid points that are stored
        in the database. The stored grid points can be inspected with
        :func:`~species.read.read_model.ReadModel.get_points`.

        Parameters
        ----------
        model_param : dict
            Model parameters and values. The values should be within the boundaries of the grid.
            The grid boundaries of the available spectra in the database can be obtained with
            :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution, achieved by smoothing with a Gaussian kernel. The original
            wavelength points are used if both ``spec_res`` and ``wavel_resample`` are set to None.
        wavel_resample : numpy.ndarray
            Wavelength points (micron) to which the spectrum is resampled. Only used if
            ``spec_res`` is set to None.
        magnitude : bool
            Normalize the spectrum with a flux calibrated spectrum of Vega and return the magnitude
            instead of flux density.
        smooth : bool
            If True, the spectral resolution is changed by smoothing with a Gaussian kernel. In
            that case the original wavelength grid is maintained. Note that this requires
            equally-spaced wavelength bins. If False, the spectrum is resampled by linearly
            interpolating the extracted spectrum. Only used if ``spec_res`` is not None.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if spec_res is not None and wavel_resample is not None:
            raise ValueError('The \'spec_res\' and \'wavel_resample\' parameters can not be used '
                             'simultaneously. Please set one of them to None.')

        if smooth and wavel_resample is not None:
            warnings.warn('The \'smooth\' parameter is ignored because it can only be used in '
                          'combination with \'spec_res\'.')

        grid_bounds = self.get_bounds()

        for key, value in model_param.items():
            if key not in ['radius', 'distance', 'mass', 'luminosity']:
                if key not in self.get_parameters():
                    raise ValueError(f'The \'{key}\' parameter is not required by \'{self.model}\''
                                     f'. The mandatory parameters are {self.get_parameters()}.')

                if value < grid_bounds[key][0]:
                    raise ValueError(f'The input value of \'{key}\' is smaller than the lower '
                                     f'boundary of the model grid ({value} < '
                                     f'{grid_bounds[key][0]}).')

                if value > grid_bounds[key][1]:
                    raise ValueError(f'The input value of \'{key}\' is larger than the upper '
                                     f'boundary of the model grid ({value} > '
                                     f'{grid_bounds[key][1]}).')

        if 'mass' in model_param:
            mass = 1e3 * model_param['mass'] * constants.M_JUP  # [g]
            radius = math.sqrt(1e3 * constants.GRAVITY * mass / (10.**model_param['logg']))  # [cm]
            model_param['radius'] = 1e-2 * radius / constants.R_JUP  # [Rjup]

        if self.spectrum_interp is None:
            self.interpolate_model()

        if self.wavel_range is None:
            wl_points = self.get_wavelengths()
            self.wavel_range = (wl_points[0], wl_points[-1])

        if self.model in ['ames-cond', 'ames-dusty', 'bt-settl']:
            parameters = [model_param['teff'],
                          model_param['logg']]

        elif self.model in ['bt-nextgen', 'drift-phoenix', 'petitcode-cool-clear']:
            parameters = [model_param['teff'],
                          model_param['logg'],
                          model_param['feh']]

        elif self.model == 'petitcode-cool-cloudy':
            parameters = [model_param['teff'],
                          model_param['logg'],
                          model_param['feh'],
                          model_param['fsed']]

        elif self.model == 'petitcode-hot-clear':
            parameters = [model_param['teff'],
                          model_param['logg'],
                          model_param['feh'],
                          model_param['co']]

        elif self.model == 'petitcode-hot-cloudy':
            parameters = [model_param['teff'],
                          model_param['logg'],
                          model_param['feh'],
                          model_param['co'],
                          model_param['fsed']]

        flux = self.spectrum_interp(parameters)[0]

        if 'radius' in model_param:
            model_param['mass'] = read_util.get_mass(model_param)

            if 'distance' in model_param:
                scaling = (model_param['radius']*constants.R_JUP)**2 / \
                          (model_param['distance']*constants.PARSEC)**2

                flux *= scaling

        if spec_res is None and wavel_resample is not None:
            flux = spectres.spectres(new_spec_wavs=wavel_resample,
                                     old_spec_wavs=self.wl_points,
                                     spec_fluxes=flux)

        elif spec_res is not None:
            index = np.where(np.isnan(flux))[0]

            if index.size > 0:
                raise ValueError('Flux values should not contains NaNs. Please make sure that '
                                 'the parameter values and the wavelength range are within '
                                 'the grid boundaries as stored in the database.')

            if smooth:
                flux = read_util.smooth_spectrum(wavelength=self.wl_points,
                                                 flux=flux,
                                                 spec_res=spec_res,
                                                 size=11)

            else:
                wavel_resample = [self.wl_points[0]]
                while wavel_resample[-1] <= self.wl_points[-1]:
                    wavel_resample.append(wavel_resample[-1] + wavel_resample[-1]/spec_res)

                wavel_resample = np.asarray(wavel_resample[:-1])

                indices = np.where((wavel_resample > self.wl_points[0]) &
                                   (wavel_resample < self.wl_points[-2]))[0]

                for i in range(10):
                    try:
                        index_error = False

                        flux = spectres.spectres(new_spec_wavs=wavel_resample[indices][i:-i],
                                                 old_spec_wavs=self.wl_points,
                                                 spec_fluxes=flux)

                    except ValueError:
                        index_error = True

                    if not index_error:
                        wavel_resample = wavel_resample[indices][i:-i]
                        break

        if magnitude:
            quantity = 'magnitude'

            with h5py.File(self.database, 'r') as h5_file:
                try:
                    h5_file['spectra/calibration/vega']

                except KeyError:
                    h5_file.close()
                    species_db = database.Database()
                    species_db.add_spectrum('vega')
                    h5_file = h5py.File(self.database, 'r')

            readcalib = read_calibration.ReadCalibration('vega', filter_name=None)
            calibbox = readcalib.get_spectrum()

            flux_vega, _ = spectres.spectres(new_spec_wavs=self.wl_points,
                                             old_spec_wavs=calibbox.wavelength,
                                             spec_fluxes=calibbox.flux,
                                             spec_errs=calibbox.error)

            flux = -2.5*np.log10(flux/flux_vega)

        else:
            quantity = 'flux'

        is_finite = np.where(np.isfinite(flux))[0]

        if wavel_resample is None:
            wavelength = self.wl_points[is_finite]
        else:
            wavelength = wavel_resample[is_finite]

        return box.create_box(boxtype='model',
                              model=self.model,
                              wavelength=wavelength,
                              flux=flux[is_finite],
                              parameters=model_param,
                              quantity=quantity)

    def get_data(self,
                 model_param):
        """
        Function for selecting a model spectrum (without interpolation) for a set of parameter
        values that coincide with the grid points. The stored grid points can be inspected with
        :func:`~species.read.read_model.ReadModel.get_points`.

        Parameters
        ----------
        model_param : dict
            Model parameters and values. Only discrete values from the original grid are possible.
            Else, the nearest grid values are selected.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        for key in model_param:
            if key not in ['radius', 'distance', 'mass', 'luminosity']:
                if key not in self.get_parameters():
                    raise ValueError(f'The \'{key}\' parameter is not required by \'{self.model}\''
                                     f'. The mandatory parameters are {self.get_parameters()}.')

        h5_file = self.open_database()

        wl_points, wl_index = self.wavelength_points(h5_file)

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])
        teff = np.asarray(h5_file[f'models/{self.model}/teff'])
        logg = np.asarray(h5_file[f'models/{self.model}/logg'])

        if self.model in ['ames-cond', 'ames-dusty', 'bt-settl']:
            teff_index = np.argwhere(teff == model_param['teff'])[0]

            if len(teff_index) == 0:
                raise ValueError('Temperature value not found.')

            teff_index = teff_index[0]
            logg_index = np.argwhere(logg == model_param['logg'])[0]

            if len(logg_index) == 0:
                raise ValueError('Surface gravity value not found.')

            logg_index = logg_index[0]

            flux = flux[teff_index, logg_index, wl_index]

        elif self.model in ['bt-nextgen', 'drift-phoenix', 'petitcode-cool-clear']:
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])

            teff_index = np.argwhere(teff == model_param['teff'])[0]

            if len(teff_index) == 0:
                raise ValueError('Temperature value not found.')

            teff_index = teff_index[0]
            logg_index = np.argwhere(logg == model_param['logg'])[0]

            if len(logg_index) == 0:
                raise ValueError('Surface gravity value not found.')

            logg_index = logg_index[0]
            feh_index = np.argwhere(feh == model_param['feh'])[0]

            if len(feh_index) == 0:
                raise ValueError('Metallicity value not found.')

            feh_index = feh_index[0]

            flux = flux[teff_index, logg_index, feh_index, wl_index]

        elif self.model == 'petitcode-cool-cloudy':
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])
            fsed = np.asarray(h5_file['models/'+self.model+'/fsed'])

            teff_index = np.argwhere(teff == model_param['teff'])[0]

            if len(teff_index) == 0:
                raise ValueError('Temperature value not found.')

            teff_index = teff_index[0]
            logg_index = np.argwhere(logg == model_param['logg'])[0]

            if len(logg_index) == 0:
                raise ValueError('Surface gravity value not found.')

            logg_index = logg_index[0]
            feh_index = np.argwhere(feh == model_param['feh'])[0]

            if len(feh_index) == 0:
                raise ValueError('Metallicity value not found.')

            feh_index = feh_index[0]
            fsed_index = np.argwhere(fsed == model_param['fsed'])[0]

            if len(fsed_index) == 0:
                raise ValueError('f_sed value not found.')

            fsed_index = fsed_index[0]

            flux = flux[teff_index, logg_index, feh_index, fsed_index, wl_index]

        elif self.model == 'petitcode-hot-clear':
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])
            co_ratio = np.asarray(h5_file['models/'+self.model+'/co'])

            teff_index = np.argwhere(teff == model_param['teff'])[0]

            if len(teff_index) == 0:
                raise ValueError('Temperature value not found.')

            teff_index = teff_index[0]
            logg_index = np.argwhere(logg == model_param['logg'])[0]

            if len(logg_index) == 0:
                raise ValueError('Surface gravity value not found.')

            logg_index = logg_index[0]
            feh_index = np.argwhere(feh == model_param['feh'])[0]

            if len(feh_index) == 0:
                raise ValueError('Metallicity value not found.')

            feh_index = feh_index[0]
            co_index = np.argwhere(co_ratio == model_param['co'])[0]

            if len(co_index) == 0:
                raise ValueError('C/O value not found.')

            co_index = co_index[0]

            flux = flux[teff_index, logg_index, feh_index, co_index, wl_index]

        elif self.model == 'petitcode-hot-cloudy':
            feh = np.asarray(h5_file['models/'+self.model+'/feh'])
            co_ratio = np.asarray(h5_file['models/'+self.model+'/co'])
            fsed = np.asarray(h5_file['models/'+self.model+'/fsed'])

            teff_index = np.argwhere(teff == model_param['teff'])[0]

            if len(teff_index) == 0:
                raise ValueError('Temperature value not found.')

            teff_index = teff_index[0]
            logg_index = np.argwhere(logg == model_param['logg'])[0]

            if len(logg_index) == 0:
                raise ValueError('Surface gravity value not found.')

            logg_index = logg_index[0]
            feh_index = np.argwhere(feh == model_param['feh'])[0]

            if len(feh_index) == 0:
                raise ValueError('Metallicity value not found.')

            feh_index = feh_index[0]
            co_index = np.argwhere(co_ratio == model_param['co'])[0]

            if len(co_index) == 0:
                raise ValueError('C/O value not found.')

            co_index = co_index[0]
            fsed_index = np.argwhere(fsed == model_param['fsed'])[0]

            if len(fsed_index) == 0:
                raise ValueError('f_sed value not found.')

            fsed_index = fsed_index[0]

            flux = flux[teff_index, logg_index, feh_index, co_index, fsed_index, wl_index]

        if 'radius' in model_param and 'distance' in model_param:
            scaling = (model_param['radius']*constants.R_JUP)**2 / \
                      (model_param['distance']*constants.PARSEC)**2

            flux *= scaling

        h5_file.close()

        return box.create_box(boxtype='model',
                              model=self.model,
                              wavelength=wl_points,
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
            Model parameters and values.
        synphot : species.analysis.photometry.SyntheticPhotometry, None
            Synthetic photometry object. The object is created if set to None.

        Returns
        -------
        float
            Average flux density (W m-2 micron-1).
        """

        if self.spectrum_interp is None:
            self.interpolate_model()

        spectrum = self.get_model(model_param, None)

        if synphot is None:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)

    def get_magnitude(self,
                      model_param):
        """
        Function for calculating the apparent and absolute magnitudes for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Model parameters and values.

        Returns
        -------
        float
            Apparent magnitude (mag).
        float
            Absolute magnitude (mag).
        """

        if self.spectrum_interp is None:
            self.interpolate_model()

        try:
            spectrum = self.get_model(model_param, None)
        except ValueError:
            warnings.warn(f'The set of model parameters {model_param} is outside the grid range '
                          f'{self.get_bounds()} so returning a NaN.')

            return np.nan, np.nan

        if spectrum.wavelength.size == 0:
            app_mag = np.nan
            abs_mag = np.nan

        else:
            synphot = photometry.SyntheticPhotometry(self.filter_name)

            if 'distance' in model_param:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength, spectrum.flux, distance=model_param['distance'])

            else:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength, spectrum.flux, distance=None)

        return app_mag, abs_mag

    def get_bounds(self):
        """
        Function for extracting the grid boundaries.

        Returns
        -------
        dict
            Boundaries of parameter grid.
        """

        h5_file = self.open_database()

        teff = h5_file['models/'+self.model+'/teff']
        logg = h5_file['models/'+self.model+'/logg']

        if self.model in ['ames-cond', 'ames-dusty', 'bt-settl']:
            bounds = {'teff': (teff[0], teff[-1]),
                      'logg': (logg[0], logg[-1])}

        elif self.model in ['drift-phoenix', 'bt-nextgen', 'petitcode-cool-clear']:
            feh = h5_file['models/'+self.model+'/feh']

            bounds = {'teff': (teff[0], teff[-1]),
                      'logg': (logg[0], logg[-1]),
                      'feh': (feh[0], feh[-1])}

        elif self.model in ['petitcode-cool-cloudy', ]:
            feh = h5_file['models/'+self.model+'/feh']
            fsed = h5_file['models/'+self.model+'/fsed']

            bounds = {'teff': (teff[0], teff[-1]),
                      'logg': (logg[0], logg[-1]),
                      'feh': (feh[0], feh[-1]),
                      'fsed': (fsed[0], fsed[-1])}

        elif self.model in ['petitcode-hot-clear', ]:
            feh = h5_file['models/'+self.model+'/feh']
            co_ratio = h5_file['models/'+self.model+'/co']

            bounds = {'teff': (teff[0], teff[-1]),
                      'logg': (logg[0], logg[-1]),
                      'feh': (feh[0], feh[-1]),
                      'co': (co_ratio[0], co_ratio[-1])}

        elif self.model in ['petitcode-hot-cloudy', ]:
            feh = h5_file['models/'+self.model+'/feh']
            co_ratio = h5_file['models/'+self.model+'/co']
            fsed = h5_file['models/'+self.model+'/fsed']

            bounds = {'teff': (teff[0], teff[-1]),
                      'logg': (logg[0], logg[-1]),
                      'feh': (feh[0], feh[-1]),
                      'co': (co_ratio[0], co_ratio[-1]),
                      'fsed': (fsed[0], fsed[-1])}

        h5_file.close()

        return bounds

    def get_wavelengths(self):
        """
        Function for extracting the wavelength points.

        Returns
        -------
        numpy.ndarray
            Wavelength points (micron).
        """

        h5_file = self.open_database()
        wavelength = np.asarray(h5_file['models/'+self.model+'/wavelength'])
        h5_file.close()

        return wavelength

    def get_points(self):
        """
        Function for extracting the grid points.

        Returns
        -------
        dict
            Parameter points of the model grid.
        """

        points = {}

        h5_file = self.open_database()

        teff = h5_file[f'models/{self.model}/teff']
        logg = h5_file[f'models/{self.model}/logg']

        points['teff'] = np.asarray(teff)
        points['logg'] = np.asarray(logg)

        feh_models = ['drift-phoenix',
                      'bt-nextgen',
                      'petitcode-cool-clear',
                      'petitcode-cool-cloudy',
                      'petitcode-hot-clear',
                      'petitcode-hot-cloudy']

        if self.model in feh_models:
            feh = h5_file[f'models/{self.model}/feh']
            points['feh'] = np.asarray(feh)

        if self.model in ['petitcode-cool-cloudy', 'petitcode-hot-cloudy']:
            fsed = h5_file[f'models/{self.model}/fsed']
            points['fsed'] = np.asarray(fsed)

        if self.model in ['petitcode-hot-clear', 'petitcode-hot-cloudy']:
            co_ratio = h5_file[f'models/{self.model}/co']
            points['co'] = np.asarray(co_ratio)

        h5_file.close()

        return points

    def get_parameters(self):
        """
        Function for extracting the parameter names.

        Returns
        -------
        list(str, )
            Model parameters.
        """

        h5_file = self.open_database()

        dset = h5_file['models/'+self.model]
        nparam = dset.attrs['nparam']

        param = []
        for i in range(nparam):
            param.append(dset.attrs['parameter'+str(i)])

        h5_file.close()

        return param
