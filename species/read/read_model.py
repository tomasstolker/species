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
            Wavelength range (um). Full spectrum is selected if set to None. Not used if
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
            h5_file[f'models/{self.model}']
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
            Wavelength points (um).
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
        Internal function for linearly interpolating the full grid of model spectra.

        Returns
        -------
        NoneType
            None
        """

        h5_file = self.open_database()

        points = []
        for item in self.get_points().values():
            points.append(item)

        if self.wl_points is None:
            self.wl_points, self.wl_index = self.wavelength_points(h5_file)

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])
        flux = flux[..., self.wl_index]

        self.spectrum_interp = RegularGridInterpolator(points=points,
                                                       values=flux,
                                                       method='linear',
                                                       bounds_error=False,
                                                       fill_value=np.nan)

    def interpolate_grid(self,
                         bounds,
                         wavel_resample=None,
                         smooth=False,
                         spec_res=None):
        """
        Internal function for linearly interpolating the grid of model spectra for a given
        filter or wavelength sampling, and grid boundaries.

        bounds : dict
            Dictionary with the parameter boundaries.
        wavel_resample : numpy.array, None
            Wavelength points for the resampling of the spectrum. The ``filter_name`` is used
            if set to None.
        smooth : bool
            Smooth the spectrum with a Gaussian line spread function. Only recommended in case the
            input wavelength sampling has a uniform spectral resolution.
        spec_res : float
            Spectral resolution that is used for the Gaussian filter when ``smooth=True``.

        Returns
        -------
        NoneType
            None
        """

        self.interpolate_model()

        if smooth and wavel_resample is None:
            raise ValueError('Smoothing is only required if the spectra are resampled to a new '
                             'wavelength grid, therefore requiring the \'wavel_resample\' '
                             'argument.')

        points = []
        for key, value in self.get_points().items():
            value_new = []

            for i, item in enumerate(value):
                if bounds[key][0] <= item <= bounds[key][-1]:
                    value_new.append(item)

            points.append(value_new)

        param = self.get_parameters()
        n_param = len(param)

        dim_size = []
        for item in points:
            dim_size.append(len(item))

        if self.filter_name is not None:
            dim_size.append(1)
        else:
            dim_size.append(wavel_resample.size)

        flux_new = np.zeros(dim_size)

        if n_param == 2:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    model_param[param[0]] = item_0
                    model_param[param[1]] = item_1

                    if self.filter_name is not None:
                        flux_new[i, j] = self.get_flux(model_param)[0]
                    else:
                        flux_new[i, j, :] = self.get_model(model_param,
                                                           spec_res=spec_res,
                                                           wavel_resample=wavel_resample,
                                                           smooth=smooth).flux

        elif n_param == 3:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        model_param[param[0]] = item_0
                        model_param[param[1]] = item_1
                        model_param[param[2]] = item_2

                        if self.filter_name is not None:
                            flux_new[i, j, k] = self.get_flux(model_param)[0]
                        else:
                            flux_new[i, j, k, :] = self.get_model(model_param,
                                                                  spec_res=spec_res,
                                                                  wavel_resample=wavel_resample,
                                                                  smooth=smooth).flux

        elif n_param == 4:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        for l, item_3 in enumerate(points[3]):
                            model_param[param[0]] = item_0
                            model_param[param[1]] = item_1
                            model_param[param[2]] = item_2
                            model_param[param[3]] = item_3

                            if self.filter_name is not None:
                                flux_new[i, j, k, l] = self.get_flux(model_param)[0]
                            else:
                                flux_new[i, j, k, l, :] = self.get_model(
                                    model_param, spec_res=spec_res, wavel_resample=wavel_resample,
                                    smooth=smooth).flux

        elif n_param == 5:
            model_param = {}

            for i, item_0 in enumerate(points[0]):
                for j, item_1 in enumerate(points[1]):
                    for k, item_2 in enumerate(points[2]):
                        for l, item_3 in enumerate(points[3]):
                            for m, item_4 in enumerate(points[4]):
                                model_param[param[0]] = item_0
                                model_param[param[1]] = item_1
                                model_param[param[2]] = item_2
                                model_param[param[3]] = item_3
                                model_param[param[4]] = item_4

                                if self.filter_name is not None:
                                    flux_new[i, j, k, l, m] = self.get_flux(model_param)[0]
                                else:
                                    flux_new[i, j, k, l, m, :] = self.get_model(
                                        model_param, spec_res=spec_res,
                                        wavel_resample=wavel_resample, smooth=smooth).flux

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wl_points = [transmission.mean_wavelength()]

        else:
            self.wl_points = wavel_resample

        self.spectrum_interp = RegularGridInterpolator(points=points,
                                                       values=flux_new,
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
        Function for extracting a model spectrum by linearly interpolating the model grid.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values should be within the
            boundaries of the grid. The grid boundaries of the spectra in the database can be
            obtained with :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum with a Gaussian kernel
            when ``smooth=True`` and/or resampling the spectrum when ``wavel_range`` of ``FitModel``
            is not None. The original wavelength points are used if both ``spec_res`` and
            ``wavel_resample`` are set to None.
        wavel_resample : numpy.ndarray, None
            Wavelength points (um) to which the spectrum is resampled. In that case, ``spec_res``
            can still be used to smooth the spectrum.
        magnitude : bool
            Normalize the spectrum with a flux calibrated spectrum of Vega and return the magnitude
            instead of flux density.
        smooth : bool
            If True, the spectrum is smoothed with a Gaussian kernel to the spectral resolution of
            ``spec_res``. This requires either a uniform spectral resolution of the input spectra
            (fast) or a uniform wavelength spacing of the input spectra (slow).

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if smooth and spec_res is None:
            warnings.warn('The \'spec_res\' argument is required for smoothing the spectrum when '
                          '\'smooth\' is set to True.')

        grid_bounds = self.get_bounds()

        extra_param = ['radius', 'distance', 'mass', 'luminosity']

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

            if model_param[key] < grid_bounds[key][0]:
                raise ValueError(f'The input value of \'{key}\' is smaller than the lower '
                                 f'boundary of the model grid ({model_param[key]} < '
                                 f'{grid_bounds[key][0]}).')

            if model_param[key] > grid_bounds[key][1]:
                raise ValueError(f'The input value of \'{key}\' is larger than the upper '
                                 f'boundary of the model grid ({model_param[key]} > '
                                 f'{grid_bounds[key][1]}).')

        for key in model_param.keys():
            if key not in self.get_parameters() and key not in extra_param:
                warnings.warn(f'The \'{key}\' parameter is not required by \'{self.model}\' so '
                              f'the parameter will be ignored. The mandatory parameters are '
                              f'{self.get_parameters()}.')

        if 'mass' in model_param and 'radius' not in model_param:
            mass = 1e3 * model_param['mass'] * constants.M_JUP  # (g)
            radius = math.sqrt(1e3 * constants.GRAVITY * mass / (10.**model_param['logg']))  # (cm)
            model_param['radius'] = 1e-2 * radius / constants.R_JUP  # (Rjup)

        if self.spectrum_interp is None:
            self.interpolate_model()

        if self.wavel_range is None:
            wl_points = self.get_wavelengths()
            self.wavel_range = (wl_points[0], wl_points[-1])

        parameters = []

        if 'teff' in model_param:
            parameters.append(model_param['teff'])

        if 'logg' in model_param:
            parameters.append(model_param['logg'])

        if 'feh' in model_param:
            parameters.append(model_param['feh'])

        if 'co' in model_param:
            parameters.append(model_param['co'])

        if 'fsed' in model_param:
            parameters.append(model_param['fsed'])

        flux = self.spectrum_interp(parameters)[0]

        if 'radius' in model_param:
            model_param['mass'] = read_util.get_mass(model_param)

            if 'distance' in model_param:
                scaling = (model_param['radius']*constants.R_JUP)**2 / \
                          (model_param['distance']*constants.PARSEC)**2

                flux *= scaling

        if smooth:
            flux = read_util.smooth_spectrum(wavelength=self.wl_points,
                                             flux=flux,
                                             spec_res=spec_res)

        if wavel_resample is not None:
            flux = spectres.spectres(wavel_resample,
                                     self.wl_points,
                                     flux)

        elif spec_res is not None:
            index = np.where(np.isnan(flux))[0]

            if index.size > 0:
                raise ValueError('Flux values should not contains NaNs. Please make sure that '
                                 'the parameter values and the wavelength range are within '
                                 'the grid boundaries as stored in the database.')

            wavel_resample = read_util.create_wavelengths(
                (self.wl_points[0], self.wl_points[-1]), spec_res)

            indices = np.where((wavel_resample > self.wl_points[0]) &
                               (wavel_resample < self.wl_points[-2]))[0]

            for i in range(1, 10):
                try:
                    index_error = False

                    flux = spectres.spectres(wavel_resample[indices][i:-i],
                                             self.wl_points,
                                             flux)

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

            if wavel_resample is not None:
                new_spec_wavs = wavel_resample
            else:
                new_spec_wavs = self.wl_points

            flux_vega, _ = spectres.spectres(new_spec_wavs,
                                             calibbox.wavelength,
                                             calibbox.flux,
                                             spec_errs=calibbox.error)

            flux = -2.5*np.log10(flux/flux_vega)

        else:
            quantity = 'flux'

        is_finite = np.where(np.isfinite(flux))[0]

        if wavel_resample is None:
            wavelength = self.wl_points[is_finite]
        else:
            wavelength = wavel_resample[is_finite]

        if wavelength.shape[0] == 0:
            raise ValueError(f'The model spectrum is empty. Perhaps the grid could not be '
                             f'interpolated at {model_param} because zeros are stored in the '
                             f'database.')

        model_box = box.create_box(boxtype='model',
                                   model=self.model,
                                   wavelength=wavelength,
                                   flux=flux[is_finite],
                                   parameters=model_param,
                                   quantity=quantity)

        if 'radius' in model_box.parameters:
            model_box.parameters['luminosity'] = 4. * np.pi * (
                model_box.parameters['radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['teff']**4. / constants.L_SUN  # (Lsun)

        return model_box

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

        for key in self.get_parameters():
            if key not in model_param.keys():
                raise ValueError(f'The \'{key}\' parameter is required by \'{self.model}\'. '
                                 f'The mandatory parameters are {self.get_parameters()}.')

        extra_param = ['radius', 'distance', 'mass', 'luminosity']

        for key in model_param.keys():
            if key not in self.get_parameters() and key not in extra_param:
                warnings.warn(f'The \'{key}\' parameter is not required by \'{self.model}\' so '
                              f'the parameter will be ignored. The mandatory parameters are '
                              f'{self.get_parameters()}.')

        h5_file = self.open_database()

        param_key = []
        param_val = []

        if 'teff' in model_param:
            param_key.append('teff')
            param_val.append(model_param['teff'])

        if 'logg' in model_param:
            param_key.append('logg')
            param_val.append(model_param['logg'])

        if 'feh' in model_param:
            param_key.append('feh')
            param_val.append(model_param['feh'])

        if 'co' in model_param:
            param_key.append('co')
            param_val.append(model_param['co'])

        if 'fsed' in model_param:
            param_key.append('fsed')
            param_val.append(model_param['fsed'])

        flux = np.asarray(h5_file[f'models/{self.model}/flux'])

        indices = []

        for item in param_key:
            data = np.asarray(h5_file[f'models/{self.model}/{item}'])
            data_index = np.argwhere(data == model_param[item])[0]

            if len(data_index) == 0:
                raise ValueError('The parameter {item}={model_val[i]} is not found.')

            indices.append(data_index[0])

        wl_points, wl_index = self.wavelength_points(h5_file)

        indices.append(wl_index)

        flux = flux[tuple(indices)]

        if 'radius' in model_param and 'distance' in model_param:
            scaling = (model_param['radius']*constants.R_JUP)**2 / \
                      (model_param['distance']*constants.PARSEC)**2

            flux *= scaling

        h5_file.close()

        model_box = box.create_box(boxtype='model',
                                   model=self.model,
                                   wavelength=wl_points,
                                   flux=flux,
                                   parameters=model_param,
                                   quantity='flux')

        if 'radius' in model_box.parameters:
            model_box.parameters['luminosity'] = 4. * np.pi * (
                model_box.parameters['radius'] * constants.R_JUP)**2 * constants.SIGMA_SB * \
                model_box.parameters['teff']**4. / constants.L_SUN  # (Lsun)

        return model_box

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
            Average flux (W m-2 um-1).
        float, None
            Uncertainty (W m-2 um-1), which is set to None.
        """

        if self.spectrum_interp is None:
            self.interpolate_model()

        spectrum = self.get_model(model_param)

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
            spectrum = self.get_model(model_param)
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
                    spectrum.wavelength, spectrum.flux, distance=(model_param['distance'], None))

            else:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength, spectrum.flux, distance=None)

        return app_mag[0], abs_mag[0]

    def get_bounds(self):
        """
        Function for extracting the grid boundaries.

        Returns
        -------
        dict
            Boundaries of parameter grid.
        """

        h5_file = self.open_database()

        parameters = self.get_parameters()

        bounds = {}

        for item in parameters:
            data = h5_file[f'models/{self.model}/{item}']
            bounds[item] = (data[0], data[-1])

        h5_file.close()

        return bounds

    def get_wavelengths(self):
        """
        Function for extracting the wavelength points.

        Returns
        -------
        numpy.ndarray
            Wavelength points (um).
        """

        with self.open_database() as h5_file:
            wavelength = np.asarray(h5_file[f'models/{self.model}/wavelength'])

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

        parameters = self.get_parameters()

        points = {}

        for item in parameters:
            data = h5_file[f'models/{self.model}/{item}']
            points[item] = np.asarray(data)

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

        dset = h5_file[f'models/{self.model}']

        if 'n_param' in dset.attrs:
            n_param = dset.attrs['n_param']
        elif 'nparam' in dset.attrs:
            n_param = dset.attrs['nparam']

        param = []
        for i in range(n_param):
            param.append(dset.attrs[f'parameter{i}'])

        h5_file.close()

        return param
