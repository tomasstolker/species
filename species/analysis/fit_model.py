"""
Module with functionalities for fitting atmospheric model spectra.
"""

import os
import math
import warnings

from typing import Optional, Union, List, Tuple, Dict
from multiprocessing import Pool, cpu_count

import emcee
import spectres
import numpy as np

# Installation of MultiNest is not possible on readthedocs
try:
    import pymultinest
except:
    warnings.warn('PyMultiNest could not be imported.')

from typeguard import typechecked

from species.analysis import photometry
from species.data import database
from species.core import constants
from species.read import read_model, read_object, read_planck, read_filter
from species.util import read_util, dust_util


@typechecked
def lnprior(param: np.ndarray,
            bounds: dict,
            param_index: Dict[str, int],
            prior: Optional[Dict[str, Tuple[float, float]]] = None):
    """
    Internal function for calculating the log prior.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Dictionary with the parameter boundaries.
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    prior : dict(str, tuple(float, float)), None
        Dictionary with Gaussian priors for one or multiple parameters. The prior can be set
        for any of the atmosphere or calibration parameters, e.g.
        ``prior={'teff': (1200., 100.)}``. Additionally, a prior can be set for the mass, e.g.
        ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup with an uncertainty of
        3 Mjup. The parameter is not used if set to ``None``.

    Returns
    -------
    floats
        Log prior.
    """

    ln_prior = 0

    for key, value in bounds.items():

        if value[0] <= param[param_index[key]] <= value[1]:
            ln_prior += 0.

        else:
            ln_prior = -np.inf
            break

    if prior is not None:
        for key, value in prior.items():
            if key == 'mass':
                mass = read_util.get_mass({'logg': param[param_index['logg']],
                                           'radius': param[param_index['radius']]})

                ln_prior += -0.5 * (mass - value[0])**2 / value[1]**2

            else:
                ln_prior += -0.5 * (param[param_index[key]] - value[0])**2 / value[1]**2

    return ln_prior


@typechecked
def lnlike(param: np.ndarray,
           bounds: dict,
           param_index: Dict[str, int],
           model: str,
           objphot: List[Optional[np.ndarray]],
           distance: Tuple[float, float],
           spectrum: Optional[dict],
           modelphot: Optional[Union[List[read_model.ReadModel],
                                     List[photometry.SyntheticPhotometry]]],
           modelspec: Optional[List[read_model.ReadModel]],
           n_planck: int,
           fit_corr: List[str]):
    """
    Internal function for calculating the log likelihood.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Dictionary with the parameter boundaries.
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    model : str
        Atmosphere model (e.g. 'bt-settl', 'exo-rem', or 'planck).
    objphot : list(np.ndarray, )
        List with the photometric fluxes and uncertainties of the object. Not photometric data
        is fitted if an empty list is provided.
    distance : tuple(float, float)
        Distance and uncertainty (pc).
    spectrum : dict(str, tuple(np.ndarray, np.ndarray, np.ndarray, float)), None
        Dictionary with the spectra stored as wavelength (um), flux (W m-2 um-1),
        and error (W m-2 um-1). Optionally the covariance matrix, the inverse of
        the covariance matrix, and the spectral resolution are included. Each
        of these three elements can be set to ``None``. No spectroscopic data is
        fitted if ``spectrum=None``.
    modelphot : list(species.read.read_model.ReadModel, ),
                list(species.analysis.photometry.SyntheticPhotometry, ), None
        List with the interpolated synthetic fluxes or list with the
        :class:`~species.analysis.photometry.SyntheticPhotometry` objects for
        calculation of synthetic photometry for Planck spectra. No photometry is fitted if set
        to ``None``.
    modelspec : list(species.read.read_model.ReadModel, ), None
        List with the interpolated synthetic spectra.
    n_planck : int
        Number of Planck components. The argument is set to zero if ``model`` is not ``'planck'``.
    fit_corr : list(str)
        List with spectrum names for which the correlation length and fractional amplitude are
        fitted (see Wang et al. 2020).

    Returns
    -------
    float
        Log likelihood.
    """

    param_dict = {}
    spec_scaling = {}
    err_offset = {}
    corr_len = {}
    corr_amp = {}

    for item in bounds:
        if item[:8] == 'scaling_' and item[8:] in spectrum:
            spec_scaling[item[8:]] = param[param_index[item]]

        elif item[:6] == 'error_' and item[6:] in spectrum:
            err_offset[item[6:]] = param[param_index[item]]

        elif item[:9] == 'corr_len_' and item[9:] in spectrum:
            corr_len[item[9:]] = 10.**param[param_index[item]]  # (um)

        elif item[:9] == 'corr_amp_' and item[9:] in spectrum:
            corr_amp[item[9:]] = param[param_index[item]]

        else:
            param_dict[item] = param[param_index[item]]

    if model == 'planck':
        param_dict['distance'] = distance[0]

    else:
        flux_scaling = (param_dict['radius']*constants.R_JUP)**2 / \
                       (distance[0]*constants.PARSEC)**2

        # The scaling is applied manually because of the interpolation
        del param_dict['radius']

    for item in spectrum:
        if item not in spec_scaling:
            spec_scaling[item] = 1.

        if item not in err_offset:
            err_offset[item] = None

    ln_like = 0.

    if model == 'planck' and n_planck > 1:
        for i in range(n_planck-1):
            if param_dict[f'teff_{i+1}'] > param_dict[f'teff_{i}']:
                return -np.inf

            if param_dict[f'radius_{i}'] > param_dict[f'radius_{i+1}']:
                return -np.inf

    for i, obj_item in enumerate(objphot):
        if model == 'planck':
            readplanck = read_planck.ReadPlanck(filter_name=modelphot[i].filter_name)
            phot_flux = readplanck.get_flux(param_dict, synphot=modelphot[i])[0]

        else:
            phot_flux = modelphot[i].spectrum_interp(list(param_dict.values()))
            phot_flux *= flux_scaling

        if obj_item.ndim == 1:
            ln_like += -0.5 * (obj_item[0] - phot_flux)**2 / obj_item[1]**2

        else:
            for j in range(obj_item.shape[1]):
                ln_like += -0.5 * (obj_item[0, j] - phot_flux)**2 / obj_item[1, j]**2

    for i, item in enumerate(spectrum.keys()):
        data_flux = spec_scaling[item]*spectrum[item][0][:, 1]
        data_var = spectrum[item][0][:, 2]**2

        if err_offset[item] is not None:
            data_var += (10.**err_offset[item])**2

        if spectrum[item][2] is not None:
            if err_offset[item] is None:
                data_cov_inv = spectrum[item][2]

            else:
                # Ratio of the inflated and original uncertainties
                sigma_ratio = np.sqrt(data_var) / spectrum[item][0][:, 2]
                sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                # Calculate the inversion of the infalted covariances
                data_cov_inv = np.linalg.inv(spectrum[item][1]*sigma_i*sigma_j)

        if model == 'planck':
            readplanck = read_planck.ReadPlanck((0.9*spectrum[item][0][0, 0],
                                                 1.1*spectrum[item][0][-1, 0]))

            model_box = readplanck.get_spectrum(param_dict, 1000., smooth=True)

            model_flux = spectres.spectres(spectrum[item][0][:, 0],
                                           model_box.wavelength,
                                           model_box.flux)

        else:
            model_flux = modelspec[i].spectrum_interp(list(param_dict.values()))[0, :]
            model_flux *= flux_scaling

        if spectrum[item][2] is not None:
            dot_tmp = np.dot(data_flux-model_flux, np.dot(data_cov_inv, data_flux-model_flux))
            ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

        else:
            if item in fit_corr:
                # Covariance model (Wang et al. 2020)
                wavel = spectrum[item][0][:, 0]  # (um)
                wavel_j, wavel_i = np.meshgrid(wavel, wavel)

                error = np.sqrt(data_var)  # (W m-2 um-1)
                error_j, error_i = np.meshgrid(error, error)

                cov_matrix = corr_amp[item]**2 * error_i * error_j * \
                    np.exp(-(wavel_i-wavel_j)**2 / (2.*corr_len[item]**2)) + \
                    (1.-corr_amp[item]**2) * np.eye(wavel.shape[0])*error_i**2

                dot_tmp = np.dot(data_flux-model_flux,
                                 np.dot(np.linalg.inv(cov_matrix), data_flux-model_flux))

                ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

            else:
                ln_like += np.nansum(-0.5 * (data_flux-model_flux)**2 / data_var -
                                     0.5 * np.log(2.*np.pi*data_var))

    return ln_like


@typechecked
def lnprob(param: np.ndarray,
           bounds: dict,
           model: str,
           param_index: Dict[str, int],
           objphot: List[Optional[np.ndarray]],
           distance: Tuple[float, float],
           prior: Optional[Dict[str, Tuple[float, float]]],
           spectrum: dict,
           modelphot: Optional[Union[List[read_model.ReadModel],
                                     List[photometry.SyntheticPhotometry]]],
           modelspec: List[read_model.ReadModel],
           n_planck: int,
           fit_corr: List[str]) -> np.float64:
    """
    Internal function for calculating the log posterior.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries.
    model : str
        Atmosphere model (e.g. 'bt-settl', 'exo-rem', or 'planck).
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    objphot : list(np.ndarray, ), None
        List with the photometric fluxes and uncertainties. No photometric data is fitted if the
        parameter is set to ``None``.
    distance : tuple(float, float)
        Distance and uncertainty (pc).
    prior : dict(str, tuple(float, float)), None
        Dictionary with Gaussian priors for one or multiple parameters. The prior can be set
        for any of the atmosphere or calibration parameters, e.g.
        ``prior={'teff': (1200., 100.)}``. Additionally, a prior can be set for the mass, e.g.
        ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup with an uncertainty of
        3 Mjup. The parameter is not used if set to ``None``.
    spectrum : dict(str, tuple(np.ndarray, np.ndarray, np.ndarray, float)), None
        Dictionary with the spectra stored as wavelength (um), flux (W m-2 um-1),
        and error (W m-2 um-1). Optionally the covariance matrix, the inverse of
        the covariance matrix, and the spectral resolution are included. Each
        of these three elements can be set to ``None``. No spectroscopic data is
        fitted if ``spectrum=None``.
    modelphot : list(species.read.read_model.ReadModel, ),
                list(species.analysis.photometry.SyntheticPhotometry, ), None
        List with the interpolated synthetic fluxes or list with the
        :class:`~species.analysis.photometry.SyntheticPhotometry` objects for
        calculation of synthetic photometry for Planck spectra. No photometry is fitted if set
        to ``None``.
    modelspec : list(species.read.read_model.ReadModel, ), None
        List with the interpolated synthetic spectra. The parameter is not used if no spectroscopic
        data is fitted ot if ``model='planck'``.
    n_planck : int
        Number of Planck components. The argument is set to zero if ``model`` is not ``'planck'``.
    fit_corr : list(str)
        List with spectrum names for which the correlation amplitude and length are fitted.

    Returns
    -------
    float
        Log posterior.
    """

    ln_prior = lnprior(param, bounds, param_index, prior)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param, bounds, param_index, model, objphot, distance, spectrum,
                                    modelphot, modelspec, n_planck, fit_corr)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitModel:
    """
    Class for fitting atmospheric model spectra or blackbody spectra to photometric and/or
    spectroscopic data.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 model: str,
                 bounds: Dict[str, Union[Tuple[float, float],
                                         Tuple[Optional[Tuple[float, float]],
                                               Optional[Tuple[float, float]]],
                                         List[Tuple[float, float]]]],
                 inc_phot: Union[bool, List[str]] = True,
                 inc_spec: Union[bool, List[str]] = True,
                 fit_corr: Optional[List[str]] = None) -> None:
        """
        The grid of spectra is linearly interpolated for each photometric point and spectrum while
        taking into account the filter profile, spectral resolution, and wavelength sampling.
        Therefore, when fitting spectra from a model grid, the computation time of the
        interpolation will depend on the wavelength range, spectral resolution, and parameter
        space of the spectra that are stored in the database.

        Parameters
        ----------
        object_name : str
            Object name as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        model : str
            Atmospheric model (e.g. 'bt-settl', 'exo-rem', or 'planck').
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform priors.

            Atmospheric model parameters (e.g. ``model='bt-settl'``):

                 - Boundaries are provided as tuple of two floats. For example,
                   ``bounds={'teff': (1000, 1500.), 'logg': (3.5, 5.), 'radius': (0.8, 1.2)}``.

                 - The grid boundaries are used if set to ``None``. For example,
                   ``bounds={'teff': None, 'logg': None}``. The radius range is set to
                   0.8-1.5 Rjup if the boundary is set to None.

            Blackbody emission parameters (``model='planck'``):

                 - Parameter boundaries have to be provided for 'teff' and 'radius'.

                 - For a single blackbody component, the values are provided as a tuple with two
                   floats. For example, ``bounds={'teff': (1000., 2000.), 'radius': (0.8, 1.2)}``.

                 - For multiple blackbody component, the values are provided as a list with tuples.
                   For example, ``bounds={'teff': [(1000., 1400.), (1200., 1600.)],
                   'radius': [(0.8, 1.5), (1.2, 2.)]}``.

                 - When fitting multiple blackbody components, a prior is used which restricts the
                   temperatures and radii to decreasing and increasing values, respectively, in the
                   order as provided in ``bounds``.

            Calibration parameters:

                 - For each spectrum/instrument, two optional parameters can be fitted to account
                   for biases in the calibration: a scaling of the flux and a constant inflation of
                   the uncertainties.

                 - For example, ``bounds={'SPHERE': ((0.8, 1.2), (-18., -14.))}`` if the scaling is
                   fitted between 0.8 and 1.2, and the error is inflated with a value between 1e-18
                   and 1e-14 W m-2 um-1.

                 - The dictionary key should be equal to the database tag of the spectrum. For
                   example, ``{'SPHERE': ((0.8, 1.2), (-18., -14.))}`` if the spectrum is stored as
                   ``'SPHERE'`` with :func:`~species.data.database.Database.add_object`.

                 - Each of the two calibration parameters can be set to ``None`` in which case the
                   parameter is not used. For example, ``bounds={'SPHERE': ((0.8, 1.2), None)}``.

                 - No calibration parameters are fitted if the spectrum name is not included in
                   ``bounds``.

            ISM extinction parameters:

                 - There are three approaches for fitting extinction. The first is with the
                   empirical relation from Cardelli et al. (1989) for ISM extinction.

                 - The extinction is parametrized by the V band extinction, A_V (``ism_ext``), and
                   the reddening, R_V (``ism_red``).

                 - The prior boundaries of ``ism_ext`` and ``ext_red`` should be provided in the
                   ``bounds`` dictionary, for example ``bounds={'ism_ext': (0., 10.),
                   'ism_red': (0., 20.)}``.

                 - Only supported by ``run_multinest``.

            Log-normal size distribution:

                 - The second approach is fitting the extinction of a log-normal size distribution
                   of grains with a crystalline MgSiO3 composition, and a homogeneous, spherical
                   structure.

                 - The size distribution is parameterized with a mean geometric radius
                   (``lognorm_radius`` in um) and a geometric standard deviation
                   (``lognorm_sigma``, dimensionless).

                 - The extinction (``lognorm_ext``) is fitted in the V band (A_V in mag) and the
                   wavelength-dependent extinction cross sections are interpolated from a
                   pre-tabulated grid.

                 - The prior boundaries of ``lognorm_radius``, ``lognorm_sigma``, and
                   ``lognorm_ext`` should be provided in the ``bounds`` dictionary, for example
                   ``bounds={'lognorm_radius': (0.01, 10.), 'lognorm_sigma': (1.2, 10.),
                   'lognorm_ext': (0., 5.)}``.

                 - A uniform prior is used for ``lognorm_sigma`` and ``lognorm_ext``, and a
                   log-uniform prior for ``lognorm_radius``.

                 - Only supported by ``run_multinest``.

            Power-law size distribution:

                 - The third approach is fitting the extinction of a power-law size distribution
                   of grains, again with a crystalline MgSiO3 composition, and a homogeneous,
                   spherical structure.

                 - The size distribution is parameterized with a maximum radius (``powerlaw_max``
                   in um) and a power-law exponent (``powerlaw_exp``, dimensionless). The
                   minimum radius is fixed to 1 nm.

                 - The extinction (``powerlaw_ext``) is fitted in the V band (A_V in mag) and the
                   wavelength-dependent extinction cross sections are interpolated from a
                   pre-tabulated grid.

                 - The prior boundaries of ``powerlaw_max``, ``powerlaw_exp``, and ``powerlaw_ext``
                   should be provided in the ``bounds`` dictionary, for example ``'powerlaw_max':
                   (0.5, 10.), 'powerlaw_exp': (-5., 5.), 'powerlaw_ext': (0., 5.)}``.

                 - A uniform prior is used for ``powerlaw_exp`` and ``powerlaw_ext``, and a
                   log-uniform prior for ``powerlaw_max``.

                 - Only supported by ``run_multinest``.

        inc_phot : bool, list(str)
            Include photometric data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of filter names (as stored in
            the database) can be provided.
        inc_spec : bool, list(str)
            Include spectroscopic data in the fit. If a boolean, either all (``True``) or none
            (``False``) of the data are selected. If a list, a subset of spectrum names (as stored
            in the database with :func:`~species.data.database.Database.add_object`) can be
            provided.
        fit_corr : list(str), None
            List with spectrum names for which the correlation length and fractional amplitude are
            fitted (see Wang et al. 2020).

        Returns
        -------
        NoneType
            None
        """

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectroscopic data has been selected.')

        if model == 'planck' and 'teff' not in bounds or 'radius' not in bounds:
            raise ValueError('The \'bounds\' dictionary should contain \'teff\' and \'radius\'.')

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()

        if fit_corr is None:
            self.fit_corr = []
        else:
            self.fit_corr = fit_corr

        self.model = model
        self.bounds = bounds

        if self.model == 'planck':
            # Fitting blackbody radiation
            if isinstance(bounds['teff'], list) and isinstance(bounds['radius'], list):
                # Update temperature and radius parameters in case of multiple blackbody components
                self.n_planck = len(bounds['teff'])

                self.modelpar = []
                self.bounds = {}

                for i, item in enumerate(bounds['teff']):
                    self.modelpar.append(f'teff_{i}')
                    self.modelpar.append(f'radius_{i}')

                    self.bounds[f'teff_{i}'] = bounds['teff'][i]
                    self.bounds[f'radius_{i}'] = bounds['radius'][i]

            else:
                # Fitting a single blackbody compoentn
                self.n_planck = 1

                self.modelpar = ['teff', 'radius']
                self.bounds = bounds

        else:
            # Fitting self-consistent atmospheric models
            if self.bounds is not None:
                readmodel = read_model.ReadModel(self.model)
                bounds_grid = readmodel.get_bounds()

                for item in bounds_grid:
                    if item not in self.bounds:
                        # Set the parameter boundaries to the grid boundaries if set to None
                        self.bounds[item] = bounds_grid[item]

            else:
                # Set all parameter boundaries to the grid boundaries
                readmodel = read_model.ReadModel(self.model, None, None)
                self.bounds = readmodel.get_bounds()

            if 'radius' not in self.bounds:
                self.bounds['radius'] = (0.8, 1.5)

            self.n_planck = 0

            self.modelpar = readmodel.get_parameters()
            self.modelpar.append('radius')

        # Select filters and spectra

        if isinstance(inc_phot, bool):
            if inc_phot:
                # Select all filters if True
                species_db = database.Database()
                objectbox = species_db.get_object(object_name)
                inc_phot = objectbox.filters

            else:
                inc_phot = []

        if isinstance(inc_spec, bool):
            if inc_spec:
                # Select all filters if True
                species_db = database.Database()
                objectbox = species_db.get_object(object_name)
                inc_spec = list(objectbox.spectrum.keys())

            else:
                inc_spec = []

        # Include photometric data

        self.objphot = []
        self.modelphot = []

        for item in inc_phot:
            if self.model == 'planck':
                # Create SyntheticPhotometry objects when fitting a Planck function
                print(f'Creating synthetic photometry: {item}...', end='', flush=True)
                self.modelphot.append(photometry.SyntheticPhotometry(item))

            else:
                # Or interpolate the model grid for each filter
                print(f'Interpolating {item}...', end='', flush=True)
                readmodel = read_model.ReadModel(self.model, filter_name=item)
                readmodel.interpolate_grid(wavel_resample=None, smooth=False, spec_res=None)
                self.modelphot.append(readmodel)

            print(' [DONE]')

            # Store the flux and uncertainty for each filter
            obj_phot = self.object.get_photometry(item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

        # Include spectroscopic data

        if inc_spec:
            # Select all spectra
            self.spectrum = self.object.get_spectrum()

            # Select the spectrum names that are not in inc_spec
            spec_remove = []
            for item in self.spectrum:
                if item not in inc_spec:
                    spec_remove.append(item)

            # Remove the spectra that are not included in inc_spec
            for item in spec_remove:
                del self.spectrum[item]

            self.n_corr_par = 0

            for item in self.spectrum:
                if item in self.fit_corr:
                    self.modelpar.append(f'corr_len_{item}')
                    self.modelpar.append(f'corr_amp_{item}')

                    self.bounds[f'corr_len_{item}'] = (-3., 0.)  # log10(corr_len) (um)
                    self.bounds[f'corr_amp_{item}'] = (0., 1.)

                    self.n_corr_par += 2

            self.modelspec = []

            if self.model != 'planck':

                for key, value in self.spectrum.items():
                    print(f'\rInterpolating {key}...', end='', flush=True)

                    wavel_range = (0.9*value[0][0, 0], 1.1*value[0][-1, 0])

                    readmodel = read_model.ReadModel(self.model, wavel_range=wavel_range)

                    readmodel.interpolate_grid(wavel_resample=self.spectrum[key][0][:, 0],
                                               smooth=True,
                                               spec_res=self.spectrum[key][3])

                    self.modelspec.append(readmodel)

                    print(' [DONE]')

        else:
            self.spectrum = {}
            self.modelspec = None
            self.n_corr_par = 0

        for item in self.spectrum:
            if item in bounds:

                if bounds[item][0] is not None:
                    # Add the flux scaling parameter
                    self.modelpar.append(f'scaling_{item}')
                    self.bounds[f'scaling_{item}'] = (bounds[item][0][0], bounds[item][0][1])

                if bounds[item][1] is not None:
                    # Add the error offset parameters
                    self.modelpar.append(f'error_{item}')
                    self.bounds[f'error_{item}'] = (bounds[item][1][0], bounds[item][1][1])

                if item in self.bounds:
                    del self.bounds[item]

        if 'lognorm_radius' in self.bounds and 'lognorm_sigma' in self.bounds and \
                'lognorm_ext' in self.bounds:

            self.cross_sections, _, _ = dust_util.interp_lognorm(inc_phot, inc_spec, self.spectrum)

            self.modelpar.append('lognorm_radius')
            self.modelpar.append('lognorm_sigma')
            self.modelpar.append('lognorm_ext')

            self.bounds['lognorm_radius'] = (np.log10(self.bounds['lognorm_radius'][0]),
                                             np.log10(self.bounds['lognorm_radius'][1]))

        elif 'powerlaw_max' in self.bounds and 'powerlaw_exp' in self.bounds and \
                'powerlaw_ext' in self.bounds:

            self.cross_sections, _, _ = dust_util.interp_powerlaw(
                inc_phot, inc_spec, self.spectrum)

            self.modelpar.append('powerlaw_max')
            self.modelpar.append('powerlaw_exp')
            self.modelpar.append('powerlaw_ext')

            self.bounds['powerlaw_max'] = (np.log10(self.bounds['powerlaw_max'][0]),
                                           np.log10(self.bounds['powerlaw_max'][1]))

        else:
            self.cross_sections = None

        if 'ism_ext' in self.bounds and 'ism_red' in self.bounds:
            self.modelpar.append('ism_ext')
            self.modelpar.append('ism_red')

        print(f'Fitting {len(self.modelpar)} parameters:')

        for item in self.modelpar:
            print(f'   - {item}')

        print('Prior boundaries:')

        for key, value in self.bounds.items():
            print(f'   - {key} = {value}')

    @typechecked
    def run_mcmc(self,
                 tag: str,
                 guess: Optional[Dict[str, Union[Optional[float],
                                                 List[Optional[float]],
                                                 Tuple[Optional[float],
                                                       Optional[float]]]]],
                 nwalkers: int = 200,
                 nsteps: int = 1000,
                 prior: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Function to run the MCMC sampler of ``emcee``.

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        guess : dict, None
            Guess for each parameter to initialize the walkers. Random values between the
            ``bounds`` are used is set to ``None``.
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        prior : dict(str, tuple(float, float)), None
            Dictionary with Gaussian priors for one or multiple parameters. The prior can be set
            for any of the atmosphere or calibration parameters, e.g.
            ``prior={'teff': (1200., 100.)}``. Additionally, a prior can be set for the mass, e.g.
            ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup with an uncertainty of
            3 Mjup. The parameter is not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print('Running MCMC...')

        ndim = len(self.bounds)

        if self.model == 'planck':

            if 'teff' in self.bounds:
                sigma = {'teff': 5., 'radius': 0.01}

            else:
                sigma = {}

                for i, item in enumerate(guess['teff']):
                    sigma[f'teff_{i}'] = 5.
                    guess[f'teff_{i}'] = guess['teff'][i]

                for i, item in enumerate(guess['radius']):
                    sigma[f'radius_{i}'] = 0.01
                    guess[f'radius_{i}'] = guess['radius'][i]

                del guess['teff']
                del guess['radius']

        else:
            sigma = {'teff': 5.,
                     'logg': 0.01,
                     'feh': 0.01,
                     'fsed': 0.01,
                     'co': 0.01,
                     'radius': 0.01}

        for item in self.spectrum:
            if item in self.fit_corr:
                sigma[f'corr_len_{item}'] = 0.01  # (dex)
                guess[f'corr_len_{item}'] = None  # (um)

                sigma[f'corr_amp_{item}'] = 0.1
                guess[f'corr_amp_{item}'] = None

        for item in self.spectrum:
            if f'scaling_{item}' in self.bounds:
                sigma[f'scaling_{item}'] = 0.01
                guess[f'scaling_{item}'] = guess[item][0]

            if f'error_{item}' in self.bounds:
                sigma[f'error_{item}'] = 0.1  # (dex)
                guess[f'error_{item}'] = guess[item][1]  # (dex)

            if item in guess:
                del guess[item]

        initial = np.zeros((nwalkers, ndim))

        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                initial[:, i] = guess[item] + np.random.normal(0, sigma[item], nwalkers)

            else:
                initial[:, i] = np.random.uniform(low=self.bounds[item][0],
                                                  high=self.bounds[item][1],
                                                  size=nwalkers)

        # Create a dictionary with the indices of the parameters

        param_index = {}
        for i, item in enumerate(self.modelpar):
            param_index[item] = i

        with Pool(processes=cpu_count()):

            ens_sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=([self.bounds,
                                                       self.model,
                                                       param_index,
                                                       self.objphot,
                                                       self.distance,
                                                       prior,
                                                       self.spectrum,
                                                       self.modelphot,
                                                       self.modelspec,
                                                       self.n_planck,
                                                       self.fit_corr]))

            ens_sampler.run_mcmc(initial, nsteps, progress=True)

        spec_labels = []
        for item in self.spectrum:
            if f'scaling_{item}' in self.bounds:
                spec_labels.append(f'scaling_{item}')

        species_db = database.Database()

        species_db.add_samples(sampler='emcee',
                               samples=ens_sampler.chain,
                               ln_prob=ens_sampler.lnprobability,
                               mean_accept=np.mean(ens_sampler.acceptance_fraction),
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance[0],
                               spec_labels=spec_labels)

    @typechecked
    def run_multinest(self,
                      tag: str,
                      n_live_points: int = 1000,
                      output: str = 'multinest/',
                      prior: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the ``MultiNest`` sampler. While
        ``PyMultiNest`` can be installed with ``pip`` from the PyPI repository, ``MultiNest``
        has to to be build manually. See the ``PyMultiNest`` documentation for details:
        http://johannesbuchner.github.io/PyMultiNest/install.html. Note that the library path
        of ``MultiNest`` should be set to the environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac. Alternatively, the variable can be
        set before importing the ``species`` package, for example:

        .. code-block:: python

            >>> import os
            >>> os.environ['DYLD_LIBRARY_PATH'] = '/path/to/MultiNest/lib'
            >>> import species

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        n_live_points : int
            Number of live points.
        output : str
            Path that is used for the output files from MultiNest.
        prior : dict(str, tuple(float, float)), None
            Dictionary with Gaussian priors for one or multiple parameters. The prior can be set
            for any of the atmosphere or calibration parameters, e.g.
            ``prior={'teff': (1200., 100.)}``. Additionally, a prior can be set for the mass, e.g.
            ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup with an uncertainty of
            3 Mjup. The parameter is not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print('Running nested sampling...')

        # Create the output folder if required

        if not os.path.exists(output):
            os.mkdir(output)

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(self.modelpar):
            cube_index[item] = i

        @typechecked
        def lnprior_multinest(cube,
                              n_dim: int,
                              n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter cube. It is not clear how to
            pass additional arguments to the function, therefore it is placed here and not merged
            with :func:`~species.analysis.fit_model.FitModel.run_mcmc`.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            NoneType
                None
            """

            for item in cube_index:
                # Uniform priors for all parameters
                cube[cube_index[item]] = self.bounds[item][0] + \
                    (self.bounds[item][1]-self.bounds[item][0])*cube[cube_index[item]]

        @typechecked
        def lnlike_multinest(cube,
                             n_dim: int,
                             n_param: int) -> np.float64:
            """
            Function for the logarithm of the likelihood, computed from the parameter cube.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            float
                Log likelihood.
            """

            param_dict = {}
            spec_scaling = {}
            err_offset = {}
            corr_len = {}
            corr_amp = {}
            dust_param = {}

            for item in self.bounds:
                if item[:8] == 'scaling_' and item[8:] in self.spectrum:
                    spec_scaling[item[8:]] = cube[cube_index[item]]

                elif item[:6] == 'error_' and item[6:] in self.spectrum:
                    err_offset[item[6:]] = cube[cube_index[item]]  # log10(um)

                elif item[:9] == 'corr_len_' and item[9:] in self.spectrum:
                    corr_len[item[9:]] = 10.**cube[cube_index[item]]  # (um)

                elif item[:9] == 'corr_amp_' and item[9:] in self.spectrum:
                    corr_amp[item[9:]] = cube[cube_index[item]]

                elif item[:8] == 'lognorm_':
                    dust_param[item] = cube[cube_index[item]]

                elif item[:9] == 'powerlaw_':
                    dust_param[item] = cube[cube_index[item]]

                elif item[:4] == 'ism_':
                    dust_param[item] = cube[cube_index[item]]

                else:
                    param_dict[item] = cube[cube_index[item]]

            if self.model == 'planck':
                param_dict['distance'] = self.distance[0]

            else:
                flux_scaling = (param_dict['radius']*constants.R_JUP)**2 / \
                               (self.distance[0]*constants.PARSEC)**2

                # The scaling is applied manually because of the interpolation
                del param_dict['radius']

            for item in self.spectrum:
                if item not in spec_scaling:
                    spec_scaling[item] = 1.

                if item not in err_offset:
                    err_offset[item] = None

            ln_like = 0.

            if self.model == 'planck' and self.n_planck > 1:
                for i in range(self.n_planck-1):
                    if param_dict[f'teff_{i+1}'] > param_dict[f'teff_{i}']:
                        return -np.inf

                    if param_dict[f'radius_{i}'] > param_dict[f'radius_{i+1}']:
                        return -np.inf

            if prior is not None:
                for key, value in prior.items():
                    if key == 'mass':
                        mass = read_util.get_mass({'logg': cube[cube_index['logg']],
                                                   'radius': cube[cube_index['radius']]})

                        ln_like += -0.5 * (mass - value[0])**2 / value[1]**2

                    else:
                        ln_like += -0.5 * (cube[cube_index[key]] - value[0])**2 / value[1]**2

            if 'lognorm_ext' in dust_param:
                cross_tmp = self.cross_sections['Generic/Bessell.V'](
                    dust_param['lognorm_sigma'], 10.**dust_param['lognorm_radius'])[0]

                n_grains = dust_param['lognorm_ext'] / cross_tmp / 2.5 / np.log10(np.exp(1.))

            elif 'powerlaw_ext' in dust_param:
                cross_tmp = self.cross_sections['Generic/Bessell.V'](
                    dust_param['powerlaw_exp'], 10.**dust_param['powerlaw_max'])

                n_grains = dust_param['powerlaw_ext'] / cross_tmp / 2.5 / np.log10(np.exp(1.))

            for i, obj_item in enumerate(self.objphot):
                if self.model == 'planck':
                    readplanck = read_planck.ReadPlanck(filter_name=self.modelphot[i].filter_name)
                    phot_flux = readplanck.get_flux(param_dict, synphot=self.modelphot[i])[0]

                else:
                    phot_flux = self.modelphot[i].spectrum_interp(list(param_dict.values()))[0][0]
                    phot_flux *= flux_scaling

                if 'lognorm_ext' in dust_param:
                    cross_tmp = self.cross_sections[self.modelphot[i].filter_name](
                        dust_param['lognorm_sigma'], 10.**dust_param['lognorm_radius'])[0]

                    phot_flux *= np.exp(-cross_tmp*n_grains)

                elif 'powerlaw_ext' in dust_param:
                    cross_tmp = self.cross_sections[self.modelphot[i].filter_name](
                        dust_param['powerlaw_exp'], 10.**dust_param['powerlaw_max'])[0]

                    phot_flux *= np.exp(-cross_tmp*n_grains)

                elif 'ism_ext' in dust_param:
                    read_filt = read_filter.ReadFilter(self.modelphot[i].filter_name)
                    filt_wavel = np.array([read_filt.mean_wavelength()])

                    ext_filt = dust_util.ism_extinction(dust_param['ism_ext'],
                                                        dust_param['ism_red'],
                                                        filt_wavel)

                    phot_flux *= 10.**(-0.4*ext_filt[0])

                if obj_item.ndim == 1:
                    ln_like += -0.5 * (obj_item[0] - phot_flux)**2 / obj_item[1]**2

                else:
                    for j in range(obj_item.shape[1]):
                        ln_like += -0.5 * (obj_item[0, j] - phot_flux)**2 / obj_item[1, j]**2

            for i, item in enumerate(self.spectrum.keys()):
                data_flux = spec_scaling[item]*self.spectrum[item][0][:, 1]
                data_var = self.spectrum[item][0][:, 2]**2

                if err_offset[item] is not None:
                    data_var += (10.**err_offset[item])**2

                if self.spectrum[item][2] is not None:
                    if err_offset[item] is None:
                        data_cov_inv = self.spectrum[item][2]

                    else:
                        # Ratio of the inflated and original uncertainties
                        sigma_ratio = np.sqrt(data_var) / self.spectrum[item][0][:, 2]
                        sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                        # Calculate the inversion of the infalted covariances
                        data_cov_inv = np.linalg.inv(self.spectrum[item][1]*sigma_i*sigma_j)

                if self.model == 'planck':
                    readplanck = read_planck.ReadPlanck((0.9*self.spectrum[item][0][0, 0],
                                                         1.1*self.spectrum[item][0][-1, 0]))

                    model_box = readplanck.get_spectrum(param_dict, 1000., smooth=True)

                    model_flux = spectres.spectres(self.spectrum[item][0][:, 0],
                                                   model_box.wavelength,
                                                   model_box.flux)

                else:
                    model_flux = self.modelspec[i].spectrum_interp(list(param_dict.values()))[0, :]
                    model_flux *= flux_scaling

                if 'lognorm_ext' in dust_param:
                    for j, cross_item in enumerate(self.cross_sections[item]):
                        cross_tmp = cross_item(dust_param['lognorm_sigma'],
                                               10.**dust_param['lognorm_radius'])[0]

                        model_flux[j] *= np.exp(-cross_tmp*n_grains)

                elif 'powerlaw_ext' in dust_param:
                    for j, cross_item in enumerate(self.cross_sections[item]):
                        cross_tmp = cross_item(dust_param['powerlaw_exp'],
                                               10.**dust_param['powerlaw_max'])[0]

                        model_flux[j] *= np.exp(-cross_tmp*n_grains)

                elif 'ism_ext' in dust_param:
                    ext_filt = dust_util.ism_extinction(dust_param['ism_ext'],
                                                        dust_param['ism_red'],
                                                        self.spectrum[item][0][:, 0])

                    model_flux *= 10.**(-0.4*ext_filt)

                if self.spectrum[item][2] is not None:
                    # Use the inverted covariance matrix
                    dot_tmp = np.dot(data_flux-model_flux,
                                     np.dot(data_cov_inv, data_flux-model_flux))

                    ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

                else:
                    if item in self.fit_corr:
                        # Covariance model (Wang et al. 2020)
                        wavel = self.spectrum[item][0][:, 0]  # (um)
                        wavel_j, wavel_i = np.meshgrid(wavel, wavel)

                        error = np.sqrt(data_var)  # (W m-2 um-1)
                        error_j, error_i = np.meshgrid(error, error)

                        cov_matrix = corr_amp[item]**2 * error_i * error_j * \
                            np.exp(-(wavel_i-wavel_j)**2 / (2.*corr_len[item]**2)) + \
                            (1.-corr_amp[item]**2) * np.eye(wavel.shape[0])*error_i**2

                        dot_tmp = np.dot(data_flux-model_flux,
                                         np.dot(np.linalg.inv(cov_matrix), data_flux-model_flux))

                        ln_like += -0.5*dot_tmp - 0.5*np.nansum(np.log(2.*np.pi*data_var))

                    else:
                        # Calculate the chi-square without a covariance matrix
                        ln_like += np.nansum(-0.5 * (data_flux-model_flux)**2 / data_var -
                                             0.5 * np.log(2.*np.pi*data_var))

            return ln_like

        pymultinest.run(lnlike_multinest,
                        lnprior_multinest,
                        len(self.modelpar),
                        outputfiles_basename=output,
                        resume=False,
                        n_live_points=n_live_points)

        # Create the Analyzer object
        analyzer = pymultinest.analyse.Analyzer(len(self.modelpar), outputfiles_basename=output)

        # Get a dictionary with the ln(Z) and its errors, the individual modes and their parameters
        # quantiles of the parameter posteriors
        stats = analyzer.get_stats()

        # Nested sampling global log-evidence
        ln_z = stats['nested sampling global log-evidence']
        ln_z_error = stats['nested sampling global log-evidence error']
        print(f'Nested sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}')

        # Nested sampling global log-evidence
        ln_z = stats['nested importance sampling global log-evidence']
        ln_z_error = stats['nested importance sampling global log-evidence error']
        print(f'Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}')

        # Get the best-fit (highest likelihood) point
        print('Sample with the highest likelihood:')
        best_params = analyzer.get_best_fit()

        max_lnlike = best_params['log_likelihood']
        print(f'   - Log-likelihood = {max_lnlike:.2f}')

        for i, item in enumerate(best_params['parameters']):
            print(f'   - {self.modelpar[i]} = {item:.2f}')

        # Get the posterior samples
        samples = analyzer.get_equal_weighted_posterior()

        spec_labels = []
        for item in self.spectrum:
            if f'scaling_{item}' in self.bounds:
                spec_labels.append(f'scaling_{item}')

        species_db = database.Database()

        species_db.add_samples(sampler='multinest',
                               samples=samples[:, :-1],
                               ln_prob=samples[:, -1],
                               mean_accept=None,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance[0],
                               spec_labels=spec_labels)
