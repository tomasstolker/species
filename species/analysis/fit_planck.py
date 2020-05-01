"""
Module with functionalities for fitting a Planck spectrum.
"""

import os
import math
import warnings

from typing import Optional, Union, List, Tuple, Dict

from multiprocessing import Pool, cpu_count

import emcee
import spectres
import numpy as np

# installation of MultiNest is not possible on readthedocs
try:
    import pymultinest
except:
    warnings.warn('PyMultiNest could not be imported.')

from typeguard import typechecked

from species.analysis import photometry
from species.data import database
from species.read import read_object, read_planck


@typechecked
def lnprior(param: np.ndarray,
            bounds: dict) -> np.float64:
    """
    Internal function for the prior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries for 'teff' and 'radius'. The values should be provided in a list
        such that multiple Planck functions can be combined, e.g. ``{'teff': [(1000., 2000.),
        (500., 1500.)], 'radius': [(0.5, 1.5), (1.5, 2.0)]}``.

    Returns
    -------
    float
        Log prior probability.
    """

    ln_prior = 0

    for i, item in enumerate(bounds):
        if bounds[item][0] <= param[i] <= bounds[item][1]:
            ln_prior += 0.

        else:
            ln_prior = -np.inf
            break

    return ln_prior


@typechecked
def lnlike(param: np.ndarray,
           bounds: dict,
           objphot: list,
           synphot: list,
           distance: float,
           spectrum: Optional[dict],
           n_planck: int) -> np.float64:
    """
    Internal function for the likelihood function.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries for 'teff' and 'radius'. The values should be provided in a list
        such that multiple Planck functions can be combined, e.g. ``{'teff': [(1000., 2000.),
        (500., 1500.)], 'radius': [(0.5, 1.5), (1.5, 2.0)]}``.
    objphot : list(tuple(float, float), )
        List with the photometric fluxes and uncertainties.
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
        List with the :class:`~species.analysis.photometry.SyntheticPhotometry` objects for
        calculation of synthetic photometry from the model spectra.
    distance : float
        Distance (pc).
    spectrum : dict, None
        Dictionary with the spectra, covariance matrix, inverse of the covariance matrix, and the
        spectral resolution. The spectrum contains columns with wavelength (um), flux (W m-2 um-1),
        and error (W m-2 um-1). Not used if set to ``None``.
    n_planck : int
        Number of Planck components.

    Returns
    -------
    float
        Log likelihood probability.
    """

    paramdict = {}
    for i, item in enumerate(bounds.keys()):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    if n_planck > 1:
        for i in range(n_planck-1):
            if paramdict[f'teff_{i+1}'] > paramdict[f'teff_{i}']:
                return -np.inf

            if paramdict[f'radius_{i}'] > paramdict[f'radius_{i+1}']:
                return -np.inf

    chisq = 0.

    if objphot is not None:
        for i, obj_item in enumerate(objphot):
            readplanck = read_planck.ReadPlanck(filter_name=synphot[i].filter_name)
            flux = readplanck.get_flux(paramdict, synphot=synphot[i])[0]

            if obj_item.ndim == 1:
                chisq += (obj_item[0]-flux)**2 / obj_item[1]**2

            else:
                for j in range(obj_item.shape[1]):
                    chisq += (obj_item[0, j]-flux)**2 / obj_item[1, j]**2

    if spectrum is not None:
        for i, item in enumerate(spectrum.keys()):
            readplanck = read_planck.ReadPlanck((0.9*spectrum[item][0][0, 0],
                                                 1.1*spectrum[item][0][-1, 0]))

            model = readplanck.get_spectrum(paramdict, 100.)

            flux_new = spectres.spectres(spectrum[item][0][:, 0],
                                         model.wavelength,
                                         model.flux,
                                         spec_errs=None)

            if spectrum[item][1] is not None:
                spec_diff = spectrum[item][0][:, 1] - flux_new

                dot_tmp = np.dot(np.transpose(spec_diff), np.linalg.inv(spectrum[item][1]))
                chisq += np.dot(dot_tmp, spec_diff)

            else:
                chisq += np.nansum((spectrum[item][0][:, 1] - flux_new)**2 /
                                   spectrum[item][0][:, 2]**2)

    return -0.5*chisq


@typechecked
def lnprob(param: np.ndarray,
           bounds: dict,
           objphot: list,
           synphot: list,
           distance: float,
           spectrum: Optional[dict],
           n_planck: int) -> np.float64:
    """
    Internal function for the posterior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries for 'teff' and 'radius'. The values should be provided in a list
        such that multiple Planck functions can be combined, e.g. ``{'teff': [(1000., 2000.),
        (500., 1500.)], 'radius': [(0.5, 1.5), (1.5, 2.0)]}``.
    objphot : list(tuple(float, float), )
        List with the photometric fluxes and uncertainties.
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
        List with the :class:`~species.analysis.photometry.SyntheticPhotometry` objects for
        calculation of synthetic photometry from the model spectra.
    distance : float
        Distance (pc).
    spectrum : dict, None
        Dictionary with the spectra, covariance matrix, inverse of the covariance matrix, and the
        spectral resolution. The spectrum contains columns with wavelength (um), flux (W m-2 um-1),
        and error (W m-2 um-1). Not used if set to ``None``.
    n_planck : int
        Number of Planck components.

    Returns
    -------
    float
        Log posterior probability.
    """

    ln_prior = lnprior(param, bounds)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param,
                                    bounds,
                                    objphot,
                                    synphot,
                                    distance,
                                    spectrum,
                                    n_planck)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitPlanck:
    """
    Class for fitting Planck spectra to photometric and/or spectroscopic data. The Planck spectra
    can consist of one or multiple components. In the latter case, a prior is used to enforce the
    temperatures and radii to decrease and increase, respectively.
    """

    @typechecked
    def __init__(self,
                 object_name: str,
                 filters: Optional[List[str]],
                 bounds: Union[Dict[str, Tuple[float, float]],
                               Dict[str, List[Tuple[float, float]]]],
                 inc_phot: bool = True,
                 inc_spec: bool = True) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filters : tuple(str, ), None
            Filter names for which the photometry is selected. All available photometric data of
            the object are used if set to ``None``.
        bounds : dict
            Parameter boundaries for 'teff' and 'radius'. The values should be provided either as
            tuple (with two float) or as list of tuples (with two floats) such that multiple Planck
            functions can be combined, e.g. ``{'teff': [(1000., 2000.), (500., 1500.)],
            'radius': [(0.5, 1.5), (1.5, 2.0)]}``.
        inc_phot : bool
            Include photometric data with the fit.
        inc_spec : bool
            Include spectroscopic data with the fit.

        Returns
        -------
        NoneType
            None
        """

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectroscopic data has been selected.')

        if 'teff' not in bounds or 'radius' not in bounds:
            raise ValueError('The \'bounds\' dictionary should contain \'teff\' and \'radius\'.')

        self.model = 'planck'

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()

        if isinstance(bounds['teff'], list) and isinstance(bounds['radius'], list):
            self.n_planck = len(bounds['teff'])
            self.modelpar = []
            self.bounds = {}

            for i, item in enumerate(bounds['teff']):
                self.modelpar.append(f'teff_{i}')
                self.modelpar.append(f'radius_{i}')

                self.bounds[f'teff_{i}'] = bounds['teff'][i]
                self.bounds[f'radius_{i}'] = bounds['radius'][i]

        else:
            self.n_planck = 1
            self.modelpar = ['teff', 'radius']
            self.bounds = bounds

        if inc_phot:
            self.synphot = []
            self.objphot = []

            if not filters:
                species_db = database.Database()
                objectbox = species_db.get_object(object_name, None)
                filters = objectbox.filters

            for item in filters:
                sphot = photometry.SyntheticPhotometry(item)
                self.synphot.append(sphot)

                obj_phot = self.object.get_photometry(item)
                self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

        else:
            self.synphot = None
            self.objphot = None

        if inc_spec:
            self.spectrum = self.object.get_spectrum()
        else:
            self.spectrum = None

    @typechecked
    def run_mcmc(self,
                 tag: str,
                 guess: Optional[Union[Dict[str, float], Dict[str, List[float]]]],
                 nwalkers: int = 200,
                 nsteps: int = 1000) -> None:
        """
        Function to run the MCMC sampler.

        Parameters
        ----------
        tag : str
            Database tag where the MCMC samples are stored.
        guess : dict, None
            Guess for the 'teff' and 'radius'. Random values between the boundary values are used
            if a value is set to None. The values should be provided either as float or in a list
            of floats such that multiple Planck functions can be combined, e.g.
            ``{'teff': [1500., 1000.], 'radius': [1., 2.]``.
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.

        Returns
        -------
        NoneType
            None
        """

        print('Running MCMC...')

        ndim = len(self.bounds)

        if ndim == 2:
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

        initial = np.zeros((nwalkers, ndim))

        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                initial[:, i] = guess[item] + np.random.normal(0, sigma[item], nwalkers)

            else:
                initial[:, i] = np.random.uniform(low=self.bounds[item][0],
                                                  high=self.bounds[item][1],
                                                  size=nwalkers)

        with Pool(processes=cpu_count()):
            ens_sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=([self.bounds,
                                                       self.objphot,
                                                       self.synphot,
                                                       self.distance[0],
                                                       self.spectrum,
                                                       self.n_planck]))

            ens_sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler='emcee',
                               samples=ens_sampler.chain,
                               ln_prob=ens_sampler.lnprobability,
                               mean_accept=np.mean(ens_sampler.acceptance_fraction),
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance[0],
                               spec_labels=None)

    @typechecked
    def run_multinest(self,
                      tag: str,
                      n_live_points: int = 4000,
                      output: str = 'multinest/') -> None:
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

        Returns
        -------
        NoneType
            None
        """

        print('Running nested sampling...')

        # create the output folder if required

        if not os.path.exists(output):
            os.mkdir(output)

        # create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(self.modelpar):
            cube_index[item] = i

        @typechecked
        def lnprior_multinest(cube,
                              n_dim: int,
                              n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter cube. It is not clear how to
            pass additional arguments to the function, therefore it is placed here.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.

            Returns
            -------
            NoneType
                None
            """

            if len(self.modelpar) == 2:

                # Effective temperature (K)
                cube[cube_index['teff']] = self.bounds['teff'][0] + \
                    (self.bounds['teff'][1]-self.bounds['teff'][0])*cube[cube_index['teff']]

                # Radius (Rjup)
                cube[cube_index['radius']] = self.bounds['radius'][0] + \
                    (self.bounds['radius'][1]-self.bounds['radius'][0])*cube[cube_index['radius']]

            else:
                for i in range(self.n_planck):
                    # Effective temperature (K)
                    cube[cube_index[f'teff_{i}']] = self.bounds[f'teff_{i}'][0] + \
                        (self.bounds[f'teff_{i}'][1]-self.bounds[f'teff_{i}'][0]) * \
                        cube[cube_index[f'teff_{i}']]

                    # Radius (Rjup)
                    cube[cube_index[f'radius_{i}']] = self.bounds[f'radius_{i}'][0] + \
                        (self.bounds[f'radius_{i}'][1]-self.bounds[f'radius_{i}'][0]) * \
                        cube[cube_index[f'radius_{i}']]

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

            Returns
            -------
            float
                The logarithm of the likelihood.
            """

            paramdict = {}

            for i, item in enumerate(self.modelpar):
                paramdict[item] = cube[cube_index[item]]

            paramdict['distance'] = self.distance[0]

            if self.n_planck > 1:
                for i in range(self.n_planck-1):
                    if paramdict[f'teff_{i+1}'] > paramdict[f'teff_{i}']:
                        return -np.inf

                    if paramdict[f'radius_{i}'] > paramdict[f'radius_{i+1}']:
                        return -np.inf

            chisq = 0.

            if self.objphot is not None:
                for i, obj_item in enumerate(self.objphot):
                    readplanck = read_planck.ReadPlanck(filter_name=self.synphot[i].filter_name)
                    flux = readplanck.get_flux(paramdict, synphot=self.synphot[i])[0]

                    if obj_item.ndim == 1:
                        chisq += (obj_item[0]-flux)**2 / obj_item[1]**2

                    else:
                        for j in range(obj_item.shape[1]):
                            chisq += (obj_item[0, j]-flux)**2 / obj_item[1, j]**2

            if self.spectrum is not None:
                for i, item in enumerate(self.spectrum.keys()):
                    readplanck = read_planck.ReadPlanck((0.9*self.spectrum[item][0][0, 0],
                                                         1.1*self.spectrum[item][0][-1, 0]))

                    model = readplanck.get_spectrum(paramdict, 100.)

                    flux_new = spectres.spectres(self.spectrum[item][0][:, 0],
                                                 model.wavelength,
                                                 model.flux,
                                                 spec_errs=None)

                    if self.spectrum[item][1] is not None:
                        spec_diff = self.spectrum[item][0][:, 1] - flux_new

                        dot_tmp = np.dot(np.transpose(spec_diff),
                                         np.linalg.inv(self.spectrum[item][1]))

                        chisq += np.dot(dot_tmp, spec_diff)

                    else:
                        chisq += np.nansum((self.spectrum[item][0][:, 1] - flux_new)**2 /
                                           self.spectrum[item][0][:, 2]**2)

            return -0.5*chisq

        pymultinest.run(lnlike_multinest,
                        lnprior_multinest,
                        len(self.modelpar),
                        outputfiles_basename=output,
                        resume=False,
                        n_live_points=n_live_points)

        samples = np.loadtxt(f'{output}/post_equal_weights.dat')

        species_db = database.Database()

        species_db.add_samples(sampler='multinest',
                               samples=samples[:, :-1],
                               ln_prob=samples[:, -1],
                               mean_accept=None,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance[0],
                               spec_labels=None)
