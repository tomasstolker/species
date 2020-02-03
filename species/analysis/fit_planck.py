"""
Module with functionalities for fitting a Planck spectrum.
"""

import math

from multiprocessing import Pool, cpu_count

import emcee
import spectres
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_object, read_planck


def lnprior(param,
            bounds):
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


def lnlike(param,
           bounds,
           objphot,
           synphot,
           distance,
           spectrum):
    """
    Internal function for the likelihood probability.

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
    spectrum : numpy.ndarray, None
        Spectrum array with the wavelength (micron), flux (W m-2 micron-1), and error
        (W m-2 micron-1). Not used if set to None.

    Returns
    -------
    float
        Log likelihood probability.
    """

    paramdict = {}
    for i, item in enumerate(bounds.keys()):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    chisq = 0.

    if objphot is not None:
        for i, obj_item in enumerate(objphot):
            readplanck = read_planck.ReadPlanck(filter_name=synphot[i].filter_name)
            flux = readplanck.get_flux(paramdict, synphot=synphot[i])

            chisq += (obj_item[0]-flux)**2 / obj_item[1]**2

    if spectrum is not None:
        for i, item in enumerate(spectrum.keys()):
            readplanck = read_planck.ReadPlanck((0.9*spectrum[item][0][0, 0],
                                                 1.1*spectrum[item][0][-1, 0]))

            model = readplanck.get_spectrum(paramdict, 100.)

            flux_new = spectres.spectres(new_spec_wavs=spectrum[item][0][:, 0],
                                         old_spec_wavs=model.wavelength,
                                         spec_fluxes=model.flux,
                                         spec_errs=None)

            if spectrum[item][1] is not None:
                spec_res = spectrum[item][0][:, 1] - flux_new

                dot_tmp = np.dot(np.transpose(spec_res), np.linalg.inv(spectrum[item][1]))
                chisq += np.dot(dot_tmp, spec_res)

            else:
                chisq += np.nansum((spectrum[item][0][:, 1] - flux_new)**2 /
                                   spectrum[item][0][:, 2]**2)

    return -0.5*chisq


def lnprob(param,
           bounds,
           objphot,
           synphot,
           distance,
           spectrum):
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
    spectrum : numpy.ndarray, None
        Spectrum array with the wavelength (micron), flux (W m-2 micron-1), and error
        (W m-2 micron-1). Not used if set to None.

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
                                    spectrum)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitPlanck:
    """
    Class for fitting Planck spectra to spectral and photometric data.
    """

    def __init__(self,
                 object_name,
                 filters,
                 bounds,
                 inc_phot=True,
                 inc_spec=True):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filters : tuple(str, )
            Filter names for which the photometry is selected. All available photometry of the
            object are used if set to None.
        bounds : dict
            Parameter boundaries for 'teff' and 'radius'. The values should be provided either as
            float or as list of floats such that multiple Planck functions can be combined,
            e.g. ``{'teff': [(1000., 2000.), (500., 1500.)], 'radius': [(0.5, 1.5), (1.5, 2.0)]}``.
        inc_phot : bool
            Include photometry data with the fit.
        inc_spec : bool
            Include spectral data with the fit.

        Returns
        -------
        NoneType
            None
        """

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectral data has been selected.')

        if 'teff' not in bounds or 'radius' not in bounds:
            raise ValueError('The \'bounds\' dictionary should contain \'teff\' and \'radius\'.')

        self.model = 'planck'

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()

        if isinstance(bounds['teff'], list) and isinstance(bounds['radius'], list):
            self.modelpar = []
            self.bounds = {}

            for i, item in enumerate(bounds['teff']):
                self.modelpar.append(f'teff_{i}')
                self.modelpar.append(f'radius_{i}')

                self.bounds[f'teff_{i}'] = bounds['teff'][i]
                self.bounds[f'radius_{i}'] = bounds['radius'][i]

        else:
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
                self.objphot.append((obj_phot[2], obj_phot[3]))

        else:
            self.synphot = None
            self.objphot = None

        if inc_spec:
            self.spectrum = self.object.get_spectrum()
        else:
            self.spectrum = None

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag):
        """
        Function to run the MCMC sampler.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        guess : dict, None
            Guess for the 'teff' and 'radius'. Random values between the boundary values are used
            if a value is set to None. The values should be provided either as float or in a list
            of floats such that multiple Planck functions can be combined, e.g.
            ``{'teff': [1500., 1000.], 'radius': [1., 2.]``.
        tag : str
            Database tag where the MCMC samples are stored.

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
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            args=([self.bounds,
                                                   self.objphot,
                                                   self.synphot,
                                                   self.distance,
                                                   self.spectrum]))

            sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance)
