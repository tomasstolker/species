"""
Module for fitting atmospheric models.
"""

import sys
import math

import emcee
import spectres
import progress.bar
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_model, read_object
from species.util import read_util


def lnprior(param,
            bounds,
            modelpar,
            prior):
    """
    Function for the prior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries.
    modelpar : list(str, )
        Parameter names.
    prior : tuple(str, float, float)
        Gaussian prior on one of the parameters. Currently only possible for the mass, e.g.
        ('mass', 13., 3.) for an expected mass of 13 Mjup with an uncertainty of 3 Mjup. Not
        used if set to None.

    Returns
    -------
    float
        Log prior probability.
    """

    if prior:

        modeldict = {}
        for i, item in enumerate(modelpar):
            modeldict[item] = param[i]

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:

            if prior is None:
                ln_prior = 0.

            elif prior[0] == 'mass':
                mass = read_util.get_mass(modeldict)
                ln_prior = -0.5*(mass-prior[1])**2/prior[2]**2

        else:
            ln_prior = -np.inf
            break

    return ln_prior


def lnlike(param,
           modelpar,
           modelphot,
           objphot,
           synphot,
           distance,
           spectrum,
           instrument,
           modelspec):
    """
    Function for the likelihood probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    modelpar : list(str, )
        Parameter names.
    modelphot : list('species.read.read_model.ReadModel, )
    objphot : list(tuple(float, float), )
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
    distance : float
        Distance (pc).
    spectrum : numpy.ndarray
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    instrument : str
        Instrument that was used for the spectrum (currently only 'gpi' possible).
    modelspec : species.read.read_model.ReadModel

    Returns
    -------
    float
        Log likelihood probability.
    """

    paramdict = {}
    for i, item in enumerate(modelpar):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    chisq = 0.

    if objphot is not None:
        for i, item in enumerate(objphot):
            flux = modelphot[i].get_photometry(paramdict, synphot[i])
            chisq += (item[0]-flux)**2 / item[1]**2

    if spectrum is not None:
        model = modelspec.get_model(paramdict, None)

        flux_new = spectres.spectres(new_spec_wavs=spectrum[:, 0],
                                     old_spec_wavs=model.wavelength,
                                     spec_fluxes=model.flux,
                                     spec_errs=None)

        # (4./float(spectrum[:, 0].size))
        chisq += np.nansum((spectrum[:, 1] - flux_new)**2/spectrum[:, 2]**2)

    return -0.5*chisq


def lnprob(param,
           bounds,
           modelpar,
           modelphot,
           objphot,
           synphot,
           distance,
           prior,
           spectrum,
           instrument,
           modelspec):
    """
    Function for the posterior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries.
    modelpar : list(str, )
        Parameter names.
    modelphot : list('species.read.read_model.ReadModel, )
    objphot : list(tuple(float, float), )
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
    distance : float
        Distance (pc).
    prior : tuple(str, float, float)
        Gaussian prior on one of the parameters. Currently only possible for the mass, e.g.
        ('mass', 13., 3.) for an expected mass of 13 Mjup with an uncertainty of 3 Mjup. Not
        used if set to None.
    spectrum : numpy.ndarray
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    instrument : str
        Instrument that was used for the spectrum (currently only 'gpi' possible).
    modelspec : species.read.read_model.ReadModel

    Returns
    -------
    float
        Log posterior probability.
    """

    ln_prior = lnprior(param, bounds, modelpar, prior)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param,
                                    modelpar,
                                    modelphot,
                                    objphot,
                                    synphot,
                                    distance,
                                    spectrum,
                                    instrument,
                                    modelspec)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitModel:
    """
    Fit atmospheric model spectra to photometric and spectral data.
    """

    def __init__(self,
                 objname,
                 filters,
                 model,
                 bounds,
                 inc_phot=True,
                 inc_spec=True):
        """
        Parameters
        ----------
        objname : str
            Object name in the database.
        filters : tuple(str, )
            Filter IDs for which the photometry is selected. All available photometry of the
            object is selected if set to None.
        model : str
            Atmospheric model.
        bounds : dict
            Parameter boundaries. Full parameter range is used if set to None or not specified.
            The radius parameter range is set to 0-5 Rjup if not specified.
        inc_phot : bool
            Include photometry data with the fit.
        inc_spec : bool
            Include spectral data with the fit.

        Returns
        -------
        NoneType
            None
        """

        self.object = read_object.ReadObject(objname)
        self.distance = self.object.get_distance()

        self.model = model
        self.bounds = bounds

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectral data has been selected.')

        if self.bounds is not None and 'teff' in self.bounds:
            teff_bound = self.bounds['teff']
        else:
            teff_bound = None

        if self.bounds is not None:
            readmodel = read_model.ReadModel(self.model, None, teff_bound)
            bounds_grid = readmodel.get_bounds()

            for item in bounds_grid:
                if item not in self.bounds:
                    self.bounds[item] = bounds_grid[item]

        else:
            readmodel = read_model.ReadModel(self.model, None, None)
            self.bounds = readmodel.get_bounds()

        if 'radius' not in self.bounds:
            self.bounds['radius'] = (0., 5.)

        if inc_phot:
            self.objphot = []
            self.modelphot = []
            self.synphot = []

            if not filters:
                species_db = database.Database()
                objectbox = species_db.get_object(objname, None)
                filters = objectbox.filter

            for item in filters:
                readmodel = read_model.ReadModel(self.model, item, teff_bound)
                readmodel.interpolate()
                self.modelphot.append(readmodel)

                sphot = photometry.SyntheticPhotometry(item)
                self.synphot.append(sphot)

                obj_phot = self.object.get_photometry(item)
                self.objphot.append((obj_phot[2], obj_phot[3]))

        else:
            self.objphot = None
            self.modelphot = None
            self.synphot = None

        if inc_spec:
            self.spectrum = self.object.get_spectrum()
            self.instrument = self.object.get_instrument()
            self.modelspec = read_model.ReadModel(self.model, (0.9, 2.5), teff_bound)

        else:
            self.spectrum = None
            self.instrument = None
            self.modelspec = None

        self.modelpar = readmodel.get_parameters()
        self.modelpar.append('radius')

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag,
                 prior=None):
        """
        Function to run the MCMC sampler.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        guess : dict
            Guess for the parameter values. Random values between the boundary values are used
            if set to None.
        tag : str
            Database tag where the MCMC samples are stored.
        prior : tuple(str, float, float)
            Gaussian prior on one of the parameters. Currently only possible for the mass, e.g.
            ('mass', 13., 3.) for an expected mass of 13 Mjup with an uncertainty of 3 Mjup. Not
            used if set to None.

        Returns
        -------
        NoneType
            None
        """

        sigma = {'teff': 5., 'logg': 0.01, 'feh': 0.01, 'radius': 0.01}

        sys.stdout.write('Running MCMC...')
        sys.stdout.flush()

        ndim = len(self.bounds)

        initial = np.zeros((nwalkers, ndim))
        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                initial[:, i] = guess[item] + np.random.normal(0, sigma[item], nwalkers)

            else:
                initial[:, i] = np.random.uniform(low=self.bounds[item][0],
                                                  high=self.bounds[item][1],
                                                  size=nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=ndim,
                                        lnpostfn=lnprob,
                                        a=2.,
                                        args=([self.bounds,
                                               self.modelpar,
                                               self.modelphot,
                                               self.objphot,
                                               self.synphot,
                                               self.distance,
                                               prior,
                                               self.spectrum,
                                               self.instrument,
                                               self.modelspec]))

        progbar = progress.bar.Bar('\rRunning MCMC...',
                                   max=nsteps,
                                   suffix='%(percent)d%%')

        for i, _ in enumerate(sampler.sample(initial, iterations=nsteps)):
            progbar.next()

        progbar.finish()

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance)
