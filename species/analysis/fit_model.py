"""
Module with functionalities for fitting atmospheric model spectra.
"""

import math

from multiprocessing import Pool, cpu_count

import emcee
import spectres
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_model, read_object
from species.util import read_util


def lnprior(param,
            bounds,
            modelpar,
            prior=None):
    """
    Internal function for the prior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries.
    modelpar : list(str, )
        Parameter names.
    prior : tuple(str, float, float), None
        Gaussian prior on one of the parameters. Currently only possible for the mass, e.g.
        ('mass', 13., 3.) for an expected mass of 13 Mjup with an uncertainty of 3 Mjup. Not
        used if set to None.

    Returns
    -------
    float
        Log prior probability.
    """

    if prior is not None:

        modeldict = {}
        for i, item in enumerate(modelpar):
            modeldict[item] = param[i]

    ln_prior = 0

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:

            if prior is not None and prior[0] == 'mass' and item == 'logg':
                mass = read_util.get_mass(modeldict)
                ln_prior += -0.5*(mass-prior[1])**2/prior[2]**2

            else:
                ln_prior += 0.

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
           modelspec):
    """
    Internal function for the likelihood probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    modelpar : list(str, )
        Parameter names.
    modelphot : list(species.read.read_model.ReadModel, )
    objphot : list(tuple(float, float), )
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
    distance : float
        Distance (pc).
    spectrum : dict
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    modelspec : list(species.read.read_model.ReadModel, )

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
            flux = modelphot[i].get_flux(paramdict, synphot[i])
            chisq += (item[0]-flux)**2 / item[1]**2

    if spectrum is not None:
        for i, item in enumerate(spectrum.keys()):
            model = modelspec[i].get_model(paramdict)

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
           modelpar,
           modelphot,
           objphot,
           synphot,
           distance,
           prior,
           spectrum,
           modelspec):
    """
    Internal function for the posterior probability.

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
    spectrum : dict
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    modelspec : list(species.read.read_model.ReadModel, )

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
                                    modelspec)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitModel:
    """
    Class for fitting atmospheric model spectra to photometric data.
    """

    def __init__(self,
                 object_name,
                 filters,
                 model,
                 bounds,
                 inc_phot=True,
                 inc_spec=True):
        """
        Parameters
        ----------
        object_name : str
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

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()

        self.model = model
        self.bounds = bounds

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectral data has been selected.')

        if self.bounds is not None:
            readmodel = read_model.ReadModel(self.model)
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
                objectbox = species_db.get_object(object_name, None)
                filters = objectbox.filters

            for item in filters:
                readmodel = read_model.ReadModel(self.model, filter_name=item)
                readmodel.interpolate_model()
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

            self.modelspec = []
            for key, value in self.spectrum.items():
                wavel_range = (0.9*value[0][0, 0], 1.1*value[0][-1, 0])
                self.modelspec.append(read_model.ReadModel(self.model, wavel_range=wavel_range))

        else:
            self.spectrum = None
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

        sigma = {'teff': 5., 'logg': 0.01, 'feh': 0.01, 'fsed': 0.01, 'co': 0.01, 'radius': 0.01}

        print('Running MCMC...')

        ndim = len(self.bounds)

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
                                                   self.modelpar,
                                                   self.modelphot,
                                                   self.objphot,
                                                   self.synphot,
                                                   self.distance,
                                                   prior,
                                                   self.spectrum,
                                                   self.modelspec]))

            sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance)
