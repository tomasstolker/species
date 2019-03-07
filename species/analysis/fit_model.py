'''
Text
'''

import sys
import math

import emcee
import progress.bar
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_model, read_object


MIN_CHISQ = np.inf
MIN_PARAM = None


def lnprior(param,
            bounds,
            modelpar,
            prior):
    '''
    :param param: Parameter values.
    :type param: numpy.ndarray
    :param bounds: Parameter boundaries.
    :type bounds: dict
    :param modelpar: Parameter names.
    :type modelpar: tuple(str, )
    :param prior: Gaussian prior on one of the parameters. Currently only possible for the mass,
                  e.g. ('mass', 13., 3.) for an expected mass of 13 Mjup with an uncertainty of
                  3 Mjup. Not used if set to None.
    :type prior: tuple(str, float, float)

    :return: Log prior probability.
    :rtype: float
    '''

    if prior:

        modeldict = {}
        for i, item in enumerate(modelpar):
            modeldict[item] = param[i]

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:

            if prior is None:
                ln_prior = 0.

            elif prior[0] == 'mass':
                mass = read_model.get_mass(modeldict)
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
           sampling,
           distance):
    '''
    :param param:
    :type param:
    :param modelpar:
    :type modelpar:
    :param modelphot:
    :type modelphot:
    :param objphot:
    :type objphot:
    :param synphot:
    :type synphot:
    :param sampling:
    :type sampling:
    :param distance:
    :type distance:

    :return: Log likelihood probability.
    :rtype: float
    '''

    global MIN_CHISQ
    global MIN_PARAM

    paramdict = {}
    for i, item in enumerate(modelpar):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    chisq = 0.

    for i, item in enumerate(objphot):
        flux = modelphot[i].get_photometry(paramdict, sampling, synphot[i])
        chisq += (item[0]-flux)**2 / item[1]**2

    if chisq < MIN_CHISQ:
        MIN_CHISQ = chisq
        MIN_PARAM = paramdict

    return -0.5*chisq


def lnprob(param,
           bounds,
           modelpar,
           modelphot,
           objphot,
           synphot,
           sampling,
           distance,
           prior):
    '''
    :param param:
    :type param:
    :param bounds:
    :type bounds:
    :param modelpar:
    :type modelpar:
    :param modelphot:
    :type modelphot:
    :param objphot:
    :type objphot:
    :param synphot:
    :type synphot:
    :param sampling:
    :type sampling:
    :param distance:
    :type distance:
    :param prior: Gaussian prior. Not used if set to None.
    :type prior: tuple(str, float, float)

    :return:
    :rtype:
    '''

    ln_prior = lnprior(param, bounds, modelpar, prior)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param,
                                    modelpar,
                                    modelphot,
                                    objphot,
                                    synphot,
                                    sampling,
                                    distance)

    return ln_prob


class FitModel:
    '''
    Text
    '''

    def __init__(self,
                 objname,
                 filters,
                 model,
                 sampling,
                 bounds):
        '''
        :param objname: Object name in the database.
        :type objname: str
        :param filters: Filter IDs for which the photometry is selected. All available
                        photometry of the object is selected if set to None.
        :type filters: tuple(str, )
        :name model: Atmospheric model.
        :type model: str
        :name sampling: Wavelength sampling for the computation of synthetic photometry
                        ('specres' or 'gaussian').
        :type sampling: tuple
        :name bounds: Parameter boundaries. Full parameter range is used if None or not specified.
                      The radius parameter range is set to 0-5 Rjup if not specified.
        :type bounds: dict

        :return: None
        '''

        self.object = read_object.ReadObject(objname)
        self.distance = self.object.get_distance()

        self.model = model
        self.sampling = sampling
        self.bounds = bounds

        self.objphot = []
        self.modelphot = []
        self.synphot = []

        if self.bounds and 'teff' in self.bounds:
            teff_bound = self.bounds['teff']
        else:
            teff_bound = None

        if self.bounds:
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

        if filters is None:
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

        self.modelpar = readmodel.get_parameters()
        self.modelpar.append('radius')

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag,
                 prior=None,
                 ncpu=1):
        '''
        :return: None
        '''

        global MIN_CHISQ
        global MIN_PARAM

        sigma = {'teff':5., 'logg':0.01, 'feh':0.01, 'radius':0.01}

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
                                               self.sampling,
                                               self.distance,
                                               prior]),
                                        threads=ncpu)

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
                               chisquare=(MIN_CHISQ, MIN_PARAM),
                               modelpar=self.modelpar,
                               distance=self.distance)
