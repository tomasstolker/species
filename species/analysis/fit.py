"""
Text
"""

import os
import sys
import math
import configparser

import h5py
import emcee
import progress.bar
import numpy as np

from scipy.optimize import minimize

from . import photometry
from .. read import read_model, read_object


def lnprior(param,
            bounds,
            modelpar):
    """
    :param param:
    :type param:
    :param bounds:
    :type bounds:
    :param modelpar:
    :type modelpar:

    :return: Log prior probability.
    :rtype: float
    """

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:
            ln_prior = 0.

        else:
            ln_prior = -np.inf
            break

    return ln_prior


def lnlike(param,
           modelpar,
           modelphot,
           objphot,
           synphot,
           coverage,
           distance):
    """
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
    :param coverage:
    :type coverage:
    :param distance:
    :type distance:

    :return: Log likelihood probability.
    :rtype: float
    """

    paramdict = {}
    for i, item in enumerate(modelpar):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    chisq = 0.

    for i, item in enumerate(objphot):
        flux = modelphot[i].get_photometry(paramdict, coverage, synphot[i])
        chisq += (item[0]-flux)**2 / item[1]**2

    return -0.5*chisq


def lnprob(param,
           bounds,
           modelpar,
           modelphot,
           objphot,
           synphot,
           coverage,
           distance):
    """
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
    :param coverage:
    :type coverage:
    :param distance:
    :type distance:

    :return:
    :rtype:
    """

    ln_prior = lnprior(param, bounds, modelpar)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param,
                                    modelpar,
                                    modelphot,
                                    objphot,
                                    synphot,
                                    coverage,
                                    distance)

    return ln_prob


class FitSpectrum:
    """
    Text
    """

    def __init__(self,
                 objname,
                 filters,
                 model,
                 coverage,
                 bounds):
        """
        :param objname:
        :type objname:

        :return: None
        """

        self.parsec = 3.08567758147e16 # [m]
        self.r_jup = 71492000. # [m]

        self.object = read_object.ReadObject(objname)
        self.distance = self.object.get_distance()

        self.model = model
        self.coverage = coverage
        self.bounds = bounds

        if self.model == 'drift-phoenix':
            self.modelpar = ('teff', 'logg', 'feh', 'radius')

        self.objphot = []
        self.modelphot = []
        self.synphot = []

        for item in filters:
            readmodel = read_model.ReadModel(self.model, item)
            readmodel.interpolate()

            self.modelphot.append(readmodel)

            mag = self.object.get_magnitude(item)

            sphot = photometry.SyntheticPhotometry(item)
            flux_obj, error_obj = sphot.magnitude_to_flux(mag[0], mag[1])

            self.synphot.append(sphot)
            self.objphot.append((flux_obj, (error_obj[0]+error_obj[1])/2.))

        if not self.bounds:
            self.bounds = readmodel.get_bounds()
            self.bounds['radius'] = (0., 3.)

    def store_samples(self,
                      samples,
                      model,
                      tag):
        """
        :param samples: MCMC samples.
        :type samples: numpy.ndarray
        :param model: Atmospheric model.
        :type model: str

        :return: None
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database = config['species']['database']

        h5_file = h5py.File(database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/mcmc' not in h5_file:
            h5_file.create_group('results/mcmc')

        if 'results/mcmc/'+tag in h5_file:
            del h5_file['results/mcmc/'+tag]

        dset = h5_file.create_dataset('results/mcmc/'+tag,
                                      data=samples,
                                      dtype='f')

        dset.attrs['model'] = str(model)

        for i, item in enumerate(self.modelpar):
            dset.attrs['parameter'+str(i+1)] = str(item)

        h5_file.close()

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag,
                 ncpu=1):
        """
        :return: None
        """

        sigma = (5., 0.01, 0.01, 0.01)

        sys.stdout.write('Running MCMC...')
        sys.stdout.flush()

        ndim = len(guess)

        initial = np.zeros((nwalkers, ndim))
        for i, item in enumerate(self.modelpar):
            initial[:, i] = guess[item] + np.random.normal(0, sigma[i], nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=ndim,
                                        lnpostfn=lnprob,
                                        a=2.,
                                        args=([self.bounds,
                                               self.modelpar,
                                               self.modelphot,
                                               self.objphot,
                                               self.synphot,
                                               self.coverage,
                                               self.distance]),
                                        threads=ncpu)

        progbar = progress.bar.Bar('\rRunning MCMC...',
                                   max=nsteps,
                                   suffix='%(percent)d%%')

        for i, _ in enumerate(sampler.sample(initial, iterations=nsteps)):
            progbar.next()

        progbar.finish()

        self.store_samples(sampler.chain, self.model, tag)

        mean_accep = np.mean(sampler.acceptance_fraction)
        print('Mean acceptance fraction: {0:.3f}'.format(mean_accep))

        int_auto = emcee.autocorr.integrated_time(sampler.flatchain)
        print('Integrated autocorrelation time = ', int_auto)

    def store_chisquare(self,
                        chisquare,
                        model,
                        tag):
        """
        :param chisquare: Chi-square values.
        :type chisquare: numpy.ndarray
        :param model: Atmospheric model.
        :type model: str
        :param tag: Atmospheric model.
        :type tag: str

        :return: None
        """

        config_file = os.path.join(os.getcwd(), 'species_config.ini')

        config = configparser.ConfigParser()
        config.read_file(open(config_file))

        database = config['species']['database']

        h5_file = h5py.File(database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/chisquare' not in h5_file:
            h5_file.create_group('results/chisquare')

        if 'results/chisquare/'+tag in h5_file:
            del h5_file['results/chisquare/'+tag]

        dset = h5_file.create_dataset('results/chisquare/'+tag,
                                      data=chisquare,
                                      dtype='f')

        dset.attrs['model'] = str(model)

        for i, item in enumerate(self.modelpar):
            dset.attrs['parameter'+str(i+1)] = str(item)

        h5_file.close()

    def run_chisquare(self,
                      tolerance,
                      step,
                      tag):
        """
        :param tolerance:
        :type tolerance: float
        :param tag:
        :type tag: str

        :return:
        """

        def _objective(arg,
                       params,
                       param_item1,
                       param_item2,
                       param_item3):
            radius = arg[0]
            chisquare = 0.

            for i, obj_item in enumerate(self.objphot):
                paramdict = {params[0]:param_item1,
                             params[1]:param_item2,
                             params[2]:param_item3}

                flux = self.modelphot[i].get_photometry(paramdict, self.coverage, self.synphot[i])

                scaling = (radius*self.r_jup)**2 / (self.distance*self.parsec)**2
                chisquare += (obj_item[0]-scaling*flux)**2 / obj_item[1]**2

            return chisquare

        sys.stdout.write('Running chi-square minimization...')
        sys.stdout.flush()

        radius_init = 1.5 # [Rjup]

        readmodel = read_model.ReadModel(self.model, None)

        if not step:
            points = readmodel.get_points()

        else:
            bounds = readmodel.get_bounds()

            points = {}
            for i, item in enumerate(bounds):
                points[item] = np.arange(bounds[item][0], bounds[item][1], step[item])

        nparam = len(points)
        params = list(points)

        shape = (len(points[params[0]]), len(points[params[1]]), len(points[params[2]]))
        chisquare = r_planet = np.zeros(shape)

        if nparam == 3:

            progbar = progress.bar.Bar('\rRunning chi-square minimization...',
                                       max=len(points[params[0]]),
                                       suffix='%(percent)d%%')

            for i, param_item1 in enumerate(points[params[0]]):
                for j, param_item2 in enumerate(points[params[1]]):
                    for k, param_item3 in enumerate(points[params[2]]):

                        result = minimize(fun=_objective,
                                          x0=[radius_init],
                                          args=(params, param_item1, param_item2, param_item3),
                                          method="Nelder-Mead",
                                          tol=None,
                                          options={'xatol':tolerance, 'fatol':float("inf")})

                        if result.success:
                            chisquare[i, j, k] = result.x
                            r_planet[i, j, k] = result.fun

                progbar.next()
            progbar.finish()

        index_min = np.argwhere(chisquare == np.min(chisquare))[0]

        dof = float(len(self.objphot))
        chisquare /= dof

        self.store_chisquare(chisquare, self.model, tag)

        print('Degrees of freedom =', str(int(dof)))
        print('Reduced chi-square =', str(np.min(chisquare)))
        print('Teff [K] =', str(points[params[0]][index_min[0]]))
        print('log g =', str(points[params[1]][index_min[1]]))
        print('[Fe/H] =', str(points[params[2]][index_min[2]]))
        print('Radius [Rjup] =', str(r_planet[index_min[0], index_min[1], index_min[2]]))
