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

from . import photometry
from .. read import read_model, read_object


min_chisq = np.inf
min_param = None


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

    global min_chisq
    global min_param

    paramdict = {}
    for i, item in enumerate(modelpar):
        paramdict[item] = param[i]

    paramdict['distance'] = distance

    chisq = 0.

    for i, item in enumerate(objphot):
        flux = modelphot[i].get_photometry(paramdict, coverage, synphot[i])
        chisq += (item[0]-flux)**2 / item[1]**2

    if chisq < min_chisq:
        min_chisq = chisq
        min_param = paramdict

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


# def chisq_minimization(param_name,
#                        modelphot,
#                        synphot,
#                        objphot,
#                        coverage,
#                        tolerance,
#                        distance,
#                        param_value):
#     """
#     :param param:
#     :type param:
#     :param modelpar:
#     :type modelpar:
#     :param modelphot:
#     :type modelphot:
#     :param synphot:
#     :type synphot:
#     :param objphot:
#     :type objphot:
#     :param coverage:
#     :type coverage:
#     :param distance:
#     :type distance:
#
#     :return:
#     :rtype:
#     """
#
#     radius_init = 1.5 # [Rjup]
#
#     def _objective(arg):
#         radius = arg[0]
#
#         parsec = 3.08567758147e16 # [m]
#         r_jup = 71492000. # [m]
#
#         chisquare = 0.
#
#         for i, obj_item in enumerate(objphot):
#             paramdict = {param_name[0]:param_value[0],
#                          param_name[1]:param_value[1],
#                          param_name[2]:param_value[2]}
#
#             flux = modelphot[i].get_photometry(paramdict, coverage, synphot[i])
#
#             scaling = (radius*r_jup)**2 / (distance*parsec)**2
#             chisquare += (obj_item[0]-scaling*flux)**2 / obj_item[1]**2
#
#         return chisquare
#
#     result = minimize(fun=_objective,
#                       x0=[radius_init],
#                       method="Nelder-Mead",
#                       tol=None,
#                       options={'xatol':tolerance, 'fatol':float("inf")})
#
#     return result.x, result.fun


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

        self.objphot = []
        self.modelphot = []
        self.synphot = []

        for item in filters:
            readmodel = read_model.ReadModel(self.model, item)
            readmodel.interpolate()
            self.modelphot.append(readmodel)

            sphot = photometry.SyntheticPhotometry(item)
            self.synphot.append(sphot)

            obj_phot = self.object.get_photometry(item)
            self.objphot.append((obj_phot[2], obj_phot[3]))

        self.modelpar = readmodel.get_parameters()
        self.modelpar.append('radius')

        if not self.bounds:
            self.bounds = readmodel.get_bounds()
            self.bounds['radius'] = (0., 3.)

    def store_samples(self,
                      sampler,
                      model,
                      tag,
                      chisquare):
        """
        :param sampler: Ensemble sampler.
        :type sampler: emcee.ensemble.EnsembleSampler
        :param model: Atmospheric model.
        :type model: str
        :param chisquare: Maximum likelihood solution. Tuple with the chi-square value and related
                          parameter values.
        :type chisquare: tuple(float, float)

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

        samples = sampler.chain

        dset = h5_file.create_dataset('results/mcmc/'+tag,
                                      data=samples,
                                      dtype='f')

        dset.attrs['model'] = str(model)
        dset.attrs['distance'] = float(self.distance)
        dset.attrs['nparam'] = int(len(self.modelpar))

        for i, item in enumerate(self.modelpar):
            dset.attrs['parameter'+str(i)] = str(item)

        dset.attrs['min_chi'] = float(chisquare[0])
        for i, item in enumerate(self.modelpar):
            dset.attrs['chisquare'+str(i)] = float(chisquare[1][item])

        mean_accep = np.mean(sampler.acceptance_fraction)
        dset.attrs['acceptance'] = float(mean_accep)
        print('Mean acceptance fraction: {0:.3f}'.format(mean_accep))

        try:
            int_auto = emcee.autocorr.integrated_time(sampler.flatchain)
            print('Integrated autocorrelation time =', int_auto)

        except emcee.autocorr.AutocorrError:
            int_auto = None

        if int_auto is not None:
            for i, item in enumerate(int_auto):
                dset.attrs['autocorrelation'+str(i)] = float(item)

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

        global min_chisq
        global min_param

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
                                               self.coverage,
                                               self.distance]),
                                        threads=ncpu)

        progbar = progress.bar.Bar('\rRunning MCMC...',
                                   max=nsteps,
                                   suffix='%(percent)d%%')

        for i, _ in enumerate(sampler.sample(initial, iterations=nsteps)):
            progbar.next()

        progbar.finish()

        self.store_samples(sampler, self.model, tag, (min_chisq, min_param))

    def store_chisquare(self,
                        chisquare,
                        points,
                        param_name,
                        model,
                        tag):
        """
        :param chisquare: Chi-square values.
        :type chisquare: numpy.ndarray
        :param points:
        :type points: dict
        :param param_name: Parameter names.
        :type param_name: list
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

        dof = len(self.objphot)
        chisquare /= float(dof)

        index_min = np.argwhere(chisquare == np.min(chisquare))[0]

        min_chisquare = np.min(chisquare)
        min_teff = points[param_name[0]][index_min[0]]
        min_logg = points[param_name[1]][index_min[1]]
        min_feh = points[param_name[2]][index_min[2]]
        min_radius = points[param_name[3]][index_min[3]]

        h5_file = h5py.File(database, 'a')

        if 'results' not in h5_file:
            h5_file.create_group('results')

        if 'results/chisquare' not in h5_file:
            h5_file.create_group('results/chisquare')

        if 'results/chisquare/'+tag in h5_file:
            del h5_file['results/chisquare/'+tag]

        h5_file.create_dataset('results/chisquare/'+tag+'/chisquare',
                               data=chisquare,
                               dtype='f')

        dset = h5_file['results/chisquare/'+tag+'']

        dset.attrs['model'] = str(model)
        dset.attrs['nparam'] = int(len(self.modelpar))
        dset.attrs['dof'] = int(dof)
        dset.attrs['chi2'] = float(min_chisquare)
        dset.attrs['teff'] = float(min_teff)
        dset.attrs['logg'] = float(min_logg)
        dset.attrs['feh'] = float(min_feh)
        dset.attrs['radius'] = float(min_radius)

        for i, item in enumerate(self.modelpar):
            dset.attrs['parameter'+str(i)] = str(item)

            h5_file.create_dataset('results/chisquare/'+tag+'/'+item,
                                   data=points[item],
                                   dtype='f')

        # print('Degrees of freedom =', dof)
        # print('Reduced chi-square =', min_chisquare)
        # print('Teff [K] =', min_teff)
        # print('log g =', min_logg)
        # print('[Fe/H] =', min_feh)
        # print('Radius [Rjup] =', min_radius)

        h5_file.close()

    def run_chisquare(self,
                      steps,
                      radius,
                      tag):
        """
        :param steps:
        :type steps: dict
        :param radius:
        :type radius: tuple(float, float, int)
        :param tag:
        :type tag: str

        :return:
        """

        sys.stdout.write('Running chi-square minimization...')
        sys.stdout.flush()

        readmodel = read_model.ReadModel(self.model, None)

        if steps:
            bounds = readmodel.get_bounds()

            points = {}
            for i, item in enumerate(bounds):
                points[item] = np.linspace(bounds[item][0], bounds[item][1], steps[item])

        else:
            points = readmodel.get_points()

        points['radius'] = np.linspace(radius[0], radius[1], radius[2])

        nparam = len(points)
        param_name = list(points)

        shape = (len(points[param_name[0]]),
                 len(points[param_name[1]]),
                 len(points[param_name[2]]),
                 len(points[param_name[3]]))

        chisquare = np.zeros(shape)

        if nparam == 4:
            for i, param_item0 in enumerate(points[param_name[0]]):
                for j, param_item1 in enumerate(points[param_name[1]]):
                    for k, param_item2 in enumerate(points[param_name[2]]):
                        for m, param_item3 in enumerate(points[param_name[3]]):

                            paramdict = {param_name[0]:param_item0,
                                         param_name[1]:param_item1,
                                         param_name[2]:param_item2,
                                         param_name[3]:param_item3,
                                         'distance':self.distance}

                            for n, obj_item in enumerate(self.objphot):
                                flux = self.modelphot[n].get_photometry(paramdict,
                                                                        self.coverage,
                                                                        self.synphot[n])

                                chisquare[i, j, k, m] += (obj_item[0]-flux)**2 / obj_item[1]**2

        self.store_chisquare(chisquare, points, param_name, self.model, tag)

        sys.stdout.write(' [DONE]\n')
        sys.stdout.flush()
