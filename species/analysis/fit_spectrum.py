"""
Text
"""

import sys
import math

import emcee
import progress.bar
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_object, read_calibration


MIN_CHISQ = np.inf
MIN_PARAM = None


def lnprob(param,
           bounds,
           objphot,
           specphot):
    """
    :param param: Value of the scaling parameter.
    :type param: numpy.ndarray
    :param bounds: Boundary of the scaling parameter
    :type bounds: dict
    :param objphot:
    :type objphot:
    :param specphot:
    :type specphot:

    :return:
    :rtype:
    """

    global MIN_CHISQ
    global MIN_PARAM

    if bounds['scaling'][0] <= param <= bounds['scaling'][1]:
        ln_prior = 0.

    else:
        ln_prior = -np.inf

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        chisq = 0.
        for i, _ in enumerate(objphot):
            chisq += (objphot[i][0]-(param*specphot[i]))**2 / objphot[i][1]**2

        if chisq < MIN_CHISQ:
            MIN_CHISQ = chisq
            MIN_PARAM = {'scaling':param}

        ln_prob = ln_prior - 0.5*chisq

    return ln_prob


class FitSpectrum:
    """
    Text
    """

    def __init__(self,
                 objname,
                 filters,
                 spectrum,
                 bounds):
        """
        :param objname: Object name in the database.
        :type objname: str
        :param filters: Filter IDs for which the photometry is selected. All available
                        photometry of the object is selected if set to None.
        :type filters: tuple(str, )
        :param spectrum: Calibration spectrum.
        :type spectrum: str
        :param bounds: Boundaries of the scaling parameter, as {'scaling':(min, max)}.
        :type bounds: dict

        :return: None
        """

        self.object = read_object.ReadObject(objname)

        self.spectrum = spectrum
        self.bounds = bounds

        self.objphot = []
        self.specphot = []

        if filters is None:
            species_db = database.Database()
            objectbox = species_db.get_object(objname, None)
            filters = objectbox.filter

        for item in filters:
            readcalib = read_calibration.ReadCalibration(self.spectrum, item)
            calibspec = readcalib.get_spectrum()

            synphot = photometry.SyntheticPhotometry(item)
            spec_phot = synphot.spectrum_to_photometry(calibspec.wavelength, calibspec.flux)
            self.specphot.append(spec_phot)

            obj_phot = self.object.get_photometry(item)
            self.objphot.append((obj_phot[2], obj_phot[3]))

        self.modelpar = ['scaling']

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag):
        """
        :param nwalkers: Number of walkers.
        :type nwalkers: int
        :param nsteps: Number of steps for each walker.
        :type nsteps: int
        :param guess: Guess of the scaling parameter.
        :type guess: float
        :param tag: Database tag for the results.
        :type tag: int

        :return: None
        """

        global MIN_CHISQ
        global MIN_PARAM

        sys.stdout.write('Running MCMC...')
        sys.stdout.flush()

        ndim = 1

        initial = np.zeros((nwalkers, ndim))
        initial[:, 0] = guess['scaling'] + np.random.normal(0, 1e-1*guess['scaling'], nwalkers)

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=ndim,
                                        lnpostfn=lnprob,
                                        a=2.,
                                        args=([self.bounds,
                                               self.objphot,
                                               self.specphot]))

        progbar = progress.bar.Bar('\rRunning MCMC...',
                                   max=nsteps,
                                   suffix='%(percent)d%%')

        for _ in sampler.sample(initial, iterations=nsteps):
            progbar.next()

        progbar.finish()

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('calibration', self.spectrum),
                               tag=tag,
                               chisquare=(MIN_CHISQ, MIN_PARAM),
                               modelpar=self.modelpar,
                               distance=None)
