"""
Module with functionalities for fitting a calibration spectrum.
"""

import math

from multiprocessing import Pool, cpu_count

import emcee
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_object, read_calibration


def lnprob(param,
           bounds,
           modelpar,
           objphot,
           specphot,
           bands):
    """
    Internal function for the posterior probability.

    Parameters
    ----------
    param : numpy.ndarray
        Values of the main scaling parameter and optionally additional band-dependent scaling
        parameters.
    bounds : dict
        Boundaries of the main scaling parameter.
    modelpar : list(str, )
        Parameter names.
    objphot : list(tuple(float, float), )
        Photometry of the object.
    specphot : list(float, )
        Synthetic photometry of the calibration spectrum for the same filters as the photometry
        of the object.
    bands : bool
        Use band-dependent scaling parameters in addition to the main scaling parameter which
        is used for the full spectrum.

    Returns
    -------
    float
        Log posterior probability.
    """

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:
            ln_prior = 0.

        else:
            ln_prior = -np.inf
            break

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        chisq = 0.
        for i, _ in enumerate(objphot):
            if bands:
                chisq += (objphot[i][0] - param[0]*param[i+1]*specphot[i])**2 / objphot[i][1]**2

            else:
                chisq += (objphot[i][0] - param[0]*specphot[i])**2 / objphot[i][1]**2

        ln_prob = ln_prior - 0.5*chisq

    return ln_prob


class FitSpectrum:
    """
    Class for fitting a calibration spectrum to photometric data.
    """

    def __init__(self,
                 object_name,
                 filters,
                 spectrum,
                 bounds):
        """
        Parameters
        ----------
        object_name : str
            Object name in the database.
        filters : tuple(str, )
            Filter IDs for which the photometry is selected. All available photometry of the
            object is selected if set to None.
        spectrum : str
            Calibration spectrum.
        bounds : dict
            Boundaries of the scaling parameter, as {'scaling':(min, max)}.

        Returns
        -------
        None
        """

        self.object = read_object.ReadObject(object_name)

        self.spectrum = spectrum
        self.bounds = bounds

        self.objphot = []
        self.specphot = []

        if filters is None:
            species_db = database.Database()

            objectbox = species_db.get_object(object_name,
                                              filters=None,
                                              inc_phot=True,
                                              inc_spec=False)

            filters = objectbox.filters

        for item in filters:
            readcalib = read_calibration.ReadCalibration(self.spectrum, item)
            calibspec = readcalib.get_spectrum()

            synphot = photometry.SyntheticPhotometry(item)
            spec_phot = synphot.spectrum_to_flux(calibspec.wavelength, calibspec.flux)
            self.specphot.append(spec_phot)

            obj_phot = self.object.get_photometry(item)
            self.objphot.append((obj_phot[2], obj_phot[3]))

        self.modelpar = ['scaling']

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag,
                 bands=False):
        """
        Function to run the MCMC sampler.

        Parameters
        ----------
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        guess : dict
            Guess of the scaling parameter.
        tag : str
            Database tag where the MCMC samples are stored.
        bands : bool
            Use band-dependent scaling parameters in addition to the main scaling parameter which
            is used for the full spectrum.

        Returns
        -------
        None
        """

        print('Running MCMC...')

        if bands:
            ndim = 1 + len(self.objphot)

        else:
            ndim = 1

        initial = np.zeros((nwalkers, ndim))
        initial[:, 0] = guess['scaling'] + np.random.normal(0, 1e-1*guess['scaling'], nwalkers)

        if ndim > 1:
            for i in range(1, ndim):
                initial[:, i] = 1. + np.random.normal(0, 0.1, nwalkers)
                self.modelpar.append('scaling'+str(i))
                self.bounds['scaling'+str(i)] = (0., 1e2)

        with Pool(processes=cpu_count()):
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            args=([self.bounds,
                                                   self.modelpar,
                                                   self.objphot,
                                                   self.specphot,
                                                   bands]))

            sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('calibration', self.spectrum),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=None)
