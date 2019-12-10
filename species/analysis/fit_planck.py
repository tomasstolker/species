"""
Module for fitting atmospheric models.
"""

import sys
import math

from multiprocessing import Pool, cpu_count

import emcee
import spectres
import numpy as np

from species.analysis import photometry
from species.data import database
from species.read import read_object, read_planck


def lnprior(param,
            bounds,
            modelpar):
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

    Returns
    -------
    float
        Log prior probability.
    """

    ln_prior = 0

    for i, item in enumerate(modelpar):

        if bounds[item][0] <= param[i] <= bounds[item][1]:
            ln_prior += 0.

        else:
            ln_prior = -np.inf
            break

    return ln_prior


def lnlike(param,
           modelpar,
           objphot,
           synphot,
           distance,
           spectrum,
           instrument,
           weighting):
    """
    Function for the likelihood probability.

    Parameters
    ----------
    param : numpy.ndarray
        Parameter values.
    modelpar : list(str, )
        Parameter names.
    objphot : list(tuple(float, float), )
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
    distance : float
        Distance (pc).
    spectrum : numpy.ndarray
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    instrument : str
        Instrument that was used for the spectrum (currently only 'gpi' possible).
    weighting : float, None
        Weighting applied to the spectrum when calculating the likelihood function in order
        to not have a spectrum dominate the chi-squared value. For example, with `weighting=3`
        then all combined spectrum points (e.g. covering the YJH bandpasses) have a weighted
        that is equal to three photometry points. The spectrum data points have an equal
        weighting as the photometry points if set to None.

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
            readplanck = read_planck.ReadPlanck(synphot[i].filter_name)
            flux = readplanck.get_photometry(paramdict, synphot=synphot[i])

            chisq += (item[0]-flux)**2 / item[1]**2

    if spectrum is not None:
        # TODO check if the wavelength range of get_planck is broad enought for spectres
        readplanck = read_planck.ReadPlanck((spectrum[0, 0], spectrum[-1, 0]))
        model = readplanck.get_spectrum(paramdict, 100.)

        flux_new = spectres.spectres(new_spec_wavs=spectrum[:, 0],
                                     old_spec_wavs=model.wavelength,
                                     spec_fluxes=model.flux,
                                     spec_errs=None)

        if weighting is None:
            chisq += np.nansum((spectrum[:, 1] - flux_new)**2/spectrum[:, 2]**2)

        else:
            chisq += (weighting/float(spectrum[:, 0].size)) * \
                      np.nansum((spectrum[:, 1] - flux_new)**2/spectrum[:, 2]**2)

    return -0.5*chisq


def lnprob(param,
           bounds,
           modelpar,
           objphot,
           synphot,
           distance,
           spectrum,
           instrument,
           weighting):
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
    objphot : list(tuple(float, float), )
    synphot : list(species.analysis.photometry.SyntheticPhotometry, )
    distance : float
        Distance (pc).
    spectrum : numpy.ndarray
        Wavelength (micron), apparent flux (W m-2 micron-1), and flux error (W m-2 micron-1).
    instrument : str
        Instrument that was used for the spectrum (currently only 'gpi' possible).
    weighting : float, None
        Weighting applied to the spectrum when calculating the likelihood function in order
        to not have a spectrum dominate the chi-squared value. For example, with `weighting=3`
        then all combined spectrum points (e.g. covering the YJH bandpasses) have a weighted
        that is equal to three photometry points. The spectrum data points have an equal
        weighting as the photometry points if set to None.

    Returns
    -------
    float
        Log posterior probability.
    """

    ln_prior = lnprior(param, bounds, modelpar)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(param,
                                    modelpar,
                                    objphot,
                                    synphot,
                                    distance,
                                    spectrum,
                                    instrument,
                                    weighting)

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitPlanck:
    """
    Fit Planck spectrum to photometric and spectral data.
    """

    def __init__(self,
                 objname,
                 filters,
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
        bounds : dict
            Parameter boundaries for 'teff' and 'radius'.
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

        self.model = 'planck'
        self.bounds = bounds
        self.modelpar = ['teff', 'radius']

        if not inc_phot and not inc_spec:
            raise ValueError('No photometric or spectral data has been selected.')

        if 'teff' not in self.bounds or 'radius' not in self.bounds:
            raise ValueError('The \'bounds\' dictionary should contain \'teff\' and \'radius\'.')

        if inc_phot:
            self.synphot = []
            self.objphot = []

            if not filters:
                species_db = database.Database()
                objectbox = species_db.get_object(objname, None)
                filters = objectbox.filter

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
            self.instrument = self.object.get_instrument()

        else:
            self.spectrum = None
            self.instrument = None

    def run_mcmc(self,
                 nwalkers,
                 nsteps,
                 guess,
                 tag,
                 weighting=None):
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
            if a dictionary value is set to None.
        tag : str
            Database tag where the MCMC samples are stored.
        weighting : float, None
            Weighting applied to the spectrum when calculating the likelihood function in order
            to not have a spectrum dominate the chi-squared value. For example, with `weighting=3`
            then all combined spectrum points (e.g. covering the YJH bandpasses) have a weighted
            that is equal to three photometry points. The spectrum data points have an equal
            weighting as the photometry points if set to None.

        Returns
        -------
        NoneType
            None
        """

        sigma = {'teff': 5., 'radius': 0.01}

        sys.stdout.write('Running MCMC...')
        sys.stdout.flush()

        ndim = 2

        initial = np.zeros((nwalkers, ndim))
        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                initial[:, i] = guess[item] + np.random.normal(0, sigma[item], nwalkers)

            else:
                initial[:, i] = np.random.uniform(low=self.bounds[item][0],
                                                  high=self.bounds[item][1],
                                                  size=nwalkers)

        with Pool(processes=cpu_count()) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            lnprob,
                                            args=([self.bounds,
                                                   self.modelpar,
                                                   self.objphot,
                                                   self.synphot,
                                                   self.distance,
                                                   self.spectrum,
                                                   self.instrument,
                                                   weighting]))

            sampler.run_mcmc(initial, nsteps, progress=True)

        species_db = database.Database()

        species_db.add_samples(sampler=sampler,
                               spectrum=('model', self.model),
                               tag=tag,
                               modelpar=self.modelpar,
                               distance=self.distance)
