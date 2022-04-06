"""
Module with functionalities for fitting atmospheric model spectra.
"""

import os
import math
import warnings

from typing import Optional, Union, List, Tuple, Dict
from multiprocessing import Pool, cpu_count

import emcee
import numpy as np
import spectres

from scipy import stats

try:
    import ultranest
except:
    warnings.warn(
        "UltraNest could not be imported. Perhaps "
        "because cython was not correctly compiled?"
    )

try:
    import pymultinest
except:
    warnings.warn(
        "PyMultiNest could not be imported. "
        "Perhaps because MultiNest was not build "
        "and/or found at the LD_LIBRARY_PATH "
        "(Linux) or DYLD_LIBRARY_PATH (Mac)?"
    )

from typeguard import typechecked

from species.analysis import photometry
from species.data import database
from species.core import constants
from species.read import read_model, read_object, read_planck, read_filter
from species.util import read_util, dust_util


warnings.filterwarnings("always", category=DeprecationWarning)


@typechecked
def lnprior(
    param: np.ndarray,
    bounds: dict,
    param_index: Dict[str, int],
    prior: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    Internal function for calculating the log prior.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Dictionary with the parameter boundaries.
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    prior : dict(str, tuple(float, float)), None
        Dictionary with Gaussian priors for one or multiple parameters.
        The prior can be set for any of the atmosphere or calibration
        parameters, e.g. ``prior={'teff': (1200., 100.)}``.
        Additionally, a prior can be set for the mass, e.g.
        ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup
        with an uncertainty of 3 Mjup. The parameter is not used if set
        to ``None``.

    Returns
    -------
    floats
        Log prior.
    """

    ln_prior = 0

    for key, value in bounds.items():

        if value[0] <= param[param_index[key]] <= value[1]:
            ln_prior += 0.0

        else:
            ln_prior = -np.inf
            break

    if prior is not None:
        for key, value in prior.items():
            if key == "mass":
                mass = read_util.get_mass(
                    param[param_index["logg"]], param[param_index["radius"]]
                )

                ln_prior += -0.5 * (mass - value[0]) ** 2 / value[1] ** 2

            else:
                ln_prior += (
                    -0.5 * (param[param_index[key]] - value[0]) ** 2 / value[1] ** 2
                )

    return ln_prior


@typechecked
def lnlike(
    param: np.ndarray,
    bounds: dict,
    param_index: Dict[str, int],
    model: str,
    objphot: List[Optional[np.ndarray]],
    distance: Tuple[float, float],
    spectrum: Optional[dict],
    modelphot: Optional[
        Union[List[read_model.ReadModel], List[photometry.SyntheticPhotometry]]
    ],
    modelspec: Optional[List[read_model.ReadModel]],
    n_planck: int,
    fit_corr: List[str],
):
    """
    Internal function for calculating the log likelihood.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Dictionary with the parameter boundaries.
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    model : str
        Atmosphere model (e.g. 'bt-settl', 'exo-rem', or 'planck).
    objphot : list(np.ndarray)
        List with the photometric fluxes and uncertainties of the
        object. Not photometric data is fitted if an empty list is
        provided.
    distance : tuple(float, float)
        Distance and uncertainty (pc).
    spectrum : dict(str, tuple(np.ndarray, np.ndarray, np.ndarray, float)), None
        Dictionary with the spectra stored as wavelength (um), flux
        (W m-2 um-1), and error (W m-2 um-1). Optionally the covariance
        matrix, the inverse of the covariance matrix, and the spectral
        resolution are included. Each of these three elements can be
        set to ``None``. No spectroscopic data is fitted if
        ``spectrum=None``.
    modelphot : list(species.read.read_model.ReadModel),
        list(species.analysis.photometry.SyntheticPhotometry), None
        List with the interpolated synthetic fluxes or list with the
        :class:`~species.analysis.photometry.SyntheticPhotometry`
        objects for calculation of synthetic photometry for Planck
        spectra. No photometry is fitted if set to ``None``.
    modelspec : list(species.read.read_model.ReadModel), None
        List with the interpolated synthetic spectra.
    n_planck : int
        Number of Planck components. The argument is set to zero
        if ``model`` is not equal to ``'planck'``.
    fit_corr : list(str)
        List with spectrum names for which the covariances are modeled
        with a Gaussian process (see Wang et al. 2020). This option can
        be used if the actual covariances as determined from the data
        are not available. The parameters that will be fitted are the
        correlation length and fractional amplitude.

    Returns
    -------
    float
        Log likelihood.
    """

    param_dict = {}
    spec_scaling = {}
    err_scaling = {}
    corr_len = {}
    corr_amp = {}

    for item in bounds:
        if item[:8] == "scaling_" and item[8:] in spectrum:
            spec_scaling[item[8:]] = param[param_index[item]]

        elif item[:6] == "error_" and item[6:] in spectrum:
            err_scaling[item[6:]] = param[param_index[item]]

        elif item[:9] == "corr_len_" and item[9:] in spectrum:
            corr_len[item[9:]] = 10.0 ** param[param_index[item]]  # (um)

        elif item[:9] == "corr_amp_" and item[9:] in spectrum:
            corr_amp[item[9:]] = param[param_index[item]]

        else:
            param_dict[item] = param[param_index[item]]

    if model == "planck":
        param_dict["distance"] = param[param_index["distance"]]

    else:
        flux_scaling = (param_dict["radius"] * constants.R_JUP) ** 2 / (
            param_dict["distance"] * constants.PARSEC
        ) ** 2

        # The scaling is applied manually because of the interpolation
        del param_dict["radius"]

    for item in spectrum:
        if item not in spec_scaling:
            spec_scaling[item] = 1.0

        if item not in err_scaling:
            err_scaling[item] = None

    ln_like = 0.0

    if model == "planck" and n_planck > 1:
        for i in range(n_planck - 1):
            if param_dict[f"teff_{i+1}"] > param_dict[f"teff_{i}"]:
                return -np.inf

            if param_dict[f"radius_{i}"] > param_dict[f"radius_{i+1}"]:
                return -np.inf

    for i, obj_item in enumerate(objphot):
        if model == "planck":
            readplanck = read_planck.ReadPlanck(filter_name=modelphot[i].filter_name)
            phot_flux = readplanck.get_flux(param_dict, synphot=modelphot[i])[0]

        else:
            phot_flux = modelphot[i].spectrum_interp(list(param_dict.values()))
            phot_flux *= flux_scaling

        if obj_item.ndim == 1:
            ln_like += -0.5 * (obj_item[0] - phot_flux) ** 2 / obj_item[1] ** 2

        else:
            for j in range(obj_item.shape[1]):
                ln_like += (
                    -0.5 * (obj_item[0, j] - phot_flux) ** 2 / obj_item[1, j] ** 2
                )

    for i, item in enumerate(spectrum.keys()):
        # Calculate or interpolate the model spectrum
        if model == "planck":
            # Calculate a blackbody spectrum
            readplanck = read_planck.ReadPlanck(
                (0.9 * spectrum[item][0][0, 0], 1.1 * spectrum[item][0][-1, 0])
            )

            model_box = readplanck.get_spectrum(param_dict, 1000.0, smooth=True)

            # Resample the spectrum to the observed wavelengths
            model_flux = spectres.spectres(
                spectrum[item][0][:, 0], model_box.wavelength, model_box.flux
            )

        else:
            # Interpolate the model spectrum
            model_flux = modelspec[i].spectrum_interp(list(param_dict.values()))[0, :]

            # Scale the spectrum by the (radius/distance)^2
            model_flux *= flux_scaling

        # Scale the spectrum data
        data_flux = spec_scaling[item] * spectrum[item][0][:, 1]

        if err_scaling[item] is None:
            # Variance without error inflation
            data_var = spectrum[item][0][:, 2] ** 2

        else:
            # Variance with error inflation (see Piette & Madhusudhan 2020)
            data_var = (
                spectrum[item][0][:, 2] ** 2 + (err_scaling[item] * model_flux) ** 2
            )

        if spectrum[item][2] is not None:
            # The inverted covariance matrix is available

            if err_scaling[item] is None:
                # Use the inverted covariance matrix directly
                data_cov_inv = spectrum[item][2]

            else:
                # Ratio of the inflated and original uncertainties
                sigma_ratio = np.sqrt(data_var) / spectrum[item][0][:, 2]
                sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                # Calculate the inverted matrix of the inflated covariances
                data_cov_inv = np.linalg.inv(spectrum[item][1] * sigma_i * sigma_j)

        if spectrum[item][2] is not None:
            # Calculate the log-likelihood with the covariance matrix
            dot_tmp = np.dot(
                data_flux - model_flux, np.dot(data_cov_inv, data_flux - model_flux)
            )
            ln_like += -0.5 * dot_tmp - 0.5 * np.nansum(np.log(2.0 * np.pi * data_var))

        else:
            if item in fit_corr:
                # Calculate the log-likelihood with the
                # covariance model (see Wang et al. 2020)
                wavel = spectrum[item][0][:, 0]  # (um)
                wavel_j, wavel_i = np.meshgrid(wavel, wavel)

                error = np.sqrt(data_var)  # (W m-2 um-1)
                error_j, error_i = np.meshgrid(error, error)

                cov_matrix = (
                    corr_amp[item] ** 2
                    * error_i
                    * error_j
                    * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len[item] ** 2))
                    + (1.0 - corr_amp[item] ** 2)
                    * np.eye(wavel.shape[0])
                    * error_i ** 2
                )

                dot_tmp = np.dot(
                    data_flux - model_flux,
                    np.dot(np.linalg.inv(cov_matrix), data_flux - model_flux),
                )

                ln_like += -0.5 * dot_tmp - 0.5 * np.nansum(
                    np.log(2.0 * np.pi * data_var)
                )

            else:
                # Calculate the log-likelihood without the covariance matrix but with the
                # normalization term in case of the errors are inflated
                ln_like += np.nansum(
                    -0.5 * (data_flux - model_flux) ** 2 / data_var
                    - 0.5 * np.log(2.0 * np.pi * data_var)
                )

    return ln_like


@typechecked
def lnprob(
    param: np.ndarray,
    bounds: dict,
    model: str,
    param_index: Dict[str, int],
    objphot: List[Optional[np.ndarray]],
    distance: Tuple[float, float],
    prior: Optional[Dict[str, Tuple[float, float]]],
    spectrum: Optional[dict],
    modelphot: Optional[
        Union[List[read_model.ReadModel], List[photometry.SyntheticPhotometry]]
    ],
    modelspec: Optional[List[read_model.ReadModel]],
    n_planck: int,
    fit_corr: List[str],
) -> np.float64:
    """
    Internal function for calculating the log posterior.

    Parameters
    ----------
    param : np.ndarray
        Parameter values.
    bounds : dict
        Parameter boundaries.
    model : str
        Atmosphere model (e.g. 'bt-settl', 'exo-rem', or 'planck).
    param_index : dict(str, int)
        Dictionary with the parameter indices of ``param``.
    objphot : list(np.ndarray), None
        List with the photometric fluxes and uncertainties. No
        photometric data is fitted if the parameter is set to ``None``.
    distance : tuple(float, float)
        Distance and uncertainty (pc).
    prior : dict(str, tuple(float, float)), None
        Dictionary with Gaussian priors for one or multiple parameters.
        The prior can be set for any of the atmosphere or calibration
        parameters, e.g. ``prior={'teff': (1200., 100.)}``.
        Additionally, a prior can be set for the mass, e.g.
        ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup
        with an uncertainty of 3 Mjup. The parameter is not used if set
        to ``None``.
    spectrum : dict(str, tuple(np.ndarray, np.ndarray, np.ndarray, float)), None
        Dictionary with the spectra stored as wavelength (um), flux
        (W m-2 um-1), and error (W m-2 um-1). Optionally the covariance
        matrix, the inverse of the covariance matrix, and the spectral
        resolution are included. Each of these three elements can be
        set to ``None``. No spectroscopic data is fitted if
        ``spectrum=None``.
    modelphot : list(species.read.read_model.ReadModel),
        list(species.analysis.photometry.SyntheticPhotometry), None
        List with the interpolated synthetic fluxes or list with the
        :class:`~species.analysis.photometry.SyntheticPhotometry`
        objects for calculation of synthetic photometry for Planck
        spectra. No photometric data is fitted if set to ``None``.
    modelspec : list(species.read.read_model.ReadModel), None
        List with the interpolated synthetic spectra. The parameter is
        set to ``None`` when no spectroscopic data is included or when
        ``model='planck'``.
    n_planck : int
        Number of Planck components. The argument is set to zero
        if ``model`` is not equal to ``'planck'``.
    fit_corr : list(str)
        List with spectrum names for which the covariances are modeled
        with a Gaussian process (see Wang et al. 2020). This option can
        be used if the actual covariances as determined from the data
        are not available. The parameters that will be fitted are the
        correlation length and fractional amplitude.

    Returns
    -------
    float
        Log posterior.
    """

    ln_prior = lnprior(param, bounds, param_index, prior)

    if math.isinf(ln_prior):
        ln_prob = -np.inf

    else:
        ln_prob = ln_prior + lnlike(
            param,
            bounds,
            param_index,
            model,
            objphot,
            distance,
            spectrum,
            modelphot,
            modelspec,
            n_planck,
            fit_corr,
        )

    if np.isnan(ln_prob):
        ln_prob = -np.inf

    return ln_prob


class FitModel:
    """
    Class for fitting atmospheric model spectra to spectra and/or
    photometric fluxes, and using Bayesian inference (``MultiNest``,
    ``UltraNest``, or ``emcee``) to estimate the posterior
    distribution and marginalized likelihood (i.e. "evidence"). The
    latter is only output from the two nested sampling algorithms.
    A grid of model spectra is linearly interpolated for each
    spectrum and photometric flux, while taking into account the
    filter profile, spectral resolution, and wavelength sampling.
    The computation time depends mostly on the number of
    free parameters and the resolution / number of data points
    of the spectra.
    """

    @typechecked
    def __init__(
        self,
        object_name: str,
        model: str,
        bounds: Optional[Dict[
            str,
            Union[
                Tuple[float, float],
                Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
                List[Tuple[float, float]],
            ],
        ]] = None,
        inc_phot: Union[bool, List[str]] = True,
        inc_spec: Union[bool, List[str]] = True,
        fit_corr: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name of the companion as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        model : str
            Name of the atmospheric model (e.g. 'bt-settl', 'exo-rem',
            'planck', or 'powerlaw').
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform priors.
            Fixing a parameter is possible by providing the same
            value as lower and upper boundary of the parameter
            (e.g. ``bounds={'logg': (4., 4.)``. An explanation of
            the various parameters can be found below.

            Atmospheric model parameters (e.g. with
            ``model='bt-settl-cifist'``; see docstring of
            :meth:`~species.data.database.Database.add_model`
            for the available model grids):

               - Boundaries are provided as tuple of two floats. For example,
                 ``bounds={'teff': (1000, 1500.), 'logg': (3.5, 5.)}``.

               - The grid boundaries (i.e. the maximum range) are
                 adopted as priors if a parameter range is set to
                 ``None`` or if a mandatory parameter is not included
                 in the dictionary of ``bounds``. For example,
                 ``bounds={'teff': (1000., 1500.), 'logg': None}``.
                 The default range for the radius is
                 :math:`0.5-5.0~R_\\mathrm{J}`. With ``bounds=None``,
                 automatic priors will be set for all mandatory
                 parameters.

               - It is possible to fit a weighted combination of two
                 atmospheric parameters from the same model. This
                 can be useful to fit data of a spectroscopic binary
                 or to account for atmospheric asymmetries of a single
                 object. For each atmospheric parameter, a tuple of
                 two tuples can be provided, for example
                 ``bounds={'teff': ((1000., 1500.), (1300., 1800.))}``.
                 Mandatory parameters that are not included are assumed
                 to be the same for both components. The grid boundaries
                 are used as parameter range if a component is set to
                 ``None``. For example, ``bounds={'teff': (None, None),
                 'logg': (4.0, 4.0), (4.5, 4.5)}`` will use the full
                 range for :math:`T_\\mathrm{eff}` of both components
                 and fixes :math:`\\log{g}` to 4.0 and 4.5,
                 respectively. The ``spec_weight`` parameter is
                 automatically included in the fit, as it sets the
                 weight of the two components. Modeling blended data
                 is only supported by ``run_ultranest`` and
                 ``run_multinest``.

            Blackbody parameters (with ``model='planck'``):

               - Parameter boundaries have to be provided for 'teff'
                 and 'radius'.

               - For a single blackbody component, the values are
                 provided as a tuple with two floats. For example,
                 ``bounds={'teff': (1000., 2000.),
                 'radius': (0.8, 1.2)}``.

               - For multiple blackbody components, the values are
                 provided as a list with tuples. For example,
                 ``bounds={'teff': [(1000., 1400.), (1200., 1600.)],
                 'radius': [(0.8, 1.5), (1.2, 2.)]}``.

               - When fitting multiple blackbody components, an
                 additional prior is used for restricting the
                 temperatures and radii to decreasing and increasing
                 values, respectively, in the order as provided in
                 ``bounds``.

            Power-law spectrum (``model='powerlaw'``):

               - Parameter boundaries have to be provided for
                 'log_powerlaw_a', 'log_powerlaw_b', and
                 'log_powerlaw_c'. For example,
                 ``bounds={'log_powerlaw_a': (-20., 0.),
                 'log_powerlaw_b': (-20., 5.), 'log_powerlaw_c':
                 (-20., 5.)}``.

               - The spectrum is parametrized as :math:`\\log10{f} =
                 a + b*\\log10{\\lambda}^c`, where :math:`a` is
                 ``log_powerlaw_a``, :math:`b` is ``log_powerlaw_b``,
                 and :math:`c` is ``log_powerlaw_c``.

               - Only implemented for fitting photometric fluxes, for
                 example the IR fluxes of a star with disk. In that way,
                 synthetic photometry can be calculated afterwards for
                 a different filter. Note that this option assumes that
                 the photometric fluxes are dominated by continuum
                 emission while spectral lines are ignored.

               - The :func:`~species.plot.plot_mcmc.plot_mag_posterior`
                 function can be used for calculating synthetic
                 photometry and error bars from the posterior
                 distributions.

               - Only supported by ``run_ultranest`` and
                 ``run_multinest``.

            Calibration parameters:

                 - For each spectrum/instrument, two optional parameters can be fitted to account
                   for biases in the calibration: a scaling of the flux and a constant inflation of
                   the uncertainties.

                 - For example, ``bounds={'SPHERE': ((0.8, 1.2), (0., 1.))}`` if the scaling is
                   fitted between 0.8 and 1.2, and the error is inflated (relative to the sampled
                   model fluxes) with a value between 0 and 1.

                 - The dictionary key should be equal to the database tag of the spectrum. For
                   example, ``{'SPHERE': ((0.8, 1.2), (0., 1.))}`` if the spectrum is stored as
                   ``'SPHERE'`` with :func:`~species.data.database.Database.add_object`.

                 - Each of the two calibration parameters can be set to ``None`` in which case the
                   parameter is not used. For example, ``bounds={'SPHERE': ((0.8, 1.2), None)}``.

                 - The errors of the photometric fluxes can be inflated to account for
                   underestimated error bars. The error inflation is relative to the actual flux
                   and is either fitted separately for a filter, or a single error inflation
                   is applied to all filters from an instrument. For the first case, the keyword
                   in the ``bounds`` dictionary should be provided in the following format:
                   ``'Paranal/NACO.Mp_error': (0., 1.)``. Here, the error of the NACO Mp
                   flux is inflated up to 100 percent of the actual flux. For the second case,
                   only the telescope/instrument part of the the filter name should be provided
                   in the ``bounds`` dictionary, so in the following format:
                   ``'Paranal/NACO_error': (0., 1.)``. This will increase the errors of all
                   NACO filters by the same (relative) amount.

                 - No calibration parameters are fitted if the spectrum name is not included in
                   ``bounds``.

            ISM extinction parameters:

                 - There are three approaches for fitting extinction.
                   The first is with the empirical relation from
                   Cardelli et al. (1989) for ISM extinction.

                 - The extinction is parametrized by the $V$ band
                   extinction, $A_V$ (``ism_ext``), and optionally the
                   reddening, R_V (``ism_red``). If ``ism_red`` is not
                   provided, its value is fixed to 3.1 and not fitted.

                 - The prior boundaries of ``ism_ext`` and ``ism_red``
                   should be provided in the ``bounds`` dictionary, for
                   example ``bounds={'ism_ext': (0., 10.),
                   'ism_red': (0., 20.)}``.

                 - Only supported by ``run_ultranest`` and
                   ``run_multinest``.

            Log-normal size distribution:

                 - The second approach is fitting the extinction of a
                   log-normal size distribution of grains with a
                   crystalline MgSiO3 composition, and a homogeneous,
                   spherical structure.

                 - The size distribution is parameterized with a mean
                   geometric radius (``lognorm_radius`` in um) and a
                   geometric standard deviation (``lognorm_sigma``,
                   dimensionless). The grid of cross sections has been
                   calculated for mean geometric radii between 0.001
                   and 10 um, and geometric standard deviations between
                   1.1 and 10.

                 - The extinction (``lognorm_ext``) is fitted in the
                   $V$ band ($A_V$ in mag) and the wavelength-dependent
                   extinction cross sections are interpolated from a
                   pre-tabulated grid.

                 - The prior boundaries of ``lognorm_radius``,
                   ``lognorm_sigma``, and ``lognorm_ext`` should be
                   provided in the ``bounds`` dictionary, for example
                   ``bounds={'lognorm_radius': (0.001, 10.),
                   'lognorm_sigma': (1.1, 10.),
                   'lognorm_ext': (0., 5.)}``.

                 - A uniform prior is used for ``lognorm_sigma`` and
                   ``lognorm_ext``, and a log-uniform prior for
                   ``lognorm_radius``.

                 - Only supported by ``run_ultranest`` and
                   ``run_multinest``.

            Power-law size distribution:

                 - The third approach is fitting the extinction of a
                   power-law size distribution of grains, again with a
                   crystalline MgSiO3 composition, and a homogeneous,
                   spherical structure.

                 - The size distribution is parameterized with a
                   maximum radius (``powerlaw_max`` in um) and a
                   power-law exponent (``powerlaw_exp``,
                   dimensionless). The minimum radius is fixed to 1 nm.
                   The grid of cross sections has been calculated for
                   maximum radii between 0.01 and 100 um, and power-law
                   exponents between -10 and 10.

                 - The extinction (``powerlaw_ext``) is fitted in the
                   $V$ band ($A_V$ in mag) and the wavelength-dependent
                   extinction cross sections are interpolated from a
                   pre-tabulated grid.

                 - The prior boundaries of ``powerlaw_max``,
                   ``powerlaw_exp``, and ``powerlaw_ext`` should be
                   provided in the ``bounds`` dictionary, for example
                   ``{'powerlaw_max': (0.01, 100.), 'powerlaw_exp':
                   (-10., 10.), 'powerlaw_ext': (0., 5.)}``.

                 - A uniform prior is used for ``powerlaw_exp`` and
                   ``powerlaw_ext``, and a log-uniform prior for
                   ``powerlaw_max``.

                 - Only supported by ``run_ultranest`` and ``run_multinest``.

            Blackbody disk emission:

                 - Additional blackbody emission can be added to the
                   atmospheric spectrum to account for thermal emission
                   from a disk.

                 - Parameter boundaries have to be provided for
                   'disk_teff' and 'disk_radius'. For example,
                   ``bounds={'teff': (2000., 3000.), 'radius': (1., 5.),
                   'logg': (3.5, 4.5), 'disk_teff': (100., 2000.),
                   'disk_radius': (1., 100.)}``.

                 - Only supported by ``run_ultranest`` and
                   ``run_multinest``.

        inc_phot : bool, list(str)
            Include photometric data in the fit. If a boolean, either
            all (``True``) or none (``False``) of the data are
            selected. If a list, a subset of filter names (as stored in
            the database) can be provided.
        inc_spec : bool, list(str)
            Include spectroscopic data in the fit. If a boolean, either
            all (``True``) or none (``False``) of the data are
            selected. If a list, a subset of spectrum names (as stored
            in the database with
            :func:`~species.data.database.Database.add_object`) can be
            provided.
        fit_corr : list(str), None
            List with spectrum names for which the covariances are
            modeled with a Gaussian process (see Wang et al. 2020).
            This option can be used if the actual covariances as
            determined from the data are not available for the spectra
            of ``object_name``. The parameters that will be fitted
            are the correlation length and the fractional amplitude.
        weights : dict(str, float), None
            Weights to be applied to the log-likelihood components of
            the different spectroscopic and photometric data that are
            provided with ``inc_spec`` and ``inc_phot``. This parameter
            can for example be used to bias the weighting of the
            photometric data points. An equal weighting is applied if
            the argument is set to ``None``. Only supported by
            ``run_ultranest`` and ``run_multinest``.

        Returns
        -------
        NoneType
            None
        """

        if not inc_phot and not inc_spec:
            raise ValueError("No photometric or spectroscopic data has been selected.")

        if model == "planck" and ("teff" not in bounds or "radius" not in bounds):
            raise ValueError(
                "The 'bounds' dictionary should contain 'teff' and 'radius'."
            )

        self.object = read_object.ReadObject(object_name)
        self.distance = self.object.get_distance()
        self.binary = False

        if fit_corr is None:
            self.fit_corr = []
        else:
            self.fit_corr = fit_corr

        self.model = model
        self.bounds = bounds

        if self.model == "bt-settl":
            warnings.warn("It is recommended to use the CIFIST "
                          "grid of the BT-Settl, because it is "
                          "a newer version. In that case, set "
                          "model='bt-settl-cifist' when using "
                          "add_model of Database.")

        if self.model == "planck":
            # Fitting blackbody radiation
            if isinstance(bounds["teff"], list) and isinstance(bounds["radius"], list):
                # Update temperature and radius parameters in case of multiple blackbody components
                self.n_planck = len(bounds["teff"])

                self.modelpar = []
                self.bounds = {}

                for i, item in enumerate(bounds["teff"]):
                    self.modelpar.append(f"teff_{i}")
                    self.modelpar.append(f"radius_{i}")

                    self.bounds[f"teff_{i}"] = bounds["teff"][i]
                    self.bounds[f"radius_{i}"] = bounds["radius"][i]

            else:
                # Fitting a single blackbody component
                self.n_planck = 1

                self.modelpar = ["teff", "radius"]
                self.bounds = bounds

            self.modelpar.append("distance")

        elif self.model == "powerlaw":
            self.n_planck = 0

            self.modelpar = ["log_powerlaw_a", "log_powerlaw_b", "log_powerlaw_c"]

        else:
            # Fitting self-consistent atmospheric models
            if self.bounds is not None:
                readmodel = read_model.ReadModel(self.model)
                bounds_grid = readmodel.get_bounds()

                for key, value in bounds_grid.items():

                    if key not in self.bounds:
                        # Set the parameter boundaries to the grid
                        # boundaries if set to None or not found
                        self.bounds[key] = bounds_grid[key]

                    elif isinstance(self.bounds[key][0], tuple):
                        self.binary = True
                        self.bounds[f"{key}_0"] = self.bounds[key][0]
                        self.bounds[f"{key}_1"] = self.bounds[key][1]
                        del self.bounds[key]

                    elif self.bounds[key][0] is None and self.bounds[key][1] is None:
                        self.binary = True
                        self.bounds[f"{key}_0"] = bounds_grid[key]
                        self.bounds[f"{key}_1"] = bounds_grid[key]
                        del self.bounds[key]

                    elif isinstance(self.bounds[key][0], tuple):
                        self.binary = True
                        self.bounds[f"{key}_0"] = self.bounds[key][0]
                        self.bounds[f"{key}_1"] = self.bounds[key][1]
                        del self.bounds[key]

                    else:
                        if self.bounds[key][0] < bounds_grid[key][0]:
                            warnings.warn(
                                f"The lower bound on {key} "
                                f"({self.bounds[key][0]}) is smaller than "
                                f"the lower bound from the available "
                                f"{self.model} model grid "
                                f"({bounds_grid[key][0]}). The lower bound "
                                f"of the {key} prior will be adjusted to "
                                f"{bounds_grid[key][0]}."
                            )
                            self.bounds[key] = (
                                bounds_grid[key][0],
                                self.bounds[key][1],
                            )

                        if self.bounds[key][1] > bounds_grid[key][1]:
                            warnings.warn(
                                f"The upper bound on {key} "
                                f"({self.bounds[key][1]}) is larger than the "
                                f"upper bound from the available {self.model} "
                                f"model grid ({bounds_grid[key][1]}). The "
                                f"bound of the {key} prior will be adjusted "
                                f"to {bounds_grid[key][1]}."
                            )
                            self.bounds[key] = (
                                self.bounds[key][0],
                                bounds_grid[key][1],
                            )

            else:
                # Set all parameter boundaries to the grid boundaries
                readmodel = read_model.ReadModel(self.model, None, None)
                self.bounds = readmodel.get_bounds()

            self.modelpar = readmodel.get_parameters()
            self.modelpar.append("radius")
            self.modelpar.append("distance")

            if self.binary:
                if "radius" in self.bounds:
                    if isinstance(self.bounds["radius"][0], tuple):
                        self.bounds["radius_0"] = self.bounds["radius"][0]
                        self.bounds["radius_1"] = self.bounds["radius"][1]
                        del self.bounds["radius"]

                else:
                    self.bounds["radius"] = (0.5, 5.0)

            elif "radius" not in self.bounds:
                self.bounds["radius"] = (0.5, 5.0)

            self.n_planck = 0

            if "disk_teff" in self.bounds and "disk_radius" in self.bounds:
                self.modelpar.append("disk_teff")
                self.modelpar.append("disk_radius")

            if self.binary:
                # Update list of model parameters

                for key in bounds:
                    if key[:-2] in self.modelpar:
                        par_index = self.modelpar.index(key[:-2])
                        self.modelpar[par_index] = key[:-2] + "_0"
                        self.modelpar.insert(par_index, key[:-2] + "_1")

                self.modelpar.append("spec_weight")

                if "spec_weight" not in self.bounds:
                    self.bounds["spec_weight"] = (0.0, 1.0)

        # Select filters and spectra

        if isinstance(inc_phot, bool):
            if inc_phot:
                # Select all filters if True
                species_db = database.Database()
                objectbox = species_db.get_object(object_name)
                inc_phot = objectbox.filters

            else:
                inc_phot = []

        if isinstance(inc_spec, bool):
            if inc_spec:
                # Select all filters if True
                species_db = database.Database()
                objectbox = species_db.get_object(object_name)
                inc_spec = list(objectbox.spectrum.keys())

            else:
                inc_spec = []

        if inc_spec and self.model == "powerlaw":
            warnings.warn(
                "The 'inc_spec' parameter is not supported when fitting a "
                "power-law spectrum to photometric data. The argument of "
                "'inc_spec' is therefore ignored."
            )

            inc_spec = []

        # Include photometric data

        self.objphot = []
        self.modelphot = []
        self.filter_name = []
        self.instr_name = []

        for item in inc_phot:
            if self.model == "planck":
                # Create SyntheticPhotometry objects when fitting a Planck function
                print(f"Creating synthetic photometry: {item}...", end="", flush=True)
                self.modelphot.append(photometry.SyntheticPhotometry(item))
                print(" [DONE]")

            elif self.model == "powerlaw":
                # Or create SyntheticPhotometry objects when fitting a power-law function
                synphot = photometry.SyntheticPhotometry(item)

                # Set the wavelength range of the filter as attribute
                synphot.zero_point()

                self.modelphot.append(synphot)

            else:
                # Or interpolate the model grid for each filter
                print(f"Interpolating {item}...", end="", flush=True)
                readmodel = read_model.ReadModel(self.model, filter_name=item)
                readmodel.interpolate_grid(
                    wavel_resample=None, smooth=False, spec_res=None
                )
                self.modelphot.append(readmodel)
                print(" [DONE]")

            # Add parameter for error inflation

            instr_filt = item.split(".")[0]

            if f"{item}_error" in self.bounds:
                self.modelpar.append(f"{item}_error")

            elif (
                f"{instr_filt}_error" in self.bounds
                and f"{instr_filt}_error" not in self.modelpar
            ):
                self.modelpar.append(f"{instr_filt}_error")

            # Store the flux and uncertainty for each filter
            obj_phot = self.object.get_photometry(item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

            self.filter_name.append(item)
            self.instr_name.append(instr_filt)

        # Include spectroscopic data

        if inc_spec:
            # Select all spectra
            self.spectrum = self.object.get_spectrum()

            # Select the spectrum names that are not in inc_spec
            spec_remove = []
            for item in self.spectrum:
                if item not in inc_spec:
                    spec_remove.append(item)

            # Remove the spectra that are not included in inc_spec
            for item in spec_remove:
                del self.spectrum[item]

            self.n_corr_par = 0

            for item in self.spectrum:
                if item in self.fit_corr:
                    if self.spectrum[item][1] is not None:
                        warnings.warn(
                            f"There is a covariance matrix included "
                            f"with the {item} data of "
                            f"{object_name} so it is not needed to "
                            f"model the covariances with a "
                            f"Gaussian process. Want to test the "
                            f"Gaussian process nonetheless? Please "
                            f"overwrite the data of {object_name} "
                            f"with add_object while setting the "
                            f"path to the covariance data to None."
                        )

                        self.fit_corr.remove(item)

                    else:
                        self.modelpar.append(f"corr_len_{item}")
                        self.modelpar.append(f"corr_amp_{item}")

                        if f"corr_len_{item}" not in self.bounds:
                            self.bounds[f"corr_len_{item}"] = (
                                -3.0,
                                0.0,
                            )  # log10(corr_len/um)

                        if f"corr_amp_{item}" not in self.bounds:
                            self.bounds[f"corr_amp_{item}"] = (0.0, 1.0)

                        self.n_corr_par += 2

            self.modelspec = []

            if self.model != "planck":

                for key, value in self.spectrum.items():
                    print(f"\rInterpolating {key}...", end="", flush=True)

                    wavel_range = (0.9 * value[0][0, 0], 1.1 * value[0][-1, 0])

                    readmodel = read_model.ReadModel(
                        self.model, wavel_range=wavel_range
                    )

                    readmodel.interpolate_grid(
                        wavel_resample=self.spectrum[key][0][:, 0],
                        smooth=True,
                        spec_res=self.spectrum[key][3],
                    )

                    self.modelspec.append(readmodel)

                    print(" [DONE]")

        else:
            self.spectrum = {}
            self.modelspec = None
            self.n_corr_par = 0

        # Get the parameter order if interpolate_grid is used

        if self.model not in ["planck", "powerlaw"]:
            readmodel = read_model.ReadModel(self.model)
            self.param_interp = readmodel.get_parameters()

            if self.binary:
                param_tmp = self.param_interp.copy()

                self.param_interp = []
                for item in param_tmp:
                    if f"{item}_0" in self.modelpar and f"{item}_1" in self.modelpar:
                        self.param_interp.append(f"{item}_0")
                        self.param_interp.append(f"{item}_1")

                    else:
                        self.param_interp.append(item)

        else:
            self.param_interp = None

        # Include blackbody disk

        self.diskphot = []
        self.diskspec = []

        if "disk_teff" in self.bounds and "disk_radius" in self.bounds:
            for item in inc_phot:
                print(f"Interpolating {item}...", end="", flush=True)
                readmodel = read_model.ReadModel("blackbody", filter_name=item)
                readmodel.interpolate_grid(
                    wavel_resample=None, smooth=False, spec_res=None
                )
                self.diskphot.append(readmodel)
                print(" [DONE]")

            for key, value in self.spectrum.items():
                print(f"\rInterpolating {key}...", end="", flush=True)

                wavel_range = (0.9 * value[0][0, 0], 1.1 * value[0][-1, 0])

                readmodel = read_model.ReadModel("blackbody", wavel_range=wavel_range)

                readmodel.interpolate_grid(
                    wavel_resample=self.spectrum[key][0][:, 0],
                    smooth=True,
                    spec_res=self.spectrum[key][3],
                )

                self.diskspec.append(readmodel)

                print(" [DONE]")

        for item in self.spectrum:
            if bounds is not None and item in bounds:

                if bounds[item][0] is not None:
                    # Add the flux scaling parameter
                    self.modelpar.append(f"scaling_{item}")
                    self.bounds[f"scaling_{item}"] = (
                        bounds[item][0][0],
                        bounds[item][0][1],
                    )

                if bounds[item][1] is not None:
                    # Add the error inflation parameters
                    self.modelpar.append(f"error_{item}")
                    self.bounds[f"error_{item}"] = (
                        bounds[item][1][0],
                        bounds[item][1][1],
                    )

                    if self.bounds[f"error_{item}"][0] < 0.0:
                        warnings.warn(
                            f"The lower bound of 'error_{item}' is smaller than 0. "
                            f"The error inflation should be given relative to the model "
                            f"fluxes  so the boundaries are typically between 0 and 1."
                        )

                    if self.bounds[f"error_{item}"][1] < 0.0:
                        warnings.warn(
                            f"The upper bound of 'error_{item}' is smaller than 0. "
                            f"The error inflation should be given relative to the model "
                            f"fluxes so the boundaries are typically between 0 and 1."
                        )

                    if self.bounds[f"error_{item}"][0] > 1.0:
                        warnings.warn(
                            f"The lower bound of 'error_{item}' is larger than 1. The "
                            f"error inflation should be given relative to the model "
                            f"fluxes so the boundaries are typically between 0 and 1."
                        )

                    if self.bounds[f"error_{item}"][1] > 1.0:
                        warnings.warn(
                            f"The upper bound of 'error_{item}' is larger than 1. The "
                            f"error inflation should be given relative to the model "
                            f"fluxes so the boundaries are typically between 0 and 1."
                        )

                if item in self.bounds:
                    del self.bounds[item]

        if (
            "lognorm_radius" in self.bounds
            and "lognorm_sigma" in self.bounds
            and "lognorm_ext" in self.bounds
        ):

            self.cross_sections, _, _ = dust_util.interp_lognorm(
                inc_phot, inc_spec, self.spectrum
            )

            self.modelpar.append("lognorm_radius")
            self.modelpar.append("lognorm_sigma")
            self.modelpar.append("lognorm_ext")

            self.bounds["lognorm_radius"] = (
                np.log10(self.bounds["lognorm_radius"][0]),
                np.log10(self.bounds["lognorm_radius"][1]),
            )

        elif (
            "powerlaw_max" in self.bounds
            and "powerlaw_exp" in self.bounds
            and "powerlaw_ext" in self.bounds
        ):

            self.cross_sections, _, _ = dust_util.interp_powerlaw(
                inc_phot, inc_spec, self.spectrum
            )

            self.modelpar.append("powerlaw_max")
            self.modelpar.append("powerlaw_exp")
            self.modelpar.append("powerlaw_ext")

            self.bounds["powerlaw_max"] = (
                np.log10(self.bounds["powerlaw_max"][0]),
                np.log10(self.bounds["powerlaw_max"][1]),
            )

        else:
            self.cross_sections = None

        if "ism_ext" in self.bounds:
            self.modelpar.append("ism_ext")

        if "ism_red" in self.bounds:
            self.modelpar.append("ism_red")

        if "veil_a" in self.bounds:
            self.modelpar.append("veil_a")

        if "veil_b" in self.bounds:
            self.modelpar.append("veil_b")

        if "veil_ref" in self.bounds:
            self.modelpar.append("veil_ref")

        self.fix_param = {}
        del_param = []

        for key, value in self.bounds.items():
            if value[0] == value[1] and value[0] is not None and value[1] is not None:
                self.fix_param[key] = value[0]
                del_param.append(key)

        if del_param:
            print(f"Fixing {len(del_param)} parameters:")

            for item in del_param:
                print(f"   - {item} = {self.fix_param[item]}")

                self.modelpar.remove(item)
                del self.bounds[item]

        print(f"Fitting {len(self.modelpar)} parameters:")

        for item in self.modelpar:
            print(f"   - {item}")

        print("Prior boundaries:")

        for key, value in self.bounds.items():
            print(f"   - {key} = {value}")

        # Create a dictionary with the cube indices of the parameters

        self.cube_index = {}
        for i, item in enumerate(self.modelpar):
            self.cube_index[item] = i

        # Weighting of the photometric and spectroscopic data

        print("Weights for the log-likelihood function:")

        if weights is None:
            self.weights = {}
        else:
            self.weights = weights

        for item in inc_spec:
            if item not in self.weights:
                self.weights[item] = 1.0

            print(f"   - {item} = {self.weights[item]:.2e}")

        for item in inc_phot:
            if item not in self.weights:
                self.weights[item] = 1.0

            print(f"   - {item} = {self.weights[item]:.2e}")

    @typechecked
    def run_mcmc(
        self,
        tag: str,
        guess: Optional[
            Dict[
                str,
                Union[
                    Optional[float],
                    List[Optional[float]],
                    Tuple[Optional[float], Optional[float]],
                ],
            ]
        ],
        nwalkers: int = 200,
        nsteps: int = 1000,
        prior: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Function to run the MCMC sampler of ``emcee``. The
        functionalities of ``run_mcmc`` are more limited than
        :func:`~species.analysis.fit_model.FitModel.run_ultranest` and
        :func:`~species.analysis.fit_model.FitModel.run_multinest`.
        Furthermore, ``run_ultranest`` ``run_multinest`` provide more
        robust results when sampling multimodal posterior distributions
        and have the additional advantage of returning the marginal
        likelihood (i.e. "evidence"). Therefore, it is recommended to
        use ``run_ultranest`` or ``run_multinest`` instead of
        ``run_mcmc``.

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        guess : dict, None
            Guess for each parameter to initialize the walkers. Random
            values between the ``bounds`` are used is set to ``None``.
        nwalkers : int
            Number of walkers.
        nsteps : int
            Number of steps per walker.
        prior : dict(str, tuple(float, float)), None
            Dictionary with Gaussian priors for one or multiple
            parameters. The prior can be set for any of the
            atmosphere or calibration parameters, e.g.
            ``prior={'teff': (1200., 100.)}``. Additionally,
            a prior can be set for the mass, e.g.
            ``prior={'mass': (13., 3.)}`` for an expected mass
            of 13 Mjup with an uncertainty of 3 Mjup. The
            parameter is not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print("Running MCMC...")

        ndim = len(self.bounds)

        if self.model == "planck":

            if "teff" in self.bounds:
                sigma = {"teff": 5.0, "radius": 0.01}

            else:
                sigma = {}

                for i, item in enumerate(guess["teff"]):
                    sigma[f"teff_{i}"] = 5.0
                    guess[f"teff_{i}"] = guess["teff"][i]

                for i, item in enumerate(guess["radius"]):
                    sigma[f"radius_{i}"] = 0.01
                    guess[f"radius_{i}"] = guess["radius"][i]

                del guess["teff"]
                del guess["radius"]

        else:
            sigma = {
                "teff": 5.0,
                "logg": 0.01,
                "feh": 0.01,
                "fsed": 0.01,
                "c_o_ratio": 0.01,
                "radius": 0.01,
            }

        for item in self.spectrum:
            if item in self.fit_corr:
                sigma[f"corr_len_{item}"] = 0.01  # log10(corr_len/um)
                guess[f"corr_len_{item}"] = None  # log10(corr_len/um)

                sigma[f"corr_amp_{item}"] = 0.1
                guess[f"corr_amp_{item}"] = None

        for item in self.spectrum:
            if f"scaling_{item}" in self.bounds:
                sigma[f"scaling_{item}"] = 0.01
                guess[f"scaling_{item}"] = guess[item][0]

            if f"error_{item}" in self.bounds:
                sigma[f"error_{item}"] = 0.1
                guess[f"error_{item}"] = guess[item][1]

            if item in guess:
                del guess[item]

        initial = np.zeros((nwalkers, ndim))

        for i, item in enumerate(self.modelpar):
            if guess[item] is not None:
                initial[:, i] = guess[item] + np.random.normal(0, sigma[item], nwalkers)

            else:
                initial[:, i] = np.random.uniform(
                    low=self.bounds[item][0], high=self.bounds[item][1], size=nwalkers
                )

        # Create a dictionary with the indices of the parameters

        param_index = {}
        for i, item in enumerate(self.modelpar):
            param_index[item] = i

        with Pool(processes=cpu_count()):

            ens_sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                lnprob,
                args=(
                    [
                        self.bounds,
                        self.model,
                        param_index,
                        self.objphot,
                        self.distance,
                        prior,
                        self.spectrum,
                        self.modelphot,
                        self.modelspec,
                        self.n_planck,
                        self.fit_corr,
                    ]
                ),
            )

            ens_sampler.run_mcmc(initial, nsteps, progress=True)

        spec_labels = []
        for item in self.spectrum:
            if f"scaling_{item}" in self.bounds:
                spec_labels.append(f"scaling_{item}")

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only possible when using a single process
            species_db = database.Database()

            species_db.add_samples(
                sampler="emcee",
                samples=ens_sampler.chain,
                ln_prob=ens_sampler.lnprobability,
                ln_evidence=None,
                mean_accept=np.mean(ens_sampler.acceptance_fraction),
                spectrum=("model", self.model),
                tag=tag,
                modelpar=self.modelpar,
                distance=self.distance[0],
                spec_labels=spec_labels,
            )

    @typechecked
    def lnlike_func(
        self, params, prior: Optional[Dict[str, Tuple[float, float]]]
    ) -> np.float64:
        """
        Function for calculating the log-likelihood for the sampled
        parameter cube.

        Parameters
        ----------
        params : np.ndarray, pymultinest.run.LP_c_double
            Cube with physical parameters.
        prior : dict(str, tuple(float, float))
            Dictionary with Gaussian priors for one or multiple
            parameters. The prior can be set for any of the atmosphere
            or calibration parameters, e.g.
            ``prior={'teff': (1200., 100.)}``. Additionally, a prior
            can be set for the mass, e.g. ``prior={'mass': (13., 3.)}``
            for an expected mass of 13 Mjup with an uncertainty of
            3 Mjup.

        Returns
        -------
        float
            Log-likelihood.
        """

        # Initilize dictionaries for different parameter types

        spec_scaling = {}
        phot_scaling = {}
        err_scaling = {}
        corr_len = {}
        corr_amp = {}
        dust_param = {}
        disk_param = {}
        veil_param = {}
        param_dict = {}

        for item in self.bounds:
            # Add the parameters from the params to their dictionaries

            if item[:8] == "scaling_" and item[8:] in self.spectrum:
                spec_scaling[item[8:]] = params[self.cube_index[item]]

            elif item[:6] == "error_" and item[6:] in self.spectrum:
                err_scaling[item[6:]] = params[self.cube_index[item]]

            elif item[:9] == "corr_len_" and item[9:] in self.spectrum:
                corr_len[item[9:]] = 10.0 ** params[self.cube_index[item]]  # (um)

            elif item[:9] == "corr_amp_" and item[9:] in self.spectrum:
                corr_amp[item[9:]] = params[self.cube_index[item]]

            elif item[-6:] == "_error" and item[:-6] in self.filter_name:
                phot_scaling[item[:-6]] = params[self.cube_index[item]]

            elif item[-6:] == "_error" and item[:-6] in self.instr_name:
                phot_scaling[item[:-6]] = params[self.cube_index[item]]

            elif item[:8] == "lognorm_":
                dust_param[item] = params[self.cube_index[item]]

            elif item[:9] == "powerlaw_":
                dust_param[item] = params[self.cube_index[item]]

            elif item[:4] == "ism_":
                dust_param[item] = params[self.cube_index[item]]

            elif item == "disk_teff":
                disk_param["teff"] = params[self.cube_index[item]]

            elif item == "disk_radius":
                disk_param["radius"] = params[self.cube_index[item]]

            elif item == "veil_a":
                veil_param["veil_a"] = params[self.cube_index[item]]

            elif item == "veil_b":
                veil_param["veil_b"] = params[self.cube_index[item]]

            elif item == "veil_ref":
                veil_param["veil_ref"] = params[self.cube_index[item]]

            elif item == "spec_weight":
                pass

            else:
                param_dict[item] = params[self.cube_index[item]]

        # Add the distance manually because it should
        # not be provided in the bounds dictionary
        param_dict["distance"] = params[self.cube_index["distance"]]

        for item in self.fix_param:
            # Add the fixed parameters to their dictionaries

            if item[:8] == "scaling_" and item[8:] in self.spectrum:
                spec_scaling[item[8:]] = self.fix_param[item]

            elif item[:6] == "error_" and item[6:] in self.spectrum:
                err_scaling[item[6:]] = self.fix_param[item]

            elif item[:9] == "corr_len_" and item[9:] in self.spectrum:
                corr_len[item[9:]] = self.fix_param[item]  # (um)

            elif item[:9] == "corr_amp_" and item[9:] in self.spectrum:
                corr_amp[item[9:]] = self.fix_param[item]

            elif item[:8] == "lognorm_":
                dust_param[item] = self.fix_param[item]

            elif item[:9] == "powerlaw_":
                dust_param[item] = self.fix_param[item]

            elif item[:4] == "ism_":
                dust_param[item] = self.fix_param[item]

            elif item == "disk_teff":
                disk_param["teff"] = self.fix_param[item]

            elif item == "disk_radius":
                disk_param["radius"] = self.fix_param[item]

            elif item == "spec_weight":
                pass

            else:
                param_dict[item] = self.fix_param[item]

        if self.model == "planck" and self.n_planck > 1:
            for i in range(self.n_planck - 1):
                if param_dict[f"teff_{i+1}"] > param_dict[f"teff_{i}"]:
                    return -np.inf

                if param_dict[f"radius_{i}"] > param_dict[f"radius_{i+1}"]:
                    return -np.inf

        if disk_param:
            if disk_param["teff"] > param_dict["teff"]:
                return -np.inf

            if disk_param["radius"] < param_dict["radius"]:
                return -np.inf

        if self.model != "powerlaw":
            if "radius_0" in param_dict and "radius_1" in param_dict:
                flux_scaling_0 = (param_dict["radius_0"] * constants.R_JUP) ** 2 / (
                    param_dict["distance"] * constants.PARSEC
                ) ** 2

                flux_scaling_1 = (param_dict["radius_1"] * constants.R_JUP) ** 2 / (
                    param_dict["distance"] * constants.PARSEC
                ) ** 2

                # The scaling is applied manually because of the interpolation
                del param_dict["radius_0"]
                del param_dict["radius_1"]

            else:
                flux_scaling = (param_dict["radius"] * constants.R_JUP) ** 2 / (
                    param_dict["distance"] * constants.PARSEC
                ) ** 2

                # The scaling is applied manually because of the interpolation
                del param_dict["radius"]

        for item in self.spectrum:
            if item not in spec_scaling:
                spec_scaling[item] = 1.0

            if item not in err_scaling:
                err_scaling[item] = None

        if self.param_interp is not None:
            # Sort the parameters in the correct order for spectrum_interp because
            # spectrum_interp creates a list in the order of the keys in param_dict
            param_tmp = param_dict.copy()

            param_dict = {}
            for item in self.param_interp:
                param_dict[item] = param_tmp[item]

        ln_like = 0.0

        for key, value in prior.items():
            if key == "mass":
                mass = read_util.get_mass(
                    params[self.cube_index["logg"]],
                    params[self.cube_index["radius"]],
                )

                ln_like += -0.5 * (mass - value[0]) ** 2 / value[1] ** 2

            else:
                ln_like += (
                    -0.5
                    * (params[self.cube_index[key]] - value[0]) ** 2
                    / value[1] ** 2
                )

        if "lognorm_ext" in dust_param:
            cross_tmp = self.cross_sections["Generic/Bessell.V"](
                dust_param["lognorm_sigma"], 10.0 ** dust_param["lognorm_radius"]
            )[0]

            n_grains = (
                dust_param["lognorm_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
            )

        elif "powerlaw_ext" in dust_param:
            cross_tmp = self.cross_sections["Generic/Bessell.V"](
                dust_param["powerlaw_exp"], 10.0 ** dust_param["powerlaw_max"]
            )

            n_grains = (
                dust_param["powerlaw_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
            )

        for i, obj_item in enumerate(self.objphot):
            # Get filter name
            phot_filter = self.modelphot[i].filter_name

            # Shortcut for weight
            weight = self.weights[phot_filter]

            if self.model == "planck":
                readplanck = read_planck.ReadPlanck(filter_name=phot_filter)
                phot_flux = readplanck.get_flux(param_dict, synphot=self.modelphot[i])[
                    0
                ]

            elif self.model == "powerlaw":
                powerl_box = read_util.powerlaw_spectrum(
                    self.modelphot[i].wavel_range, param_dict
                )

                phot_flux = self.modelphot[i].spectrum_to_flux(
                    powerl_box.wavelength, powerl_box.flux
                )[0]

            else:
                if self.binary:
                    # Star 0

                    param_0 = read_util.binary_to_single(param_dict, 0)

                    phot_flux_0 = self.modelphot[i].spectrum_interp(
                        list(param_0.values())
                    )[0][0]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        phot_flux_0 *= flux_scaling

                    elif "radius_0" in self.modelpar:
                        phot_flux_0 *= flux_scaling_0

                    # Star 1

                    param_1 = read_util.binary_to_single(param_dict, 1)

                    phot_flux_1 = self.modelphot[i].spectrum_interp(
                        list(param_1.values())
                    )[0][0]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        phot_flux_1 *= flux_scaling

                    elif "radius_1" in self.modelpar:
                        phot_flux_1 *= flux_scaling_1

                    # Weighted flux of two stars

                    phot_flux = (
                        params[self.cube_index["spec_weight"]] * phot_flux_0
                        + (1.0 - params[self.cube_index["spec_weight"]]) * phot_flux_1
                    )

                else:
                    phot_flux = self.modelphot[i].spectrum_interp(
                        list(param_dict.values())
                    )[0][0]

                    phot_flux *= flux_scaling

            if disk_param:
                phot_tmp = self.diskphot[i].spectrum_interp([disk_param["teff"]])[0][0]

                phot_flux += (
                    phot_tmp
                    * (disk_param["radius"] * constants.R_JUP) ** 2
                    / (param_dict["distance"] * constants.PARSEC) ** 2
                )

            if "lognorm_ext" in dust_param:
                cross_tmp = self.cross_sections[phot_filter](
                    dust_param["lognorm_sigma"], 10.0 ** dust_param["lognorm_radius"]
                )[0]

                phot_flux *= np.exp(-cross_tmp * n_grains)

            elif "powerlaw_ext" in dust_param:
                cross_tmp = self.cross_sections[phot_filter](
                    dust_param["powerlaw_exp"], 10.0 ** dust_param["powerlaw_max"]
                )[0]

                phot_flux *= np.exp(-cross_tmp * n_grains)

            elif "ism_ext" in dust_param:
                read_filt = read_filter.ReadFilter(phot_filter)
                filt_wavel = np.array([read_filt.mean_wavelength()])

                ism_reddening = dust_param.get("ism_red", 3.1)

                ext_filt = dust_util.ism_extinction(
                    dust_param["ism_ext"], ism_reddening, filt_wavel
                )

                phot_flux *= 10.0 ** (-0.4 * ext_filt[0])

            if obj_item.ndim == 1:
                phot_var = obj_item[1] ** 2

                # Get the telescope/instrument name
                instr_check = phot_filter.split(".")[0]

                if phot_filter in phot_scaling:
                    # Inflate photometric error for filter
                    phot_var += phot_scaling[phot_filter] ** 2 * obj_item[0] ** 2

                elif instr_check in phot_scaling:
                    # Inflate photometric error for instrument
                    phot_var += phot_scaling[instr_check] ** 2 * obj_item[0] ** 2

                ln_like += -0.5 * weight * (obj_item[0] - phot_flux) ** 2 / phot_var

                # Only required when fitting an error inflation
                ln_like += -0.5 * weight * np.log(2.0 * np.pi * phot_var)

            else:
                for j in range(obj_item.shape[1]):
                    phot_var = obj_item[1, j] ** 2

                    if (
                        self.model == "powerlaw"
                        and f"{phot_filter}_error" in param_dict
                    ):
                        phot_var += (
                            param_dict[f"{phot_filter}_error"] ** 2
                            * obj_item[0, j] ** 2
                        )

                    ln_like += (
                        -0.5 * weight * (obj_item[0, j] - phot_flux) ** 2 / phot_var
                    )

                    # Only required when fitting an error inflation
                    ln_like += -0.5 * weight * np.log(2.0 * np.pi * phot_var)

        for i, item in enumerate(self.spectrum.keys()):
            # Calculate or interpolate the model spectrum

            # Shortcut for the weight
            weight = self.weights[item]

            if self.model == "planck":
                # Calculate a blackbody spectrum
                readplanck = read_planck.ReadPlanck(
                    (
                        0.9 * self.spectrum[item][0][0, 0],
                        1.1 * self.spectrum[item][0][-1, 0],
                    )
                )

                model_box = readplanck.get_spectrum(param_dict, 1000.0, smooth=True)

                # Resample the spectrum to the observed wavelengths
                model_flux = spectres.spectres(
                    self.spectrum[item][0][:, 0], model_box.wavelength, model_box.flux
                )

            else:
                # Interpolate the model spectrum from the grid

                if self.binary:
                    # Star 1

                    param_0 = read_util.binary_to_single(param_dict, 0)

                    model_flux_0 = self.modelspec[i].spectrum_interp(
                        list(param_0.values())
                    )[0, :]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        model_flux_0 *= flux_scaling

                    elif "radius_1" in self.modelpar:
                        model_flux_0 *= flux_scaling_0

                    # Star 2

                    param_1 = read_util.binary_to_single(param_dict, 1)

                    model_flux_1 = self.modelspec[i].spectrum_interp(
                        list(param_1.values())
                    )[0, :]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        model_flux_1 *= flux_scaling

                    elif "radius_1" in self.modelpar:
                        model_flux_1 *= flux_scaling_1

                    # Weighted flux of two stars

                    model_flux = (
                        params[self.cube_index["spec_weight"]] * model_flux_0
                        + (1.0 - params[self.cube_index["spec_weight"]]) * model_flux_1
                    )

                else:
                    model_flux = self.modelspec[i].spectrum_interp(
                        list(param_dict.values())
                    )[0, :]

                    # Scale the spectrum by (radius/distance)^2
                    model_flux *= flux_scaling

            # Veiling
            if (
                "veil_a" in veil_param
                and "veil_b" in veil_param
                and "veil_ref" in veil_param
            ):
                if item == "MUSE":
                    lambda_ref = 0.5  # (um)

                    veil_flux = veil_param["veil_ref"] + veil_param["veil_b"] * (
                        self.spectrum[item][0][:, 0] - lambda_ref
                    )

                    model_flux = veil_param["veil_a"] * model_flux + veil_flux

            # Scale the spectrum data
            data_flux = spec_scaling[item] * self.spectrum[item][0][:, 1]

            if err_scaling[item] is None:
                # Variance without error inflation
                data_var = self.spectrum[item][0][:, 2] ** 2

            else:
                # Variance with error inflation (see Piette & Madhusudhan 2020)
                data_var = (
                    self.spectrum[item][0][:, 2] ** 2
                    + (err_scaling[item] * model_flux) ** 2
                )

            if self.spectrum[item][2] is not None:
                # The inverted covariance matrix is available

                if err_scaling[item] is None:
                    # Use the inverted covariance matrix directly
                    data_cov_inv = self.spectrum[item][2]

                else:
                    # Ratio of the inflated and original uncertainties
                    sigma_ratio = np.sqrt(data_var) / self.spectrum[item][0][:, 2]
                    sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                    # Calculate the inverted matrix of the inflated covariances
                    data_cov_inv = np.linalg.inv(
                        self.spectrum[item][1] * sigma_i * sigma_j
                    )

            if disk_param:
                model_tmp = self.diskspec[i].spectrum_interp([disk_param["teff"]])[0, :]

                model_tmp *= (disk_param["radius"] * constants.R_JUP) ** 2 / (
                    param_dict["distance"] * constants.PARSEC
                ) ** 2

                model_flux += model_tmp

            if "lognorm_ext" in dust_param:
                for j, cross_item in enumerate(self.cross_sections[item]):
                    cross_tmp = cross_item(
                        dust_param["lognorm_sigma"],
                        10.0 ** dust_param["lognorm_radius"],
                    )[0]

                    model_flux[j] *= np.exp(-cross_tmp * n_grains)

            elif "powerlaw_ext" in dust_param:
                for j, cross_item in enumerate(self.cross_sections[item]):
                    # For loop over all wavelengths of a spectrum
                    cross_tmp = cross_item(
                        dust_param["powerlaw_exp"], 10.0 ** dust_param["powerlaw_max"]
                    )[0]

                    model_flux[j] *= np.exp(-cross_tmp * n_grains)

            elif "ism_ext" in dust_param:
                ism_reddening = dust_param.get("ism_red", 3.1)

                ext_filt = dust_util.ism_extinction(
                    dust_param["ism_ext"], ism_reddening, self.spectrum[item][0][:, 0]
                )

                model_flux *= 10.0 ** (-0.4 * ext_filt)

            if self.spectrum[item][2] is not None:
                # Use the inverted covariance matrix
                ln_like += (
                    -0.5
                    * weight
                    * np.dot(
                        data_flux - model_flux,
                        np.dot(data_cov_inv, data_flux - model_flux),
                    )
                )

                ln_like += -0.5 * weight * np.nansum(np.log(2.0 * np.pi * data_var))

            else:
                if item in self.fit_corr:
                    # Covariance model (Wang et al. 2020)
                    wavel = self.spectrum[item][0][:, 0]  # (um)
                    wavel_j, wavel_i = np.meshgrid(wavel, wavel)

                    error = np.sqrt(data_var)  # (W m-2 um-1)
                    error_j, error_i = np.meshgrid(error, error)

                    cov_matrix = (
                        corr_amp[item] ** 2
                        * error_i
                        * error_j
                        * np.exp(
                            -((wavel_i - wavel_j) ** 2) / (2.0 * corr_len[item] ** 2)
                        )
                        + (1.0 - corr_amp[item] ** 2)
                        * np.eye(wavel.shape[0])
                        * error_i ** 2
                    )

                    dot_tmp = np.dot(
                        data_flux - model_flux,
                        np.dot(np.linalg.inv(cov_matrix), data_flux - model_flux),
                    )

                    ln_like += -0.5 * weight * dot_tmp
                    ln_like += -0.5 * np.nansum(np.log(2.0 * np.pi * data_var))

                else:
                    # Calculate the chi-square without a covariance matrix
                    chi_sq = -0.5 * weight * (data_flux - model_flux) ** 2 / data_var
                    chi_sq += -0.5 * weight * np.log(2.0 * np.pi * data_var)

                    ln_like += np.nansum(chi_sq)

        return ln_like

    @typechecked
    def run_multinest(
        self,
        tag: str,
        n_live_points: int = 1000,
        output: str = "multinest/",
        prior: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the
        ``MultiNest`` sampler. While ``PyMultiNest`` can be
        installed with ``pip`` from the PyPI repository,
        ``MultiNest`` has to to be build manually. See the
        ``PyMultiNest`` documentation for details:
        http://johannesbuchner.github.io/PyMultiNest/install.html.
        Note that the library path of ``MultiNest`` should be set
        to the environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac.
        Alternatively, the variable can be set before importing
        the ``species`` package, for example:

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
        prior : dict(str, tuple(float, float)), None
            Dictionary with Gaussian priors for one or multiple
            parameters. The prior can be set for any of the
            atmosphere or calibration parameters, for example
            ``prior={'teff': (1200., 100.)}``. Additionally, a
            prior can be set for the mass, for example
            ``prior={'mass': (13., 3.)}`` for an expected mass
            of 13 Mjup with an uncertainty of 3 Mjup. The
            parameter is not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print("Running nested sampling with MultiNest...")

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        # Add distance to dictionary with Gaussian priors

        if prior is None:
            prior = {}

        prior['distance'] = self.distance

        @typechecked
        def lnprior_multinest(cube, n_dim: int, n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter cube. It is not clear how to
            pass additional arguments to the function, therefore it is placed here and not merged
            with :func:`~species.analysis.fit_model.FitModel.run_mcmc`.

            Parameters
            ----------
            cube : pymultinest.run.LP_c_double
                Unit cube.
            n_dim : int
                Number of dimensions.
            n_param : int
                Number of parameters.

            Returns
            -------
            NoneType
                None
            """

            for item in self.cube_index:
                if item == "distance":
                    # Gaussian prior for the distance
                    cube[self.cube_index[item]] = stats.norm.ppf(
                        cube[self.cube_index[item]],
                        loc=self.distance[0],
                        scale=self.distance[1]
                    )

                else:
                    # Uniform priors for all parameters
                    cube[self.cube_index[item]] = (
                        self.bounds[item][0]
                        + (self.bounds[item][1] - self.bounds[item][0])
                        * cube[self.cube_index[item]]
                    )

        @typechecked
        def lnlike_multinest(params, n_dim: int, n_param: int) -> np.float64:
            """
            Function for return the log-likelihood for the sampled parameter cube.

            Parameters
            ----------
            params : pymultinest.run.LP_c_double
                Cube with physical parameters.
            n_dim : int
                Number of dimensions. This parameter is mandatory but not used by the function.
            n_param : int
                Number of parameters. This parameter is mandatory but not used by the function.

            Returns
            -------
            float
                Log-likelihood.
            """

            return self.lnlike_func(params, prior=prior)

        pymultinest.run(
            lnlike_multinest,
            lnprior_multinest,
            len(self.modelpar),
            outputfiles_basename=output,
            resume=False,
            n_live_points=n_live_points,
        )

        # Create the Analyzer object
        analyzer = pymultinest.analyse.Analyzer(
            len(self.modelpar), outputfiles_basename=output
        )

        # Get a dictionary with the ln(Z) and its errors, the
        # individual modes and their parameters quantiles of
        # the parameter posteriors
        sampling_stats = analyzer.get_stats()

        # Nested sampling global log-evidence
        ln_z = sampling_stats["nested sampling global log-evidence"]
        ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"Nested sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Nested sampling global log-evidence
        ln_z = sampling_stats["nested importance sampling global log-evidence"]
        ln_z_error = sampling_stats["nested importance sampling global log-evidence error"]
        print(
            f"Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}"
        )

        # Get the best-fit (highest likelihood) point
        print("Sample with the highest likelihood:")
        best_params = analyzer.get_best_fit()

        max_lnlike = best_params["log_likelihood"]
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(best_params["parameters"]):
            print(f"   - {self.modelpar[i]} = {item:.2f}")

        # Get the posterior samples
        samples = analyzer.get_equal_weighted_posterior()

        spec_labels = []
        for item in self.spectrum:
            if f"scaling_{item}" in self.bounds:
                spec_labels.append(f"scaling_{item}")

        ln_prob = samples[:, -1]
        samples = samples[:, :-1]

        # Adding the fixed parameters to the samples

        for key, value in self.fix_param.items():
            self.modelpar.append(key)

            app_param = np.full(samples.shape[0], value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only possible when using a single process
            species_db = database.Database()

            species_db.add_samples(
                sampler="multinest",
                samples=samples,
                ln_prob=ln_prob,
                ln_evidence=(ln_z, ln_z_error),
                mean_accept=None,
                spectrum=("model", self.model),
                tag=tag,
                modelpar=self.modelpar,
                distance=self.distance[0],
                spec_labels=spec_labels,
            )

    @typechecked
    def run_ultranest(
        self,
        tag: str,
        min_num_live_points=400,
        output: str = "ultranest/",
        prior: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Function to run ``UltraNest`` for constructing the posterior
        probability distributions on model parameters and computing
        the marginal likelihood (i.e. "evidence").

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        min_num_live_points : int
            Minimum number of live points. The default of 400 is a reasonable number (see
            https://johannesbuchner.github.io/UltraNest/issues.html). In principle, choosing a very
            low number allows nested sampling to make very few iterations and go to the peak
            quickly. However, the space will be poorly sampled, giving a large region and thus low
            efficiency, and potentially not seeing interesting modes. Therefore, a value above 100
            is typically useful.
        output : str
            Path that is used for the output files from ``UltraNest``.
        prior : dict(str, tuple(float, float)), None
            Dictionary with Gaussian priors for one or multiple parameters. The prior can be set
            for any of the atmosphere or calibration parameters, e.g.
            ``prior={'teff': (1200., 100.)}``. Additionally, a prior can be set for the mass, e.g.
            ``prior={'mass': (13., 3.)}`` for an expected mass of 13 Mjup with an uncertainty of
            3 Mjup. The parameter is not used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print("Running nested sampling with UltraNest...")

        # Print warning if n_live_points from PyMultiNest is used

        # if n_live_points is not None:
        #     warnings.warn('The \'n_live_points\' parameter has been deprecated because UltraNest '
        #                   'is used instead of PyMultiNest. UltraNest can be executed with the '
        #                   '\'run_ultranest\' method of \'FitModel\' and uses the '
        #                   '\'min_num_live_points\' parameter (see documentation for details).',
        #                   DeprecationWarning)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        # Add distance to dictionary with Gaussian priors

        if prior is None:
            prior = {}

        prior['distance'] = self.distance

        @typechecked
        def lnprior_ultranest(cube: np.ndarray) -> np.ndarray:
            """
            Function to transform the unit cube into the parameter cube. It is not clear how to
            pass additional arguments to the function, therefore it is placed here and not merged
            with :func:`~species.analysis.fit_model.FitModel.run_mcmc`.

            Parameters
            ----------
            cube : np.ndarray
                Array with unit parameters.

            Returns
            -------
            np.ndarray
                Array with physical parameters.
            """

            params = cube.copy()

            for item in self.cube_index:
                # Uniform priors for all parameters
                params[self.cube_index[item]] = (
                    self.bounds[item][0]
                    + (self.bounds[item][1] - self.bounds[item][0])
                    * params[self.cube_index[item]]
                )

            return params

        @typechecked
        def lnlike_ultranest(params: np.ndarray) -> np.float64:
            """
            Function for returning the log-likelihood for the sampled parameter cube.

            Parameters
            ----------
            params : np.ndarray
                Array with physical parameters.

            Returns
            -------
            float
                Log-likelihood.
            """

            return self.lnlike_func(params, prior=prior)

        sampler = ultranest.ReactiveNestedSampler(
            self.modelpar,
            lnlike_ultranest,
            transform=lnprior_ultranest,
            resume="subfolder",
            log_dir=output,
        )

        result = sampler.run(
            show_status=True,
            viz_callback=False,
            min_num_live_points=min_num_live_points,
        )

        # Log-evidence

        ln_z = result["logz"]
        ln_z_error = result["logzerr"]
        print(f"Log-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Best-fit parameters

        print("Best-fit parameters (mean +/- std):")

        for i, item in enumerate(self.modelpar):
            mean = np.mean(result["samples"][:, i])
            std = np.std(result["samples"][:, i])

            print(f"   - {item} = {mean:.2e} +/- {std:.2e}")

        # Maximum likelihood sample

        print("Maximum likelihood sample:")

        max_lnlike = result["maximum_likelihood"]["logl"]
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(result["maximum_likelihood"]["point"]):
            print(f"   - {self.modelpar[i]} = {item:.2f}")

        # Create a list with scaling labels

        spec_labels = []
        for item in self.spectrum:
            if f"scaling_{item}" in self.bounds:
                spec_labels.append(f"scaling_{item}")

        # Posterior samples
        samples = result["samples"]

        # Log-likelihood
        ln_prob = result["weighted_samples"]["logl"]

        # Adding the fixed parameters to the samples

        for key, value in self.fix_param.items():
            self.modelpar.append(key)

            app_param = np.full(samples.shape[0], value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only possible when using a single process
            species_db = database.Database()

            species_db.add_samples(
                sampler="ultranest",
                samples=samples,
                ln_prob=ln_prob,
                ln_evidence=(ln_z, ln_z_error),
                mean_accept=None,
                spectrum=("model", self.model),
                tag=tag,
                modelpar=self.modelpar,
                distance=self.distance[0],
                spec_labels=spec_labels,
            )
