"""
Module with functionalities for fitting atmospheric model spectra.
"""

import os
import sys
import warnings

from typing import Optional, Union, List, Tuple, Dict

import dynesty
import numpy as np

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
        "Perhaps because MultiNest was not built "
        "and/or found at the LD_LIBRARY_PATH "
        "(Linux) or DYLD_LIBRARY_PATH (Mac)?"
    )

from schwimmbad import MPIPool
from scipy.interpolate import interp1d
from scipy.stats import norm, truncnorm
from typeguard import typechecked

from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_model import ReadModel
from species.read.read_object import ReadObject
from species.read.read_planck import ReadPlanck
from species.read.read_filter import ReadFilter
from species.util.convert_util import logg_to_mass
from species.util.core_util import print_section
from species.util.dust_util import (
    interp_lognorm,
    interp_powerlaw,
)
from species.util.model_util import (
    binary_to_single,
    check_nearest_spec,
    extract_disk_param,
    apply_obs,
    powerlaw_spectrum,
)

warnings.filterwarnings("always", category=DeprecationWarning)


class FitModel:
    """
    Class for fitting atmospheric model spectra to spectra and/or
    photometric fluxes, and using Bayesian inference (with
    ``MultiNest`` or ``UltraNest``) to estimate the posterior
    distribution and marginalized likelihood (i.e. "evidence").
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
        bounds: Optional[
            Dict[
                str,
                Union[
                    Optional[Tuple[Optional[float], Optional[float]]],
                    Tuple[
                        Optional[Tuple[Optional[float], Optional[float]]],
                        Optional[Tuple[Optional[float], Optional[float]]],
                    ],
                    Tuple[
                        Optional[Tuple[float, float]],
                        Optional[Tuple[float, float]],
                        Optional[Tuple[float, float]],
                    ],
                    List[Tuple[float, float]],
                ],
            ]
        ] = None,
        inc_phot: Union[bool, List[str]] = True,
        inc_spec: Union[bool, List[str]] = True,
        fit_corr: Optional[List[str]] = None,
        apply_weights: Union[bool, Dict[str, Union[float, np.ndarray]]] = False,
        normal_prior: Optional[Dict[str, Tuple[float, float]]] = None,
        ext_model: Optional[str] = None,
        binary_prior: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        object_name : str
            Object name of the companion as stored in the database with
            :func:`~species.data.database.Database.add_object` or
            :func:`~species.data.database.Database.add_companion`.
        model : str
            Name of the atmospheric model (see
            :func:`~species.data.database.Database.available_models`).
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform or
            log-uniform priors. Mandatory parameters are
            automatically added if not already included in
            ``bounds``. Fixing a parameter is possible by
            providing the same value as lower and upper boundary
            of the parameter (e.g. ``bounds={'logg': (4., 4.)``.
            An explanation of the various parameters can be found
            below. See also the ``normal_prior`` parameter for
            using priors with a normal distribution.

            Atmospheric model parameters (e.g. with
            ``model='bt-settl-cifist'``; see docstring of
            :func:`~species.data.database.Database.add_model`
            for the available model grids):

               - Boundaries are provided as tuple of two floats. For example,
                 ``bounds={'teff': (1000, 1500.), 'logg': (3.5, 5.)}``.

               - The grid boundaries (i.e. the maximum range) are
                 adopted as prior if a parameter range is set to
                 ``None`` (instead of a tuple with two values),
                 or if a mandatory parameter is not included
                 in the dictionary of ``bounds``. For example,
                 ``bounds={'teff': (1000., 1500.), 'logg': None}``.
                 The default range for the radius is
                 :math:`0.5-5.0~R_\\mathrm{J}`. With ``bounds=None``,
                 automatic priors will be set for all mandatory
                 parameters.

               - Radial velocity can be included with the ``rad_vel``
                 parameter (km/s). This parameter will only be relevant
                 if the radial velocity shift can be detected given
                 the instrument resolution. When including ``rad_vel``,
                 a single RV will be fitted for all spectra. Or, it is
                 also possible by fitting an RV for individual spectra
                 by for example including the parameter as
                 ``rad_vel_CRIRES`` in case the spectrum name is
                 ``CRIRES``, that is, the name used as tag when adding
                 the spectrum to the database with
                 :func:`~species.data.database.Database.add_object`.

               - Rotational broadening can be fitted by including the
                 ``vsini`` parameter (km/s). This parameter will only
                 be relevant if the rotational broadening is stronger
                 than or comparable to the instrumental broadening,
                 so typically when the data has a high spectral
                 resolution. The resolution is set when adding a
                 spectrum to the database with
                 :func:`~species.data.database.Database.add_object`.
                 The broadening is applied with the function from
                 `Carvalho & Johns-Krull (2023) <https://ui.adsabs.
                 harvard.edu/abs/2023RNAAS...7...91C/abstract>`_.
                 A single broadening parameter, ``vsini``, can be
                 fitted, so it is applied for all spectra. Or, it is
                 also possible to fit the broadening for individual
                 spectra by for example including the parameter as
                 ``vsini_CRIRES`` in case the spectrum name is
                 ``CRIRES``, that is, the name used as tag when
                 adding the spectrum to the database with
                 :func:`~species.data.database.Database.add_object`.

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
                 weight of the two components in case a single radius
                 is fitted, so when simulating horizontal
                 inhomogeneities in the atmosphere. When fitting the
                 combined photometry from two stars, but with known
                 flux ratios in specific filters, it is possible to
                 apply a prior for the known flux ratios. The filters
                 with the known flux ratios can be different from
                 the filters with (combined) photometric fluxes.
                 For example, when the flux ratio is known in filter
                 Paranal/ERIS.H then the parameter to add is
                 ``ratio_Paranal/ERIS.H``. For a uniform prior, the
                 ratio parameter should be added to ``bounds`` and
                 for a normal prior it is added to ``normal_prior``.
                 The flux ratio is defined as the flux of the
                 secondary star divided by the flux of the primary
                 star.

               - Instead of fitting the radius and parallax, it is also
                 possible to fit a scaling parameter directly, either
                 linearly sampled (``flux_scaling``) or logarithmically
                 sampled (``log_flux_scaling``). Additionally, it is
                 also possible to fit a flux offset (``flux_offset``),
                 which adds a constant flux (in W m-2 um-1) to the
                 model spectrum.

            Blackbody disk emission:

               - Blackbody parameters can be fitted to account for
                 thermal emission from one or multiple disk
                 components, in addition to the atmospheric
                 emission. These parameters should therefore be
                 combined with an atmospheric model.

               - Parameter boundaries have to be provided for
                 'disk_teff' (in K) and 'disk_radius' (in Rjup)
                 For example, ``bounds={'teff': (2000., 3000.),
                 'radius': (1., 5.), 'logg': (3.5, 4.5), 'disk_teff':
                 (100., 2000.), 'disk_radius': (1., 100.)}`` for
                 fitting a single blackbody component, in addition
                 to the atmospheric
                 parameters. Or, ``bounds={'teff': (2000., 3000.),
                 'radius': (1., 5.), 'logg': (3.5, 4.5),
                 'disk_teff': [(2000., 500.), (1000., 20.)],
                 'disk_radius': [(1., 100.), (50., 1000.)]}`` for
                 fitting two blackbody components. Any number of
                 blackbody components can be fitted by including
                 additional priors in the lists of ``'disk_teff'``
                 and ``'disk_radius'``.

            Blackbody parameters (only with ``model='planck'``):

               - This implementation fits both the atmospheric emission
                 and possible disk emission with blackbody components.
                 Parameter boundaries have to be provided for 'teff'
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

            Calibration parameters:

                 - For each spectrum, a scaling of the fluxes can be
                   fitted, to account for an inaccuracy of the
                   absolute flux calibration. For example,
                   ``bounds={'scaling_SPHERE': (0.8, 1.2)}`` if the
                   scaling is fitted between 0.8 and 1.2, and the
                   spectrum is stored as ``'SPHERE'`` with
                   :func:`~species.data.database.Database.add_object`.

                 - For each spectrum, a error inflation can be fitted,
                   to account for an underestimation of the flux
                   uncertainties. The errors are inflated relative
                   to the sampled model fluxes. For example,
                   ``bounds={'error_SPHERE': (0.0, 2.0)}`` will
                   scale each sampled model spectrum by a factor between
                   0.0 and 2.0, and add this scaled model spectrum
                   in quadrature to the uncertainties of the
                   observed spectrum. In this case, applied to the
                   spectrum that is stored as ``'SPHERE'`` with
                   :func:`~species.data.database.Database.add_object`.
                   This approach has been adopted from `Piette &
                   Madhusudhan (2020) <https://ui.adsabs.harvard.edu/
                   abs/2020MNRAS.497.5136P/abstract>`_. The error
                   inflation parameter is :math:`x_\\mathrm{tol}` in
                   Eq. 8 from their work.

                 - The errors of the photometric fluxes can also be
                   inflated, to account for an underestimated
                   uncertainty. The errors are inflated relative
                   to the synthetic photometry from the model. The
                   inflation can be fitted either for individual
                   filters, or a single error inflation is applied to
                   all filters from an instrument. For the first case,
                   the keyword in the ``bounds`` dictionary should be
                   provided in the following format:
                   ``'error_Paranal/NACO.Mp': (0., 1.)``. Here, the
                   error of the NACO :math:`M'` flux is inflated up to
                   100 percent of the model flux. For the second case,
                   only the telescope/instrument part of the the filter
                   name should be provided in the ``bounds``
                   dictionary, so in the following format:
                   ``'error_Paranal/NACO': (0., 1.)``. This will
                   increase the errors of all NACO photometry by the
                   same amount relative to their model fluxes.

            ISM extinction parameters:

                 - Fit the extinction with any of the models in the
                   ``dust-extinction`` package (see `list of
                   available models <https://dust-extinction.
                   readthedocs.io/en/latest/dust_extinction/
                   choose_model.html>`_). This is done by setting the
                   the argument of ``ext_model``, for example to
                   ``'G23'`` for using the relation from
                   `Gordon et al. (2023) <https://ui.adsabs.harvard.
                   edu/abs/2023ApJ...950...86G>`_.

                 - The extinction is parametrized by the visual
                   extinction, $A_V$ (``ext_av``), and optionally the
                   reddening, R_V (``ext_rv``). If ``ext_rv`` is not
                   provided, its value is fixed to 3.1 and not fitted.

                 - The prior boundaries of ``ext_av`` and ``ext_rv``
                   should be provided in the ``bounds`` dictionary, for
                   example ``bounds={'ext_av': (0., 5.),
                   'ext_rv': (1., 5.)}``.

            ISM extinction parameters (DEPRECATED):

                 - Fit the extinction with the relation from
                   `Gordon et al. (2023) <https://ui.adsabs.harvard.
                   edu/abs/2023ApJ...950...86G>`_. This relation is
                   defined for wavelengths in the range of 0.09-32
                   µm, so no extinction is applied to the fluxes
                   with wavelengths outside this range.

                 - The extinction is parametrized by the $V$ band
                   extinction, $A_V$ (``ism_ext``), and optionally the
                   reddening, R_V (``ism_red``). If ``ism_red`` is not
                   provided, its value is fixed to 3.1 and not fitted.

                 - The prior boundaries of ``ism_ext`` and ``ism_red``
                   should be provided in the ``bounds`` dictionary, for
                   example ``bounds={'ism_ext': (0., 10.),
                   'ism_red': (1., 10.)}``. It is recommended to set the
                   minimum value of ``ism_red`` to at least 1.0 since the
                   extinction will be negative for the infrared
                   wavelength regime otherwise.

            Log-normal size distribution:

                 - Fitting the extinction of a log-normal size
                   distribution of grains with a crystalline MgSiO3
                   composition, and a homogeneous, spherical structure.

                 - The extinction (``lognorm_ext``) is fitted in the
                   $V$ band ($A_V$ in mag) and the wavelength-dependent
                   extinction cross sections are interpolated from a
                   pre-tabulated grid. The prior boundary of should be
                   provided in the ``bounds`` dictionary, for example
                   ``bounds={'lognorm_ext': (0., 5.)}``.

                 - The size distribution is parameterized by the
                   logarithm (base 10) of the mean geometric
                   radius (``lognorm_radius``) and the
                   geometric standard deviation (``lognorm_sigma``).
                   The grid of cross sections has been calculated for
                   logarithmic radii between -3 and 1, so 0.001 to
                   10 um, and geometric standard deviations between
                   1.1 and 10 (dimensionless). These two parameters
                   are automatically included so only ``lognorm_ext``
                   needs to be added to ``bounds``.

                 - Including the prior boundaries of ``lognorm_radius``
                   and ``lognorm_sigma`` is optional, for example
                   ``bounds={'lognorm_radius': (-1., 0.),
                   'lognorm_sigma': (2., 4.)}``. The full available
                   range is automatically used otherwise.

                 - Uniform priors are used for ``lognorm_radius``,
                   ``lognorm_sigma`` and ``lognorm_ext``.                   .

            Power-law size distribution:

                 - Fitting the extinction of a power-law size
                   distribution of grains, with a crystalline MgSiO3
                   composition, and a homogeneous, spherical structure.

                 - The extinction (``powerlaw_ext``) is fitted in the
                   $V$ band ($A_V$ in mag) and the wavelength-dependent
                   extinction cross sections are interpolated from a
                   pre-tabulated grid. The prior boundary of should be
                   provided in the ``bounds`` dictionary, for example
                   ``bounds={'powerlaw_ext': (0., 5.)}``.

                 - The size distribution is parameterized by the
                   logarithm (base 10) of the maximum radius
                   (``powerlaw_max``) and a power-law exponent
                   (``powerlaw_exp``). The minimum radius is fixed to
                   1 nm. The grid of cross sections has been calculated
                   for maximum radii between 0.01 and 100 um, so for
                   ``powerlaw_max`` between -2 and 2, and power-law
                   exponents between -10 and 10 (dimensionless). These
                   two parameters are automatically included so only
                   ``powerlaw_ext`` needs to be added to ``bounds``.

                 - Including the prior boundaries of ``powerlaw_max``,
                   and ``powerlaw_exp`` is optional, for example
                   ``{'powerlaw_max': (-1., 1.), 'powerlaw_exp':
                   (-2., 3.)}``. The full available range is
                   automatically used otherwise.

                 - Uniform priors are used for ``powerlaw_max``,
                   ``powerlaw_exp``, and ``powerlaw_ext``.

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
            determined from the data are not available with a spectrum
            that was added to the database with
            :func:`~species.data.database.Database.add_object`).
            The parameters that will be fitted are the correlation
            length and the fractional amplitude.
        apply_weights : bool, dict
            Weights to be applied to the log-likelihood components of
            the spectra and photometric fluxes that are provided with
            ``inc_spec`` and ``inc_phot``. This parameter can for
            example be used to increase the weighting of the
            photometric fluxes relative to a spectrum that consists
            of many wavelength points. By setting the argument to
            ``True``, the weighting factors are automatically set,
            based on the FWHM of the filter profiles or the wavelength
            spacing calculated from the spectral resolution. By
            setting the argument to ``False``, there will be no
            weighting applied. Alternatively, a dictionary can
            be used as argument, which includes the names and
            weightings of spectra and/or photometric fluxes.
        normal_prior : dict(str, tuple(float, float)), None
            Dictionary with normal priors for one or multiple
            parameters. The prior can be set for any of the
            atmosphere or calibration parameters, e.g.
            ``normal_prior={'teff': (1200., 100.)}``, for a prior
            distribution with a mean of 1200 K and a standard
            deviation of 100 K. Additionally, a prior can be set for
            the mass, e.g. ``normal_prior={'mass': (13., 3.)}`` for
            an expected mass of 13 Mjup with an uncertainty of 3
            Mjup. A normal prior for the parallax is automatically
            included so does not need to be set with
            ``normal_prior``. The parameter is not used if the
            argument is set to ``None``. See also the ``bounds``
            parameter for including priors with a (log-)uniform
            distribution. Setting a parameter both in ``bounds``
            and ``normal_prior`` will create a prior from a
            truncated normal distribution.
        ext_model : str, None
            Name with the extinction model from the
            ``dust-extinction`` package (see `list of available
            models <https://dust-extinction.readthedocs.io/en/
            latest/dust_extinction/choose_model.html>`_). For example,
            set the argument to ``'G23'`` to use the extinction
            relation from `Gordon et al. (2023) <https://ui.adsabs.
            harvard.edu/abs/2023ApJ...950...86G>`_. To use the
            ``ext_model``, it is mandatory to add the ``ext_av``
            parameter to the ``bounds`` dictionary for setting the
            prior boundaries of :math:`A_V`. The reddening
            can be optionally added to the ``bounds`` with the
            ``ext_rv`` parameter. Otherwise, it is set to the
            default of :math:`R_V = 3.1`.
        binary_prior : bool
            When fitting a binary system (i.e. two sets of atmosphere
            parameters to spectra and/or photometry), a prior is
            applied such that the :math:`T_\\mathrm{eff}` and
            :math:`R` of the primary object will be larger than
            those parameters of the secondary object.

        Returns
        -------
        NoneType
            None
        """

        print_section("Fit model spectra")

        if not inc_phot and not inc_spec:
            raise ValueError("No photometric or spectroscopic data has been selected.")

        if model == "planck" and ("teff" not in bounds or "radius" not in bounds):
            raise ValueError(
                "The 'bounds' dictionary should contain 'teff' and 'radius'."
            )

        if model == "bt-settl":
            warnings.warn(
                "It is recommended to use the CIFIST "
                "grid of the BT-Settl, because it is "
                "a newer version. In that case, set "
                "model='bt-settl-cifist' when using "
                "add_model of Database."
            )

        # Set attributes

        self.object = ReadObject(object_name)
        self.obj_parallax = self.object.get_parallax()
        self.binary = False
        self.param_interp = None
        self.cross_sections = None
        self.ln_z = None
        self.ln_z_error = None
        self.n_planck = 0
        self.n_disk = 0

        self.binary_prior = binary_prior
        self.ext_model = ext_model

        if fit_corr is None:
            self.fit_corr = []
        else:
            self.fit_corr = fit_corr

        self.model = model
        self.bounds = bounds

        if normal_prior is None:
            self.normal_prior = {}
        else:
            self.normal_prior = normal_prior

        # Set default prior range if bounds=None

        readmodel = ReadModel(self.model)

        if self.bounds is None:
            self.bounds = {}

            param_bounds = readmodel.get_bounds()
            for param_key, param_value in param_bounds.items():
                self.bounds[param_key] = param_value

        # Teff range for which the grid will be interpolated

        self.teff_range = readmodel.get_bounds()["teff"]

        if "teff" in self.bounds and self.bounds["teff"] is not None:
            # List with tuples for blackbody components or
            # tuple with two tuples for a binary system
            if isinstance(self.bounds["teff"][0], tuple):
                teff_min = np.inf
                teff_max = -np.inf

                for teff_item in self.bounds["teff"]:
                    if teff_item[0] < teff_min:
                        teff_min = teff_item[0]

                    if teff_item[1] > teff_max:
                        teff_max = teff_item[1]

                self.teff_range = (teff_min, teff_max)

            elif self.bounds["teff"][0] != self.bounds["teff"][1]:
                self.teff_range = self.bounds["teff"]

        # Models that do not require a grid interpolation

        self.non_interp_model = ["planck", "powerlaw"]

        # Check if deprecated ism_ext parameter is used

        if "ism_ext" in self.bounds:
            warnings.warn(
                "The use of 'ism_ext' for fitting the extinction is "
                "deprecated. Please use the 'ext_model' parameter of "
                "'FitModel' in combination with setting 'ext_av' in "
                "the 'bounds' dictionary.",
                DeprecationWarning,
            )

        # Set model parameters and boundaries

        if self.model == "planck":
            if isinstance(bounds["teff"], list) and isinstance(bounds["radius"], list):
                # Fitting multiple blackbody components

                self.n_planck = len(bounds["teff"])

                self.modelpar = []
                self.bounds = {}

                for teff_idx in range(len(bounds["teff"])):
                    self.modelpar.append(f"teff_{teff_idx}")
                    self.modelpar.append(f"radius_{teff_idx}")

                    self.bounds[f"teff_{teff_idx}"] = bounds["teff"][teff_idx]
                    self.bounds[f"radius_{teff_idx}"] = bounds["radius"][teff_idx]

            else:
                # Fitting a single blackbody component

                self.n_planck = 1

                self.modelpar = ["teff", "radius"]
                self.bounds = bounds

            self.modelpar.append("parallax")

        elif self.model == "powerlaw":
            self.modelpar = ["log_powerlaw_a", "log_powerlaw_b", "log_powerlaw_c"]

        else:
            # Fitting self-consistent atmospheric models
            if self.bounds is not None:
                readmodel = ReadModel(self.model)
                bounds_grid = readmodel.get_bounds()

                for key, value in bounds_grid.items():
                    if key not in self.bounds or self.bounds[key] is None:
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

                    else:
                        if self.bounds[key][0] is None:
                            self.bounds[key] = (
                                bounds_grid[key][0],
                                self.bounds[key][1],
                            )

                        if self.bounds[key][1] is None:
                            self.bounds[key] = (
                                self.bounds[key][0],
                                bounds_grid[key][1],
                            )

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

                    if self.binary:
                        for i in range(2):
                            if self.bounds[f"{key}_{i}"][0] < bounds_grid[key][0]:
                                warnings.warn(
                                    f"The lower bound on {key}_{i} "
                                    f"({self.bounds[f'{key}_{i}'][0]}) "
                                    f"is smaller than the lower bound "
                                    f"from the available {self.model} "
                                    f"model grid ({bounds_grid[key][0]}). "
                                    f"The lower bound of the {key}_{i} "
                                    f"prior will be adjusted to "
                                    f"{bounds_grid[key][0]}."
                                )
                                self.bounds[f"{key}_{i}"] = (
                                    bounds_grid[key][0],
                                    self.bounds[f"{key}_{i}"][1],
                                )

                            if self.bounds[f"{key}_{i}"][1] > bounds_grid[key][1]:
                                warnings.warn(
                                    f"The upper bound on {key}_{i} "
                                    f"({self.bounds[f'{key}_{i}'][0]}) "
                                    f"is larger than the upper bound "
                                    f"from the available {self.model} "
                                    f"model grid ({bounds_grid[key][1]}). "
                                    f"The upper bound of the {key}_{i} "
                                    f"prior will be adjusted to "
                                    f"{bounds_grid[key][1]}."
                                )
                                self.bounds[f"{key}_{i}"] = (
                                    self.bounds[f"{key}_{i}"][0],
                                    bounds_grid[key][1],
                                )

            else:
                # Set all parameter boundaries to the grid boundaries
                readmodel = ReadModel(self.model)
                self.bounds = readmodel.get_bounds()

            self.modelpar = readmodel.get_parameters()

            if "flux_scaling" in self.bounds:
                # Fit arbitrary flux scaling
                # Instead of using radius and parallax
                self.modelpar.append("flux_scaling")

            elif "log_flux_scaling" in self.bounds:
                # Fit arbitrary log flux scaling
                # Instead of using radius and parallax
                self.modelpar.append("log_flux_scaling")

            else:
                self.modelpar.append("radius")

                if self.binary:
                    if "parallax" in self.bounds:
                        if isinstance(self.bounds["parallax"][0], tuple):
                            self.modelpar.append("parallax_0")
                            self.modelpar.append("parallax_1")
                            self.bounds["parallax_0"] = self.bounds["parallax"][0]
                            self.bounds["parallax_1"] = self.bounds["parallax"][1]
                            del self.bounds["parallax"]

                    if "parallax_0" in self.normal_prior:
                        self.modelpar.append("parallax_0")

                    if "parallax_1" in self.normal_prior:
                        self.modelpar.append("parallax_1")

                    if "ism_ext" in self.bounds:
                        if isinstance(self.bounds["ism_ext"][0], tuple):
                            self.modelpar.append("ism_ext_0")
                            self.modelpar.append("ism_ext_1")
                            self.bounds["ism_ext_0"] = self.bounds["ism_ext"][0]
                            self.bounds["ism_ext_1"] = self.bounds["ism_ext"][1]
                            del self.bounds["ism_ext"]

                    if "ext_av" in self.bounds:
                        if isinstance(self.bounds["ext_av"][0], tuple):
                            self.modelpar.append("ext_av_0")
                            self.modelpar.append("ext_av_1")
                            self.bounds["ext_av_0"] = self.bounds["ext_av"][0]
                            self.bounds["ext_av_1"] = self.bounds["ext_av"][1]
                            del self.bounds["ext_av"]

                if (
                    "parallax_0" not in self.modelpar
                    or "parallax_1" not in self.modelpar
                ):
                    self.modelpar.append("parallax")

            if "flux_offset" in self.bounds:
                self.modelpar.append("flux_offset")

            # Add radius

            if self.binary:
                if "radius" in self.bounds:
                    if isinstance(self.bounds["radius"][0], tuple):
                        self.bounds["radius_0"] = self.bounds["radius"][0]
                        self.bounds["radius_1"] = self.bounds["radius"][1]
                        del self.bounds["radius"]

                else:
                    self.bounds["radius"] = (0.5, 5.0)

            elif "radius" not in self.bounds and "radius" in self.modelpar:
                self.bounds["radius"] = (0.5, 5.0)

            # Add blackbody disk components

            if "disk_teff" in self.bounds and "disk_radius" in self.bounds:
                teff_range = ReadModel("blackbody").get_bounds()["teff"]

                if isinstance(bounds["disk_teff"], list) and isinstance(
                    bounds["disk_radius"], list
                ):
                    # Fitting multiple blackbody components

                    self.n_disk = len(self.bounds["disk_teff"])

                    # Update temperature and radius parameters

                    for disk_idx in range(len(self.bounds["disk_teff"])):
                        self.modelpar.append(f"disk_teff_{disk_idx}")
                        self.modelpar.append(f"disk_radius_{disk_idx}")

                        self.bounds[f"disk_teff_{disk_idx}"] = self.bounds["disk_teff"][
                            disk_idx
                        ]

                        self.bounds[f"disk_radius_{disk_idx}"] = self.bounds[
                            "disk_radius"
                        ][disk_idx]

                        if self.bounds[f"disk_teff_{disk_idx}"][0] < teff_range[0]:
                            warnings.warn(
                                "The lower bound on the prior of "
                                f"'disk_teff_{disk_idx}' "
                                f"({self.bounds['disk_teff'][0]} K) is below "
                                "the minimum value available in the blackbody "
                                f"grid ({teff_range[0]} K). The lower bound "
                                "is therefore adjusted to the minimum value "
                                "from the grid."
                            )

                            self.bounds[f"disk_teff_{disk_idx}"] = (
                                teff_range[0],
                                self.bounds[f"disk_teff_{disk_idx}"][1],
                            )

                        if self.bounds[f"disk_teff_{disk_idx}"][1] > teff_range[1]:
                            warnings.warn(
                                "The upper bound on the prior of "
                                f"'disk_teff_{disk_idx}' "
                                f"({self.bounds['disk_teff'][1]} K) is above "
                                "the maximum value available in the blackbody "
                                f"grid ({teff_range[1]} K). The upper bound "
                                "is therefore adjusted to the maximum value "
                                "from the grid."
                            )

                            self.bounds[f"disk_teff_{disk_idx}"] = (
                                self.bounds[f"disk_teff_{disk_idx}"][0],
                                teff_range[1],
                            )

                    del self.bounds["disk_teff"]
                    del self.bounds["disk_radius"]

                else:
                    # Fitting a single blackbody component

                    self.n_disk = 1

                    self.modelpar.append("disk_teff")
                    self.modelpar.append("disk_radius")

                    if self.bounds["disk_teff"][0] < teff_range[0]:
                        warnings.warn(
                            "The lower bound on the prior of 'disk_teff' "
                            f"({self.bounds['disk_teff'][0]} K) is below "
                            "the minimum value available in the blackbody "
                            f"grid ({teff_range[0]} K). The lower bound "
                            "is therefore adjusted to the minimum value "
                            "from the grid."
                        )

                        self.bounds["disk_teff"] = (
                            teff_range[0],
                            self.bounds["disk_teff"][1],
                        )

                    if self.bounds["disk_teff"][1] > teff_range[1]:
                        warnings.warn(
                            "The upper bound on the prior of 'disk_teff' "
                            f"({self.bounds['disk_teff'][1]} K) is above "
                            "the maximum value available in the blackbody "
                            f"grid ({teff_range[1]} K). The upper bound "
                            "is therefore adjusted to the maximum value "
                            "from the grid."
                        )

                        self.bounds["disk_teff"] = (
                            self.bounds["disk_teff"][0],
                            teff_range[1],
                        )

            # Update parameters of binary system

            if self.binary:
                # Update list of model parameters

                for key in bounds:
                    if key[:-2] in self.modelpar:
                        par_index = self.modelpar.index(key[:-2])
                        self.modelpar[par_index] = key[:-2] + "_0"
                        self.modelpar.insert(par_index, key[:-2] + "_1")

                if "radius" in self.modelpar:
                    # Fit a weighting for the two spectra in case this
                    # is a single object, so not an actual binary star.
                    # In that case the combination of two spectra is
                    # used to account for atmospheric assymetries
                    self.modelpar.append("spec_weight")

                    if "spec_weight" not in self.bounds:
                        self.bounds["spec_weight"] = (0.0, 1.0)

        # Print info

        print(f"Object name: {object_name}")
        print(f"Model tag: {model}")
        print(f"Binary star: {self.binary}")

        if self.binary:
            print(f"Binary prior: {self.binary}")

        if self.model not in self.non_interp_model:
            print(f"Blackbody components: {self.n_disk}")
            print(f"Teff interpolation range: {self.teff_range}")

        # Select filters and spectra

        if isinstance(inc_phot, bool):
            if inc_phot:
                # Select all filters if inc_phot=True
                inc_phot = self.object.list_filters(verbose=False)

            else:
                inc_phot = []

        if isinstance(inc_spec, bool):
            if inc_spec:
                # Select all spectra if inc_spec=True
                inc_spec = list(self.object.get_spectrum().keys())

            else:
                inc_spec = []

        if inc_spec and self.model == "powerlaw":
            warnings.warn(
                "The 'inc_spec' parameter is not supported when "
                "fitting a power-law spectrum to photometric data. "
                "The argument of 'inc_spec' is therefore ignored."
            )

            inc_spec = []

        # Include photometric data

        self.objphot = []
        self.modelphot = []
        self.synphot = []
        self.filter_name = []
        self.instr_name = []
        self.prior_phot = {}

        if self.model not in self.non_interp_model:
            print()

        for filter_item in inc_phot:
            self.synphot.append(SyntheticPhotometry(filter_item))

            if self.model not in self.non_interp_model:
                print(f"Interpolating {filter_item}...", end="", flush=True)

                # Check wavelength range of filter profile
                read_filt = ReadFilter(filter_item)
                filt_wavel = read_filt.wavelength_range()

                # Check wavelength range of model spectrum
                readmodel = ReadModel(self.model, filter_name=filter_item)
                model_wavel = readmodel.get_wavelengths()

                if filt_wavel[0] < model_wavel[0] or filt_wavel[1] > model_wavel[-1]:
                    raise ValueError(
                        f"The wavelength range of the {filter_item} "
                        f"filter profile, {filt_wavel[0]:.2f}-"
                        f"{filt_wavel[1]:.2f} um, extends beyond the "
                        "wavelength range of the model spectrum, "
                        f"{model_wavel[0]:.2f}-{model_wavel[-1]:.2f} "
                        "um. Please set 'wavel_range=None' when "
                        "adding the model grid with `add_model()' and "
                        "optionally set the 'fit_from' and "
                        "'extend_from' parameters for extending the "
                        "model spectra with a Rayleigh-Jeans slope."
                    )

                # Interpolate the model grid for each filter
                readmodel.interpolate_grid(teff_range=self.teff_range)
                self.modelphot.append(readmodel)

                print(" [DONE]")

            # Add parameter for error inflation

            instr_filt = filter_item.split(".")[0]

            if f"error_{filter_item}" in self.bounds:
                self.modelpar.append(f"error_{filter_item}")

            elif (
                f"error_{instr_filt}" in self.bounds
                and f"error_{instr_filt}" not in self.modelpar
            ):
                self.modelpar.append(f"error_{instr_filt}")

            elif f"log_error_{filter_item}" in self.bounds:
                self.modelpar.append(f"log_error_{filter_item}")

            elif (
                f"log_error_{instr_filt}" in self.bounds
                and f"log_error_{instr_filt}" not in self.modelpar
            ):
                self.modelpar.append(f"log_error_{instr_filt}")

            # Store the flux and uncertainty for each filter

            obj_phot = self.object.get_photometry(filter_item)
            self.objphot.append(np.array([obj_phot[2], obj_phot[3]]))

            self.filter_name.append(filter_item)
            self.instr_name.append(instr_filt)

        # Include spectroscopic data

        if inc_spec:
            # Select all spectra
            self.spectrum = self.object.get_spectrum()

            # Select the spectrum names that are not in inc_spec
            spec_remove = []
            for spec_item in self.spectrum:
                if spec_item not in inc_spec:
                    spec_remove.append(spec_item)

            # Remove the spectra that are not included in inc_spec
            for spec_item in spec_remove:
                del self.spectrum[spec_item]

            self.n_corr_par = 0

            for spec_item in self.fit_corr:
                if spec_item not in self.spectrum:
                    warnings.warn(
                        f"The '{spec_item}' spectrum that is set "
                        "in 'fit_corr' does not exist in the "
                        f"database with '{object_name}'."
                    )

            for spec_item in self.spectrum:
                if spec_item in self.fit_corr:
                    if self.spectrum[spec_item][1] is not None:
                        warnings.warn(
                            f"There is a covariance matrix included "
                            f"with the {spec_item} data of "
                            f"{object_name} so it is not needed to "
                            f"model the covariances with a "
                            f"Gaussian process. Want to test the "
                            f"Gaussian process nonetheless? Please "
                            f"overwrite the data of {object_name} "
                            f"with add_object while setting the "
                            f"path to the covariance data to None."
                        )

                        self.fit_corr.remove(spec_item)

                    else:
                        self.modelpar.append(f"corr_len_{spec_item}")
                        self.modelpar.append(f"corr_amp_{spec_item}")

                        if f"corr_len_{spec_item}" not in self.bounds:
                            # Default prior log10(corr_len/um)
                            self.bounds[f"corr_len_{spec_item}"] = (
                                -3.0,
                                0.0,
                            )

                        if f"corr_amp_{spec_item}" not in self.bounds:
                            self.bounds[f"corr_amp_{spec_item}"] = (0.0, 1.0)

                        self.n_corr_par += 2

            self.modelspec = []

            if self.model not in self.non_interp_model:
                for spec_key, spec_value in self.spectrum.items():
                    print(f"\rInterpolating {spec_key}...", end="", flush=True)

                    # The ReadModel class will make sure that wavelength range of
                    # the selected subgrid fully includes the requested wavel_range

                    wavel_range = (
                        spec_value[0][0, 0],
                        spec_value[0][-1, 0],
                    )

                    readmodel = ReadModel(self.model, wavel_range=wavel_range)
                    model_wavel = readmodel.get_wavelengths()

                    if (
                        spec_value[0][0, 0] < model_wavel[0]
                        or spec_value[0][-1, 0] > model_wavel[-1]
                    ):
                        raise ValueError(
                            f"The wavelength range of the {spec_key} "
                            f"spectrum, {spec_value[0][0, 0]:.2f}-"
                            f"{spec_value[0][-1, 0]:.2f} um, extends "
                            "beyond the wavelength range of the model "
                            f"spectrum, {model_wavel[0]:.2f}-"
                            f"{model_wavel[-1]:.2f} um. Please set "
                            "'wavel_range=None' when adding the model "
                            "grid with `add_model()' and optionally "
                            "set the 'fit_from' and 'extend_from' "
                            "parameters for extending the model "
                            "spectra with a Rayleigh-Jeans slope."
                        )

                    readmodel.interpolate_grid(teff_range=self.teff_range)

                    self.modelspec.append(readmodel)

                    print(" [DONE]")

        else:
            self.spectrum = {}
            self.modelspec = None
            self.n_corr_par = 0

        # Optional parameters, either global or for each instrument/spectrum

        for param_item in ["vsini", "rad_vel"]:
            if param_item in self.bounds:
                if self.binary:
                    if isinstance(self.bounds[param_item][0], tuple):
                        self.modelpar.append(f"{param_item}_0")
                        self.modelpar.append(f"{param_item}_1")
                        self.bounds[f"{param_item}_0"] = self.bounds[param_item][0]
                        self.bounds[f"{param_item}_1"] = self.bounds[param_item][1]
                        del self.bounds[param_item]

                    else:
                        self.modelpar.append(param_item)

                else:
                    self.modelpar.append(param_item)

            else:
                for spec_item in self.spectrum:
                    param_spec = f"{param_item}_{spec_item}"

                    if param_spec in self.bounds:
                        if self.binary:
                            if isinstance(self.bounds[param_spec][0], tuple):
                                self.modelpar.append(f"{param_spec}_0")
                                self.modelpar.append(f"{param_spec}_1")

                                self.bounds[f"{param_spec}_0"] = self.bounds[
                                    param_spec
                                ][0]

                                self.bounds[f"{param_spec}_1"] = self.bounds[
                                    param_spec
                                ][1]

                                del self.bounds[param_spec]

                            else:
                                self.modelpar.append(param_spec)

                        else:
                            self.modelpar.append(param_spec)

        # Get the parameter order if interpolate_grid is used

        if self.model not in self.non_interp_model:
            readmodel = ReadModel(self.model)
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

        # Include blackbody disk

        self.diskphot = []
        self.diskspec = []

        if self.n_disk > 0:
            for filter_item in inc_phot:
                print(f"Interpolating {filter_item}...", end="", flush=True)
                readmodel = ReadModel("blackbody", filter_name=filter_item)
                readmodel.interpolate_grid(teff_range=None)
                self.diskphot.append(readmodel)
                print(" [DONE]")

            for spec_key, spec_value in self.spectrum.items():
                # Use the wavelength range of the selected atmospheric model
                # because the sampled blackbody spectrum will get interpolated
                # to the wavelengths of the model spectrum instead of the data
                spec_idx = list(self.spectrum.keys()).index(spec_key)
                print(f"\rInterpolating {spec_key}...", end="", flush=True)
                # wavel_range = (spec_value[0][0, 0], spec_value[0][-1, 0])
                wavel_range = (
                    self.modelspec[spec_idx].wl_points[0],
                    self.modelspec[spec_idx].wl_points[-1],
                )
                readmodel = ReadModel("blackbody", wavel_range=wavel_range)
                readmodel.interpolate_grid(teff_range=None)
                self.diskspec.append(readmodel)
                print(" [DONE]")

        # Optional flux scaling and error inflation parameters of spectra

        for spec_item in self.spectrum:
            if f"scaling_{spec_item}" in self.bounds:
                self.modelpar.append(f"scaling_{spec_item}")

                if self.bounds[f"scaling_{spec_item}"][0] < 0.0:
                    raise ValueError(
                        f"The lower bound of 'scaling_{spec_item}' "
                        "is smaller than 0. The flux scaling "
                        "should be larger than 0."
                    )

                if self.bounds[f"scaling_{spec_item}"][1] < 0.0:
                    raise ValueError(
                        f"The upper bound of 'scaling_{spec_item}' "
                        "is smaller than 0. The flux scaling "
                        "should be larger than 0."
                    )

            if f"error_{spec_item}" in self.bounds:
                self.modelpar.append(f"error_{spec_item}")

                if self.bounds[f"error_{spec_item}"][0] < 0.0:
                    raise ValueError(
                        f"The lower bound of 'error_{spec_item}' "
                        "is smaller than 0. The error inflation "
                        "should be given relative to the model "
                        "fluxes so the boundaries should be "
                        "larger than 0."
                    )

                if self.bounds[f"error_{spec_item}"][1] < 0.0:
                    raise ValueError(
                        f"The upper bound of 'error_{spec_item}' "
                        "is smaller than 0. The error inflation "
                        "should be given relative to the model "
                        "fluxes so the boundaries should be "
                        "larger than 0."
                    )

            if f"log_error_{spec_item}" in self.bounds:
                self.modelpar.append(f"log_error_{spec_item}")

        # Exctinction parameters

        if "lognorm_ext" in self.bounds:
            self.cross_sections, _, _ = interp_lognorm(verbose=False)

            self.modelpar.append("lognorm_radius")
            self.modelpar.append("lognorm_sigma")
            self.modelpar.append("lognorm_ext")

            if "lognorm_radius" in self.bounds:
                self.bounds["lognorm_radius"] = (
                    self.bounds["lognorm_radius"][0],
                    self.bounds["lognorm_radius"][1],
                )

            else:
                self.bounds["lognorm_radius"] = (
                    np.log10(self.cross_sections.grid[1][0]),
                    np.log10(self.cross_sections.grid[1][-1]),
                )

            if "lognorm_sigma" not in self.bounds:
                self.bounds["lognorm_sigma"] = (
                    self.cross_sections.grid[2][0],
                    self.cross_sections.grid[2][-1],
                )

        elif "powerlaw_ext" in self.bounds:
            self.cross_sections, _, _ = interp_powerlaw(verbose=False)

            self.modelpar.append("powerlaw_max")
            self.modelpar.append("powerlaw_exp")
            self.modelpar.append("powerlaw_ext")

            if "powerlaw_max" in self.bounds:
                self.bounds["powerlaw_max"] = (
                    self.bounds["powerlaw_max"][0],
                    self.bounds["powerlaw_max"][1],
                )

            else:
                self.bounds["powerlaw_max"] = (
                    np.log10(self.cross_sections.grid[1][0]),
                    np.log10(self.cross_sections.grid[1][-1]),
                )

            if "powerlaw_exp" not in self.bounds:
                self.bounds["powerlaw_exp"] = (
                    self.cross_sections.grid[2][0],
                    self.cross_sections.grid[2][-1],
                )

        elif "ism_ext" in self.bounds or "ism_ext" in self.normal_prior:
            self.modelpar.append("ism_ext")

            if "ism_red" in self.bounds or "ism_red" in self.normal_prior:
                self.modelpar.append("ism_red")

        elif "ism_ext" not in self.bounds and "ism_red" in self.bounds:
            warnings.warn(
                "The 'ism_red' parameter is set in the "
                "bounds dictionary but the 'ism_ext' "
                "parameter is not included. The 'ism_red' "
                "parameter is therefore ignored since the "
                "reddening can only be fitted in "
                "combination with the extinction."
            )

        elif ext_model is not None:
            self.ext_model = ext_model

            if self.binary:
                # ext_av_0 and ext_av_1 were already added to self.modelpar
                if "ext_rv_0" in self.bounds or "ext_rv_0" in self.normal_prior:
                    self.modelpar.append("ext_rv_0")

                if "ext_rv_1" in self.bounds or "ext_rv_1" in self.normal_prior:
                    self.modelpar.append("ext_rv_1")

            else:
                if "ext_av" in self.bounds or "ext_av" in self.normal_prior:
                    self.modelpar.append("ext_av")

                    if "ext_rv" in self.bounds or "ext_rv" in self.normal_prior:
                        self.modelpar.append("ext_rv")

                else:
                    self.ext_model = None

                    warnings.warn(
                        "The 'ext_model' is set but the 'ext_av' "
                        "parameter is missing in the 'bounds' "
                        "dictionary so the 'ext_model' parameter "
                        "will be ignored and no extinction will "
                        "be fitted."
                    )

        elif ext_model is None:
            if "ext_av" in self.bounds:
                del self.bounds["ext_av"]

                warnings.warn(
                    "The 'ext_av' parameter is set in the 'bounds' "
                    "dictionary but the 'ext_model' is not set. "
                    "The 'ext_av' parameter will therefore be "
                    "ignored and no extinction will be fitted."
                )

            if "ext_rv" in self.bounds:
                del self.bounds["ext_rv"]

                warnings.warn(
                    "The 'ext_rv' parameter is set in the 'bounds' "
                    "dictionary but the 'ext_model' is not set. "
                    "The 'ext_rv' parameter will therefore be "
                    "ignored and no extinction will be fitted."
                )

        # Veiling parameters

        if "veil_a" in self.bounds:
            self.modelpar.append("veil_a")

        if "veil_b" in self.bounds:
            self.modelpar.append("veil_b")

        if "veil_ref" in self.bounds:
            self.modelpar.append("veil_ref")

        # Interpolate filters of flux ratio priors

        self.flux_ratio = {}

        for param_item in self.bounds:
            if param_item[:6] == "ratio_":
                print(f"Interpolating {param_item[6:]}...", end="", flush=True)
                read_model = ReadModel(self.model, filter_name=param_item[6:])
                read_model.interpolate_grid(teff_range=self.teff_range)
                self.flux_ratio[param_item[6:]] = read_model
                print(" [DONE]")

        for param_item in self.normal_prior:
            if param_item[:6] == "ratio_":
                print(f"Interpolating {param_item[6:]}...", end="", flush=True)
                read_model = ReadModel(self.model, filter_name=param_item[6:])
                read_model.interpolate_grid(teff_range=self.teff_range)
                self.flux_ratio[param_item[6:]] = read_model
                print(" [DONE]")

        for filter_item in self.flux_ratio:
            self.prior_phot[filter_item] = SyntheticPhotometry(filter_item)

        # Fixed parameters

        self.fix_param = {}
        del_param = []

        for key, value in self.bounds.items():
            if value[0] == value[1] and value[0] is not None and value[1] is not None:
                self.fix_param[key] = value[0]
                del_param.append(key)

        if del_param:
            print(f"\nFixing {len(del_param)} parameters:")

            for item in del_param:
                print(f"   - {item} = {self.fix_param[item]}")

                self.modelpar.remove(item)
                del self.bounds[item]

        print(f"\nFitting {len(self.modelpar)} parameters:")

        for item in self.modelpar:
            print(f"   - {item}")

        # Add parallax to dictionary with normal priors

        if (
            "parallax" in self.modelpar
            and "parallax" not in self.fix_param
            and "parallax" not in self.bounds
        ):
            self.normal_prior["parallax"] = (self.obj_parallax[0], self.obj_parallax[1])

        # Printing uniform and normal priors

        print("\nUniform priors (min, max):")

        for param_key, param_value in self.bounds.items():
            print(f"   - {param_key} = {param_value}")

        if len(self.normal_prior) > 0:
            print("\nNormal priors (mean, sigma):")
            for param_key, param_value in self.normal_prior.items():
                if -0.1 < param_value[0] < 0.1:
                    print(
                        f"   - {param_key} = {param_value[0]:.2e} +/- {param_value[1]:.2e}"
                    )
                else:
                    print(
                        f"   - {param_key} = {param_value[0]:.2f} +/- {param_value[1]:.2f}"
                    )

        # Create a dictionary with the cube indices of the parameters

        self.cube_index = {}
        for i, item in enumerate(self.modelpar):
            self.cube_index[item] = i

        # Weighting of the photometric and spectroscopic data

        print("\nWeights for the log-likelihood function:")

        if isinstance(apply_weights, bool):
            self.weights = {}

            if apply_weights:
                for spec_item in inc_spec:
                    spec_size = self.spectrum[spec_item][0].shape[0]

                    if spec_item not in self.weights:
                        # Set weight for spectrum to lambda/R
                        spec_wavel = self.spectrum[spec_item][0][:, 0]
                        spec_res = self.spectrum[spec_item][3]
                        self.weights[spec_item] = spec_wavel / spec_res

                    elif not isinstance(self.weights[spec_item], np.ndarray):
                        self.weights[spec_item] = np.full(
                            spec_size, self.weights[spec_item]
                        )

                    if np.all(self.weights[spec_item] == self.weights[spec_item][0]):
                        print(f"   - {spec_item} = {self.weights[spec_item][0]:.2e}")

                    else:
                        print(
                            f"   - {spec_item} = {np.amin(self.weights[spec_item]):.2e} "
                            f"- {np.amax(self.weights[spec_item]):.2e}"
                        )

                for filter_item in inc_phot:
                    if filter_item not in self.weights:
                        # Set weight for photometry to FWHM of filter
                        read_filt = ReadFilter(filter_item)
                        self.weights[filter_item] = read_filt.filter_fwhm()
                        print(f"   - {filter_item} = {self.weights[filter_item]:.2e}")

            else:
                for spec_item in inc_spec:
                    spec_size = self.spectrum[spec_item][0].shape[0]
                    self.weights[spec_item] = np.full(spec_size, 1.0)
                    print(f"   - {spec_item} = {self.weights[spec_item][0]:.2f}")

                for filter_item in inc_phot:
                    # Set weight to 1 if apply_weights=False
                    self.weights[filter_item] = 1.0
                    print(f"   - {filter_item} = {self.weights[filter_item]:.2f}")

        else:
            self.weights = apply_weights

            for spec_item in inc_spec:
                spec_size = self.spectrum[spec_item][0].shape[0]

                if spec_item not in self.weights:
                    # Set weight for spectrum to lambda/R
                    spec_wavel = self.spectrum[spec_item][0][:, 0]
                    spec_res = self.spectrum[spec_item][3]
                    self.weights[spec_item] = spec_wavel / spec_res

                elif not isinstance(self.weights[spec_item], np.ndarray):
                    self.weights[spec_item] = np.full(
                        spec_size, self.weights[spec_item]
                    )

                if np.all(self.weights[spec_item] == self.weights[spec_item][0]):
                    print(f"   - {spec_item} = {self.weights[spec_item][0]:.2e}")

                else:
                    print(
                        f"   - {spec_item} = {np.amin(self.weights[spec_item]):.2e} "
                        f"- {np.amax(self.weights[spec_item]):.2e}"
                    )

            for filter_item in inc_phot:
                if filter_item not in self.weights:
                    # Set weight for photometry to FWHM of filter
                    read_filt = ReadFilter(filter_item)
                    self.weights[filter_item] = read_filt.filter_fwhm()

                print(f"   - {filter_item} = {self.weights[filter_item]:.2e}")

    @typechecked
    def _prior_transform(
        self, cube, bounds: Dict[str, Tuple[float, float]], cube_index: Dict[str, int]
    ):
        """
        Function to transform the sampled unit cube into a
        cube with sampled model parameters.

        Parameters
        ----------
        cube : LP_c_double, np.ndarray
            Unit cube.
        bounds : dict(str, tuple(float, float))
            Dictionary with the prior boundaries.
        cube_index : dict(str, int)
            Dictionary with the indices for selecting the model
            parameters in the ``cube``.

        Returns
        -------
        np.ndarray
            Cube with the sampled model parameters.
        """

        if isinstance(cube, np.ndarray):
            # Create new array for UltraNest
            param_out = cube.copy()
        else:
            # Convert from ctypes.c_double to np.ndarray
            # Only required with MultiNest
            # n_modelpar = len(self.modelpar)
            # cube = np.ctypeslib.as_array(cube, shape=(n_modelpar,))

            # Use the same object with MultiNest
            param_out = cube

        for param_item in cube_index:
            if param_item in self.normal_prior:
                if param_item in bounds:
                    # Truncated normal prior

                    # The truncation values are given in number of
                    # standard deviations relative to the mean
                    # of the normal distribution

                    a_trunc = (
                        bounds[param_item][0] - self.normal_prior[param_item][0]
                    ) / self.normal_prior[param_item][1]

                    b_trunc = (
                        bounds[param_item][1] - self.normal_prior[param_item][0]
                    ) / self.normal_prior[param_item][1]

                    param_out[cube_index[param_item]] = truncnorm.ppf(
                        param_out[cube_index[param_item]],
                        a_trunc,
                        b_trunc,
                        loc=self.normal_prior[param_item][0],
                        scale=self.normal_prior[param_item][1],
                    )

                else:
                    # Normal prior

                    param_out[cube_index[param_item]] = norm.ppf(
                        param_out[cube_index[param_item]],
                        loc=self.normal_prior[param_item][0],
                        scale=self.normal_prior[param_item][1],
                    )

            else:
                # Uniform prior

                param_out[cube_index[param_item]] = (
                    bounds[param_item][0]
                    + (bounds[param_item][1] - bounds[param_item][0])
                    * param_out[cube_index[param_item]]
                )

        return param_out

    @typechecked
    def _lnlike_func(
        self,
        params,
    ) -> Union[np.float64, float]:
        """
        Function for calculating the log-likelihood for the sampled
        parameter cube. The model spectrum will be compared with
        the photometric and spectral fluxes.

        Parameters
        ----------
        params : LP_c_double, np.ndarray
            Cube with sampled model parameters.

        Returns
        -------
        float
            Log-likelihood.
        """

        # Create dictionary with the sampled parameters

        all_param = {}

        for param_item in self.cube_index.keys():
            all_param[param_item] = params[self.cube_index[param_item]]

        # Add fixed parameters to parameter dictionary

        for param_item in self.fix_param.keys():
            all_param[param_item] = self.fix_param[param_item]

        # Add the flux offset to the parameter dictionary

        # if "flux_offset" in self.cube_index:
        #     all_param["flux_offset"] = params[self.cube_index["flux_offset"]]
        # else:
        #     all_param["flux_offset"] = 0.0

        # Check if the blackbody temperatures/radii are decreasing/increasing

        if self.model == "planck" and self.n_planck > 1:
            for planck_idx in range(self.n_planck - 1):
                if all_param[f"teff_{planck_idx+1}"] > all_param[f"teff_{planck_idx}"]:
                    return -np.inf

                if (
                    all_param[f"radius_{planck_idx}"]
                    > all_param[f"radius_{planck_idx+1}"]
                ):
                    return -np.inf

        # Enforce decreasing Teff and increasing R

        if self.n_disk == 1:
            if all_param["disk_teff"] > all_param["disk_teff"]:
                return -np.inf

            if all_param["disk_radius"] < all_param["disk_radius"]:
                return -np.inf

        elif self.n_disk > 1:
            for disk_idx in range(self.n_disk):
                if disk_idx == 0:
                    if all_param["disk_teff_0"] > all_param["teff"]:
                        return -np.inf

                    if all_param["disk_radius_0"] < all_param["radius"]:
                        return -np.inf

                else:
                    if (
                        all_param[f"disk_teff_{disk_idx}"]
                        > all_param[f"disk_teff_{disk_idx-1}"]
                    ):
                        return -np.inf

                    if (
                        all_param[f"disk_radius_{disk_idx}"]
                        < all_param[f"disk_radius_{disk_idx-1}"]
                    ):
                        return -np.inf

        # Check if the primary star has a higher Teff and larger R

        if self.binary and self.binary_prior:
            if all_param["teff_1"] > all_param["teff_0"]:
                return -np.inf

            if all_param["radius_1"] > all_param["radius_0"]:
                return -np.inf

        # Sort the parameters in the correct order for
        # spectrum_interp because it creates a list in
        # the order of the keys in param_dict

        if self.param_interp is not None:
            param_dict = {}
            for param_item in self.param_interp:
                param_dict[param_item] = all_param[param_item]

        else:
            param_dict = None

        # Initialize the log-likelihood sum

        ln_like = 0.0

        # Add normal priors to the log-likelihood function

        for prior_key, prior_value in self.normal_prior.items():
            if prior_key == "mass":
                if "logg" in all_param and "radius" in all_param:
                    mass = logg_to_mass(all_param["logg"], all_param["radius"])

                    ln_like += (mass - prior_value[0]) ** 2 / prior_value[1] ** 2

                else:
                    if "logg" not in all_param:
                        warnings.warn(
                            "The 'logg' parameter is not used "
                            f"by the '{self.model}' model so "
                            "the mass prior cannot be applied."
                        )

                    elif "radius" not in all_param:
                        warnings.warn(
                            "The 'radius' parameter is not fitted "
                            "so the mass prior cannot be applied."
                        )

            elif self.binary and prior_key in ["mass_0", "mass_1"]:
                bin_idx = prior_key[-1]

                if f"logg_{bin_idx}" in all_param and f"radius_{bin_idx}" in all_param:
                    mass = logg_to_mass(
                        all_param[f"logg_{bin_idx}"], all_param[f"radius_{bin_idx}"]
                    )

                    ln_like += (mass - prior_value[0]) ** 2 / prior_value[1] ** 2

                else:
                    if f"logg_{bin_idx}" not in all_param:
                        warnings.warn(
                            f"The 'logg_{bin_idx}' parameter is not "
                            "fitted so the mass prior can't be applied."
                        )

                    elif f"radius_{bin_idx}" not in all_param:
                        warnings.warn(
                            f"The 'radius_{bin_idx}' parameter is not "
                            "fitted so the mass prior can't be applied."
                        )

            elif self.binary and prior_key == "mass_ratio":
                if (
                    "logg_0" in all_param
                    and "radius_0" in all_param
                    and "logg_1" in all_param
                    and "radius_1" in all_param
                ):
                    mass_0 = logg_to_mass(all_param["logg_0"], all_param["radius_0"])
                    mass_1 = logg_to_mass(all_param["logg_1"], all_param["radius_1"])

                    ln_like += (mass_1 / mass_0 - prior_value[0]) ** 2 / prior_value[
                        1
                    ] ** 2

                else:
                    for param_item in ["logg_0", "logg_1", "radius_0", "radius_1"]:
                        if param_item not in all_param:
                            warnings.warn(
                                f"The '{param_item}' parameter is "
                                "not fitted so the mass_ratio prior "
                                "can't be applied."
                            )

            elif prior_key[:6] == "ratio_":
                filter_name = prior_key[6:]

                # Star 0

                param_0 = binary_to_single(param_dict, 0)

                model_flux_0 = self.flux_ratio[filter_name].spectrum_interp(
                    list(param_0.values())
                )[0]

                # Apply extinction and flux scaling

                all_param_0 = binary_to_single(all_param, 0)

                model_flux_0 = apply_obs(
                    model_flux=model_flux_0,
                    model_wavel=self.flux_ratio[filter_name].wl_points,
                    model_param=all_param_0,
                    cross_sections=self.cross_sections,
                    ext_model=self.ext_model,
                )

                phot_flux_0 = self.prior_phot[filter_name].spectrum_to_flux(
                    self.flux_ratio[filter_name].wl_points, model_flux_0
                )[0]

                # Star 1

                param_1 = binary_to_single(param_dict, 1)

                model_flux_1 = self.flux_ratio[filter_name].spectrum_interp(
                    list(param_1.values())
                )[0]

                # Apply extinction and flux scaling

                all_param_1 = binary_to_single(all_param, 1)

                model_flux_1 = apply_obs(
                    model_flux=model_flux_1,
                    model_wavel=self.flux_ratio[filter_name].wl_points,
                    model_param=all_param_1,
                    cross_sections=self.cross_sections,
                    ext_model=self.ext_model,
                )

                phot_flux_1 = self.prior_phot[filter_name].spectrum_to_flux(
                    self.flux_ratio[filter_name].wl_points, model_flux_1
                )[0]

                # Uniform prior for the flux ratio

                if f"ratio_{filter_name}" in self.bounds:
                    ratio_prior = self.bounds[f"ratio_{filter_name}"]

                    if phot_flux_1 / phot_flux_0 < ratio_prior[0]:
                        return -np.inf

                    if phot_flux_1 / phot_flux_0 > ratio_prior[1]:
                        return -np.inf

                # Normal prior for the flux ratio

                if f"ratio_{filter_name}" in self.normal_prior:
                    ratio_prior = self.normal_prior[f"ratio_{filter_name}"]

                    ln_like += (
                        phot_flux_1 / phot_flux_0 - ratio_prior[0]
                    ) ** 2 / ratio_prior[1] ** 2

            else:
                ln_like += (
                    params[self.cube_index[prior_key]] - prior_value[0]
                ) ** 2 / prior_value[1] ** 2

        # Compare photometry with model

        for phot_idx, phot_item in enumerate(self.objphot):
            filter_name = self.synphot[phot_idx].filter_name

            if self.model == "planck":
                readplanck = ReadPlanck(filter_name=filter_name)

                phot_flux = readplanck.get_flux(
                    all_param, synphot=self.synphot[phot_idx]
                )[0]

            elif self.model == "powerlaw":
                powerl_box = powerlaw_spectrum(
                    self.synphot[phot_idx].wavel_range, param_dict
                )

                phot_flux = self.synphot[phot_idx].spectrum_to_flux(
                    powerl_box.wavelength, powerl_box.flux
                )[0]

            else:
                if self.binary:
                    # Star 0

                    param_0 = binary_to_single(param_dict, 0)

                    model_flux_0 = self.modelphot[phot_idx].spectrum_interp(
                        list(param_0.values())
                    )[0]

                    # Apply extinction and flux scaling

                    all_param_0 = binary_to_single(all_param, 0)

                    model_flux_0 = apply_obs(
                        model_flux=model_flux_0,
                        model_wavel=self.modelphot[phot_idx].wl_points,
                        model_param=all_param_0,
                        cross_sections=self.cross_sections,
                        ext_model=self.ext_model,
                    )

                    # Star 1

                    param_1 = binary_to_single(param_dict, 1)

                    model_flux_1 = self.modelphot[phot_idx].spectrum_interp(
                        list(param_1.values())
                    )[0]

                    # Apply extinction and flux scaling

                    all_param_1 = binary_to_single(all_param, 1)

                    model_flux_1 = apply_obs(
                        model_flux=model_flux_1,
                        model_wavel=self.modelphot[phot_idx].wl_points,
                        model_param=all_param_1,
                        cross_sections=self.cross_sections,
                        ext_model=self.ext_model,
                    )

                    # Weighted flux of two spectra for atmospheric asymmetries
                    # Or adding the two objects in case of a binary system

                    if "spec_weight" in self.cube_index:
                        model_flux = (
                            params[self.cube_index["spec_weight"]] * model_flux_0
                            + (1.0 - params[self.cube_index["spec_weight"]])
                            * model_flux_1
                        )

                    else:
                        model_flux = model_flux_0 + model_flux_1

                else:
                    # Interpolate model spectrum

                    model_flux = self.modelphot[phot_idx].spectrum_interp(
                        list(param_dict.values())
                    )[0]

                    # Apply extinction and flux scaling

                    model_flux = apply_obs(
                        model_flux=model_flux,
                        model_wavel=self.modelphot[phot_idx].wl_points,
                        model_param=all_param,
                        cross_sections=self.cross_sections,
                        ext_model=self.ext_model,
                    )

                # Calculate synthetic photometry

                phot_flux = self.synphot[phot_idx].spectrum_to_flux(
                    self.modelphot[phot_idx].wl_points, model_flux
                )[0]

                # Add blackbody disk components

                if self.n_disk == 1:
                    disk_flux = self.diskphot[phot_idx].spectrum_interp(
                        [all_param["disk_teff"]]
                    )[0]

                    # Apply extinction and flux scaling

                    disk_param = extract_disk_param(all_param)

                    disk_flux = apply_obs(
                        model_flux=disk_flux,
                        model_wavel=self.diskphot[phot_idx].wl_points,
                        model_param=disk_param,
                        cross_sections=self.cross_sections,
                        ext_model=self.ext_model,
                    )

                    # Calculate synthetic photometry

                    phot_flux += self.synphot[phot_idx].spectrum_to_flux(
                        self.diskphot[phot_idx].wl_points, disk_flux
                    )[0]

                elif self.n_disk > 1:
                    for disk_idx in range(self.n_disk):
                        disk_flux = self.diskphot[phot_idx].spectrum_interp(
                            [all_param[f"disk_teff_{disk_idx}"]]
                        )[0]

                        # Apply extinction and flux scaling

                        disk_param = extract_disk_param(all_param, disk_index=disk_idx)

                        disk_flux = apply_obs(
                            model_flux=disk_flux,
                            model_wavel=self.diskphot[phot_idx].wl_points,
                            model_param=disk_param,
                            cross_sections=self.cross_sections,
                            ext_model=self.ext_model,
                        )

                        # Calculate synthetic photometry

                        phot_flux += self.synphot[phot_idx].spectrum_to_flux(
                            self.diskphot[phot_idx].wl_points, disk_flux
                        )[0]

                # Optional flux offset

                if "flux_offset" in self.cube_index:
                    phot_flux += params[self.cube_index["flux_offset"]]

            # Calculate log-likelihood for photometry

            if phot_item.ndim == 1:
                phot_var = phot_item[1] ** 2

                # Get the telescope/instrument name
                instr_check = filter_name.split(".")[0]

                if f"error_{filter_name}" in all_param:
                    # Inflate photometric uncertainty for filter
                    # Scale relative to the model flux
                    phot_var += all_param[f"error_{filter_name}"] ** 2 * phot_flux**2

                elif f"error_{instr_check}" in all_param:
                    # Inflate photometric uncertainty for instrument
                    # Scale relative to the model flux
                    phot_var += all_param[f"error_{instr_check}"] ** 2 * phot_flux**2

                elif f"log_error_{filter_name}" in all_param:
                    # Inflate photometric uncertainty for filter
                    # Scale relative to the model flux
                    phot_var += (
                        10.0 ** all_param[f"log_error_{filter_name}"]
                    ) ** 2 * phot_flux**2

                elif f"log_error_{instr_check}" in all_param:
                    # Inflate photometric uncertainty for instrument
                    # Scale relative to the model flux
                    phot_var += (
                        10.0 ** all_param[f"log_error_{instr_check}"]
                    ) ** 2 * phot_flux**2

                ln_like += (
                    self.weights[filter_name]
                    * (phot_item[0] - phot_flux) ** 2
                    / phot_var
                )

                # Only required when fitting an error inflation
                ln_like += self.weights[filter_name] * np.log(2.0 * np.pi * phot_var)

            else:
                for phot_idx in range(phot_item.shape[1]):
                    phot_var = phot_item[1, phot_idx] ** 2

                    # Get the telescope/instrument name
                    instr_check = filter_name.split(".")[0]

                    if f"error_{filter_name}" in all_param:
                        # Inflate photometric uncertainty for filter
                        # Scale relative to the model flux
                        phot_var += (
                            all_param[f"error_{filter_name}"] ** 2 * phot_flux**2
                        )

                    elif f"error_{instr_check}" in all_param:
                        # Inflate photometric uncertainty for instrument
                        # Scale relative to the model flux
                        phot_var += (
                            all_param[f"error_{instr_check}"] ** 2 * phot_flux**2
                        )

                    elif f"log_error_{filter_name}" in all_param:
                        # Inflate photometric uncertainty for filter
                        # Scale relative to the model flux
                        phot_var += (
                            10.0 ** all_param[f"log_error_{filter_name}"]
                        ) ** 2 * phot_flux**2

                    elif f"log_error_{instr_check}" in all_param:
                        # Inflate photometric uncertainty for instrument
                        # Scale relative to the model flux
                        phot_var += (
                            10.0 ** all_param[f"log_error_{instr_check}"]
                        ) ** 2 * phot_flux**2

                    ln_like += (
                        self.weights[filter_name]
                        * (phot_item[0, phot_idx] - phot_flux) ** 2
                        / phot_var
                    )

                    # Only required when fitting an error inflation
                    ln_like += self.weights[filter_name] * np.log(
                        2.0 * np.pi * phot_var
                    )

        # Compare spectra with model

        for spec_idx, spec_item in enumerate(self.spectrum.keys()):

            if self.model == "planck":
                # Calculate a blackbody spectrum from the sampled parameters

                readplanck = ReadPlanck(
                    wavel_range=(
                        0.9 * self.spectrum[spec_item][0][0, 0],
                        1.1 * self.spectrum[spec_item][0][-1, 0],
                    )
                )

                model_box = readplanck.get_spectrum(all_param, spec_res=1000.0)

                # Resample the spectrum to the observed wavelengths

                flux_interp = interp1d(model_box.wavelength, model_box.flux)
                model_flux = flux_interp(self.spectrum[spec_item][0][:, 0])

            else:
                # Interpolate the model spectrum from the grid

                if self.binary:
                    # Star 0

                    param_0 = binary_to_single(param_dict, 0)

                    model_flux_0 = self.modelspec[spec_idx].spectrum_interp(
                        list(param_0.values())
                    )[0]

                    # Set rotational broadening

                    if "vsini" in self.modelpar:
                        rot_broad_0 = params[self.cube_index["vsini"]]

                    elif "vsini_0" in self.modelpar:
                        rot_broad_0 = params[self.cube_index["vsini_0"]]

                    elif f"vsini_{spec_item}_0" in self.modelpar:
                        rot_broad_0 = params[self.cube_index[f"vsini_{spec_item}_0"]]

                    else:
                        rot_broad_0 = None

                    # Set radial velocity

                    if "rad_vel" in self.modelpar:
                        rad_vel_0 = params[self.cube_index["rad_vel"]]

                    elif "rad_vel_0" in self.modelpar:
                        rad_vel_0 = params[self.cube_index["rad_vel_0"]]

                    elif f"rad_vel_{spec_item}_0" in self.modelpar:
                        rad_vel_0 = params[self.cube_index[f"rad_vel_{spec_item}_0"]]

                    else:
                        rad_vel_0 = None

                    # Apply extinction, flux scaling, vsin(i), and RV

                    all_param_0 = binary_to_single(all_param, 0)

                    model_flux_0 = apply_obs(
                        model_flux=model_flux_0,
                        model_wavel=self.modelspec[spec_idx].wl_points,
                        model_param=all_param_0,
                        cross_sections=self.cross_sections,
                        rot_broad=rot_broad_0,
                        rad_vel=rad_vel_0,
                        ext_model=self.ext_model,
                    )

                    # Star 1

                    param_1 = binary_to_single(param_dict, 1)

                    model_flux_1 = self.modelspec[spec_idx].spectrum_interp(
                        list(param_1.values())
                    )[0]

                    # Set rotational broadening

                    if "vsini" in self.modelpar:
                        rot_broad_1 = params[self.cube_index["vsini"]]

                    elif "vsini_1" in self.modelpar:
                        rot_broad_1 = params[self.cube_index["vsini_1"]]

                    elif f"vsini_{spec_item}_1" in self.modelpar:
                        rot_broad_1 = params[self.cube_index[f"vsini_{spec_item}_1"]]

                    else:
                        rot_broad_1 = None

                    # Set radial velocity

                    if "rad_vel" in self.modelpar:
                        rad_vel_1 = params[self.cube_index["rad_vel"]]

                    elif "rad_vel_1" in self.modelpar:
                        rad_vel_1 = params[self.cube_index["rad_vel_1"]]

                    elif f"rad_vel_{spec_item}_1" in self.modelpar:
                        rad_vel_1 = params[self.cube_index[f"rad_vel_{spec_item}_1"]]

                    else:
                        rad_vel_1 = None

                    # Apply extinction, flux scaling, vsin(i), and RV

                    all_param_1 = binary_to_single(all_param, 1)

                    model_flux_1 = apply_obs(
                        model_flux=model_flux_1,
                        model_wavel=self.modelspec[spec_idx].wl_points,
                        model_param=all_param_1,
                        cross_sections=self.cross_sections,
                        rot_broad=rot_broad_1,
                        rad_vel=rad_vel_1,
                        ext_model=self.ext_model,
                    )

                    # Weighted flux of two spectra for atmospheric asymmetries
                    # Or simply the same in case of an actual binary system

                    if "spec_weight" in self.cube_index:
                        model_flux = (
                            params[self.cube_index["spec_weight"]] * model_flux_0
                            + (1.0 - params[self.cube_index["spec_weight"]])
                            * model_flux_1
                        )
                    else:
                        model_flux = model_flux_0 + model_flux_1

                else:
                    # Set rotational broadening

                    if "vsini" in self.modelpar:
                        rot_broad = params[self.cube_index["vsini"]]

                    elif f"vsini_{spec_item}" in self.modelpar:
                        rot_broad = params[self.cube_index[f"vsini_{spec_item}"]]

                    else:
                        rot_broad = None

                    # Set radial velocity

                    if "rad_vel" in self.modelpar:
                        rad_vel = params[self.cube_index["rad_vel"]]

                    elif f"rad_vel_{spec_item}" in self.modelpar:
                        rad_vel = params[self.cube_index[f"rad_vel_{spec_item}"]]

                    else:
                        rad_vel = None

                    # Interpolate model spectrum

                    model_flux = self.modelspec[spec_idx].spectrum_interp(
                        list(param_dict.values())
                    )[0]

                    # Apply extinction, flux scaling, vsin(i), and RV

                    model_flux = apply_obs(
                        model_param=all_param,
                        model_flux=model_flux,
                        model_wavel=self.modelspec[spec_idx].wl_points,
                        cross_sections=self.cross_sections,
                        rot_broad=rot_broad,
                        rad_vel=rad_vel,
                        ext_model=self.ext_model,
                    )

                # Add blackbody disk components

                if self.n_disk > 0:
                    disk_wavel = self.diskspec[spec_idx].wl_points

                    if self.n_disk == 1:
                        flux_tmp = self.diskspec[spec_idx].spectrum_interp(
                            [all_param["disk_teff"]]
                        )[0]

                        # Apply extinction and flux scaling

                        disk_param = extract_disk_param(all_param)

                        disk_flux = apply_obs(
                            model_param=disk_param,
                            model_flux=flux_tmp,
                            model_wavel=disk_wavel,
                            cross_sections=self.cross_sections,
                            ext_model=self.ext_model,
                        )

                    elif self.n_disk > 1:
                        disk_flux = 0.0

                        for disk_idx in range(self.n_disk):
                            flux_tmp = self.diskspec[spec_idx].spectrum_interp(
                                [all_param[f"disk_teff_{disk_idx}"]]
                            )[0]

                            # Apply extinction and flux scaling

                            disk_param = extract_disk_param(
                                all_param, disk_index=disk_idx
                            )

                            disk_flux += apply_obs(
                                model_param=disk_param,
                                model_flux=flux_tmp,
                                model_wavel=disk_wavel,
                                cross_sections=self.cross_sections,
                                ext_model=self.ext_model,
                            )

                    # Interpolate blackbody spectrum to the atmosphere spectrum

                    flux_interp = interp1d(disk_wavel, disk_flux)
                    model_flux += flux_interp(self.modelspec[spec_idx].wl_points)

            # Extinction, flux scaling, vsin(i), and RV have already been applied

            model_flux = apply_obs(
                model_flux=model_flux,
                model_wavel=self.modelspec[spec_idx].wl_points,
                data_wavel=self.spectrum[spec_item][0][:, 0],
                spec_res=self.spectrum[spec_item][3],
            )

            # Optional flux offset

            if "flux_offset" in self.cube_index:
                model_flux += params[self.cube_index["flux_offset"]]

            # Apply veiling by adding continuuum source

            # if (
            #     "veil_a" in all_param
            #     and "veil_b" in all_param
            #     and "veil_ref" in all_param
            # ):
            #     if spec_item == "MUSE":
            #         lambda_ref = 0.5  # (um)
            #
            #         veil_flux = all_param["veil_ref"] + all_param["veil_b"] * (
            #             self.spectrum[spec_item][0][:, 0] - lambda_ref
            #         )
            #
            #         model_flux = all_param["veil_a"] * model_flux + veil_flux

            # Optional flux scaling to account for calibration inaccuracy

            if f"scaling_{spec_item}" in all_param:
                spec_scaling = all_param[f"scaling_{spec_item}"]
            else:
                spec_scaling = 1.0

            data_flux = spec_scaling * self.spectrum[spec_item][0][:, 1]
            data_var = spec_scaling**2 * self.spectrum[spec_item][0][:, 2] ** 2

            # Optional variance inflation (see Piette & Madhusudhan 2020)

            if f"error_{spec_item}" in all_param:
                data_var += (all_param[f"error_{spec_item}"] * model_flux) ** 2

            elif f"log_error_{spec_item}" in all_param:
                data_var += (
                    10.0 ** all_param[f"log_error_{spec_item}"] * model_flux
                ) ** 2

            # Select the inverted covariance matrix

            if self.spectrum[spec_item][2] is not None:
                if (
                    f"error_{spec_item}" in all_param
                    or f"log_error_{spec_item}" in all_param
                ):
                    # Ratio of the inflated and original uncertainties
                    sigma_ratio = np.sqrt(data_var) / (
                        spec_scaling * self.spectrum[spec_item][0][:, 2]
                    )
                    sigma_j, sigma_i = np.meshgrid(sigma_ratio, sigma_ratio)

                    # Calculate the inverted matrix of the inflated covariances
                    data_cov_inv = np.linalg.inv(
                        spec_scaling**2
                        * sigma_i
                        * sigma_j
                        * self.spectrum[spec_item][1]
                    )

                else:
                    if spec_scaling == 1.0:
                        # Use the inverted covariance matrix directly
                        data_cov_inv = self.spectrum[spec_item][2]
                    else:
                        # Apply flux scaling to covariance matrix
                        # and then invert the covariance matrix
                        data_cov_inv = np.linalg.inv(
                            spec_scaling**2 * self.spectrum[spec_item][1]
                        )

            # Calculate the log-likelihood

            if self.spectrum[spec_item][2] is not None:
                # Use the inverted covariance matrix

                ln_like += (
                    self.weights[spec_item]
                    * (data_flux - model_flux)
                    @ data_cov_inv
                    @ (data_flux - model_flux)
                )

                ln_like += np.nansum(
                    self.weights[spec_item] * np.log(2.0 * np.pi * data_var)
                )

            else:
                if spec_item in self.fit_corr:
                    # Covariance model (Wang et al. 2020)

                    wavel = self.spectrum[spec_item][0][:, 0]  # (um)
                    wavel_j, wavel_i = np.meshgrid(wavel, wavel)

                    error = np.sqrt(data_var)  # (W m-2 um-1)
                    error_j, error_i = np.meshgrid(error, error)

                    corr_len = 10.0 ** all_param[f"corr_len_{spec_item}"]  # (um)
                    corr_amp = all_param[f"corr_amp_{spec_item}"]

                    cov_matrix = (
                        corr_amp**2
                        * error_i
                        * error_j
                        * np.exp(-((wavel_i - wavel_j) ** 2) / (2.0 * corr_len**2))
                        + (1.0 - corr_amp**2) * np.eye(wavel.shape[0]) * error_i**2
                    )

                    ln_like += (
                        self.weights[spec_item]
                        * (data_flux - model_flux)
                        @ np.linalg.inv(cov_matrix)
                        @ (data_flux - model_flux)
                    )

                    ln_like += np.nansum(
                        self.weights[spec_item] * np.log(2.0 * np.pi * data_var)
                    )

                else:
                    # Calculate the log-likelihood without a covariance matrix

                    ln_like += np.nansum(
                        self.weights[spec_item]
                        * (data_flux - model_flux) ** 2
                        / data_var
                    )

                    ln_like += np.nansum(
                        self.weights[spec_item] * np.log(2.0 * np.pi * data_var)
                    )

        return -0.5 * ln_like

    @typechecked
    def _create_attr_dict(self):
        """
        Internal function for creating a dictionary with attributes
        that will be stored in the database with the results when
        calling :func:`~species.data.database.Database.add_samples`.

        Returns
        -------
        NoneType
            None
        """

        attr_dict = {
            "model_type": "atmosphere",
            "model_name": self.model,
            "ln_evidence": (self.ln_z, self.ln_z_error),
            "parallax": self.obj_parallax[0],
            "binary": self.binary,
        }

        if self.ext_model is not None:
            attr_dict["ext_model"] = self.ext_model

        return attr_dict

    @typechecked
    def run_multinest(
        self,
        tag: str,
        n_live_points: int = 1000,
        resume: bool = False,
        output: str = "multinest/",
        kwargs_multinest: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the
        ``MultiNest`` sampler. While ``PyMultiNest`` can be
        installed with ``pip`` from the PyPI repository,
        ``MultiNest`` has to be built manually. See the
        `PyMultiNest documentation <http://johannesbuchner.
        github.io/PyMultiNest/install.html>`_. The library
        path of ``MultiNest`` should be set to the
        environmental variable ``LD_LIBRARY_PATH`` on a
        Linux machine and ``DYLD_LIBRARY_PATH`` on a Mac.
        Alternatively, the variable can be set before
        importing the ``species`` package, for example:

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
        resume : bool
            Resume the posterior sampling from a previous run.
        output : str
            Path that is used for the output files from MultiNest.
        kwargs_multinest : dict, None
            Dictionary with keyword arguments that can be used to
            adjust the parameters of the `run() function
            <https://github.com/JohannesBuchner/PyMultiNest/blob/
            master/pymultinest/run.py>`_ of the ``PyMultiNest``
            sampler. See also the `documentation of MultiNest
            <https://github.com/JohannesBuchner/MultiNest>`_.

        Returns
        -------
        NoneType
            None
        """

        print_section("Nested sampling with MultiNest")

        print(f"Database tag: {tag}")
        print(f"Number of live points: {n_live_points}")
        print(f"Resume previous fit: {resume}")
        print(f"Output folder: {output}")
        print()

        # Set attributes

        if "prior" in kwargs:
            warnings.warn(
                "The 'prior' parameter has been deprecated "
                "and will be removed in a future release. "
                "Please use the 'normal_prior' of FitModel "
                "instead.",
                DeprecationWarning,
            )

            if kwargs["prior"] is not None:
                self.normal_prior = kwargs["prior"]

        # Create empty dictionary if needed

        if kwargs_multinest is None:
            kwargs_multinest = {}

        # Check kwargs_multinest keywords

        if "n_live_points" in kwargs_multinest:
            warnings.warn(
                "Please specify the number of live points "
                "as argument of 'n_live_points' instead "
                "of using 'kwargs_multinest'."
            )

            del kwargs_multinest["n_live_points"]

        if "resume" in kwargs_multinest:
            warnings.warn(
                "Please use the 'resume' parameter "
                "instead of setting the value with "
                "'kwargs_multinest'."
            )

            del kwargs_multinest["resume"]

        if "outputfiles_basename" in kwargs_multinest:
            warnings.warn(
                "Please use the 'output' parameter "
                "instead of setting the value of "
                "'outputfiles_basename' in "
                "'kwargs_multinest'."
            )

            del kwargs_multinest["outputfiles_basename"]

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        @typechecked
        def _lnprior_multinest(cube, n_dim: int, n_param: int) -> None:
            """
            Function to transform the unit cube into the parameter
            cube. It is not clear how to pass additional arguments
            to the function, therefore it is placed here.

            Parameters
            ----------
            cube : LP_c_double
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

            self._prior_transform(cube, self.bounds, self.cube_index)

        @typechecked
        def _lnlike_multinest(
            params, n_dim: int, n_param: int
        ) -> Union[float, np.float64]:
            """
            Function for return the log-likelihood for the
            sampled parameter cube.

            Parameters
            ----------
            params : LP_c_double
                Cube with sampled model parameters.
            n_dim : int
                Number of dimensions. This parameter is mandatory
                but not used by the function.
            n_param : int
                Number of parameters. This parameter is mandatory
                but not used by the function.

            Returns
            -------
            float
                Log-likelihood.
            """

            return self._lnlike_func(params)

        pymultinest.run(
            _lnlike_multinest,
            _lnprior_multinest,
            len(self.modelpar),
            outputfiles_basename=output,
            resume=resume,
            n_live_points=n_live_points,
            **kwargs_multinest,
        )

        # Create the Analyzer object
        analyzer = pymultinest.analyse.Analyzer(
            len(self.modelpar),
            outputfiles_basename=output,
            verbose=False,
        )

        # Get a dictionary with the ln(Z) and its errors, the
        # individual modes and their parameters quantiles of
        # the parameter posteriors
        sampling_stats = analyzer.get_stats()

        # Nested sampling log-evidence
        self.ln_z = sampling_stats["nested sampling global log-evidence"]
        self.ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"\nlog-evidence = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Nested importance sampling log-evidence
        imp_ln_z = sampling_stats["nested importance sampling global log-evidence"]
        imp_ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            "log-evidence (importance sampling) = "
            f"{imp_ln_z:.2f} +/- {imp_ln_z_error:.2f}"
        )

        # Get the sample with the maximum likelihood

        best_params = analyzer.get_best_fit()
        max_lnlike = best_params["log_likelihood"]

        print("\nSample with the maximum likelihood:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        param_check = {}
        for param_idx, param_item in enumerate(best_params["parameters"]):
            param_check[self.modelpar[param_idx]] = param_item
            if -0.1 < param_item < 0.1:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2e}")
            else:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2f}")

        # Check nearest grid points

        for param_key, param_value in self.fix_param.items():
            param_check[param_key] = param_value

        if self.binary:
            param_0 = binary_to_single(param_check, 0)
            check_nearest_spec(self.model, param_0)

            param_1 = binary_to_single(param_check, 1)
            check_nearest_spec(self.model, param_1)

        else:
            check_nearest_spec(self.model, param_check)

        # Get the posterior samples

        post_samples = analyzer.get_equal_weighted_posterior()

        # Create list with the spectrum labels that are used
        # for fitting an additional scaling parameter

        spec_labels = []
        for spec_item in self.spectrum:
            if f"scaling_{spec_item}" in self.bounds:
                spec_labels.append(f"scaling_{spec_item}")

        # Samples and ln(L)

        ln_prob = post_samples[:, -1]
        samples = post_samples[:, :-1]

        # Adding the fixed parameters to the samples

        for param_key, param_value in self.fix_param.items():
            self.modelpar.append(param_key)

            app_param = np.full(samples.shape[0], param_value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                tag=tag,
                sampler="multinest",
                samples=samples,
                ln_prob=ln_prob,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                fixed_param=self.fix_param,
                spec_labels=spec_labels,
                attr_dict=self._create_attr_dict(),
            )

    @typechecked
    def run_ultranest(
        self,
        tag: str,
        min_num_live_points: int = 400,
        resume: Union[bool, str] = False,
        output: str = "ultranest/",
        kwargs_ultranest: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Function to run ``UltraNest`` for estimating the posterior
        distributions of model parameters and computing the
        marginalized likelihood (i.e. "evidence").

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        min_num_live_points : int
            Minimum number of live points. The default of 400 is a
            `reasonable number <https://johannesbuchner.github.io/
            UltraNest/issues.html>`_. In principle, choosing a very
            low number allows nested sampling to make very few
            iterations and go to the peak quickly. However, the space
            will be poorly sampled, giving a large region and thus low
            efficiency, and potentially not seeing interesting modes.
            Therefore, a value above 100 is typically useful.
        resume : bool, str
            Resume the posterior sampling from a previous run. The
            ``UltraNest`` documentation provides a description of the
            `possible arguments <https://johannesbuchner.github.io/
            UltraNest/ultranest.html#ultranest.integrator.
            ReactiveNestedSampler>`_ (``True``, ``'resume'``,
            ``'resume-similar'``, ``'overwrite'``, ``'subfolder'``).
            Setting the argument to ``False`` is identical to
            ``'subfolder'``.
        kwargs_ultranest : dict, None
            Dictionary with keyword arguments that can be used to
            adjust the parameters of the `run() method
            <https://johannesbuchner.github.io/UltraNest/ultranest
            .html#ultranest.integrator.ReactiveNestedSampler.run>`_
            of the ``UltraNest`` sampler.
        output : str
            Path that is used for the output files from ``UltraNest``.

        Returns
        -------
        NoneType
            None
        """

        print_section("Nested sampling with UltraNest")

        print(f"Database tag: {tag}")
        print(f"Minimum number of live points: {min_num_live_points}")
        print(f"Resume previous fit: {resume}")
        print(f"Output folder: {output}")
        print()

        # Check if resume is set to a non-UltraNest value

        if isinstance(resume, bool) and not resume:
            resume = "subfolder"

        # Set attributes

        if "prior" in kwargs:
            warnings.warn(
                "The 'prior' parameter has been deprecated "
                "and will be removed in a future release. "
                "Please use the 'normal_prior' of FitModel "
                "instead.",
                DeprecationWarning,
            )

            if kwargs["prior"] is not None:
                self.normal_prior = kwargs["prior"]

        # Create empty dictionary if needed

        if kwargs_ultranest is None:
            kwargs_ultranest = {}

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        @typechecked
        def _lnprior_ultranest(cube: np.ndarray) -> np.ndarray:
            """
            Function to transform the unit cube into the parameter
            cube. It is not clear how to pass additional arguments
            to the function, therefore it is placed here.

            Parameters
            ----------
            cube : np.ndarray
                Unit cube.

            Returns
            -------
            np.ndarray
                Cube with the sampled model parameters.
            """

            return self._prior_transform(cube, self.bounds, self.cube_index)

        @typechecked
        def _lnlike_ultranest(params: np.ndarray) -> Union[float, np.float64]:
            """
            Function for returning the log-likelihood for the
            sampled parameter cube.

            Parameters
            ----------
            params : np.ndarray
                Array with sampled model parameters.

            Returns
            -------
            float
                Log-likelihood.
            """

            ln_like = self._lnlike_func(params)

            if not np.isfinite(ln_like):
                # UltraNest cannot handle np.inf in the likelihood
                ln_like = -1e100

            return ln_like

        sampler = ultranest.ReactiveNestedSampler(
            self.modelpar,
            _lnlike_ultranest,
            transform=_lnprior_ultranest,
            resume=resume,
            log_dir=output,
        )

        if "show_status" not in kwargs_ultranest:
            kwargs_ultranest["show_status"] = True

        if "viz_callback" not in kwargs_ultranest:
            kwargs_ultranest["viz_callback"] = False

        if "min_num_live_points" in kwargs_ultranest:
            warnings.warn(
                "Please specify the minimum number of live "
                "points as argument of 'min_num_live_points' "
                "instead of using 'kwargs_ultranest'."
            )

            del kwargs_ultranest["min_num_live_points"]

        result = sampler.run(
            min_num_live_points=min_num_live_points, **kwargs_ultranest
        )

        # Log-evidence

        self.ln_z = result["logz"]
        self.ln_z_error = result["logzerr"]
        print(f"\nlog-evidence = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Best-fit parameters

        print("\nBest-fit parameters (mean +/- sigma):")

        for param_idx, param_item in enumerate(self.modelpar):
            mean = np.mean(result["samples"][:, param_idx])
            sigma = np.std(result["samples"][:, param_idx])

            if -0.1 < mean < 0.1:
                print(f"   - {param_item} = {mean:.2e} +/- {sigma:.2e}")
            else:
                print(f"   - {param_item} = {mean:.2f} +/- {sigma:.2f}")

        # Get the sample with the maximum likelihood

        max_lnlike = result["maximum_likelihood"]["logl"]

        print("\nSample with the maximum likelihood:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        param_check = {}
        for param_idx, param_item in enumerate(result["maximum_likelihood"]["point"]):
            param_check[self.modelpar[param_idx]] = param_item
            if -0.1 < param_item < 0.1:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2e}")
            else:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2f}")

        # Check nearest grid points

        for param_key, param_value in self.fix_param.items():
            param_check[param_key] = param_value

        if self.binary:
            param_0 = binary_to_single(param_check, 0)
            check_nearest_spec(self.model, param_0)

            param_1 = binary_to_single(param_check, 1)
            check_nearest_spec(self.model, param_1)

        else:
            check_nearest_spec(self.model, param_check)

        # Create a list with scaling labels

        spec_labels = []
        for spec_item in self.spectrum:
            if f"scaling_{spec_item}" in self.bounds:
                spec_labels.append(f"scaling_{spec_item}")

        # Samples and ln(L)

        samples = result["samples"]
        ln_prob = result["weighted_samples"]["logl"]

        # Adding the fixed parameters to the samples

        for param_key, param_value in self.fix_param.items():
            self.modelpar.append(param_key)

            app_param = np.full(samples.shape[0], param_value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                tag=tag,
                sampler="ultranest",
                samples=samples,
                ln_prob=ln_prob,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                fixed_param=self.fix_param,
                spec_labels=spec_labels,
                attr_dict=self._create_attr_dict(),
            )

    @typechecked
    def run_dynesty(
        self,
        tag: str,
        n_live_points: int = 1000,
        resume: bool = False,
        output: str = "dynesty/",
        evidence_tolerance: float = 0.5,
        dynamic: bool = False,
        sample_method: str = "auto",
        bound: str = "multi",
        n_pool: Optional[int] = None,
        mpi_pool: bool = False,
    ) -> None:
        """
        Function for running the fit with a grid of model spectra.
        The parameter estimation and computation of the marginalized
        likelihood (i.e. model evidence), are done with ``Dynesty``.

        When using MPI, it is also required to install ``mpi4py`` (e.g.
        ``pip install mpi4py``), otherwise an error may occur when the
        ``output_folder`` is created by multiple processes.

        Parameters
        ----------
        tag : str
            Database tag where the samples will be stored.
        n_live_points : int
            Number of live points used by the nested sampling
            with ``Dynesty``.
        resume : bool
            Resume the posterior sampling from a previous run.
        output : str
            Path that is used for the output files from ``Dynesty``.
        evidence_tolerance : float
            The dlogZ value used to terminate a nested sampling run,
            or the initial dlogZ value passed to a dynamic nested
            sampling run.
        dynamic : bool
            Whether to use static or dynamic nested sampling (see
            `Dynesty documentation <https://dynesty.readthedocs.io/
            en/stable/dynamic.html>`_).
        sample_method : str
            The sampling method that should be used ('auto', 'unif',
            'rwalk', 'slice', 'rslice' (see `sampling documentation
            <https://dynesty.readthedocs.io/en/stable/
            quickstart.html#nested-sampling-with-dynesty>`_).
        bound : str
            Method used to approximately bound the prior using the
            current set of live points ('none', 'single', 'multi',
            'balls', 'cubes'). `Conditions the sampling methods
            <https://dynesty.readthedocs.io/en/stable/
            quickstart.html#nested-sampling-with-dynesty>`_ used
            to propose new live points
        n_pool : int
            The number of processes for the local multiprocessing. The
            parameter is not used when the argument is set to ``None``.
        mpi_pool : bool
            Distribute the workers to an ``MPIPool`` on a cluster,
            using ``schwimmbad``.

        Returns
        -------
        NoneType
            None
        """

        print_section("Nested sampling with Dynesty")

        print(f"Database tag: {tag}")
        print(f"Number of live points: {n_live_points}")
        print(f"Resume previous fit: {resume}")

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            print(f"Creating output folder: {output}")
            os.mkdir(output)

        else:
            print(f"Output folder: {output}")

        print()

        out_basename = os.path.join(output, "")

        if not mpi_pool:
            if n_pool is not None:
                with dynesty.pool.Pool(
                    n_pool,
                    self._lnlike_func,
                    self._prior_transform,
                    ptform_args=[self.bounds, self.cube_index],
                ) as pool:
                    print(f"Initialized a Dynesty.pool with {n_pool} workers")

                    if dynamic:
                        if resume:
                            dsampler = dynesty.DynamicNestedSampler.restore(
                                fname=out_basename + "dynesty.save",
                                pool=pool,
                            )

                            print(
                                "Resumed a Dynesty run from "
                                f"{out_basename}dynesty.save"
                            )

                        else:
                            dsampler = dynesty.DynamicNestedSampler(
                                loglikelihood=pool.loglike,
                                prior_transform=pool.prior_transform,
                                ndim=len(self.modelpar),
                                pool=pool,
                                sample=sample_method,
                                bound=bound,
                            )

                        dsampler.run_nested(
                            dlogz_init=evidence_tolerance,
                            nlive_init=n_live_points,
                            checkpoint_file=out_basename + "dynesty.save",
                            resume=resume,
                        )

                    else:
                        if resume:
                            dsampler = dynesty.NestedSampler.restore(
                                fname=out_basename + "dynesty.save",
                                pool=pool,
                            )

                            print(
                                "Resumed a Dynesty run from "
                                f"{out_basename}dynesty.save"
                            )

                        else:
                            dsampler = dynesty.NestedSampler(
                                loglikelihood=pool.loglike,
                                prior_transform=pool.prior_transform,
                                ndim=len(self.modelpar),
                                pool=pool,
                                nlive=n_live_points,
                                sample=sample_method,
                                bound=bound,
                            )

                        dsampler.run_nested(
                            dlogz=evidence_tolerance,
                            checkpoint_file=out_basename + "dynesty.save",
                            resume=resume,
                        )
            else:
                if dynamic:
                    if resume:
                        dsampler = dynesty.DynamicNestedSampler.restore(
                            fname=out_basename + "dynesty.save"
                        )

                        print(f"Resumed a Dynesty run from {out_basename}dynesty.save")

                    else:
                        dsampler = dynesty.DynamicNestedSampler(
                            loglikelihood=self._lnlike_func,
                            prior_transform=self._prior_transform,
                            ndim=len(self.modelpar),
                            ptform_args=[self.bounds, self.cube_index],
                            sample=sample_method,
                            bound=bound,
                        )

                    dsampler.run_nested(
                        dlogz_init=evidence_tolerance,
                        nlive_init=n_live_points,
                        checkpoint_file=out_basename + "dynesty.save",
                        resume=resume,
                    )

                else:
                    if resume:
                        dsampler = dynesty.NestedSampler.restore(
                            fname=out_basename + "dynesty.save"
                        )

                        print(f"Resumed a Dynesty run from {out_basename}dynesty.save")

                    else:
                        dsampler = dynesty.NestedSampler(
                            loglikelihood=self._lnlike_func,
                            prior_transform=self._prior_transform,
                            ndim=len(self.modelpar),
                            ptform_args=[self.bounds, self.cube_index],
                            sample=sample_method,
                            bound=bound,
                        )

                    dsampler.run_nested(
                        dlogz=evidence_tolerance,
                        checkpoint_file=out_basename + "dynesty.save",
                        resume=resume,
                    )

        else:
            pool = MPIPool()

            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            print("Created an MPIPool object.")

            if dynamic:
                if resume:
                    dsampler = dynesty.DynamicNestedSampler.restore(
                        fname=out_basename + "dynesty.save",
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.DynamicNestedSampler(
                        loglikelihood=self._lnlike_func,
                        prior_transform=self._prior_transform,
                        ndim=len(self.modelpar),
                        ptform_args=[self.bounds, self.cube_index],
                        pool=pool,
                        sample=sample_method,
                        bound=bound,
                    )

                dsampler.run_nested(
                    dlogz_init=evidence_tolerance,
                    nlive_init=n_live_points,
                    checkpoint_file=out_basename + "dynesty.save",
                    resume=resume,
                )

            else:
                if resume:
                    dsampler = dynesty.NestedSampler.restore(
                        fname=out_basename + "dynesty.save",
                        pool=pool,
                    )

                else:
                    dsampler = dynesty.NestedSampler(
                        loglikelihood=self._lnlike_func,
                        prior_transform=self._prior_transform,
                        ndim=len(self.modelpar),
                        ptform_args=[self.bounds, self.cube_index],
                        pool=pool,
                        nlive=n_live_points,
                        sample=sample_method,
                        bound=bound,
                    )

                dsampler.run_nested(
                    dlogz=evidence_tolerance,
                    checkpoint_file=out_basename + "dynesty.save",
                    resume=resume,
                )

        # Samples and ln(L)

        results = dsampler.results
        samples = results.samples_equal()
        ln_prob = results.logl

        print(f"\nSamples shape: {samples.shape}")
        print(f"Number of iterations: {results.niter}")

        out_file = out_basename + "post_equal_weights.dat"
        print(f"Storing samples: {out_file}")
        np.savetxt(out_file, np.c_[samples, ln_prob])

        # Nested sampling log-evidence

        self.ln_z = results.logz[-1]
        self.ln_z_error = results.logzerr[-1]
        print(f"\nlog-evidence = {self.ln_z:.2f} +/- {self.ln_z_error:.2f}")

        # Get the sample with the maximum likelihood

        max_idx = np.argmax(ln_prob)
        max_lnlike = ln_prob[max_idx]
        best_params = samples[max_idx]

        print("\nSample with the maximum likelihood:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        param_check = {}
        for param_idx, param_item in enumerate(best_params):
            param_check[self.modelpar[param_idx]] = param_item
            if -0.1 < param_item < 0.1:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2e}")
            else:
                print(f"   - {self.modelpar[param_idx]} = {param_item:.2f}")

        # Check nearest grid points

        for param_key, param_value in self.fix_param.items():
            param_check[param_key] = param_value

        if self.binary:
            param_0 = binary_to_single(param_check, 0)
            check_nearest_spec(self.model, param_0)

            param_1 = binary_to_single(param_check, 1)
            check_nearest_spec(self.model, param_1)

        else:
            check_nearest_spec(self.model, param_check)

        # Create a list with scaling labels

        spec_labels = []
        for spec_item in self.spectrum:
            if f"scaling_{spec_item}" in self.bounds:
                spec_labels.append(f"scaling_{spec_item}")

        # Adding the fixed parameters to the samples

        for param_key, param_value in self.fix_param.items():
            self.modelpar.append(param_key)

            app_param = np.full(samples.shape[0], param_value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ImportError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                tag=tag,
                sampler="dynesty",
                samples=samples,
                ln_prob=ln_prob,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                fixed_param=self.fix_param,
                spec_labels=spec_labels,
                attr_dict=self._create_attr_dict(),
            )
