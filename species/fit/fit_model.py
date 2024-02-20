"""
Module with functionalities for fitting atmospheric model spectra.
"""

import os
import sys
import warnings

from typing import Optional, Union, List, Tuple, Dict

import dynesty
import numpy as np
import spectres

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

from PyAstronomy.pyasl import fastRotBroad
from schwimmbad import MPIPool
from scipy import interpolate, stats
from typeguard import typechecked

from species.core import constants
from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_model import ReadModel
from species.read.read_object import ReadObject
from species.read.read_planck import ReadPlanck
from species.read.read_filter import ReadFilter
from species.util.convert_util import logg_to_mass
from species.util.core_util import print_section
from species.util.dust_util import (
    convert_to_av,
    interp_lognorm,
    interp_powerlaw,
    ism_extinction,
)
from species.util.model_util import binary_to_single, powerlaw_spectrum


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
                    Tuple[float, float],
                    Tuple[Optional[Tuple[float, float]]],
                    Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
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
        ext_filter: Optional[str] = None,
        normal_prior: Optional[Dict[str, Tuple[float, float]]] = None,
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
                 if the radial velocity shift can be spectrally
                 resolved given the instrument resolution.

               - Rotational broadening can be fitted by including the
                 ``vsini`` parameter (km/s). This parameter will only
                 be relevant if the rotational broadening is stronger
                 than or comparable to the instrumental broadening,
                 so typically when the data has a high spectral
                 resolution. The resolution is set when adding a
                 spectrum to the database with
                 :func:`~species.data.database.Database.add_object`.
                 Note that the broadening is applied with the
                 `fastRotBroad <https://pyastronomy.readthedocs.io/
                 en/latest/pyaslDoc/aslDoc/rotBroad.html#PyAstronomy.
                 pyasl.fastRotBroad>`_ function from ``PyAstronomy``.
                 The rotational broadening is only accurate if the
                 wavelength range of the data is somewhat narrow.
                 For example, when fitting a medium- or
                 high-resolution spectrum across multiple bands
                 (e.g. $JHK$ bands) then it is best to split up the
                 data into the separate bands when adding them with
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
                 weight of the two components.

               - Instead of fitting the radius and parallax, it is also
                 possible to fit a scaling parameter directly, either
                 linearly sampled (``flux_scaling``) or logarithmically
                 sampled (``log_flux_scaling``). Additionally, it is
                 also possible to fit a flux offset (``flux_offset``),
                 which adds a constant flux (in W m-2 um-1) to the
                 model spectrum.

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

            Calibration parameters:

                 - For each spectrum/instrument, three optional
                   parameters can be fitted to account for biases in
                   the calibration: a scaling of the flux, a
                   relative inflation of the uncertainties, and a
                   radial velocity (RV) shift. The last parameter can
                   account for an actual RV shift by the source or
                   an inaccuracy in the wavelength solution.

                 - For example, ``bounds={'SPHERE': ((0.8, 1.2),
                   (0., 1.), (-50., 50.))}`` if the scaling is
                   fitted between 0.8 and 1.2, the error is
                   inflated (relative to the sampled model fluxes)
                   with a value between 0 and 1, and the RV is
                   fitted between -50 and 50 km/s.

                 - The dictionary key should be the same as the
                   database tag of the spectrum. For example,
                   ``{'SPHERE': ((0.8, 1.2), (0., 1.), (-50., 50.))}``
                   if the spectrum is stored as ``'SPHERE'`` with
                   :func:`~species.data.database.Database.add_object`.

                 - Each of the three calibration parameters can be set to
                   ``None`` in which case the parameter is not used. For
                   example,
                   ``bounds={'SPHERE': ((0.8, 1.2), None, None)}``.

                 - The errors of the photometric fluxes can be inflated
                   to account for underestimated error bars. The error
                   inflation is relative to the actual flux and is
                   either fitted separately for a filter, or a single
                   error inflation is applied to all filters from an
                   instrument. For the first case, the keyword in the
                   ``bounds`` dictionary should be provided in the
                   following format:
                   ``'Paranal/NACO.Mp_error': (0., 1.)``. Here, the
                   error of the NACO :math:`M'` flux is inflated up to
                   100 percent of the actual flux. For the second case,
                   only the telescope/instrument part of the the filter
                   name should be provided in the ``bounds``
                   dictionary, so in the following format:
                   ``'Paranal/NACO_error': (0., 1.)``. This will
                   increase the errors of all NACO filters by the same
                   (relative) amount.

                 - No calibration parameters are fitted if the
                   spectrum name is not included in ``bounds``.

            ISM extinction parameters:

                 - There are three approaches for fitting extinction.
                   The first is with the empirical relation from
                   `Cardelli et al. (1989)
                   <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_
                   for ISM extinction.

                 - The extinction is parametrized by the $V$ band
                   extinction, $A_V$ (``ism_ext``), and optionally the
                   reddening, R_V (``ism_red``). If ``ism_red`` is not
                   provided, its value is fixed to 3.1 and not fitted.

                 - The prior boundaries of ``ism_ext`` and ``ism_red``
                   should be provided in the ``bounds`` dictionary, for
                   example ``bounds={'ism_ext': (0., 10.),
                   'ism_red': (0., 20.)}``.

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

            Blackbody disk emission:

                 - Additional blackbody emission can be added to the
                   atmospheric spectrum to account for thermal emission
                   from a disk.

                 - Parameter boundaries have to be provided for
                   'disk_teff' and 'disk_radius'. For example,
                   ``bounds={'teff': (2000., 3000.), 'radius': (1., 5.),
                   'logg': (3.5, 4.5), 'disk_teff': (100., 2000.),
                   'disk_radius': (1., 100.)}``.

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
            weighting applied.
        ext_filter : str, None
            Filter that is associated with the (optional) extinction
            parameter, ``ism_ext``. When the argument of ``ext_filter``
            is set to ``None``, the extinction is defined in the visual
            (i.e. :math:`A_V`). By providing a filter name from the
            `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/
            svo/theory/fps/>`_ as argument then the extinction
            ``ism_ext`` is fitted in that filter instead of the
            $V$ band.
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
            distribution.

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
        self.parallax = self.object.get_parallax()
        self.binary = False
        self.ext_filter = ext_filter

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

        # Set model parameters and boundaries

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

            self.modelpar.append("parallax")

        elif self.model == "powerlaw":
            self.n_planck = 0

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
                                    f"is larger than the lower bound "
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
                readmodel = ReadModel(self.model, None, None)
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
                self.modelpar.append("parallax")

            if "flux_offset" in self.bounds:
                self.modelpar.append("flux_offset")

            # Optional rotational broading

            if "vsini" in self.bounds:
                # Add vsin(i) parameter (km s-1)
                self.modelpar.append("vsini")
                self.bounds["vsini"] = (bounds["vsini"][0], bounds["vsini"][1])

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
                # Select all filters if inc_phot=True
                inc_phot = self.object.list_filters()

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
        self.filter_name = []
        self.instr_name = []

        for item in inc_phot:
            if self.model == "planck":
                # Create SyntheticPhotometry objects when fitting a Planck function
                print(f"Creating synthetic photometry: {item}...", end="", flush=True)
                self.modelphot.append(SyntheticPhotometry(item))
                print(" [DONE]")

            elif self.model == "powerlaw":
                # Or create SyntheticPhotometry objects when fitting a power-law function
                synphot = SyntheticPhotometry(item)

                # Set the wavelength range of the filter as attribute
                synphot.zero_point()

                self.modelphot.append(synphot)

            else:
                # Or interpolate the model grid for each filter
                print(f"Interpolating {item}...", end="", flush=True)
                readmodel = ReadModel(self.model, filter_name=item)
                readmodel.interpolate_grid(wavel_resample=None, spec_res=None)
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
                for spec_key, spec_value in self.spectrum.items():
                    print(f"\rInterpolating {spec_key}...", end="", flush=True)

                    wavel_range = (
                        0.9 * spec_value[0][0, 0],
                        1.1 * spec_value[0][-1, 0],
                    )

                    readmodel = ReadModel(self.model, wavel_range=wavel_range)

                    readmodel.interpolate_grid(
                        wavel_resample=spec_value[0][:, 0],
                        spec_res=spec_value[3],
                    )

                    self.modelspec.append(readmodel)

                    print(" [DONE]")

        else:
            self.spectrum = {}
            self.modelspec = None
            self.n_corr_par = 0

        # Get the parameter order if interpolate_grid is used

        if self.model not in ["planck", "powerlaw"]:
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

        else:
            self.param_interp = None

        # Include blackbody disk

        self.diskphot = []
        self.diskspec = []

        if "disk_teff" in self.bounds and "disk_radius" in self.bounds:
            for item in inc_phot:
                print(f"Interpolating {item}...", end="", flush=True)
                readmodel = ReadModel("blackbody", filter_name=item)
                readmodel.interpolate_grid(wavel_resample=None, spec_res=None)
                self.diskphot.append(readmodel)
                print(" [DONE]")

            for spec_key, spec_value in self.spectrum.items():
                print(f"\rInterpolating {spec_key}...", end="", flush=True)

                wavel_range = (0.9 * spec_value[0][0, 0], 1.1 * spec_value[0][-1, 0])

                readmodel = ReadModel("blackbody", wavel_range=wavel_range)

                readmodel.interpolate_grid(
                    wavel_resample=spec_value[0][:, 0],
                    spec_res=spec_value[3],
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

                if len(bounds[item]) > 1 and bounds[item][1] is not None:
                    # Add the error inflation parameters
                    self.modelpar.append(f"error_{item}")
                    self.bounds[f"error_{item}"] = (
                        bounds[item][1][0],
                        bounds[item][1][1],
                    )

                    if self.bounds[f"error_{item}"][1] < 0.0:
                        warnings.warn(
                            f"The lower bound of 'error_{item}' is "
                            "smaller than 0. The error inflation "
                            "should be given relative to the model "
                            "fluxes so the boundaries should be "
                            "larger than 0."
                        )

                    if self.bounds[f"error_{item}"][1] < 0.0:
                        warnings.warn(
                            f"The upper bound of 'error_{item}' is "
                            "smaller than 0. The error inflation "
                            "should be given relative to the model "
                            "fluxes so the boundaries should be "
                            "larger than 0."
                        )

                if len(bounds[item]) > 2 and bounds[item][2] is not None:
                    # Add radial velocity parameter (km s-1)
                    self.modelpar.append(f"radvel_{item}")
                    self.bounds[f"radvel_{item}"] = (
                        bounds[item][2][0],
                        bounds[item][2][1],
                    )

                if item in self.bounds:
                    del self.bounds[item]

        if (
            "lognorm_radius" in self.bounds
            and "lognorm_sigma" in self.bounds
            and "lognorm_ext" in self.bounds
        ):
            self.cross_sections, _, _ = interp_lognorm(inc_phot, inc_spec)

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
            self.cross_sections, _, _ = interp_powerlaw(inc_phot, inc_spec)

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
            if self.ext_filter is not None:
                self.modelpar.append(f"phot_ext_{self.ext_filter}")
                self.bounds[f"phot_ext_{self.ext_filter}"] = self.bounds["ism_ext"]
                del self.bounds["ism_ext"]

            else:
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

        print(f"\nFitting {len(self.modelpar)} parameters:")

        for item in self.modelpar:
            print(f"   - {item}")

        # Add parallax to dictionary with Gaussian priors

        if "parallax" in self.modelpar:
            self.normal_prior["parallax"] = (self.parallax[0], self.parallax[1])

        # Printing uniform and normal priors

        print("\nUniform priors (min, max):")

        for key, value in self.bounds.items():
            print(f"   - {key} = {value}")

        if len(self.normal_prior) > 0:
            print("\nNormal priors (mean, sigma):")
            for key, value in self.normal_prior.items():
                print(f"   - {key} = {value}")

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
                            f"   - {spec_item} = {np.amin(self.weights[spec_item]):.2e} - {np.amax(self.weights[spec_item]):.2e}"
                        )

                for phot_item in inc_phot:
                    if phot_item not in self.weights:
                        # Set weight for photometry to FWHM of filter
                        read_filt = ReadFilter(phot_item)
                        self.weights[phot_item] = read_filt.filter_fwhm()
                        print(f"   - {phot_item} = {self.weights[phot_item]:.2e}")

            else:
                for spec_item in inc_spec:
                    spec_size = self.spectrum[spec_item][0].shape[0]
                    self.weights[spec_item] = np.full(spec_size, 1.0)
                    print(f"   - {spec_item} = {self.weights[spec_item][0]:.2f}")

                for phot_item in inc_phot:
                    # Set weight to 1 if apply_weights=False
                    self.weights[phot_item] = 1.0
                    print(f"   - {phot_item} = {self.weights[phot_item]:.2f}")

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
                        f"   - {spec_item} = {np.amin(self.weights[spec_item]):.2e} - {np.amax(self.weights[spec_item]):.2e}"
                    )

            for phot_item in inc_phot:
                if phot_item not in self.weights:
                    # Set weight for photometry to FWHM of filter
                    read_filt = ReadFilter(phot_item)
                    self.weights[phot_item] = read_filt.filter_fwhm()

                print(f"   - {phot_item} = {self.weights[phot_item]:.2e}")

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

        for item in cube_index:
            if item in self.normal_prior:
                # Gaussian prior
                param_out[cube_index[item]] = stats.norm.ppf(
                    param_out[cube_index[item]],
                    loc=self.normal_prior[item][0],
                    scale=self.normal_prior[item][1],
                )

            else:
                # Uniform prior
                param_out[cube_index[item]] = (
                    bounds[item][0]
                    + (bounds[item][1] - bounds[item][0]) * param_out[cube_index[item]]
                )

        return param_out

    @typechecked
    def _lnlike_func(
        self,
        params,
    ) -> Union[np.float64, float]:
        """
        Function for calculating the log-likelihood for the sampled
        parameter cube.

        Parameters
        ----------
        params : LP_c_double, np.ndarray
            Cube with sampled model parameters.

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
        rad_vel = {}

        for item in self.bounds:
            # Add the parameters from the params to their dictionaries

            if item[:8] == "scaling_" and item[8:] in self.spectrum:
                spec_scaling[item[8:]] = params[self.cube_index[item]]

            elif item[:6] == "error_" and item[6:] in self.spectrum:
                err_scaling[item[6:]] = params[self.cube_index[item]]

            elif item[:7] == "radvel_":
                rad_vel[item[7:]] = params[self.cube_index[item]]

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

            elif self.ext_filter is not None and item == f"phot_ext_{self.ext_filter}":
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

        # Add the parallax manually because it should
        # not be provided in the bounds dictionary

        if self.model != "powerlaw":
            if "parallax" in self.cube_index:
                parallax = params[self.cube_index["parallax"]]
            else:
                parallax = None

        for item in self.fix_param:
            # Add the fixed parameters to their dictionaries

            if item[:8] == "scaling_" and item[8:] in self.spectrum:
                spec_scaling[item[8:]] = self.fix_param[item]

            elif item[:6] == "error_" and item[6:] in self.spectrum:
                err_scaling[item[6:]] = self.fix_param[item]

            elif item[:7] == "radvel_" and item[7:] in self.spectrum:
                rad_vel[item[7:]] = self.fix_param[item]

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

            elif item[:9] == "phot_ext_":
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
                    1e3 * constants.PARSEC / parallax
                ) ** 2

                flux_scaling_1 = (param_dict["radius_1"] * constants.R_JUP) ** 2 / (
                    1e3 * constants.PARSEC / parallax
                ) ** 2

                # The scaling is applied manually because of the interpolation
                del param_dict["radius_0"]
                del param_dict["radius_1"]

            else:
                if parallax is None:
                    if "flux_scaling" in self.cube_index:
                        flux_scaling = params[self.cube_index["flux_scaling"]]
                    else:
                        flux_scaling = (
                            10.0 ** params[self.cube_index["log_flux_scaling"]]
                        )

                else:
                    try:
                        flux_scaling = (param_dict["radius"] * constants.R_JUP) ** 2 / (
                            1e3 * constants.PARSEC / parallax
                        ) ** 2

                    except ZeroDivisionError:
                        warnings.warn(
                            f"Encountered a ZeroDivisionError when"
                            f"calculating the flux scaling with "
                            f"parallax = {parallax}. This error "
                            f"should not have happened. Setting "
                            f"the scaling to 1e100."
                        )

                        flux_scaling = 1e100

                    # The scaling is applied manually because of the interpolation
                    del param_dict["radius"]

                if "flux_offset" in self.cube_index:
                    flux_offset = params[self.cube_index["flux_offset"]]
                else:
                    flux_offset = 0.0

        for item in self.spectrum:
            if item not in spec_scaling:
                spec_scaling[item] = 1.0

            if item not in err_scaling:
                err_scaling[item] = None

        if self.param_interp is not None:
            # Sort the parameters in the correct order for
            # spectrum_interp because spectrum_interp creates
            # a list in the order of the keys in param_dict
            param_tmp = param_dict.copy()

            param_dict = {}
            for item in self.param_interp:
                param_dict[item] = param_tmp[item]

        ln_like = 0.0

        for key, value in self.normal_prior.items():
            if key == "mass":
                if "logg" in self.modelpar and "radius" in self.modelpar:
                    mass = logg_to_mass(
                        params[self.cube_index["logg"]],
                        params[self.cube_index["radius"]],
                    )

                    ln_like += -0.5 * (mass - value[0]) ** 2 / value[1] ** 2

                else:
                    if "logg" not in self.modelpar:
                        warnings.warn(
                            "The 'logg' parameter is not used "
                            f"by the '{self.model}' model so "
                            "the mass prior can not be applied."
                        )

                    if "radius" not in self.modelpar:
                        warnings.warn(
                            "The 'radius' parameter is not fitted "
                            "so the mass prior can not be applied."
                        )

            else:
                ln_like += (
                    -0.5
                    * (params[self.cube_index[key]] - value[0]) ** 2
                    / value[1] ** 2
                )

        if "lognorm_ext" in dust_param:
            cross_tmp = self.cross_sections["Generic/Bessell.V"](
                (10.0 ** dust_param["lognorm_radius"], dust_param["lognorm_sigma"])
            )

            n_grains = (
                dust_param["lognorm_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
            )

        elif "powerlaw_ext" in dust_param:
            cross_tmp = self.cross_sections["Generic/Bessell.V"](
                (10.0 ** dust_param["powerlaw_max"], dust_param["powerlaw_exp"])
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
                readplanck = ReadPlanck(filter_name=phot_filter)
                phot_flux = readplanck.get_flux(param_dict, synphot=self.modelphot[i])[
                    0
                ]

            elif self.model == "powerlaw":
                powerl_box = powerlaw_spectrum(
                    self.modelphot[i].wavel_range, param_dict
                )

                phot_flux = self.modelphot[i].spectrum_to_flux(
                    powerl_box.wavelength, powerl_box.flux
                )[0]

            else:
                if self.binary:
                    # Star 0

                    param_0 = binary_to_single(param_dict, 0)

                    phot_flux_0 = self.modelphot[i].spectrum_interp(
                        list(param_0.values())
                    )[0][0]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        phot_flux_0 *= flux_scaling
                        phot_flux_0 += flux_offset

                    elif "radius_0" in self.modelpar:
                        phot_flux_0 *= flux_scaling_0
                        phot_flux_0 += flux_offset

                    # Star 1

                    param_1 = binary_to_single(param_dict, 1)

                    phot_flux_1 = self.modelphot[i].spectrum_interp(
                        list(param_1.values())
                    )[0][0]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        phot_flux_1 *= flux_scaling
                        phot_flux_1 += flux_offset

                    elif "radius_1" in self.modelpar:
                        phot_flux_1 *= flux_scaling_1
                        phot_flux_1 += flux_offset

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
                    phot_flux += flux_offset

            if disk_param:
                phot_tmp = self.diskphot[i].spectrum_interp([disk_param["teff"]])[0][0]

                phot_flux += (
                    phot_tmp
                    * (disk_param["radius"] * constants.R_JUP) ** 2
                    / (1e3 * constants.PARSEC / parallax) ** 2
                )

            if "lognorm_ext" in dust_param:
                cross_tmp = self.cross_sections[phot_filter](
                    (10.0 ** dust_param["lognorm_radius"], dust_param["lognorm_sigma"])
                )

                phot_flux *= np.exp(-cross_tmp * n_grains)

            elif "powerlaw_ext" in dust_param:
                cross_tmp = self.cross_sections[phot_filter](
                    (10.0 ** dust_param["powerlaw_max"], dust_param["powerlaw_exp"])
                )

                phot_flux *= np.exp(-cross_tmp * n_grains)

            elif "ism_ext" in dust_param:
                read_filt = ReadFilter(phot_filter)
                phot_wavel = np.array([read_filt.mean_wavelength()])

                ism_reddening = dust_param.get("ism_red", 3.1)

                ext_filt = ism_extinction(
                    dust_param["ism_ext"], ism_reddening, phot_wavel
                )

                phot_flux *= 10.0 ** (-0.4 * ext_filt[0])

            elif self.ext_filter is not None:
                readmodel = ReadModel(self.model, filter_name=phot_filter)

                param_dict[f"phot_ext_{self.ext_filter}"] = dust_param[
                    f"phot_ext_{self.ext_filter}"
                ]
                param_dict["ism_red"] = dust_param.get("ism_red", 3.1)

                phot_flux = readmodel.get_flux(param_dict)[0]
                phot_flux *= flux_scaling
                phot_flux += flux_offset

                del param_dict[f"phot_ext_{self.ext_filter}"]
                del param_dict["ism_red"]

            if obj_item.ndim == 1:
                phot_var = obj_item[1] ** 2

                # Get the telescope/instrument name
                instr_check = phot_filter.split(".")[0]

                if phot_filter in phot_scaling:
                    # Inflate photometric uncertainty for filter

                    # Scale relative to the uncertainty
                    phot_var += phot_scaling[phot_filter] ** 2 * obj_item[1] ** 2

                elif instr_check in phot_scaling:
                    # Inflate photometric uncertainty for instrument

                    # Scale relative to the uncertainty
                    phot_var += phot_scaling[instr_check] ** 2 * obj_item[1] ** 2

                ln_like += -0.5 * weight * (obj_item[0] - phot_flux) ** 2 / phot_var

                # Only required when fitting an error inflation
                ln_like += -0.5 * np.log(2.0 * np.pi * phot_var)

            else:
                for j in range(obj_item.shape[1]):
                    phot_var = obj_item[1, j] ** 2

                    # Get the telescope/instrument name
                    instr_check = phot_filter.split(".")[0]

                    if phot_filter in phot_scaling:
                        # Inflate photometric uncertainty for filter

                        # Scale relative to the uncertainty
                        phot_var += phot_scaling[phot_filter] ** 2 * obj_item[1, j] ** 2

                    elif instr_check in phot_scaling:
                        # Inflate photometric uncertainty for instrument

                        # Scale relative to the uncertainty
                        phot_var += phot_scaling[instr_check] ** 2 * obj_item[1, j] ** 2

                    ln_like += (
                        -0.5 * weight * (obj_item[0, j] - phot_flux) ** 2 / phot_var
                    )

                    # Only required when fitting an error inflation
                    ln_like += -0.5 * np.log(2.0 * np.pi * phot_var)

        for i, item in enumerate(self.spectrum.keys()):
            # Calculate or interpolate the model spectrum

            # Shortcut for the weight
            weight = self.weights[item]

            if self.model == "planck":
                # Calculate a blackbody spectrum
                readplanck = ReadPlanck(
                    (
                        0.9 * self.spectrum[item][0][0, 0],
                        1.1 * self.spectrum[item][0][-1, 0],
                    )
                )

                model_box = readplanck.get_spectrum(param_dict, spec_res=1000.0)

                # Resample the spectrum to the observed wavelengths
                model_flux = spectres.spectres(
                    self.spectrum[item][0][:, 0], model_box.wavelength, model_box.flux
                )

            else:
                # Interpolate the model spectrum from the grid

                if self.binary:
                    # Star 1

                    param_0 = binary_to_single(param_dict, 0)

                    model_flux_0 = self.modelspec[i].spectrum_interp(
                        list(param_0.values())
                    )[0, :]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        model_flux_0 *= flux_scaling
                        model_flux_0 += flux_offset

                    elif "radius_1" in self.modelpar:
                        model_flux_0 *= flux_scaling_0
                        model_flux_0 += flux_offset

                    # Star 2

                    param_1 = binary_to_single(param_dict, 1)

                    model_flux_1 = self.modelspec[i].spectrum_interp(
                        list(param_1.values())
                    )[0, :]

                    # Scale the spectrum by (radius/distance)^2

                    if "radius" in self.modelpar:
                        model_flux_1 *= flux_scaling
                        model_flux_1 += flux_offset

                    elif "radius_1" in self.modelpar:
                        model_flux_1 *= flux_scaling_1
                        model_flux_1 += flux_offset

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
                    model_flux += flux_offset

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

            # Apply radial velocity shift

            if item in rad_vel:
                wavel_shift = (
                    rad_vel[item] * 1e3 * self.spectrum[item][0][:, 0] / constants.LIGHT
                )

                spec_interp = interpolate.interp1d(
                    self.spectrum[item][0][:, 0] + wavel_shift,
                    model_flux,
                    fill_value="extrapolate",
                )

                model_flux = spec_interp(self.spectrum[item][0][:, 0])

            # Apply rotational broadening

            if "vsini" in self.modelpar:
                spec_interp = interpolate.interp1d(
                    self.spectrum[item][0][:, 0], model_flux
                )

                wavel_new = np.linspace(
                    self.spectrum[item][0][0, 0],
                    self.spectrum[item][0][-1, 0],
                    2 * self.spectrum[item][0].shape[0],
                )

                flux_broad = fastRotBroad(
                    wvl=wavel_new,
                    flux=spec_interp(wavel_new),
                    epsilon=0.0,
                    vsini=params[self.cube_index["vsini"]],
                    effWvl=None,
                )

                spec_interp = interpolate.interp1d(wavel_new, flux_broad)

                model_flux = spec_interp(self.spectrum[item][0][:, 0])

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
                    1e3 * constants.PARSEC / parallax
                ) ** 2

                model_flux += model_tmp

            if "lognorm_ext" in dust_param:
                cross_tmp = self.cross_sections["spectrum"](
                    (
                        self.spectrum[item][0][:, 0],
                        10.0 ** dust_param["lognorm_radius"],
                        dust_param["lognorm_sigma"],
                    )
                )

                model_flux *= np.exp(-cross_tmp * n_grains)

            elif "powerlaw_ext" in dust_param:
                cross_tmp = self.cross_sections["spectrum"](
                    (
                        self.spectrum[item][0][:, 0],
                        10.0 ** dust_param["powerlaw_max"],
                        dust_param["powerlaw_exp"],
                    )
                )

                model_flux *= np.exp(-cross_tmp * n_grains)

            elif "ism_ext" in dust_param:
                ism_reddening = dust_param.get("ism_red", 3.1)

                ext_spec = ism_extinction(
                    dust_param["ism_ext"], ism_reddening, self.spectrum[item][0][:, 0]
                )

                model_flux *= 10.0 ** (-0.4 * ext_spec)

            elif self.ext_filter is not None:
                ism_reddening = dust_param.get("ism_red", 3.1)

                av_required = convert_to_av(
                    filter_name=self.ext_filter,
                    filter_ext=dust_param[f"phot_ext_{self.ext_filter}"],
                    v_band_red=ism_reddening,
                )

                ext_spec = ism_extinction(
                    av_required, ism_reddening, self.spectrum[item][0][:, 0]
                )

                model_flux *= 10.0 ** (-0.4 * ext_spec)

            if self.spectrum[item][2] is not None:
                # Use the inverted covariance matrix
                ln_like += -0.5 * np.dot(
                    weight * (data_flux - model_flux),
                    np.dot(data_cov_inv, data_flux - model_flux),
                )

                ln_like += -0.5 * np.nansum(np.log(2.0 * np.pi * data_var))

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
                        * error_i**2
                    )

                    dot_tmp = np.dot(
                        weight * (data_flux - model_flux),
                        np.dot(np.linalg.inv(cov_matrix), data_flux - model_flux),
                    )

                    ln_like += -0.5 * dot_tmp
                    ln_like += -0.5 * np.nansum(np.log(2.0 * np.pi * data_var))

                else:
                    # Calculate the chi-square without a covariance matrix
                    chi_sq = -0.5 * weight * (data_flux - model_flux) ** 2 / data_var
                    chi_sq += -0.5 * np.log(2.0 * np.pi * data_var)

                    ln_like += np.nansum(chi_sq)

        return ln_like

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
        ``MultiNest`` has to to be build manually. See the
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

        except ModuleNotFoundError:
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
            len(self.modelpar), outputfiles_basename=output
        )

        # Get a dictionary with the ln(Z) and its errors, the
        # individual modes and their parameters quantiles of
        # the parameter posteriors
        sampling_stats = analyzer.get_stats()

        # Nested sampling global log-evidence
        ln_z = sampling_stats["nested sampling global log-evidence"]
        ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"\nNested sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Nested sampling global log-evidence
        ln_z = sampling_stats["nested importance sampling global log-evidence"]
        ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            f"Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}"
        )

        # Get the best-fit (highest likelihood) point

        print("\nSample with the highest probability:")
        best_params = analyzer.get_best_fit()

        max_lnlike = best_params["log_likelihood"]
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(best_params["parameters"]):
            if -0.1 < item < 0.1:
                print(f"   - {self.modelpar[i]} = {item:.2e}")
            else:
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

        # Dictionary with attributes that will be stored

        attr_dict = {
            "spec_type": "model",
            "spec_name": self.model,
            "ln_evidence": (ln_z, ln_z_error),
            "parallax": self.parallax[0],
        }

        if self.ext_filter is not None:
            attr_dict["ext_filter"] = self.ext_filter

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                sampler="multinest",
                samples=samples,
                ln_prob=ln_prob,
                tag=tag,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                spec_labels=spec_labels,
                attr_dict=attr_dict,
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

        except ModuleNotFoundError:
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
                # UltraNest can not handle np.inf in the likelihood
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

        ln_z = result["logz"]
        ln_z_error = result["logzerr"]
        print(f"\nLog-evidence = {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Best-fit parameters

        print("\nBest-fit parameters (mean +/- sigma):")

        for i, item in enumerate(self.modelpar):
            mean = np.mean(result["samples"][:, i])
            sigma = np.std(result["samples"][:, i])

            print(f"   - {item} = {mean:.2e} +/- {sigma:.2e}")

        # Get the best-fit (highest likelihood) point

        max_lnlike = result["maximum_likelihood"]["logl"]

        print("\nSample with the highest probability:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(result["maximum_likelihood"]["point"]):
            if -0.1 < item < 0.1:
                print(f"   - {self.modelpar[i]} = {item:.2e}")
            else:
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

        # Dictionary with attributes that will be stored

        attr_dict = {
            "spec_type": "model",
            "spec_name": self.model,
            "ln_evidence": (ln_z, ln_z_error),
            "parallax": self.parallax[0],
        }

        if self.ext_filter is not None:
            attr_dict["ext_filter"] = self.ext_filter

        # Get the MPI rank of the process

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                sampler="ultranest",
                samples=samples,
                ln_prob=ln_prob,
                tag=tag,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                spec_labels=spec_labels,
                attr_dict=attr_dict,
            )

    @typechecked
    def run_dynesty(
        self,
        tag: str,
        n_live_points: int = 2000,
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
        Function for running the atmospheric retrieval. The parameter
        estimation and computation of the marginalized likelihood (i.e.
        model evidence), is done with ``Dynesty``.

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

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            print(f"Creating output folder: {output}")
            os.mkdir(output)

        else:
            print(f"Output folder: {output}")

        print()

        out_basename = os.path.join(output, "retrieval_")

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

                        print(
                            f"Resumed a Dynesty run from {out_basename}dynesty.save"
                        )

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

                        print(
                            f"Resumed a Dynesty run from {out_basename}dynesty.save"
                        )

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

        results = dsampler.results
        samples = results.samples_equal()
        ln_prob = results.logl

        print(f"\nSamples shape: {samples.shape}")
        print(f"Number of iterations: {results.niter}")

        out_file = out_basename + "post_equal_weights.dat"
        print(f"Storing samples: {out_file}")
        np.savetxt(out_file, np.c_[samples, ln_prob])

        # Nested sampling global log-evidence
        # TODO check if selecting the last index is correct

        ln_z = results.logz[-1]
        ln_z_error = results.logzerr[-1]
        print(f"\nNested sampling log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Get the best-fit (highest likelihood) point

        max_idx = np.argmax(ln_prob)
        max_lnlike = ln_prob[max_idx]
        best_params = samples[max_idx]

        print("\nSample with the highest probability:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(self.modelpar):
            if -0.1 < best_params[i] < 0.1:
                print(f"   - {self.modelpar[i]} = {best_params[i]:.2e}")
            else:
                print(f"   - {self.modelpar[i]} = {best_params[i]:.2f}")

        spec_labels = []
        for item in self.spectrum:
            if f"scaling_{item}" in self.bounds:
                spec_labels.append(f"scaling_{item}")

        # Adding the fixed parameters to the samples

        for key, value in self.fix_param.items():
            self.modelpar.append(key)

            app_param = np.full(samples.shape[0], value)
            app_param = app_param[..., np.newaxis]

            samples = np.append(samples, app_param, axis=1)

        # Dictionary with attributes that will be stored

        attr_dict = {
            "spec_type": "model",
            "spec_name": self.model,
            "ln_evidence": (ln_z, ln_z_error),
            "parallax": self.parallax[0],
        }

        if self.ext_filter is not None:
            attr_dict["ext_filter"] = self.ext_filter

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            from species.data.database import Database

            species_db = Database()

            species_db.add_samples(
                sampler="dynesty",
                samples=samples,
                ln_prob=ln_prob,
                tag=tag,
                modelpar=self.modelpar,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                spec_labels=spec_labels,
                attr_dict=attr_dict,
            )
