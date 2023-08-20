"""
Module for generating atmospheric model spectra with ``petitRADTRANS``.
Details on the radiative transfer, atmospheric setup, and opacities
can be found in `Mollière et al. (2019) <https://ui.adsabs.harvard.edu
/abs/2019A%26A...627A..67M/abstract>`_.
"""

import warnings

from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spectres

from matplotlib.ticker import MultipleLocator
from PyAstronomy.pyasl import fastRotBroad
from scipy.interpolate import interp1d
from typeguard import typechecked

from species.analysis import photometry
from species.core import box, constants
from species.read import read_filter
from species.util import dust_util, read_util, retrieval_util


class ReadRadtrans:
    """
    Class for generating a model spectrum with ``petitRADTRANS``.
    """

    @typechecked
    def __init__(
        self,
        line_species: Optional[List[str]] = None,
        cloud_species: Optional[List[str]] = None,
        scattering: bool = False,
        wavel_range: Optional[Tuple[float, float]] = None,
        filter_name: Optional[str] = None,
        pressure_grid: str = "smaller",
        res_mode: str = "c-k",
        cloud_wavel: Optional[Tuple[float, float]] = None,
        max_press: float = None,
        pt_manual: Optional[np.ndarray] = None,
        lbl_opacity_sampling: Optional[Union[int, np.int_]] = None,
    ) -> None:
        """
        Parameters
        ----------
        line_species : list, None
            List with the line species. No line species are used if set
            to ``None``.
        cloud_species : list, None
            List with the cloud species. No clouds are used if set to
            ``None``.
        scattering : bool
            Include scattering in the radiative transfer.
        wavel_range : tuple(float, float), None
            Wavelength range (:math:`\\mu`m). The wavelength range is
            set to 0.8-10.0 :math:`\\mu`m if set to ``None`` or not
            used if ``filter_name`` is not ``None``.
        filter_name : str, None
            Filter name that is used for the wavelength range. The
            ``wavel_range`` is used if ``filter_name`` is set to
            ``None``.
        pressure_grid : str
            The type of pressure grid that is used for the radiative
            transfer. Either 'standard', to use 180 layers both for
            the atmospheric structure (e.g. when interpolating the
            abundances) and 180 layers with the radiative transfer,
            or 'smaller' to use 60 (instead of 180) with the radiative
            transfer, or 'clouds' to start with 1440 layers but
            resample to ~100 layers (depending on the number of cloud
            species) with a refinement around the cloud decks. For
            cloudless atmospheres it is recommended to use 'smaller',
            which runs faster than 'standard' and provides sufficient
            accuracy. For cloudy atmosphere, one can test with
            'smaller' but it is recommended to use 'clouds' for
            improved accuracy fluxes.
        res_mode : str
            Resolution mode ('c-k' or 'lbl'). The low-resolution mode
            ('c-k') calculates the spectrum with the correlated-k
            assumption at :math:`\\lambda/\\Delta \\lambda = 1000`. The
            high-resolution mode ('lbl') calculates the spectrum with a
            line-by-line treatment at
            :math:`\\lambda/\\Delta \\lambda = 10^6`.
        cloud_wavel : tuple(float, float), None
            Tuple with the wavelength range (:math:`\\mu`m) that is
            used for calculating the median optical depth of the
            clouds at the gas-only photosphere and then scaling the
            cloud optical depth to the value of ``log_tau_cloud``.
            The range of ``cloud_wavel`` should be encompassed by
            the range of ``wavel_range``.  The full wavelength
            range (i.e. ``wavel_range``) is used if the argument is
            set to ``None``.
        max_pressure : float, None
            Maximum pressure (bar) for the free temperature nodes.
            The default value is set to 1000 bar.
        pt_manual : np.ndarray, None
            A 2D array that contains the P-T profile that is used
            when ``pressure_grid="manual"``. The shape of array should
            be (n_pressure, 2), with pressure (bar) as first column
            and temperature (K) as second column. It is recommended
            that the pressures are logarithmically spaced.
        lbl_opacity_sampling : int, None
            This is the same parameter as in ``petitRADTRANS`` which is
            used with ``res_mode='lbl'`` to downsample the line-by-line
            opacities by selecting every ``lbl_opacity_sampling``-th
            wavelength from the original sampling of
            :math:`\\lambda/\\Delta \\lambda = 10^6`. Setting this
            parameter will lower the computation time. By setting the
            argument to ``None``, the original sampling is used so no
            downsampling is applied.

        Returns
        -------
        NoneType
            None
        """

        # Set several of the required ReadRadtrans attributes

        self.filter_name = filter_name
        self.wavel_range = wavel_range
        self.scattering = scattering
        self.pressure_grid = pressure_grid
        self.cloud_wavel = cloud_wavel
        self.pt_manual = pt_manual
        self.lbl_opacity_sampling = lbl_opacity_sampling

        # Set maximum pressure

        if max_press is None:
            self.max_press = 1e3
        else:
            self.max_press = max_press

        # Set the wavelength range

        if self.filter_name is not None:
            transmission = read_filter.ReadFilter(self.filter_name)
            self.wavel_range = transmission.wavelength_range()
            self.wavel_range = (0.9 * self.wavel_range[0], 1.2 * self.wavel_range[1])

        elif self.wavel_range is None:
            self.wavel_range = (0.8, 10.0)

        # Set the list with line species

        if line_species is None:
            self.line_species = []
        else:
            self.line_species = line_species

        # Set the list with cloud species and the number of P-T points

        if cloud_species is None:
            self.cloud_species = []
        else:
            self.cloud_species = cloud_species

        # Set the number of pressures

        if self.pressure_grid in ["standard", "smaller"]:
            # Initiate 180 pressure layers but use only
            # 60 layers during the radiative transfer
            # when pressure_grid is set to 'smaller'
            n_pressure = 180

        elif self.pressure_grid == "clouds":
            # Initiate 1140 pressure layers but use fewer
            # layers (~100) during the radiative tranfer
            # after running make_half_pressure_better
            n_pressure = 1440

        else:
            raise ValueError(
                f"The argument of pressure_grid "
                f"('{self.pressure_grid}') is "
                f"not recognized. Please use "
                f"'standard', 'smaller', or 'clouds'."
            )

        # Create 180 pressure layers in log space

        if self.pressure_grid == "manual":
            if self.pt_manual is None:
                raise UserWarning(
                    "A 2D array with the P-T profile "
                    "should be provided as argument "
                    "of pt_manual when using "
                    "pressure_grid='manual'."
                )

            self.pressure = self.pt_manual[:, 0]

        else:
            self.pressure = np.logspace(-6, np.log10(self.max_press), n_pressure)

        # Import petitRADTRANS here because it is slow

        print("Importing petitRADTRANS...", end="", flush=True)
        from petitRADTRANS.radtrans import Radtrans

        print(" [DONE]")

        # Create the Radtrans object

        self.rt_object = Radtrans(
            line_species=self.line_species,
            rayleigh_species=["H2", "He"],
            cloud_species=self.cloud_species,
            continuum_opacities=["H2-H2", "H2-He"],
            wlen_bords_micron=self.wavel_range,
            mode=res_mode,
            test_ck_shuffle_comp=self.scattering,
            do_scat_emis=self.scattering,
            lbl_opacity_sampling=lbl_opacity_sampling,
        )

        # Setup the opacity arrays

        if self.pressure_grid == "standard":
            self.rt_object.setup_opa_structure(self.pressure)

        elif self.pressure_grid == "manual":
            self.rt_object.setup_opa_structure(self.pressure)

        elif self.pressure_grid == "smaller":
            self.rt_object.setup_opa_structure(self.pressure[::3])

        elif self.pressure_grid == "clouds":
            self.rt_object.setup_opa_structure(self.pressure[::24])

    @typechecked
    def get_model(
        self,
        model_param: Dict[str, float],
        quenching: Optional[str] = None,
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
        plot_contribution: Optional[Union[bool, str]] = False,
        temp_nodes: Optional[Union[int, np.integer]] = None,
    ) -> box.ModelBox:
        """
        Function for calculating a model spectrum with
        radiative transfer code of ``petitRADTRANS``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters. Various
            parameterizations can be used for the
            pressure-temperature (P-T) profile, abundances
            (chemical equilibrium or free abundances), and
            the cloud properties. The type of parameterizations
            that will be used depend on the parameters provided
            in the dictionary of ``model_param``. Below is an
            (incomplete) list of the supported parameters.

            Mandatory parameters:

                - The surface gravity, ``logg``, should always
                  be included. It is provided in cgs units as
                  :math:`\\log_{10}{g}`.

            Scaling parameters (optional):

                - The radius (:math:`R_\\mathrm{J}`), ``radius``,
                  and parallax (mas), ``parallax``, are optional
                  parameters that can be included for scaling the
                  flux from the planet surface to the observer.

                - Instead of ``parallax``, it is also possible to
                  provided the distance (pc) with the ``distance``
                  parameter.

            Chemical abundances (mandatory -- one of the options
            should be used):

                - Chemical equilibrium requires the ``metallicity``,
                  ``c_o_ratio`` parameters. Optionally, the
                  ``log_p_quench`` (as :math:`\\log_{10}P/\\mathrm{bar}`)
                  can be included for setting a quench pressure for
                  CO/CH$_4$/H$_2$O. If this last parameter is used,
                  then the argument of ``quenching`` should be set
                  to ``'pressure'``.

                - Free abundances requires the parameters that have the
                  names from ``line_species`` and ``cloud_species``.
                  These will be used as :math:`\\log_{10}` mass fraction
                  of the line and cloud species. For example, if
                  ``line_species`` includes ``H2O_HITEMP``
                  then ``model_param`` should contain the ``H2O_HITEMP``
                  parameter. For a mass fraction of :math:`10^{-3}` the
                  dictionary value can be set to -3. Or, if
                  ``cloud_species`` contains ``MgSiO3(c)_cd`` then
                  ``model_param`` should contain the ``MgSiO3(c)``
                  parameter. So it is provided without the suffix,
                  ``_cd``, for the particle shape and structure.

            Pressure-temperature (P-T) profiles (mandatory -- one of
            the options should be used):

                - Eddington approximation requires the ``tint``
                  and ``log_delta`` parameters.

                - Parametrization from `Mollière et al (2020)
                  <https://ui.adsabs.harvard.edu/abs/2020A%26A...640A.
                  131M/abstract>`_ that was used for HR 8799 e. It
                  requires ``tint``, ``alpa``, ``log_delta``, ``t1``,
                  ``t2``, and ``t3`` as parameters.

                - Arbitrary number of free temperature nodes requires
                  parameters ``t0``, ``t1``, ``t2``, etc. So counting
                  from zero up to the number of nodes that are
                  required. The nodes will be interpolated to a larger
                  number of points in log-pressure space (set with the
                  ``pressure_grid`` parameter) by using a cubic spline.
                  Optionally, the ``pt_smooth`` parameter can also be
                  included in ``model_param``, which is used for
                  smoothing the interpolated P-T profile with a
                  Gaussian kernel in :math:`\\log{P/\\mathrm{bar}}`.
                  A recommended value for the kernel is 0.3 dex,
                  so ``pt_smooth=0.3``.

                - Instead of a parametrization, it is also possible
                  to provide a manual P-T profile as ``numpy`` array
                  with the argument of ``pt_manual``.

            Cloud models (optional -- one of the options can be used):

                - Physical clouds as in `Mollière et al (2020)
                  <https://ui.adsabs.harvard.edu/abs/2020A%26A...640A.
                  131M/abstract>`_ require the parameters ``fsed``,
                  ``log_kzz``, and ``sigma_lnorm``. Cloud abundances
                  are either specified relative to the equilibrium
                  abundances (when using chemical equilibrium
                  abundances for the line species) or as free
                  abundances (when using free abundances for the line
                  species). For the first case, the relative mass
                  fractions are specified for example with the
                  ``mgsio3_fraction`` parameter if the list with
                  ``cloud_species`` contains ``MgSiO3(c)_cd``.

                - With the physical clouds, instead of including the
                  mass fraction with the ``_fraction`` parameters,
                  it is also possible to enforce the clouds (to ensure
                  an effect on the spectrum) by scaling the opacities
                  with the ``log_tau_cloud`` parameter. This is the
                  wavelength-averaged optical depth of the clouds down
                  to the gas-only photosphere. The abundances are
                  now specified relative to the first cloud species
                  that is listed in ``cloud_species``. The ratio
                  parameters should be provided with the ``_ratio``
                  suffix. For example, if
                  ``cloud_species=['MgSiO3(c)_cd', 'Fe(c)_cd',
                  'Al2O3(c)_cd']`` then the ``fe_mgsio3_ratio`` and
                  ``al2o3_mgsio3_ratio`` parameters are required.

                - Instead of a single sedimentation parameter,
                  ``fsed``, it is also possible to include two values,
                  ``fsed_1`` and ``fsed_2``. This will calculate a
                  weighted combination of two cloudy spectra, to mimic
                  horizontal cloud variations. The weight should be
                  provided with the ``f_clouds`` parameter (between
                  0 and 1) in the ``model_param`` dictionary.

                - Parametrized cloud opacities with a cloud absorption
                  opacity, ``log_kappa_abs``, and powerlaw index,
                  ``opa_abs_index``. Furthermore, ``log_p_base`` and
                  ``fsed`` are required parameters. In addition to
                  absorption, parametrized scattering opacities are
                  added with the optional ``log_kappa_sca`` and
                  ``opa_sca_index`` parameters. Optionally, the
                  ``lambda_ray`` can be included, which is the
                  wavelength at which the opacity changes to a
                  :math:`\\lambda^{-4}` dependence in the Rayleigh
                  regime. It is also possible to include
                  ``log_tau_cloud``, which can be used for
                  enforcing clouds in the photospheric region by
                  scaling the cloud opacities.

                - Parametrized cloud opacities with a total cloud
                  opacity, ``log_kappa_0``, and a single scattering
                  albedo, ``albedo``. Furthermore, ``opa_index``,
                  ``log_p_base``, and ``fsed``, are required
                  parameters. This is `cloud model 2` from
                  `Mollière et al (2020) <https://ui.adsabs.harvard.
                  edu/abs/2020A%26A...640A.131M/abstract>`_
                  Optionally, ``log_tau_cloud`` can be used for
                  enforcing clouds in the photospheric region by
                  scaling the cloud opacities.

                - Gray clouds are simply parametrized with the
                  ``log_kappa_gray`` and ``log_cloud_top``
                  parameters. These clouds extend from the bottom
                  of the atmosphere up to the cloud top pressure and
                  have a constant opacity. Optionally, a single
                  scattering albedo, ``albedo``, can be specified.
                  Also ``log_tau_cloud`` can be used for enforcing
                  clouds in the photospheric region by scaling the
                  cloud opacities.

            Extinction (optional):

                 - Extinction can optionally be applied to the spectrum
                   by including the ``ism_ext`` parameter, which is the
                   the visual extinction, $A_V$. The empirical relation
                   from `Cardelli et al. (1989) <https://ui.adsabs.
                   harvard.edu/abs/1989ApJ...345..245C/abstract>`_
                   is used for calculating the extinction at other
                   wavelengths.

                 - When using ``ism_ext``, the reddening, $R_V$, can
                   also be optionaly set with the ``ism_red``
                   parameter. Otherwise it is set to the standard
                   value for the diffuse ISM, $R_V = 3.1$.

            Radial velocity and broadening (optional):

                 - Radial velocity shift can be applied by adding the
                   ``rad_vel`` parameter. This shifts the spectrum
                   by a constant velocity (km/s).

                 - Rotational broadening can be applied by adding the
                   ``vsini`` parameter, which is the projected spin
                   velocity (km/s), :math:`v\\sin{i}`. The broadening
                   is applied with the ``fastRotBroad`` function from
                   ``PyAstronomy`` (see for details the `documentation
                   <https://pyastronomy.readthedocs.io/en/latest/
                   pyaslDoc/aslDoc/ rotBroad.html#fastrotbroad-a-
                   faster-algorithm>`_).

        quenching : str, None
            Quenching type for CO/CH$_4$/H$_2$O abundances. Either
            the quenching pressure (bar) is a free parameter
            (``quenching='pressure'``) or the quenching pressure is
            calculated from the mixing and chemical timescales
            (``quenching='diffusion'``). The quenching is not applied
            if the argument is set to ``None``.
        spec_res : float, None
            Spectral resolution, achieved by smoothing with a Gaussian
            kernel. No smoothing is applied when the argument is set
            to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (:math:`\\mu`m) to which the spectrum
            will be resampled. The original wavelengths points will
            be used if the argument is set to ``None``.
        plot_contribution : bool, str, None
            Filename for the plot with the emission contribution. The
            plot is not created if the argument is set to ``False`` or
            ``None``. If set to ``True``, the plot is shown in an
            interface window instead of written to a file.
        temp_nodes : int, None
            Number of free temperature nodes.

        Returns
        -------
        species.core.box.ModelBox
            Box with the petitRADTRANS model spectrum.
        """

        # Set chemistry type

        if "metallicity" in model_param and "c_o_ratio" in model_param:
            chemistry = "equilibrium"

        else:
            chemistry = "free"

            check_nodes = {}

            for line_item in self.line_species:
                abund_count = 0

                for node_idx in range(100):
                    if f"{line_item}_{node_idx}" in model_param:
                        abund_count += 1
                    else:
                        break

                check_nodes[line_item] = abund_count

            # Check if there are an equal number of
            # abundance nodes for all the line species

            nodes_list = list(check_nodes.values())

            if not all(value == nodes_list[0] for value in nodes_list):
                raise ValueError(
                    "The number of abundance nodes is "
                    "not equal for all the lines "
                    f"species: {check_nodes}"
                )

            if all(value == 0 for value in nodes_list):
                abund_nodes = None
            else:
                abund_nodes = nodes_list[0]

            for line_item in self.line_species:
                if abund_nodes is None:
                    if line_item not in model_param:
                        raise RuntimeError(
                            f"The abundance of {line_item} is not "
                            "found in the dictionary with parameters "
                            "of 'model_param'. Please add the log10 "
                            f"mass fraction of {line_item}."
                        )

                else:
                    for node_idx in range(abund_nodes):
                        if f"{line_item}_{node_idx}" not in model_param:
                            raise RuntimeError(
                                f"The abundance of {line_item} is not "
                                "found in the dictionary with parameters "
                                "of 'model_param'. Please add the log10 "
                                f"mass fraction of {line_item}."
                            )

        # Check quenching parameter

        if not hasattr(self, "quenching"):
            self.quenching = quenching

        if self.quenching is not None and chemistry != "equilibrium":
            raise ValueError(
                "The 'quenching' parameter can only be used in combination with "
                "chemistry='equilibrium'."
            )

        if self.quenching is not None and self.quenching not in [
            "pressure",
            "diffusion",
        ]:
            raise ValueError(
                "The argument of 'quenching' should be of the following: "
                "'pressure', 'diffusion', or None."
            )

        # Abundance nodes

        if chemistry == "free" and abund_nodes is not None:
            knot_press_abund = np.logspace(
                np.log10(self.pressure[0]), np.log10(self.pressure[-1]), abund_nodes
            )

        else:
            knot_press_abund = None

        # C/O and [Fe/H]

        if chemistry == "equilibrium":
            # Equilibrium chemistry
            metallicity = model_param["metallicity"]
            c_o_ratio = model_param["c_o_ratio"]

            log_x_abund = None

        elif chemistry == "free":
            # Free chemistry

            # TODO Set [Fe/H] = 0 for Molliere P-T profile and
            # cloud condensation profiles
            metallicity = 0.0

            # Get smoothing parameter for abundance profiles

            if "abund_smooth" in model_param:
                abund_smooth = model_param["abund_smooth"]

            else:
                abund_smooth = None

            # Create a dictionary with the mass fractions

            if abund_nodes is None:
                log_x_abund = {}
                for line_item in self.line_species:
                    log_x_abund[line_item] = model_param[line_item]

                _, _, c_o_ratio = retrieval_util.calc_metal_ratio(
                    log_x_abund, self.line_species
                )

            else:
                log_x_abund = {}
                for line_item in self.line_species:
                    for node_idx in range(abund_nodes):
                        log_x_abund[f"{line_item}_{node_idx}"] = model_param[
                            f"{line_item}_{node_idx}"
                        ]

                # TODO Set C/O = 0.55 for Molliere P-T profile
                # and cloud condensation profiles
                c_o_ratio = 0.55

        # Create the P-T profile

        if self.pressure_grid == "manual":
            temp = self.pt_manual[:, 1]

        elif (
            "tint" in model_param
            and "log_delta" in model_param
            and "alpha" in model_param
        ):
            temp, _, _ = retrieval_util.pt_ret_model(
                np.array([model_param["t1"], model_param["t2"], model_param["t3"]]),
                10.0 ** model_param["log_delta"],
                model_param["alpha"],
                model_param["tint"],
                self.pressure,
                metallicity,
                c_o_ratio,
            )

        elif "tint" in model_param and "log_delta" in model_param:
            tau = self.pressure * 1e6 * 10.0 ** model_param["log_delta"]
            temp = (0.75 * model_param["tint"] ** 4.0 * (2.0 / 3.0 + tau)) ** 0.25

        else:
            if temp_nodes is None:
                temp_nodes = 0

                for temp_idx in range(100):
                    if f"t{temp_idx}" in model_param:
                        temp_nodes += 1
                    else:
                        break

            knot_press = np.logspace(
                np.log10(self.pressure[0]), np.log10(self.pressure[-1]), temp_nodes
            )

            knot_temp = []
            for temp_idx in range(temp_nodes):
                knot_temp.append(model_param[f"t{temp_idx}"])

            knot_temp = np.asarray(knot_temp)

            if "pt_smooth" in model_param:
                pt_smooth = model_param["pt_smooth"]

            else:
                pt_smooth = None

            temp = retrieval_util.pt_spline_interp(
                knot_press,
                knot_temp,
                self.pressure,
                pt_smooth=pt_smooth,
            )

        # Set the log quenching pressure, log(P/bar)

        if self.quenching == "pressure":
            p_quench = 10.0 ** model_param["log_p_quench"]

        elif self.quenching == "diffusion":
            p_quench = retrieval_util.quench_pressure(
                self.pressure,
                temp,
                model_param["metallicity"],
                model_param["c_o_ratio"],
                model_param["logg"],
                model_param["log_kzz"],
            )

        else:
            if "log_p_quench" in model_param:
                warnings.warn(
                    "The 'model_param' dictionary contains the "
                    "'log_p_quench' parameter but 'quenching=None'. "
                    "The quenching pressure from the dictionary is "
                    "therefore ignored."
                )

            p_quench = None

        if (
            len(self.cloud_species) > 0
            or "log_kappa_0" in model_param
            or "log_kappa_gray" in model_param
            or "log_kappa_abs" in model_param
        ):
            tau_cloud = None
            log_x_base = None

            if (
                "log_kappa_0" in model_param
                or "log_kappa_gray" in model_param
                or "log_kappa_abs" in model_param
            ):
                if "log_tau_cloud" in model_param:
                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    tau_cloud = model_param["tau_cloud"]

            elif chemistry == "equilibrium":
                # Create the dictionary with the mass fractions of the
                # clouds relative to the maximum values allowed from
                # elemental abundances

                cloud_fractions = {}

                for item in self.cloud_species:
                    if f"{item[:-3].lower()}_fraction" in model_param:
                        cloud_fractions[item] = model_param[
                            f"{item[:-3].lower()}_fraction"
                        ]

                    elif f"{item[:-3].lower()}_tau" in model_param:
                        # Import the chemistry module here because it is slow

                        from poor_mans_nonequ_chem.poor_mans_nonequ_chem import (
                            interpol_abundances,
                        )

                        # Interpolate the abundances, following chemical equilibrium

                        abund_in = interpol_abundances(
                            np.full(self.pressure.size, c_o_ratio),
                            np.full(self.pressure.size, metallicity),
                            temp,
                            self.pressure,
                            Pquench_carbon=p_quench,
                        )

                        # Extract the mean molecular weight

                        mmw = abund_in["MMW"]

                        # Calculate the scaled mass fraction of the clouds

                        cloud_fractions[item] = retrieval_util.scale_cloud_abund(
                            model_param,
                            self.rt_object,
                            self.pressure,
                            temp,
                            mmw,
                            "equilibrium",
                            abund_in,
                            item,
                            model_param[f"{item[:-3].lower()}_tau"],
                            pressure_grid=self.pressure_grid,
                        )

                if "log_tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = model_param["tau_cloud"]

                if tau_cloud is not None:
                    for i, item in enumerate(self.cloud_species):
                        if i == 0:
                            cloud_fractions[item] = 0.0

                        else:
                            cloud_1 = item[:-3].lower()
                            cloud_2 = self.cloud_species[0][:-3].lower()

                            cloud_fractions[item] = model_param[
                                f"{cloud_1}_{cloud_2}_ratio"
                            ]

                # Create a dictionary with the log mass fractions at the cloud base

                log_x_base = retrieval_util.log_x_cloud_base(
                    c_o_ratio, metallicity, cloud_fractions
                )

            elif chemistry == "free":
                # Add the log10 mass fractions of the clouds to the dictionary

                log_x_base = {}

                if "log_tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = 10.0 ** model_param["log_tau_cloud"]

                elif "tau_cloud" in model_param:
                    # Set the log mass fraction to zero and use the
                    # optical depth parameter to scale the cloud mass
                    # fraction with petitRADTRANS

                    tau_cloud = model_param["tau_cloud"]

                if tau_cloud is None:
                    for item in self.cloud_species:
                        # Set the log10 of the mass fractions at th
                        # cloud base equal to the value from the
                        # parameter dictionary
                        log_x_base[item[:-3]] = model_param[item]

                else:
                    # Set the log10 of the mass fractions with the
                    # ratios from the parameter dictionary and
                    # scale to the actual mass fractions with
                    # tau_cloud that is used in calc_spectrum_clouds
                    for i, item in enumerate(self.cloud_species):
                        if i == 0:
                            log_x_base[item[:-3]] = 0.0

                        else:
                            cloud_1 = item[:-3].lower()
                            cloud_2 = self.cloud_species[0][:-3].lower()

                            log_x_base[item[:-3]] = model_param[
                                f"{cloud_1}_{cloud_2}_ratio"
                            ]

            # Calculate the petitRADTRANS spectrum
            # for a cloudy atmosphere

            if "fsed_1" in model_param and "fsed_2" in model_param:
                cloud_dict = model_param.copy()
                cloud_dict["fsed"] = cloud_dict["fsed_1"]

                (
                    wavelength,
                    flux_1,
                    emission_contr_1,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    cloud_dict,
                    model_param["logg"],
                    chemistry=chemistry,
                    knot_press_abund=knot_press_abund,
                    abund_smooth=abund_smooth,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

                cloud_dict = model_param.copy()
                cloud_dict["fsed"] = cloud_dict["fsed_2"]

                (
                    wavelength,
                    flux_2,
                    emission_contr_2,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    cloud_dict,
                    model_param["logg"],
                    chemistry=chemistry,
                    knot_press_abund=knot_press_abund,
                    abund_smooth=abund_smooth,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

                flux = (
                    model_param["f_clouds"] * flux_1
                    + (1.0 - model_param["f_clouds"]) * flux_2
                )

                emission_contr = (
                    model_param["f_clouds"] * emission_contr_1
                    + (1.0 - model_param["f_clouds"]) * emission_contr_2
                )

            else:
                (
                    wavelength,
                    flux,
                    emission_contr,
                    _,
                ) = retrieval_util.calc_spectrum_clouds(
                    self.rt_object,
                    self.pressure,
                    temp,
                    c_o_ratio,
                    metallicity,
                    p_quench,
                    log_x_abund,
                    log_x_base,
                    model_param,
                    model_param["logg"],
                    chemistry=chemistry,
                    knot_press_abund=knot_press_abund,
                    abund_smooth=abund_smooth,
                    pressure_grid=self.pressure_grid,
                    plotting=False,
                    contribution=True,
                    tau_cloud=tau_cloud,
                    cloud_wavel=self.cloud_wavel,
                )

        elif chemistry == "equilibrium":
            # Calculate the petitRADTRANS spectrum for a clear atmosphere

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object,
                self.pressure,
                temp,
                model_param["logg"],
                model_param["c_o_ratio"],
                model_param["metallicity"],
                p_quench,
                None,
                pressure_grid=self.pressure_grid,
                chemistry=chemistry,
                knot_press_abund=knot_press_abund,
                abund_smooth=abund_smooth,
                contribution=True,
            )

        elif chemistry == "free":
            log_x_abund = {}

            if abund_nodes is None:
                for line_item in self.rt_object.line_species:
                    log_x_abund[line_item] = model_param[line_item]

            else:
                for line_item in self.rt_object.line_species:
                    for node_idx in range(abund_nodes):
                        log_x_abund[f"{line_item}_{node_idx}"] = model_param[
                            f"{line_item}_{node_idx}"
                        ]

            wavelength, flux, emission_contr = retrieval_util.calc_spectrum_clear(
                self.rt_object,
                self.pressure,
                temp,
                model_param["logg"],
                None,
                None,
                None,
                log_x_abund,
                chemistry=chemistry,
                knot_press_abund=knot_press_abund,
                abund_smooth=abund_smooth,
                pressure_grid=self.pressure_grid,
                contribution=True,
            )

        if "radius" in model_param:
            # Calculate the planet mass from log(g) and radius

            model_param["mass"] = read_util.get_mass(
                model_param["logg"], model_param["radius"]
            )

            # Scale the flux to the observer

            if "parallax" in model_param:
                scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                    1e3 * constants.PARSEC / model_param["parallax"]
                ) ** 2

                flux *= scaling

            elif "distance" in model_param:
                scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                    model_param["distance"] * constants.PARSEC
                ) ** 2

                flux *= scaling

        # Apply ISM extinction

        if "ism_ext" in model_param:
            if "ism_red" in model_param:
                ism_reddening = model_param["ism_red"]

            else:
                # Use default ISM reddening (R_V = 3.1) if ism_red is not provided
                ism_reddening = 3.1

            flux = dust_util.apply_ism_ext(
                wavelength, flux, model_param["ism_ext"], ism_reddening
            )

        # Plot 2D emission contribution

        if plot_contribution:
            # Calculate the total optical depth (line and continuum opacities)
            # self.rt_object.calc_opt_depth(10.**model_param['logg'])

            # From Paul: The first axis of total_tau is the coordinate
            # of the cumulative opacity distribution function (ranging
            # from 0 to 1). A correct average is obtained by
            # multiplying the first axis with self.w_gauss, then
            # summing them. This is then the actual wavelength-mean.

            if self.scattering:
                # From petitRADTRANS: Only use 0 index for species
                # because for lbl or test_ck_shuffle_comp = True
                # everything has been moved into the 0th index
                w_gauss = self.rt_object.w_gauss[..., np.newaxis, np.newaxis]
                optical_depth = np.sum(
                    w_gauss * self.rt_object.total_tau[:, :, 0, :], axis=0
                )

            else:
                # TODO Is this correct?
                w_gauss = self.rt_object.w_gauss[
                    ..., np.newaxis, np.newaxis, np.newaxis
                ]
                optical_depth = np.sum(
                    w_gauss * self.rt_object.total_tau[:, :, :, :], axis=0
                )

                # Sum over all species
                optical_depth = np.sum(optical_depth, axis=1)

            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"

            plt.figure(figsize=(8.0, 4.0))
            gridsp = mpl.gridspec.GridSpec(1, 1)
            gridsp.update(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            ax = plt.subplot(gridsp[0, 0])

            ax.tick_params(
                axis="both",
                which="major",
                colors="black",
                labelcolor="black",
                direction="in",
                width=1,
                length=5,
                labelsize=12,
                top=True,
                bottom=True,
                left=True,
                right=True,
            )

            ax.tick_params(
                axis="both",
                which="minor",
                colors="black",
                labelcolor="black",
                direction="in",
                width=1,
                length=3,
                labelsize=12,
                top=True,
                bottom=True,
                left=True,
                right=True,
            )

            ax.set_xlabel("Wavelength (µm)", fontsize=13)
            ax.set_ylabel("Pressure (bar)", fontsize=13)

            ax.get_xaxis().set_label_coords(0.5, -0.09)
            ax.get_yaxis().set_label_coords(-0.07, 0.5)

            ax.set_yscale("log")

            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))

            press_bar = 1e-6 * self.rt_object.press  # (Ba) -> (Bar)

            xx_grid, yy_grid = np.meshgrid(wavelength, press_bar)

            emission_contr = np.nan_to_num(emission_contr)

            ax.pcolormesh(
                xx_grid,
                yy_grid,
                emission_contr,
                cmap=plt.cm.bone_r,
                shading="gouraud",
            )

            photo_press = np.zeros(wavelength.shape[0])

            for i in range(photo_press.shape[0]):
                press_interp = interp1d(optical_depth[i, :], self.rt_object.press)
                photo_press[i] = press_interp(1.0) * 1e-6  # cgs to (bar)

            ax.plot(wavelength, photo_press, lw=0.5, color="gray")

            ax.set_xlim(np.amin(wavelength), np.amax(wavelength))
            ax.set_ylim(np.amax(press_bar), np.amin(press_bar))

            if isinstance(plot_contribution, str):
                plt.savefig(plot_contribution, bbox_inches="tight")
            else:
                plt.show()

            plt.clf()
            plt.close()

        # Convolve with a broadening kernel for vsin(i)

        if "vsini" in model_param:
            # fastRotBroad requires a regular
            # wavelength sampling while pRT uses
            # a logarithmic wavelength sampling
            wavel_even = np.linspace(
                np.amin(wavelength), np.amax(wavelength), wavelength.size * 10
            )

            # So change temporarily to a linear sampling
            # with a factor 10 larger number of wavelengths
            spec_interp = interp1d(wavelength, flux)
            flux_even = spec_interp(wavel_even)

            # Apply the rotational broadening
            spec_broad = fastRotBroad(
                wvl=wavel_even,
                flux=flux_even,
                epsilon=1.0,
                vsini=model_param["vsini"],
                effWvl=None,
            )

            # The rotBroad function is much slower than
            # fastRotBroad when tested on a large array
            # spec_broad = rotBroad(wvl=wavel_even,
            #                       flux=flux_even,
            #                       epsilon=1.,
            #                       vsini=model_param['vsini'],
            #                       edgeHandling='firstlast')

            # And change back to the original (logarithmic)
            # wavelength sampling, with constant R
            spec_interp = interp1d(wavel_even, spec_broad)
            flux = spec_interp(wavelength)

        # Convolve the spectrum with a Gaussian LSF

        if spec_res is not None:
            flux = retrieval_util.convolve(wavelength, flux, spec_res)

        # Apply a radial velocity shift to the wavelengths

        if "rad_vel" in model_param:
            # Change speed of light from (m/s) to (km/s)
            wavelength *= 1.0 - model_param["rad_vel"] / (constants.LIGHT * 1e-3)

        # Resample the spectrum

        if wavel_resample is not None:
            flux = spectres.spectres(
                wavel_resample,
                wavelength,
                flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

            wavelength = wavel_resample

        if hasattr(self.rt_object, "h_bol"):
            pressure = 1e-6 * self.rt_object.press  # (bar)
            f_bol = -4.0 * np.pi * self.rt_object.h_bol
            f_bol *= 1e-3  # (erg s-1 cm-2) -> (W m-2)
            bol_flux = np.column_stack((pressure, f_bol))

        else:
            bol_flux = None

        return box.create_box(
            boxtype="model",
            model="petitradtrans",
            wavelength=wavelength,
            flux=flux,
            parameters=model_param,
            quantity="flux",
            contribution=emission_contr,
            bol_flux=bol_flux,
        )

    @typechecked
    def get_flux(self, model_param: Dict[str, float]) -> Tuple[float, None]:
        """
        Function for calculating the filter-weighted flux density
        for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.

        Returns
        -------
        float
            Flux (W m-2 um-1).
        NoneType
            Uncertainty (W m-2 um-1). Always set to ``None``.
        """

        if "log_p_quench" in model_param:
            quenching = "pressure"
        else:
            quenching = None

        spectrum = self.get_model(model_param, quenching=quenching)

        synphot = photometry.SyntheticPhotometry(self.filter_name)

        return synphot.spectrum_to_flux(spectrum.wavelength, spectrum.flux)

    @typechecked
    def get_magnitude(self, model_param: Dict[str, float]) -> Tuple[float, None]:
        """
        Function for calculating the magnitude for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.

        Returns
        -------
        float
            Magnitude.
        NoneType
            Uncertainty. Always set to ``None``.
        """

        if "log_p_quench" in model_param:
            quenching = "pressure"
        else:
            quenching = None

        spectrum = self.get_model(model_param, quenching=quenching)

        synphot = photometry.SyntheticPhotometry(self.filter_name)
        app_mag, _ = synphot.spectrum_to_magnitude(spectrum.wavelength, spectrum.flux)

        return app_mag
