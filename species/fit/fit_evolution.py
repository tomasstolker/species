"""
Module with functionalities for retrieving the age and
bulk parameters of one or multiple planets and/or brown
dwarfs in a system.
"""

import configparser
import os
import sys
import warnings

from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from scipy.stats import norm, truncnorm
from tqdm.auto import tqdm
from typeguard import typechecked

from species.core import constants
from species.read.read_isochrone import ReadIsochrone
from species.util.core_util import print_section


class FitEvolution:
    """
    Class for retrieving evolutionary parameters from the bolometric
    luminosity of one or multiple planets in a planetary system.
    Optionally, dynamical mass and/or radius priors can be applied
    A single age is retrieved, so assuming that a difference
    in formation time is negligible at the age of the system.
    """

    @typechecked
    def __init__(
        self,
        evolution_model: str,
        log_lum: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
        age_prior: Optional[Tuple[float, float, float]] = None,
        mass_prior: Optional[
            Union[Tuple[float, float], List[Optional[Tuple[float, float]]]]
        ] = None,
        radius_prior: Optional[
            Union[Tuple[float, float], List[Optional[Tuple[float, float]]]]
        ] = None,
        bounds: Optional[
            Dict[
                str,
                Union[
                    Tuple[float, float],
                    Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]],
                    List[Tuple[float, float]],
                ],
            ]
        ] = None,
        interp_method: str = "linear",
    ) -> None:
        """
        Parameters
        ----------
        evolution_model : str
            Database tag of the isochrone data (e.g. 'ames-cond',
            'sonora+0.0', 'atmo-ceq'). When using an incorrect
            argument, and error message is printed that includes
            a list with the isochrone models that are available
            in the current ``species`` database.
        log_lum : tuple(float, float), list(tuple(float, float))
            List with tuples that contain :math:`\\log10{L/L_\\odot}`
            and the related uncertainty for one or multiple objects.
            The list should follow the alphabetical order of companion
            characters (i.e. b, c, d, etc.) to make sure that the
            labels are correctly shown when plotting results.
        age_prior : tuple(float, float, float), None
            Tuple with an optional (asymmetric) normal prior for the
            age (Myr). The tuple should contain three values, for
            example, ``age_prior=(20., -5., +2.)``. The prior is not
            applied if the argument is set to ``None``.
        mass_prior : tuple(float, float), list(tuple(float, float)), None
            Optional list with tuples that contain the (dynamical)
            masses and the related uncertainty for one or multiple
            objects. These masses will be used as normal prior with
            the fit. The order should be identical to ``log_lum``.
            For fitting multiple objects, an item in the list can be
            to ``None`` to not apply the normal prior on a specific
            object.
        radius_prior : tuple(float, float), list(tuple(float, float)), None
            Optional list with tuples that contain the radii (e.g.
            from and SED fit) and the related uncertainty for one
            or multiple objects. These radii will be used as normal
            prior with the fit. The order should be identical to
            ``log_lum``. For fitting multiple objects, an item in
            the list can be to ``None`` to not apply the normal
            prior on a specific object.
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform or
            log-uniform priors. Fixing a parameter is possible by
            providing the same value as lower and upper boundary
            of the parameter (e.g. ``bounds={'mass_0': (5.0, 5.0)``.
        interp_method : str
            Interpolation method for the isochrone data. The argument
            should be either 'linear', for using a linear 2D
            interpolation with `LinearNDInterpolator
            <https://docs.scipy.org/doc/scipy/reference/
            generated/scipy.interpolate.LinearNDInterpolator.html>`_,
            or 'cubic', for using a 2D cubic interpolation with
            `CloughTocher2DInterpolator <https://docs.scipy.org/
            doc/scipy/reference/generated/scipy.interpolate.
            CloughTocher2DInterpolator.html>`_ (default: 'linear').

        Returns
        -------
        NoneType
            None
        """

        print_section("Fit evolutionary model")

        self.evolution_model = evolution_model
        self.log_lum = log_lum
        self.mass_prior = mass_prior
        self.radius_prior = radius_prior
        self.bounds = bounds
        self.age_prior = age_prior
        self.normal_prior = {}
        self.fix_param = {}
        self.interp_method = interp_method

        print(f"Evolution model: {self.evolution_model}")
        print(f"Luminosity log(L/Lsun): {self.log_lum}")

        if isinstance(self.log_lum, tuple):
            self.log_lum = [self.log_lum]

        self.n_planets = len(self.log_lum)

        if self.mass_prior is None:
            self.mass_prior = []
            for i in range(self.n_planets):
                self.mass_prior.append(None)

        elif isinstance(self.mass_prior, tuple):
            self.mass_prior = [self.mass_prior]

        if self.radius_prior is None:
            self.radius_prior = []
            for i in range(self.n_planets):
                self.radius_prior.append(None)

        elif isinstance(self.radius_prior, tuple):
            self.radius_prior = [self.radius_prior]

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.database_path = config["species"]["database"]

        # Add grid with evolution data

        with h5py.File(self.database_path, "r") as hdf_file:
            found_group = bool("isochrones" in hdf_file)

            if found_group:
                found_model = bool(f"isochrones/{evolution_model}" in hdf_file)
                tag_list = list(hdf_file["isochrones"])
            else:
                found_model = False
                tag_list = None

        if not found_model:
            raise ValueError(
                f"The isochrones of '{evolution_model}' "
                "are not found in the database. Please "
                "add the isochrones by using the "
                "add_isochrones method of Database. The "
                "following isochrone data are found in "
                f"the database: {tag_list}"
            )

        # Read isochrone grid

        self.read_iso = ReadIsochrone(
            tag=self.evolution_model,
            create_regular_grid=False,
            verbose=False,
            interp_method=self.interp_method,
        )

        # Model parameters

        self.model_par = ["age"]

        for i in range(self.n_planets):
            self.model_par.append(f"mass_{i}")

        if "s_init" in self.read_iso.get_points():
            for i in range(self.n_planets):
                self.model_par.append(f"s_init_{i}")

        # Check if the log_lum values are within the
        # available range of the evolutionary grid

        iso_data = self.read_iso.read_data()

        for planet_idx in range(self.n_planets):
            if self.log_lum[planet_idx][0] < np.amin(iso_data["log_lum"]):
                warnings.warn(
                    f"The luminosity of the object with index {planet_idx}, "
                    f"log_lum={self.log_lum[planet_idx][0]}, is smaller than "
                    f"the minimum luminosity in the '{self.evolution_model}' "
                    f"grid, log(L/Lsun)={np.amin(iso_data['log_lum']):.2f}."
                )

            if self.log_lum[planet_idx][0] > np.amax(iso_data["log_lum"]):
                warnings.warn(
                    f"The luminosity of the object with index {planet_idx}, "
                    f"log_lum={self.log_lum[planet_idx][0]}, is larger than "
                    f"the maximum luminosity in the '{self.evolution_model}' "
                    f"grid, log(L/Lsun)={np.amax(iso_data['log_lum']):.2f}."
                )

        # Prior boundaries

        if self.bounds is not None:
            # Set manual prior boundaries

            bounds_grid = {}
            for param_key, param_value in self.read_iso.get_points().items():
                if param_key in ["age", "mass", "s_init"]:
                    bounds_grid[param_key] = (
                        np.amin(param_value),
                        np.amax(param_value),
                    )

            for param_key, param_value in bounds_grid.items():
                for planet_idx in range(self.n_planets):
                    if param_key == "age":
                        param_new = param_key
                    else:
                        param_new = f"{param_key}_{planet_idx}"

                    if param_new not in self.bounds:
                        # Set the parameter boundaries to the grid
                        # boundaries if set to None or not found
                        self.bounds[param_new] = bounds_grid[param_key]

                    else:
                        if self.bounds[param_new][0] < bounds_grid[param_key][0]:
                            warnings.warn(
                                f"The lower bound on {param_new} "
                                f"({self.bounds[param_new][0]}) is smaller than "
                                f"the lower bound from the available "
                                f"evolution model grid "
                                f"({bounds_grid[param_key][0]}). The lower bound "
                                f"of the {param_new} prior will be adjusted to "
                                f"{bounds_grid[param_key][0]}."
                            )

                            self.bounds[param_new] = (
                                bounds_grid[param_key][0],
                                self.bounds[param_new][1],
                            )

                        if self.bounds[param_new][1] > bounds_grid[param_key][1]:
                            warnings.warn(
                                f"The upper bound on {param_new} "
                                f"({self.bounds[param_new][1]}) is larger than the "
                                f"upper bound from the available evolution "
                                f"model grid ({bounds_grid[param_key][1]}). The "
                                f"bound of the {param_new} prior will be adjusted "
                                f"to {bounds_grid[param_key][1]}."
                            )

                            self.bounds[param_new] = (
                                self.bounds[param_new][0],
                                bounds_grid[param_key][1],
                            )

            for i in range(self.n_planets):
                if f"inflate_lbol{i}" in self.bounds:
                    self.model_par.append(f"inflate_lbol{i}")

            for i in range(self.n_planets):
                if f"inflate_mass{i}" in self.bounds:
                    if self.mass_prior[i] is None:
                        warnings.warn(
                            f"The mass_prior with index "
                            f"{i} is set to None so the "
                            f"inflate_mass{i} parameter "
                            f"will be excluded."
                        )

                        del self.bounds[f"inflate_mass{i}"]

                    else:
                        self.model_par.append(f"inflate_mass{i}")

        else:
            # Set all prior boundaries to the grid boundaries

            self.bounds = {}
            for key, value in self.read_iso.get_points().items():
                if key == "age":
                    self.bounds[key] = (np.amin(value), np.amax(value))

                elif key in ["mass", "s_init"]:
                    for i in range(self.n_planets):
                        self.bounds[f"{key}_{i}"] = (np.amin(value), np.amax(value))

        # Check if parameters are fixed

        del_param = []

        for key, value in self.bounds.items():
            if value[0] == value[1] and value[0] is not None and value[1] is not None:
                self.fix_param[key] = value[0]
                del_param.append(key)

        for item in del_param:
            del self.bounds[item]
            self.model_par.remove(item)

        if self.fix_param:
            print(f"\nFixing {len(self.fix_param)} parameters:")

            for key, value in self.fix_param.items():
                print(f"   - {key} = {value}")

        print(f"\nFitting {len(self.model_par)} parameters:")

        for item in self.model_par:
            print(f"   - {item}")

        # Printing uniform and normal priors

        print("\nUniform priors (min, max):")

        for param_key, param_value in self.bounds.items():
            print(f"   - {param_key} = {param_value}")

        print(f"\nNormal priors (mean, sigma):")
        print(f"   - Age (Myr): {self.age_prior}")
        print(f"   - Mass (Mjup): {self.mass_prior}")
        print(f"   - Radius (Rjup): {self.radius_prior}")

        if len(self.normal_prior) > 0:
            # Not used by the current implementation
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

    @typechecked
    def run_multinest(
        self,
        tag: str,
        n_live_points: int = 200,
        output: str = "multinest/",
    ) -> None:
        """
        Function to run the ``PyMultiNest`` wrapper of the
        ``MultiNest`` sampler. While ``PyMultiNest`` can be
        installed with ``pip`` from the PyPI repository,
        ``MultiNest`` has to be built manually. See the
        ``PyMultiNest`` documentation for details:
        http://johannesbuchner.github.io/PyMultiNest/install.html.
        Note that the library path of ``MultiNest`` should be set
        to the environmental variable ``LD_LIBRARY_PATH`` on a
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
        output : str
            Path that is used for the output files from MultiNest.

        Returns
        -------
        NoneType
            None
        """

        print_section("Nested sampling with MultiNest")

        print(f"Database tag: {tag}")
        print(f"Number of live points: {n_live_points}")
        print(f"Output folder: {output}")
        print()

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        # Create a dictionary with the cube indices of the parameters

        cube_index = {}
        for i, item in enumerate(self.model_par):
            cube_index[item] = i

        @typechecked
        def ln_prior(cube, n_dim, n_param) -> None:
            """
            Function to transform the unit cube into the parameter
            cube. It is not clear how to pass additional arguments
            to the function, therefore it is placed here.

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

            for i, key in enumerate(self.model_par):
                if key != "age":
                    obj_idx = int(key.split("_")[-1])

                if key == "age" and self.age_prior is not None:
                    # Asymmetric and truncated normal age prior

                    if cube[cube_index["age"]] < 0.5:
                        # Use lower errorbar on the age
                        # Truncated normal prior

                        # The truncation values are given in number of
                        # standard deviations relative to the mean
                        # of the normal distribution

                        if "age" in self.bounds:
                            a_trunc = (
                                self.bounds["age"][0] - self.age_prior[0]
                            ) / np.abs(self.age_prior[1])

                            b_trunc = (
                                self.bounds["age"][1] - self.age_prior[0]
                            ) / np.abs(self.age_prior[1])

                        else:
                            a_trunc = -self.age_prior[0] / np.abs(self.age_prior[1])
                            b_trunc = np.inf

                        cube[cube_index["age"]] = truncnorm.ppf(
                            cube[cube_index["age"]],
                            a_trunc,
                            b_trunc,
                            loc=self.age_prior[0],
                            scale=np.abs(self.age_prior[1]),
                        )

                    else:
                        # Use upper errorbar on the age

                        if "age" in self.bounds:
                            # Truncated normal prior

                            a_trunc = (
                                self.bounds["age"][0] - self.age_prior[0]
                            ) / self.age_prior[2]

                            b_trunc = (
                                self.bounds["age"][1] - self.age_prior[0]
                            ) / self.age_prior[2]

                            cube[cube_index["age"]] = truncnorm.ppf(
                                cube[cube_index["age"]],
                                a_trunc,
                                b_trunc,
                                loc=self.age_prior[0],
                                scale=self.age_prior[2],
                            )

                        else:
                            # Regular normal prior
                            cube[cube_index["age"]] = norm.ppf(
                                cube[cube_index["age"]],
                                loc=self.age_prior[0],
                                scale=self.age_prior[2],
                            )

                elif key[:4] == "mass" and self.mass_prior[obj_idx] is not None:
                    # Normal mass prior
                    sigma = self.mass_prior[obj_idx][1]

                    if f"inflate_mass{obj_idx}" in self.bounds:
                        sigma += (
                            self.bounds[f"inflate_mass{obj_idx}"][0]
                            + (
                                self.bounds[f"inflate_mass{obj_idx}"][1]
                                - self.bounds[f"inflate_mass{obj_idx}"][0]
                            )
                            * cube[cube_index[f"inflate_mass{obj_idx}"]]
                        )

                    cube[cube_index[f"mass_{obj_idx}"]] = norm.ppf(
                        cube[cube_index[f"mass_{obj_idx}"]],
                        loc=self.mass_prior[obj_idx][0],
                        scale=sigma,
                    )

                else:
                    cube[i] = (
                        self.bounds[key][0]
                        + (self.bounds[key][1] - self.bounds[key][0]) * cube[i]
                    )

        @typechecked
        def ln_like(params, n_dim, n_param) -> Union[float, np.float64]:
            """
            Function for return the log-likelihood for
            the sampled parameter cube.

            Parameters
            ----------
            params : pymultinest.run.LP_c_double
                Cube with physical parameters.
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

            chi_square = 0.0

            for planet_idx in range(self.n_planets):
                # param_names = [
                #     "age",
                #     f"mass_{planet_idx}",
                # ]

                # if f"s_init_{planet_idx}" in self.model_par:
                #     param_names.append(f"s_init_{planet_idx}")

                # param_val = []
                #
                # for param_item in param_names:
                #     if param_item in self.fix_param:
                #         param_val.append(self.fix_param[param_item])
                #
                #     else:
                #         param_val.append(params[cube_index[param_item]])

                if "age" in self.fix_param:
                    age_param = self.fix_param["age"]
                else:
                    age_param = params[cube_index["age"]]

                if f"inflate_lbol{planet_idx}" in self.bounds:
                    lbol_var = (
                        self.log_lum[planet_idx][1]
                        + params[cube_index[f"inflate_lbol{planet_idx}"]]
                    ) ** 2.0
                else:
                    lbol_var = self.log_lum[planet_idx][1] ** 2

                param_interp = ["log_lum"]
                if self.radius_prior[planet_idx] is not None:
                    param_interp.append("radius")

                if f"s_init_{planet_idx}" in self.model_par:
                    s_init = params[cube_index[f"s_init_{planet_idx}"]]
                else:
                    s_init = None

                iso_box = self.read_iso.get_isochrone(
                    age=age_param,
                    s_init=s_init,
                    masses=np.array([params[cube_index[f"mass_{planet_idx}"]]]),
                    filters_color=None,
                    filter_mag=None,
                    param_interp=param_interp,
                )

                chi_square += (
                    self.log_lum[planet_idx][0] - iso_box.log_lum[0]
                ) ** 2 / lbol_var

                # Only required when fitting the
                # inflation on the Lbol variance
                chi_square += np.log(2.0 * np.pi * lbol_var)

                if self.mass_prior[planet_idx] is not None:
                    # Only required when fitting the
                    # inflation on the mass variance
                    if f"inflate_mass{planet_idx}" in self.bounds:
                        mass_var = (
                            self.mass_prior[planet_idx][1]
                            + params[cube_index[f"inflate_mass{planet_idx}"]]
                        ) ** 2.0
                    else:
                        mass_var = self.mass_prior[planet_idx][1] ** 2

                    chi_square += np.log(2.0 * np.pi * mass_var)

                # Radius prior
                if self.radius_prior[planet_idx] is not None:
                    chi_square += (
                        self.radius_prior[planet_idx][0] - iso_box.radius[0]
                    ) ** 2 / self.radius_prior[planet_idx][1] ** 2

                # ln_like += -0.5 * weight * (obj_item[0] - phot_flux) ** 2 / phot_var
                # ln_like += -0.5 * weight * np.log(2.0 * np.pi * phot_var)

            if not np.isfinite(chi_square):
                log_like = -np.inf

            else:
                log_like = -0.5 * chi_square

            return log_like

        import pymultinest

        pymultinest.run(
            ln_like,
            ln_prior,
            len(self.model_par),
            outputfiles_basename=output,
            resume=False,
            n_live_points=n_live_points,
        )

        # Create the Analyzer object
        analyzer = pymultinest.analyse.Analyzer(
            len(self.model_par),
            outputfiles_basename=output,
            verbose=False,
        )

        # Get a dictionary with the ln(Z) and its errors, the
        # individual modes and their parameters quantiles of
        # the parameter posteriors
        sampling_stats = analyzer.get_stats()

        # Nested sampling global log-evidence
        ln_z = sampling_stats["nested sampling global log-evidence"]
        ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"\nNested sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

        # Nested importance sampling global log-evidence
        ln_z = sampling_stats["nested importance sampling global log-evidence"]
        ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            f"Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}"
        )

        # Get the maximum likelihood sample

        best_params = analyzer.get_best_fit()
        max_lnlike = best_params["log_likelihood"]

        print("\nSample with the maximum likelihood:")
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for param_idx, param_item in enumerate(best_params["parameters"]):
            if -0.1 < param_item < 0.1:
                print(f"   - {self.model_par[param_idx]} = {param_item:.2e}")
            else:
                print(f"   - {self.model_par[param_idx]} = {param_item:.2f}")

        # Get the posterior samples

        post_samples = analyzer.get_equal_weighted_posterior()

        # Samples and ln(L)

        ln_prob = post_samples[:, -1]
        samples = post_samples[:, :-1]

        # Adding the fixed parameters to the samples

        if self.fix_param:
            samples_tmp = np.copy(samples)
            self.model_par = ["age"]

            for planet_idx in range(self.n_planets):
                self.model_par.append(f"mass_{planet_idx}")

            for planet_idx in range(self.n_planets):
                if f"inflate_lbol{planet_idx}" in self.bounds:
                    self.model_par.append(f"inflate_lbol{planet_idx}")

            for planet_idx in range(self.n_planets):
                if f"inflate_mass{planet_idx}" in self.bounds:
                    self.model_par.append(f"inflate_mass{planet_idx}")

            samples = np.zeros((samples.shape[0], len(self.model_par)))

            for param_idx, param_item in enumerate(self.model_par):
                if param_item in self.fix_param:
                    samples[:, param_idx] = np.full(
                        samples_tmp.shape[0], self.fix_param[param_item]
                    )
                else:
                    samples[:, param_idx] = samples_tmp[:, cube_index[param_item]]

        # Recreate cube_index dictionary because the fix_param
        # parameters have been included in the samples array

        cube_index = {}
        for param_idx, param_item in enumerate(self.model_par):
            cube_index[param_item] = param_idx

        # Add atmospheric parameters: R, Teff, and log(g)

        print("\nExtracting the posteriors of Teff, R, and log(g):")

        radius = np.zeros((samples.shape[0], self.n_planets))
        log_g = np.zeros((samples.shape[0], self.n_planets))
        t_eff = np.zeros((samples.shape[0], self.n_planets))

        for planet_idx in tqdm(range(self.n_planets)):
            for sample_idx in tqdm(range(samples.shape[0]), leave=False):
                age = samples[sample_idx, cube_index["age"]]
                mass = samples[sample_idx, cube_index[f"mass_{planet_idx}"]]

                if f"s_init_{planet_idx}" in cube_index:
                    s_init = samples[sample_idx, cube_index[f"s_init_{planet_idx}"]]
                else:
                    s_init = None

                iso_box = self.read_iso.get_isochrone(
                    age=age,
                    s_init=s_init,
                    masses=np.array([mass]),
                    filters_color=None,
                    filter_mag=None,
                    param_interp=["log_lum", "teff", "logg", "radius"],
                )

                if iso_box.radius is not None:
                    radius[sample_idx, planet_idx] = iso_box.radius[0]

                if iso_box.logg is not None:
                    log_g[sample_idx, planet_idx] = iso_box.logg[0]

                if iso_box.teff is not None:
                    t_eff[sample_idx, planet_idx] = iso_box.teff[0]

        for planet_idx in range(self.n_planets):
            self.model_par.append(f"teff_{planet_idx}")

        for planet_idx in range(self.n_planets):
            self.model_par.append(f"radius_{planet_idx}")

        for planet_idx in range(self.n_planets):
            self.model_par.append(f"logg_{planet_idx}")

        samples = np.hstack((samples, t_eff, radius, log_g))

        # Recreate cube_index dictionary because of
        # derived parameters that were included

        cube_index = {}
        for param_idx, param_item in enumerate(self.model_par):
            cube_index[param_item] = param_idx

        # Remove outliers

        # percent = np.percentile(samples, (1.0, 99.0), axis=0)
        #
        # for i, item in enumerate(percent[0]):
        #     if i == 0:
        #         indices = samples[:, i] < item
        #     else:
        #         indices += samples[:, i] < item
        #
        # samples = samples[~indices, :]
        #
        # for i, item in enumerate(percent[1]):
        #     if i == 0:
        #         indices = samples[:, i] > item
        #     else:
        #         indices += samples[:, i] > item
        #
        # samples = samples[~indices, :]

        # Apply uncertainty inflation

        for planet_idx in range(self.n_planets):
            if f"inflate_lbol{planet_idx}" in self.bounds:
                # sigma_add = np.median(samples[:, cube_index[f"inflate_lbol{planet_idx}"]])
                index_prob = np.argmax(ln_prob)
                sigma_add = samples[index_prob, cube_index[f"inflate_lbol{planet_idx}"]]

                self.log_lum[planet_idx] = (
                    self.log_lum[planet_idx][0],
                    self.log_lum[planet_idx][1] + sigma_add,
                )

        for planet_idx in range(self.n_planets):
            if f"inflate_mass{planet_idx}" in self.bounds:
                # sigma_add = np.median(samples[:, cube_index[f"inflate_lbol{planet_idx}"]])
                index_prob = np.argmax(ln_prob)
                sigma_add = samples[index_prob, cube_index[f"inflate_mass{planet_idx}"]]

                self.mass_prior[planet_idx] = (
                    self.mass_prior[planet_idx][0],
                    self.mass_prior[planet_idx][1] + sigma_add,
                )

        # Set radius_prior to posterior value if argument was None

        # for planet_idx, planet_item in enumerate(self.radius_prior):
        #     if planet_item is None:
        #         radius_samples = samples[:, cube_index[f"radius_{planet_idx}"]]
        #         self.radius_prior[planet_idx] = (
        #             np.mean(radius_samples),
        #             np.std(radius_samples),
        #         )

        # Adjust mass_prior to posterior value

        # for i, item in enumerate(self.mass_prior):
        #     mass_samples = samples[:, cube_index[f"mass_{i}"]]
        #     self.mass_prior[i] = (np.mean(mass_samples), np.std(mass_samples))

        # Adjust radius_prior to posterior value

        # for i, item in enumerate(self.radius_prior):
        #     radius_samples = samples[:, cube_index[f"radius_{i}"]]
        #     self.radius_prior[i] = (np.mean(radius_samples), np.std(radius_samples))

        # Set age_prior to NaN if no prior was provided

        if self.age_prior is None:
            self.age_prior = [np.nan]

        elif "age" in self.fix_param:
            self.age_prior = (self.fix_param["age"], 0.0)

        # Set mass_prior to NaN if no prior was provided

        for planet_idx, planet_item in enumerate(self.mass_prior):
            if f"mass_{planet_idx}" in self.fix_param:
                self.mass_prior[planet_idx] = (
                    self.fix_param[f"mass_{planet_idx}"],
                    0.0,
                )

            elif planet_item is None:
                self.mass_prior[planet_idx] = np.nan

        # Set radius_prior to NaN if no prior was provided

        for planet_idx, planet_item in enumerate(self.radius_prior):
            if f"radius_{planet_idx}" in self.fix_param:
                self.radius_prior[planet_idx] = (
                    self.fix_param[f"radius_{planet_idx}"],
                    0.0,
                )

            elif planet_item is None:
                self.radius_prior[planet_idx] = np.nan

        # Dictionary with attributes that will be stored

        attr_dict = {
            "model_type": "evolution",
            "model_name": self.evolution_model,
            "ln_evidence": (ln_z, ln_z_error),
            "n_planets": self.n_planets,
            "log_lum": self.log_lum,
            "age_prior": self.age_prior,
            "mass_prior": self.mass_prior,
            "radius_prior": self.radius_prior,
            "regular_grid": self.read_iso.regular_grid,
        }

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
                modelpar=self.model_par,
                bounds=self.bounds,
                normal_prior=self.normal_prior,
                fixed_param=self.fix_param,
                attr_dict=attr_dict,
            )
