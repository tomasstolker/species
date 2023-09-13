"""
Module with functionalities for retrieving the age and bulk parameters
of one or multiple planets and/or brown dwarfs in a system.
"""

import configparser
import os
import warnings

from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

try:
    import pymultinest
except:
    warnings.warn(
        "PyMultiNest could not be imported. "
        "Perhaps because MultiNest was not build "
        "and/or found at the LD_LIBRARY_PATH "
        "(Linux) or DYLD_LIBRARY_PATH (Mac)?"
    )

from scipy import stats
from tqdm.auto import tqdm
from typeguard import typechecked

from species.core import constants
from species.data import database
from species.read import read_isochrone


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
        object_lbol: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]],
        object_mass: Optional[
            Union[Tuple[float, float], List[Optional[Tuple[float, float]]]]
        ] = None,
        object_radius: Optional[
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
        object_lbol : tuple(float, float), list(tuple(float, float))
            List with tuples that contain :math:`\\log10{L/L_\\odot}`
            and the related uncertainty for one or multiple objects.
            The list should follow the alphabetical order of companion
            characters (i.e. b, c, d, etc.) to make sure that the
            labels are correctly shown when plotting results.
        object_mass : tuple(float, float), list(tuple(float, float)), None
            Optional list with tuples that contain the (dynamical)
            masses and the related uncertainty for one or multiple
            objects. These masses we be used as Gaussian prior with
            the fit. The order should be identical to ``object_lbol``.
        object_radius : tuple(float, float), list(tuple(float, float)), None
            Optional list with tuples that contain the radii (e.g.
            from and SED fit) and the related uncertainty for one
            or multiple objects. These radii we be used as Gaussian
            prior with the fit. The order should be identical to
            ``object_lbol``.
        bounds : dict(str, tuple(float, float)), None
            The boundaries that are used for the uniform or
            log-uniform priors. Fixing a parameter is possible by
            providing the same value as lower and upper boundary
            of the parameter (e.g. ``bounds={'y_frac': (0.25, 0.25)``.

        Returns
        -------
        NoneType
            None
        """

        self.evolution_model = evolution_model
        self.object_lbol = object_lbol
        self.object_mass = object_mass
        self.object_radius = object_radius
        self.bounds = bounds
        self.fix_param = {}

        if isinstance(self.object_lbol, tuple):
            self.object_lbol = [self.object_lbol]
            self.object_mass = [self.object_mass]
            self.n_planets = 1

        if isinstance(self.object_lbol, list):
            self.n_planets = len(self.object_lbol)

        else:
            self.n_planets = 1

        if self.object_mass is None:
            self.object_mass = []
            for i in range(self.n_planets):
                self.object_mass.append(None)

        if self.object_radius is None:
            self.object_radius = []
            for i in range(self.n_planets):
                self.object_radius.append(None)

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.database_path = config["species"]["database"]
        self.interp_method = config["species"]["interp_method"]

        # Add grid with evolution data

        with h5py.File(self.database_path, "r") as h5_file:
            found_model = bool(f"isochrones/{evolution_model}" in h5_file)
            tag_list = list(h5_file["isochrones"])

        if not found_model:
            raise ValueError(
                f"The isochrones of '{evolution_model}' "
                "are not found in the database. Please "
                "add the isochrones by using the "
                "add_isochrones method of Database. The "
                "following isochrone data are found in "
                f"the database: {tag_list}"
            )

        # Model parameters

        self.model_par = ["age"]

        for i in range(self.n_planets):
            self.model_par.append(f"mass_{i}")

    @typechecked
    def run_multinest(
        self,
        tag: str,
        n_live_points: int = 1000,
        output: str = "multinest/",
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

        Returns
        -------
        NoneType
            None
        """

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()

        except ModuleNotFoundError:
            mpi_rank = 0

        # Create the output folder if required

        if mpi_rank == 0 and not os.path.exists(output):
            os.mkdir(output)

        read_iso = read_isochrone.ReadIsochrone(
            tag=self.evolution_model,
            create_regular_grid=False,
        )

        # Prior boundaries

        if self.bounds is not None:
            bounds_grid = {}
            for key, value in read_iso.grid_points().items():
                if key in ["age", "mass"]:
                    bounds_grid[key] = (value[0], value[-1])

            for key, value in bounds_grid.items():
                if key not in self.bounds:
                    # Set the parameter boundaries to the grid
                    # boundaries if set to None or not found
                    if key == "age":
                        self.bounds[key] = bounds_grid[key]

                    else:
                        for i in range(self.n_planets):
                            if f"{key}_{i}" not in self.bounds:
                                self.bounds[f"{key}_{i}"] = bounds_grid[key]

                else:
                    if self.bounds[key][0] < bounds_grid[key][0]:
                        warnings.warn(
                            f"The lower bound on {key} "
                            f"({self.bounds[key][0]}) is smaller than "
                            f"the lower bound from the available "
                            f"evolution model grid "
                            f"({bounds_grid[key][0]}). The lower bound "
                            f"of the {key} prior will be adjusted to "
                            f"{bounds_grid[key][0]}."
                        )

                        if key == "age":
                            self.bounds[key] = (
                                bounds_grid[key][0],
                                self.bounds[key][1],
                            )

                        else:
                            for i in range(self.n_planets):
                                self.bounds[f"{key}_{i}"] = (
                                    bounds_grid[key][0],
                                    self.bounds[key][1],
                                )

                    if self.bounds[key][1] > bounds_grid[key][1]:
                        warnings.warn(
                            f"The upper bound on {key} "
                            f"({self.bounds[key][1]}) is larger than the "
                            f"upper bound from the available evolution "
                            f"model grid ({bounds_grid[key][1]}). The "
                            f"bound of the {key} prior will be adjusted "
                            f"to {bounds_grid[key][1]}."
                        )

                        if key == "age":
                            self.bounds[key] = (
                                self.bounds[key][0],
                                bounds_grid[key][1],
                            )

                        else:
                            for i in range(self.n_planets):
                                if f"{key}_{i}" in self.bounds:
                                    self.bounds[f"{key}_{i}"] = (
                                        self.bounds[f"{key}_{i}"][0],
                                        bounds_grid[key][1],
                                    )

                                else:
                                    self.bounds[f"{key}_{i}"] = (
                                        self.bounds[key][0],
                                        bounds_grid[key][1],
                                    )

                    if key != "age":
                        for i in range(self.n_planets):
                            self.bounds[f"{key}_{i}"] = self.bounds[key]

                        del self.bounds[key]

            for i in range(self.n_planets):
                if f"inflate_lbol{i}" in self.bounds:
                    self.model_par.append(f"inflate_lbol{i}")

            for i in range(self.n_planets):
                if f"inflate_mass{i}" in self.bounds:
                    if self.object_mass[i] is None:
                        warnings.warn(
                            f"The object_mass with index "
                            f"{i} is set to None so the "
                            f"inflate_mass{i} parameter "
                            f"will be excluded."
                        )

                        del self.bounds[f"inflate_mass{i}"]

                    else:
                        self.model_par.append(f"inflate_mass{i}")

        else:
            # Set all parameter boundaries to the grid boundaries
            self.bounds = {}
            for key, value in read_iso.grid_points().items():
                if key == "age":
                    self.bounds[key] = (value[0], value[-1])

                elif key == "mass":
                    for i in range(self.n_planets):
                        self.bounds[f"{key}_{i}"] = (value[0], value[-1])

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
            print(f"Fixing {len(self.fix_param)} parameters:")

            for key, value in self.fix_param.items():
                print(f"   - {key} = {value}")

        print(f"Fitting {len(self.model_par)} parameters:")

        for item in self.model_par:
            print(f"   - {item}")

        print("Prior boundaries:")

        for key, value in self.bounds.items():
            print(f"   - {key} = {value}")

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

                if key[:4] == "mass" and self.object_mass[obj_idx] is not None:
                    # Gaussian mass prior
                    sigma = self.object_mass[obj_idx][1]

                    if f"inflate_mass{obj_idx}" in self.bounds:
                        sigma += (
                            self.bounds[f"inflate_mass{obj_idx}"][0]
                            + (
                                self.bounds[f"inflate_mass{obj_idx}"][1]
                                - self.bounds[f"inflate_mass{obj_idx}"][0]
                            )
                            * cube[cube_index[f"inflate_mass{obj_idx}"]]
                        )

                    cube[cube_index[f"mass_{obj_idx}"]] = stats.norm.ppf(
                        cube[cube_index[f"mass_{obj_idx}"]],
                        loc=self.object_mass[obj_idx][0],
                        scale=sigma,
                    )

                else:
                    cube[i] = (
                        self.bounds[key][0]
                        + (self.bounds[key][1] - self.bounds[key][0]) * cube[i]
                    )

        @typechecked
        def ln_like(params, n_dim, n_param) -> np.float64:
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

            for i in range(self.n_planets):
                param_names = [
                    "age",
                    f"mass_{i}",
                ]

                param_val = []

                for item in param_names:
                    if item in self.fix_param:
                        param_val.append(self.fix_param[item])

                    else:
                        param_val.append(params[cube_index[item]])

                if f"inflate_lbol{i}" in self.bounds:
                    lbol_var = (
                        self.object_lbol[i][1] + params[cube_index[f"inflate_lbol{i}"]]
                    ) ** 2.0
                else:
                    lbol_var = self.object_lbol[i][1] ** 2

                iso_box = read_iso.get_isochrone(
                    age=params[cube_index["age"]],
                    masses=np.array([params[cube_index[f"mass_{i}"]]]),
                    filters_color=None,
                    filter_mag=None,
                )

                chi_square += (
                    self.object_lbol[i][0] - iso_box.log_lum[0]
                ) ** 2 / lbol_var

                # Only required when fitting the
                # inflation on the Lbol variance
                chi_square += np.log(2.0 * np.pi * lbol_var)

                if self.object_mass[i] is not None:
                    # Only required when fitting the
                    # inflation on the mass variance
                    if f"inflate_mass{i}" in self.bounds:
                        mass_var = (
                            self.object_mass[i][1]
                            + params[cube_index[f"inflate_mass{i}"]]
                        ) ** 2.0
                    else:
                        mass_var = self.object_mass[i][1] ** 2

                    chi_square += np.log(2.0 * np.pi * mass_var)

                # Radius prior
                if self.object_radius[i] is not None:
                    chi_square += (
                        self.object_radius[i][0] - iso_box.radius[0]
                    ) ** 2 / self.object_radius[i][1] ** 2

                # ln_like += -0.5 * weight * (obj_item[0] - phot_flux) ** 2 / phot_var
                # ln_like += -0.5 * weight * np.log(2.0 * np.pi * phot_var)

            if np.isnan(chi_square):
                log_like = -np.inf

            elif np.isinf(chi_square):
                log_like = -np.inf

            else:
                log_like = -0.5 * chi_square

            return log_like

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
            len(self.model_par), outputfiles_basename=output
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
        ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            f"Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}"
        )

        # Get the best-fit (highest likelihood) point
        print("Sample with the highest likelihood:")
        best_params = analyzer.get_best_fit()

        max_lnlike = best_params["log_likelihood"]
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(best_params["parameters"]):
            print(f"   - {self.model_par[i]} = {item:.2f}")

        # Get the posterior samples
        samples = analyzer.get_equal_weighted_posterior()

        analyzer = pymultinest.analyse.Analyzer(
            len(self.model_par), outputfiles_basename=output
        )

        sampling_stats = analyzer.get_stats()

        ln_z = sampling_stats["nested sampling global log-evidence"]
        ln_z_error = sampling_stats["nested sampling global log-evidence error"]
        print(f"Nested sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

        ln_z = sampling_stats["nested importance sampling global log-evidence"]
        ln_z_error = sampling_stats[
            "nested importance sampling global log-evidence error"
        ]
        print(
            f"Nested importance sampling global log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}"
        )

        print("Sample with the highest likelihood:")
        best_params = analyzer.get_best_fit()

        max_lnlike = best_params["log_likelihood"]
        print(f"   - Log-likelihood = {max_lnlike:.2f}")

        for i, item in enumerate(best_params["parameters"]):
            print(f"   - {self.model_par[i]} = {item:.2f}")

        samples = analyzer.get_equal_weighted_posterior()

        ln_prob = samples[:, -1]

        # Adding the fixed parameters to the samples

        if self.fix_param:
            samples_tmp = samples[:, :-1]
            self.model_par = ["age"]

            for i in range(self.n_planets):
                self.model_par.append(f"mass_{i}")

            for i in range(self.n_planets):
                if f"inflate_lbol{i}" in self.bounds:
                    self.model_par.append(f"inflate_lbol{i}")

            for i in range(self.n_planets):
                if f"inflate_mass{i}" in self.bounds:
                    self.model_par.append(f"inflate_mass{i}")

            samples = np.zeros((samples_tmp.shape[0], len(self.model_par)))

            for i, key in enumerate(self.model_par):
                if key in self.fix_param:
                    samples[:, i] = np.full(samples_tmp.shape[0], self.fix_param[key])
                else:
                    samples[:, i] = samples_tmp[:, cube_index[key]]

        else:
            samples = samples[:, :-1]

        # Recreate cube_index dictionary because of included fix_param

        cube_index = {}
        for i, item in enumerate(self.model_par):
            cube_index[item] = i

        # Add atmospheric parameters (R, Teff, and log(g))

        print("Extracting the posteriors of Teff, R, and log(g)...")

        radius = np.zeros((samples.shape[0], self.n_planets))
        log_g = np.zeros((samples.shape[0], self.n_planets))
        t_eff = np.zeros((samples.shape[0], self.n_planets))

        for j in tqdm(range(self.n_planets)):
            for i in tqdm(range(samples.shape[0]), leave=False):
                age = samples[i, cube_index["age"]]
                mass = samples[i, cube_index[f"mass_{j}"]]

                iso_box = read_iso.get_isochrone(
                    age=age,
                    masses=np.array([mass]),
                    filters_color=None,
                    filter_mag=None,
                )

                radius[i, j] = iso_box.radius[0]
                log_g[i, j] = iso_box.logg[0]

                l_bol = 10.0 ** iso_box.log_lum[0] * constants.L_SUN

                t_eff[i, j] = (
                    l_bol
                    / (
                        4.0
                        * np.pi
                        * (radius[i, j] * constants.R_JUP) ** 2
                        * constants.SIGMA_SB
                    )
                ) ** 0.25

        for i in range(self.n_planets):
            self.model_par.append(f"teff_evol_{i}")

        for i in range(self.n_planets):
            self.model_par.append(f"radius_evol_{i}")

        for i in range(self.n_planets):
            self.model_par.append(f"logg_evol_{i}")

        samples = np.hstack((samples, t_eff, radius, log_g))

        # Recreate cube_index dictionary because of
        # derived parameters that were included

        cube_index = {}
        for i, item in enumerate(self.model_par):
            cube_index[item] = i

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

        for i in range(self.n_planets):
            if f"inflate_lbol{i}" in self.bounds:
                # sigma_add = np.median(samples[:, cube_index[f"inflate_lbol{i}"]])
                index_prob = np.argmax(ln_prob)
                sigma_add = samples[index_prob, cube_index[f"inflate_lbol{i}"]]

                self.object_lbol[i] = (
                    self.object_lbol[i][0],
                    self.object_lbol[i][1] + sigma_add,
                )

        for i in range(self.n_planets):
            if f"inflate_mass{i}" in self.bounds:
                # sigma_add = np.median(samples[:, cube_index[f"inflate_lbol{i}"]])
                index_prob = np.argmax(ln_prob)
                sigma_add = samples[index_prob, cube_index[f"inflate_mass{i}"]]

                self.object_mass[i] = (
                    self.object_mass[i][0],
                    self.object_mass[i][1] + sigma_add,
                )

        # Set object_radius to posterior value if argument was None

        for i, item in enumerate(self.object_radius):
            if item is None:
                radius_samples = samples[:, cube_index[f"radius_evol_{i}"]]
                self.object_radius[i] = (
                    np.mean(radius_samples),
                    np.std(radius_samples),
                )

        # Adjust object_mass to posterior value

        # for i, item in enumerate(self.object_mass):
        #     mass_samples = samples[:, cube_index[f"mass_{i}"]]
        #     self.object_mass[i] = (np.mean(mass_samples), np.std(mass_samples))

        # Adjust object_radius to posterior value

        # for i, item in enumerate(self.object_radius):
        #     radius_samples = samples[:, cube_index[f"radius_evol_{i}"]]
        #     self.object_radius[i] = (np.mean(radius_samples), np.std(radius_samples))

        # Set object_mass and object_radius to NaN if no prior was provided

        for i, item in enumerate(self.object_mass):
            if f"mass_{i}" in self.fix_param:
                self.object_mass[i] = (self.fix_param[f"mass_{i}"], 0.0)
            elif item is None:
                self.object_mass[i] = np.nan

        for i, item in enumerate(self.object_radius):
            if item is None:
                self.object_radius[i] = np.nan

        # Dictionary with attributes that will be stored

        attr_dict = {
            "spec_type": "model",
            "spec_name": "evolution",
            "evolution_model": self.evolution_model,
            "ln_evidence": (ln_z, ln_z_error),
            "n_planets": self.n_planets,
            "object_lbol": self.object_lbol,
            "object_mass": self.object_mass,
            "object_radius": self.object_radius,
        }

        # Add samples to the database

        if mpi_rank == 0:
            # Writing the samples to the database is only
            # possible when using a single process
            species_db = database.Database()

            species_db.add_samples(
                sampler="multinest",
                samples=samples,
                ln_prob=ln_prob,
                tag=tag,
                modelpar=self.model_par,
                attr_dict=attr_dict,
            )
