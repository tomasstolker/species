"""
Module with reading functionalities for isochrones and cooling curves.
"""

import os
import warnings

from configparser import ConfigParser
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
from typeguard import typechecked

from species.core.box import (
    ColorMagBox,
    ColorColorBox,
    CoolingBox,
    IsochroneBox,
    ModelBox,
    PhotometryBox,
    create_box,
)
from species.read.read_model import ReadModel
from species.util.convert_util import apparent_to_absolute
from species.util.core_util import print_section
from species.util.plot_util import update_labels


class ReadIsochrone:
    """
    Class for reading isochrone data from the database. This class
    interpolates the evolutionary tracks. Please carefully check
    for interpolation effects since some grids show non-linear
    variations between grid points. Setting ``masses=None`` in
    :func:`~species.read.read_isochrone.ReadIsochrone.get_isochrone`
    extracts the isochrones at the masses of the original
    grid, so using that option helps with comparing results
    for which the masses have been interpolated. Similarly, by
    setting ``ages=None`` with the :func:`~species.read.
    read_isochrone.ReadIsochrone.get_cooling_track` method
    will set the ages to those of the original grid.
    """

    @typechecked
    def __init__(
        self,
        tag: Optional[str] = None,
        create_regular_grid: bool = False,
        verbose: bool = True,
        interp_method: str = "linear",
    ) -> None:
        """
        Parameters
        ----------
        tag : str, None
            Database tag of the isochrone data (e.g. 'ames-cond',
            'sonora+0.0', 'atmo-ceq'). A list with the isochrone
            data that are stored in the current database is
            printed if the argument of ``tag`` is set to ``None``.
        create_regular_grid : bool
            Evolutionary grids can be irregular in the (age, mass)
            space. By setting ``create_regular_grid=True``, the
            grid will be interpolated and extrapolated onto
            a regular grid that is based on all the unique age
            and mass values of the original grid. The
            resampling of the evolutionary parameters
            (i.e. :math:`T_\\mathrm{eff}` (K),
            :math:`\\log{(L/L_\\odot)}`, :math:`\\log{(g)}`, and
            :math:`R`) is done as function of mass, separately
            for each age. Setting ``create_regular_grid=True``
            can be helpful if some values are missing at the edge
            of the grid. However, please carefully check results
            since there might be inaccuracies in the extrapolated
            parts of the parameter space, in particular for the
            cooling tracks extracted with :func:`~species.read.
            read_isochrone.ReadIsochrone.get_cooling_track`, so
            because ``get_cooling_track`` is an experimental
            parameter.
        verbose : bool
            Print output information.
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

        self.tag = tag
        self.create_regular_grid = create_regular_grid
        self.verbose = verbose
        self.regular_grid = None

        if self.verbose:
            print_section("Read isochrone grid")
            print(f"Database tag: {self.tag}")
            print(f"Create regular grid: {self.create_regular_grid}")

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]

        with h5py.File(self.database, "r") as hdf5_file:
            if "isochrones" not in hdf5_file:
                raise ValueError(
                    "There are no isochrone data stored in the "
                    "database. Please use the add_isochrones "
                    "method of Database to add a grid of "
                    "isochrones."
                )

        if self.tag is None:
            with h5py.File(self.database, "r") as hdf5_file:
                tag_list = list(hdf5_file["isochrones"])

            self.tag = input(
                "Please select one of the following "
                "isochrone tags that are stored in "
                "the database or use 'add_isochrones' "
                "to add another model to the database:"
                f"\n{tag_list}:\n"
            )

        with h5py.File(self.database, "r") as hdf5_file:
            if f"isochrones/{self.tag}" not in hdf5_file:
                tag_list = list(hdf5_file["isochrones"])

                raise ValueError(
                    f"There are no isochrone data stored with the "
                    f"selected tag '{tag}'. The following isochrone "
                    f"tags are found in the database: {tag_list}"
                )

        # Connect isochrone model with atmosphere model
        # key = isochrone model, value = atmosphere model, extra_param
        self.match_model = {
            "ames-cond": ("ames-cond", None),
            "ames-dusty": ("ames-dusty", None),
            "atmo-ceq": ("atmo-ceq", None),
            "atmo-neq-strong": ("atmo-neq-strong", None),
            "atmo-neq-weak": ("atmo-neq-weak", None),
            "atmo-ceq-chabrier2023": ("atmo-ceq", None),
            "atmo-neq-strong-chabrier2023": ("atmo-neq-strong", None),
            "atmo-neq-weak-chabrier2023": ("atmo-neq-weak", None),
            "bt-settl": ("bt-settl", None),
            "linder2019-petitCODE-metal_-0.4": (
                "petitcode-linder2019-clear",
                {"feh": -0.4},
            ),
            "linder2019-petitCODE-metal_0.0": (
                "petitcode-linder2019-clear",
                {"feh": 0.0},
            ),
            "linder2019-petitCODE-metal_0.4": (
                "petitcode-linder2019-clear",
                {"feh": 0.4},
            ),
            "linder2019-petitCODE-metal_0.8": (
                "petitcode-linder2019-clear",
                {"feh": 0.8},
            ),
            "linder2019-petitCODE-metal_1.2": (
                "petitcode-linder2019-clear",
                {"feh": 1.2},
            ),
            "linder2019-petitCODE-metal_-0.4-fsed_1.0": (
                "petitcode-linder2019-cloudy",
                {"feh": -0.4},
            ),
            "linder2019-petitCODE-metal_0.0-fsed_1.0": (
                "petitcode-linder2019-cloudy",
                {"feh": 0.0},
            ),
            "linder2019-petitCODE-metal_0.4-fsed_1.0": (
                "petitcode-linder2019-cloudy",
                {"feh": 0.4},
            ),
            "linder2019-petitCODE-metal_0.8-fsed_1.0": (
                "petitcode-linder2019-cloudy",
                {"feh": 0.8},
            ),
            "linder2019-petitCODE-metal_1.2-fsed_1.0": (
                "petitcode-linder2019-cloudy",
                {"feh": 1.0},
            ),
            "saumon2008-nc_solar": ("saumon2008-clear", None),
            "saumon2008-f2_solar": ("saumon2008-cloudy", None),
            "sonora-0.5": ("sonora-bobcat", {"feh": -0.5}),
            "sonora+0.0": ("sonora-bobcat", {"feh": 0.0}),
            "sonora+0.5": ("sonora-bobcat", {"feh": 0.5}),
            "sonora-diamondback-hybrid-0.5": ("sonora-diamondback", {"feh": -0.5}),
            "sonora-diamondback-hybrid+0.0": ("sonora-diamondback", {"feh": 0.0}),
            "sonora-diamondback-hybrid+0.5": ("sonora-diamondback", {"feh": 0.5}),
            "sonora-diamondback-hybrid-grav-0.5": ("sonora-diamondback", {"feh": -0.5}),
            "sonora-diamondback-hybrid-grav+0.0": ("sonora-diamondback", {"feh": 0.0}),
            "sonora-diamondback-hybrid-grav+0.5": ("sonora-diamondback", {"feh": 0.5}),
            "sonora-diamondback-nc-0.5": ("sonora-bobcat", {"feh": -0.5}),
            "sonora-diamondback-nc+0.0": ("sonora-bobcat", {"feh": 0.0}),
            "sonora-diamondback-nc+0.5": ("sonora-bobcat", {"feh": 0.5}),
        }

        # Set attribute with extra parameters
        # The attribute will be overwritten by any of
        # the methods that have the extra_param parameter

        if self.tag in self.match_model:
            self.extra_param = self.match_model[self.tag][1]

            if self.verbose:
                print(f"\nSetting 'extra_param' attribute: {self.extra_param}")

        else:
            self.extra_param = None

        self.teff_interp = None
        self.loglum_interp = None
        self.logg_interp = None
        self.radius_interp = None
        self.mass_interp = None
        self.sinit_interp = None
        self.interp_method = interp_method

        if self.interp_method not in ["linear", "cubic"]:
            raise ValueError(
                "The argument of 'interp_method' should "
                "be set to 'linear' or 'cubic'."
            )

    @typechecked
    def read_data(
        self,
    ) -> Dict[str, np.ndarray]:
        """
        Internal function for reading the evolutionary
        data from the database.

        Returns
        -------
        dict
            Dictionary with arrays containing the age (Myr),
        np.ndarray
            Array with the mass (:math:`M_\\mathrm{J}`).
        np.ndarray
            Array with the :math:`T_\\mathrm{eff}` (K).
        np.ndarray
            Array with the :math:`\\log{(L/L_\\odot)}`.
        np.ndarray
            Array with the :math:`\\log{(g)}`.
        np.ndarray
            Array with the radius (:math:`R_\\mathrm{J}`).
        np.ndarray
            Optional array with the initial entropy
            (:math:`k_B/\\mathrm{baryon}`).
        np.ndarray, None
            Optional array with the absolute magnitudes. The
            array has two axes with the length of the second
            axis equal to the number of filters for which
            there are magnitudes available.
        """

        iso_data = {}
        iso_mag = None
        new_mag = None

        with h5py.File(self.database, "r") as hdf5_file:
            # if "model" in hdf5_file[f"isochrones/{self.tag}"].attrs:
            #     model_name = hdf5_file[f"isochrones/{self.tag}"].attrs["model"]
            # else:
            #     model_name = hdf5_file[f"isochrones/{self.tag}/age"].attrs["model"]

            if "regular_grid" in hdf5_file[f"isochrones/{self.tag}"].attrs:
                self.regular_grid = hdf5_file[f"isochrones/{self.tag}"].attrs[
                    "regular_grid"
                ]
            else:
                self.regular_grid = False

            iso_data["age"] = np.asarray(hdf5_file[f"isochrones/{self.tag}/age"])
            iso_data["mass"] = np.asarray(hdf5_file[f"isochrones/{self.tag}/mass"])
            iso_data["log_lum"] = np.asarray(
                hdf5_file[f"isochrones/{self.tag}/log_lum"]
            )

            if "teff" in hdf5_file[f"isochrones/{self.tag}"]:
                iso_data["teff"] = np.asarray(hdf5_file[f"isochrones/{self.tag}/teff"])

            if "log_g" in hdf5_file[f"isochrones/{self.tag}"]:
                iso_data["log_g"] = np.asarray(
                    hdf5_file[f"isochrones/{self.tag}/log_g"]
                )

            if "radius" in hdf5_file[f"isochrones/{self.tag}"]:
                iso_data["radius"] = np.asarray(
                    hdf5_file[f"isochrones/{self.tag}/radius"]
                )

            if "s_init" in hdf5_file[f"isochrones/{self.tag}"]:
                iso_data["s_init"] = np.asarray(
                    hdf5_file[f"isochrones/{self.tag}/s_init"]
                )

            if f"isochrones/{self.tag}/magnitudes" in hdf5_file:
                iso_data["mag"] = np.asarray(
                    hdf5_file[f"isochrones/{self.tag}/magnitudes"]
                )

        if self.create_regular_grid and not self.regular_grid:
            age_unique = np.unique(iso_data["age"])
            mass_unique = np.unique(iso_data["mass"])

            n_ages = age_unique.shape[0]
            n_masses = mass_unique.shape[0]

            new_age = np.zeros((n_ages * n_masses))
            new_mass = np.zeros((n_ages * n_masses))
            new_teff = np.zeros((n_ages * n_masses))
            new_loglum = np.zeros((n_ages * n_masses))
            new_logg = np.zeros((n_ages * n_masses))
            new_radius = np.zeros((n_ages * n_masses))

            if "mag" in iso_data:
                new_mag = np.zeros(((n_ages * n_masses, iso_data["mag"].shape[1])))

            for age_idx, age_item in enumerate(age_unique):
                age_select = iso_data["age"] == age_item
                ages_tmp = np.full(n_masses, age_item)

                new_age[age_idx * n_masses : (age_idx + 1) * n_masses] = ages_tmp
                new_mass[age_idx * n_masses : (age_idx + 1) * n_masses] = mass_unique

                interp_teff = interp1d(
                    iso_data["mass"][age_select],
                    iso_data["teff"][age_select],
                    fill_value="extrapolate",
                )

                new_teff[age_idx * n_masses : (age_idx + 1) * n_masses] = interp_teff(
                    mass_unique
                )

                interp_loglum = interp1d(
                    iso_data["mass"][age_select],
                    iso_data["log_lum"][age_select],
                    fill_value="extrapolate",
                )

                new_loglum[age_idx * n_masses : (age_idx + 1) * n_masses] = (
                    interp_loglum(mass_unique)
                )

                interp_logg = interp1d(
                    iso_data["mass"][age_select],
                    iso_data["log_g"][age_select],
                    fill_value="extrapolate",
                )

                new_logg[age_idx * n_masses : (age_idx + 1) * n_masses] = interp_logg(
                    mass_unique
                )

                interp_radius = interp1d(
                    iso_data["mass"][age_select],
                    iso_data["radius"][age_select],
                    fill_value="extrapolate",
                )

                new_radius[age_idx * n_masses : (age_idx + 1) * n_masses] = (
                    interp_radius(mass_unique)
                )

                if iso_data["mag"] is not None:
                    for mag_idx in range(iso_data["mag"].shape[1]):
                        interp_mag = interp1d(
                            iso_data["mass"][age_select],
                            iso_data["mag"][age_select, mag_idx],
                            fill_value="extrapolate",
                        )

                        new_mag[
                            age_idx * n_masses : (age_idx + 1) * n_masses, mag_idx
                        ] = interp_mag(mass_unique)

            iso_data["age"] = new_age.copy()
            iso_data["mass"] = new_mass.copy()
            iso_data["teff"] = new_teff.copy()
            iso_data["log_lum"] = new_loglum.copy()
            iso_data["log_g"] = new_logg.copy()
            iso_data["radius"] = new_radius.copy()

            if "mag" in iso_data:
                iso_data["mag"] = new_mag.copy()

        return iso_data

    @typechecked
    def _check_model(self, atmospheric_model: Optional[str]) -> str:
        """
        Function for matching the atmospheric model with
        the evolutionary model and checking if the expected
        atmospheric model is used.

        Parameters
        ----------
        atmospheric_model : str, None
            Name of the atmospheric model. By setting the argument to
            ``None``, the atmospheric model associated with the
            evolutionary model is automatically selected.

        Returns
        -------
        str
            Name of the atmospheric model.
        """

        if atmospheric_model is None:
            if self.tag in self.match_model:
                atmospheric_model = self.match_model[self.tag][0]
            else:
                raise ValueError(
                    "Can't find the atmosphere model "
                    f"associated with the '{self.tag}' "
                    "evolutionary model. Please contact "
                    "the code maintainer."
                )

        elif self.tag in self.match_model:
            if atmospheric_model != self.match_model[self.tag][0]:
                warnings.warn(
                    "Please note that you have selected "
                    f"'{atmospheric_model}' as "
                    f"atmospheric model for '{self.tag}' "
                    f"while '{self.match_model[self.tag][0]}'"
                    " is the atmospheric model that is "
                    f"self-consistently associated with "
                    f"'{self.tag}'. It is recommended "
                    "to set 'atmospheric_model=None' to "
                    "automatically select the correct "
                    "grid with model spectra."
                )

        return atmospheric_model

    @typechecked
    def _update_param(
        self,
        atmospheric_model: str,
        model_param: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
        extra_param: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Internal function for updating the dictionary with model
        parameters for the atmospheric model. Parameters that are
        not part of the evolutionary model but are required for the
        atmospheric model will be included by either adopting them
        from the ``extra_param`` dictionary or asking for manual
        input in case a parameter is missing. In the latter case,
        the ``extra_param`` dictionary will also be updated.

        Parameters
        ----------
        atmospheric_model : str
            Name of the atmospheric model.
        model_param : dict
            Dictionary with the parameters at which the atmospheric
            model spectra will be interpolated.
        param_bounds : dict
            Dictionary with the parameter boundaries of grid with
            atmospheric model spectra.
        extra_param : dict
            Dictionary with additional parameters that are optionally
            required for the atmospheric model but are not part of
            the evolutionary model grid. In case additional
            parameters are required for the atmospheric model but
            they are not provided in ``extra_param`` then a manual
            input will be requested when running the
            ``get_photometry`` method. Typically the ``extra_param``
            parameter is not needed so the argument can be set to
            ``None``. It will only be required if a non-self-consistent
            approach will be tested, that is, the calculation of
            synthetic photometry from an atmospheric model that is
            not associated with the evolutionary model.

        Returns
        -------
        dict
            Updated dictionary with parameters for the interpolation
            of atmospheric model grid.
        dict
            Updated dictionary with only the parameters that
            are required for the atmospheric model but not for
            the evolutionary model.
        """

        if len(extra_param) == 0 and self.extra_param is not None:
            extra_param = self.extra_param.copy()

        if self.tag in [
            "sonora-diamondback-hybrid-0.5",
            "sonora-diamondback-hybrid+0.0",
            "sonora-diamondback-hybrid+0.5",
            "sonora-diamondback-hybrid-grav-0.5",
            "sonora-diamondback-hybrid-grav+0.0",
            "sonora-diamondback-hybrid-grav+0.5",
        ]:
            # TODO For Teff < 1300 K there were no clouds included in
            # the model so best to use Sonora Bobcat for the spectra?
            # See https://zenodo.org/records/12735103
            if model_param["teff"] > 1300.0:
                extra_param["fsed"] = 2.0

        for key, value in param_bounds.items():
            if key not in model_param:
                if key in extra_param:
                    model_param[key] = extra_param[key]

                else:
                    param_name = update_labels([key])[0]

                    input_value = input(
                        f"The '{atmospheric_model}' atmospheric model "
                        f"requires the '{key}' parameter (i.e. "
                        f"{param_name}) as input, while it is not "
                        f"part of the '{self.tag}' evolutionary model. "
                        "Please provide a value within the available "
                        f"range from {value[0]} to {value[1]}: "
                    )

                    model_param[key] = float(input_value)
                    extra_param[key] = float(input_value)

        return model_param, extra_param

    @typechecked
    def get_points(self) -> Dict[str, np.ndarray]:
        """
        Function for returning a dictionary with the unique grid points
        of the parameters of the evolutionary data. The grid may not
        be regularly sampled unless ``create_regular_grid=True``.

        Returns
        -------
        dict(str, np.array)
            Dictionary with the parameter names and the arrays with
            the unique values in the grid of evolutionary data.
        """

        iso_data = self.read_data()

        grid_points = {
            "age": iso_data["age"],
            "mass": iso_data["mass"],
        }

        if "s_init" in iso_data:
            grid_points["s_init"] = iso_data["s_init"]

        return grid_points

    @typechecked
    def get_isochrone(
        self,
        age: float,
        s_init: Optional[float] = None,
        masses: Optional[np.ndarray] = None,
        filter_mag: Optional[str] = None,
        filters_color: Optional[Tuple[str, str]] = None,
        param_interp: Optional[List[str]] = None,
    ) -> IsochroneBox:
        """
        Function for interpolating an isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data will get
            interpolated.
        s_init : float, None
            Initial entropy (k_b/baryon) at which the isochrone
            data will get interpolated. This parameter is only
            needed by the ``tag='marleau'`` model and can be
            set to ``None`` otherwise.
        masses : np.ndarray, None
            Masses (:math:`M_\\mathrm{J}`) at which the isochrone
            data is interpolated. The masses are not interpolated
            if the argument is set to ``None``, in which case the
            mass sampling from the evolutionary data is used.
        filter_mag : str, None
            Filter name for the absolute magnitude as listed in the
            file with the isochrone data. Not selected if set to
            ``None`` or if only evolutionary tracks are available.
        filters_color : tuple(str, str), None
            Filter names for the color as listed in the file with the
            isochrone data. Not selected if set to ``None`` or if only
            evolutionary tracks are available.
        param_interp : list(str), None
            List with the parameters that will be interpolated from
            the isochrone grid. By default, all parameters will
            be interpolated and stored in the ``IsochroneBox``
            when the argument is set to ``None``. However, to
            decrease the computation time, a subset of parameters
            can be selected from 'log_lum', 'teff', 'logg', 'radius'.

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone.
        """

        if self.interp_method == "linear":
            from scipy.interpolate import LinearNDInterpolator as grid_interp

        elif self.interp_method == "cubic":
            from scipy.interpolate import CloughTocher2DInterpolator as grid_interp

        color = None
        mag_abs = None

        # Read isochrone data

        iso_data = self.read_data()

        if masses is None:
            if self.regular_grid:
                masses = iso_data["mass"]

            else:
                idx_min = (np.abs(iso_data["age"] - age)).argmin()
                age_select = iso_data["age"] == iso_data["age"][idx_min]
                masses = np.unique(iso_data["mass"][age_select])  # (Mjup)

                if masses.size < 5:
                    # This can happen if the age sampling in the
                    # isochrone grid was different for each mass,
                    # for example baraffe+2015
                    masses = np.unique(iso_data["mass"])  # (Mjup)

        # Check if initial entropy is provided

        if s_init is None and "s_init" in iso_data:
            raise ValueError(
                "The initial entropy is a parameter of the "
                f"'{self.tag}' model so please set the 's_init' "
                "parameter."
            )

        # Create array with grid points

        age_points = np.full(masses.shape[0], age)  # (Myr)

        if self.regular_grid:
            grid_points = [iso_data["age"], iso_data["mass"]]

            if "s_init" in iso_data:
                grid_points.append(iso_data["s_init"])

        else:
            if "s_init" in iso_data:
                grid_points = np.column_stack(
                    [iso_data["age"], iso_data["mass"], iso_data["s_init"]]
                )
            else:
                grid_points = np.column_stack([iso_data["age"], iso_data["mass"]])

        # Parameter values to interpolate the evolutionary tracks

        if "s_init" in iso_data:
            s_i_points = np.full(masses.shape[0], s_init)  # (k_b/baryon)
            interp_values = np.column_stack([age_points, masses, s_i_points])

        else:
            interp_values = np.column_stack([age_points, masses])

        # Check if the isochrone table has magnitudes

        filters = self.get_filters()

        if filters is not None:
            if filters_color is not None:
                if filters_color[0] in filters:
                    index_color_1 = filters.index(filters_color[0])

                else:
                    raise ValueError(
                        f"Magnitudes for the selected "
                        f"'{filters_color[0]}' filter "
                        f"are not found in the "
                        f"'{self.tag}' data. Please "
                        f"select one of the following "
                        f"filters: {filters}"
                    )

                if filters_color[1] in filters:
                    index_color_2 = filters.index(filters_color[1])

                else:
                    raise ValueError(
                        f"Magnitudes for the selected "
                        f"'{filters_color[1]}' filter "
                        f"are not found in the "
                        f"'{self.tag}' data. Please "
                        f"select one of the following "
                        f"filters: {filters}"
                    )

                if self.regular_grid:
                    mag_color_1 = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_color_1],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_color_1 = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_color_1],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

                if self.regular_grid:
                    mag_color_2 = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_color_2],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_color_2 = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_color_2],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                if filter_mag in filters:
                    index_mag = filters.index(filter_mag)

                else:
                    raise ValueError(
                        f"Magnitudes for the selected "
                        f"'{filter_mag}' filter are not "
                        f"found in the '{self.tag}' data. "
                        f"Please select one of the "
                        f"following filters: {filters}"
                    )

                if self.regular_grid:
                    mag_abs = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_mag],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_abs = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_mag],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

        # Interpolation of Teff

        if (
            "teff" in iso_data
            and self.teff_interp is None
            and (param_interp is None or "teff" in param_interp)
        ):
            if self.regular_grid:
                self.teff_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["teff"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.teff_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["teff"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.teff_interp is not None:
            teff = self.teff_interp(interp_values)
        else:
            teff = None

        # Interpolation of log(L/Lsun)

        if self.loglum_interp is None and (
            param_interp is None or "log_lum" in param_interp
        ):
            if self.regular_grid:
                self.loglum_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["log_lum"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.loglum_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["log_lum"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.loglum_interp is not None:
            log_lum = self.loglum_interp(interp_values)
        else:
            log_lum = None

        # Interpolation of log(g)

        if (
            "log_g" in iso_data
            and self.logg_interp is None
            and (param_interp is None or "logg" in param_interp)
        ):
            if self.regular_grid:
                self.logg_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["log_g"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.logg_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["log_g"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.logg_interp is not None:
            logg = self.logg_interp(interp_values)
        else:
            logg = None

        # Interpolation of radius

        if (
            "radius" in iso_data
            and self.radius_interp is None
            and (param_interp is None or "radius" in param_interp)
        ):
            if self.regular_grid:
                self.radius_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["radius"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.radius_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["radius"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.radius_interp is not None:
            radius = self.radius_interp(interp_values)
        else:
            radius = None

        # Check if magnitude and color are found

        if mag_abs is None and filter_mag is not None:
            warnings.warn(
                f"The isochrones of {self.tag} do not have "
                f"magnitudes for the {filter_mag} filter so "
                f"setting the argument of 'filter_mag' to None."
            )

            filter_mag = None

        if color is None and filters_color is not None:
            warnings.warn(
                f"The isochrones of {self.tag} do not have "
                f"magnitudes for the {filters_color} filters so "
                f"setting the argument of 'filter_color' to None."
            )

            filters_color = None

        return create_box(
            boxtype="isochrone",
            model=self.tag,
            age=age,
            s_init=s_init,
            filters_color=filters_color,
            filter_mag=filter_mag,
            color=color,
            magnitude=mag_abs,
            log_lum=log_lum,
            teff=teff,
            logg=logg,
            radius=radius,
            masses=masses,
        )

    @typechecked
    def get_cooling_track(
        self,
        mass: float,
        s_init: Optional[float] = None,
        ages: Optional[np.ndarray] = None,
        filter_mag: Optional[str] = None,
        filters_color: Optional[Tuple[str, str]] = None,
    ) -> CoolingBox:
        """
        Function for interpolating a cooling curve.

        Parameters
        ----------
        mass : float
            Mass (:math:`M_\\mathrm{J}`) for which the cooling
            curve will be interpolated.
        s_init : float, None
            Initial entropy (k_b/baryon) at which the isochrone
            data will get interpolated. This parameter is only
            needed by the ``tag='marleau'`` model and can be
            set to ``None`` otherwise.
        ages : np.ndarray, None
            Ages (Myr) at which the cooling curve will be
            interpolated. The ages are not interpolated
            if the argument is set to ``None``, in which case the
            age sampling from the evolutionary data is used.
        filter_mag : str, None
            Filter name for the absolute magnitude as listed in the
            file with the isochrone data. Not selected if set to
            ``None`` or if only evolutionary tracks are available.
        filters_color : tuple(str, str), None
            Filter names for the color as listed in the file with the
            isochrone data. Not selected if set to ``None`` or if only
            evolutionary tracks are available.

        Returns
        -------
        species.core.box.CoolingBox
            Box with the cooling curve.
        """

        if self.interp_method == "linear":
            from scipy.interpolate import LinearNDInterpolator as grid_interp

        elif self.interp_method == "cubic":
            from scipy.interpolate import CloughTocher2DInterpolator as grid_interp

        color = None
        mag_abs = None

        # Read isochrone data

        iso_data = self.read_data()

        if ages is None:
            if self.regular_grid:
                ages = iso_data["age"]

            else:
                idx_min = (np.abs(iso_data["mass"] - mass)).argmin()
                mass_select = iso_data["mass"] == iso_data["mass"][idx_min]
                ages = np.unique(iso_data["age"][mass_select])  # (Myr)

                if ages.size < 5:
                    # This can happen if the age sampling in the
                    # isochrone grid was different for each mass,
                    # for example baraffe+2015
                    ages = np.unique(iso_data["age"])  # (Mjup)

        # Check if initial entropy is provided

        if s_init is None and "s_init" in iso_data:
            raise ValueError(
                "The initial entropy is a parameter of the "
                f"'{self.tag}' model so please set the 's_init' "
                "parameter."
            )

        # Create array with grid points

        mass_points = np.full(ages.shape[0], mass)  # (Mjup)

        if self.regular_grid:
            grid_points = [iso_data["age"], iso_data["mass"]]

            if "s_init" in iso_data:
                grid_points.append(iso_data["s_init"])

        else:
            if "s_init" in iso_data:
                grid_points = np.column_stack(
                    [iso_data["age"], iso_data["mass"], iso_data["s_init"]]
                )
            else:
                grid_points = np.column_stack([iso_data["age"], iso_data["mass"]])

        # Parameter values to interpolate the evolutionary tracks

        if "s_init" in iso_data:
            s_i_points = np.full(ages.shape[0], s_init)  # (k_b/baryon)
            interp_values = np.column_stack([ages, mass_points, s_i_points])

        else:
            interp_values = np.column_stack([ages, mass_points])

        # Check if the isochrone table has magnitudes

        filters = self.get_filters()

        if filters is not None:
            if filters_color is not None:
                index_color_1 = filters.index(filters_color[0])
                index_color_2 = filters.index(filters_color[1])

            if filter_mag is not None:
                index_mag = filters.index(filter_mag)

            if filters_color is not None:
                if self.regular_grid:
                    mag_color_1 = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_color_1],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_color_1 = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_color_1],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

                if self.regular_grid:
                    mag_color_2 = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_color_2],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_color_2 = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_color_2],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                if self.regular_grid:
                    mag_abs = RegularGridInterpolator(
                        grid_points,
                        iso_data["mag"][:, index_mag],
                        method=self.interp_method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                else:
                    mag_abs = griddata(
                        points=grid_points,
                        values=iso_data["mag"][:, index_mag],
                        xi=interp_values,
                        method="linear",
                        fill_value="nan",
                        rescale=False,
                    )

        if "teff" in iso_data and self.teff_interp is None:
            if self.regular_grid:
                self.teff_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["teff"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.teff_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["teff"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.teff_interp is not None:
            teff = self.teff_interp(interp_values)
        else:
            teff = None

        if self.loglum_interp is None:
            if self.regular_grid:
                self.loglum_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["log_lum"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.loglum_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["log_lum"],
                    fill_value="nan",
                    rescale=False,
                )

        log_lum = self.loglum_interp(interp_values)

        if "log_g" in iso_data and self.logg_interp is None:
            if self.regular_grid:
                self.logg_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["log_g"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.logg_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["log_g"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.logg_interp is not None:
            logg = self.logg_interp(interp_values)
        else:
            logg = None

        if "radius" in iso_data and self.radius_interp is None:
            if self.regular_grid:
                self.radius_interp = RegularGridInterpolator(
                    grid_points,
                    iso_data["radius"],
                    method=self.interp_method,
                    bounds_error=False,
                    fill_value=np.nan,
                )

            else:
                self.radius_interp = grid_interp(
                    points=grid_points,
                    values=iso_data["radius"],
                    fill_value="nan",
                    rescale=False,
                )

        if self.radius_interp is not None:
            radius = self.radius_interp(interp_values)
        else:
            radius = None

        if mag_abs is None and filter_mag is not None:
            warnings.warn(
                f"The isochrones of {self.tag} do not have "
                f"magnitudes for the {filter_mag} filter so "
                f"setting the argument of 'filter_mag' to None."
            )

            filter_mag = None

        if mag_abs is None and filters_color is not None:
            warnings.warn(
                f"The isochrones of {self.tag} do not have "
                f"magnitudes for the {filters_color} filters so "
                f"setting the argument of 'filter_color' to None."
            )

            filters_color = None

        return create_box(
            boxtype="cooling",
            model=self.tag,
            mass=mass,
            s_init=s_init,
            filters_color=filters_color,
            filter_mag=filter_mag,
            color=color,
            magnitude=mag_abs,
            ages=ages,
            log_lum=log_lum,
            teff=teff,
            logg=logg,
            radius=radius,
        )

    @typechecked
    def get_color_magnitude(
        self,
        age: float,
        masses: Optional[np.ndarray],
        filters_color: Tuple[str, str],
        filter_mag: str,
        adapt_logg: bool = False,
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
    ) -> ColorMagBox:
        """
        Function for calculating color-magnitude pairs
        from a selected isochrone. The function selects the
        corresponding atmosphere model and computes synthetic
        photometry by interpolating and integrating the
        spectra for any given filters.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray, None
            Masses (:math:`M_\\mathrm{J}`) at which the isochrone
            data is interpolated. The masses at the nearest age
            in the grid with evolutionary data are selected if
            the argument is set to ``None``.
        filters_color : tuple(str, str)
            Filter names for the color as listed in the file with the
            isochrone data. The filter names should be provided in the
            format of the SVO Filter Profile Service.
        filter_mag : str
            Filter name for the absolute magnitude as listed in the
            file with the isochrone data. The value should be equal
            to one of the ``filters_color`` values.
        adapt_logg : bool
            Adapt :math:`\\log(g)` to the upper or lower boundary of
            the atmospheric model grid whenever the :math:`\\log(g)`
            that has been calculated from the isochrone mass and
            radius lies outside the available range of the synthetic
            spectra. Typically :math:`\\log(g)` has only a minor
            impact on the broadband magnitudes and colors.
        atmospheric_model : str, None
            Atmospheric model used to compute the synthetic photometry.
            The argument can be set to ``None`` such that the correct
            atmospheric model is automatically selected that is
            associated with the evolutionary model. If the user
            nonetheless wants to test a non-self-consistent approach
            by using a different atmospheric model, then the argument
            can be set to any of the models that can be added with
            :func:`~species.data.database.Database.add_model`.
        extra_param : dict, None
            Optional dictionary with additional parameters that are
            required for the atmospheric model but are not part of
            the evolutionary model grid. In case additional
            parameters are required for the atmospheric model but
            they are not provided in ``extra_param`` then a manual
            input will be requested when running the
            ``get_photometry`` method. Typically the ``extra_param``
            parameter is not needed so the argument can be set to
            ``None``. It will only be required if a non-self-consistent
            approach will be tested, that is, the calculation of
            synthetic photometry from an atmospheric model that is
            not associated with the evolutionary model.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the color-magnitude data.
        """

        if extra_param is None:
            extra_param = {}

        atmospheric_model = self._check_model(atmospheric_model)

        isochrone = self.get_isochrone(
            age=age, masses=masses, filters_color=None, filter_mag=None
        )

        model_1 = ReadModel(model=atmospheric_model, filter_name=filters_color[0])
        model_2 = ReadModel(model=atmospheric_model, filter_name=filters_color[1])

        param_bounds = model_1.get_bounds()

        mag1 = np.zeros(isochrone.mass.shape[0])
        mag2 = np.zeros(isochrone.mass.shape[0])

        for i in range(isochrone.mass.size):
            model_param = {
                "teff": isochrone.teff[i],
                "logg": isochrone.logg[i],
                "radius": isochrone.radius[i],
                "distance": 10.0,
            }

            if atmospheric_model == "sonora-bobcat":
                model_param["feh"] = float(self.tag[-4:])

            # The get_bounds of model_1 and model_2 are the
            # same since the same atmospheric_model is used
            model_param, extra_param = self._update_param(
                atmospheric_model, model_param, model_1.get_bounds(), extra_param
            )

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan

                warnings.warn(
                    f"The value of Teff is NaN for the following "
                    f"isochrone sample: {model_param}. Setting "
                    f"the magnitudes to NaN."
                )

            else:
                for item_bounds in param_bounds:
                    if model_param[item_bounds] < param_bounds[item_bounds][0]:
                        if adapt_logg and item_bounds == "logg":
                            warnings.warn(
                                f"The log(g) is {model_param[item_bounds]} but the "
                                f"lower boundary of the model grid is "
                                f"{param_bounds[item_bounds][0]}. Adapting "
                                f"log(g) to {param_bounds[item_bounds][0]} since "
                                f"adapt_logg=True."
                            )

                            model_param["logg"] = param_bounds["logg"][0]

                        else:
                            mag1[i] = np.nan
                            mag2[i] = np.nan

                            warnings.warn(
                                f"The value of {item_bounds} is "
                                f"{model_param[item_bounds]}, which is below "
                                f"the lower bound of the model grid "
                                f"({param_bounds[item_bounds][0]}). Setting the "
                                f"magnitudes to NaN for the following isochrone "
                                f"sample: {model_param}."
                            )

                    elif model_param[item_bounds] > param_bounds[item_bounds][1]:
                        if adapt_logg and item_bounds == "logg":
                            warnings.warn(
                                f"The log(g) is {model_param[item_bounds]} but "
                                f"the upper boundary of the model grid is "
                                f"{param_bounds[item_bounds][1]}. Adapting "
                                f"log(g) to {param_bounds[item_bounds][1]} "
                                f"since adapt_logg=True."
                            )

                            model_param["logg"] = param_bounds["logg"][1]

                        else:
                            mag1[i] = np.nan
                            mag2[i] = np.nan

                            warnings.warn(
                                f"The value of {item_bounds} is "
                                f"{model_param[item_bounds]}, which is above "
                                f"the upper bound of the model grid "
                                f"({param_bounds[item_bounds][1]}). Setting the "
                                f"magnitudes to NaN for the following isochrone "
                                f"sample: {model_param}."
                            )

                if not np.isnan(mag1[i]):
                    mag1[i], _ = model_1.get_magnitude(model_param)
                    mag2[i], _ = model_2.get_magnitude(model_param)

        if filter_mag == filters_color[0]:
            abs_mag = mag1

        elif filter_mag == filters_color[1]:
            abs_mag = mag2

        else:
            raise ValueError(
                "The argument of filter_mag should be equal to "
                "one of the two filter values of filters_color."
            )

        return create_box(
            boxtype="colormag",
            library=atmospheric_model,
            object_type="model",
            filters_color=filters_color,
            filter_mag=filter_mag,
            color=mag1 - mag2,
            magnitude=abs_mag,
            mass=isochrone.mass,
            radius=isochrone.radius,
            iso_tag=self.tag,
            age=age,
        )

    @typechecked
    def get_color_color(
        self,
        age: float,
        masses: Optional[np.ndarray],
        filters_colors: Tuple[Tuple[str, str], Tuple[str, str]],
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
    ) -> ColorColorBox:
        """
        Function for calculating color-color pairs
        from a selected isochrone. The function selects the
        corresponding atmosphere model and computes synthetic
        photometry by interpolating and integrating the spectra
        for any given filters.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray, None
            Masses (:math:`M_\\mathrm{J}`) at which the isochrone
            data is interpolated. The masses at the nearest age
            in the grid with evolutionary data are selected if
            the argument is set to ``None``.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names for the colors as listed in the file with the
            isochrone data. The filter names should be provided in the
            format of the SVO Filter Profile Service.
        atmospheric_model : str, None
            Atmospheric model used to compute the synthetic photometry.
            The argument can be set to ``None`` such that the correct
            atmospheric model is automatically selected that is
            associated with the evolutionary model. If the user
            nonetheless wants to test a non-self-consistent approach
            by using a different atmospheric model, then the argument
            can be set to any of the models that can be added with
            :func:`~species.data.database.Database.add_model`.
        extra_param : dict, None
            Optional dictionary with additional parameters that are
            required for the atmospheric model but are not part of
            the evolutionary model grid. In case additional
            parameters are required for the atmospheric model but
            they are not provided in ``extra_param`` then a manual
            input will be requested when running the
            ``get_photometry`` method. Typically the ``extra_param``
            parameter is not needed so the argument can be set to
            ``None``. It will only be required if a non-self-consistent
            approach will be tested, that is, the calculation of
            synthetic photometry from an atmospheric model that is
            not associated with the evolutionary model.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the color-color data.
        """

        if extra_param is None:
            extra_param = {}

        atmospheric_model = self._check_model(atmospheric_model)

        isochrone = self.get_isochrone(
            age=age, masses=masses, filters_color=None, filter_mag=None
        )

        model_1 = ReadModel(model=atmospheric_model, filter_name=filters_colors[0][0])
        model_2 = ReadModel(model=atmospheric_model, filter_name=filters_colors[0][1])
        model_3 = ReadModel(model=atmospheric_model, filter_name=filters_colors[1][0])
        model_4 = ReadModel(model=atmospheric_model, filter_name=filters_colors[1][1])

        mag1 = np.zeros(isochrone.mass.shape[0])
        mag2 = np.zeros(isochrone.mass.shape[0])
        mag3 = np.zeros(isochrone.mass.shape[0])
        mag4 = np.zeros(isochrone.mass.shape[0])

        for i in range(isochrone.mass.size):
            model_param = {
                "teff": isochrone.teff[i],
                "logg": isochrone.logg[i],
                "radius": isochrone.radius[i],
                "distance": 10.0,
            }

            if atmospheric_model == "sonora-bobcat":
                model_param["feh"] = float(self.tag[-4:])

            # The get_bounds of model_1 and model_2 are the
            # same since the same atmospheric_model is used
            model_param, extra_param = self._update_param(
                atmospheric_model, model_param, model_1.get_bounds(), extra_param
            )

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan
                mag3[i] = np.nan
                mag4[i] = np.nan

                warnings.warn(
                    "The value of Teff is NaN for the following "
                    f"isochrone sample: {model_param}. Setting "
                    "the magnitudes to NaN."
                )

            else:
                for item_bounds in model_1.get_bounds():
                    if model_param[item_bounds] < model_1.get_bounds()[item_bounds][0]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(
                            f"The value of {item_bounds} is "
                            f"{model_param[item_bounds]}, which is "
                            f"below the lower bound of the model grid "
                            f" ({model_1.get_bounds()[item_bounds][0]}). "
                            f"Setting the magnitudes to NaN for the "
                            f"following isochrone sample: {model_param}."
                        )

                    elif (
                        model_param[item_bounds] > model_1.get_bounds()[item_bounds][1]
                    ):
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(
                            f"The value of {item_bounds} is "
                            f"{model_param[item_bounds]}, which is above "
                            f"the upper bound of the model grid "
                            f"({model_1.get_bounds()[item_bounds][1]}). "
                            f"Setting the magnitudes to NaN for the "
                            f"following isochrone sample: {model_param}."
                        )

                if (
                    not np.isnan(mag1[i])
                    and not np.isnan(mag2[i])
                    and not np.isnan(mag3[i])
                    and not np.isnan(mag4[i])
                ):
                    mag1[i], _ = model_1.get_magnitude(model_param)
                    mag2[i], _ = model_2.get_magnitude(model_param)
                    mag3[i], _ = model_3.get_magnitude(model_param)
                    mag4[i], _ = model_4.get_magnitude(model_param)

        return create_box(
            boxtype="colorcolor",
            library=atmospheric_model,
            object_type="model",
            filters=filters_colors,
            color1=mag1 - mag2,
            color2=mag3 - mag4,
            mass=isochrone.mass,
            radius=isochrone.radius,
            iso_tag=self.tag,
            age=age,
        )

    @typechecked
    def get_mass(
        self,
        age: float,
        log_lum: np.ndarray,
    ) -> np.ndarray:
        """
        Function for interpolating a mass for a given
        age and array with bolometric luminosities.

        Parameters
        ----------
        age : float
            Age (Myr) at which the masses will be interpolated.
        log_lum : np.ndarray
            Array with the bolometric luminosities,
            :math:`\\log{(L/L_\\odot)}`, for which the
            masses will be interpolated.

        Returns
        -------
        np.ndarray
            Array with masses (:math:`M_\\mathrm{J}`).
        """

        if self.interp_method == "linear":
            from scipy.interpolate import LinearNDInterpolator as grid_interp

        elif self.interp_method == "cubic":
            from scipy.interpolate import CloughTocher2DInterpolator as grid_interp

        # Read isochrone data

        iso_data = self.read_data()

        # Check for a regular grid

        if self.regular_grid:
            # Using a regular grid is not possible because
            # log_lum is not part of the regular grid points

            raise NotImplementedError(
                f"The 'get_mass()' method does not support the '{self.tag}' model."
            )

        # Create array with grid points

        grid_points = np.column_stack([iso_data["age"], iso_data["log_lum"]])

        # Parameter values to interpolate the evolutionary tracks

        age_points = np.full(log_lum.size, age)  # (Myr)
        interp_values = np.column_stack([age_points, log_lum])

        # Interpolate masses

        if self.mass_interp is None:
            self.mass_interp = grid_interp(
                points=grid_points,
                values=iso_data["mass"],
                fill_value="nan",
                rescale=False,
            )

        return self.mass_interp(interp_values)

    @typechecked
    def get_radius(
        self,
        age: float,
        log_lum: np.ndarray,
    ) -> np.ndarray:
        """
        Function for interpolating a radius for a given
        age and array with bolometric luminosities.

        Parameters
        ----------
        age : float
            Age (Myr) at which the masses will be interpolated.
        log_lum : np.ndarray
            Array with the bolometric luminosities,
            :math:`\\log{(L/L_\\odot)}`, for which the
            masses will be interpolated.

        Returns
        -------
        np.ndarray
            Array with radii (:math:`R_\\mathrm{J}`).
        """

        if self.interp_method == "linear":
            from scipy.interpolate import LinearNDInterpolator as grid_interp

        elif self.interp_method == "cubic":
            from scipy.interpolate import CloughTocher2DInterpolator as grid_interp

        # Read isochrone data

        iso_data = self.read_data()

        # Check for a regular grid

        if self.regular_grid:
            # Using a regular grid is not possible because
            # log_lum is not part of the regular grid points

            raise NotImplementedError(
                f"The 'get_radius()' method does not support the '{self.tag}' model."
            )

        # Create array with grid points

        grid_points = np.column_stack([iso_data["age"], iso_data["log_lum"]])

        # Parameter values to interpolate the evolutionary tracks

        age_points = np.full(log_lum.size, age)  # (Myr)
        interp_values = np.column_stack([age_points, log_lum])

        # Interpolate radii

        if self.radius_interp is None:
            self.radius_interp = grid_interp(
                points=grid_points,
                values=iso_data["radius"],
                fill_value="nan",
                rescale=False,
            )

        return self.radius_interp(interp_values)

    @typechecked
    def get_filters(self) -> Optional[List[str]]:
        """
        Function for get a list with filter names for which there
        are are magnitudes stored with the isochrone data.

        Returns
        -------
        list(str), None
            List with filter names. A ``None`` is returned if
            there are no filters and magnitudes stored with
            the isochrone data.
        """

        with h5py.File(self.database, "r") as hdf5_file:
            if "filters" in hdf5_file[f"isochrones/{self.tag}"]:
                filters = list(hdf5_file[f"isochrones/{self.tag}/filters/"])

                # Convert from bytes to strings
                for i, item in enumerate(filters):
                    if isinstance(item, bytes):
                        filters[i] = item.decode("utf-8")

            else:
                filters = None

        return filters

    @typechecked
    def get_photometry(
        self,
        age: float,
        mass: float,
        distance: float,
        filter_name: str,
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
    ) -> PhotometryBox:
        """
        Function for computing synthetic photometry by interpolating
        and integrating the associated spectra. Bulk and atmosphere
        parameters are interpolated from the evolutionary data for
        the requested age and mass. The output from the evolutionary
        data is then used as input for the atmospheric model. This
        function is useful if the required magnitudes or fluxes
        are not part of the available filters of the evolutionary
        data (i.e. the filters returned by
        :func:`~species.read.read_isochrone.ReadIsochrone.get_filters`).
        The atmospheric model that is associated with the evolutionary
        model is by default automatically selected and added to the
        database if needed.

        Parameters
        ----------
        age : float
            Age (Myr) at which the bulk parameters will be
            interpolated from the grid with evolutionary data.
        mass : float
            Mass (:math:`M_\\mathrm{J}`) at which the bulk
            parameters will be interpolated from the grid with
            evolutionary data.
        distance : float
            Distance (pc) that is used for scaling the fluxes
            from the atmosphere to the observer.
        filter_name : tuple(str, str), None
            Filter name for which the synthetic photometry will be
            computed. Any filter name from the `SVO Filter Profile
            Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_
            can be used as argument.
        atmospheric_model : str, None
            Atmospheric model used to compute the synthetic photometry.
            The argument can be set to ``None`` such that the correct
            atmospheric model is automatically selected that is
            associated with the evolutionary model. If the user
            nonetheless wants to test a non-self-consistent approach
            by using a different atmospheric model, then the argument
            can be set to any of the models that can be added with
            :func:`~species.data.database.Database.add_model`.
        extra_param : dict, None
            Optional dictionary with additional parameters that are
            required for the atmospheric model but are not part of
            the evolutionary model grid. In case additional
            parameters are required for the atmospheric model but
            they are not provided in ``extra_param`` then a manual
            input will be requested when running the
            ``get_photometry`` method. Typically the ``extra_param``
            parameter is not needed so the argument can be set to
            ``None``. It will only be required if a non-self-consistent
            approach will be tested, that is, the calculation of
            synthetic photometry from an atmospheric model that is
            not associated with the evolutionary model.

        Returns
        -------
        species.core.box.PhotometryBox
            Box with the synthetic photometry (magnitude and flux).
        """

        if extra_param is None:
            extra_param = {}

        atmospheric_model = self._check_model(atmospheric_model)

        iso_box = self.get_isochrone(age=age, masses=np.array([mass]))

        model_param = {
            "teff": iso_box.teff[0],
            "logg": iso_box.logg[0],
            "radius": iso_box.radius[0],
            "distance": distance,
        }

        model_reader = ReadModel(model=atmospheric_model, filter_name=filter_name)

        model_param, _ = self._update_param(
            atmospheric_model, model_param, model_reader.get_bounds(), extra_param
        )

        phot_box = model_reader.get_flux(model_param=model_param, return_box=True)

        return phot_box

    @typechecked
    def get_spectrum(
        self,
        age: float,
        mass: float,
        distance: float,
        wavel_range: Optional[Tuple[float, float]] = None,
        spec_res: Optional[float] = None,
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
    ) -> ModelBox:
        """
        Function for interpolating the model spectrum at a specified
        age and mass. Bulk and atmosphere parameters are interpolated
        from the evolutionary data for the requested age and mass.
        The output from the evolutionary data is then used as input
        for the atmospheric model. The atmospheric model that is
        associated with the evolutionary model is by default
        automatically selected and added to the database if needed.

        Parameters
        ----------
        age : float
            Age (Myr) at which the bulk parameters will be
            interpolated from the grid with evolutionary data.
        mass : float
            Mass (:math:`M_\\mathrm{J}`) at which the bulk
            parameters will be interpolated from the grid with
            evolutionary data.
        distance : float
            Distance (pc) that is used for scaling the fluxes
            from the atmosphere to the observer.
        wavel_range : tuple(float, float), None
            Wavelength range (um). Full spectrum is selected if
            the argument is set to ``None``.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum
            with a Gaussian kernel. No smoothing is applied when the
            argument is set to ``None``.
        atmospheric_model : str, None
            Atmospheric model used to compute the synthetic photometry.
            The argument can be set to ``None`` such that the correct
            atmospheric model is automatically selected that is
            associated with the evolutionary model. If the user
            nonetheless wants to test a non-self-consistent approach
            by using a different atmospheric model, then the argument
            can be set to any of the models that can be added with
            :func:`~species.data.database.Database.add_model`.
        extra_param : dict, None
            Optional dictionary with additional parameters that are
            required for the atmospheric model but are not part of
            the evolutionary model grid. In case additional
            parameters are required for the atmospheric model but
            they are not provided in ``extra_param`` then a manual
            input will be requested when running the
            ``get_photometry`` method. Typically the ``extra_param``
            parameter is not needed so the argument can be set to
            ``None``. It will only be required if a non-self-consistent
            approach will be tested, that is, the calculation of
            synthetic photometry from an atmospheric model that is
            not associated with the evolutionary model.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if extra_param is None:
            extra_param = {}

        atmospheric_model = self._check_model(atmospheric_model)

        iso_box = self.get_isochrone(age=age, masses=np.array([mass]))

        model_param = {
            "teff": iso_box.teff[0],
            "logg": iso_box.logg[0],
            "radius": iso_box.radius[0],
            "distance": distance,
        }

        model_reader = ReadModel(model=atmospheric_model, wavel_range=wavel_range)

        model_param, _ = self._update_param(
            atmospheric_model, model_param, model_reader.get_bounds(), extra_param
        )

        model_box = model_reader.get_model(
            model_param=model_param, spec_res=spec_res, smooth=True
        )

        return model_box

    @typechecked
    def contrast_to_mass(
        self,
        age: float,
        distance: float,
        filter_name: str,
        star_mag: float,
        contrast: Union[List[float], np.ndarray],
        use_mag: bool = True,
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
        calc_phot: bool = False,
    ) -> np.ndarray:
        """
        Function for converting contrast values into masses. This
        can be used to convert a list/array with detection
        limits of companions into mass limits. Either
        one of the available filter names from the isochrone grid
        can be selected (i.e. the filters returned by
        :func:`~species.read.read_isochrone.ReadIsochrone.get_filters`),
        or any of the filters from the `SVO Filter Profile
        Service <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_. For
        the first case, the magnitudes will be directly interpolated
        from the grid of evolution data. For the second case, the
        associated model spectra will be used for calculating
        synthetic photometry for the isochrone age and selected
        filter. These will then be interpolated to the requested
        contrast values. The atmospheric model that is associated
        with the evolutionary model is by default automatically
        selected and added to the database if needed.

        Parameters
        ----------
        age : float
            Age (Myr) at which the bulk parameters will be
            interpolated from the grid with evolutionary data.
        distance : float
            Distance (pc) that is used for scaling the fluxes
            from the atmosphere to the observer.
        filter_name : str
            Filter name for which the magnitudes will be interpolated,
            either directly from the isochrone grid or by calculating
            synthetic photometry from the associated model spectra.
            The first case only works for the filters that are
            returned by the
            :func:`~species.read.read_isochrone.ReadIsochrone.get_filters`
            method of :class:`~species.read.read_isochrone.ReadIsochrone`
            because these will have pre-calculated magnitudes. The
            second case will work for any of the filter names from the
            `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_. This will
            require more disk space and a bit more computation time.
        star_mag : float
            Stellar apparent magnitude for the filter that is set
            as argument of `filter_name`.
        contrast : list(float), np.ndarray
            List or array with the contrast values between a companion
            and the star. The magnitude of the star should be provided
            as argument of ``star_mag``. The contrast values will be
            converted into masses, while taking into account the
            stellar magnitude. The values should be provided
            either as ratio (e.g. ``[1e-2, 1e-3, 1e-4]``) or as
            magnitudes (e.g. ``[5.0, 7.5, 10.0]``). For ratios,
            it is important to set ``use_mag=False``.
        use_mag : bool
            Set to ``True`` if the values of ``contrast`` are given as
            magnitudes. Set to ``False`` if the values of ``contrast``
            are given as ratios. The default is set to ``True``.
        atmospheric_model : str, None
            Atmospheric model used to compute the synthetic photometry
            in case the ``filter_name`` is set to a value from the
            SVO Filter Profile Service. The argument can be set to
            ``None`` such that the correct atmospheric model is
            automatically selected that is associated with the
            evolutionary model. If the user nonetheless wants to test
            a non-self-consistent approach by using a different
            atmospheric model, then the argument can be set to any of
            the models that can be added with
            :func:`~species.data.database.Database.add_model`.
        extra_param : dict, None
            Optional dictionary with additional parameters that are
            required for the atmospheric model but are not part of
            the evolutionary model grid, for example because they
            were implicitly set by the evolution model (e.g.
            solar metallicity). In case additional parameters are
            required for the atmospheric model but they are not
            provided in ``extra_param`` then a manual input will
            be requested when running the ``get_photometry`` method.
            Typically the ``extra_param`` parameter is not needed so
            the argument can be set to ``None``. It will only be
            required if a non-self-consistent approach will be tested,
            that is, the calculation of synthetic photometry from an
            atmospheric model that is not associated with the
            evolutionary model.
        calc_phot : bool
            Calculate synthetic photometry from the model spectra
            regardless if pre-calculated magnitudes for the
            ``filter_name`` are already available with the isochrone
            data. Typically the argument can be set to ``False``,
            but to force the calculation of synthetic photometry the
            argument can be set to ``True``.

        Returns
        -------
        np.ndarray
            Array with the masses (in :math:`M_\\mathrm{J}`) for the
            requested contrast values.
        """

        if isinstance(contrast, list):
            contrast = np.array(contrast)

        if use_mag and np.all(contrast < 1.0):
            warnings.warn(
                "All values in the array of 'contrast' are "
                "smaller than 1.0 but the argument of "
                "'use_mag' is set to True. Please set the "
                "argument of 'magnitude' to False in case "
                "the values of 'contrast' are given as "
                "ratios instead of magnitudes."
            )

        if not use_mag:
            # Convert contrast from ratio to magnitude
            contrast = -2.5 * np.log10(contrast)

        if extra_param is None:
            extra_param = {}

        atmospheric_model = self._check_model(atmospheric_model)

        app_mag = star_mag + contrast
        abs_mag = apparent_to_absolute((app_mag, None), (distance, None))[0]

        filter_list = self.get_filters()

        if filter_list is not None and filter_name in filter_list and not calc_phot:
            print(
                f"The '{filter_name}' filter is found in the list "
                "of available filters from the isochrone data of "
                f"'{self.tag}'.\nThe requested contrast values "
                "will be directly interpolated from the grid with "
                "pre-calculated magnitudes."
            )

            iso_box = self.get_isochrone(age=age, masses=None, filter_mag=filter_name)

            # x (=iso_box.magnitude) must be increasing in np.interp
            mass_array = np.interp(
                abs_mag,
                iso_box.magnitude[::-1],
                iso_box.mass[::-1],
                left=np.nan,
                right=np.nan,
            )

        else:
            if filter_list is not None and filter_name not in filter_list:
                print(
                    f"The '{filter_name}' filter is not found in the "
                    "list of available filters from the isochrone "
                    f"data of '{self.tag}'.\nIt will be tried to "
                    "download the filter profile (if needed) and to "
                    "use the associated atmospheric model spectra "
                    "for calculating synthetic photometry."
                )

            model_reader = ReadModel(model=atmospheric_model, filter_name=filter_name)

            iso_box = self.get_isochrone(age=age, masses=None, filter_mag=None)

            model_abs_mag = np.zeros(iso_box.mass.size)

            for i in range(iso_box.mass.size):
                model_param = {
                    "teff": iso_box.teff[i],
                    "logg": iso_box.logg[i],
                    "radius": iso_box.radius[i],
                    "distance": distance,
                }

                model_param, _ = self._update_param(
                    atmospheric_model,
                    model_param,
                    model_reader.get_bounds(),
                    extra_param,
                )

                # The get_magnitude method returns the
                # apparent magnitude and absolute magnitude
                _, model_abs_mag[i] = model_reader.get_magnitude(
                    model_param=model_param, return_box=False
                )

            # x (=model_abs_mag) must be increasing in np.interp
            mass_array = np.interp(
                abs_mag,
                model_abs_mag[::-1],
                iso_box.mass[::-1],
                left=np.nan,
                right=np.nan,
            )

        return mass_array
