"""
Module with reading functionalities for isochrones and cooling curves.
"""

import configparser
import os
import warnings

from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from scipy import interpolate
from typeguard import typechecked

from species.core import box
from species.read import read_model
from species.util import plot_util


class ReadIsochrone:
    """
    Class for reading isochrone data from the database.
    This class interpolates the evolutionary track or
    isochrone data. Please carefully check for interpolation
    effects. Setting ``masses=None`` in
    :func:`~species.read.read_isochrone.ReadIsochrone.get_isochrone`
    extracts the isochrones at the masses of the original
    grid, so using that option helps with comparing results
    for which the masses have been interpolated. Similarly, by
    setting ``ages=None`` with the
    :func:`~species.read.read_isochrone.ReadIsochrone.get_isochrone`
    method will fix the ages to those of the original grid.
    """

    @typechecked
    def __init__(
        self, tag: Optional[str] = None, create_regular_grid: bool = False, extrapolate: bool = False
    ) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag of the isochrone data (e.g. 'ames-cond',
            'sonora+0.0', 'atmo-ceq'). When using an incorrect
            argument, and error message is printed that includes
            a list with the isochrone models that are available
            in the current ``species`` database.
        create_regular_grid : bool
            Evolutionary grids can be irregular in the (age, mass)
            space. By setting ``create_regular_grid=True``, the
            grid will be interpolated and extrapolate onto
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
            cooling curves extracted with
            :func:`~species.read.read_isochrone.ReadIsochrone.get_cooling_curve`.
        extrapolate : str
            DEPRECATED: This parameter has been renamed to
            ``create_regular_grid`` and will be removed in a future
            release. Please use the ``create_regular_grid``
            parameter instead.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag

        self.extrapolate = extrapolate
        self.create_regular_grid = create_regular_grid

        if self.extrapolate:
            warnings.warn(
                "The 'extrapolate' parameter has been "
                "renamed to 'create_regular_grid' and "
                "will be removed in a future release.",
                DeprecationWarning,
            )

            if not self.create_regular_grid:
                warnings.warn(
                    "Setting 'create_regular_grid=True' since 'extrapolate=True'."
                )

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.interp_method = config["species"]["interp_method"]

        if self.tag is None:
            with h5py.File(self.database, "r") as h5_file:
                tag_list = list(h5_file["isochrones"])

            self.tag = input("Please select one of the following "
                             "isochrone tags that are stored in "
                             "the database or use 'add_isochrones' "
                             "to add another model to the database:"
                             f"\n{tag_list}:\n")

        with h5py.File(self.database, "r") as h5_file:
            if f"isochrones/{self.tag}" not in h5_file:
                tag_list = list(h5_file["isochrones"])

                raise ValueError(
                    f"There is no isochrone data stored with the "
                    f"selected tag '{tag}'. The following isochrone "
                    f"tags are found in the database: {tag_list}"
                )

        self.mag_models = ["ames", "atmo", "baraffe", "bt-settl", "linder2019", "manual", "nextgen"]

        # Connect isochrone model with atmosphere model
        # key = isochrone model, value = atmosphere model
        self.match_model = {
            "ames-cond": "ames-cond",
            "ames-dusty": "ames-dusty",
            "atmo-ceq": "atmo-ceq",
            "atmo-neq-strong": "atmo-neq-strong",
            "atmo-neq-weak": "atmo-neq-weak",
            "bt-settl": "bt-settl",
            "saumon2008-nc_solar": "saumon2008-clear",
            "saumon2008-f2_solar": "saumon2008-cloudy",
            "sonora+0.0": "sonora-bobcat",
        }

    @typechecked
    def _read_data(
        self,
    ) -> Tuple[
        str,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
    ]:
        """
        Internal function for reading the evolutionary
        data from the database.

        Returns
        -------
        str
            Model name.
        np.ndarray
            Array with the age (Myr).
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
        np.ndarray, None
            Optional array with the absolute magnitudes. The
            array has two axes with the length of the second
            axis equal to the number of filters for which
            there are magnitudes available.
        """

        with h5py.File(self.database, "r") as h5_file:
            model_name = h5_file[f"isochrones/{self.tag}/age"].attrs["model"]

            iso_age = np.asarray(h5_file[f"isochrones/{self.tag}/age"])
            iso_mass = np.asarray(h5_file[f"isochrones/{self.tag}/mass"])
            iso_teff = np.asarray(h5_file[f"isochrones/{self.tag}/teff"])
            iso_loglum = np.asarray(h5_file[f"isochrones/{self.tag}/log_lum"])
            iso_logg = np.asarray(h5_file[f"isochrones/{self.tag}/log_g"])
            iso_radius = np.asarray(h5_file[f"isochrones/{self.tag}/radius"])

            if f"isochrones/{self.tag}/magnitudes" in h5_file:
                iso_mag = np.asarray(h5_file[f"isochrones/{self.tag}/magnitudes"])
            else:
                iso_mag = None

        if self.create_regular_grid:
            age_unique = np.unique(iso_age)
            mass_unique = np.unique(iso_mass)

            n_ages = age_unique.shape[0]
            n_masses = mass_unique.shape[0]

            new_age = np.zeros((n_ages * n_masses))
            new_mass = np.zeros((n_ages * n_masses))
            new_teff = np.zeros((n_ages * n_masses))
            new_loglum = np.zeros((n_ages * n_masses))
            new_logg = np.zeros((n_ages * n_masses))
            new_radius = np.zeros((n_ages * n_masses))

            if iso_mag is not None:
                new_mag = np.zeros(((n_ages * n_masses, iso_mag.shape[1])))

            for j, age_item in enumerate(age_unique):
                age_select = iso_age == age_item
                ages_tmp = np.full(n_masses, age_item)

                new_age[j * n_masses : (j + 1) * n_masses] = ages_tmp
                new_mass[j * n_masses : (j + 1) * n_masses] = mass_unique

                interp_teff = interpolate.interp1d(
                    iso_mass[age_select],
                    iso_teff[age_select],
                    fill_value="extrapolate",
                )

                new_teff[j * n_masses : (j + 1) * n_masses] = interp_teff(mass_unique)

                interp_loglum = interpolate.interp1d(
                    iso_mass[age_select],
                    iso_loglum[age_select],
                    fill_value="extrapolate",
                )

                new_loglum[j * n_masses : (j + 1) * n_masses] = interp_loglum(
                    mass_unique
                )

                interp_logg = interpolate.interp1d(
                    iso_mass[age_select],
                    iso_logg[age_select],
                    fill_value="extrapolate",
                )

                new_logg[j * n_masses : (j + 1) * n_masses] = interp_logg(mass_unique)

                interp_radius = interpolate.interp1d(
                    iso_mass[age_select],
                    iso_radius[age_select],
                    fill_value="extrapolate",
                )

                new_radius[j * n_masses : (j + 1) * n_masses] = interp_radius(
                    mass_unique
                )

                for k in range(iso_mag.shape[1]):
                    interp_mag = interpolate.interp1d(
                        iso_mass[age_select],
                        iso_mag[age_select, k],
                        fill_value="extrapolate",
                    )

                    new_mag[j * n_masses : (j + 1) * n_masses, k] = interp_mag(
                        mass_unique
                    )

            iso_age = new_age.copy()
            iso_mass = new_mass.copy()
            iso_teff = new_teff.copy()
            iso_loglum = new_loglum.copy()
            iso_logg = new_logg.copy()
            iso_radius = new_radius.copy()

            if iso_mag is not None:
                iso_mag = new_mag.copy()

        return (
            model_name,
            iso_age,
            iso_mass,
            iso_teff,
            iso_loglum,
            iso_logg,
            iso_radius,
            iso_mag,
        )

    @typechecked
    def _check_model(self, atmospheric_model: Optional[str]) -> str:
        """
        Internal function for matching the atmospheric model with
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
                atmospheric_model = self.match_model[self.tag]
            else:
                raise ValueError(
                    "Can not find the atmosphere model "
                    f"associated with the '{self.tag}' "
                    "evolutionary model. Please contact "
                    "the code maintainer."
                )

        elif self.tag in self.match_model:
            if atmospheric_model != self.match_model[self.tag]:
                warnings.warn(
                    "Please note that you have selected "
                    f"'{atmospheric_model}' as "
                    f"atmospheric model for '{self.tag}' "
                    f"while '{self.match_model[self.tag]}'"
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

        for key, value in param_bounds.items():
            if key not in model_param:
                if key in extra_param:
                    model_param[key] = extra_param[key]

                else:
                    param_name = plot_util.update_labels([key])[0]

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
    def grid_points(self) -> Dict[str, np.ndarray]:
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

        (
            model,
            iso_age,
            iso_mass,
            iso_teff,
            iso_loglum,
            iso_logg,
            iso_radius,
            iso_mag,
        ) = self._read_data()

        grid_points = {
            "age": iso_age,
            "mass": iso_mass,
            "radius": iso_radius,
            "log_lum": iso_loglum,
            "teff": iso_teff,
            "logg": iso_logg,
        }

        return grid_points

    @typechecked
    def get_isochrone(
        self,
        age: float,
        masses: Optional[np.ndarray] = None,
        filter_mag: Optional[str] = None,
        filters_color: Optional[Tuple[str, str]] = None,
    ) -> box.IsochroneBox:
        """
        Function for interpolating an isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
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

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone.
        """

        color = None
        mag_abs = None

        # Read isochrone data

        (
            model,
            iso_age,
            iso_mass,
            iso_teff,
            iso_loglum,
            iso_logg,
            iso_radius,
            iso_mag,
        ) = self._read_data()

        if masses is None:
            idx_min = (np.abs(iso_age - age)).argmin()
            age_select = iso_age == iso_age[idx_min]
            masses = np.unique(iso_mass[age_select])  # (Mjup)

        age_points = np.full(masses.shape[0], age)  # (Myr)
        grid_points = np.column_stack([iso_age, iso_mass])

        filters = self.get_filters()

        if model in self.mag_models:
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

            if filters_color is not None:
                mag_color_1 = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_color_1],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                mag_color_2 = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_color_2],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                mag_abs = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_mag],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

        teff = interpolate.griddata(
            points=grid_points,
            values=iso_teff,
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        log_lum = interpolate.griddata(
            points=grid_points,
            values=iso_loglum,
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        logg = interpolate.griddata(
            points=grid_points,
            values=iso_logg,
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        radius = interpolate.griddata(
            points=grid_points,
            values=iso_radius,
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

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

        return box.create_box(
            boxtype="isochrone",
            model=self.tag,
            age=age,
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
    def get_cooling_curve(
        self,
        mass: float,
        ages: Optional[np.ndarray] = None,
        filter_mag: Optional[str] = None,
        filters_color: Optional[Tuple[str, str]] = None,
    ) -> box.CoolingBox:
        """
        Function for interpolating a cooling curve.

        Parameters
        ----------
        mass : float
            Mass (:math:`M_\\mathrm{J}`) for which the cooling
            curve will be interpolated.
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

        color = None
        mag_abs = None

        # Read isochrone data

        (
            model,
            iso_age,
            iso_mass,
            iso_teff,
            iso_loglum,
            iso_logg,
            iso_radius,
            iso_mag,
        ) = self._read_data()

        if ages is None:
            idx_min = (np.abs(iso_mass - mass)).argmin()
            mass_select = iso_mass == iso_mass[idx_min]
            ages = np.unique(iso_age[mass_select])  # (Myr)

        mass_points = np.full(ages.shape[0], mass)  # (Mjup)
        grid_points = np.column_stack([iso_age, iso_mass])

        filters = self.get_filters()

        if model in self.mag_models:
            if filters_color is not None:
                index_color_1 = filters.index(filters_color[0])
                index_color_2 = filters.index(filters_color[1])

            if filter_mag is not None:
                index_mag = filters.index(filter_mag)

            if filters_color is not None:
                mag_color_1 = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_color_1],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                mag_color_2 = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_color_2],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                mag_abs = interpolate.griddata(
                    points=grid_points,
                    values=iso_mag[:, index_mag],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

        teff = interpolate.griddata(
            points=grid_points,
            values=iso_teff,
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        log_lum = interpolate.griddata(
            points=grid_points,
            values=iso_loglum,
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        logg = interpolate.griddata(
            points=grid_points,
            values=iso_logg,
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        radius = interpolate.griddata(
            points=grid_points,
            values=iso_radius,
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

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

        return box.create_box(
            boxtype="cooling",
            model=self.tag,
            mass=mass,
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
    ) -> box.ColorMagBox:
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

        model_1 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_color[0]
        )
        model_2 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_color[1]
        )

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

        return box.create_box(
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
        )

    @typechecked
    def get_color_color(
        self,
        age: float,
        masses: Optional[np.ndarray],
        filters_colors: Tuple[Tuple[str, str], Tuple[str, str]],
        atmospheric_model: Optional[str] = None,
        extra_param: Optional[Dict[str, float]] = None,
    ) -> box.ColorColorBox:
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

        model_1 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_colors[0][0]
        )
        model_2 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_colors[0][1]
        )
        model_3 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_colors[1][0]
        )
        model_4 = read_model.ReadModel(
            model=atmospheric_model, filter_name=filters_colors[1][1]
        )

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

        return box.create_box(
            boxtype="colorcolor",
            library=atmospheric_model,
            object_type="model",
            filters=filters_colors,
            color1=mag1 - mag2,
            color2=mag3 - mag4,
            mass=isochrone.mass,
            radius=isochrone.radius,
            iso_tag=self.tag,
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

        # Read isochrone data

        (
            _,
            iso_age,
            iso_mass,
            _,
            iso_loglum,
            _,
            _,
            _,
        ) = self._read_data()

        # Interpolate masses

        grid_points = np.stack((iso_age, iso_loglum), axis=1)

        age_points = np.full(log_lum.size, age)  # (Myr)

        mass = interpolate.griddata(
            points=grid_points,
            values=iso_mass,
            xi=np.stack((age_points, log_lum), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        return mass

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

        # Read isochrone data

        (
            _,
            iso_age,
            _,
            _,
            iso_loglum,
            _,
            iso_radius,
            _,
        ) = self._read_data()

        # Interpolate radius

        grid_points = np.stack((iso_age, iso_loglum), axis=1)

        age_points = np.full(log_lum.size, age)  # (Myr)

        radius = interpolate.griddata(
            points=grid_points,
            values=iso_radius,
            xi=np.stack((age_points, log_lum), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        return radius

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

        with h5py.File(self.database, "r") as h5_file:
            if "filters" in h5_file[f"isochrones/{self.tag}"]:
                filters = list(h5_file[f"isochrones/{self.tag}/filters/"])

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
    ) -> box.PhotometryBox:
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

        model_reader = read_model.ReadModel(
            model=atmospheric_model, filter_name=filter_name
        )

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
    ) -> box.ModelBox:
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

        model_reader = read_model.ReadModel(
            model=atmospheric_model, wavel_range=wavel_range
        )

        model_param, _ = self._update_param(
            atmospheric_model, model_param, model_reader.get_bounds(), extra_param
        )

        model_box = model_reader.get_model(
            model_param=model_param, spec_res=spec_res, smooth=True
        )

        return model_box
