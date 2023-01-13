"""
Module with reading functionalities for isochrones and cooling curves.
"""

import configparser
import os
import warnings

from typing import List, Optional, Tuple

import h5py
import numpy as np

from scipy import interpolate
from typeguard import typechecked

from species.core import box
from species.read import read_model
from species.util import read_util


class ReadIsochrone:
    """
    Class for reading isochrone data from the database. This class
    interpolates the evolutionary track or isochrone data.
    Please carefully check for interpolation effects. Setting
    ``masses=None`` in
    :func:`~species.read.read_isochrone.ReadIsochrone.get_isochrone`
    extracts the isochrones at the masses of the original grid,
    so using that option helps with comparing results for which
    the masses have been interpolated. Similar, by setting
    ``ages=None`` with the
    :func:`~species.read.read_isochrone.ReadIsochrone.get_isochrone`
    method will fix the ages to those of the original grid.
    """

    @typechecked
    def __init__(self, tag: str, extrapolate: bool = False) -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag of the isochrone data (e.g. 'ames-cond',
            'ames-dusty', 'sonora+0.0', 'sonora-0.5', 'sonora+0.5',
            'saumon-2008', 'nextgen').
        extrapolate : str
            Extrapolate :math:`T_\\mathrm{eff}` (K),
            :math:`\\log{(L/L_\\odot)}`, and :math:`\\log{(g)}`
            to a regular grid of masses. Please check any results
            obtained with ``extrapolate=True`` carefully there
            might be inaccuracies in the extrapolated parts of
            the parameter space.


        Returns
        -------
        NoneType
            None
        """

        self.tag = tag
        self.extrapolate = extrapolate

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.interp_method = config["species"]["interp_method"]

        with h5py.File(self.database, "r") as h5_file:
            if f"isochrones/{self.tag}" not in h5_file:
                tag_list = []
                for item in h5_file["isochrones"]:
                    tag_list.append(item)

                raise ValueError(
                    f"There is no isochrone data stored with the "
                    f"selected tag '{tag}'. The following isochrone "
                    f"tags are found in the database: {tag_list}"
                )

    @typechecked
    def _read_data(self) -> Tuple[str, np.ndarray]:
        """
        Internal function for reading the evolutionary
        data from the database.

        Returns
        -------
        str
            Model name.
        np.ndarray
            Evolutionary data. The array has 2 dimensions with the
            shape (n_data_points, 5). The columns are age (Myr),
            mass (:math:`M_\\mathrm{J}`), :math:`T_\\mathrm{eff}` (K),
            :math:`\\log{(L/L_\\odot)}`, and :math:`\\log{(g)}`.
        """

        with h5py.File(self.database, "r") as h5_file:
            model = h5_file[f"isochrones/{self.tag}/evolution"].attrs["model"]
            evolution = np.asarray(h5_file[f"isochrones/{self.tag}/evolution"])

        age_unique = np.unique(evolution[:, 0])
        mass_unique = np.unique(evolution[:, 1])

        n_ages = age_unique.shape[0]
        n_masses = mass_unique.shape[0]

        if self.extrapolate:
            evol_new = np.zeros((n_ages * n_masses, 5))

            for j, age_item in enumerate(age_unique):
                indices = evolution[:, 0] == age_item

                evol_new[j * n_masses : (j + 1) * n_masses, 0] = np.full(
                    n_masses, age_item
                )
                evol_new[j * n_masses : (j + 1) * n_masses, 1] = mass_unique

                for i in range(evolution.shape[1] - 2):
                    interp_param = interpolate.interp1d(
                        evolution[indices, 1],
                        evolution[indices, 2 + i],
                        fill_value="extrapolate",
                    )

                    evol_new[j * n_masses : (j + 1) * n_masses, 2 + i] = interp_param(
                        mass_unique
                    )

            evolution = evol_new.copy()

        return model, evolution

    @typechecked
    def get_isochrone(
        self,
        age: float,
        masses: Optional[np.ndarray] = None,
        filters_color: Optional[Tuple[str, str]] = None,
        filter_mag: Optional[str] = None,
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
        filters_color : tuple(str, str), None
            Filter names for the color as listed in the file with the
            isochrone data. Not selected if set to ``None`` or if only
            evolutionary tracks are available.
        filter_mag : str, None
            Filter name for the absolute magnitude as listed in the
            file with the isochrone data. Not selected if set to
            ``None`` or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.IsochroneBox
            Box with the isochrone.
        """

        color = None
        mag_abs = None

        index_age = 0
        index_mass = 1
        index_teff = 2
        index_log_lum = 3
        index_logg = 4

        # Read isochrone data

        model, evolution = self._read_data()

        if masses is None:
            idx_min = (np.abs(evolution[:, index_age] - age)).argmin()
            age_select = evolution[:, index_age] == evolution[idx_min, index_age]
            masses = np.unique(evolution[age_select, index_mass])  # (Mjup)

        age_points = np.full(masses.shape[0], age)  # (Myr)

        if model in ["baraffe", "phoenix", "manual"]:
            filters = self.get_filters()

            with h5py.File(self.database, "r") as h5_file:
                magnitudes = np.asarray(h5_file[f"isochrones/{self.tag}/magnitudes"])

        if model in ["baraffe", "phoenix", "manual"]:
            if filters_color is not None:
                if filters_color[0] in filters:
                    index_color_1 = filters.index(filters_color[0])

                else:
                    raise ValueError(f"Magnitudes for the selected "
                                     f"\'{filters_color[0]}\' filter "
                                     f"are not found in the "
                                     f"\'{self.tag}\' data. Please "
                                     f"select one of the following "
                                     f"filters: {filters}")

                if filters_color[1] in filters:
                    index_color_2 = filters.index(filters_color[1])

                else:
                    raise ValueError(f"Magnitudes for the selected "
                                     f"\'{filters_color[1]}\' filter "
                                     f"are not found in the "
                                     f"\'{self.tag}\' data. Please "
                                     f"select one of the following "
                                     f"filters: {filters}")

            if filter_mag is not None:
                if filter_mag in filters:
                    index_mag = filters.index(filter_mag)

                else:
                    raise ValueError(f"Magnitudes for the selected "
                                     f"\'{filter_mag}\' filter are not "
                                     f"found in the \'{self.tag}\' data. "
                                     f"Please select one of the "
                                     f"following filters: {filters}")

            if filters_color is not None:
                mag_color_1 = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_color_1],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                mag_color_2 = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_color_2],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                mag_abs = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_mag],
                    xi=np.stack((age_points, masses), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

        teff = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_teff],
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        log_lum = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_log_lum],
            xi=np.stack((age_points, masses), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        logg = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_logg],
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
            masses=masses,
        )

    @typechecked
    def get_cooling_curve(
        self,
        mass: float,
        ages: Optional[np.ndarray] = None,
        filters_color: Optional[Tuple[str, str]] = None,
        filter_mag: Optional[str] = None,
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
        filters_color : tuple(str, str), None
            Filter names for the color as listed in the file with the
            isochrone data. Not selected if set to ``None`` or if only
            evolutionary tracks are available.
        filter_mag : str, None
            Filter name for the absolute magnitude as listed in the
            file with the isochrone data. Not selected if set to
            ``None`` or if only evolutionary tracks are available.

        Returns
        -------
        species.core.box.CoolingBox
            Box with the cooling curve.
        """

        color = None
        mag_abs = None

        index_age = 0
        index_mass = 1
        index_teff = 2
        index_log_lum = 3
        index_logg = 4

        # Read isochrone data

        model, evolution = self._read_data()

        if ages is None:
            idx_min = (np.abs(evolution[:, index_mass] - mass)).argmin()
            mass_select = evolution[:, index_mass] == evolution[idx_min, index_mass]
            ages = np.unique(evolution[mass_select, index_age])  # (Myr)

        mass_points = np.full(ages.shape[0], mass)  # (Mjup)

        if model in ["baraffe", "phoenix", "manual"]:
            filters = self.get_filters()

            with h5py.File(self.database, "r") as h5_file:
                magnitudes = np.asarray(h5_file[f"isochrones/{self.tag}/magnitudes"])

        if model in ["baraffe", "phoenix", "manual"]:
            if filters_color is not None:
                index_color_1 = filters.index(filters_color[0])
                index_color_2 = filters.index(filters_color[1])

            if filter_mag is not None:
                index_mag = filters.index(filter_mag)

            if filters_color is not None:
                mag_color_1 = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_color_1],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                mag_color_2 = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_color_2],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

                color = mag_color_1 - mag_color_2

            if filter_mag is not None:
                mag_abs = interpolate.griddata(
                    points=evolution[:, :2],
                    values=magnitudes[:, index_mag],
                    xi=np.stack((ages, mass_points), axis=1),
                    method=self.interp_method,
                    fill_value="nan",
                    rescale=False,
                )

        teff = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_teff],
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        log_lum = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_log_lum],
            xi=np.stack((ages, mass_points), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        logg = interpolate.griddata(
            points=evolution[:, :2],
            values=evolution[:, index_logg],
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
        )

    @typechecked
    def get_color_magnitude(
        self,
        age: float,
        masses: np.ndarray,
        model: str,
        filters_color: Tuple[str, str],
        filter_mag: str,
        adapt_logg: bool = False,
    ) -> box.ColorMagBox:
        """
        Function for calculating color-magnitude
        combinations from a selected isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray
            Masses (:math:`M_\\mathrm{J}`) at which the isochrone
            data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
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

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the color-magnitude data.
        """

        isochrone = self.get_isochrone(
            age=age, masses=masses, filters_color=None, filter_mag=None
        )

        model1 = read_model.ReadModel(model=model, filter_name=filters_color[0])
        model2 = read_model.ReadModel(model=model, filter_name=filters_color[1])

        param_bounds = model1.get_bounds()

        if model1.get_parameters() == ["teff", "logg", "feh"]:
            if model == "sonora-bobcat":
                iso_feh = float(self.tag[-4:])
            else:
                iso_feh = 0.0

        elif model1.get_parameters() != ["teff", "logg"]:
            raise ValueError(
                "Creating synthetic colors and magnitudes from "
                "isochrones is currently only implemented for "
                "models with only Teff and log(g) as free parameters. "
                "Please contact Tomas Stolker if additional "
                "functionalities are required."
            )

        else:
            iso_feh = None

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])
        radius = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {
                "teff": isochrone.teff[i],
                "logg": isochrone.logg[i],
                "mass": mass_item,
                "distance": 10.0,
            }

            if iso_feh is not None:
                model_param["feh"] = iso_feh

            radius[i] = read_util.get_radius(
                model_param["logg"], model_param["mass"]
            )  # (Rjup)

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
                    mag1[i], _ = model1.get_magnitude(model_param)
                    mag2[i], _ = model2.get_magnitude(model_param)

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
            library=model,
            object_type="model",
            filters_color=filters_color,
            filter_mag=filter_mag,
            color=mag1 - mag2,
            magnitude=abs_mag,
            names=None,
            sptype=masses,
            mass=masses,
            radius=radius,
            iso_tag=self.tag,
        )

    @typechecked
    def get_color_color(
        self,
        age: float,
        masses: np.ndarray,
        model: str,
        filters_colors: Tuple[Tuple[str, str], Tuple[str, str]],
    ) -> box.ColorColorBox:
        """
        Function for calculating color-magnitude combinations from a
        selected isochrone.

        Parameters
        ----------
        age : float
            Age (Myr) at which the isochrone data is interpolated.
        masses : np.ndarray
            Masses (:math:`M_\\mathrm{J}`) at which the isochrone
            data is interpolated.
        model : str
            Atmospheric model used to compute the synthetic photometry.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names for the colors as listed in the file with the
            isochrone data. The filter names should be provided in the
            format of the SVO Filter Profile Service.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the color-color data.
        """

        isochrone = self.get_isochrone(
            age=age, masses=masses, filters_color=None, filter_mag=None
        )

        model1 = read_model.ReadModel(model=model, filter_name=filters_colors[0][0])
        model2 = read_model.ReadModel(model=model, filter_name=filters_colors[0][1])
        model3 = read_model.ReadModel(model=model, filter_name=filters_colors[1][0])
        model4 = read_model.ReadModel(model=model, filter_name=filters_colors[1][1])

        if model1.get_parameters() == ["teff", "logg", "feh"]:
            if model == "sonora-bobcat":
                iso_feh = float(self.tag[-4:])
            else:
                iso_feh = 0.0

        elif model1.get_parameters() != ["teff", "logg"]:
            raise ValueError(
                "Creating synthetic colors and magnitudes from "
                "isochrones is currently only implemented for "
                "models with only Teff and log(g) as free parameters. "
                "Please contact Tomas Stolker if additional "
                "functionalities are required."
            )

        else:
            iso_feh = None

        mag1 = np.zeros(isochrone.masses.shape[0])
        mag2 = np.zeros(isochrone.masses.shape[0])
        mag3 = np.zeros(isochrone.masses.shape[0])
        mag4 = np.zeros(isochrone.masses.shape[0])
        radius = np.zeros(isochrone.masses.shape[0])

        for i, mass_item in enumerate(isochrone.masses):
            model_param = {
                "teff": isochrone.teff[i],
                "logg": isochrone.logg[i],
                "mass": mass_item,
                "distance": 10.0,
            }

            if iso_feh is not None:
                model_param["feh"] = iso_feh

            radius[i] = read_util.get_radius(
                model_param["logg"], model_param["mass"]
            )  # (Rjup)

            if np.isnan(isochrone.teff[i]):
                mag1[i] = np.nan
                mag2[i] = np.nan
                mag3[i] = np.nan
                mag4[i] = np.nan

                warnings.warn(
                    f"The value of Teff is NaN for the following isochrone "
                    f"sample: {model_param}. Setting the magnitudes to NaN."
                )

            else:
                for item_bounds in model1.get_bounds():
                    if model_param[item_bounds] < model1.get_bounds()[item_bounds][0]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(
                            f"The value of {item_bounds} is "
                            f"{model_param[item_bounds]}, which is "
                            f"below the lower bound of the model grid "
                            f" ({model1.get_bounds()[item_bounds][0]}). "
                            f"Setting the magnitudes to NaN for the "
                            f"following isochrone sample: {model_param}."
                        )

                    elif model_param[item_bounds] > model1.get_bounds()[item_bounds][1]:
                        mag1[i] = np.nan
                        mag2[i] = np.nan
                        mag3[i] = np.nan
                        mag4[i] = np.nan

                        warnings.warn(
                            f"The value of {item_bounds} is "
                            f"{model_param[item_bounds]}, which is above "
                            f"the upper bound of the model grid "
                            f"({model1.get_bounds()[item_bounds][1]}). "
                            f"Setting the magnitudes to NaN for the "
                            f"following isochrone sample: {model_param}."
                        )

                if (
                    not np.isnan(mag1[i])
                    and not np.isnan(mag2[i])
                    and not np.isnan(mag3[i])
                    and not np.isnan(mag4[i])
                ):
                    mag1[i], _ = model1.get_magnitude(model_param)
                    mag2[i], _ = model2.get_magnitude(model_param)
                    mag3[i], _ = model3.get_magnitude(model_param)
                    mag4[i], _ = model4.get_magnitude(model_param)

        return box.create_box(
            boxtype="colorcolor",
            library=model,
            object_type="model",
            filters=filters_colors,
            color1=mag1 - mag2,
            color2=mag3 - mag4,
            names=None,
            sptype=masses,
            mass=masses,
            radius=radius,
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

        index_age = 0
        index_mass = 1
        index_log_lum = 3

        # Read isochrone data

        _, evolution = self._read_data()

        # Interpolate masses

        points = np.stack(
            (evolution[:, index_age], evolution[:, index_log_lum]), axis=1
        )

        age_points = np.full(log_lum.shape[0], age)  # (Myr)

        mass = interpolate.griddata(
            points=points,
            values=evolution[:, index_mass],
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

        index_age = 0
        index_log_lum = 3
        index_logg = 4

        # Read isochrone data

        _, evolution = self._read_data()

        # Interpolate masses

        mass = self.get_mass(age, log_lum)

        # Interpolate log(g)

        points = np.stack(
            (evolution[:, index_age], evolution[:, index_log_lum]), axis=1
        )

        age_points = np.full(log_lum.shape[0], age)  # (Myr)

        log_g = interpolate.griddata(
            points=points,
            values=evolution[:, index_logg],
            xi=np.stack((age_points, log_lum), axis=1),
            method=self.interp_method,
            fill_value="nan",
            rescale=False,
        )

        return read_util.get_radius(log_g, mass)

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
