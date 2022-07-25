"""
Module with reading functionalities for evolutionary data.
"""

import configparser
import os
import warnings

from typing import Dict, List, Tuple

import h5py
import numpy as np

from scipy import interpolate
from typeguard import typechecked

from species.data import database


class ReadEvolution:
    """
    Class for reading evolutionary data from the database.
    """

    @typechecked
    def __init__(self, tag: str = "evolution") -> None:
        """
        Parameters
        ----------
        tag : str
            Database tag with the evolutionary data.

        Returns
        -------
        NoneType
            None
        """

        self.tag = tag

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database_path = config["species"]["database"]
        self.interp_method = config["species"]["interp_method"]

        with h5py.File(self.database_path, "r") as h5_file:
            add_grid = bool(self.tag not in h5_file)

        if add_grid:
            species_db = database.Database()
            species_db.add_evolution()

        self.interp_lbol = None
        self.interp_radius = None
        self.grid_points = None

        self.default_values = {"d_frac": 2e-05, "y_frac": (0.25), "m_core": 0.0}

    @typechecked
    def interpolate_grid(self) -> None:
        """
        Internal function for interpolating the grid of
        bolometric luminosities and radii.

        Returns
        -------
        NoneType
            None
        """

        self.grid_points = {}

        with h5py.File(self.database_path, "r") as h5_file:
            grid_lbol = np.asarray(h5_file[f"{self.tag}/grid_lbol"])
            grid_radius = np.asarray(h5_file[f"{self.tag}/grid_radius"])

            self.grid_points["age"] = np.asarray(h5_file[f"{self.tag}/points/age"])
            self.grid_points["mass"] = np.asarray(h5_file[f"{self.tag}/points/mass"])
            self.grid_points["s_i"] = np.asarray(h5_file[f"{self.tag}/points/s_i"])
            self.grid_points["d_frac"] = np.asarray(
                h5_file[f"{self.tag}/points/d_frac"]
            )
            self.grid_points["y_frac"] = np.asarray(
                h5_file[f"{self.tag}/points/y_frac"]
            )
            self.grid_points["m_core"] = np.asarray(
                h5_file[f"{self.tag}/points/m_core"]
            )

        # Change D_frac from linear to log10
        self.grid_points["d_frac"] = np.log10(self.grid_points["d_frac"])

        points = []
        for item in self.grid_points.values():
            points.append(item)

        self.interp_lbol = interpolate.RegularGridInterpolator(
            points,
            grid_lbol,
            method=self.interp_method,
            bounds_error=False,
            fill_value=np.nan,
        )

        self.interp_radius = interpolate.RegularGridInterpolator(
            points,
            grid_radius,
            method=self.interp_method,
            bounds_error=False,
            fill_value=np.nan,
        )

    @typechecked
    def get_luminosity(
        self,
        model_param: Dict[str, float],
    ) -> np.float64:
        """
        Function for interpolating the bolometric luminosity
        for a set of specified parameters. The 'age', 'mass',
        and 's_i' (initial entropy) are mandatory parameters,
        while the other parameters will be set to default
        values in case they are missing from the parameter
        dictionary.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.
            The values should be within the boundaries of the
            grid. The grid boundaries can be inspected with
            :func:`~species.read.read_evolution.ReadEvolution.get_bounds()`.

        Returns
        -------
        np.float64
            Bolometric luminosity (:math:`L_\\odot`).
        """

        parameters = self.get_parameters()

        if self.interp_lbol is None:
            self.interpolate_grid()

        param_copy = model_param.copy()

        for item in parameters:
            if item not in model_param:
                if item in self.default_values:
                    warnings.warn(
                        f"The {item} parameter is missing in "
                        f"the dictionary of 'model_param'. "
                        f"Setting {item} to a default value "
                        f"of {self.default_values[item]}."
                    )

                    param_copy[item] = self.default_values[item]

                else:
                    raise ValueError(
                        f"The {item} parameter is missing in "
                        f"the dictionary of 'model_param'."
                    )

        log_lbol = self.interp_lbol(
            [
                param_copy["age"],
                param_copy["mass"],
                param_copy["s_i"],
                np.log10(param_copy["d_frac"]),
                param_copy["y_frac"],
                param_copy["m_core"],
            ]
        )[0]

        return 10.0**log_lbol

    @typechecked
    def get_radius(
        self,
        model_param: Dict[str, float],
    ) -> np.float64:
        """
        Function for interpolating the radius for a set of
        specified parameters. The 'age', 'mass', and 's_i'
        (initial entropy) are mandatory parameters, while
        the other parameters will be set to default values
        in case they are missing from the parameter dictionary.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values.
            The values should be within the boundaries of the
            grid. The grid boundaries can be inspected with
            :func:`~species.read.read_evolution.ReadEvolution.get_bounds()`.

        Returns
        -------
        np.float64
            Radius (:math:`R_\\mathrm{J}`).
        """

        parameters = self.get_parameters()

        if self.interp_radius is None:
            self.interpolate_grid()

        param_copy = model_param.copy()

        for item in parameters:
            if item not in model_param:
                if item in self.default_values:
                    warnings.warn(
                        f"The {item} parameter is missing in "
                        f"the dictionary of 'model_param'. "
                        f"Setting {item} to a default value "
                        f"of {self.default_values[item]}."
                    )

                    param_copy[item] = self.default_values[item]

                else:
                    raise ValueError(
                        f"The {item} parameter is missing in "
                        f"the dictionary of 'model_param'."
                    )

        radius = self.interp_radius(
            [
                param_copy["age"],
                param_copy["mass"],
                param_copy["s_i"],
                np.log10(param_copy["d_frac"]),
                param_copy["y_frac"],
                param_copy["m_core"],
            ]
        )[0]

        return radius

    @typechecked
    def get_parameters(self) -> List[str]:
        """
        Function for extracting the parameter names.

        Returns
        -------
        list(str)
            Model parameters.
        """

        with h5py.File(self.database_path, "r") as h5_file:
            dset = h5_file[self.tag]
            n_param = dset.attrs["n_param"]

            param = []
            for i in range(n_param):
                param.append(dset.attrs[f"parameter{i}"])

        return param

    @typechecked
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Function for extracting the grid boundaries.

        Returns
        -------
        dict
            Boundaries of parameter grid.
        """

        parameters = self.get_parameters()

        with h5py.File(self.database_path, "r") as h5_file:
            bounds = {}

            for item in parameters:
                data = h5_file[f"{self.tag}/points/{item}"]
                bounds[item] = (data[0], data[-1])

        return bounds

    @typechecked
    def get_points(self) -> Dict[str, np.ndarray]:
        """
        Function for extracting the grid points.

        Returns
        -------
        dict
            Parameter points of the model grid.
        """

        parameters = self.get_parameters()

        with h5py.File(self.database_path, "r") as h5_file:
            points = {}

            for item in parameters:
                data = h5_file[f"{self.tag}/points/{item}"]
                points[item] = np.asarray(data)

        return points
