"""
Module with reading functionalities for atmospheric model spectra.
"""

import os
import warnings

from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dust_extinction.parameter_averages as dust_ext
import h5py
import numpy as np

from astropy import units as u
from typeguard import typechecked
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from spectres.spectral_resampling_numba import spectres_numba

from species.core import constants
from species.core.box import (
    ColorColorBox,
    ColorMagBox,
    ModelBox,
    PhotometryBox,
    create_box,
)
from species.data.model_data.model_spectra import add_model_grid
from species.data.spec_data.spec_vega import add_vega
from species.phot.syn_phot import SyntheticPhotometry
from species.read.read_filter import ReadFilter
from species.read.read_planck import ReadPlanck
from species.util.convert_util import logg_to_mass
from species.util.dust_util import (
    convert_to_av,
    interp_lognorm,
    interp_powerlaw,
    ism_extinction,
)
from species.util.model_util import binary_to_single, check_nearest_spec, rot_int_cmj
from species.util.spec_util import smooth_spectrum


class ReadModel:
    """
    Class for reading a model spectrum from the database.
    Extinction is applied by adding the ``ext_model`` parameter
    to the ``model_param`` dictionary of any of the ``ReadModel``
    methods. The value of ``ext_model`` should be the name of
    any of the extinction models from the ``dust-extinction``
    package (see `list of available models <https://
    dust-extinction.readthedocs.io/en/latest/dust_extinction/
    choose_model.html>`_). For example, set the value to
    ``'G23'`` to use the extinction relation from `Gordon et al.
    (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJ...950...86G>`_.
    When setting the ``ext_model``, the ``ext_av`` should be included
    in ``model_param`` to specify the visual extinction,
    :math:`A_V`, and optionally ``ext_rv``, to specify the
    reddening, :math:`R_V`.
    """

    @typechecked
    def __init__(
        self,
        model: str,
        wavel_range: Optional[Tuple[float, float]] = None,
        filter_name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model : str
            Name of the atmospheric model.
        wavel_range : tuple(float, float), None
            Wavelength range (um). Full spectrum is selected if set to
            ``None``. Not used if ``filter_name`` is not ``None``.
        filter_name : str, None
            Filter name that is used for the wavelength range. The
            ``wavel_range`` is used if set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        self.model = model

        if self.model == "bt-settl":
            warnings.warn(
                "It is recommended to use the CIFIST "
                "grid of the BT-Settl, because it is "
                "a newer version. In that case, set "
                "model='bt-settl-cifist' when using "
                "add_model of Database."
            )

        self.spectrum_interp = None
        self.wl_points = None
        self.wl_index = None

        self.filter_name = filter_name
        self.wavel_range = wavel_range

        if self.filter_name is not None:
            read_filter = ReadFilter(self.filter_name)
            self.wavel_range = read_filter.wavelength_range()
            self.mean_wavelength = read_filter.mean_wavelength()

        else:
            self.mean_wavelength = None

        if "SPECIES_CONFIG" in os.environ:
            config_file = os.environ["SPECIES_CONFIG"]
        else:
            config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = ConfigParser()
        config.read(config_file)

        self.database = config["species"]["database"]
        self.data_folder = config["species"]["data_folder"]

        self.extra_param = [
            "radius",
            "distance",
            "parallax",
            "mass",
            "log_lum",
            "log_lum_atm",
            "log_lum_disk",
            "lognorm_radius",
            "lognorm_sigma",
            "lognorm_ext",
            "ism_ext",
            "ism_red",
            "ext_model",
            "ext_av",
            "ext_rv",
            "powerlaw_max",
            "powerlaw_exp",
            "powerlaw_ext",
            "disk_teff",
            "disk_radius",
            "veil_a",
            "veil_b",
            "veil_ref",
            "vsini",
        ]

        for i in range(10):
            self.extra_param.append(f"disk_teff_{i}")
            self.extra_param.append(f"disk_radius_{i}")

        # Test if the spectra are present in the database
        hdf5_file = self.open_database()
        hdf5_file.close()

    @typechecked
    def open_database(self) -> h5py._hl.files.File:
        """
        Internal function for opening the HDF5 database.

        Returns
        -------
        h5py._hl.files.File
            The HDF5 database.
        """

        with h5py.File(self.database, "r") as hdf5_file:
            if f"models/{self.model}" in hdf5_file:
                model_found = True

            else:
                model_found = False

                warnings.warn(
                    f"The '{self.model}' model spectra are not present "
                    "in the database. Will try to add the model grid. "
                    "If this does not work (e.g. currently without an "
                    "internet connection), then please use the "
                    "'add_model' method of 'Database' to add the "
                    "grid of spectra at a later moment."
                )

        if not model_found:
            # This will not work when using multiprocessing.
            # Model spectra should be added to the database
            # before running FitModel with MPI
            with h5py.File(self.database, "a") as hdf5_file:
                add_model_grid(self.model, self.data_folder, hdf5_file)

        return h5py.File(self.database, "r")

    @typechecked
    def wavelength_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal function for extracting the wavelength points and
        indices that are used.

        Returns
        -------
        np.ndarray
            Wavelength points (um).
        np.ndarray
            Array with the size of the original wavelength grid. The
            booleans indicate if a wavelength point was used.
        """

        with self.open_database() as hdf5_file:
            wl_points = np.array(hdf5_file[f"models/{self.model}/wavelength"])

        if self.wavel_range is None:
            wl_index = np.ones(wl_points.shape[0], dtype=bool)

        else:
            wl_index = (wl_points >= self.wavel_range[0]) & (
                wl_points <= self.wavel_range[1]
            )
            index = np.where(wl_index)[0]

            # Add extra wavelength points at the boundary to make
            # sure that the wavelength range of a filter profile
            # is fully included by the model spectrum.

            # Adding 1 wavelength at both boundaries is not
            # sufficient because of the way that spectres
            # treats the edges with the resampling.

            for i in range(1, 20):
                if index[0] - i >= 0:
                    wl_index[index[0] - i] = True

                if index[-1] + i < wl_index.size:
                    wl_index[index[-1] + i] = True

        return wl_points[wl_index], wl_index

    @typechecked
    def interpolate_grid(
        self, teff_range: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Internal function for linearly interpolating the grid of
        model spectra for a requested wavelength range.

        Parameters
        ----------
        teff_range : tuple(float, float), None
            Effective temperature (K) range, (min, max) for which the
            grid will be interpolated. The full grid as stored in the
            database will be interpolated if the argument if set to
            ``None``.

        Returns
        -------
        NoneType
            None
        """

        # Get the grid points

        grid_points = self.get_points()

        # Select the required Teff points of the grid

        if "teff" in grid_points and teff_range is not None:
            teff_select = (teff_range[0] <= grid_points["teff"]) & (
                grid_points["teff"] <= teff_range[1]
            )

            # Add extra Teff points at the boundary to make sure
            # sure that the Teff prior of a fit is fully included
            # in the Teff range that is interpolated

            first_teff = np.where(teff_select)[0][0]

            if first_teff - 1 >= 0:
                teff_select[first_teff - 1] = True

            last_teff = np.where(teff_select)[0][-1]

            if last_teff + 1 < teff_select.size:
                teff_select[last_teff + 1] = True

            grid_points["teff"] = grid_points["teff"][teff_select]

        else:
            teff_select = np.ones(grid_points["teff"].size, dtype=bool)

        # Create list with grid points

        grid_points = list(grid_points.values())

        # Get the boolean array for selecting the fluxes
        # within the requested wavelength range

        if self.wl_index is None:
            self.wl_points, self.wl_index = self.wavelength_points()

        # Open de HDF5 database and read the model fluxes

        with self.open_database() as hdf5_file:
            grid_flux = np.array(hdf5_file[f"models/{self.model}/flux"])
            grid_flux = grid_flux[..., self.wl_index]
            grid_flux = grid_flux[teff_select, ...]

        # Interpolate the grid of model spectra

        self.spectrum_interp = RegularGridInterpolator(
            grid_points,
            grid_flux,
            method="linear",
            bounds_error=True,
            fill_value=np.nan,
        )

    @typechecked
    def apply_lognorm_ext(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        lognorm_radius: float,
        lognorm_sigma: float,
        lognorm_ext: float,
    ) -> np.ndarray:
        """
        Internal function for applying extinction by dust to a spectrum.

        wavelength : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        lognorm_radius : float
            Logarithm (base 10) of the mean geometric radius (um)
            of the log-normal size distribution.
        lognorm_sigma : float
            Geometric standard deviation (dimensionless) of the
            log-normal size distribution.
        lognorm_ext : float
            The extinction (mag) in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        # Interpolate cross sections as function of wavelength,
        # geometric radius, and geometric standard deviation

        dust_interp, _, _ = interp_lognorm(verbose=False)
        dust_wavel = dust_interp.grid[0]

        if wavelength[0] < dust_wavel[0]:
            raise ValueError(
                f"The shortest wavelength ({wavelength[0]:.2e} um) "
                "for which the spectrum will be calculated is smaller "
                f"than the shortest wavelength ({dust_wavel[0]:.2e} "
                "um) of the grid with dust cross sections."
            )

        if wavelength[-1] > dust_wavel[-1]:
            raise ValueError(
                f"The longest wavelength ({wavelength[-1]:.2e} um) "
                "for which the spectrum  will be calculated is "
                "larger than the longest wavelength "
                f"({dust_wavel[-1]:.2e} um) of the grid with dust "
                "cross sections."
            )

        # For each radius-sigma pair, cross sections are normalized
        # by the integrated cross section in the V-band

        cross_sections = dust_interp((wavelength, 10.0**lognorm_radius, lognorm_sigma))

        return flux * np.exp(-lognorm_ext * cross_sections)

    @typechecked
    def apply_powerlaw_ext(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        powerlaw_max: float,
        powerlaw_exp: float,
        powerlaw_ext: float,
    ) -> np.ndarray:
        """
        Internal function for applying extinction by dust to a
        spectrum.

        wavelength : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        powerlaw_max : float
            Logarithm (base 10) of the maximum radius (um)
            of the power-law size distribution.
        powerlaw_exp : float
            Exponent of the power-law size distribution.
        powerlaw_ext : float
            The extinction (mag) in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        # Interpolate cross sections as function of wavelength,
        # geometric radius, and geometric standard deviation

        dust_interp, _, _ = interp_powerlaw(verbose=False)
        dust_wavel = dust_interp.grid[0]

        if wavelength[0] < dust_wavel[0]:
            raise ValueError(
                f"The shortest wavelength ({wavelength[0]:.2e} um) for which the "
                f"spectrum will be calculated is smaller than the shortest "
                f"wavelength ({dust_wavel[0]:.2e} um) of the grid with dust cross "
                f"sections."
            )

        if wavelength[-1] > dust_wavel[-1]:
            raise ValueError(
                f"The longest wavelength ({wavelength[-1]:.2e} um) for which the "
                f"spectrum  will be calculated is larger than the longest wavelength "
                f"({dust_wavel[-1]:.2e} um) of the grid with dust cross sections."
            )

        # For each radius-sigma pair, cross sections are normalized
        # by the integrated cross section in the V-band

        cross_sections = dust_interp((wavelength, 10.0**powerlaw_max, powerlaw_exp))

        return flux * np.exp(-powerlaw_ext * cross_sections)

    @staticmethod
    @typechecked
    def apply_ext_ism(
        wavelengths: np.ndarray, flux: np.ndarray, v_band_ext: float, v_band_red: float
    ) -> np.ndarray:
        """
        Internal function for applying ISM extinction to a spectrum.

        wavelengths : np.ndarray
            Wavelengths (um) of the spectrum.
        flux : np.ndarray
            Fluxes (W m-2 um-1) of the spectrum.
        v_band_ext : float
            Extinction (mag) in the V band.
        v_band_red : float
            Reddening in the V band.

        Returns
        -------
        np.ndarray
            Fluxes (W m-2 um-1) with the extinction applied.
        """

        ext_mag = ism_extinction(v_band_ext, v_band_red, wavelengths)

        return flux * 10.0 ** (-0.4 * ext_mag)

    @typechecked
    def get_model(
        self,
        model_param: Dict[str, float],
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
        magnitude: bool = False,
        ext_filter: Optional[str] = None,
        **kwargs,
    ) -> ModelBox:
        """
        Function for extracting a model spectrum by linearly
        interpolating the model grid.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values
            should be within the boundaries of the grid. The grid
            boundaries of the spectra in the database can be obtained
            with
            :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum
            with a Gaussian kernel. No smoothing is applied if the
            argument is set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum is resampled.
            Optional smoothing with ``spec_res`` is applied for
            resampling with ``wavel_resample``. The wavelength points
            as stored in the database are used if the argument is set
            to ``None``.
        magnitude : bool
            Normalize the spectrum with a flux calibrated spectrum of
            Vega and return the magnitude instead of flux density.
        ext_filter : str, None
            Filter that is associated with the (optional) extinction
            parameter, ``ism_ext``. When the argument of ``ext_filter``
            is set to ``None``, the extinction is defined in the visual
            as usual (i.e. :math:`A_V`). By providing a filter name
            from the `SVO Filter Profile Service <http://svo2.cab.
            inta-csic.es/svo/theory/fps/>`_ as argument then the
            extinction ``ism_ext`` is defined in that filter instead
            of the $V$ band.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if "smooth" in kwargs:
            warnings.warn(
                "The 'smooth' parameter has been "
                "deprecated. Please set only the "
                "'spec_res' argument, which can be set "
                "to None for not applying a smoothing.",
                DeprecationWarning,
            )

            if not kwargs["smooth"] and spec_res is not None:
                spec_res = None

        # Check nearest grid points

        check_nearest_spec(self.model, model_param)

        # Get grid boundaries

        grid_bounds = self.get_bounds()

        # Check if all parameters are present and within the grid boundaries

        param_list = self.get_parameters()

        for key in param_list:
            if key not in model_param.keys():
                raise ValueError(
                    f"The '{key}' parameter is required by '{self.model}'. "
                    f"The mandatory parameters are {param_list}."
                )

            if model_param[key] < grid_bounds[key][0]:
                raise ValueError(
                    f"The input value of '{key}' is smaller than the lower "
                    f"boundary of the model grid ({model_param[key]} < "
                    f"{grid_bounds[key][0]})."
                )

            if model_param[key] > grid_bounds[key][1]:
                raise ValueError(
                    f"The input value of '{key}' is larger than the upper "
                    f"boundary of the model grid ({model_param[key]} > "
                    f"{grid_bounds[key][1]})."
                )

        # Print a warning if redundant parameters are included in the dictionary

        ignore_param = []

        param_list = self.get_parameters()

        for key in model_param.keys():
            if (
                key not in param_list
                and key not in self.extra_param
                and not key.startswith("phot_ext_")
            ):
                warnings.warn(
                    f"The '{key}' parameter is not required by "
                    f"'{self.model}' so the parameter will be "
                    f"ignored. The mandatory parameters are "
                    f"{param_list}."
                )

                ignore_param.append(key)

        # Interpolate the model grid

        if self.spectrum_interp is None:
            self.interpolate_grid()

        # Set the wavelength range

        if self.wavel_range is None:
            wl_points = self.get_wavelengths()
            self.wavel_range = (wl_points[0], wl_points[-1])

        # Create a list with the parameter values

        check_param = [
            "teff",
            "logg",
            "feh",
            "c_o_ratio",
            "fsed",
            "log_kzz",
            "ad_index",
            "log_co_iso",
        ]

        parameters = []
        for item in check_param:
            if item in model_param and item not in ignore_param:
                parameters.append(model_param[item])

        # Check if the ext_filter should be adjusted
        # to the name that is extracted from the
        # phot_ext_{ext_filter} parameter

        if ext_filter is None:
            for param_item in model_param:
                if param_item.startswith("phot_ext_"):
                    ext_filter = param_item[9:]

        # Interpolate the spectrum from the grid

        flux = self.spectrum_interp(parameters)[0]

        # Add the radius to the parameter dictionary if the mass if given

        if "mass" in model_param and "radius" not in model_param:
            mass = 1e3 * model_param["mass"] * constants.M_JUP  # (g)
            radius = np.sqrt(
                1e3 * constants.GRAVITY * mass / (10.0 ** model_param["logg"])
            )  # (cm)
            model_param["radius"] = 1e-2 * radius / constants.R_JUP  # (Rjup)

        # Apply (radius/distance)^2 scaling

        if "radius" in model_param and "parallax" in model_param:
            scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                1e3 * constants.PARSEC / model_param["parallax"]
            ) ** 2

            flux *= scaling

        elif "radius" in model_param and "distance" in model_param:
            scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                model_param["distance"] * constants.PARSEC
            ) ** 2

            flux *= scaling

        elif "flux_scaling" in model_param:
            flux *= model_param["flux_scaling"]

        elif "log_flux_scaling" in model_param:
            flux *= 10.0 ** model_param["log_flux_scaling"]

        # Add optional offset to the flux

        if "flux_offset" in model_param:
            flux += model_param["flux_offset"]

        # Add blackbody disk component to the spectrum

        n_disk = 0

        if "disk_teff" in model_param and "disk_radius" in model_param:
            n_disk = 1

        else:
            for disk_idx in range(100):
                if (
                    f"disk_teff_{disk_idx}" in model_param
                    and f"disk_radius_{disk_idx}" in model_param
                ):
                    n_disk += 1
                else:
                    break

        if n_disk == 1:
            readplanck = ReadPlanck(
                (0.9 * self.wavel_range[0], 1.1 * self.wavel_range[-1])
            )

            disk_param = {
                "teff": model_param["disk_teff"],
                "radius": model_param["disk_radius"],
            }

            if "parallax" in model_param:
                disk_param["parallax"] = model_param["parallax"]

            elif "distance" in model_param:
                disk_param["distance"] = model_param["distance"]

            planck_box = readplanck.get_spectrum(disk_param, spec_res=spec_res)

            flux += spectres_numba(
                self.wl_points,
                planck_box.wavelength,
                planck_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

        elif n_disk > 1:
            readplanck = ReadPlanck(
                (0.9 * self.wavel_range[0], 1.1 * self.wavel_range[-1])
            )

            for disk_idx in range(n_disk):
                disk_param = {
                    "teff": model_param[f"disk_teff_{disk_idx}"],
                    "radius": model_param[f"disk_radius_{disk_idx}"],
                }

                if "parallax" in model_param:
                    disk_param["parallax"] = model_param["parallax"]

                elif "distance" in model_param:
                    disk_param["distance"] = model_param["distance"]

                planck_box = readplanck.get_spectrum(disk_param, spec_res=spec_res)

                flux += spectres_numba(
                    self.wl_points,
                    planck_box.wavelength,
                    planck_box.flux,
                    spec_errs=None,
                    fill=np.nan,
                    verbose=True,
                )

        # Create ModelBox with the spectrum

        model_box = create_box(
            boxtype="model",
            model=self.model,
            wavelength=self.wl_points,
            flux=flux,
            parameters=model_param,
            quantity="flux",
            spec_res=spec_res,
        )

        # Apply rotational broadening vsin(i) in km/s

        if "vsini" in model_param:
            model_box.flux = rot_int_cmj(
                wavel=model_box.wavelength,
                flux=model_box.flux,
                vsini=model_param["vsini"],
                eps=0.0,
            )

        # Apply veiling

        if (
            "veil_a" in model_param
            and "veil_b" in model_param
            and "veil_ref" in model_param
        ):
            lambda_ref = 0.5  # (um)

            veil_flux = model_param["veil_ref"] + model_param["veil_b"] * (
                model_box.wavelength - lambda_ref
            )

            model_box.flux = model_param["veil_a"] * model_box.flux + veil_flux

        # Apply extinction

        if (
            "lognorm_radius" in model_param
            and "lognorm_sigma" in model_param
            and "lognorm_ext" in model_param
        ):
            model_box.flux = self.apply_lognorm_ext(
                model_box.wavelength,
                model_box.flux,
                model_param["lognorm_radius"],
                model_param["lognorm_sigma"],
                model_param["lognorm_ext"],
            )

        if (
            "powerlaw_max" in model_param
            and "powerlaw_exp" in model_param
            and "powerlaw_ext" in model_param
        ):
            model_box.flux = self.apply_powerlaw_ext(
                model_box.wavelength,
                model_box.flux,
                model_param["powerlaw_max"],
                model_param["powerlaw_exp"],
                model_param["powerlaw_ext"],
            )

        if "ism_ext" in model_param or ext_filter is not None:
            ism_reddening = model_param.get("ism_red", 3.1)

            if ext_filter is not None:
                ism_ext_av = convert_to_av(
                    filter_name=ext_filter,
                    filter_ext=model_param[f"phot_ext_{ext_filter}"],
                    v_band_red=ism_reddening,
                )

            else:
                ism_ext_av = model_param["ism_ext"]

            model_box.flux = self.apply_ext_ism(
                model_box.wavelength,
                model_box.flux,
                ism_ext_av,
                ism_reddening,
            )

        if "ext_av" in model_param:
            if "ext_model" in model_param:
                ext_model = getattr(dust_ext, model_param["ext_model"])()

                if "ext_rv" in model_param:
                    ext_model.Rv = model_param["ext_rv"]

                # Wavelength range (um) for which the extinction is defined
                ext_wavel = (1.0 / ext_model.x_range[1], 1.0 / ext_model.x_range[0])

                if (
                    model_box.wavelength[0] < ext_wavel[0]
                    or model_box.wavelength[-1] > ext_wavel[1]
                ):
                    warnings.warn(
                        "The wavelength range of the model spectrum "
                        f"({model_box.wavelength[0]:.3f}-"
                        f"{model_box.wavelength[-1]:.3f} um) "
                        "does not fully lie within the available "
                        "wavelength range of the extinction model "
                        f"({ext_wavel[0]:.3f}-{ext_wavel[1]:.3f} um). "
                        "The extinction will therefore not be applied "
                        "to fluxes of which the wavelength lies "
                        "outside the range of the extinction model."
                    )

                wavel_select = (model_box.wavelength > ext_wavel[0]) & (
                    model_box.wavelength < ext_wavel[1]
                )

                model_box.flux[wavel_select] *= ext_model.extinguish(
                    model_box.wavelength[wavel_select] * u.micron,
                    Av=model_param["ext_av"],
                )

            else:
                warnings.warn(
                    "The 'ext_av' parameter is included in the "
                    "'model_param' dictionary but the 'ext_model' "
                    "parameter is missing. Therefore, the 'ext_av' "
                    "parameter is ignored and no extinction is "
                    "applied to the spectrum."
                )

        # Apply radial velocity shift to the wavelengths

        if "rad_vel" in model_param:
            # Wavelength shift in um
            # rad_vel in km s-1 and constants.LIGHT in m s-1

            wavel_shift = (
                model_box.wavelength * 1e3 * model_param["rad_vel"] / constants.LIGHT
            )

            # Resampling will introduce a few NaNs at the edge of the
            # flux array. Resampling is needed because shifting the
            # wavelength array does not work when combining two spectra
            # of a binary system of which the two stars have different RVs.

            model_box.flux = spectres_numba(
                model_box.wavelength,
                model_box.wavelength + wavel_shift,
                model_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=False,
            )

        # Smooth the spectrum

        if spec_res is not None:
            model_box.flux = smooth_spectrum(
                model_box.wavelength, model_box.flux, spec_res
            )

        # Resample the spectrum

        if wavel_resample is not None:
            model_box.flux = spectres_numba(
                wavel_resample,
                model_box.wavelength,
                model_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

            model_box.wavelength = wavel_resample

        # Convert flux to magnitude

        if magnitude:
            with h5py.File(self.database, "r") as hdf5_file:
                # Check if the Vega spectrum is found in
                # 'r' mode because the 'a' mode is not
                # possible when using multiprocessing
                vega_found = "spectra/calibration/vega" in hdf5_file

            if not vega_found:
                with h5py.File(self.database, "a") as hdf5_file:
                    add_vega(self.data_folder, hdf5_file)

            with h5py.File(self.database, "r") as hdf5_file:
                vega_spec = np.array(hdf5_file["spectra/calibration/vega"])

            flux_vega = spectres_numba(
                model_box.wavelength,
                vega_spec[0,],
                vega_spec[1,],
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

            model_box.flux = -2.5 * np.log10(model_box.flux / flux_vega)
            model_box.quantity = "magnitude"

        # Check if the contains NaNs

        if np.isnan(np.sum(model_box.flux)):
            warnings.warn(
                "The resampled spectrum contains "
                f"{np.sum(np.isnan(model_box.flux))} "
                "NaNs, probably because the original "
                "wavelength range does not fully encompass "
                "the new wavelength range. This happened with "
                f"the following parameters: {model_param}."
            )

        # Add the luminosity to the parameter dictionary

        lum_total = 0.0

        if "radius" in model_box.parameters:
            lum_atm = (
                4.0
                * np.pi
                * (model_box.parameters["radius"] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * model_box.parameters["teff"] ** 4.0
                / constants.L_SUN
            )  # (Lsun)

            lum_total += lum_atm
            model_box.parameters["log_lum_atm"] = np.log10(lum_atm)

        # Add the blackbody disk components to the luminosity

        if n_disk == 1:
            lum_disk = (
                4.0
                * np.pi
                * (model_box.parameters["disk_radius"] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * model_box.parameters["disk_teff"] ** 4.0
                / constants.L_SUN
            )  # (Lsun)

            lum_total += lum_disk
            model_box.parameters["log_lum_disk"] = np.log10(lum_disk)

        elif n_disk > 1:
            for disk_idx in range(n_disk):
                lum_disk = (
                    4.0
                    * np.pi
                    * (
                        model_box.parameters[f"disk_radius_{disk_idx}"]
                        * constants.R_JUP
                    )
                    ** 2
                    * constants.SIGMA_SB
                    * model_box.parameters[f"disk_teff_{disk_idx}"] ** 4.0
                    / constants.L_SUN
                )  # (Lsun)

                lum_total += lum_disk
                model_box.parameters[f"log_lum_disk_{disk_idx}"] = np.log10(lum_disk)

        if lum_total > 0.0:
            model_box.parameters["log_lum"] = np.log10(lum_total)

        # Add the planet mass to the parameter dictionary

        if "radius" in model_param and "logg" in model_param:
            model_param["mass"] = logg_to_mass(
                model_param["logg"], model_param["radius"]
            )

        return model_box

    @typechecked
    def get_data(
        self,
        model_param: Dict[str, float],
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
        ext_filter: Optional[str] = None,
    ) -> ModelBox:
        """
        Function for selecting a model spectrum (without interpolation)
        for a set of parameter values that coincide with the grid
        points. The stored grid points can be inspected with
        :func:`~species.read.read_model.ReadModel.get_points`.

        Parameters
        ----------
        model_param : dict
            Model parameters and values. Only discrete values from the
            original grid are possible. Else, the nearest grid values
            are selected.
        spec_res : float, None
            Spectral resolution that is used for smoothing the spectrum
            with a Gaussian kernel. No smoothing is applied to the
            spectrum if the argument is set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum will be
            resampled. In that case, ``spec_res`` can still be used for
            smoothing the spectrum with a Gaussian kernel. The original
            wavelength points are used if the argument is set to
            ``None``.
        ext_filter : str, None
            Filter that is associated with the (optional) extinction
            parameter, ``ism_ext``. When the argument of ``ext_filter``
            is set to ``None``, the extinction is defined in the visual
            as usual (i.e. :math:`A_V`). By providing a filter name
            from the `SVO Filter Profile Service <http://svo2.cab.
            inta-csic.es/svo/theory/fps/>`_ as argument then the
            extinction ``ism_ext`` is defined in that filter instead
            of the $V$ band.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        # Check if all parameters are present

        param_list = self.get_parameters()

        for key in param_list:
            if key not in model_param.keys():
                raise ValueError(
                    f"The '{key}' parameter is required by '{self.model}'. "
                    f"The mandatory parameters are {param_list}."
                )

        # Print a warning if redundant parameters are included in the dictionary

        ignore_param = []

        for key in model_param.keys():
            if key not in param_list and key not in self.extra_param:
                warnings.warn(
                    f"The '{key}' parameter is not required by "
                    f"'{self.model}' so the parameter will be "
                    f"ignored. The mandatory parameters are "
                    f"{param_list}."
                )

                ignore_param.append(key)

        # Get wavelength points for wavelength range of
        # a filter in case filter_name was set or a
        # spectrum in case wavel_range was set

        self.wl_points, self.wl_index = self.wavelength_points()

        # Model parameters to check

        check_param = [
            "teff",
            "logg",
            "feh",
            "c_o_ratio",
            "fsed",
            "log_kzz",
            "ad_index",
            "log_co_iso",
        ]

        # Create lists with the parameter names and values

        param_key = []
        param_val = []

        for param_item in check_param:
            if param_item in model_param and param_item not in ignore_param:
                param_key.append(param_item)
                param_val.append(model_param[param_item])

        # Check if the ext_filter should be adjusted
        # to the name that is extracted from the
        # phot_ext_{ext_filter} parameter

        if ext_filter is None:
            for param_item in model_param:
                if param_item.startswith("phot_ext_"):
                    ext_filter = param_item[9:]

        # Open de HDF5 database

        with self.open_database() as hdf5_file:
            # Read the grid of fluxes from the database

            flux = np.array(hdf5_file[f"models/{self.model}/flux"])
            flux = flux[..., self.wl_index]

            # Find the indices of the grid points for which the spectrum will be extracted

            indices = []

            for i, item in enumerate(param_key):
                data = np.array(hdf5_file[f"models/{self.model}/{item}"])
                data_index = np.argwhere(
                    np.round(data, 4) == np.round(model_param[item], 4)
                )

                if len(data_index) == 0:
                    raise ValueError(
                        f"The parameter {item}={param_val[i]} is not found."
                    )

                data_index = data_index[0]

                indices.append(data_index[0])

        # Extract the spectrum at the requested grid point

        flux = flux[tuple(indices)]

        # Apply (radius/distance)^2 scaling

        if "radius" in model_param and "parallax" in model_param:
            scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                1e3 * constants.PARSEC / model_param["parallax"]
            ) ** 2

            flux *= scaling

        elif "radius" in model_param and "distance" in model_param:
            scaling = (model_param["radius"] * constants.R_JUP) ** 2 / (
                model_param["distance"] * constants.PARSEC
            ) ** 2

            flux *= scaling

        elif "flux_scaling" in model_param:
            flux *= model_param["flux_scaling"]

        elif "log_flux_scaling" in model_param:
            flux *= 10.0 ** model_param["log_flux_scaling"]

        # Add optional offset to the flux

        if "flux_offset" in model_param:
            flux += model_param["flux_offset"]

        # Add blackbody disk component to the spectrum

        n_disk = 0

        if "disk_teff" in model_param and "disk_radius" in model_param:
            n_disk = 1

        else:
            for disk_idx in range(100):
                if "disk_teff_{disk_idx}" in model_param:
                    n_disk += 1
                else:
                    break

        if n_disk == 1:
            readplanck = ReadPlanck(
                (0.9 * self.wavel_range[0], 1.1 * self.wavel_range[-1])
            )

            disk_param = {
                "teff": model_param["disk_teff"],
                "radius": model_param["disk_radius"],
            }

            if "parallax" in model_param:
                disk_param["parallax"] = model_param["parallax"]

            elif "distance" in model_param:
                disk_param["distance"] = model_param["distance"]

            planck_box = readplanck.get_spectrum(disk_param, spec_res=spec_res)

            flux += spectres_numba(
                self.wl_points,
                planck_box.wavelength,
                planck_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

        elif n_disk > 1:
            readplanck = ReadPlanck(
                (0.9 * self.wavel_range[0], 1.1 * self.wavel_range[-1])
            )

            for disk_idx in range(n_disk):
                disk_param = {
                    "teff": model_param[f"disk_teff_{disk_idx}"],
                    "radius": model_param[f"disk_radius_{disk_idx}"],
                }

                if "parallax" in model_param:
                    disk_param["parallax"] = model_param["parallax"]

                elif "distance" in model_param:
                    disk_param["distance"] = model_param["distance"]

                planck_box = readplanck.get_spectrum(disk_param, spec_res=spec_res)

                flux += spectres_numba(
                    self.wl_points,
                    planck_box.wavelength,
                    planck_box.flux,
                    spec_errs=None,
                    fill=np.nan,
                    verbose=True,
                )

        # Create ModelBox with the spectrum

        model_box = create_box(
            boxtype="model",
            model=self.model,
            wavelength=self.wl_points,
            flux=flux,
            parameters=model_param,
            quantity="flux",
            spec_res=spec_res,
        )

        # Apply rotational broadening vsin(i) in km/s

        if "vsini" in model_param:
            model_box.flux = rot_int_cmj(
                wavel=model_box.wavelength,
                flux=model_box.flux,
                vsini=model_param["vsini"],
                eps=0.0,
            )

        # Apply veiling

        if (
            "veil_a" in model_param
            and "veil_b" in model_param
            and "veil_ref" in model_param
        ):
            lambda_ref = 0.5  # (um)

            veil_flux = model_param["veil_ref"] + model_param["veil_b"] * (
                model_box.wavelength - lambda_ref
            )

            model_box.flux = model_param["veil_a"] * model_box.flux + veil_flux

        # Apply extinction

        if (
            "lognorm_radius" in model_param
            and "lognorm_sigma" in model_param
            and "lognorm_ext" in model_param
        ):
            raise NotImplementedError(
                "Log-normal dust is nog yet implemented for get_data."
            )

        if (
            "powerlaw_max" in model_param
            and "powerlaw_exp" in model_param
            and "powerlaw_ext" in model_param
        ):
            raise NotImplementedError(
                "Power-law dust is nog yet implemented for get_data."
            )

        if "ism_ext" in model_param or ext_filter is not None:
            ism_reddening = model_param.get("ism_red", 3.1)

            if ext_filter is not None:
                ism_ext_av = convert_to_av(
                    filter_name=ext_filter,
                    filter_ext=model_param[f"phot_ext_{ext_filter}"],
                    v_band_red=ism_reddening,
                )

            else:
                ism_ext_av = model_param["ism_ext"]

            model_box.flux = self.apply_ext_ism(
                model_box.wavelength,
                model_box.flux,
                ism_ext_av,
                ism_reddening,
            )

        if "ext_av" in model_param:
            if "ext_model" in model_param:
                ext_model = getattr(dust_ext, model_param["ext_model"])()

                if "ext_rv" in model_param:
                    ext_model.Rv = model_param["ext_rv"]

                # Wavelength range (um) for which the extinction is defined
                ext_wavel = (1.0 / ext_model.x_range[1], 1.0 / ext_model.x_range[0])

                if (
                    model_box.wavelength[0] < ext_wavel[0]
                    or model_box.wavelength[-1] > ext_wavel[1]
                ):
                    warnings.warn(
                        "The wavelength range of the model spectrum "
                        f"({model_box.wavelength[0]:.3f}-"
                        f"{model_box.wavelength[-1]:.3f} um) "
                        "does not fully lie within the available "
                        "wavelength range of the extinction model "
                        f"({ext_wavel[0]:.3f}-{ext_wavel[1]:.3f} um). "
                        "The extinction will therefore not be applied "
                        "to fluxes of which the wavelength lies "
                        "outside the range of the extinction model."
                    )

                wavel_select = (model_box.wavelength > ext_wavel[0]) & (
                    model_box.wavelength < ext_wavel[1]
                )

                model_box.flux[wavel_select] *= ext_model.extinguish(
                    model_box.wavelength[wavel_select] * u.micron,
                    Av=model_param["ext_av"],
                )

            else:
                warnings.warn(
                    "The 'ext_av' parameter is included in the "
                    "'model_param' dictionary but the 'ext_model' "
                    "parameter is missing. Therefore, the 'ext_av' "
                    "parameter is ignored and no extinction is "
                    "applied to the spectrum."
                )

        # Apply radial velocity shift to the wavelengths

        if "rad_vel" in model_param:
            # Wavelength shift in um
            # rad_vel in km s-1 and constants.LIGHT in m s-1

            wavel_shift = (
                model_box.wavelength * 1e3 * model_param["rad_vel"] / constants.LIGHT
            )

            # Resampling will introduce a few NaNs at the edge of the
            # flux array. Resampling is needed because shifting the
            # wavelength array does not work when combining two spectra
            # of a binary system of which the two stars have different RVs.

            model_box.flux = spectres_numba(
                model_box.wavelength,
                model_box.wavelength + wavel_shift,
                model_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=False,
            )

        # Smooth the spectrum

        if spec_res is not None:
            model_box.flux = smooth_spectrum(
                model_box.wavelength, model_box.flux, spec_res
            )

        # Resample the spectrum

        if wavel_resample is not None:
            model_box.flux = spectres_numba(
                wavel_resample,
                model_box.wavelength,
                model_box.flux,
                spec_errs=None,
                fill=np.nan,
                verbose=True,
            )

            model_box.wavelength = wavel_resample

        # Add the luminosity to the parameter dictionary

        lum_total = 0.0

        if "radius" in model_box.parameters:
            lum_atm = (
                4.0
                * np.pi
                * (model_box.parameters["radius"] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * model_box.parameters["teff"] ** 4.0
                / constants.L_SUN
            )  # (Lsun)

            lum_total += lum_atm
            model_box.parameters["log_lum_atm"] = np.log10(lum_atm)

        # Add the blackbody disk components to the luminosity

        if n_disk == 1:
            lum_disk = (
                4.0
                * np.pi
                * (model_box.parameters["disk_radius"] * constants.R_JUP) ** 2
                * constants.SIGMA_SB
                * model_box.parameters["disk_teff"] ** 4.0
                / constants.L_SUN
            )  # (Lsun)

            lum_total += lum_disk
            model_box.parameters["log_lum_disk"] = np.log10(lum_disk)

        elif n_disk > 1:
            for disk_idx in range(n_disk):
                lum_disk = (
                    4.0
                    * np.pi
                    * (
                        model_box.parameters[f"disk_radius_{disk_idx}"]
                        * constants.R_JUP
                    )
                    ** 2
                    * constants.SIGMA_SB
                    * model_box.parameters["disk_teff"] ** 4.0
                    / constants.L_SUN
                )  # (Lsun)

                lum_total += lum_disk
                model_box.parameters[f"log_lum_disk_{disk_idx}"] = np.log10(lum_disk)

        if lum_total > 0.0:
            model_box.parameters["log_lum"] = np.log10(lum_total)

        # Add the planet mass to the parameter dictionary

        if "radius" in model_param and "logg" in model_param:
            model_param["mass"] = logg_to_mass(
                model_param["logg"], model_param["radius"]
            )

        return model_box

    @typechecked
    def get_flux(
        self, model_param: Dict[str, float], synphot=None, return_box: bool = False
    ) -> Union[Tuple[Optional[float], Optional[float]], PhotometryBox]:
        """
        Function for calculating the average flux density for the
        ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Model parameters and values.
        synphot : SyntheticPhotometry, None
            Synthetic photometry object. The object is created if set
            to ``None``.
        return_box : bool
            Return a :class:`~species.core.box.PhotometryBox`
            if set to ``True`` or return the two values that are
            specified below if set to ``False``. By default, the
            argument is set to ``False``. The advantage of
            returning the output in a
            :class:`~species.core.box.PhotometryBox` is that it can
            directly be provided as input to
            :func:`~species.plot.plot_spectrum.plot_spectrum`.

        Returns
        -------
        float
            Average flux (W m-2 um-1).
        float, None
            Uncertainty (W m-2 um-1), which is set to ``None``.
        """

        param_list = self.get_parameters()

        for key in param_list:
            if key not in model_param.keys():
                raise ValueError(
                    f"The '{key}' parameter is required by '{self.model}'. "
                    f"The mandatory parameters are {param_list}."
                )

        if self.spectrum_interp is None:
            self.interpolate_grid()

        model_box = self.get_model(model_param)

        if synphot is None:
            synphot = SyntheticPhotometry(self.filter_name)

        model_flux = synphot.spectrum_to_flux(model_box.wavelength, model_box.flux)

        if return_box:
            model_mag = self.get_magnitude(model_param)

            phot_box = create_box(
                boxtype="photometry",
                name=self.model,
                wavelength=[self.mean_wavelength],
                flux=[model_flux],
                app_mag=[(model_mag[0], None)],
                abs_mag=[(model_mag[1], None)],
                filter_name=[self.filter_name],
            )

            return phot_box

        return model_flux

    @typechecked
    def get_magnitude(
        self,
        model_param: Dict[str, float],
        return_box: bool = False,
    ) -> Union[Tuple[Optional[float], Optional[float]], PhotometryBox]:
        """
        Function for calculating the apparent and absolute magnitudes
        for the ``filter_name``.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters. A ``radius`` (Rjup),
            and ``parallax`` (mas) or ``distance`` (pc) are required
            for the apparent magnitude (i.e. to scale the flux from
            the planet to the observer). Only a ``radius`` is
            required for the absolute magnitude.
        return_box : bool
            Return a :class:`~species.core.box.PhotometryBox`
            if set to ``True`` or return the two values that are
            specified below if set to ``False``. By default, the
            argument is set to ``False``. The advantage of
            returning the output in a
            :class:`~species.core.box.PhotometryBox` is that it can
            directly be provided as input to
            :func:`~species.plot.plot_spectrum.plot_spectrum`.

        Returns
        -------
        float
            Apparent magnitude. A ``None`` is returned if the
            dictionary of ``model_param`` does not contain a
            ``radius``, and ``parallax`` or ``distance``.
        float, None
            Absolute magnitude. A ``None`` is returned if the
            dictionary of ``model_param`` does not contain a
            ``radius``.
        """

        param_list = self.get_parameters()

        for key in param_list:
            if key not in model_param.keys():
                raise ValueError(
                    f"The '{key}' parameter is required by '{self.model}'. "
                    f"The mandatory parameters are {param_list}."
                )

        if self.spectrum_interp is None:
            self.interpolate_grid()

        try:
            spectrum = self.get_model(model_param)

        except ValueError:
            warnings.warn(
                f"The set of model parameters {model_param} is outside "
                f"the grid range {self.get_bounds()} so returning a NaN."
            )

            return np.nan, np.nan

        if spectrum.wavelength.size == 0:
            app_mag = (np.nan, None)
            abs_mag = (np.nan, None)

        else:
            synphot = SyntheticPhotometry(self.filter_name)

            if "radius" in model_param and "parallax" in model_param:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength,
                    spectrum.flux,
                    parallax=(model_param["parallax"], None),
                )

            elif "radius" in model_param and "distance" in model_param:
                app_mag, abs_mag = synphot.spectrum_to_magnitude(
                    spectrum.wavelength,
                    spectrum.flux,
                    distance=(model_param["distance"], None),
                )

            else:
                app_mag = (None, None)
                abs_mag = (None, None)

                if "radius" in model_param:
                    distance = 10.0  # (pc)

                    spectrum.flux *= (model_param["radius"] * constants.R_JUP) ** 2
                    spectrum.flux /= (distance * constants.PARSEC) ** 2

                    _, abs_mag = synphot.spectrum_to_magnitude(
                        spectrum.wavelength, spectrum.flux, distance=(distance, None)
                    )

        if return_box:
            model_flux = self.get_flux(model_param)

            phot_box = create_box(
                boxtype="photometry",
                name=self.model,
                wavelength=[self.mean_wavelength],
                flux=[model_flux],
                app_mag=[(app_mag[0], None)],
                abs_mag=[(abs_mag[0], None)],
                filter_name=[self.filter_name],
            )

            return phot_box

        return app_mag[0], abs_mag[0]

    @typechecked
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Function for extracting the grid boundaries.

        Returns
        -------
        dict
            Boundaries of parameter grid.
        """

        param_list = self.get_parameters()

        with self.open_database() as hdf5_file:
            bounds = {}

            for param_item in param_list:
                data = hdf5_file[f"models/{self.model}/{param_item}"]
                bounds[param_item] = (data[0], data[-1])

        return bounds

    @typechecked
    def get_wavelengths(self) -> np.ndarray:
        """
        Function for extracting the wavelength points.

        Returns
        -------
        np.ndarray
            Wavelength points (um).
        """

        with self.open_database() as hdf5_file:
            wavelength = np.array(hdf5_file[f"models/{self.model}/wavelength"])

        return wavelength

    @typechecked
    def get_points(self) -> Dict[str, np.ndarray]:
        """
        Function for extracting the grid points.

        Returns
        -------
        dict
            Parameter points of the model grid.
        """

        param_list = self.get_parameters()

        points = {}

        with self.open_database() as hdf5_file:
            points = {}

            for param_item in param_list:
                data = hdf5_file[f"models/{self.model}/{param_item}"]
                points[param_item] = np.array(data)

        return points

    @typechecked
    def get_parameters(self) -> List[str]:
        """
        Function for extracting the parameter names.

        Returns
        -------
        list(str)
            Model parameters.
        """

        with self.open_database() as hdf5_file:
            dset = hdf5_file[f"models/{self.model}"]

            if "n_param" in dset.attrs:
                n_param = dset.attrs["n_param"]
            else:
                n_param = dset.attrs["nparam"]

            param = []
            for i in range(n_param):
                param.append(dset.attrs[f"parameter{i}"])

        return param

    @typechecked
    def get_sampling(self) -> float:
        """
        Function for returning the wavelength sampling,
        :math:`\\lambda/\\Delta\\lambda`, of the
        model spectra as stored in the database.

        Returns
        -------
        float
            Wavelength sampling, :math:`\\lambda/\\Delta\\lambda`.
        """

        wavel_points = self.get_wavelengths()

        wavel_mean = (wavel_points[1:] + wavel_points[:-1]) / 2.0
        wavel_sampling = wavel_mean / np.diff(wavel_points)

        return np.mean(wavel_sampling)

    @typechecked
    def binary_spectrum(
        self,
        model_param: Dict[str, float],
        spec_res: Optional[float] = None,
        wavel_resample: Optional[np.ndarray] = None,
        **kwargs,
    ) -> ModelBox:
        """
        Function for extracting a model spectrum of a binary system.
        A weighted combination of two spectra will be returned. The
        ``model_param`` dictionary should contain the parameters
        for both components (e.g. ``teff_0`` and ``teff_1``, instead
        of ``teff``). Apart from that, the same parameters are used
        as with :func:`~species.read.read_model.ReadModel.get_model`.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values
            should be within the boundaries of the grid. The grid
            boundaries of the spectra in the database can be obtained
            with
            :func:`~species.read.read_model.ReadModel.get_bounds()`.
        spec_res : float, None
            Spectral resolution that is used for smoothing the
            spectrum with a Gaussian kernel. No smoothing is applied
            if the argument is set to ``None``.
        wavel_resample : np.ndarray, None
            Wavelength points (um) to which the spectrum is resampled.
            In that case, ``spec_res`` can still be used for smoothing
            the spectrum with a Gaussian kernel. The original
            wavelength points are used if the argument is set to
            ``None``.

        Returns
        -------
        species.core.box.ModelBox
            Box with the model spectrum.
        """

        if "smooth" in kwargs:
            warnings.warn(
                "The 'smooth' parameter has been "
                "deprecated. Please set only the "
                "'spec_res' argument, which can be set "
                "to None for not applying a smoothing.",
                DeprecationWarning,
            )

            if not kwargs["smooth"] and spec_res is not None:
                spec_res = None

        # Get grid boundaries

        param_0 = binary_to_single(model_param, 0)

        model_box_0 = self.get_model(
            param_0,
            spec_res=spec_res,
            wavel_resample=wavel_resample,
        )

        param_1 = binary_to_single(model_param, 1)

        model_box_1 = self.get_model(
            param_1,
            spec_res=spec_res,
            wavel_resample=wavel_resample,
        )

        # Weighted flux of two spectra for atmospheric asymmetries
        # Or simply the same in case of an actual binary system

        if "spec_weight" in model_param:
            flux_comb = (
                model_param["spec_weight"] * model_box_0.flux
                + (1.0 - model_param["spec_weight"]) * model_box_1.flux
            )

        else:
            flux_comb = model_box_0.flux + model_box_1.flux

        model_box = create_box(
            boxtype="model",
            model=self.model,
            wavelength=model_box_0.wavelength,
            flux=flux_comb,
            parameters=model_param,
            quantity="flux",
            spec_res=spec_res,
        )

        return model_box

    @typechecked
    def integrate_spectrum(
        self, model_param: Dict[str, float]
    ) -> Tuple[float, Optional[float]]:
        """
        Function for calculating the bolometric flux by integrating
        a model spectrum at the requested parameters. Therefore, when
        extinction is applied to the spectrum, the luminosity is the
        extinct luminosity and not the intrinsic luminosity. Without
        applying extinction, the integrated luminosity should in
        principle be the same as the luminosity calculated directly
        from the :math:`T_\\mathrm{eff}` and radius parameters, unless
        the radiative-convective model had not fully converged for a
        particular set of input parameters. It can thus be useful
        to check if the integrated luminosity is indeed consistent
        with the :math:`T_\\mathrm{eff}` of the model.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values
            should be within the boundaries of the grid. The grid
            boundaries of the spectra in the database can be obtained
            with
            :func:`~species.read.read_model.ReadModel.get_bounds()`.

        Returns
        -------
        float
            Effective temperature (K) calculated with the
            Stefan–Boltzmann law from the integrated flux.
        float, None
            Bolometric luminosity (:math:`\\log{(L/L_\\odot)}`).
            The returned value is set to ``None`` in case
            the ``model_param`` dictionary does not contain
            the radius parameter.
        """

        wavel_points = self.get_wavelengths()

        if self.wavel_range is None:
            self.wavel_range = (wavel_points[0], wavel_points[-1])

        if (
            self.wavel_range[0] != wavel_points[0]
            or self.wavel_range[1] != wavel_points[-1]
        ):
            warnings.warn(
                "The 'wavel_range' is not set to the maximum "
                "available range. To maximize the accuracy when "
                "calculating the bolometric luminosity, it is "
                "recommended to set 'wavel_range=None'."
            )

        param_copy = model_param.copy()

        if "parallax" in param_copy:
            del param_copy["parallax"]

        if "distance" in param_copy:
            del param_copy["distance"]

        model_box = self.get_model(param_copy)

        flux_int = simpson(y=model_box.flux, x=model_box.wavelength)

        if "radius" in param_copy:
            bol_lum = (
                4.0 * np.pi * (param_copy["radius"] * constants.R_JUP) ** 2 * flux_int
            )
            log_lum = np.log10(bol_lum / constants.L_SUN)
        else:
            log_lum = None

            warnings.warn(
                "Please include the 'radius' parameter in the "
                "'model_param' dictionary, if the bolometric "
                "luminosity should be calculated."
            )

        teff_int = (flux_int / constants.SIGMA_SB) ** 0.25

        return teff_int, log_lum

    @typechecked
    def create_color_magnitude(
        self,
        model_param: Dict[str, float],
        filters_color: Tuple[str, str],
        filter_mag: str,
    ) -> ColorMagBox:
        """
        Function for creating a :class:`~species.core.box.
        ColorMagBox` for a given set of filter names and model
        parameters. The effective temperature, :math:`T_\\mathrm{eff}`,
        is varied such that the returned :class:`~species.core.box.
        ColorMagBox` contains the colors as function of
        :math:`T_\\mathrm{eff}` and can be provide as input to the
        :func:`~species.plot.plot_color.plot_color_magnitude` function.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values
            should be within the boundaries of the grid. The boundaries
            of the model grid can be inspected by using the
            :func:`~species.read.read_model.ReadModel.get_bounds()`
            method. The effective temperature, :math:`T_\\mathrm{eff}`,
            does not need to be included in the dictionary since it
            is varied. The values of :math:`T_\\mathrm{eff}` are set to
            the grid points. The grid points can be inspected with the
            :func:`~species.read.read_model.ReadModel.get_points()`
            method.
        filters_color : tuple(str, str)
            Filter names that are used for the color. Any of
            the filter names from the `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_ are
            compatible.
        filter_mag : str
            Filter name that is used for the magnitude. Any of
            the filter names from the `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_ are
            compatible.

        Returns
        -------
        species.core.box.ColorMagBox
            Box with the colors and magnitudes.
        """

        if "distance" not in model_param:
            model_param["distance"] = 10.0

        if "radius" not in model_param:
            model_param["radius"] = 1.0

        if "parallax" in model_param:
            del model_param["parallax"]

        read_model_1 = ReadModel(self.model, filter_name=filters_color[0])
        read_model_2 = ReadModel(self.model, filter_name=filters_color[0])
        read_model_3 = ReadModel(self.model, filter_name=filter_mag)

        model_points = self.get_points()

        param_list = []
        color_list = []
        mag_list = []

        for param_item in model_points["teff"]:
            model_param["teff"] = param_item

            mag_1 = read_model_1.get_magnitude(model_param)
            mag_2 = read_model_2.get_magnitude(model_param)
            mag_3 = read_model_3.get_magnitude(model_param)

            param_list.append(param_item)
            color_list.append(mag_1[0] - mag_2[0])
            mag_list.append(mag_3[0])

        return create_box(
            "colormag",
            library=self.model,
            object_type="spectra",
            filters_color=filters_color,
            filter_mag=filter_mag,
            color=color_list,
            magnitude=mag_list,
            sptype=param_list,
        )

    @typechecked
    def create_color_color(
        self,
        model_param: Dict[str, float],
        filters_colors: Tuple[Tuple[str, str], Tuple[str, str]],
    ) -> ColorColorBox:
        """
        Function for creating a :class:`~species.core.box.
        ColorColorBox` for a given set of filter names and model
        parameters. The effective temperature, :math:`T_\\mathrm{eff}`,
        is varied such that the returned :class:`~species.core.box.
        ColorColorBox` contains the colors as function of
        :math:`T_\\mathrm{eff}` and can be provide as input to the
        :func:`~species.plot.plot_color.plot_color_color` function.

        Parameters
        ----------
        model_param : dict
            Dictionary with the model parameters and values. The values
            should be within the boundaries of the grid. The boundaries
            of the model grid can be inspected by using the
            :func:`~species.read.read_model.ReadModel.get_bounds()`
            method. The effective temperature, :math:`T_\\mathrm{eff}`,
            does not need to be included in the dictionary since it
            is varied. The values of :math:`T_\\mathrm{eff}` are set to
            the grid points. The grid points can be inspected with the
            :func:`~species.read.read_model.ReadModel.get_points()`
            method.
        filters_colors : tuple(tuple(str, str), tuple(str, str))
            Filter names that are used for the two colors. Any of
            the filter names from the `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_ are
            compatible.

        Returns
        -------
        species.core.box.ColorColorBox
            Box with the colors.
        """

        if "distance" not in model_param:
            model_param["distance"] = 10.0

        if "radius" not in model_param:
            model_param["radius"] = 1.0

        if "parallax" in model_param:
            del model_param["parallax"]

        read_model_1 = ReadModel(self.model, filter_name=filters_colors[0][0])
        read_model_2 = ReadModel(self.model, filter_name=filters_colors[0][1])
        read_model_3 = ReadModel(self.model, filter_name=filters_colors[1][0])
        read_model_4 = ReadModel(self.model, filter_name=filters_colors[1][1])

        model_points = self.get_points()

        param_list = []
        color_1_list = []
        color_2_list = []

        for param_item in model_points["teff"]:
            model_param["teff"] = param_item

            mag_1 = read_model_1.get_magnitude(model_param)
            mag_2 = read_model_2.get_magnitude(model_param)
            mag_3 = read_model_3.get_magnitude(model_param)
            mag_4 = read_model_4.get_magnitude(model_param)

            param_list.append(param_item)
            color_1_list.append(mag_1[0] - mag_2[0])
            color_2_list.append(mag_3[0] - mag_4[0])

        return create_box(
            "colorcolor",
            library=self.model,
            object_type="spectra",
            filters=filters_colors,
            color1=color_1_list,
            color2=color_2_list,
            sptype=param_list,
        )

    # def test_accuracy(self):
    #     with self.open_database() as hdf5_file:
    #         wl_points = np.array(hdf5_file[f"models/{self.model}/wavelength"])
    #         grid_flux = np.array(hdf5_file[f"models/{self.model}/flux"])
    #
    #     import matplotlib.pyplot as plt
    #
    #     for i in range(grid_flux.shape[0]):
    #         if i == 0:
    #             continue
    #         if i == grid_flux.shape[0]-1:
    #             continue
    #
    #         # Get the grid points
    #
    #         grid_points = self.get_points()
    #
    #         # Select the required Teff points of the grid
    #
    #         # if "teff" in grid_points and teff_range is not None:
    #         #     teff_select = (teff_range[0] <= grid_points["teff"]) & (
    #         #         grid_points["teff"] <= teff_range[1]
    #         #     )
    #         #
    #         #     grid_points["teff"] = grid_points["teff"][teff_select]
    #         #
    #         # else:
    #         #     teff_select = np.ones(grid_points["teff"].size, dtype=bool)
    #
    #         # Create list with grid points
    #
    #         grid_points = list(grid_points.values())
    #
    #         # Get the boolean array for selecting the fluxes
    #         # within the requested wavelength range
    #
    #         # if self.wl_index is None:
    #         #     self.wl_points, self.wl_index = self.wavelength_points()
    #
    #         # Open de HDF5 database and read the model fluxes
    #
    #         # with self.open_database() as hdf5_file:
    #         #     grid_flux = np.array(hdf5_file[f"models/{self.model}/flux"])
    #         #     grid_flux = grid_flux[..., self.wl_index]
    #         #     grid_flux = grid_flux[teff_select, ...]
    #
    #         grid_points_select = grid_points.copy()
    #         grid_points_select[0] = np.delete(grid_points_select[0], i)
    #
    #         grid_flux_select = np.delete(grid_flux, i, axis=0)
    #
    #         # Interpolate the grid of model spectra
    #
    #         spectrum_interp = RegularGridInterpolator(
    #             grid_points_select,
    #             grid_flux_select,
    #             method="linear",
    #             bounds_error=False,
    #             fill_value=np.nan,
    #         )
    #
    #         param_test = (grid_points[0][i], grid_points[1][0])
    #         flux_interp = spectrum_interp(param_test)
    #         flux_true = grid_flux[i, 0]
    #         diff = (flux_true-flux_interp)/flux_true
    #         # plt.figure(figsize=(10., 3.))
    #         # plt.plot(wl_points, 100.*diff, '-', lw=0.3)
    #         # plt.yscale('log')
    #         # plt.show()
    #         # exit()
    #
    #         # grid_flux = grid_flux[..., self.wl_index]
    #         # grid_flux = grid_flux[teff_select, ...]
