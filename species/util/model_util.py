"""
Utility functions for model spectra.
"""

import json
import warnings

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from PyAstronomy.pyasl import fastRotBroad
from scipy.interpolate import interp1d, RegularGridInterpolator
from typeguard import typechecked

from species.core import constants
from species.core.box import ModelBox, create_box
from species.util.dust_util import ism_extinction
from species.util.spec_util import create_wavelengths, smooth_spectrum


@typechecked
def convert_model_name(in_name: str) -> str:
    """
    Function for updating a model name for use in plots.

    Parameters
    ----------
    in_name : str
        Model name as used by species.

    Returns
    -------
    str
        Updated model name for plots.
    """

    data_file = (
        Path(__file__).parent.resolve().parents[0] / "data/model_data/model_data.json"
    )

    with open(data_file, "r", encoding="utf-8") as json_file:
        model_data = json.load(json_file)

    if in_name in model_data.keys():
        out_name = model_data[in_name]["name"]

    elif in_name == "planck":
        out_name = "Blackbody"

    else:
        out_name = in_name

        warnings.warn(
            f"The model name '{in_name}' is not known "
            "so the output name will not get adjusted "
            "for plot purposes"
        )

    return out_name


@typechecked
def powerlaw_spectrum(
    wavel_range: Union[Tuple[float, float], Tuple[np.float32, np.float32]],
    model_param: Dict[str, float],
    spec_res: float = 100.0,
) -> ModelBox:
    """
    Function for calculating a power-law spectrum. The power-law
    function is calculated in log(wavelength)-log(flux) space but
    stored in the :class:`~species.core.box.ModelBox` in linear
    wavelength-flux space.

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Tuple with the minimum and maximum wavelength (um).
    model_param : dict
        Dictionary with the model parameters. Should contain
        `'log_powerlaw_a'`, `'log_powerlaw_b'`, and `'log_powerlaw_c'`.
    spec_res : float
        Spectral resolution (default: 100).

    Returns
    -------
    species.core.box.ModelBox
        Box with the power-law spectrum.
    """

    wavel = create_wavelengths((wavel_range[0], wavel_range[1]), spec_res)

    wavel *= 1e3  # (um) -> (nm)

    log_flux = (
        model_param["log_powerlaw_a"]
        + model_param["log_powerlaw_b"]
        * np.log10(wavel) ** model_param["log_powerlaw_c"]
    )

    model_box = create_box(
        boxtype="model",
        model="powerlaw",
        wavelength=1e-3 * wavel,  # (um)
        flux=10.0**log_flux,  # (W m-2 um-1)
        parameters=model_param,
        quantity="flux",
    )

    return model_box


@typechecked
def gaussian_spectrum(
    wavel_range: Union[Tuple[float, float], Tuple[np.float32, np.float32]],
    model_param: Dict[str, float],
    spec_res: float = 100.0,
    double_gaussian: bool = False,
) -> ModelBox:
    """
    Function for calculating a Gaussian spectrum (i.e. for an emission
    line).

    Parameters
    ----------
    wavel_range : tuple(float, float)
        Tuple with the minimum and maximum wavelength (um).
    model_param : dict
        Dictionary with the model parameters. Should contain
        ``'gauss_amplitude'``, ``'gauss_mean'``, ``'gauss_sigma'``,
        and optionally ``'gauss_offset'``.
    spec_res : float
        Spectral resolution (default: 100).
    double_gaussian : bool
        Set to ``True`` for returning a double Gaussian function.
        In that case, ``model_param`` should also contain
        ``'gauss_amplitude_2'``, ``'gauss_mean_2'``, and
        ``'gauss_sigma_2'``.

    Returns
    -------
    species.core.box.ModelBox
        Box with the Gaussian spectrum.
    """

    wavel = create_wavelengths((wavel_range[0], wavel_range[1]), spec_res)

    gauss_exp = np.exp(
        -0.5
        * (wavel - model_param["gauss_mean"]) ** 2
        / model_param["gauss_sigma"] ** 2
    )

    flux = model_param["gauss_amplitude"] * gauss_exp

    if double_gaussian:
        gauss_exp = np.exp(
            -0.5
            * (wavel - model_param["gauss_mean_2"]) ** 2
            / model_param["gauss_sigma_2"] ** 2
        )

        flux += model_param["gauss_amplitude_2"] * gauss_exp

    if "gauss_offset" in model_param:
        flux += model_param["gauss_offset"]

    model_box = create_box(
        boxtype="model",
        model="gaussian",
        wavelength=wavel,
        flux=flux,
        parameters=model_param,
        quantity="flux",
    )

    return model_box


# @typechecked
# def add_luminosity(modelbox: ModelBox) -> ModelBox:
#     """
#     Function to add the luminosity of a model spectrum to the parameter
#     dictionary of the box.
#
#     Parameters
#     ----------
#     modelbox : species.core.box.ModelBox
#         Box with the model spectrum. Should also contain the dictionary
#         with the model parameters, the radius in particular.
#
#     Returns
#     -------
#     species.core.box.ModelBox
#         The input box with the luminosity added in the parameter
#         dictionary.
#     """
#
#     print("Calculating the luminosity...", end="", flush=True)
#
#     if modelbox.model == "planck":
#         readmodel = ReadPlanck(wavel_range=(1e-1, 1e3))
#         fullspec = readmodel.get_spectrum(
#             model_param=modelbox.parameters, spec_res=1000.0
#         )
#
#     else:
#         readmodel = ReadModel(modelbox.model)
#         fullspec = readmodel.get_model(modelbox.parameters)
#
#     flux = simps(fullspec.flux, fullspec.wavelength)
#
#     if "parallax" in modelbox.parameters:
#         luminosity = (
#             4.0
#             * np.pi
#             * (1e3 * constants.PARSEC / fullspec.parameters["parallax"]) ** 2
#             * flux
#         )  # (W)
#
#     elif "distance" in modelbox.parameters:
#         luminosity = (
#             4.0
#             * np.pi
#             * (fullspec.parameters["distance"] * constants.PARSEC) ** 2
#             * flux
#         )  # (W)
#
#         # Analytical solution for a single-component Planck function
#         # luminosity = 4.*np.pi*(modelbox.parameters['radius']*constants.R_JUP)**2* \
#         #     constants.SIGMA_SB*modelbox.parameters['teff']**4.
#
#     else:
#         luminosity = (
#             4.0 * np.pi * (fullspec.parameters["radius"] * constants.R_JUP) ** 2 * flux
#         )  # (W)
#
#     modelbox.parameters["luminosity"] = luminosity / constants.L_SUN  # (Lsun)
#
#     print(" [DONE]")
#
#     print(
#         f"Wavelength range (um): {fullspec.wavelength[0]:.2e} - "
#         f"{fullspec.wavelength[-1]:.2e}"
#     )
#
#     print(f"Luminosity (Lsun): {luminosity/constants.L_SUN:.2e}")
#
#     return modelbox


@typechecked
def binary_to_single(param_dict: Dict[str, float], star_index: int) -> Dict[str, float]:
    """
    Function for converting a dictionary with atmospheric parameters
    of a binary system to a dictionary of parameters for one of the
    two stars.

    Parameters
    ----------
    param_dict : dict
        Dictionary with the atmospheric parameters of both stars. The
        keywords end either with ``_0`` or ``_1`` that correspond with
        ``star_index=0`` or ``star_index=1``.
    star_index : int
        Star index (0 or 1) that is used for the parameters in
        ``param_dict``.

    Returns
    -------
    dict
        Dictionary with the parameters of the selected star.
    """

    new_dict = {}

    for param_key, param_value in param_dict.items():
        if star_index == 0 and param_key[-1] == "0":
            new_dict[param_key[:-2]] = param_value

        elif star_index == 1 and param_key[-1] == "1":
            new_dict[param_key[:-2]] = param_value

        elif param_key in [
            "teff",
            "logg",
            "feh",
            "c_o_ratio",
            "fsed",
            "radius",
            "distance",
            "parallax",
            "ism_ext",
        ]:
            new_dict[param_key] = param_value

    return new_dict


@typechecked
def extract_disk_param(
    param_dict: Dict[str, float], disk_index: Optional[int] = None
) -> Dict[str, float]:
    """
    Function for extracting the blackbody disk parameters from a
    dictionary with a mix of atmospheric and blackbody parameters.


    Parameters
    ----------
    param_dict : dict
        Dictionary with the model parameters.
    disk_index : int, None
        Disk index that is used for extracting the blackbody
        parameters from ``param_dict``. A single disk component
        will be used by setting the argument to ``None``.

    Returns
    -------
    dict
        Dictionary with the extracted blackbody parameters.
    """

    new_dict = {}

    if disk_index is None:
        new_dict["teff"] = param_dict["disk_teff"]
        new_dict["radius"] = param_dict["disk_radius"]

    else:
        new_dict["teff"] = param_dict[f"disk_teff_{disk_index}"]
        new_dict["radius"] = param_dict[f"disk_radius_{disk_index}"]

    for param_key, param_value in param_dict.items():
        if param_key in [
            "distance",
            "parallax",
            "ism_ext",
        ]:
            new_dict[param_key] = param_value

    return new_dict


@typechecked
def apply_obs(
    model_flux: np.ndarray,
    model_wavel: Optional[np.ndarray] = None,
    model_param: Optional[Dict[str, float]] = None,
    data_wavel: Optional[np.ndarray] = None,
    spec_res: Optional[float] = None,
    rot_broad: Optional[float] = None,
    rad_vel: Optional[float] = None,
    cross_sections: Optional[RegularGridInterpolator] = None,
) -> np.ndarray:
    """
    Function for post-processing of a model spectrum. This will
    apply a rotational broadening, radial velocity shift, extinction,
    scaling, instrumental broadening, and wavelength resampling.
    Each of the steps are optional, depending on the arguments that
    are set and the parameters in the ``model_param`` dictionary.

    Parameters
    ----------
    model_flux : np.ndarray
        Array with the fluxes of the model spectrum.
    model_wavel : np.ndarray, None
        Array with the wavelengths of the model spectrum. Used by
        most of the steps, except the flux scaling.
    model_param : dict(str, float), None
        Dictionary with the model parameters. Not used if the
        argument is set to ``None``.
    data_wavel : np.ndarray, None
        Array with the wavelengths of the data used for the
        wavelength resampling. Not applied if the argument is
        set to ``None``.
    spec_res : float, None
        Spectral resolution of the data used for the instrumental
        broadening. Not applied if the argument is set to ``None``.
    rot_broad : float, None
        Rotational broadening :math:`v\\sin{i}` (km/s). Not
        applied if the argument is set to ``None``.
    rad_vel : float, None
        Radial velocity (km/s). Not applied if the argument
        is set to ``None``.
    cross_sections : RegularGridInterpolator, None
        Interpolated cross sections for fitting extinction by dust
        grains with a log-normal or power-law size distribution.

    Returns
    -------
    np.ndarray
        Array with the processed model fluxes.
    """

    # Set empty dictionary if None

    if model_param is None:
        model_param = {}

    # Apply rotational broadening

    if rot_broad is not None:
        # The fastRotBroad requires constant wavelength steps
        # Upsample by a factor of 4 to not lose spectral information

        spec_interp = interp1d(model_wavel, model_flux)

        wavel_new = np.linspace(
            model_wavel[0],
            model_wavel[-1],
            4 * model_wavel.size,
        )

        flux_new = spec_interp(wavel_new)

        # Apply fast rotational broadening
        # Only to be used on a limited wavelength range

        flux_broad = fastRotBroad(
            wvl=wavel_new,
            flux=flux_new,
            epsilon=0.0,
            vsini=rot_broad,
            effWvl=None,
        )

        # Interpolate back to the original wavelength sampling

        spec_interp = interp1d(wavel_new, flux_broad)
        model_flux = spec_interp(model_wavel)

    # Apply radial velocity shift

    if rad_vel is not None:
        wavel_shift = rad_vel * 1e3 * model_wavel / constants.LIGHT

        spec_interp = interp1d(
            model_wavel + wavel_shift,
            model_flux,
            fill_value="extrapolate",
        )

        model_flux = spec_interp(model_wavel)

    # Apply extinction

    if "ism_ext" in model_param:
        ism_reddening = model_param.get("ism_red", 3.1)

        ext_filt = ism_extinction(
            model_param["ism_ext"],
            ism_reddening,
            model_wavel,
        )

        model_flux *= 10.0 ** (-0.4 * ext_filt)

    # elif "lognorm_ext" in model_param:
    #     cross_tmp = cross_sections["Generic/Bessell.V"](
    #         (10.0 ** model_param["lognorm_radius"], model_param["lognorm_sigma"])
    #     )
    #
    #     n_grains = (
    #         model_param["lognorm_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
    #     )
    #
    #     cross_tmp = cross_sections["spectrum"](
    #         (
    #             model_wavel,
    #             10.0 ** model_param["lognorm_radius"],
    #             model_param["lognorm_sigma"],
    #         )
    #     )
    #
    #     n_grains = (
    #         model_param["lognorm_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
    #     )
    #     print(n_grains)
    #
    #     model_flux *= np.exp(-cross_tmp * n_grains)
    #
    # elif "powerlaw_ext" in model_param:
    #     cross_tmp = cross_sections["Generic/Bessell.V"](
    #         (10.0 ** model_param["powerlaw_max"], model_param["powerlaw_exp"])
    #     )
    #
    #     n_grains = (
    #         model_param["powerlaw_ext"] / cross_tmp / 2.5 / np.log10(np.exp(1.0))
    #     )
    #
    #     cross_tmp = cross_sections["spectrum"](
    #         (
    #             model_wavel,
    #             10.0 ** model_param["powerlaw_max"],
    #             model_param["powerlaw_exp"],
    #         )
    #     )
    #
    #     model_flux *= np.exp(-cross_tmp * n_grains)

    # elif self.ext_filter is not None:
    #     ism_reddening = all_param.get("ism_red", 3.1)
    #
    #     av_required = convert_to_av(
    #         filter_name=self.ext_filter,
    #         filter_ext=all_param[f"phot_ext_{self.ext_filter}"],
    #         v_band_red=ism_reddening,
    #     )
    #
    #     ext_spec = ism_extinction(
    #         av_required, ism_reddening, self.spectrum[spec_item][0][:, 0]
    #     )
    #
    #     model_flux *= 10.0 ** (-0.4 * ext_spec)

    # Scale the spectrum by (radius/distance)^2

    if "radius" in model_param:
        model_flux *= (model_param["radius"] * constants.R_JUP) ** 2 / (
            1e3 * constants.PARSEC / model_param["parallax"]
        ) ** 2

    elif "flux_scaling" in model_param:
        model_flux *= model_param["flux_scaling"]

    elif "log_flux_scaling" in model_param:
        model_flux *= 10.0 ** model_param["log_flux_scaling"]

    # Apply instrument broadening

    if spec_res is not None:
        model_flux = smooth_spectrum(model_wavel, model_flux, spec_res)

    # Resample wavelengths to data

    if data_wavel is not None:
        flux_interp = interp1d(model_wavel, model_flux, bounds_error=True)
        model_flux = flux_interp(data_wavel)

    # Shift the spectrum by a constant

    # if "flux_offset" in model_param:
    #     model_flux += model_param["flux_offset"]

    return model_flux
