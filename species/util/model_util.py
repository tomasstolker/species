"""
Utility functions for model spectra.
"""

import warnings

from typing import Dict, Tuple, Union

import numpy as np

from typeguard import typechecked

from species.core.box import ModelBox, create_box
from species.util.spec_util import create_wavelengths


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

    if in_name == "drift-phoenix":
        out_name = "DRIFT-PHOENIX"

    elif in_name == "ames-cond":
        out_name = "AMES-Cond"

    elif in_name == "ames-dusty":
        out_name = "AMES-Dusty"

    elif in_name == "atmo":
        out_name = "ATMO"

    elif in_name == "atmo-ceq":
        out_name = "ATMO CEQ"

    elif in_name == "atmo-neq-weak":
        out_name = "ATMO NEQ weak"

    elif in_name == "atmo-neq-strong":
        out_name = "ATMO NEQ strong"

    elif in_name == "petrus2023":
        out_name = "ATMO (Petrus et al. 2023)"

    elif in_name == "atmo-petrus2023":
        out_name = "ATMO (Petrus et al. 2023)"

    elif in_name == "bt-cond":
        out_name = "BT-Cond"

    elif in_name == "bt-cond-feh":
        out_name = "BT-Cond"

    elif in_name == "bt-settl":
        out_name = "BT-Settl"

    elif in_name == "bt-settl-cifist":
        out_name = "BT-Settl"

    elif in_name == "bt-nextgen":
        out_name = "BT-NextGen"

    elif in_name == "bt-nextgen-subsolar":
        out_name = "BT-NextGen"

    elif in_name == "koester-wd":
        out_name = "Koester (2010)"

    elif in_name == "petitcode-cool-clear":
        out_name = "petitCODE clear"

    elif in_name == "petitcode-cool-cloudy":
        out_name = "petitCODE cloudy"

    elif in_name == "petitcode-hot-clear":
        out_name = "petitCODE clear"

    elif in_name == "petitcode-hot-cloudy":
        out_name = "petitCODE cloudy"

    elif in_name == "petitcode-linder2019-clear":
        out_name = "petitCODE clear (Linder et al. 2019)"

    elif in_name == "petitcode-linder2019-cloudy":
        out_name = "petitCODE cloudy (Linder et al. 2019)"

    elif in_name == "exo-rem":
        out_name = "Exo-REM"

    elif in_name == "exo-rem-highres":
        out_name = "Exo-REM"

    elif in_name == "planck":
        out_name = "Blackbody"

    elif in_name == "zhu2015":
        out_name = "Zhu (2015)"

    elif in_name == "saumon2008-clear":
        out_name = "Saumon & Marley (2008)"

    elif in_name == "saumon2008-cloudy":
        out_name = "Saumon & Marley (2008)"

    elif in_name == "sonora-cholla":
        out_name = "Sonora Cholla"

    elif in_name == "sonora-bobcat":
        out_name = "Sonora Bobcat"

    elif in_name == "sonora-bobcat-co":
        out_name = "Sonora Bobcat C/O"

    elif in_name == "sphinx":
        out_name = "SPHINX"

    elif in_name == "petitradtrans":
        out_name = "petitRADTRANS"

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

    for key, value in param_dict.items():
        if star_index == 0 and key[-1] == "0":
            new_dict[key[:-2]] = value

        elif star_index == 1 and key[-1] == "1":
            new_dict[key[:-2]] = value

        elif key in [
            "teff",
            "logg",
            "feh",
            "c_o_ratio",
            "fsed",
            "radius",
            "distance",
            "parallax",
        ]:
            new_dict[key] = value

    return new_dict
