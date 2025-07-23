"""
Utility functions for model spectra.
"""

import json
import warnings

from itertools import product
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from astropy import units as u

# from PyAstronomy.pyasl import fastRotBroad
from scipy.interpolate import RegularGridInterpolator
from spectres.spectral_resampling_numba import spectres_numba
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

        elif param_key[-2:] not in ["_0", "_1"]:
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
            "ext_av",
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
    ext_model: Optional[str] = None,
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
        Spectral resolution of the data used for the
        instrumental broadening. Not applied if the argument
        is set to ``None`` or np.nan.
    rot_broad : float, None
        Rotational broadening :math:`v\\sin{i}` (km/s). Not
        applied if the argument is set to ``None``.
    rad_vel : float, None
        Radial velocity (km/s). Not applied if the argument
        is set to ``None``.
    cross_sections : RegularGridInterpolator, None
        Interpolated cross sections for fitting extinction by dust
        grains with a log-normal or power-law size distribution.
    ext_model : str, None
        Name with the extinction model from the ``dust-extinction``
        package (see `list of available models
        <https://dust-extinction.readthedocs.io/en/latest/
        dust_extinction/choose_model.html>`_). For example,
        set the argument to ``'CCM89'`` to use the extinction
        relation from `Cardelli et al. (1989) <https://ui.adsabs.
        harvard.edu/abs/1989ApJ...345..245C/abstract>`_.

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
        model_flux = rot_int_cmj(
            wavel=model_wavel,
            flux=model_flux,
            vsini=rot_broad,
            eps=0.0,
        )

    # Apply extinction

    if "ism_ext" in model_param:
        ism_reddening = model_param.get("ism_red", 3.1)

        ext_filt = ism_extinction(
            model_param["ism_ext"],
            ism_reddening,
            model_wavel,
        )

        model_flux *= 10.0 ** (-0.4 * ext_filt)

    if "ext_av" in model_param:
        import dust_extinction.parameter_averages as dust_ext

        ext_object = getattr(dust_ext, ext_model)()

        if "ext_rv" in model_param:
            ext_object.Rv = model_param["ext_rv"]

        # Wavelength range (um) for which the extinction is defined
        ext_wavel = (1.0 / ext_object.x_range[1], 1.0 / ext_object.x_range[0])

        if model_wavel[0] < ext_wavel[0] or model_wavel[-1] > ext_wavel[1]:
            warnings.warn(
                "The wavelength range of the model spectrum "
                f"({model_wavel[0]:.3f}-{model_wavel[-1]:.3f} "
                "um) does not fully lie within the available "
                "wavelength range of the extinction model "
                f"({ext_wavel[0]:.3f}-{ext_wavel[1]:.3f} um). "
                "The extinction will therefore not be applied "
                "to fluxes of which the wavelength lies "
                "outside the range of the extinction model."
            )

        wavel_select = (model_wavel > ext_wavel[0]) & (model_wavel < ext_wavel[1])

        model_flux[wavel_select] *= ext_object.extinguish(
            model_wavel[wavel_select] * u.micron, Av=model_param["ext_av"]
        )

    elif "lognorm_ext" in model_param:
        cross_tmp = cross_sections(
            (
                model_wavel,
                10.0 ** model_param["lognorm_radius"],
                model_param["lognorm_sigma"],
            )
        )

        model_flux *= np.exp(-model_param["lognorm_ext"] * cross_tmp)

    elif "powerlaw_ext" in model_param:
        cross_tmp = cross_sections(
            (
                model_wavel,
                10.0 ** model_param["powerlaw_max"],
                model_param["powerlaw_exp"],
            )
        )

        model_flux *= np.exp(-model_param["powerlaw_ext"] * cross_tmp)

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

    # Apply radial velocity shift

    if rad_vel is not None:
        # Wavelength shift in um
        # rad_vel in km s-1 and constants.LIGHT in m s-1
        wavel_shift = model_wavel * 1e3 * rad_vel / constants.LIGHT

        # Resampling will introduce a few NaNs at the edge of the flux
        # array but that should not influence the fit given the total
        # number of wavelength points of a typical spectrum

        model_flux = spectres_numba(
            model_wavel,
            model_wavel + wavel_shift,
            model_flux,
            spec_errs=None,
            fill=np.nan,
            verbose=False,
        )

    # Apply instrument broadening

    if spec_res is not None and not np.isnan(spec_res):
        model_flux = smooth_spectrum(model_wavel, model_flux, spec_res)

    # Resample wavelengths to data

    if data_wavel is not None:
        # The 'fill' should not happen because model_wavel is
        # 20 wavelength points broader than data_wavel, but
        # spectres sometimes sets the outer fluxes to 'fill'
        # depending on the spectral resolution of the data

        model_flux = spectres_numba(
            data_wavel,
            model_wavel,
            model_flux,
            spec_errs=None,
            fill=np.nan,
            verbose=False,
        )

        # Set the flux to the neighboring wavelength
        # if a NaN is present at the edge.
        # TODO this is quick hack and not the best solution

        if np.isnan(model_flux[0]):
            model_flux[0] = model_flux[1]

        if np.isnan(model_flux[-1]):
            model_flux[0] = model_flux[-2]

    # Shift the spectrum by a constant

    # if "flux_offset" in model_param:
    #     model_flux += model_param["flux_offset"]

    return model_flux


@typechecked
def rot_int_cmj(
    wavel: np.ndarray,
    flux: np.ndarray,
    vsini: float,
    eps: float = 0.6,
    nr: int = 10,
    ntheta: int = 100,
    dif: float = 0.0,
):
    """
    A routine to quickly rotationally broaden a spectrum in linear time.
    This function has been adopted from `Carvalho & Johns-Krull (2023)
    <https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...91C/abstract>`_.

    Parameters
    ----------
    wavel : np.ndarray
        Array with the wavelengths.
    flux : np.ndarray
        Array with the fluxes
    vsini : float
        Projected rotational velocity (km s-1).
    eps : float
        Coefficient of the limb darkening law (default: 0.0).
    nr : int
        Number of radial bins on the projected disk (default: 10).
    ntheta : int
        Number of azimuthal bins in the largest radial annulus
        (default: 100). Note: the number of bins at each r is
        int(r*ntheta) where r < 1.
    dif : float
        Differential rotation coefficient (default = 0.0), applied
        according to the law Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2)
        cos(2 th)). Dif = 0.675 reproduces the law proposed by Smith,
        1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS.
        Dif = 0.23 is similar to observed solar differential rotation.
        Note: the th in the expression above is the stellar co-latitude,
        which is not the same as the integration variable used in the
        function. This is a disk integration routine.

    Returns
    -------
    np.ndarray
        Array with the rotationally broadened spectrum.
    """

    ns = np.zeros(flux.shape)
    tarea = 0.0
    dr = 1.0 / nr

    for j in range(nr):
        r = dr / 2.0 + j * dr
        area = (
            ((r + dr / 2.0) ** 2 - (r - dr / 2.0) ** 2)
            / int(ntheta * r)
            * (1.0 - eps + eps * np.cos(np.arcsin(r)))
        )

        for k in range(int(ntheta * r)):
            th = np.pi / int(ntheta * r) + k * 2.0 * np.pi / int(ntheta * r)

            if dif != 0:
                vl = (
                    vsini
                    * r
                    * np.sin(th)
                    * (
                        1.0
                        - dif / 2.0
                        - dif / 2.0 * np.cos(2.0 * np.arccos(r * np.cos(th)))
                    )
                )
            else:
                vl = r * vsini * np.sin(th)

            ns += area * np.interp(
                wavel + wavel * vl / (1e-3 * constants.LIGHT), wavel, flux
            )
            tarea += area

    return ns / tarea


@typechecked
def check_nearest_spec(model_name: str, model_param: Dict[str, float]):
    """
    Check if the nearest grid points of the requested model parameters
    have a spectrum stored in the database. For some grids, spectra
    are missing for certain parameters, in which case a spectrum with
    zero fluxes has been stored in the database. Interpolating from
    such a grid point will therefore give an inaccurate spectrum, so
    it is important to check if for example the best-fit parameters
    from a fit are close to grid points with a missing spectrum.

    Parameters
    ----------
    model_name : str
        Name of the atmosphere model.
    model_param : dict
        Dictionary with the model parameters.

    Returns
    -------
    NoneType
        None
    """

    from species.read.read_model import ReadModel

    read_model = ReadModel(model_name)
    model_points = read_model.get_points()

    near_low = {}
    near_high = {}
    param_idx = []
    for param_key, param_value in model_points.items():
        near_idx = np.argsort(np.abs(model_param[param_key] - param_value))
        near_low[param_key] = param_value[near_idx[0]]
        near_high[param_key] = param_value[near_idx[1]]
        param_idx.append((near_idx[0], near_idx[1]))

    for comb_item in list(product(*param_idx)):
        model_param = {}
        for param_idx, param_key in enumerate(model_points):
            model_param[param_key] = model_points[param_key][comb_item[param_idx]]

        model_box = read_model.get_data(model_param)

        if np.count_nonzero(model_box.flux) == 0:
            warnings.warn(
                "The selected parameters have a nearest grid point for "
                f"which a spectrum is not available: {model_param}. "
                "Therefore, zeros had been stored for the spectrum at "
                "this grid point. Interpolating from this grid point "
                "will therefore be inaccurate. When using FitModel, it "
                "is best to adjust the prior range in the 'bounds' "
                "parameter accordingly to exclude the parameter space "
                "for which model spectrum is missing. See also details "
                "printed when the model spectra are added to the "
                "database with add_model()."
            )
