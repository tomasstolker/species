"""
Utility functions for boxes.
"""

import warnings

from typing import Dict, Optional

import numpy as np

from typeguard import typechecked

from species.core import constants
from species.core.box import ObjectBox
from species.read.read_model import ReadModel


@typechecked
def update_objectbox(
    objectbox: ObjectBox, model_param: Dict[str, float], model: Optional[str] = None
) -> ObjectBox:
    """
    Function for updating the spectra and/or photometric fluxes in
    an :class:`~species.core.box_types.ObjectBox`, for example by
    applying a flux scaling and/or error inflation.

    Parameters
    ----------
    objectbox : species.core.box.ObjectBox
        Box with the object's data, including the spectra
        and/or photometric fluxes.
    model_param : dict
        Dictionary with the model parameters. Should contain the
        value(s) of the flux scaling and/or the error inflation.
    model : str, None
        Name of the atmospheric model. Only required for inflating
        the errors of spectra. Otherwise, the argument can be set
        to ``None``. Not required when ``model='petitradtrans'``
        because the error inflation is implemented differently
        with :class:`~species.fit.retrieval.AtmosphericRetrieval`.

    Returns
    -------
    species.core.box.ObjectBox
        The input box which includes the spectra with the
        scaled fluxes and/or inflated errors.
    """

    if objectbox.flux is not None:
        for key, value in objectbox.flux.items():
            instr_name = key.split(".")[0]

            if f"{key}_error" in model_param:
                # Inflate the photometric uncertainty of a filter

                # Scale relative to the uncertainty
                infl_factor = model_param[f"{key}_error"]
                var_add = infl_factor**2 * value[1] ** 2

            elif f"{instr_name}_error" in model_param:
                # Inflate photometric uncertainty of an instrument

                # Scale relative to the uncertainty
                infl_factor = model_param[f"{instr_name}_error"]
                var_add = infl_factor**2 * value[1] ** 2

            else:
                # No inflation required
                var_add = None

            if var_add is not None:
                message = (
                    f"Inflating the uncertainty of {key} by a "
                    + f"factor {infl_factor:.2f} to "
                    + f"{np.sqrt(var_add):.2e} (W m-2 um-1)..."
                )

                print(message, end="", flush=True)

                value[1] = np.sqrt(value[1] ** 2 + var_add)

                print(" [DONE]")

                objectbox.flux[key] = value

    if objectbox.spectrum is not None:
        # Check if there are any spectra

        for key, value in objectbox.spectrum.items():
            # Get the spectrum (3 columns)
            spec_tmp = value[0]

            if f"scaling_{key}" in model_param:
                # Scale the flux of the spectrum
                scaling = model_param[f"scaling_{key}"]

                print(
                    f"Scaling the flux of {key} by: {scaling:.2f}...",
                    end="",
                    flush=True,
                )
                spec_tmp[:, 1] *= model_param[f"scaling_{key}"]
                print(" [DONE]")

            if f"error_{key}" in model_param:
                if model is None:
                    warnings.warn(
                        "The dictionary with model parameters "
                        f"contains the error inflation for {key} "
                        "but the argument of 'model' is set to "
                        "'None'. Inflation of the errors is "
                        "therefore not possible."
                    )

                elif model == "petitradtrans":
                    # Increase the errors by a constant value
                    add_error = 10.0 ** model_param[f"error_{key}"]
                    log_msg = (
                        f"Inflating the uncertainties of {key} "
                        + "by a constant value of "
                        + f"{add_error:.2e} (W m-2 um-1)..."
                    )

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] += add_error
                    print(" [DONE]")

                else:
                    # Calculate the model spectrum
                    wavel_range = (0.9 * spec_tmp[0, 0], 1.1 * spec_tmp[-1, 0])
                    readmodel = ReadModel(model, wavel_range=wavel_range)

                    model_box = readmodel.get_model(
                        model_param,
                        spec_res=value[3],
                        wavel_resample=spec_tmp[:, 0],
                    )

                    # Inflate the uncertainties relative to
                    # the fluxes of the model spectrum
                    infl_factor = model_param[f"error_{key}"]
                    log_msg = (
                        f"Inflating the uncertainties of {key} "
                        + "by a factor {infl_factor:.2f}..."
                    )

                    print(log_msg, end="", flush=True)
                    spec_tmp[:, 2] = np.sqrt(
                        spec_tmp[:, 2] ** 2 + (infl_factor * model_box.flux) ** 2
                    )
                    print(" [DONE]")

            if f"radvel_{key}" in model_param:
                # Shift the wavelengths of the data by
                # the radial velocity in opposite direction
                wavel_shift = (
                    -1.0
                    * model_param[f"radvel_{key}"]
                    * 1e3
                    * spec_tmp[:, 0]
                    / constants.LIGHT
                )

                mean_shift = np.mean(wavel_shift) * 1e3  # (nm)

                print(
                    f"Mean wavelength shift (nm) for {key}: {mean_shift:.2f}...",
                    end="",
                    flush=True,
                )
                spec_tmp[:, 0] += wavel_shift
                print(" [DONE]")

            # Store the spectra with the scaled fluxes and/or errors
            # The other three elements (i.e. the covariance matrix,
            # the inverted covariance matrix, and the spectral
            # resolution) remain unaffected
            objectbox.spectrum[key] = (spec_tmp, value[1], value[2], value[3])

    return objectbox
