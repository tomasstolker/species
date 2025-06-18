"""
Utility functions for boxes.
"""

import warnings

from typing import Dict, Optional

import numpy as np

from typeguard import typechecked

from species.core.box import ObjectBox
from species.read.read_model import ReadModel
from species.util.core_util import print_section


@typechecked
def update_objectbox(
    objectbox: ObjectBox, model_param: Dict[str, float], model: Optional[str] = None
) -> ObjectBox:
    """
    Function for updating the spectra and/or photometric fluxes in
    an :class:`~species.core.box_types.ObjectBox`, for example to
    apply a flux scaling and/or error inflation.

    Parameters
    ----------
    objectbox : species.core.box.ObjectBox
        Box with the object's data, including the spectra
        and/or photometric fluxes.
    model_param : dict
        Dictionary with the model parameters. Should contain the
        value(s) of the flux scaling and/or the error inflation.
    model : str, None
        Name of the atmospheric model. Not required when
        ``model='petitradtrans'`` because the error inflation is
        implemented differently with
        :class:`~species.fit.retrieval.AtmosphericRetrieval`.

    Returns
    -------
    species.core.box.ObjectBox
        The input box which includes the spectra with the
        scaled fluxes and/or inflated errors.
    """

    print_section("Update ObjectBox")

    if objectbox.flux is not None:
        for phot_key, phot_value in objectbox.flux.items():
            instr_name = phot_key.split(".")[0]

            if (
                f"error_{phot_key}" in model_param
                or f"log_error_{phot_key}" in model_param
            ):
                # Inflate the photometric uncertainty of a filter

                if model is None:
                    warnings.warn(
                        "The dictionary with model parameters "
                        "contains the error inflation for "
                        f"'{phot_key}' but the argument of "
                        "'model' is set to 'None'. Inflation "
                        "of the errors is therefore not possible."
                    )

                else:
                    readmodel = ReadModel(model, filter_name=phot_key)
                    model_flux = readmodel.get_flux(model_param)[0]

                    if f"error_{phot_key}" in model_param:
                        infl_factor = model_param[f"error_{phot_key}"]
                    else:
                        infl_factor = 10.0 ** model_param[f"log_error_{phot_key}"]

                    var_add = infl_factor**2 * model_flux**2

            elif (
                f"error_{instr_name}" in model_param
                or f"log_error_{instr_name}" in model_param
            ):
                # Inflate photometric uncertainty of an instrument

                if model is None:
                    warnings.warn(
                        "The dictionary with model parameters "
                        "contains the error inflation for "
                        f"'{instr_name}' but the argument of "
                        "'model' is set to 'None'. Inflation "
                        "of the errors is therefore not possible."
                    )

                    var_add = None

                else:
                    readmodel = ReadModel(model, filter_name=phot_key)
                    model_flux = readmodel.get_flux(model_param)[0]

                    if f"error_{instr_name}" in model_param:
                        infl_factor = model_param[f"error_{instr_name}"]
                    else:
                        infl_factor = 10.0 ** model_param[f"log_error_{instr_name}"]

                    var_add = infl_factor**2 * model_flux**2

            else:
                # No inflation required
                var_add = None

            if var_add is not None:
                if infl_factor < 0.01:
                    message = (
                        f"Inflating the uncertainty of {phot_key} by a "
                        + f"factor {infl_factor:.2e}..."
                    )

                else:
                    message = (
                        f"Inflating the uncertainty of {phot_key} by a "
                        + f"factor {infl_factor:.2f}..."
                    )

                print(message, end="", flush=True)

                phot_value[1] = np.sqrt(phot_value[1] ** 2 + var_add)

                print(" [DONE]")

                objectbox.flux[phot_key] = phot_value

    if objectbox.spectrum is not None:
        # Check if there are any spectra

        for spec_key, spec_value in objectbox.spectrum.items():
            # Get the spectrum (3 columns)
            spec_tmp = spec_value[0]

            if f"scaling_{spec_key}" in model_param:
                # Scale the fluxes and uncertainties of the spectrum
                scaling = model_param[f"scaling_{spec_key}"]

                if scaling < 0.01:
                    print(
                        f"Scaling the flux of {spec_key} by: {scaling:.2e}...",
                        end="",
                        flush=True,
                    )

                else:
                    print(
                        f"Scaling the flux of {spec_key} by: {scaling:.2f}...",
                        end="",
                        flush=True,
                    )

                spec_tmp[:, 1] *= model_param[f"scaling_{spec_key}"]
                spec_tmp[:, 2] *= model_param[f"scaling_{spec_key}"]

                print(" [DONE]")

            if (
                f"error_{spec_key}" in model_param
                or f"log_error_{spec_key}" in model_param
            ):
                if model is None:
                    warnings.warn(
                        "The dictionary with model parameters "
                        "contains the error inflation for "
                        f"'{spec_key}' but the argument of "
                        "'model' is set to 'None'. Inflation "
                        "of the errors is therefore not possible."
                    )

                elif model == "petitradtrans":
                    # Increase the errors by a constant value
                    add_error = 10.0 ** model_param[f"error_{spec_key}"]

                    log_msg = (
                        f"Inflating the uncertainties of {spec_key} "
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
                        spec_res=spec_value[3],
                        wavel_resample=spec_tmp[:, 0],
                    )

                    # Inflate the uncertainties relative to
                    # the fluxes of the model spectrum

                    if f"error_{spec_key}" in model_param:
                        infl_factor = model_param[f"error_{spec_key}"]
                    else:
                        infl_factor = 10.0 ** model_param[f"log_error_{spec_key}"]

                    log_msg = (
                        f"Inflating the uncertainties of {spec_key} "
                        + f"by a factor {infl_factor:.2f}..."
                    )

                    print(log_msg, end="", flush=True)

                    spec_tmp[:, 2] = np.sqrt(
                        spec_tmp[:, 2] ** 2 + (infl_factor * model_box.flux) ** 2
                    )

                    print(" [DONE]")

            # Store the spectra with the scaled fluxes and/or errors
            # The other three elements (i.e. the covariance matrix,
            # the inverted covariance matrix, and the spectral
            # resolution) remain unaffected

            objectbox.spectrum[spec_key] = (
                spec_tmp,
                spec_value[1],
                spec_value[2],
                spec_value[3],
            )

    return objectbox
