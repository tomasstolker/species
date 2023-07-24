"""
Utility functions for photometry.
"""

import math
import warnings

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import spectres

from typeguard import typechecked

from species.analysis import photometry
from species.core import box
from species.read import (
    read_calibration,
    read_filter,
    read_model,
    read_planck,
    read_radtrans,
)
from species.util import read_util, retrieval_util


@typechecked
def multi_photometry(
    datatype: str,
    spectrum: str,
    filters: List[str],
    parameters: Dict[str, float],
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
) -> box.SynphotBox:
    """
    Function for calculating synthetic photometry for a list of
    filters and a specified atmosphere model and related parameters.
    This function can for example be used for calculating the
    synthetic photometry from a best-fit model spectrum. It returns
    a :class:`~species.core.box.SynphotBox` that can be provided
    as input to :func:`~species.plot.plot_spectrum.plot_spectrum`.

    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Spectrum name (e.g., 'drift-phoenix', 'bt-settl-cifist',
        planck', 'powerlaw', 'petitradtrans').
    filters : list(str)
        List with the filter names.
    parameters : dict
        Dictionary with the model parameters.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        Only required with ``spectrum='petitradtrans'`. Make sure that
        the ``wavel_range`` of the ``ReadRadtrans`` instance is
        sufficiently broad to cover all the ``filters``. The argument
        can be set to ``None`` for any other model than petitRADTRANS.

    Returns
    -------
    species.core.box.SynphotBox
        Box with synthetic photometry.
    """

    print("Calculating synthetic photometry...", end="", flush=True)

    mean_wavel = {}

    for item in filters:
        read_filt = read_filter.ReadFilter(item)
        mean_wavel[item] = read_filt.mean_wavelength

    flux = {}

    if datatype == "model":
        if spectrum == "petitradtrans":
            # Calculate the petitRADTRANS spectrum only once
            radtrans_box = radtrans.get_model(parameters)

        for item in filters:
            if spectrum == "petitradtrans":
                # Use an instance of SyntheticPhotometry instead
                # of get_flux from ReadRadtrans in order to not
                # recalculate the spectrum
                syn_phot = photometry.SyntheticPhotometry(item)

                flux[item], _ = syn_phot.spectrum_to_flux(
                    radtrans_box.wavelength, radtrans_box.flux
                )

            elif spectrum == "powerlaw":
                synphot = photometry.SyntheticPhotometry(item)

                # Set the wavel_range attribute
                synphot.zero_point()

                powerl_box = read_util.powerlaw_spectrum(
                    synphot.wavel_range, parameters
                )
                flux[item] = synphot.spectrum_to_flux(
                    powerl_box.wavelength, powerl_box.flux
                )[0]

            else:
                if spectrum == "planck":
                    readmodel = read_planck.ReadPlanck(filter_name=item)

                else:
                    readmodel = read_model.ReadModel(spectrum, filter_name=item)

                try:
                    if "teff_0" in parameters and "teff_1" in parameters:
                        # Binary system

                        param_0 = read_util.binary_to_single(parameters, 0)
                        model_flux_0 = readmodel.get_flux(param_0)[0]

                        param_1 = read_util.binary_to_single(parameters, 1)
                        model_flux_1 = readmodel.get_flux(param_1)[0]

                        flux[item] = (
                            parameters["spec_weight"] * model_flux_0
                            + (1.0 - parameters["spec_weight"]) * model_flux_1
                        )

                    else:
                        # Single object

                        flux[item] = readmodel.get_flux(parameters)[0]

                except IndexError:
                    flux[item] = np.nan

                    warnings.warn(
                        f"The wavelength range of the {item} filter does not "
                        f"match with the wavelength range of {spectrum}. The "
                        f"flux is set to NaN."
                    )

    elif datatype == "calibration":
        for item in filters:
            readcalib = read_calibration.ReadCalibration(spectrum, filter_name=item)
            flux[item] = readcalib.get_flux(parameters)[0]

    app_mag = {}
    abs_mag = {}

    for key, value in flux.items():
        syn_phot = photometry.SyntheticPhotometry(key)
        if "parallax" in parameters:
            app_mag[key], abs_mag[key] = syn_phot.flux_to_magnitude(
                flux=value, error=None, parallax=(parameters["parallax"], None)
            )

        elif "distance" in parameters:
            app_mag[key], abs_mag[key] = syn_phot.flux_to_magnitude(
                flux=value, error=None, distance=(parameters["distance"], None)
            )

        else:
            app_mag[key], abs_mag[key] = syn_phot.flux_to_magnitude(
                flux=value, error=None
            )

    print(" [DONE]")

    return box.create_box(
        "synphot",
        name="synphot",
        flux=flux,
        wavelength=mean_wavel,
        app_mag=app_mag,
        abs_mag=abs_mag,
    )


@typechecked
def apparent_to_absolute(
    app_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an apparent magnitude into an absolute
    magnitude. The uncertainty on the distance is propagated into the
    uncertainty on the absolute magnitude.

    Parameters
    ----------
    app_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Apparent magnitude and uncertainty (mag). The returned error
        on the absolute magnitude is set to None if the error on the
        apparent magnitude is set to None, for example
        ``app_mag=(15., None)``.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Distance and uncertainty (pc). The error is not propagated
        into the error on the absolute magnitude if set to None, for
        example ``distance=(20., None)``.

    Returns
    -------
    float, np.ndarray
        Absolute magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    abs_mag = app_mag[0] - 5.0 * np.log10(distance[0]) + 5.0

    if app_mag[1] is not None and distance[1] is not None:
        dist_err = distance[1] * (5.0 / (distance[0] * math.log(10.0)))
        abs_err = np.sqrt(app_mag[1] ** 2 + dist_err**2)

    elif app_mag[1] is not None and distance[1] is None:
        abs_err = app_mag[1]

    else:
        abs_err = None

    return abs_mag, abs_err


@typechecked
def absolute_to_apparent(
    abs_mag: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
    distance: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting an absolute magnitude
    into an apparent magnitude.

    Parameters
    ----------
    abs_mag : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Tuple with the absolute magnitude and uncertainty (mag).
        The uncertainty on the returned apparent magnitude is
        simply adopted from the absolute magnitude. Providing the
        uncertainty is optional and can be set to ``None``.
    distance : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Tuple with the distance and uncertainty (pc). The uncertainty
        is optional and can be set to ``None``. The distance
        uncertainty is currently not used by this function but
        included so it can be implemented at some point into the
        error budget.

    Returns
    -------
    float, np.ndarray
        Apparent magnitude (mag).
    float, np.ndarray, None
        Uncertainty (mag).
    """

    app_mag = abs_mag[0] + 5.0 * np.log10(distance[0]) - 5.0

    return app_mag, abs_mag[1]


@typechecked
def get_residuals(
    datatype: str,
    spectrum: str,
    parameters: Dict[str, float],
    objectbox: box.ObjectBox,
    inc_phot: Union[bool, List[str]] = True,
    inc_spec: Union[bool, List[str]] = True,
    radtrans: Optional[read_radtrans.ReadRadtrans] = None,
) -> box.ResidualsBox:
    """
    Function for calculating the residuals from fitting model or
    calibration spectra to a set of spectra and/or photometry.

    Parameters
    ----------
    datatype : str
        Data type ('model' or 'calibration').
    spectrum : str
        Name of the atmospheric model or calibration spectrum.
    parameters : dict
        Parameters and values for the spectrum
    objectbox : species.core.box.ObjectBox
        Box with the photometry and/or spectra of an object. A scaling
        and/or error inflation of the spectra should be applied with
        :func:`~species.util.read_util.update_objectbox` beforehand.
    inc_phot : bool, list(str)
        Include photometric data in the fit. If a boolean, either all
        (``True``) or none (``False``) of the data are selected. If a
        list, a subset of filter names (as stored in the database) can
        be provided.
    inc_spec : bool, list(str)
        Include spectroscopic data in the fit. If a boolean, either all
        (``True``) or none (``False``) of the data are selected. If a
        list, a subset of spectrum names (as stored in the database
        with :func:`~species.data.database.Database.add_object`) can be
        provided.
    radtrans : read_radtrans.ReadRadtrans, None
        Instance of :class:`~species.read.read_radtrans.ReadRadtrans`.
        Only required with ``spectrum='petitradtrans'`. Make sure that
        the ``wavel_range`` of the ``ReadRadtrans`` instance is
        sufficiently broad to cover all the photometric and
        spectroscopic data of ``inc_phot`` and ``inc_spec``. Not used
        if set to ``None``.

    Returns
    -------
    species.core.box.ResidualsBox
        Box with the residuals.
    """

    if inc_phot and objectbox.filters is not None:
        if isinstance(inc_phot, bool) and inc_phot:
            inc_phot = objectbox.filters

        model_phot = multi_photometry(
            datatype=datatype,
            spectrum=spectrum,
            filters=inc_phot,
            parameters=parameters,
            radtrans=radtrans,
        )

        res_phot = {}

        for item in inc_phot:
            transmission = read_filter.ReadFilter(item)
            res_phot[item] = np.zeros(objectbox.flux[item].shape)

            if objectbox.flux[item].ndim == 1:
                res_phot[item][0] = transmission.mean_wavelength()
                res_phot[item][1] = (
                    objectbox.flux[item][0] - model_phot.flux[item]
                ) / objectbox.flux[item][1]

            elif objectbox.flux[item].ndim == 2:
                for j in range(objectbox.flux[item].shape[1]):
                    res_phot[item][0, j] = transmission.mean_wavelength()
                    res_phot[item][1, j] = (
                        objectbox.flux[item][0, j] - model_phot.flux[item]
                    ) / objectbox.flux[item][1, j]

    else:
        res_phot = None

    if inc_spec and objectbox.spectrum is not None:
        res_spec = {}

        if spectrum == "petitradtrans":
            # Calculate the petitRADTRANS spectrum only once
            # Smoothing and resampling not with get_model
            model = radtrans.get_model(parameters)

        for key in objectbox.spectrum:
            if isinstance(inc_spec, bool) or key in inc_spec:
                wavel_range = (
                    0.9 * objectbox.spectrum[key][0][0, 0],
                    1.1 * objectbox.spectrum[key][0][-1, 0],
                )

                wl_new = objectbox.spectrum[key][0][:, 0]
                spec_res = objectbox.spectrum[key][3]

                if spectrum == "planck":
                    readmodel = read_planck.ReadPlanck(wavel_range=wavel_range)

                    model = readmodel.get_spectrum(
                        model_param=parameters, spec_res=1000.0
                    )

                    # Separate resampling to the new wavelength points

                    flux_new = spectres.spectres(
                        wl_new,
                        model.wavelength,
                        model.flux,
                        spec_errs=None,
                        fill=0.0,
                        verbose=True,
                    )

                elif spectrum == "petitradtrans":
                    # Smoothing to the instrument resolution
                    flux_smooth = retrieval_util.convolve(
                        model.wavelength, model.flux, spec_res
                    )

                    # Resampling to the new wavelength points
                    flux_new = spectres.spectres(
                        wl_new,
                        model.wavelength,
                        flux_smooth,
                        spec_errs=None,
                        fill=0.0,
                        verbose=True,
                    )

                else:
                    # Resampling to the new wavelength points
                    # is done by the get_model method

                    readmodel = read_model.ReadModel(spectrum, wavel_range=wavel_range)

                    if "teff_0" in parameters and "teff_1" in parameters:
                        # Binary system

                        param_0 = read_util.binary_to_single(parameters, 0)

                        model_spec_0 = readmodel.get_model(
                            param_0,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                        param_1 = read_util.binary_to_single(parameters, 1)

                        model_spec_1 = readmodel.get_model(
                            param_1,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                        flux_comb = (
                            parameters["spec_weight"] * model_spec_0.flux
                            + (1.0 - parameters["spec_weight"]) * model_spec_1.flux
                        )

                        model_spec = box.create_box(
                            boxtype="model",
                            model=spectrum,
                            wavelength=wl_new,
                            flux=flux_comb,
                            parameters=parameters,
                            quantity="flux",
                        )

                    else:
                        # Single object

                        model_spec = readmodel.get_model(
                            parameters,
                            spec_res=spec_res,
                            wavel_resample=wl_new,
                            smooth=True,
                        )

                    flux_new = model_spec.flux

                data_spec = objectbox.spectrum[key][0]
                res_tmp = (data_spec[:, 1] - flux_new) / data_spec[:, 2]

                res_spec[key] = np.column_stack([wl_new, res_tmp])

    else:
        res_spec = None

    print("Calculating residuals... [DONE]")

    print("Residuals (sigma):")

    if res_phot is not None:
        for item in inc_phot:
            if res_phot[item].ndim == 1:
                print(f"   - {item}: {res_phot[item][1]:.2f}")

            elif res_phot[item].ndim == 2:
                for j in range(res_phot[item].shape[1]):
                    print(f"   - {item}: {res_phot[item][1, j]:.2f}")

    if res_spec is not None:
        for key in objectbox.spectrum:
            if isinstance(inc_spec, bool) or key in inc_spec:
                print(
                    f"   - {key}: min: {np.nanmin(res_spec[key]):.2f}, "
                    f"max: {np.nanmax(res_spec[key]):.2f}"
                )

    chi2_stat = 0
    n_dof = 0

    if res_phot is not None:
        for key, value in res_phot.items():
            if value.ndim == 1:
                chi2_stat += value[1] ** 2
                n_dof += 1

            elif value.ndim == 2:
                for i in range(value.shape[1]):
                    chi2_stat += value[1][i] ** 2
                    n_dof += 1

    if res_spec is not None:
        for key, value in res_spec.items():
            count_nan = np.sum(np.isnan(value[:, 1]))
            chi2_stat += np.nansum(value[:, 1] ** 2)

            n_dof += value.shape[0]
            n_dof -= count_nan

    for item in parameters:
        if item not in ["mass", "luminosity", "distance"]:
            n_dof -= 1

    chi2_red = chi2_stat / n_dof

    print(f"Reduced chi2 = {chi2_red:.2f}")
    print(f"Number of degrees of freedom = {n_dof}")

    return box.create_box(
        boxtype="residuals",
        name=objectbox.name,
        photometry=res_phot,
        spectrum=res_spec,
        chi2_red=chi2_red,
    )


@typechecked
def parallax_to_distance(
    parallax: Union[
        Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]
    ],
) -> Union[Tuple[float, Optional[float]], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Function for converting from parallax to distance.

    Parameters
    ----------
    parallax : tuple(float, float), tuple(np.ndarray, np.ndarray)
        Parallax and optional uncertainty (mas). The
        uncertainty is not used if set to ``None``,
        for example, ``parallax=(2., None)``.

    Returns
    -------
    float, np.ndarray
        Distance (pc).
    float, np.ndarray, None
        Uncertainty (pc).
    """

    # From parallax (mas) to distance (pc)
    distance = 1e3 / parallax[0]

    if parallax[1] is None:
        distance_error = None

    else:
        distance_minus = distance - 1.0 / ((parallax[0] + parallax[1]) * 1e-3)
        distance_plus = 1.0 / ((parallax[0] - parallax[1]) * 1e-3) - distance
        distance_error = (distance_plus + distance_minus) / 2.0

    return distance, distance_error
