"""
Module for adding a custom grid of model spectra to the database.
"""

import warnings

from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from spectres.spectral_resampling_numba import spectres_numba
from typeguard import typechecked

from species.util.core_util import print_section
from species.util.data_util import sort_data, write_data, add_missing
from species.util.spec_util import create_wavelengths


@typechecked
def add_custom_model_grid(
    model_name: str,
    data_path: Union[str, Path],
    parameters: List[str],
    database: h5py._hl.files.File,
    wavel_range: Optional[Tuple[float, float]],
    teff_range: Optional[Tuple[float, float]],
    wavel_sampling: Optional[float],
    fit_from: Optional[float] = None,
    extend_from: Optional[float] = None,
) -> None:
    """
    Function for adding a custom grid of model spectra to the
    database. The spectra are read from the ``data_path`` and
    should contain the ``model_name`` and ``parameters`` in
    the filenames in the following format example:
    `model-name_teff_1000_logg_4.0_feh_0.0_spec.dat`. The
    list with ``parameters`` should contain the same parameters
    as are included in the filename. Each datafile should contain
    two columns with the wavelengths in :math:`\\mu\\text{m}` and
    the fluxes in :math:`\\text{W} \\text{m}^{-2} \\mu\\text{m}^{-1}`.
    Each file should contain the same number and values of wavelengths.
    The wavelengths should be logarithmically sampled, so with a
    constant :math:`\\lambda/\\Delta\\lambda`. If not, then the
    ``wavel_range`` and ``wavel_sampling`` parameters should be used
    such that the wavelengths are resampled when reading the data
    into the ``species`` database.

    Parameters
    ----------
    model_name : str
        Name of the model grid. Should be identical to the model
        name that is included in the filenames.
    data_path : str, Path
        Path where the files with the model spectra are located.
        Either a relative or absolute path. Either a string or
        a ``Path`` object from ``pathlib``.
    parameters : list(str)
        List with the model parameters. The following parameters
        are supported: ``teff`` (for :math:`T_\\mathrm{eff}`),
        ``logg`` (for :math:`\\log\\,g`), ``feh`` (for [Fe/H]),
        ``co`` (for C/O), ``fsed`` (for :math:`f_\\mathrm{sed}`),
        ``logkzz`` (for :math:`\\log\\,K_\\mathrm{zz}`), and
        ``adindex`` (for :math:`\\gamma_\\mathrm{ad}`). Please
        contact the code maintainer if support for other parameters
        should be added.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float), None
        Wavelength range (:math:`\\mu\\text{m}`). The original
        wavelength points are used if the argument is set to ``None``.
    teff_range : tuple(float, float), None
        Effective temperature range (K) for adding a subset of the
        model grid. The full parameter grid will be added if the
        argument is set to ``None``.
    wavel_sampling : float, None
        Wavelength spacing :math:`\\lambda/\\Delta\\lambda` to which
        the spectra will be resampled. Typically this parameter is
        not needed so the argument can be set to ``None``. The only
        benefit of using this parameter is limiting the storage
        in the HDF5 database. The parameter should be used in
        combination with setting the ``wavel_range``.

    Returns
    -------
    NoneType
        None
    """

    print_section("Add custom grid of model spectra")

    if wavel_sampling is not None and wavel_range is None:
        warnings.warn(
            "The 'wavel_sampling' parameter can only be "
            "used in combination with the 'wavel_range' "
            "parameter. The model spectra are therefore "
            "not resampled."
        )

        wavel_sampling = None

    teff = []

    if "logg" in parameters:
        logg = []
    else:
        logg = None

    if "feh" in parameters:
        feh = []
    else:
        feh = None

    if "co" in parameters:
        c_o_ratio = []
        co_index = parameters.index("co")
        parameters[co_index] = "c_o_ratio"

    else:
        c_o_ratio = None

    if "fsed" in parameters:
        fsed = []
    else:
        fsed = None

    if "logkzz" in parameters:
        log_kzz = []
    else:
        log_kzz = None

    if "adindex" in parameters:
        ad_index = []
    else:
        ad_index = None

    if "log_co_iso" in parameters:
        log_co_iso = []
    else:
        log_co_iso = None

    flux = []

    if wavel_range is not None:
        print(f"Wavelength range (um) = {wavel_range[0]} - {wavel_range[1]}")

    if wavel_range is not None and wavel_sampling is not None:
        wavelength = create_wavelengths(wavel_range, wavel_sampling)
        resample_spectra = True

    else:
        wavelength = None
        resample_spectra = False

    print(f"Sampling (lambda/d_lambda) = {wavel_sampling}")

    if teff_range is not None:
        print(f"Teff range (K) = {teff_range[0]} - {teff_range[1]}")

    print_message = ""
    count = 0

    data_path = Path(data_path)
    model_files = sorted(data_path.glob("*"))

    check_plot = False

    for file_item in model_files:
        if file_item.stem[: len(model_name)] == model_name:
            file_split = file_item.stem.split("_")

            param_index = file_split.index("teff") + 1
            teff_val = float(file_split[param_index])

            if teff_range is not None:
                if teff_val < teff_range[0] or teff_val > teff_range[1]:
                    continue

            teff.append(teff_val)

            if logg is not None:
                param_index = file_split.index("logg") + 1
                logg.append(float(file_split[param_index]))

            if feh is not None:
                param_index = file_split.index("feh") + 1
                feh.append(float(file_split[param_index]))

            if c_o_ratio is not None:
                param_index = file_split.index("co") + 1
                c_o_ratio.append(float(file_split[param_index]))

            if fsed is not None:
                param_index = file_split.index("fsed") + 1
                fsed.append(float(file_split[param_index]))

            if log_kzz is not None:
                param_index = file_split.index("logkzz") + 1
                log_kzz.append(float(file_split[param_index]))

            if ad_index is not None:
                param_index = file_split.index("adindex") + 1
                ad_index.append(float(file_split[param_index]))

            if log_co_iso is not None:
                param_index = file_split.index("coiso") + 1
                log_co_iso.append(float(file_split[param_index]))

            empty_message = len(print_message) * " "
            print(f"\r{empty_message}", end="")

            print_message = f"Adding {model_name} model spectra... {file_item}"
            print(f"\r{print_message}", end="")

            if file_item.suffix == ".npy":
                data = np.load(str(file_item))
                data_wavel = data[:, 0]
                data_flux = data[:, 1]

            else:
                data_wavel, data_flux = np.loadtxt(str(file_item), unpack=True)

            if fit_from is not None:
                if fit_from > data_wavel[-1]:
                    raise ValueError(
                        "The argument of 'fit_from', "
                        f"{fit_from} um, is larger than the "
                        f"longest wavelength of the {model_name} "
                        f"model spectra, {data_wavel[-1]:.2f} um."
                    )

                def linear_func(log_wavel, a_param):
                    return a_param - 4.0 * log_wavel

                fit_select = data_wavel > fit_from

                popt_fit, _ = curve_fit(
                    linear_func,
                    np.log10(data_wavel[fit_select]),
                    np.log10(data_flux[fit_select]),
                    p0=[1.0],
                    maxfev=10000,
                )

                if extend_from is None:
                    wavel_ext = create_wavelengths(
                        (data_wavel[-1], 6000.0), wavel_sampling
                    )

                    extend_select = np.zeros(data_wavel.size, dtype=bool)

                else:
                    extend_select = data_wavel > extend_from

                    if np.sum(extend_select) == 0:
                        wavel_ext = create_wavelengths(
                            (data_wavel[-1], 6000.0), wavel_sampling
                        )

                    else:
                        wavel_ext = create_wavelengths(
                            (data_wavel[extend_select][0], 6000.0), wavel_sampling
                        )

                flux_ext = 10.0 ** linear_func(
                    np.log10(wavel_ext),
                    popt_fit[0],
                )

                wavel_combined = np.hstack((data_wavel[~extend_select], wavel_ext[1:]))
                flux_combined = np.hstack((data_flux[~extend_select], flux_ext[1:]))

                wavel_new = create_wavelengths((data_wavel[2], 5000.0), wavel_sampling)

                flux_new = spectres_numba(
                    wavel_new,
                    wavel_combined,
                    flux_combined,
                    spec_errs=None,
                    fill=np.nan,
                    verbose=True,
                )

                if not check_plot:
                    check_plot = True

                    plt.figure(figsize=(6, 3))

                    plt.plot(
                        data_wavel,
                        data_flux,
                        ls="-",
                        lw=0.6,
                        color="tab:orange",
                        label="Original",
                    )

                    plt.plot(
                        wavel_new,
                        flux_new,
                        ls="-",
                        lw=0.2,
                        color="black",
                        label="Extended",
                    )

                    plt.xscale("log")
                    plt.yscale("log")
                    plt.xlabel("Wavelength (μm)")
                    plt.ylabel("Flux (W m$^{-2}$ μm$^{-1}$)")
                    plt.legend(frameon=False)
                    plt.show()

                    user_check = input(
                        "Does the extended model spectrum seem fine? (y/n)? "
                    )

                    if user_check not in ["y", "Y", "yes", "Yes"]:
                        raise UserWarning(
                            "Please adjust the 'fit_from' and/or "
                            "'extend_from' arguments, and run "
                            "again 'add_model()'."
                        )

                data_wavel = wavel_new
                data_flux = flux_new

            if wavel_range is None:
                if wavelength is None:
                    wavelength = np.copy(data_wavel)  # (um)

                    if np.all(np.diff(wavelength) < 0):
                        raise ValueError(
                            "The wavelengths are not all sorted by increasing value."
                        )

                flux.append(data_flux)  # (W m-2 um-1)

            else:
                if not resample_spectra:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                        if np.all(np.diff(wavelength) < 0):
                            raise ValueError(
                                "The wavelengths are not all sorted by increasing value."
                            )

                        wavel_select = (wavel_range[0] < wavelength) & (
                            wavelength < wavel_range[1]
                        )
                        wavelength = wavelength[wavel_select]

                    flux.append(data_flux[wavel_select])  # (W m-2 um-1)

                else:
                    flux_interp = interp1d(data_wavel, data_flux)
                    flux_resample = flux_interp(wavelength)

                    if np.isnan(np.sum(flux_resample)):
                        raise ValueError(
                            f"Resampling is only possible if the new wavelength "
                            f"range ({wavelength[0]} - {wavelength[-1]} um) falls "
                            f"sufficiently far within the wavelength range "
                            f"({data_wavel[0]} - {data_wavel[-1]} um) of the input "
                            f"spectra."
                        )

                    flux.append(flux_resample)  # (W m-2 um-1)

            count += 1

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = f"Adding {model_name} model spectra... [DONE]"
    print(f"\r{print_message}")

    if logg is not None:
        logg = np.asarray(logg)

    if feh is not None:
        feh = np.asarray(feh)

    if c_o_ratio is not None:
        c_o_ratio = np.asarray(c_o_ratio)

    if fsed is not None:
        fsed = np.asarray(fsed)

    if log_kzz is not None:
        log_kzz = np.asarray(log_kzz)

    if ad_index is not None:
        ad_index = np.asarray(ad_index)

    if log_co_iso is not None:
        log_co_iso = np.asarray(log_co_iso)

    if wavelength is None:
        raise ValueError(
            "No files have been found. Please check "
            "the arguments of 'model', 'data_path', "
            "and 'parameters' of the add_custom_model "
            "method to make sure that the correct "
            "folder and files names are selected."
        )

    data_sorted = sort_data(
        np.asarray(teff),
        logg,
        feh,
        c_o_ratio,
        fsed,
        log_kzz,
        ad_index,
        log_co_iso,
        wavelength,
        np.asarray(flux),
    )

    if wavel_sampling is None:
        wavel_sampling = np.mean(
            0.5 * (wavelength[1:] + wavelength[:-1]) / np.diff(wavelength)
        )

    write_data(model_name, parameters, wavel_sampling, database, data_sorted)

    add_missing(model_name, parameters, database)
