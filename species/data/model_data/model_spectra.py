"""
Module for adding a grid of model spectra to the database.
"""

import hashlib
import json
import tarfile
import warnings

from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pooch

from scipy.optimize import curve_fit
from spectres.spectral_resampling_numba import spectres_numba
from typeguard import typechecked

from species.util.core_util import print_section
from species.util.data_util import add_missing, extract_tarfile, sort_data, write_data
from species.util.model_util import convert_model_name
from species.util.spec_util import create_wavelengths


@typechecked
def add_model_grid(
    model_tag: str,
    input_path: str,
    database: h5py._hl.files.File,
    wavel_range: Optional[Tuple[float, float]] = None,
    teff_range: Optional[Tuple[float, float]] = None,
    wavel_sampling: Optional[float] = None,
    unpack_tar: bool = True,
    fit_from: Optional[float] = None,
    extend_from: Optional[float] = None,
) -> None:
    """
    Function for adding a grid of model spectra to the database.
    The original spectra had been resampled to logarithmically-
    spaced wavelengths, so with at a constant
    :math:`\\lambda/\\Delta\\lambda`. This function downloads
    the model grid, unpacks the tar file, and adds the spectra
    and parameters to the database.

    Parameters
    ----------
    model_tag : str
        Tag of the grid of model spectra.
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        HDF5 database.
    wavel_range : tuple(float, float), None
        Wavelength range (:math:`\\mu\\text{m}`). The original
        wavelength points are used if the argument is set to ``None``.
    teff_range : tuple(float, float), None
        Range of effective temperatures (K) for which the spectra will
        be extracted from the TAR file and added to the database. All
        spectra are selected if the argument is set to ``None``.
    wavel_sampling : float, None
        Wavelength spacing :math:`\\lambda/\\Delta\\lambda` to which
        the spectra will be resampled. Typically this parameter is
        not needed so the argument can be set to ``None``. The only
        benefit of using this parameter is limiting the storage
        in the HDF5 database. The parameter should be used in
        combination with setting the ``wavel_range``.
    unpack_tar : bool
        Unpack the TAR file with the model spectra in the
        ``data_folder``. The argument can be set to ``False`` if
        the TAR file had already been unpacked previously.
    fit_from : float, None
        Extend the spectra with a Rayleigh-Jeans slope. To do so, the
        red end of the spectra will be fitted by setting ``fit_from``
        to the minimum wavelength (in um) beyond which fluxes will be
        included in the least-squares fit. The spectra are not
        extended when setting the argument to ``None``.
    extend_from : float, None
        This parameter can be used in combination with ``fit_from``.
        The argument of ``extend_from`` is the minimum wavelength
        (in um) from which the spectra will be extended with the
        Rayleigh-Jeans slope. The spectra will be extended from the
        last available wavelength when setting the argument to
        ``None``. Typically, the value of ``fit_from`` will be
        smaller than the value of ``extend_from``.

    Returns
    -------
    NoneType
        None
    """

    print_section("Add grid of model spectra")

    model_name = convert_model_name(model_tag)
    print(f"Database tag: {model_tag}")
    print(f"Model name: {model_name}")

    data_file = Path(__file__).parent.resolve() / "model_data.json"

    with open(data_file, "r", encoding="utf-8") as json_file:
        model_data = json.load(json_file)

    if model_tag in model_data.keys():
        model_info = model_data[model_tag]

    else:
        raise ValueError(
            f"The '{model_tag}' atmospheric model is "
            "not available. Please choose one of the "
            f"following models: {list(model_data.keys())}"
        )

    if model_tag == "bt-settl":
        warnings.warn(
            "It is recommended to use the CIFIST "
            "grid of the BT-Settl, because it is "
            "a newer version. In that case, set "
            "model='bt-settl-cifist' when using "
            "add_model of Database."
        )

    elif model_tag == "exo-rem-highres":
        if teff_range is None:
            warnings.warn(
                "Adding the full high-resolution grid of Exo-Rem to the "
                "HDF5 database may not be feasible since it requires "
                "a large amount of memory. Please consider using the "
                "'teff_range' parameter to only add a subset of the "
                "model spectra to the database."
            )

        if wavel_range is None:
            warnings.warn(
                "Adding the full high-resolution grid of Exo-Rem to the "
                "HDF5 database may not be feasible since it requires "
                "a large amount of memory. Please consider using the "
                "'wavel_range' parameter to reduce the data size."
            )

    if wavel_sampling is not None and wavel_range is None:
        warnings.warn(
            "The 'wavel_sampling' parameter can only be "
            "used in combination with the 'wavel_range' "
            "parameter. The model spectra are therefore "
            "not resampled."
        )

        wavel_sampling = None

    input_file = f"{model_tag}.tgz"

    data_folder = Path(input_path) / model_tag
    data_file = Path(input_path) / input_file

    if not data_folder.exists():
        data_folder.mkdir()

    url = f"https://home.strw.leidenuniv.nl/~stolker/species/{model_tag}.tgz"

    if data_file.exists():
        sha256_hash = hashlib.sha256(open(str(data_file), "rb").read()).hexdigest()

        if sha256_hash != model_info["checksum"]:
            warnings.warn(
                f"The hash of the '{model_tag}' file is not as "
                "expected, probably because the model grid has "
                "been updated on the server. Please remove the "
                "following file such that the latest version "
                f"will be downloaded: {str(data_file)}"
            )

    else:
        print()

        pooch.retrieve(
            url=url,
            known_hash=f"sha256:{model_info['checksum']}",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if unpack_tar:
        with tarfile.open(str(data_file)) as tar_open:
            # Get a list of all TAR members
            tar_members = tar_open.getmembers()

        if teff_range is None:
            member_list = None
            n_members = len(tar_members)

        else:
            # Only include and extract TAR members
            # within the specified Teff range
            member_list = []

            for tar_item in tar_members:
                file_split = tar_item.name.split("_")
                param_index = file_split.index("teff") + 1
                teff_val = float(file_split[param_index])

                if teff_range[0] <= teff_val <= teff_range[1]:
                    member_list.append(tar_item)

            n_members = len(member_list)

        print(
            f"\nUnpacking {n_members}/{len(tar_members)} model spectra "
            f"from {model_info['name']} ({model_info['file size']})...",
            end="",
            flush=True,
        )

        extract_tarfile(str(data_file), str(data_folder), member_list=member_list)

        print(" [DONE]")

    print_newline = False

    if "information" in model_info:
        if not print_newline:
            print()
            print_newline = True

        print(f"Model information: {model_info['information']}")

    if "reference" in model_info:
        if not print_newline:
            print()
            print_newline = True

        print(
            f"Please cite {model_info['reference']} when "
            f"using {model_info['name']} in a publication"
        )

    if "url" in model_info:
        if not print_newline:
            print()
            print_newline = True

        print(f"Reference URL: {model_info['url']}")

    teff = []

    if "logg" in model_info["parameters"]:
        logg = []
    else:
        logg = None

    if "feh" in model_info["parameters"]:
        feh = []
    else:
        feh = None

    if "c_o_ratio" in model_info["parameters"]:
        c_o_ratio = []
    else:
        c_o_ratio = None

    if "fsed" in model_info["parameters"]:
        fsed = []
    else:
        fsed = None

    if "log_kzz" in model_info["parameters"]:
        log_kzz = []
    else:
        log_kzz = None

    if "ad_index" in model_info["parameters"]:
        ad_index = []
    else:
        ad_index = None

    if "log_co_iso" in model_info["parameters"]:
        log_co_iso = []
    else:
        log_co_iso = None

    flux = []

    print()

    if wavel_range is None:
        print(
            f"Wavelength range (um) = "
            f"{model_info['wavelength range'][0]} - "
            f"{model_info['wavelength range'][1]}"
        )

    else:
        print(f"Wavelength range (um) = {wavel_range[0]} - {wavel_range[1]}")

    if wavel_range is not None and wavel_sampling is not None:
        wavelength = create_wavelengths(wavel_range, wavel_sampling)
        resample_spectra = True

    else:
        wavelength = None
        resample_spectra = False
        wavel_sampling = model_info["lambda/d_lambda"]

    print(f"Sampling (lambda/d_lambda) = {wavel_sampling}")

    if teff_range is None:
        print(
            f"Teff range (K) = {model_info['teff range'][0]} - {model_info['teff range'][1]}"
        )
    else:
        print(f"Teff range (K) = {teff_range[0]} - {teff_range[1]}")

    print()
    print_message = ""

    model_files = sorted(data_folder.glob("*"))

    if len(model_files) == 0:
        raise ValueError(
            f"Could not find any model spectrum of {model_tag} in "
            f" the data_folder: {data_folder}. Perhaps the "
            "argument of 'unpack_tar' should be set to True?"
        )

    check_plot = False

    for file_item in model_files:
        if file_item.stem[: len(model_tag)] == model_tag:
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

            print_message = f"Adding {model_info['name']} model spectra... {file_item}"
            print(f"\r{print_message}", end="", flush=True)

            if file_item.suffix == ".dat":
                data_wavel, data_flux = np.loadtxt(str(file_item), unpack=True)

            else:
                data = np.load(str(file_item))

                data_wavel = data[:, 0]
                data_flux = data[:, 1]

            if fit_from is not None:
                if fit_from > data_wavel[-1]:
                    raise ValueError(
                        "The argument of 'fit_from', "
                        f"{fit_from} um, is larger than the "
                        f"longest wavelength of the {model_tag} "
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

            if np.all(np.diff(data_wavel) < 0):
                raise ValueError(
                    f"The wavelengths of {file_item.stem} are "
                    "not all monotonically increasing."
                )

            if wavel_range is None:
                if wavelength is None:
                    wavelength = np.copy(data_wavel)  # (um)

                flux.append(data_flux)  # (W m-2 um-1)

            else:
                if not resample_spectra:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                        wavel_select = (wavel_range[0] < wavelength) & (
                            wavelength < wavel_range[1]
                        )
                        wavelength = wavelength[wavel_select]

                    flux.append(data_flux[wavel_select])  # (W m-2 um-1)

                else:
                    flux_resample = spectres_numba(
                        wavelength,
                        data_wavel,
                        data_flux,
                        spec_errs=None,
                        fill=np.nan,
                        verbose=True,
                    )

                    if np.isnan(np.sum(flux_resample)):
                        raise ValueError(
                            f"Resampling is only possible if the new wavelength "
                            f"range ({wavelength[0]} - {wavelength[-1]} um) falls "
                            f"sufficiently far within the wavelength range "
                            f"({data_wavel[0]} - {data_wavel[-1]} um) of the input "
                            f"spectra."
                        )

                    flux.append(flux_resample)  # (W m-2 um-1)

    print()

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

    write_data(
        model_tag, model_info["parameters"], wavel_sampling, database, data_sorted
    )

    add_missing(model_tag, model_info["parameters"], database)
