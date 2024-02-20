"""
Module for adding a grid of model spectra to the database.
"""

import json
import os
import pathlib
import tarfile
import warnings

from typing import Optional, Tuple

import h5py
import numpy as np
import pooch
import spectres

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
        Wavelength range (um). The original wavelength
        points are used if set to ``None``.
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

    Returns
    -------
    NoneType
        None
    """

    print_section("Add grid of model spectra")

    model_name = convert_model_name(model_tag)
    print(f"Database tag: {model_tag}")
    print(f"Model name: {model_name}")

    data_file = pathlib.Path(__file__).parent.resolve() / "model_data.json"

    with open(data_file, "r", encoding="utf-8") as json_file:
        model_data = json.load(json_file)

    if model_tag in model_data.keys():
        model_info = model_data[model_tag]

    else:
        raise ValueError(
            f"The '{model_tag}' atmospheric model is not available. "
            "Please choose one of the following models: "
            "'ames-cond', 'ames-dusty', 'atmo', 'bt-settl', "
            "'bt-nextgen', 'drift-phoexnix', 'petitcode-cool-clear', "
            "'petitcode-cool-cloudy', 'petitcode-hot-clear', "
            "'petitcode-hot-cloudy', 'exo-rem', 'bt-settl-cifist', "
            "'bt-cond', 'bt-cond-feh', 'blackbody', 'sonora-cholla', "
            "'sonora-bobcat', 'sonora-bobcat-co', 'koester-wd', "
            "'saumon2008-clear', 'saumon2008-cloudy', 'petrus2023', "
            "'sphinx'"
        )

    if model_tag == "bt-settl":
        warnings.warn(
            "It is recommended to use the CIFIST "
            "grid of the BT-Settl, because it is "
            "a newer version. In that case, set "
            "model='bt-settl-cifist' when using "
            "add_model of Database."
        )

    elif model_tag == "exo-rem":
        warnings.warn(
            "The Exo-Rem grid has been updated to the latest version "
            "from https://lesia.obspm.fr/exorem/YGP_grids/. Please "
            "consider removing the grid from the 'data_folder' if "
            "needed such that the latest version of the grid will "
            "be downloaded and added to the HDF5 database."
        )

    elif model_tag == "exo-rem-highres" and teff_range is None:
        warnings.warn(
            "Adding the full high-resolution grid of Exo-Rem to the "
            "HDF5 database may not be feasible since it requires "
            "a large amount of memory. Please consider using the "
            "'teff_range' parameter to only add a small "
            "Teff range of model spectra to the database."
        )

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    input_file = f"{model_tag}.tgz"

    data_folder = os.path.join(input_path, model_tag)
    data_file = os.path.join(input_path, input_file)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    url = f"https://home.strw.leidenuniv.nl/~stolker/species/{model_tag}.tgz"

    if not os.path.isfile(data_file):
        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if unpack_tar:
        with tarfile.open(data_file) as tar_open:
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

        extract_tarfile(data_file, data_folder, member_list=member_list)

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

    flux = []

    print()

    if wavel_range is not None and wavel_sampling is not None:
        wavelength = create_wavelengths(wavel_range, wavel_sampling)
        print(f"Wavelength range (um) = {wavel_range[0]} - {wavel_range[1]}")

    else:
        wavelength = None
        print(
            f"Wavelength range (um) = "
            f"{model_info['wavelength range'][0]} - "
            f"{model_info['wavelength range'][1]}"
        )
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

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):
            if filename[: len(model_tag)] == model_tag:
                file_split = filename.split("_")

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

                empty_message = len(print_message) * " "
                print(f"\r{empty_message}", end="")

                print_message = (
                    f"Adding {model_info['name']} model spectra... {filename}"
                )
                print(f"\r{print_message}", end="", flush=True)

                data_wavel, data_flux = np.loadtxt(
                    os.path.join(data_folder, filename), unpack=True
                )

                if wavel_range is None:
                    if wavelength is None:
                        wavelength = np.copy(data_wavel)  # (um)

                        if np.all(np.diff(wavelength) < 0):
                            raise ValueError(
                                "The wavelengths are not all sorted by increasing value."
                            )

                    flux.append(data_flux)  # (W m-2 um-1)

                else:
                    flux_resample = spectres.spectres(
                        wavelength,
                        data_wavel,
                        data_flux,
                        spec_errs=None,
                        fill=np.nan,
                        verbose=False,
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

    data_sorted = sort_data(
        np.asarray(teff),
        logg,
        feh,
        c_o_ratio,
        fsed,
        log_kzz,
        ad_index,
        wavelength,
        np.asarray(flux),
    )

    write_data(
        model_tag, model_info["parameters"], wavel_sampling, database, data_sorted
    )

    add_missing(model_tag, model_info["parameters"], database)
