"""
Module for adding a grid of model spectra to the database.
"""

import json
import os
import pathlib
import tarfile
import urllib.request
import warnings

from typing import Optional, Tuple

import h5py
import spectres
import numpy as np

from typeguard import typechecked

from species.util import data_util, read_util


@typechecked
def add_model_grid(
    model_name: str,
    input_path: str,
    database: h5py._hl.files.File,
    wavel_range: Optional[Tuple[float, float]],
    teff_range: Optional[Tuple[float, float]],
    spec_res: Optional[float],
) -> None:
    """
    Function for adding a grid of model spectra to the database.
    The original spectra had been resampled to logarithmically-
    spaced wavelengths, so at a constant resolution,
    :math:`\\lambda/\\Delta\\lambda`. This function downloads
    the model grid, unpacks the tar file, and adds the spectra
    and parameters to the database.

    Parameters
    ----------
    model_name : str
        Name of the model grid.
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    wavel_range : tuple(float, float), None
        Wavelength range (um). The original wavelength
        points are used if set to ``None``.
    teff_range : tuple(float, float), None
        Effective temperature range (K). All temperatures
        are selected if set to ``None``.
    spec_res : float, None
        Spectral resolution for resampling. Not used if
        ``wavel_range`` is set to ``None`` and/or
        ``spec_res`` is set to ``None``

    Returns
    -------
    NoneType
        None
    """

    data_file = pathlib.Path(__file__).parent.resolve() / "model_data.json"

    with open(data_file, "r", encoding="utf-8") as json_file:
        model_data = json.load(json_file)

    if model_name in model_data.keys():
        model_info = model_data[model_name]

    else:
        raise ValueError(
            f"The {model_name} atmospheric model is not available. "
            f"Please choose one of the following models: "
            f"'ames-cond', 'ames-dusty', 'atmo', 'bt-settl', "
            f"'bt-nextgen', 'drift-phoexnix', 'petitcode-cool-clear', "
            f"'petitcode-cool-cloudy', 'petitcode-hot-clear', "
            f"'petitcode-hot-cloudy', 'exo-rem', 'bt-settl-cifist', "
            f"'bt-cond', 'bt-cond-feh', 'blackbody', 'sonora-cholla', "
            f"'sonora-bobcat', 'sonora-bobcat-co'"
        )

    if model_name == "bt-settl":
        warnings.warn("It is recommended to use the CIFIST "
                      "grid of the BT-Settl, because it is "
                      "a newer version. In that case, set "
                      "model='bt-settl-cifist' when using "
                      "add_model of Database.")

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    input_file = f"{model_name}.tgz"

    data_folder = os.path.join(input_path, model_name)
    data_file = os.path.join(input_path, input_file)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    url = f"https://home.strw.leidenuniv.nl/~stolker/species/{model_name}.tgz"

    if not os.path.isfile(data_file):
        print(
            f"Downloading {model_info['name']} model "
            f"spectra ({model_info['file size']})...",
            end="",
            flush=True,
        )
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    print(
        f"Unpacking {model_info['name']} model "
        f"spectra ({model_info['file size']})...",
        end="",
        flush=True,
    )
    tar = tarfile.open(data_file)
    tar.extractall(data_folder)
    tar.close()
    print(" [DONE]")

    if "information" in model_info:
        print(f"Model information: {model_info['information']}")

    if "reference" in model_info:
        print(f"Please cite {model_info['reference']} when "
              f"using {model_info['name']} in a publication")

    if "url" in model_info:
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

    flux = []

    if wavel_range is not None and spec_res is not None:
        wavelength = read_util.create_wavelengths(wavel_range, spec_res)
        print(f"Wavelength range (um) = {wavel_range[0]} - {wavel_range[1]}")
        print(f"Spectral resolution = {spec_res}")

    else:
        wavelength = None
        print(f"Wavelength range (um) = "
              f"{model_info['wavelength range'][0]} -"
              f"{model_info['wavelength range'][1]}")
        print(f"Spectral resolution = {model_info['resolution']}")

    if teff_range is None:
        print(f"Teff range (K) = {model_info['teff range'][0]} - {model_info['teff range'][1]}")
    else:
        print(f"Teff range (K) = {teff_range[0]} - {teff_range[1]}")

    print_message = ""

    for _, _, file_list in os.walk(data_folder):
        for filename in sorted(file_list):

            if filename[: len(model_name)] == model_name:
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

                empty_message = len(print_message) * " "
                print(f"\r{empty_message}", end="")

                print_message = (
                    f"Adding {model_info['name']} model spectra... {filename}"
                )
                print(f"\r{print_message}", end="")

                data_wavel, data_flux = np.loadtxt(
                    os.path.join(data_folder, filename), unpack=True
                )

                if wavel_range is None or spec_res is None:
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

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = f"Adding {model_info['name']} model spectra... [DONE]"
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

    data_sorted = data_util.sort_data(
        np.asarray(teff),
        logg,
        feh,
        c_o_ratio,
        fsed,
        log_kzz,
        wavelength,
        np.asarray(flux),
    )

    data_util.write_data(model_name, model_info["parameters"], database, data_sorted)
