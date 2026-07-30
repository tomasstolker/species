"""
Module for adding 1-6 micron SPHEREx spectra of BDs dervied using SPIFF from
`Gagne et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G/abstract>`_ to the database. In particular, utilizes only the binned, knwon UCD sample (2304 objects).
"""

import os
import gzip
from zipfile import ZipFile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pooch

from astropy.io import fits
from beartype import beartype

from species.util.data_util import extract_tarfile, remove_directory


@beartype
def add_gagne2026(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding 1-6 micron SPHEREx spectra of BDs dervied using SPIFF from
    `Gagne et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G/abstract>`_ to the database. In particular, utilizes only the binned, knwon UCD sample (2304 objects)

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        The HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    print_text = "SPHEREx SPIFF spectra of ultra cool dwarfs from Gagne et al. 2026"

    url = "https://zenodo.org/records/19051216/files/spiff_known_bds_binned_csv.zip"
    input_file = "spiff_known_bds_binned_csv.zip"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "gagne+2026/"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if data_folder.exists():
        remove_directory(data_folder)

    print(f"\nUnpacking {print_text} (2.3 MB)...", end="", flush=True)
    with ZipFile(data_file) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, data_folder)
    print(" [DONE]")

    spec_dict = {}

    spec_files = sorted(data_folder.glob("*"))

    for line in spec_files:
        name = str(line).split("_moca_")[0].split("gagne+2026/")[-1]
        files = str(line)
        sptype = str(line).split("_spt_")[-1].replace("_spherex_spectrum.csv", "")

        spec_dict[name] = {"name": name, "sptype": sptype, "files": files}

    print_message = ""
    print()

    for file_item in spec_files:

        data = pd.read_csv(file_item)

        for spec_key, spec_value in spec_dict.items():
            if file_item.name in spec_value["files"]:

                spec_value["SPIFF"] = data.dropna().to_numpy(dtype=float)

    for spec_key, spec_value in spec_dict.items():
        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {spec_key}"
        print(f"\r{print_message}", end="")

        if "SPIFF" in spec_value:
            sp_data = spec_value["SPIFF"]

        dset = database.create_dataset(f"spectra/gagne+2026/{spec_key}", data=sp_data)

        dset.attrs["name"] = str(spec_key).encode()
        dset.attrs["sptype"] = str(spec_value["sptype"]).encode()

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")


@beartype
def add_gagnetemplates2026(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding 1-6 micron SPHEREx templates of BDs combined from individual SPIFF spectra from
    `Gagne et al. (2026) <https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G/abstract>`_ to the database. In particular, uses the "raw" spectral templates.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        The HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    print_text = "SPHEREx SPIFF templates of ultra cool dwarfs from Gagne et al. 2026"

    url = "https://zenodo.org/records/19051216/files/spiff_templates_raw_csv.zip"
    input_file = "spiff_template_sptypes.zip"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "gagne-templates+2026/"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if data_folder.exists():
        remove_directory(data_folder)

    print(f"\nUnpacking {print_text} (2.3 MB)...", end="", flush=True)
    with ZipFile(data_file) as zip:
        for zip_info in zip.infolist():
            if zip_info.is_dir():
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            zip.extract(zip_info, data_folder)
    print(" [DONE]")

    spec_dict = {}

    spec_files = sorted(data_folder.glob("*"))

    for line in spec_files:
        name = (
            "TEMPLATE_" + str(line).split("_raw_")[0].split("gagne-templates+2026/")[-1]
        )
        files = str(line)
        sptype = str(line).split("_raw_")[0].split("gagne-templates+2026/")[-1]

        spec_dict[name] = {"name": name, "sptype": sptype, "files": files}

    print_message = ""
    print()

    for file_item in spec_files:

        data = pd.read_csv(file_item)

        for spec_key, spec_value in spec_dict.items():
            if file_item.name in spec_value["files"]:

                spec_value["SPIFF-TEMPLATE"] = data.dropna().to_numpy(dtype=float)

    for spec_key, spec_value in spec_dict.items():
        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {spec_key}"
        print(f"\r{print_message}", end="")

        if "SPIFF-TEMPLATE" in spec_value:
            sp_data = spec_value["SPIFF-TEMPLATE"]

        dset = database.create_dataset(
            f"spectra/gagne-templates+2026/{spec_key}", data=sp_data
        )

        dset.attrs["name"] = str(spec_key).encode()
        dset.attrs["sptype"] = str(spec_value["sptype"]).encode()

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
