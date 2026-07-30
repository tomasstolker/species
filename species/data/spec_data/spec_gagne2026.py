"""
Module for adding the 1–6 μm SPHEREx spectra of ultracool dwarfs (UCDs)
derived with SPIFF from `Gagné et al. (2026)
<https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G>`_ to the
database. Contains a function that adds the 2,304 binned spectra
and a function that adds 44 spectral templates.
"""

import os
import re

from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
import pooch

from beartype import beartype
from mocapy import MocaEngine

from species.util.data_util import remove_directory


@beartype
def add_gagne2026(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the 1–6 μm SPHEREx spectra of ultracool dwarfs (UCDs)
    derived with SPIFF from `Gagné et al. (2026)
    <https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G>`_ to the
    database. The function imports the binned spectra if the 2,304 objects.

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

    print_text = "SPHEREx SPIFF spectra of UCDs from Gagne et al. 2026"

    url = "https://zenodo.org/records/19051216/files/spiff_known_bds_binned_csv.zip"
    input_file = "spiff_known_bds_binned_csv.zip"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "gagne+2026/"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="d8d16ee156074a1ade74cdd9c024e0119347eb11781e83b849680001f1e916ec",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if data_folder.exists():
        remove_directory(data_folder)

    print(f"\nUnpacking {print_text} (7.2 MB)...", end="", flush=True)

    with ZipFile(data_file) as zip_file:
        for zip_info in zip_file.infolist():
            if zip_info.is_dir():
                continue

            zip_info.filename = os.path.basename(zip_info.filename)
            zip_file.extract(zip_info, data_folder)

    print(" [DONE]")

    spec_files = sorted(data_folder.glob("*.csv"))

    moca_oid_list = []

    for file_item in spec_files:
        match = re.match(
            r"(?P<name>.+)_moca_oid_(?P<moca_oid>\d+)"
            r"_spt_(?P<sptype>.+?)_spherex_spectrum\.csv$",
            file_item.name,
        )

        name = match["name"]
        moca_oid = match["moca_oid"]
        sptype = match["sptype"]
        moca_oid_list.append(moca_oid)

    print("\nQuerying parallaxes in MOCAdb...", end="", flush=True)

    moca = MocaEngine()

    mocadb_query = f"""
    SELECT
        moca_oid,
        parallax_mas
    FROM
        summary_all_members
    WHERE
        moca_oid IN ({", ".join(map(str, moca_oid_list))});
    """

    df = moca.query(mocadb_query)

    print(" [DONE]")

    print_message = ""
    print()

    for file_item in spec_files:
        match = re.match(
            r"(?P<name>.+)_moca_oid_(?P<moca_oid>\d+)"
            r"_spt_(?P<sptype>.+?)_spherex_spectrum\.csv$",
            file_item.name,
        )

        name = match["name"]
        moca_oid = match["moca_oid"]
        sptype = match["sptype"]

        parallax = df["parallax_mas"][df["moca_oid"] == int(moca_oid)]

        if len(parallax) == 0:
            parallax = np.nan
        else:
            parallax = parallax.iloc[0]

        data = pd.read_csv(file_item)

        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {name}"
        print(f"\r{print_message}", end="")

        data = data.dropna().to_numpy(dtype=float)
        dset = database.create_dataset(f"spectra/gagne+2026/{name}", data=data)

        dset.attrs["name"] = name.encode()
        dset.attrs["sptype"] = sptype.encode()
        dset.attrs["moca_oid"] = moca_oid.encode()
        dset.attrs["file_name"] = file_item.name.encode()
        dset.attrs["parallax"] = parallax
        dset.attrs["parallax_error"] = np.nan

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")


@beartype
def add_gagnetemplates2026(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the 1–6 μm SPHEREx spectral templates of
    ultracool dwarfs (UCDs) constructed by combining individual
    SPIFF spectra from `Gagné et al. (2026)
    <https://ui.adsabs.harvard.edu/abs/2026arXiv260422012G>`_ to
    the database. The function imports the 44 "raw" spectral templates.

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

    print_text = "SPHEREx SPIFF templates of UCDs from Gagne et al. 2026"

    url = "https://zenodo.org/records/19051216/files/spiff_templates_raw_csv.zip"
    input_file = "spiff_templates_raw_csv.zip"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "gagne-templates+2026/"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="342bcef7d550acdacb4835491c1bf1ca4f2a9fbcce3d397bc681719e0ad94e9f",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if data_folder.exists():
        remove_directory(data_folder)

    print(f"\nUnpacking {print_text} (75 kB)...", end="", flush=True)

    with ZipFile(data_file) as zip_file:
        for zip_info in zip_file.infolist():
            if zip_info.is_dir():
                continue

            zip_info.filename = os.path.basename(zip_info.filename)
            zip_file.extract(zip_info, data_folder)

    print(" [DONE]")

    spec_files = sorted(data_folder.glob("*_raw_*.csv"))

    print_message = ""
    print()

    for file_item in spec_files:
        match = re.match(r"(?P<sptype>.+?)_raw_.*\.csv$", file_item.name)

        sptype = match["sptype"]
        name = f"{sptype} template"

        data = pd.read_csv(file_item)

        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {name}"
        print(f"\r{print_message}", end="")

        data = data.dropna().to_numpy(dtype=float)

        dset = database.create_dataset(
            f"spectra/gagne-templates+2026/{name}",
            data=data,
        )

        dset.attrs["name"] = name.encode()
        dset.attrs["sptype"] = sptype.encode()
        dset.attrs["file_name"] = file_item.name.encode()

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
