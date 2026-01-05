"""
Module for adding young, M- and L-type dwarf spectra from
`Bonnefoy et al. (2014) <https://ui.adsabs.harvard.edu/abs/
2014A%26A...562A.127B>`_ to the database.
"""

import gzip

from pathlib import Path

import h5py
import numpy as np
import pooch

from astropy.io import fits
from typeguard import typechecked

from species.util.data_util import extract_tarfile, remove_directory


@typechecked
def add_bonnefoy2014(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the SINFONI spectra of young, M- and L-type
    dwarfs from `Bonnefoy et al. (2014) <https://ui.adsabs.harvard.
    edu/abs/2014A%26A...562A.127B>`_ to the database.

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

    print_text = "spectra of young M/L type objects from Bonnefoy et al. 2014"

    url = "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/A+A/562/A127/"
    input_file = "J_A+A_562_A127.tar.gz"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "bonnefoy+2014/"

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
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    spec_dict = {}

    data_file = Path(data_folder) / "stars.dat.gz"

    with gzip.open(data_file, "r") as gzip_file:
        for line in gzip_file:
            name = line[:13].decode().strip()
            files = line[80:].decode().strip().split()
            sptype = line[49:56].decode().strip()

            if name == "NAME 2M1207A":
                name = "2M1207A"

            if len(sptype) == 0:
                sptype = None
            elif "." in sptype:
                sptype = sptype[:4]
            else:
                sptype = sptype[:2]

            if name == "Cha1109":
                sptype = "M9"
            elif name == "DH Tau B":
                sptype = "M9"
            elif name == "TWA 22A":
                sptype = "M6"
            elif name == "TWA 22B":
                sptype = "M6"
            elif name == "CT Cha b":
                sptype = "M9"

            spec_dict[name] = {"name": name, "sptype": sptype, "files": files}

    fits_folder = Path(data_folder) / "sp"

    print_message = ""
    print()

    spec_files = sorted(fits_folder.glob("*"))

    for file_item in spec_files:
        fname_split = file_item.stem.split("_")

        data = fits.getdata(file_item)

        for spec_key, spec_value in spec_dict.items():
            if file_item.name in spec_value["files"]:
                if spec_key == "TWA 22AB":
                    # Binary spectrum
                    continue

                if "JHK" in fname_split:
                    spec_value["JHK"] = data

                elif "J" in fname_split:
                    spec_value["J"] = data

                elif "H+K" in fname_split or "HK" in fname_split:
                    spec_value["HK"] = data

    for spec_key, spec_value in spec_dict.items():
        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {spec_key}"
        print(f"\r{print_message}", end="")

        if "JHK" in spec_value:
            sp_data = spec_value["JHK"]

        elif "J" in spec_value and "HK" in spec_value:
            sp_data = np.vstack((spec_value["J"], spec_value["HK"]))

        else:
            # Binary spectrum
            continue

        dset = database.create_dataset(
            f"spectra/bonnefoy+2014/{spec_key}", data=sp_data
        )

        dset.attrs["name"] = str(spec_key).encode()
        dset.attrs["sptype"] = str(spec_value["sptype"]).encode()

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
