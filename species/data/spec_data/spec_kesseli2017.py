"""
Module for adding O5 through L3 SDSS stellar spectra from
Kesseli et al. (2017) to the database.
"""

from pathlib import Path

import h5py
import numpy as np
import pooch

from astropy.io import fits
from typeguard import typechecked

from species.util.data_util import extract_tarfile, remove_directory


@typechecked
def add_kesseli2017(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the SDSS stellar spectra from
    Kesseli et al. (2017) to the database.

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

    url = "https://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/ApJS/230/16"
    input_file = "J_ApJS_230_16.tar.gz"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "kesseli+2017/"

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

    print(
        "\nUnpacking SDSS spectra from Kesseli et al. 2017 (145 MB)...",
        end="",
        flush=True,
    )
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    fits_folder = Path(data_folder) / "fits"

    print_message = ""
    print()

    spec_files = sorted(fits_folder.glob("*"))

    for file_item in spec_files:
        data = fits.getdata(file_item, ext=1)

        wavelength = 1e-4 * 10.0 ** data["LogLam"]  # (um)
        flux = data["Flux"]  # Normalized units
        error = data["PropErr"]  # Normalized units

        name = file_item.stem.replace("_", " ")

        file_split = file_item.stem.split("_")
        sptype = file_split[0].split(".")[0]

        spdata = np.column_stack([wavelength, flux, error])

        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {name}"
        print(f"\r{print_message}", end="")

        dset = database.create_dataset(f"spectra/kesseli+2017/{name}", data=spdata)

        dset.attrs["name"] = str(name).encode()
        dset.attrs["sptype"] = str(sptype).encode()

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
