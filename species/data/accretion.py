"""
Module for the accretion luminosity relation.
"""

import os
import urllib.request

import h5py
import numpy as np

from typeguard import typechecked

from species.core import constants


@typechecked
def add_accretion_relation(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the accretion relation from `Aoyama et al.
    (2021) <https://ui. adsabs.harvard.edu/abs/
    2021ApJ...917L..30A/abstract>`_ to the database. It provides
    coefficients to convert line to accretion luminosities.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    None
        NoneType
    """

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url = (
        "https://home.strw.leidenuniv.nl/~stolker/"
        "species/ab-Koeffienzenten_mehrStellen.dat"
    )

    data_file = os.path.join(input_path, "ab-Koeffienzenten_mehrStellen.dat")

    if not os.path.isfile(data_file):
        print(
            "Downloading coefficients for accretion relation (1.1 kB)...",
            end="",
            flush=True,
        )
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    print("Adding coefficients for accretion relation (1.1 kB)...", end="", flush=True)

    data = np.genfromtxt(
        data_file,
        dtype=None,
        skip_header=1,
        encoding=None,
        names=True,
        usecols=[0, 1, 2, 3, 4]
    )

    line_names = data["name"]
    coefficients = np.column_stack([data["a"], data["b"]])

    # Rest wavelength (um) from Rydberg formula
    wavelengths = 1e6 / (
        constants.RYDBERG * (1.0 / data["nf"] ** 2 - 1.0 / data["ni"] ** 2)
    )

    database.create_dataset("accretion/wavelengths", data=wavelengths)
    database.create_dataset("accretion/coefficients", data=coefficients)

    dtype = h5py.special_dtype(vlen=str)
    dset = database.create_dataset(
        "accretion/hydrogen_lines", (np.size(line_names),), dtype=dtype
    )
    dset[...] = line_names

    print(" [DONE]")

    print(f"Please cite Aoyama et al. (2021) when using "
          f"the accretion relation in a publication")
