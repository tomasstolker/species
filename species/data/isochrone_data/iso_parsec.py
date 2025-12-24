"""
Module with a function for adding the PARSEC
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants


@typechecked
def add_parsec(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the PARSEC v2.0 isochrone data to the database.

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    input_path : str
        Folder where the data is located.

    Returns
    -------
    NoneType
        None
    """

    iso_tag = "parsec"

    url = "https://home.strw.leidenuniv.nl/~stolker/species/parsec_evolution.dat"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="c41f60460ac0bf89390b16645238f9cc692316ce158543634763e6c928115b6e",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    iso_data = np.loadtxt(
        data_file,
        comments="#",
        delimiter=None,
        usecols=[2, 5, 6, 7, 8, 32, 33],
        unpack=False,
    )

    idx_bad = iso_data[:, 2] == -9.999
    iso_data = iso_data[~idx_bad, :]

    log_age, mass, log_lum, log_teff, log_g, radius_pol, radius_eq = iso_data.T

    age = 1e-6 * 10.0**log_age  # (Myr)
    mass *= constants.M_SUN / constants.M_JUP  # (Msun) -> (Mjup)
    teff = 10.0**log_teff  # (K)
    radius = (radius_pol + radius_eq) / 2.0  # (Rsun)
    radius *= constants.R_SUN / constants.R_JUP  # (Rjup)

    print("Adding isochrones: PARSEC v2.0...", end="", flush=True)

    database.create_dataset(f"isochrones/{iso_tag}/mass", data=mass)  # (Mjup)
    dset = database.create_dataset(f"isochrones/{iso_tag}/age", data=age)  # (Myr)
    database.create_dataset(f"isochrones/{iso_tag}/teff", data=teff)  # (K)
    database.create_dataset(
        f"isochrones/{iso_tag}/log_lum", data=log_lum
    )  # log(L/Lsun)
    database.create_dataset(f"isochrones/{iso_tag}/radius", data=radius)  # (Rjup)
    database.create_dataset(f"isochrones/{iso_tag}/log_g", data=log_g)  # log(g)

    dset.attrs["model"] = iso_tag

    print(" [DONE]")
    print(f"Database tag: {iso_tag}")
