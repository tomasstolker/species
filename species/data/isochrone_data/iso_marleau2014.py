"""
Module with a function for adding the Marleau & Cumming (2014)
evolutionary tracks to the database.
"""

import json
import os

from pathlib import Path

import h5py
import numpy as np
import pooch

from astropy.io import fits
from beartype import beartype

from species.core import constants


@beartype
def add_marleau2014(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the `Marleau & Cumming (2014)
    <https://ui.adsabs.harvard.edu/abs/2014MNRAS.437.1378M>`_
    isochrone data to the database.

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

    iso_tag = "marleau2014"

    url = "https://home.strw.leidenuniv.nl/~stolker/species/marleau2014_lbol.fits"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="76e47eb6b8124148bc618c1ec88b0885992e82f57606ad7ae24a1e77fa7011e4",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    url = "https://home.strw.leidenuniv.nl/~stolker/species/marleau2014_radius.fits"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="e69e78cd2fe5d642f493fa4e53dc7049db7269a9de9ad2aa0e447d02303b31eb",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    url = "https://home.strw.leidenuniv.nl/~stolker/species/marleau2014_points.json"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="dd2d5695e3eb1b07851523b9dfb518fd70a55ac6f85a2ccfe2b83d97d869d548",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    print("\nAdding isochrones: Marleau & Cumming (2014)...", end="", flush=True)

    # Bolometric luminosity log10(L/Lsun)
    data_lbol = fits.getdata(os.path.join(input_path, "marleau2014_lbol.fits"))
    data_lbol = np.log10(data_lbol)

    # Radius (Rjup)
    data_radius = fits.getdata(os.path.join(input_path, "marleau2014_radius.fits"))

    file_name = os.path.join(input_path, "marleau2014_points.json")
    with open(file_name, "r", encoding="utf-8") as json_file:
        grid_points = json.load(json_file)

    # Select primordial helium and feuterium fraction, and zero core mass
    # These values were also used in Marleau et al.
    # TODO Allow these parameters to be interpolated
    data_lbol = data_lbol[..., 1, 0, 0]
    data_radius = data_radius[..., 1, 0, 0]

    model_param = ["age", "mass", "s_init"]

    dgroup = database.create_group(f"isochrones/{iso_tag}")

    dgroup.attrs["model"] = iso_tag
    dgroup.attrs["regular_grid"] = True
    dgroup.attrs["n_param"] = len(model_param)

    for i, item in enumerate(model_param):
        dgroup.attrs[f"parameter{i}"] = item

    data_teff = (
        (10.0**data_lbol * constants.L_SUN)
        / (4.0 * np.pi * (data_radius * constants.R_JUP) ** 2 * constants.SIGMA_SB)
    ) ** (1.0 / 4.0)

    data_mass = np.array(grid_points["mass"])
    data_mass = np.broadcast_to(data_mass[None, :, None], data_lbol.shape)

    # 1e2 to convert from SI to cgs
    data_logg = np.log10(
        1e2
        * constants.GRAVITY
        * (data_mass * constants.M_JUP)
        / (data_radius * constants.R_JUP) ** 2
    )

    database.create_dataset(
        f"isochrones/{iso_tag}/mass", data=grid_points["mass"]
    )  # (Mjup)

    database.create_dataset(
        f"isochrones/{iso_tag}/age", data=grid_points["age"]
    )  # (Myr)

    database.create_dataset(
        f"isochrones/{iso_tag}/log_lum", data=data_lbol
    )  # log(L/Lsun)

    database.create_dataset(f"isochrones/{iso_tag}/radius", data=data_radius)  # (Rjup)

    database.create_dataset(
        f"isochrones/{iso_tag}/s_init", data=grid_points["s_i"]
    )  # (k_b/baryon)

    database.create_dataset(f"isochrones/{iso_tag}/teff", data=data_teff)  # (K)

    database.create_dataset(f"isochrones/{iso_tag}/log_g", data=data_logg)  # log(g)

    print(" [DONE]")
