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
from typeguard import typechecked


@typechecked
def add_marleau(database: h5py._hl.files.File, input_path: str) -> None:
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

    # M      age     S_0             L          S(t)            R        Teff
    # (M_J)  (Gyr)   (k_B/baryon)    (L_sol)    (k_B/baryon)    (R_J)    (K)
    # mass, age, _, luminosity, _, radius, teff = np.loadtxt(file_name, unpack=True)
    #
    # age *= 1e3  # (Myr)
    # luminosity = np.log10(luminosity)
    #
    # mass_cgs = 1e3 * mass * constants.M_JUP  # (g)
    # radius_cgs = 1e2 * radius * constants.R_JUP  # (cm)
    #
    # logg = np.log10(1e3 * constants.GRAVITY * mass_cgs / radius_cgs**2)

    # isochrones = np.vstack((age, mass, teff, luminosity, logg))
    # isochrones = np.transpose(isochrones)
    #
    # index_sort = np.argsort(isochrones[:, 0])
    # isochrones = isochrones[index_sort, :]
    #
    # dset = database.create_dataset(f"isochrones/{tag}/evolution", data=isochrones)
    #
    # dset.attrs["model"] = "marleau"

    iso_tag = "marleau"

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_luminosity.fits"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="218bd69105206373389c8d513969c500ad39ec32a5bd98e248aedf0b3e29af67",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_radius.fits"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="6aa0e75b3e54a427caa124ac9f1906ac180245927238a00a57acda8d5e7bf75d",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_points.json"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="592415d9a5425211b6e1bdc1b0e36c7e45d317165dca639fbea7d77cbfcafb9d",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    print("\nAdding isochrones: Marleau & Cumming (2014)...", end="", flush=True)

    # Bolometric luminosity log10(L/Lsun)
    data_lbol = fits.getdata(os.path.join(input_path, "evolution_luminosity.fits"))
    data_lbol = np.log10(data_lbol)

    # Radius (Rjup)
    data_radius = fits.getdata(os.path.join(input_path, "evolution_radius.fits"))

    file_name = os.path.join(input_path, "evolution_points.json")
    with open(file_name, "r", encoding="utf-8") as json_file:
        grid_points = json.load(json_file)

    # Select Solar Helium and Deuterium fraction, and zero core mass
    data_lbol = data_lbol[..., 1, 1, 0]
    data_radius = data_radius[..., 1, 1, 0]

    model_param = ["age", "mass", "log_lum", "radius", "s_i"]

    age_list = []
    mass_list = []
    lbol_list = []
    radius_list = []
    s_i_list = []

    for age_idx, age_item in enumerate(grid_points["age"]):
        for mass_idx, mass_item in enumerate(grid_points["mass"]):
            for s_i_idx, s_i_item in enumerate(grid_points["s_i"]):
                age_list.append(age_item)
                mass_list.append(mass_item)
                lbol_list.append(data_lbol[age_idx, mass_idx, s_i_idx])
                radius_list.append(data_lbol[age_idx, mass_idx, s_i_idx])
                s_i_list.append(s_i_item)

    dgroup = database.create_group(f"isochrones/{iso_tag}")
    dgroup.attrs["model"] = iso_tag
    dgroup.attrs["n_param"] = len(model_param)
    for i, item in enumerate(model_param):
        dgroup.attrs[f"parameter{i}"] = item

    # TODO dummy Teff and log(g)
    teff_list = np.full(len(age_list), 10000.)
    logg_list = np.full(len(age_list), 4.0)

    database.create_dataset(f"isochrones/{iso_tag}/mass", data=mass_list)  # (Mjup)
    database.create_dataset(f"isochrones/{iso_tag}/age", data=age_list)  # (Myr)
    database.create_dataset(
        f"isochrones/{iso_tag}/log_lum", data=lbol_list
    )  # log(L/Lsun)
    database.create_dataset(f"isochrones/{iso_tag}/radius", data=radius_list)  # (Rjup)
    database.create_dataset(f"isochrones/{iso_tag}/s_init", data=s_i_list)  # (k_b/baryon)
    database.create_dataset(f"isochrones/{iso_tag}/teff", data=teff_list)  # (K)
    database.create_dataset(f"isochrones/{iso_tag}/log_g", data=logg_list)  # log(g)

    print(" [DONE]")
