"""
Module for evolutionary data.
"""

import json
import os
import urllib.request

import h5py
import numpy as np

from astropy.io import fits
from typeguard import typechecked


@typechecked
def add_evolution(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for downloading a grid of evolutionary data
    and adding the data to the database.

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

    # Download grid with bolometric luminosities

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_luminosity.fits"

    data_file = os.path.join(input_path, "evolution_luminosity.fits")

    if not os.path.isfile(data_file):
        print(
            "Downloading evolution grid with luminosities (557 MB)...",
            end="",
            flush=True,
        )
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    # Download grid with radii

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_radius.fits"

    data_file = os.path.join(input_path, "evolution_radius.fits")

    if not os.path.isfile(data_file):
        print("Downloading evolution grid with radii (557 MB)...", end="", flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    # Download JSON file with grid points

    url = "https://home.strw.leidenuniv.nl/~stolker/species/evolution_points.json"

    data_file = os.path.join(input_path, "evolution_points.json")

    if not os.path.isfile(data_file):
        print("Downloading evolution grid with radii (17 kB)...", end="", flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    # Add data to the database

    print("Adding grid with evolution data:")

    # Bolometric luminositry log10(L/Lsun)
    data = fits.getdata(os.path.join(input_path, "evolution_luminosity.fits"))
    database.create_dataset("evolution/grid_lbol", data=np.log10(data))
    print(f"   - Luminosity grid shape: {data.shape}")

    # Planet radius (Rjup)
    data = fits.getdata(os.path.join(input_path, "evolution_radius.fits"))
    database.create_dataset("evolution/grid_radius", data=data)
    print(f"   - Radius grid shape: {data.shape}")

    file_name = os.path.join(input_path, "evolution_points.json")
    with open(file_name, "r", encoding="utf-8") as json_file:
        grid_points = json.load(json_file)

    for key, value in grid_points.items():
        database.create_dataset(f"evolution/{key}", data=value)
        if key == "age":
            print(
                f"   - Number of {key} points = {len(value)}, from {value[0]} to {value[-1]:.1f}"
            )
        else:
            print(
                f"   - Number of {key} points = {len(value)}, from {value[0]} to {value[-1]}"
            )
