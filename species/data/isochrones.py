"""
Module for isochrone data from evolutionary models.
"""

import os
import tarfile
import urllib.request

import h5py
import numpy as np

from species.core import constants


def add_baraffe(database, tag, file_name):
    """
    Function for adding the `Baraffe et al. (2003)
    <https://ui.adsabs.harvard.edu/abs/2003A%26A...402..701B/>`_
    isochrone data to the database. Any of the isochrones from
    https://phoenix.ens-lyon.fr/Grids/ can be used as input.

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # Read in all the data, ignoring empty lines or lines with "---"

    data = []

    with open(file_name, encoding="utf-8") as data_file:
        for line in data_file:
            if "---" in line or line == "\n":
                continue

            data.append(list(filter(None, line.rstrip().split(" "))))

    isochrones = []

    for line in data:
        if "(Gyr)" in line:
            age = line[-1]

        elif "lg(g)" in line:
            header = ["M/Ms", "Teff(K)"] + line[1:]

        else:
            line.insert(0, age)
            isochrones.append(line)

    header = np.asarray(header, dtype=str)
    isochrones = np.asarray(isochrones, dtype=float)

    isochrones[:, 0] *= 1e3  # (Myr)
    isochrones[:, 1] *= constants.M_SUN / constants.M_JUP  # (Mjup)

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    print(f"Adding isochrones: {tag}...", end="", flush=True)

    dtype = h5py.special_dtype(vlen=str)

    dset = database.create_dataset(
        f"isochrones/{tag}/filters", (np.size(header[7:]),), dtype=dtype
    )

    dset[...] = header[7:]

    database.create_dataset(f"isochrones/{tag}/magnitudes", data=isochrones[:, 8:])

    dset = database.create_dataset(
        f"isochrones/{tag}/evolution", data=isochrones[:, 0:8]
    )

    dset.attrs["model"] = "baraffe"

    print(" [DONE]")
    print(f"Database tag: {tag}")


def add_marleau(database, tag, file_name):
    """
    Function for adding the Marleau et al. isochrone data
    to the database. The isochrone data can be requested
    from Gabriel Marleau.

    https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..20M/abstract

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # M      age     S_0             L          S(t)            R        Teff
    # (M_J)  (Gyr)   (k_B/baryon)    (L_sol)    (k_B/baryon)    (R_J)    (K)
    mass, age, _, luminosity, _, radius, teff = np.loadtxt(file_name, unpack=True)

    age *= 1e3  # (Myr)
    luminosity = np.log10(luminosity)

    mass_cgs = 1e3 * mass * constants.M_JUP  # (g)
    radius_cgs = 1e2 * radius * constants.R_JUP  # (cm)

    logg = np.log10(1e3 * constants.GRAVITY * mass_cgs / radius_cgs ** 2)

    print(f"Adding isochrones: {tag}...", end="", flush=True)

    isochrones = np.vstack((age, mass, teff, luminosity, logg))
    isochrones = np.transpose(isochrones)

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    dset = database.create_dataset(f"isochrones/{tag}/evolution", data=isochrones)

    dset.attrs["model"] = "marleau"

    print(" [DONE]")


def add_sonora(database, input_path):
    """
    Function for adding the
    `Sonora Bobcat <https://zenodo.org/record/5063476>`_
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

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url = "https://zenodo.org/record/5063476/files/" \
          "evolution_and_photometery.tar.gz"

    input_file = "evolution_and_photometery.tar.gz"
    data_file = os.path.join(input_path, input_file)
    sub_folder = input_file.split(".", maxsplit=1)[0]
    data_folder = os.path.join(input_path, sub_folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.isfile(data_file):
        print("Downloading Sonora Bobcat evolution (929 kB)...", end="", flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    print("Unpacking Sonora Bobcat evolution (929 kB)", end="", flush=True)
    with tarfile.open(data_file) as tar:
        tar.extractall(data_folder)
    print(" [DONE]")

    iso_files = ["evo_tables+0.0/nc+0.0_co1.0_mass_age",
                 "evo_tables+0.5/nc+0.5_co1.0_mass_age",
                 "evo_tables-0.5/nc-0.5_co1.0_mass_age"]

    labels = ["[M/H] = +0.0", "[M/H] = +0.5", "[M/H] = -0.5"]

    for i, item in enumerate(iso_files):
        iso_file = f"evolution_tables/{item}"
        iso_path = os.path.join(data_folder, iso_file)

        # Teff      log g          Mass     Radius     log L    log age
        # (K)      (cm/s2)        (Msun)    (Rsun)    (Lsun)      (yr)
        teff, logg, mass, _, luminosity, age = np.loadtxt(iso_path, unpack=True, skiprows=2)

        age = 1e-6*10.**age  # (Myr)
        mass *= constants.M_SUN / constants.M_JUP  # (Mjup)

        print(f"Adding isochrones: Sonora {labels[i]}...", end="", flush=True)

        isochrones = np.vstack((age, mass, teff, luminosity, logg))
        isochrones = np.transpose(isochrones)

        index_sort = np.argsort(isochrones[:, 0])
        isochrones = isochrones[index_sort, :]

        metal = labels[i].split(" ")[2]

        dset = database.create_dataset(
            f"isochrones/sonora{metal}/evolution", data=isochrones)

        dset.attrs["model"] = "sonora"

        print(" [DONE]")
        print(f"Database tag: sonora{metal}")


def add_ames(database, input_path):
    """
    Function for adding the
    `Sonora Bobcat <https://zenodo.org/record/5063476>`_
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

    if not os.path.exists(input_path):
        os.makedirs(input_path)

    url_list = ["https://home.strw.leidenuniv.nl/~stolker/species/"
                "model.AMES-Cond-2000.M-0.0.MKO.Vega",
                "https://home.strw.leidenuniv.nl/~stolker/species/"
                "model.AMES-dusty.M-0.0.MKO.Vega"]

    iso_tags = ["AMES-Cond", "AMES-Dusty"]
    iso_size = ["235 kB", "182 kB"]

    for i, url_item in enumerate(url_list):
        input_file = url_item.split("/")[-1]
        data_file = os.path.join(input_path, input_file)

        if not os.path.isfile(data_file):
            print(f"Downloading {iso_tags[i]} isochrones "
                  f"({iso_size[i]})...", end="", flush=True)
            urllib.request.urlretrieve(url_item, data_file)
            print(" [DONE]")

        add_baraffe(database=database,
                    tag=iso_tags[i].lower(),
                    file_name=data_file)
