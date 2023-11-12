import glob
import os
import urllib.request

import h5py
import numpy as np

from species.core import constants
from species.util.data_util import extract_tarfile


def add_atmo(database, input_path):
    """
    Function for adding the AMES-Cond and AMES-Dusty
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

    url_iso = (
        "https://home.strw.leidenuniv.nl/~stolker/"
        "species/atmo_evolutionary_tracks.tgz"
    )

    # iso_tags = ["ATMO-CEQ", "ATMO-NEQ-weak", , "ATMO-NEQ-strong"]
    # iso_size = ["235 kB", "182 kB"]

    iso_tag = "ATMO"
    iso_size = "9.6 MB"

    data_folder = os.path.join(input_path, "atmo_evolutionary_tracks")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    print(f"Unpacking {iso_tag} isochrones ({iso_size})...", end="", flush=True)
    extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_files = [
        "ATMO_CEQ",
        "ATMO_NEQ_weak",
        "ATMO_NEQ_strong",
    ]

    labels = [
        "ATMO equilibrium chemistry",
        "ATMO non-equilibrium chemistry (weak)",
        "ATMO non-equilibrium chemistry (strong)",
    ]

    db_tags = [
        "atmo-ceq",
        "atmo-neq-weak",
        "atmo-neq-strong",
    ]

    for j, iso_item in enumerate(iso_files):
        iso_path = os.path.join(data_folder, iso_item)
        iso_path = os.path.join(iso_path, "MKO_WISE_IRAC")

        file_list = sorted(glob.glob(iso_path + "/*.txt"))

        for i, file_item in enumerate(file_list):
            # Mass (Msun) - Age (Gyr) - Teff (K) - log(L/Lsun) - Radius (Rsun) - log(g)
            if i == 0:
                iso_data = np.loadtxt(file_item)

            else:
                iso_load = np.loadtxt(file_item)
                iso_data = np.vstack((iso_data, iso_load))

            with open(file_item, encoding="utf-8") as open_file:
                parameters = open_file.readline()
                filter_names = parameters.split()[7:]

        iso_data[:, 0] *= constants.M_SUN / constants.M_JUP  # (Msun) -> (Mjup)
        iso_data[:, 1] *= 1e3  # (Gyr) -> (Myr)
        iso_data[:, 4] *= constants.R_SUN / constants.R_JUP  # (Rsun) -> (Rjup)

        print(f"Adding isochrones: {labels[j]}...", end="", flush=True)

        dtype = h5py.string_dtype(encoding="utf-8", length=None)

        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/filters", (np.size(filter_names),), dtype=dtype
        )

        dset[...] = filter_names

        database.create_dataset(
            f"isochrones/{db_tags[j]}/mass", data=iso_data[:, 0]
        )  # (Mjup)
        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/age", data=iso_data[:, 1]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/teff", data=iso_data[:, 2]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_lum", data=iso_data[:, 3]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/radius", data=iso_data[:, 4]
        )  # (Rjup)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_g", data=iso_data[:, 5]
        )  # log(g)

        database.create_dataset(
            f"isochrones/{db_tags[j]}/magnitudes", data=iso_data[:, 6:]
        )

        dset.attrs["model"] = "atmo"

        print(" [DONE]")
        print(f"Database tag: {db_tags[j]}")
