import os
import urllib.request

import numpy as np

from species.core import constants
from species.util.data_util import extract_tarfile


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

    url = "https://zenodo.org/record/5063476/files/evolution_and_photometery.tar.gz"

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

    print("Unpacking Sonora Bobcat evolution (929 kB)...", end="", flush=True)
    extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_files = [
        "evo_tables+0.0/nc+0.0_co1.0_age",
        "evo_tables+0.5/nc+0.5_co1.0_age",
        "evo_tables-0.5/nc-0.5_co1.0_age",
    ]

    labels = ["[M/H] = +0.0", "[M/H] = +0.5", "[M/H] = -0.5"]

    for i, item in enumerate(iso_files):
        iso_file = f"evolution_tables/{item}"
        iso_path = os.path.join(data_folder, iso_file)

        iso_data = []

        with open(iso_path, encoding="utf-8") as open_file:
            for j, line in enumerate(open_file):
                if j == 0 or " " not in line.strip():
                    continue

                # age(Gyr)  M/Msun  log(L/Lsun)  Teff(K)  log(g)  R/Rsun
                param = list(filter(None, line.strip().split(" ")))
                param = list(map(float, param))

                param[0] = 1e3 * param[0]  # (Gyr) -> (Myr)
                param[1] = (
                    param[1] * constants.M_SUN / constants.M_JUP
                )  # (Msun) -> (Mjup)
                param[5] = (
                    param[5] * constants.R_SUN / constants.R_JUP
                )  # (Rsun) -> (Rjup)

                iso_data.append(
                    [param[0], param[1], param[2], param[3], param[4], param[5]]
                )

            print(f"Adding isochrones: Sonora {labels[i]}...", end="", flush=True)

            iso_data = np.array(iso_data)

            metallicity = labels[i].split(" ")[2]

            dset = database.create_dataset(
                f"isochrones/sonora{metallicity}/age", data=iso_data[:, 0]
            )  # (Myr)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/mass", data=iso_data[:, 1]
            )  # (Mjup)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/log_lum", data=iso_data[:, 2]
            )  # log(L/Lsun)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/teff", data=iso_data[:, 3]
            )  # (K)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/log_g", data=iso_data[:, 4]
            )  # log(g)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/radius", data=iso_data[:, 5]
            )  # (Rjup)

            dset.attrs["model"] = "sonora"

            print(" [DONE]")
            print(f"Database tag: sonora{metallicity}")
