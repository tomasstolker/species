import os
import urllib.request

import numpy as np

from species.core import constants
from species.util.data_util import extract_tarfile


def add_saumon2008(database, input_path):
    """
    Function for adding the Saumon & Marley (2008)
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

    url_iso = "https://home.strw.leidenuniv.nl/~stolker/species/BD_evolution.tgz"

    iso_tag = "Saumon & Marley (2008)"
    iso_size = "800 kB"

    data_folder = os.path.join(input_path, "saumon_marley_2008")

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
        "nc_solar_age",
        "nc-0.3_age",
        "nc+0.3_age",
        "f2_solar_age",
        "hybrid_solar_age",
    ]

    labels = [
        "Cloudless [M/H] = 0.0",
        "Cloudless [M/H] = -0.3",
        "Cloudless [M/H] = +0.3",
        "Cloudy f_sed = 2",
        "Hybrid (cloudless / f_sed = 2)",
    ]

    db_tags = [
        "saumon2008-nc_solar",
        "saumon2008-nc_-0.3",
        "saumon2008-nc_+0.3",
        "saumon2008-f2_solar",
        "saumon2008-hybrid_solar",
    ]

    for j, item in enumerate(iso_files):
        iso_path = os.path.join(data_folder, item)

        iso_data = []

        with open(iso_path, encoding="utf-8") as open_file:
            for i, line in enumerate(open_file):
                if i == 0 or " " not in line.strip():
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

        print(f"Adding isochrones: {iso_tag} {labels[j]}...", end="", flush=True)

        iso_data = np.array(iso_data)

        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/age", data=iso_data[:, 0]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/mass", data=iso_data[:, 1]
        )  # (Mjup)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_lum", data=iso_data[:, 2]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/teff", data=iso_data[:, 3]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_g", data=iso_data[:, 4]
        )  # log(g)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/radius", data=iso_data[:, 5]
        )  # (Rjup)

        dset.attrs["model"] = "saumon2008"

        print(" [DONE]")
        print(f"Database tag: {db_tags[j]}")
