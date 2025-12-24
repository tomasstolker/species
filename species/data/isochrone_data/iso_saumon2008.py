"""
Module with a function for adding the Saumon & Marley (2008)
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants
from species.util.data_util import extract_tarfile


@typechecked
def add_saumon2008(database: h5py._hl.files.File, input_path: str) -> None:
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

    url = "https://home.strw.leidenuniv.nl/~stolker/species/BD_evolution.tgz"

    iso_tag = "Saumon & Marley (2008)"
    iso_size = "800 kB"

    data_folder = Path(input_path) / "saumon_marley_2008"

    if not data_folder.exists():
        data_folder.mkdir()

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="fb64793b74a4503f13b9b1daa7d04e9594e9ba6f87353a0dbb50b73257961c88",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    print(f"\nUnpacking {iso_tag} isochrones ({iso_size})...", end="", flush=True)
    extract_tarfile(str(data_file), str(data_folder))
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

    for iso_idx, iso_item in enumerate(iso_files):
        iso_path = Path(data_folder) / iso_item

        iso_data = []

        with open(str(iso_path), encoding="utf-8") as open_file:
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

        print(
            f"\nAdding isochrones: {iso_tag} {labels[iso_idx]}...", end="", flush=True
        )

        iso_data = np.array(iso_data)

        dset = database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/age", data=iso_data[:, 0]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/mass", data=iso_data[:, 1]
        )  # (Mjup)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/log_lum", data=iso_data[:, 2]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/teff", data=iso_data[:, 3]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/log_g", data=iso_data[:, 4]
        )  # log(g)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/radius", data=iso_data[:, 5]
        )  # (Rjup)

        dset.attrs["model"] = "saumon2008"

        print(" [DONE]")
        print(f"Database tag: {db_tags[iso_idx]}")
