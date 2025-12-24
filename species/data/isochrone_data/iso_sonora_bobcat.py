"""
Module with a function for adding the Sonora Bobcat
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
def add_sonora_bobcat(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the isochrone data of `Sonora Bobcat
    <https://zenodo.org/record/5063476>`_ to the database.

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

    url = "https://zenodo.org/record/5063476/files/evolution_and_photometery.tar.gz"

    input_file = "evolution_and_photometery.tar.gz"
    data_folder = Path(input_path) / "sonora-bobcat-evolution"
    data_file = data_folder / input_file

    if not data_folder.exists():
        data_folder.mkdir()

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="2198426d1ca0e410fda7b63c3b7f45f3890a8d9f2fcf0a3a1e36e14185283ca5",
            fname=input_file,
            path=data_folder,
            progressbar=True,
        )

    print("\nUnpacking Sonora Bobcat evolution (929 kB)...", end="", flush=True)
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    iso_files = [
        "evo_tables+0.0/nc+0.0_co1.0_age",
        "evo_tables+0.5/nc+0.5_co1.0_age",
        "evo_tables-0.5/nc-0.5_co1.0_age",
    ]

    labels = ["[M/H] = +0.0", "[M/H] = +0.5", "[M/H] = -0.5"]

    for iso_idx, iso_item in enumerate(iso_files):
        iso_file = f"evolution_tables/{iso_item}"
        iso_path = Path(data_folder) / iso_file

        iso_data = []

        with open(str(iso_path), encoding="utf-8") as open_file:
            for line_idx, line_item in enumerate(open_file):
                if line_idx == 0 or " " not in line_item.strip():
                    continue

                # age(Gyr)  M/Msun  log(L/Lsun)  Teff(K)  log(g)  R/Rsun
                param = list(filter(None, line_item.strip().split(" ")))
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
                f"\nAdding isochrones: Sonora {labels[iso_idx]}...", end="", flush=True
            )

            iso_data = np.array(iso_data)

            metallicity = labels[iso_idx].split(" ")[2]

            dset = database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/age", data=iso_data[:, 0]
            )  # (Myr)

            database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/mass", data=iso_data[:, 1]
            )  # (Mjup)

            database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/log_lum", data=iso_data[:, 2]
            )  # log(L/Lsun)

            database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/teff", data=iso_data[:, 3]
            )  # (K)

            database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/log_g", data=iso_data[:, 4]
            )  # log(g)

            database.create_dataset(
                f"isochrones/sonora-bobcat{metallicity}/radius", data=iso_data[:, 5]
            )  # (Rjup)

            dset.attrs["model"] = "sonora-bobcat"

            print(" [DONE]")
            print(f"Database tag: sonora{metallicity}")
