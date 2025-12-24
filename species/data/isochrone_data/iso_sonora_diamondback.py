"""
Module with a function for adding the Sonora Diamondback
evolutionary tracks to the database.
"""

from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants


@typechecked
def add_sonora_diamondback(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the isochrone data of  `Sonora Diamondback
    <https://zenodo.org/records/12735103>`_ to the database.

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

    url = "https://zenodo.org/records/12735103/files/evolution.zip"

    input_file = "evolution.zip"
    data_folder = Path(input_path) / "sonora-diamondback-evolution"
    data_file = Path(data_folder) / input_file

    if not data_folder.exists():
        data_folder.mkdir()

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="1efb4e5297060fa7b0329dec363e0bfff4f6132d2d11b655281cabea091d78ee",
            fname=input_file,
            path=data_folder,
            progressbar=True,
        )

    print("\nUnpacking Sonora Diamondback evolution (830 kB)...", end="", flush=True)
    with ZipFile(str(data_file)) as zip_object:
        zip_object.extractall(path=str(data_folder))
    print(" [DONE]")

    url = "https://zenodo.org/records/12735103/files/photometry.zip"

    input_file = "photometry.zip"
    data_file = Path(data_folder) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="7210f358c6da317d60a19dffb5b18a398e20a565aebaaa297e521ee9076bbc9c",
            fname=input_file,
            path=data_folder,
            progressbar=True,
        )

    print("\nUnpacking Sonora Diamondback photometry (239 kB)...", end="", flush=True)
    with ZipFile(str(data_file)) as zip_object:
        zip_object.extractall(path=str(data_folder))
    print(" [DONE]")

    iso_files = [
        "nc_m-0.5_age",
        "nc_m0.0_age",
        "nc_m+0.5_age",
        "hybrid_f2_m-0.5_age",
        "hybrid_f2_m0.0_age",
        "hybrid_f2_m+0.5_age",
        "hybrid-grav_f2_m-0.5_age",
        "hybrid-grav_f2_m0.0_age",
        "hybrid-grav_f2_m+0.5_age",
    ]

    labels = [
        "cloud-free, [M/H] = -0.5",
        "cloud-free, [M/H] = +0.0",
        "cloud-free, [M/H] = +0.5",
        "hybrid, fsed = 2, [M/H] = -0.5",
        "hybrid, fsed = 2, [M/H] = +0.0",
        "hybrid, fsed = 2, [M/H] = +0.5",
        "hybrid-grav, fsed = 2, [M/H] = -0.5",
        "hybrid-grav, fsed = 2, [M/H] = +0.0",
        "hybrid-grav, fsed = 2, [M/H] = +0.5",
    ]

    iso_tags = [
        "nc-0.5",
        "nc+0.0",
        "nc+0.5",
        "hybrid-0.5",
        "hybrid+0.0",
        "hybrid+0.5",
        "hybrid-grav-0.5",
        "hybrid-grav+0.0",
        "hybrid-grav+0.5",
    ]

    for iso_idx, iso_item in enumerate(iso_files):
        iso_file = f"evolution/{iso_item}"
        iso_path = Path(data_folder) / iso_file

        iso_data = []

        with open(str(iso_path), encoding="utf-8") as open_file:
            for line_idx, line_item in enumerate(open_file):
                if line_idx == 0 or "Gyr" in line_item:
                    continue

                # age(Gyr)  M/Msun  log(L/Lsun)  Teff(K)  log(g)  R/Rjup
                # The data files have R/Rsun, but should be R/Rjup
                param = list(filter(None, line_item.strip().split(" ")))
                param = list(map(float, param))

                param[0] = 1e3 * param[0]  # (Gyr) -> (Myr)

                param[1] = (
                    param[1] * constants.M_SUN / constants.M_JUP
                )  # (Msun) -> (Mjup)

                iso_data.append(
                    [param[0], param[1], param[2], param[3], param[4], param[5]]
                )

            print(
                f"\nAdding isochrones: Sonora {labels[iso_idx]}...", end="", flush=True
            )

            iso_data = np.array(iso_data)

            dset = database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/age",
                data=iso_data[:, 0],
            )  # (Myr)

            database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/mass",
                data=iso_data[:, 1],
            )  # (Mjup)

            database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/log_lum",
                data=iso_data[:, 2],
            )  # log(L/Lsun)

            database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/teff",
                data=iso_data[:, 3],
            )  # (K)

            database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/log_g",
                data=iso_data[:, 4],
            )  # log(g)

            database.create_dataset(
                f"isochrones/sonora-diamondback-{iso_tags[iso_idx]}/radius",
                data=iso_data[:, 5],
            )  # (Rjup)

            dset.attrs["model"] = "sonora-diamondback"

            print(" [DONE]")
            print(f"Database tag: sonora-diamondback-{iso_tags[iso_idx]}")
