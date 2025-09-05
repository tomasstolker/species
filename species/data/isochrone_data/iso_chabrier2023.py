from pathlib import Path

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants
from species.util.data_util import extract_tarfile


@typechecked
def add_chabrier2023(database: h5py._hl.files.File, input_path: str) -> None:
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

    url = "https://home.strw.leidenuniv.nl/~stolker/species/chabrier2023_tracks.tgz"

    iso_tag = "ATMO (Chabrier et al. 2023"
    iso_size = "12 MB"

    data_folder = Path(input_path) / "chabrier2023_tracks"

    if not data_folder.exists():
        data_folder.mkdir()

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="2f268839107d7084f2512152e4da4be9fb1220793619899806c1f3fc4b3e4034",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    print(f"\nUnpacking {iso_tag} isochrones ({iso_size})...", end="", flush=True)
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    iso_files = [
        "ATMO_CEQ",
        "ATMO_NEQ_weak",
        "ATMO_NEQ_strong",
    ]

    labels = [
        "ATMO equilibrium chemistry (Chabrier et al. 2023)",
        "ATMO weak non-equilibrium chemistry (Chabrier et al. 2023)",
        "ATMO strong non-equilibrium chemistry (Chabrier et al. 2023)",
    ]

    db_tags = [
        "atmo-ceq-chabrier2023",
        "atmo-neq-weak-chabrier2023",
        "atmo-neq-strong-chabrier2023",
    ]

    for iso_idx, iso_item in enumerate(iso_files):
        tar_file = str(data_folder / iso_item) + "_neweos.tar.gz"
        print(f"\nUnpacking {iso_item} isochrones...", end="", flush=True)
        extract_tarfile(tar_file, str(data_folder))
        print(" [DONE]")

        iso_path = Path(data_folder) / iso_item / "MKO_WISE_IRAC_vega"

        # Ignore hidden files
        file_list = sorted(iso_path.glob("[!.]*.txt"))

        for file_idx, file_item in enumerate(file_list):
            # Mass (Msun) - Age (Gyr) - Teff (K) - log(L/Lsun) - Radius (Rsun) - log(g)
            if file_idx == 0:
                iso_data = np.loadtxt(str(file_item))

            else:
                iso_load = np.loadtxt(str(file_item))
                iso_data = np.vstack((iso_data, iso_load))

            with open(str(file_item), encoding="utf-8") as open_file:
                parameters = open_file.readline()
                filter_names = parameters.split()[7:]

        iso_data[:, 0] *= constants.M_SUN / constants.M_JUP  # (Msun) -> (Mjup)
        iso_data[:, 1] *= 1e3  # (Gyr) -> (Myr)
        iso_data[:, 4] *= constants.R_SUN / constants.R_JUP  # (Rsun) -> (Rjup)

        print(f"Adding isochrones: {labels[iso_idx]}...", end="", flush=True)

        dtype = h5py.string_dtype(encoding="utf-8", length=None)

        dset = database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/filters",
            (np.size(filter_names),),
            dtype=dtype,
        )

        dset[...] = filter_names

        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/mass", data=iso_data[:, 0]
        )  # (Mjup)
        dset = database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/age", data=iso_data[:, 1]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/teff", data=iso_data[:, 2]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/log_lum", data=iso_data[:, 3]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/radius", data=iso_data[:, 4]
        )  # (Rjup)
        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/log_g", data=iso_data[:, 5]
        )  # log(g)

        database.create_dataset(
            f"isochrones/{db_tags[iso_idx]}/magnitudes", data=iso_data[:, 6:]
        )

        dset.attrs["model"] = "atmo"

        print(" [DONE]")
        print(f"Database tag: {db_tags[iso_idx]}")
