"""
Module with a function for adding manual PHOENIX
evolutionary tracks to the database.
"""

import h5py
import numpy as np

from typeguard import typechecked

from species.core import constants


@typechecked
def add_manual(
    database: h5py._hl.files.File, tag: str, file_name: str, model_name: str
) -> None:
    """
    Function for adding any of the isochrones from
    https://phoenix.ens-lyon.fr/Grids/ or
    https://perso.ens-lyon.fr/isabelle.baraffe/ to
    the database.

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.
    model_name : str
        Model name that is stored as attribute of the
        isochrone dataset in the HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    # Read in all the data, ignoring empty lines or lines with "---"

    data = []

    check_baraffe = False
    baraffe_continue = False

    with open(file_name, encoding="utf-8") as open_file:
        for line in open_file:
            if "BHAC15" in line:
                check_baraffe = True
                continue

            if not baraffe_continue:
                if "(Gyr)" in line:
                    baraffe_continue = True
                else:
                    continue

            if line[0] == "!":
                line = line[1:]

            elif line[:2] == " !":
                line = line[2:]

            if "---" in line or line == "\n":
                continue

            data.append(list(filter(None, line.rstrip().split(" "))))

    iso_data = []

    for line in data:
        if "(Gyr)" in line:
            age = line[-1]

        elif "lg(g)" in line:
            # Isochrones from Phoenix website
            header = ["M/Ms", "Teff(K)"] + line[1:]

        elif "M/Ms" in line:
            # Isochrones from Baraffe et al. (2015)
            header = line.copy()

        else:
            line.insert(0, age)
            iso_data.append(line)

    header = np.asarray(header, dtype=str)
    iso_data = np.asarray(iso_data, dtype=float)

    iso_data[:, 0] *= 1e3  # (Myr)
    iso_data[:, 1] *= constants.M_SUN / constants.M_JUP  # (Mjup)

    if check_baraffe:
        iso_data[:, 5] *= constants.R_SUN / constants.R_JUP  # (Rjup)
    else:
        iso_data[:, 5] *= 1e9  # (cm)
        iso_data[:, 5] *= 1e-2 / constants.R_JUP  # (cm) -> (Rjup)

    index_sort = np.argsort(iso_data[:, 0])
    iso_data = iso_data[index_sort, :]

    print(f"\nAdding isochrones: {tag}...", end="", flush=True)

    if check_baraffe:
        filters = header[6:]
    else:
        filters = header[7:]

    dtype = h5py.string_dtype(encoding="utf-8", length=None)

    dset = database.create_dataset(
        f"isochrones/{tag}/filters", (np.size(filters),), dtype=dtype
    )

    dset[...] = filters

    dset = database.create_dataset(
        f"isochrones/{tag}/age", data=iso_data[:, 0]
    )  # (Myr)
    database.create_dataset(f"isochrones/{tag}/mass", data=iso_data[:, 1])  # (Mjup)
    database.create_dataset(f"isochrones/{tag}/teff", data=iso_data[:, 2])  # (K)
    database.create_dataset(
        f"isochrones/{tag}/log_lum", data=iso_data[:, 3]
    )  # log(L/Lsun)
    database.create_dataset(f"isochrones/{tag}/log_g", data=iso_data[:, 4])  # log(g)
    database.create_dataset(f"isochrones/{tag}/radius", data=iso_data[:, 5])  # (Rjup)

    if check_baraffe:
        database.create_dataset(f"isochrones/{tag}/lithium", data=iso_data[:, 6])
        database.create_dataset(f"isochrones/{tag}/magnitudes", data=iso_data[:, 7:])

    else:
        database.create_dataset(f"isochrones/{tag}/deuterium", data=iso_data[:, 6])
        database.create_dataset(f"isochrones/{tag}/lithium", data=iso_data[:, 7])
        database.create_dataset(f"isochrones/{tag}/magnitudes", data=iso_data[:, 8:])

    dset.attrs["model"] = model_name

    print(" [DONE]")
    print(f"Database tag: {tag}")
