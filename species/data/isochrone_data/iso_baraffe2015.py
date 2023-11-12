import os
import urllib.request

import numpy as np

from species.core import constants


def add_baraffe2015(database, input_path):
    """
    Function for adding the Baraffe et al. (2015)
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
        "http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure"
    )

    iso_tag = "Baraffe et al. (2015)"
    iso_size = "1.4 MB"
    db_tag = "baraffe2015"

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    # M/Ms, log t(yr), Teff, log(L/Ls), log(g), R/Rs,
    # Log(Li/Li0), log(Tc), log(ROc), Mrad, Rrad, k2conv, k2rad
    mass, log_age, teff, log_lum, log_g, radius, _, _, _, _, _, _, _ = np.loadtxt(
        data_file, unpack=True, skiprows=45, comments="!"
    )

    age = 1e-6 * 10.0**log_age  # (Myr)
    mass *= constants.M_SUN / constants.M_JUP  # (Msun) -> (Mjup)
    radius *= constants.R_SUN / constants.R_JUP  # (Msun) -> (Mjup)

    iso_data = np.column_stack([age, mass, teff, log_lum, log_g, radius])

    print(f"Adding isochrones: {iso_tag}...", end="", flush=True)

    dset = database.create_dataset(
        f"isochrones/{db_tag}/age", data=iso_data[:, 0]
    )  # (Myr)
    database.create_dataset(f"isochrones/{db_tag}/mass", data=iso_data[:, 1])  # (Mjup)
    database.create_dataset(f"isochrones/{db_tag}/teff", data=iso_data[:, 2])  # (K)
    database.create_dataset(
        f"isochrones/{db_tag}/log_lum", data=iso_data[:, 3]
    )  # log(L/Lsun)
    database.create_dataset(f"isochrones/{db_tag}/log_g", data=iso_data[:, 4])  # log(g)
    database.create_dataset(
        f"isochrones/{db_tag}/radius", data=iso_data[:, 5]
    )  # (Rjup)

    dset.attrs["model"] = db_tag

    print(" [DONE]")
    print(f"Database tag: {db_tag}")
