import glob
import os
import shutil
import urllib.request

import h5py
import numpy as np

from species.core import constants
from species.util.data_util import extract_tarfile


def add_linder2019(database, input_path):
    """
    Function for adding the `Linder et al. (2019)
    <https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/623/A85#/browse>`_
    isochrones data to the database.

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

    filters = (
        "Paranal/NACO.J",
        "Paranal/NACO.H",
        "Paranal/NACO.Ks",
        "Paranal/NACO.Lp",
        "Paranal/NACO.Mp",
        "Generic/Cousins.R",
        "Generic/Cousins.I",
        "WISE/WISE.W1",
        "WISE/WISE.W2",
        "WISE/WISE.W3",
        "WISE/WISE.W4",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
        "JWST/MIRI.F560W",
        "JWST/MIRI.F770W",
        "JWST/MIRI.F1000W",
        "JWST/MIRI.F1280W",
        "JWST/MIRI.F1500W",
        "JWST/MIRI.F1800W",
        "JWST/MIRI.F2100W",
        "JWST/MIRI.F2550W",
        "Paranal/VISIR.B87",
        "Paranal/VISIR.SiC",
        "Paranal/SPHERE.IRDIS_B_Y",
        "Paranal/SPHERE.IRDIS_B_J",
        "Paranal/SPHERE.IRDIS_B_H",
        "Paranal/SPHERE.IRDIS_B_Ks",
        "Paranal/SPHERE.IRDIS_D_J23_2",
        "Paranal/SPHERE.IRDIS_D_J23_3",
        "Paranal/SPHERE.IRDIS_D_H23_2",
        "Paranal/SPHERE.IRDIS_D_H23_3",
        "Paranal/SPHERE.IRDIS_D_K12_1",
        "Paranal/SPHERE.IRDIS_D_K12_2",
    )

    data_folder = os.path.join(input_path, "linder_2019")

    if os.path.exists(data_folder):
        # The folder should be removed if the TAR file was previously
        # unpacked because the file permissions are set to read-only
        # such that the extract_tarfile will cause an error if the
        # files need to be overwritten
        shutil.rmtree(data_folder)

    os.makedirs(data_folder)

    url = "https://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/A+A/623/A85"

    input_file = "J_A+A_623_A85.tar.gz"

    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(
            "Downloading Linder et al. (2019) isochrones (536 kB)...",
            end="",
            flush=True,
        )
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    print("Unpacking Linder et al. (2019) isochrones (536 kB)...", end="", flush=True)
    extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_folder = os.path.join(data_folder, "isochrones")
    iso_files = sorted(glob.glob(iso_folder + "/*"))

    for iso_item in iso_files:
        file_name = iso_item.split("/")[-1]
        file_param = file_name[:-4].split("_")

        if int(file_param[3]) == -2:
            atm_model = "petitCODE"
        elif int(file_param[3]) == -3:
            atm_model = "HELIOS"
        else:
            raise ValueError("Atmospheric model not recognized.")

        metallicity = float(file_param[5])

        if len(file_param) == 7:
            # Skip _brighter and _fainter files
            continue

        if len(file_param) == 8:
            fsed = float(file_param[7])
        else:
            fsed = None

        iso_data = np.loadtxt(iso_item)

        print(
            f"Adding isochrones: Linder et al. (2019) {atm_model}...",
            end="",
            flush=True,
        )

        age = 1e-6 * 10.0 ** iso_data[:, 0]  # (Myr)
        mass = iso_data[:, 1] * constants.M_EARTH / constants.M_JUP  # (Mjup)
        radius = iso_data[:, 2]  # (Rjup)
        log_lum = np.log10(8.710e-10 * iso_data[:, 3])  # log(L/Lsun)
        teff = iso_data[:, 4]  # (K)
        logg = iso_data[:, 5]  # log(g/cgs)
        magnitudes = iso_data[:, 6:]

        if fsed is None:
            tag_label = f"linder2019-{atm_model}-metal_{metallicity}"
        else:
            tag_label = f"linder2019-{atm_model}-metal_{metallicity}-fsed_{fsed}"

        dtype = h5py.string_dtype(encoding="utf-8", length=None)

        dset = database.create_dataset(
            f"isochrones/{tag_label}/filters", (np.size(filters),), dtype=dtype
        )

        dset[...] = filters

        dset = database.create_dataset(f"isochrones/{tag_label}/age", data=age)  # (Myr)
        database.create_dataset(f"isochrones/{tag_label}/mass", data=mass)  # (Mjup)
        database.create_dataset(
            f"isochrones/{tag_label}/log_lum", data=log_lum
        )  # log(L/Lsun)
        database.create_dataset(f"isochrones/{tag_label}/teff", data=teff)  # (K)
        database.create_dataset(f"isochrones/{tag_label}/log_g", data=logg)  # log(g)
        database.create_dataset(f"isochrones/{tag_label}/radius", data=radius)  # (Rjup)
        database.create_dataset(f"isochrones/{tag_label}/magnitudes", data=magnitudes)

        dset.attrs["model"] = "linder2019"

        print(" [DONE]")
        print(f"Database tag: {tag_label}")
