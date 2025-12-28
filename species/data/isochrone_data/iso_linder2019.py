"""
Module with a function for adding the Linder et al. (2019)
evolutionary tracks to the database.
"""

from requests.exceptions import HTTPError, SSLError
from pathlib import Path

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants
from species.util.data_util import extract_tarfile, remove_directory


@typechecked
def add_linder2019(database: h5py._hl.files.File, input_path: str) -> None:
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

    data_folder = Path(input_path) / "linder_2019"

    if data_folder.exists():
        # The folder should be removed if the TAR file was previously
        # unpacked because the file permissions are set to read-only
        # such that the extract_tarfile will cause an error if the
        # files need to be overwritten
        remove_directory(data_folder)

    data_folder.mkdir()

    input_file = "J_A+A_623_A85.tar.gz"
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        try:
            url = "https://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/tar.gz?J/A+A/623/A85"

            pooch.retrieve(
                url=url,
                known_hash=None,
                fname=input_file,
                path=input_path,
                progressbar=True,
            )

        except (HTTPError, SSLError):
            url = (
                "https://home.strw.leidenuniv.nl/~stolker/species/J_A+A_623_A85.tar.gz"
            )

            pooch.retrieve(
                url=url,
                known_hash="83bbc673a10207838983e0155ec21915caedd6465d6926fba23675562797923d",
                fname=input_file,
                path=input_path,
                progressbar=True,
            )

    print("\nUnpacking Linder et al. (2019) isochrones (536 kB)...", end="", flush=True)
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    iso_folder = Path(data_folder) / "isochrones"
    iso_files = sorted(iso_folder.glob("*"))

    for iso_item in iso_files:
        file_param = iso_item.stem.split("_")
        mags_idx = file_param.index("mags")

        if int(file_param[mags_idx + 1]) == -2:
            atm_model = "petitCODE"
        elif int(file_param[mags_idx + 1]) == -3:
            atm_model = "HELIOS"
        else:
            raise ValueError("Atmospheric model not recognized.")

        mh_idx = file_param.index("MH")
        metallicity = float(file_param[mh_idx + 1])

        if "brighter" in file_param or "fainter" in file_param:
            # Skip _brighter and _fainter files
            continue

        if "fsed" in file_param:
            fsed_idx = file_param.index("fsed")
            fsed = float(file_param[fsed_idx + 1])
        else:
            fsed = None

        iso_data = np.loadtxt(iso_item)

        print(
            f"\nAdding isochrones: Linder et al. (2019) {atm_model}...",
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
