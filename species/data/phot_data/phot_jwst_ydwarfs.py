"""
Module for the photometric data and parallaxes from the JWST late T and
early Y dwarf observations recorded in Beiler et al. 2024 (ApJ).
"""

from pathlib import Path

import h5py
import numpy as np
import pooch

from astropy.io import fits
from typeguard import typechecked

from species.util.data_util import update_sptype


@typechecked
def add_jwst_ydwarfs(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the synthesized photometry from 'Precise
    Bolometric Luminosities and Effective Temperatures of 23 Late-T
    and Y Dwarfs Obtained with JWST' by Beiler et al. (2024, ApJ).

    Parameters
    ----------
    input_path : str
        Data folder.
    database : h5py._hl.files.File
        The HDF5 database that has been opened.

    Returns
    -------
    NoneType
        None
    """

    input_file = "Beiler2024_synth_phot.fits"
    data_file = Path(input_path) / "Beiler2024_synth_phot.fits"
    url = "https://home.strw.leidenuniv.nl/~stolker/species/Beiler2024_synth_phot.fits"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    database.create_group("photometry/beiler2024")

    with fits.open(data_file, mode="update") as hdu_list:
        phot_data = hdu_list[1].data

    parallax = np.array(phot_data["PLX"], dtype=float)  # (mas)
    parallax_error = np.array(phot_data["EPLX"], dtype=float)  # (mas)

    name = np.array(phot_data["Name"]).astype("str")

    sptype_nir = np.array(phot_data["SpT"]).astype("str")
    sptype_nir = np.core.defchararray.strip(sptype_nir)

    sptype = update_sptype(sptype_nir)

    dtype = h5py.special_dtype(vlen=str)

    flag = np.repeat("null", np.size(name))

    dset = database.create_dataset(
        "photometry/beiler2024/flag", (np.size(flag),), dtype=dtype
    )
    dset[...] = flag

    dset = database.create_dataset(
        "photometry/beiler2024/name", (np.size(name),), dtype=dtype
    )
    dset[...] = name

    dset = database.create_dataset(
        "photometry/beiler2024/sptype", (np.size(sptype),), dtype=dtype
    )
    dset[...] = sptype

    database.create_dataset("photometry/beiler2024/parallax", data=parallax)
    database.create_dataset("photometry/beiler2024/parallax_error", data=parallax_error)

    nirc_filts = [
        "F090W",
        "F115W",
        "F140M",
        "F150W",
        "F150W2",
        "F162M",
        "F164N",
        "F182M",
        "F187N",
        "F200W",
        "F210M",
        "F212N",
        "F250M",
        "F277W",
        "F300M",
        "F322W2",
        "F323N",
        "F335M",
        "F356W",
        "F360M",
        "F405N",
        "F410M",
        "F430M",
        "F444W",
        "F460M",
        "F466N",
        "F470N",
        "F480M",
    ]

    for filt in nirc_filts:
        database.create_dataset(
            f"photometry/beiler2024/JWST/NIRCAM.{filt}",
            data=np.array(phot_data[f"{filt}"], dtype=float),
        )

    miri_filts = ["F560W", "F770W", "F1065C", "F1130W", "F1140C"]

    for filt in miri_filts:
        database.create_dataset(
            f"photometry/beiler2024/JWST/MIRI.{filt}",
            data=np.array(phot_data[f"{filt}"], dtype=float),
        )

    database.close()
