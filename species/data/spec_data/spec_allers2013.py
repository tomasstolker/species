"""
Module for adding young, M- and L-type dwarf spectra from
`Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/
2013ApJ...772...79A>`_ to the . These spectra are
also available in the `SpeX Prism Library Analysis Toolkit
<https://github.com/aburgasser/splat>`_.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pooch

from astropy.io import fits
from astroquery.simbad import Simbad
from typeguard import typechecked

from species.util.data_util import extract_tarfile, remove_directory
from species.util.query_util import get_simbad


@typechecked
def add_allers2013(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the spectra of young, M- and L-type dwarfs
    from `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/
    2013ApJ...772...79A>`_  to the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        The HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    Simbad.add_votable_fields("plx", "plx_error")

    url = "https://home.strw.leidenuniv.nl/~stolker/species/parallax.dat"
    input_file = "parallax.dat"
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="e2fe0719a919dc98d24627a12f535862a107e473bc67f09298a40ad474cdd491",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    parallax_data = pd.pandas.read_csv(
        data_file,
        usecols=[0, 1, 2],
        names=["object", "parallax", "parallax_error"],
        delimiter=",",
        dtype={"object": str, "parallax": float, "parallax_error": float},
    )

    print_text = "spectra of young M/L type objects from Allers & Liu 2013"

    url = "https://home.strw.leidenuniv.nl/~stolker/species/allers_liu_2013.tgz"
    input_file = "allers_liu_2013.tgz"
    data_file = Path(input_path) / input_file
    data_folder = Path(input_path) / "allers+2013/"

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    if data_folder.exists():
        remove_directory(data_folder)

    print(f"\nUnpacking {print_text} (173 kB)...", end="", flush=True)
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    data_file = Path(data_folder) / "sources.csv"

    sources = np.genfromtxt(
        data_file,
        delimiter=",",
        dtype=None,
        encoding="ASCII",
    )

    source_names = sources[:, 0]
    source_sptype = sources[:, 7]

    print_message = ""
    print()

    spec_files = sorted(data_folder.glob("*"))

    for file_item in spec_files:
        if file_item.suffix == ".fits":
            sp_data, header = fits.getdata(file_item, header=True)

        else:
            continue

        sp_data = np.transpose(sp_data)

        # (erg s-1 cm-2 A-1) -> (W m-2 um-1)
        sp_data[:, 1:] *= 10.0

        name = header["OBJECT"]

        if "RES" in header:
            spec_res = header["RES"]
        elif "RP" in header:
            spec_res = header["RP"]

        simbad_id = get_simbad(name)

        if simbad_id is not None:
            if not isinstance(simbad_id, str):
                simbad_id = simbad_id.decode("utf-8")

            par_select = parallax_data[parallax_data["object"] == simbad_id]

            if not par_select.empty:
                parallax = (
                    par_select["parallax"].values[0],
                    par_select["parallax_error"].values[0],
                )

            else:
                parallax = (np.nan, np.nan)

        else:
            parallax = (np.nan, np.nan)

        if np.isnan(parallax[0]) and simbad_id is not None:
            simbad_result = Simbad.query_object(simbad_id)

            if simbad_result is not None and len(simbad_result) > 0:
                if "PLX_VALUE" in simbad_result.columns:
                    if not simbad_result["PLX_VALUE"].mask[0]:
                        parallax = (
                            simbad_result["PLX_VALUE"].value[0],
                            simbad_result["PLX_ERROR"].value[0],
                        )

                else:
                    if not simbad_result["plx_value"].mask[0]:
                        parallax = (
                            simbad_result["plx_value"].value[0],
                            simbad_result["plx_err"].value[0],
                        )

        index = np.argwhere(source_names == name)

        if len(index) == 0:
            sptype = None
        else:
            sptype = source_sptype[index][0][0][:2]

        empty_message = len(print_message) * " "
        print(f"\r{empty_message}", end="")

        print_message = f"Adding spectra... {name}"
        print(f"\r{print_message}", end="")

        dset = database.create_dataset(f"spectra/allers+2013/{name}", data=sp_data)

        dset.attrs["name"] = str(name).encode()
        dset.attrs["sptype"] = str(sptype).encode()
        dset.attrs["simbad"] = str(simbad_id).encode()
        dset.attrs["parallax"] = float(parallax[0])  # (mas)
        dset.attrs["parallax_error"] = float(parallax[1])  # (mas)
        dset.attrs["spec_res"] = float(spec_res)

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
