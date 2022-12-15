"""
Module for adding young, M- and L-type dwarf spectra from
`Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/
2013ApJ...772...79A/abstract>`_ to the . These spectra are
also available in the `SpeX Prism Library Analysis Toolkit
<https://github.com/aburgasser/splat>`_.
"""

import os
import shutil
import urllib.request

import h5py
import numpy as np
import pandas as pd

from astropy.io import fits
from astroquery.simbad import Simbad
from typeguard import typechecked

from species.util import data_util, query_util


@typechecked
def add_allers2013(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the spectra of young, M- and L-type dwarfs
    from `Allers & Liu (2013) <https://ui.adsabs.harvard.edu/abs/
    2013ApJ...772...79A/abstract>`_  to the database.

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

    parallax_url = "https://home.strw.leidenuniv.nl/~stolker/species/parallax.dat"
    parallax_file = os.path.join(input_path, "parallax.dat")

    if not os.path.isfile(parallax_file):
        urllib.request.urlretrieve(parallax_url, parallax_file)

    parallax_data = pd.pandas.read_csv(
        parallax_file,
        usecols=[0, 1, 2],
        names=["object", "parallax", "parallax_error"],
        delimiter=",",
        dtype={"object": str, "parallax": float, "parallax_error": float},
    )

    print_text = "spectra of young M/L type objects from Allers & Liu 2013"

    data_url = "https://home.strw.leidenuniv.nl/~stolker/species/allers_liu_2013.tgz"
    data_file = os.path.join(input_path, "allers_liu_2013.tgz")
    data_folder = os.path.join(input_path, "allers+2013/")

    if not os.path.isfile(data_file):
        print(f"Downloading {print_text} (173 kB)...", end="", flush=True)
        urllib.request.urlretrieve(data_url, data_file)
        print(" [DONE]")

    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    print(f"Unpacking {print_text} (173 kB)...", end="", flush=True)
    data_util.extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    sources = np.genfromtxt(
        os.path.join(data_folder, "sources.csv"),
        delimiter=",",
        dtype=None,
        encoding="ASCII",
    )

    source_names = sources[:, 0]
    source_sptype = sources[:, 7]

    database.create_group("spectra/allers+2013")

    print_message = ""

    for _, _, files in os.walk(data_folder):
        for _, filename in enumerate(files):
            if filename.endswith(".fits"):
                sp_data, header = fits.getdata(
                    os.path.join(data_folder, filename), header=True
                )

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

            simbad_id = query_util.get_simbad(name)

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

            if np.isnan(parallax[0]):
                simbad = Simbad.query_object(simbad_id)
                if simbad is not None and not simbad["PLX_VALUE"].mask[0]:
                    parallax = (
                        simbad["PLX_VALUE"].value[0],
                        simbad["PLX_ERROR"].value[0],
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

    database.close()
