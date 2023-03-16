"""
Module for adding the IRTF Spectral Library to the database.
"""

import os
import urllib.request

from typing import Optional, List

import h5py
import numpy as np
import pandas as pd

from astropy.io import fits
from typeguard import typechecked

from species.util import data_util, query_util


@typechecked
def add_irtf(
    input_path: str, database: h5py._hl.files.File, sptypes: Optional[List[str]] = None
) -> None:
    """
    Function for adding the IRTF Spectral Library to the database.

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.
    sptypes : list(str), None
        List with the spectral types ('F', 'G', 'K', 'M', 'L', 'T'). All spectral types are
        included if set to ``None``.

    Returns
    -------
    NoneType
        None
    """

    if sptypes is None:
        sptypes = ["F", "G", "K", "M", "L", "T"]

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

    data_folder = os.path.join(input_path, "irtf")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    data_file = {
        "F": os.path.join(input_path, "irtf/F_fits_091201.tar"),
        "G": os.path.join(input_path, "irtf/G_fits_091201.tar"),
        "K": os.path.join(input_path, "irtf/K_fits_091201.tar"),
        "M": os.path.join(input_path, "irtf/M_fits_091201.tar"),
        "L": os.path.join(input_path, "irtf/L_fits_091201.tar"),
        "T": os.path.join(input_path, "irtf/T_fits_091201.tar"),
    }

    data_folder = {
        "F": os.path.join(input_path, "irtf/F_fits_091201"),
        "G": os.path.join(input_path, "irtf/G_fits_091201"),
        "K": os.path.join(input_path, "irtf/K_fits_091201"),
        "M": os.path.join(input_path, "irtf/M_fits_091201"),
        "L": os.path.join(input_path, "irtf/L_fits_091201"),
        "T": os.path.join(input_path, "irtf/T_fits_091201"),
    }

    main_folder = os.path.join(input_path, "irtf/")

    data_type = {
        "F": "F stars (4.4 MB)",
        "G": "G stars (5.6 MB)",
        "K": "K stars (5.5 MB)",
        "M": "M stars (7.5 MB)",
        "L": "L dwarfs (850 kB)",
        "T": "T dwarfs (100 kB)",
    }

    url_root = "http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/"

    url = {
        "F": url_root + "F_fits_091201.tar",
        "G": url_root + "G_fits_091201.tar",
        "K": url_root + "K_fits_091201.tar",
        "M": url_root + "M_fits_091201.tar",
        "L": url_root + "L_fits_091201.tar",
        "T": url_root + "T_fits_091201.tar",
    }

    for item in sptypes:
        if not os.path.isfile(data_file[item]):
            print(
                f"Downloading IRTF Spectral Library - {data_type[item]}...",
                end="",
                flush=True,
            )
            urllib.request.urlretrieve(url[item], data_file[item])
            print(" [DONE]")

    print("Unpacking IRTF Spectral Library...", end="", flush=True)

    for item in sptypes:
        data_util.extract_tarfile(data_file[item], main_folder)

    print(" [DONE]")

    database.create_group("spectra/irtf")

    print_message = ""

    for item in sptypes:
        for root, _, files in os.walk(data_folder[item]):

            for _, filename in enumerate(files):
                if filename[-9:] != "_ext.fits":
                    fitsfile = os.path.join(root, filename)

                    spdata, header = fits.getdata(fitsfile, header=True)
                    spdata = np.transpose(spdata)

                    name = header["OBJECT"]
                    sptype = header["SPTYPE"]

                    if name[-2:] == "AB":
                        name = name[:-2]
                    elif name[-3:] == "ABC":
                        name = name[:-3]

                    spt_split = sptype.split()

                    if item in ["L", "T"] or spt_split[1][0] == "V":
                        empty_message = len(print_message) * " "
                        print(f"\r{empty_message}", end="")

                        print_message = f"Adding spectra... {name}"
                        print(f"\r{print_message}", end="")

                        simbad_id = query_util.get_simbad(name)

                        if simbad_id is not None:
                            # For backward compatibility
                            if not isinstance(simbad_id, str):
                                simbad_id = simbad_id.decode("utf-8")

                            par_select = parallax_data[
                                parallax_data["object"] == simbad_id
                            ]

                            if not par_select.empty:
                                parallax = (
                                    par_select["parallax"],
                                    par_select["parallax_error"],
                                )
                            else:
                                simbad_id, parallax = query_util.get_parallax(name)

                        else:
                            parallax = (np.nan, np.nan)

                        sptype = data_util.update_sptype(np.array([sptype]))[0]

                        dset = database.create_dataset(
                            f"spectra/irtf/{name}", data=spdata
                        )

                        dset.attrs["name"] = str(name).encode()
                        dset.attrs["sptype"] = str(sptype).encode()
                        dset.attrs["simbad"] = str(simbad_id).encode()
                        dset.attrs["parallax"] = parallax[0]
                        dset.attrs["parallax_error"] = parallax[1]

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")

    database.close()
