"""
Module for adding the IRTF Spectral Library to the database.
"""

from pathlib import Path
from typing import Optional, List

import h5py
import numpy as np
import pandas as pd
import pooch

from astropy.io import fits
from typeguard import typechecked

from species.util.data_util import extract_tarfile, update_sptype
from species.util.query_util import get_parallax, get_simbad


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

    data_folder = Path(input_path) / "irtf"

    if not data_folder.exists():
        data_folder.mkdir()

    data_file = {
        "F": Path(input_path) / "irtf/F_fits_091201.tar",
        "G": Path(input_path) / "irtf/G_fits_091201.tar",
        "K": Path(input_path) / "irtf/K_fits_091201.tar",
        "M": Path(input_path) / "irtf/M_fits_091201.tar",
        "L": Path(input_path) / "irtf/L_fits_091201.tar",
        "T": Path(input_path) / "irtf/T_fits_091201.tar",
    }

    data_folder = {
        "F": Path(input_path) / "irtf/F_fits_091201",
        "G": Path(input_path) / "irtf/G_fits_091201",
        "K": Path(input_path) / "irtf/K_fits_091201",
        "M": Path(input_path) / "irtf/M_fits_091201",
        "L": Path(input_path) / "irtf/L_fits_091201",
        "T": Path(input_path) / "irtf/T_fits_091201",
    }

    irtf_folder = Path(input_path) / "irtf/"

    known_hash = {
        "F": "2bc34cfc1262582a8825c4b16d97c47f93cedb46dfcafa43b9db6dfb6fabe9c8",
        "G": "555c19d41dcb5278b796609e6eaceb5d1e8e484cb3b030fda5437722a1a84238",
        "K": "6c5b234a01681c174c3174366e7ee7b90ff35c542f762b9846cedb1d796c50a0",
        "M": "5488656537062593af43c175f9ee5068d0b7dece19b26175cce97d0593232436",
        "L": "2c2f6507c1dca2b81b5f085b3422664d2c21c6e77997037ab053dd77cc2d3381",
        "T": "74675599470c2e86803e9f395ec90fd62f8682a29bcbb281dd1ccd12750033e1",
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

    for spt_item in sptypes:
        if not data_file[spt_item].exists():
            print()

            pooch.retrieve(
                url=url[spt_item],
                known_hash=known_hash[spt_item],
                fname=data_file[spt_item].name,
                path=irtf_folder,
                progressbar=True,
            )

    print("\nUnpacking IRTF Spectral Library...", end="", flush=True)

    for spt_item in sptypes:
        extract_tarfile(str(data_file[spt_item]), str(irtf_folder))

    print(" [DONE]")

    print_message = ""
    print()

    for spt_item in sptypes:
        spec_files = sorted(data_folder[spt_item].glob("*"))

        for file_item in spec_files:
            if file_item.stem[-4:] != "_ext":
                spdata, header = fits.getdata(file_item, header=True)
                spdata = np.transpose(spdata)

                name = header["OBJECT"]
                sptype = header["SPTYPE"]

                if name[-2:] == "AB":
                    name = name[:-2]
                elif name[-3:] == "ABC":
                    name = name[:-3]

                spt_split = sptype.split()

                if spt_item in ["L", "T"] or spt_split[1][0] == "V":
                    empty_message = len(print_message) * " "
                    print(f"\r{empty_message}", end="")

                    print_message = f"Adding spectra... {name}"
                    print(f"\r{print_message}", end="")

                    simbad_id = get_simbad(name)

                    if simbad_id is not None:
                        # For backward compatibility
                        if not isinstance(simbad_id, str):
                            simbad_id = simbad_id.decode("utf-8")

                        par_select = parallax_data[parallax_data["object"] == simbad_id]

                        if not par_select.empty:
                            parallax = (
                                par_select["parallax"],
                                par_select["parallax_error"],
                            )
                        else:
                            simbad_id, parallax = get_parallax(name)

                    else:
                        parallax = (np.nan, np.nan)

                    sptype = update_sptype(np.array([sptype]))[0]

                    dset = database.create_dataset(f"spectra/irtf/{name}", data=spdata)

                    dset.attrs["name"] = str(name).encode()
                    dset.attrs["sptype"] = str(sptype).encode()
                    dset.attrs["simbad"] = str(simbad_id).encode()
                    dset.attrs["parallax"] = parallax[0]
                    dset.attrs["parallax_error"] = parallax[1]

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
