"""
Module for adding the SpeX Prism Spectral Libraries to the database.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pooch

from astropy.io.votable import parse_single_table
from typeguard import typechecked

from species.phot.syn_phot import SyntheticPhotometry
from species.util.data_util import update_sptype
from species.util.query_util import get_simbad


@typechecked
def add_spex(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding the SpeX Prism Spectral Library
    to the database.

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

    data_folder = Path(input_path) / "spex"

    if not data_folder.exists():
        data_folder.mkdir()

    url = (
        "http://svo2.cab.inta-csic.es/vocats/v2/spex/cs.php?"
        "RA=180.000000&DEC=0.000000&SR=180.000000&VERB=2"
    )

    input_file = "spex.xml"
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash=None,
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    table = parse_single_table(data_file)
    # name = table.array['name']
    twomass = table.array["name2m"]
    url = table.array["access_url"]

    unique_id = []

    for url_idx, url_item in enumerate(url):
        if twomass[url_idx] not in unique_id:
            input_file = twomass[url_idx] + ".xml"
            data_file = Path(data_folder) / input_file

            if not data_file.exists():
                print()

                pooch.retrieve(
                    url=url_item,
                    known_hash=None,
                    fname=input_file,
                    path=data_folder,
                    progressbar=True,
                )

            table = parse_single_table(data_file)
            name = table.array["ID"]
            url_spec = table.array["access_url"]

            if isinstance(name[0], str):
                name = name[0]
            else:
                name = name[0].decode("utf-8")

            input_file = f"spex_{name}.xml"
            data_file = Path(data_folder) / input_file

            if not data_file.exists():
                print()

                pooch.retrieve(
                    url=url_spec[0],
                    known_hash=None,
                    fname=input_file,
                    path=data_folder,
                    progressbar=True,
                )

            unique_id.append(twomass[url_idx])

    # 2MASS H band zero point for 0 mag (Cogen et al. 2003)
    zp_hband = 1.133e-9  # (W m-2 um-1)

    h_twomass = SyntheticPhotometry("2MASS/2MASS.H", zero_point=zp_hband)

    spec_files = sorted(data_folder.glob("*"))

    print_message = ""

    for file_item in spec_files:
        if file_item.stem.startswith("spex_") and file_item.suffix == ".xml":
            table = parse_single_table(file_item)

            wavelength = 1e-4 * table.array["wavelength"]  # (A) -> (um)
            flux = table.array["flux"]  # Normalized units
            spec_res = table.get_field_by_id("res").value

            wavelength = np.array(wavelength)
            flux = np.array(flux)
            error = np.full(flux.size, np.nan)

            # 2MASS magnitudes
            j_mag = table.get_field_by_id("jmag").value
            h_mag = table.get_field_by_id("hmag").value
            ks_mag = table.get_field_by_id("ksmag").value

            if not isinstance(j_mag, str):
                j_mag = j_mag.decode("utf-8")

            if not isinstance(h_mag, str):
                h_mag = h_mag.decode("utf-8")

            if not isinstance(ks_mag, str):
                ks_mag = ks_mag.decode("utf-8")

            if j_mag == "":
                j_mag = np.nan
            else:
                j_mag = float(j_mag)

            if h_mag == "":
                h_mag = np.nan
            else:
                h_mag = float(h_mag)

            if ks_mag == "":
                ks_mag = np.nan
            else:
                ks_mag = float(ks_mag)

            name = table.get_field_by_id("name").value

            if not isinstance(name, str):
                name = name.decode("utf-8")

            twomass_id = table.get_field_by_id("name2m").value

            if not isinstance(twomass_id, str):
                twomass_id = twomass_id.decode("utf-8")

            # Optical spectral type

            try:
                sptype_opt = table.get_field_by_id("optspty").value

                if not isinstance(sptype_opt, str):
                    sptype_opt = sptype_opt.decode("utf-8")

                sptype_opt = update_sptype(np.array([sptype_opt]))[0]

            except KeyError:
                sptype_opt = None

            # Near-infrared spectral type

            try:
                sptype_nir = table.get_field_by_id("nirspty").value

                if not isinstance(sptype_nir, str):
                    sptype_nir = sptype_nir.decode("utf-8")

                sptype_nir = update_sptype(np.array([sptype_nir]))[0]

            except KeyError:
                sptype_nir = None

            if np.isnan(h_mag):
                continue

            h_flux, _ = h_twomass.magnitude_to_flux(h_mag, error=None)
            phot = h_twomass.spectrum_to_flux(wavelength, flux)  # Normalized units

            flux *= h_flux / phot[0]  # (W m-2 um-1)

            spdata = np.column_stack([wavelength, flux, error])

            simbad_id = get_simbad(f"2MASS {twomass_id}")

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

            empty_message = len(print_message) * " "
            print(f"\r{empty_message}", end="")

            print_message = f"Adding spectra... {name}"
            print(f"\r{print_message}", end="")

            dset = database.create_dataset(f"spectra/spex/{name}", data=spdata)

            dset.attrs["name"] = str(name).encode()

            if sptype_opt is not None:
                dset.attrs["sptype"] = str(sptype_opt).encode()
            elif sptype_nir is not None:
                dset.attrs["sptype"] = str(sptype_nir).encode()
            else:
                dset.attrs["sptype"] = str("None").encode()

            dset.attrs["simbad"] = str(simbad_id).encode()
            dset.attrs["2MASS/2MASS.J"] = float(j_mag)
            dset.attrs["2MASS/2MASS.H"] = float(h_mag)
            dset.attrs["2MASS/2MASS.Ks"] = float(ks_mag)
            dset.attrs["parallax"] = float(parallax[0])  # (mas)
            dset.attrs["parallax_error"] = float(parallax[1])  # (mas)
            dset.attrs["spec_res"] = float(spec_res)

    empty_message = len(print_message) * " "
    print(f"\r{empty_message}", end="")

    print_message = "Adding spectra... [DONE]"
    print(f"\r{print_message}")
