"""
Module with a function for adding the Spiegel & Burrows (2012)
evolutionary tracks to the database.
"""

from pathlib import Path

import h5py
import numpy as np
import pooch

from typeguard import typechecked

from species.core import constants
from species.util.data_util import extract_tarfile, remove_directory


@typechecked
def add_spiegel2012(database: h5py._hl.files.File, input_path: str) -> None:
    """
    Function for adding the `Spiegel & Burrows (2012)
    <https://ui.adsabs.harvard.edu/abs/2012ApJ...745..174S/>`_
    isochrone data to the database. The spectra data don't
    contain the radius evolution, which is needed to calculate
    the effective temperature and surface gravity.

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

    url = "https://www.astro.princeton.edu/~burrows/warmstart/spectra.tar.gz"

    input_file = url.rsplit("/", maxsplit=1)[-1]
    data_file = Path(input_path) / input_file

    if not data_file.exists():
        print()

        pooch.retrieve(
            url=url,
            known_hash="428030c5f5853595f5fdaf277edbb45aa9aaf9f4d4e5bd0a56be6bfde6703a71",
            fname=input_file,
            path=input_path,
            progressbar=True,
        )

    data_folder = Path(input_path) / "spiegel_2012"

    if data_folder.exists():
        remove_directory(data_folder)

    data_folder.mkdir()

    print(
        "\nUnpacking Spiegel & Burrows (2012) isochrones (107 MB)...",
        end="",
        flush=True,
    )
    extract_tarfile(str(data_file), str(data_folder))
    print(" [DONE]")

    # hy1s = hybrid clouds, solar abundances
    # hy3s = hybrid clouds, 3x solar abundances
    # cf1s = cloud-free, solar abundances
    # cf3s = cloud-free, 3x solar abundances

    model_type = ["hy1s", "hy3s", "cf1s", "cf3s"]

    labels = [
        "hybrid [M/H] = +0.0",
        "hybrid [M/H] = +0.5",
        "cloud-free [M/H] = +0.0",
        "cloud-free [M/H] = +0.5",
    ]

    iso_tags = [
        "spiegel2012-hybrid+0.0",
        "spiegel2012-hybrid+0.5",
        "spiegel2012-cloudfree+0.0",
        "spiegel2012-cloudfree+0.5",
    ]

    for model_idx, model_item in enumerate(model_type):
        print(
            f"\nAdding isochrones: Spiegel & Burrows (2012) {labels[model_idx]}...",
            end="",
            flush=True,
        )

        data_files = data_folder.glob(f"spectra/spec_{model_item}_*.txt")

        mass_list = []
        age_list = []
        s_init_list = []
        log_lum_list = []

        for file_item in data_files:
            # The first number indicates the mass (in units of
            # Jupiter's) and the second indicates the age (in Myr).
            #
            # Each file contains the following rows:
            #
            #     Row #: Value
            #
            # 1: column 1: Age (Myr); columns 2-601: wavelength
            # (in microns, in range 0.8-15.0)
            #
            # 2-end: column 1: initial entropy;
            # columns 2-601 Fnu (in mJy for a source at 10 pc)

            file_split = file_item.name.split("_")

            mass = float(file_split[3])  # (Mjup)
            age = float(file_split[5][:-4])  # (Myr)

            data = np.loadtxt(file_item)
            # age = data[0, 0]  # (Myr)
            wavel = data[0, 1:]  # (um)

            # (um) -> (m)
            wavel *= 1e-6

            # Distance (pc)
            distance = 10.0 * constants.PARSEC

            # Remove the first line with the age and wavelengths
            data = data[1:,]

            for i in range(data.shape[0]):
                # Initial entropy (k_B/baryon)
                s_init = data[i, 0]

                # Flux density at 10 pc (mJy)
                flux_nu = data[i, 1:]

                # (mJy) -> (W m-2 Hz-1)
                flux_nu *= 1e-3 * 1e-26

                # (W m-2 Hz-1) -> (W m-2 m-1)
                flux_lambda = flux_nu * constants.LIGHT / wavel**2

                # Integrated flux (W m-2)
                flux_bol = np.trapezoid(flux_lambda, wavel)

                # Luminosity: L = 4π d^2 f_bol
                lum_bol = 4.0 * np.pi * distance**2 * flux_bol
                log_lum = np.log10(lum_bol / constants.L_SUN)

                mass_list.append(mass)
                age_list.append(age)
                s_init_list.append(s_init)
                log_lum_list.append(log_lum)

        model_param = ["age", "mass", "s_init"]

        dgroup = database.create_group(f"isochrones/{iso_tags[model_idx]}")

        dgroup.attrs["model"] = iso_tags[model_idx]
        dgroup.attrs["regular_grid"] = False
        dgroup.attrs["n_param"] = len(model_param)

        for i, item in enumerate(model_param):
            dgroup.attrs[f"parameter{i}"] = item

        database.create_dataset(
            f"isochrones/{iso_tags[model_idx]}/mass", data=mass_list
        )  # (Mjup)

        database.create_dataset(
            f"isochrones/{iso_tags[model_idx]}/age", data=age_list
        )  # (Myr)

        database.create_dataset(
            f"isochrones/{iso_tags[model_idx]}/log_lum", data=log_lum_list
        )  # log(L/Lsun)

        database.create_dataset(
            f"isochrones/{iso_tags[model_idx]}/s_init", data=s_init_list
        )  # (k_b/baryon)

        print(" [DONE]")
        print(f"Database tag: {iso_tags[model_idx]}")
