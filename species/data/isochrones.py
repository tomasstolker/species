"""
Module for isochrone data from evolutionary models.
"""

import glob
import os
import shutil
import urllib.request

import h5py
import numpy as np

from species.core import constants
from species.util import data_util


def add_manual(database, tag, file_name, model_name="manual"):
    """
    Function for adding any of the isochrones from
    https://phoenix.ens-lyon.fr/Grids/ or
    https://perso.ens-lyon.fr/isabelle.baraffe/ to
    the database.

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.
    model_name : str
        Model name that is stored as attribute of the
        isochrone dataset in the HDF5 database.

    Returns
    -------
    NoneType
        None
    """

    # Read in all the data, ignoring empty lines or lines with "---"

    data = []

    check_baraffe = False
    baraffe_continue = False

    with open(file_name, encoding="utf-8") as open_file:
        for line in open_file:
            if "BHAC15" in line:
                check_baraffe = True
                continue

            if not baraffe_continue:
                if "(Gyr)" in line:
                    baraffe_continue = True
                else:
                    continue

            if line[0] == "!":
                line = line[1:]

            elif line[:2] == " !":
                line = line[2:]

            if "---" in line or line == "\n":
                continue

            data.append(list(filter(None, line.rstrip().split(" "))))

    iso_data = []

    for line in data:
        if "(Gyr)" in line:
            age = line[-1]

        elif "lg(g)" in line:
            # Isochrones from Phoenix website
            header = ["M/Ms", "Teff(K)"] + line[1:]

        elif "M/Ms" in line:
            # Isochrones from Baraffe et al. (2015)
            header = line.copy()

        else:
            line.insert(0, age)
            iso_data.append(line)

    header = np.asarray(header, dtype=str)
    iso_data = np.asarray(iso_data, dtype=float)

    iso_data[:, 0] *= 1e3  # (Myr)
    iso_data[:, 1] *= constants.M_SUN / constants.M_JUP  # (Mjup)
    iso_data[:, 5] *= 1e9  # (cm)
    iso_data[:, 5] *= 1e-2 / constants.R_JUP  # (cm) -> (Rjup)

    index_sort = np.argsort(iso_data[:, 0])
    iso_data = iso_data[index_sort, :]

    print(f"Adding isochrones: {tag}...", end="", flush=True)

    if check_baraffe:
        filters = header[6:]
    else:
        filters = header[7:]

    dtype = h5py.string_dtype(encoding="utf-8", length=None)

    dset = database.create_dataset(
        f"isochrones/{tag}/filters", (np.size(filters),), dtype=dtype
    )

    dset[...] = filters

    dset = database.create_dataset(
        f"isochrones/{tag}/age", data=iso_data[:, 0]
    )  # (Myr)
    database.create_dataset(f"isochrones/{tag}/mass", data=iso_data[:, 1])  # (Mjup)
    database.create_dataset(f"isochrones/{tag}/teff", data=iso_data[:, 2])  # (K)
    database.create_dataset(
        f"isochrones/{tag}/log_lum", data=iso_data[:, 3]
    )  # log(L/Lsun)
    database.create_dataset(f"isochrones/{tag}/log_g", data=iso_data[:, 4])  # log(g)
    database.create_dataset(f"isochrones/{tag}/radius", data=iso_data[:, 5])  # (Rjup)
    database.create_dataset(f"isochrones/{tag}/deuterium", data=iso_data[:, 6])
    database.create_dataset(f"isochrones/{tag}/lithium", data=iso_data[:, 7])
    database.create_dataset(f"isochrones/{tag}/magnitudes", data=iso_data[:, 8:])

    dset.attrs["model"] = model_name

    print(" [DONE]")
    print(f"Database tag: {tag}")


def add_ames(database, input_path):
    """
    Function for adding the AMES-Cond and AMES-Dusty
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

    url_list = [
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-Cond-2000.M-0.0.MKO.Vega",
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.AMES-dusty.M-0.0.MKO.Vega",
    ]

    iso_tags = ["AMES-Cond", "AMES-Dusty"]
    iso_size = ["235 kB", "182 kB"]

    for i, url_item in enumerate(url_list):
        input_file = url_item.split("/")[-1]
        data_file = os.path.join(input_path, input_file)

        if not os.path.isfile(data_file):
            print(
                f"Downloading {iso_tags[i]} isochrones ({iso_size[i]})...",
                end="",
                flush=True,
            )
            urllib.request.urlretrieve(url_item, data_file)
            print(" [DONE]")

        add_manual(
            database=database,
            tag=iso_tags[i].lower(),
            file_name=data_file,
            model_name="ames",
        )


def add_atmo(database, input_path):
    """
    Function for adding the AMES-Cond and AMES-Dusty
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
        "https://home.strw.leidenuniv.nl/~stolker/"
        "species/atmo_evolutionary_tracks.tgz"
    )

    # iso_tags = ["ATMO-CEQ", "ATMO-NEQ-weak", , "ATMO-NEQ-strong"]
    # iso_size = ["235 kB", "182 kB"]

    iso_tag = "ATMO"
    iso_size = "9.6 MB"

    data_folder = os.path.join(input_path, "atmo_evolutionary_tracks")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    print(f"Unpacking {iso_tag} isochrones ({iso_size})...", end="", flush=True)
    data_util.extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_files = [
        "ATMO_CEQ",
        "ATMO_NEQ_weak",
        "ATMO_NEQ_strong",
    ]

    labels = [
        "ATMO equilibrium chemistry",
        "ATMO non-equilibrium chemistry (weak)",
        "ATMO non-equilibrium chemistry (strong)",
    ]

    db_tags = [
        "atmo-ceq",
        "atmo-neq-weak",
        "atmo-neq-strong",
    ]

    for j, iso_item in enumerate(iso_files):
        iso_path = os.path.join(data_folder, iso_item)
        iso_path = os.path.join(iso_path, "MKO_WISE_IRAC")

        file_list = sorted(glob.glob(iso_path + "/*.txt"))

        for i, file_item in enumerate(file_list):
            # Mass (Msun) - Age (Gyr) - Teff (K) - log(L/Lsun) - Radius (Rsun) - log(g)
            if i == 0:
                iso_data = np.loadtxt(file_item)

            else:
                iso_load = np.loadtxt(file_item)
                iso_data = np.vstack((iso_data, iso_load))

            with open(file_item, encoding="utf-8") as open_file:
                parameters = open_file.readline()
                filter_names = parameters.split()[7:]

        iso_data[:, 0] *= constants.M_SUN / constants.M_JUP  # (Msun) -> (Mjup)
        iso_data[:, 1] *= 1e3  # (Gyr) -> (Myr)
        iso_data[:, 4] *= constants.R_SUN / constants.R_JUP  # (Rsun) -> (Rjup)

        print(f"Adding isochrones: {labels[j]}...", end="", flush=True)

        dtype = h5py.string_dtype(encoding="utf-8", length=None)

        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/filters", (np.size(filter_names),), dtype=dtype
        )

        dset[...] = filter_names

        database.create_dataset(
            f"isochrones/{db_tags[j]}/mass", data=iso_data[:, 0]
        )  # (Mjup)
        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/age", data=iso_data[:, 1]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/teff", data=iso_data[:, 2]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_lum", data=iso_data[:, 3]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/radius", data=iso_data[:, 4]
        )  # (Rjup)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_g", data=iso_data[:, 5]
        )  # log(g)

        database.create_dataset(
            f"isochrones/{db_tags[j]}/magnitudes", data=iso_data[:, 6:]
        )

        dset.attrs["model"] = "atmo"

        print(" [DONE]")
        print(f"Database tag: {db_tags[j]}")


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

    dset.attrs["model"] = "baraffe2015"

    print(" [DONE]")


def add_btsettl(database, input_path):
    """
    Function for adding the BT-Settl isochrone data to the database.

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
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.BT-Settl.M-0.0.MKO.Vega"
    )

    iso_tag = "BT-Settl"
    iso_size = "113 kB"

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    add_manual(
        database=database,
        tag=iso_tag.lower(),
        file_name=data_file,
        model_name="bt-settl",
    )


def add_marleau(database, tag, file_name):
    """
    Function for adding the Marleau et al. isochrone data
    to the database. The isochrone data can be requested
    from Gabriel Marleau.

    https://ui.adsabs.harvard.edu/abs/2019A%26A...624A..20M/abstract

    Parameters
    ----------
    database : h5py._hl.files.File
        Database.
    tag : str
        Tag name in the database.
    file_name : str
        Filename with the isochrones data.

    Returns
    -------
    NoneType
        None
    """

    # M      age     S_0             L          S(t)            R        Teff
    # (M_J)  (Gyr)   (k_B/baryon)    (L_sol)    (k_B/baryon)    (R_J)    (K)
    mass, age, _, luminosity, _, radius, teff = np.loadtxt(file_name, unpack=True)

    age *= 1e3  # (Myr)
    luminosity = np.log10(luminosity)

    mass_cgs = 1e3 * mass * constants.M_JUP  # (g)
    radius_cgs = 1e2 * radius * constants.R_JUP  # (cm)

    logg = np.log10(1e3 * constants.GRAVITY * mass_cgs / radius_cgs**2)

    print(f"Adding isochrones: {tag}...", end="", flush=True)

    isochrones = np.vstack((age, mass, teff, luminosity, logg))
    isochrones = np.transpose(isochrones)

    index_sort = np.argsort(isochrones[:, 0])
    isochrones = isochrones[index_sort, :]

    dset = database.create_dataset(f"isochrones/{tag}/evolution", data=isochrones)

    dset.attrs["model"] = "marleau"

    print(" [DONE]")


def add_nextgen(database, input_path):
    """
    Function for adding the NextGen isochrone data to the database.

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
        "https://home.strw.leidenuniv.nl/~stolker/species/"
        "model.NextGen.M-0.0.MKO.Vega"
    )

    iso_tag = "NextGen"
    iso_size = "177 kB"

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    add_manual(
        database=database,
        tag=iso_tag.lower(),
        file_name=data_file,
        model_name="nextgen",
    )


def add_saumon(database, input_path):
    """
    Function for adding the Saumon & Marley (2008)
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

    url_iso = "https://home.strw.leidenuniv.nl/~stolker/species/BD_evolution.tgz"

    iso_tag = "Saumon & Marley (2008)"
    iso_size = "800 kB"

    data_folder = os.path.join(input_path, "saumon_marley_2008")

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    input_file = url_iso.rsplit("/", maxsplit=1)[-1]
    data_file = os.path.join(input_path, input_file)

    if not os.path.isfile(data_file):
        print(f"Downloading {iso_tag} isochrones ({iso_size})...", end="", flush=True)
        urllib.request.urlretrieve(url_iso, data_file)
        print(" [DONE]")

    print(f"Unpacking {iso_tag} isochrones ({iso_size})...", end="", flush=True)
    data_util.extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_files = [
        "nc_solar_age",
        "nc-0.3_age",
        "nc+0.3_age",
        "f2_solar_age",
        "hybrid_solar_age",
    ]

    labels = [
        "Cloudless [M/H] = 0.0",
        "Cloudless [M/H] = -0.3",
        "Cloudless [M/H] = +0.3",
        "Cloudy f_sed = 2",
        "Hybrid (cloudless / f_sed = 2)",
    ]

    db_tags = [
        "saumon2008-nc_solar",
        "saumon2008-nc_-0.3",
        "saumon2008-nc_+0.3",
        "saumon2008-f2_solar",
        "saumon2008-hybrid_solar",
    ]

    for j, item in enumerate(iso_files):
        iso_path = os.path.join(data_folder, item)

        iso_data = []

        with open(iso_path, encoding="utf-8") as open_file:
            for i, line in enumerate(open_file):
                if i == 0 or " " not in line.strip():
                    continue

                # age(Gyr)  M/Msun  log(L/Lsun)  Teff(K)  log(g)  R/Rsun
                param = list(filter(None, line.strip().split(" ")))
                param = list(map(float, param))

                param[0] = 1e3 * param[0]  # (Gyr) -> (Myr)
                param[1] = (
                    param[1] * constants.M_SUN / constants.M_JUP
                )  # (Msun) -> (Mjup)
                param[5] = (
                    param[5] * constants.R_SUN / constants.R_JUP
                )  # (Rsun) -> (Rjup)

                iso_data.append(
                    [param[0], param[1], param[2], param[3], param[4], param[5]]
                )

        print(f"Adding isochrones: {iso_tag} {labels[j]}...", end="", flush=True)

        iso_data = np.array(iso_data)

        dset = database.create_dataset(
            f"isochrones/{db_tags[j]}/age", data=iso_data[:, 0]
        )  # (Myr)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/mass", data=iso_data[:, 1]
        )  # (Mjup)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_lum", data=iso_data[:, 2]
        )  # log(L/Lsun)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/teff", data=iso_data[:, 3]
        )  # (K)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/log_g", data=iso_data[:, 4]
        )  # log(g)
        database.create_dataset(
            f"isochrones/{db_tags[j]}/radius", data=iso_data[:, 5]
        )  # (Rjup)

        dset.attrs["model"] = "saumon2008"

        print(" [DONE]")
        print(f"Database tag: {db_tags[j]}")


def add_sonora(database, input_path):
    """
    Function for adding the
    `Sonora Bobcat <https://zenodo.org/record/5063476>`_
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

    url = "https://zenodo.org/record/5063476/files/evolution_and_photometery.tar.gz"

    input_file = "evolution_and_photometery.tar.gz"
    data_file = os.path.join(input_path, input_file)
    sub_folder = input_file.split(".", maxsplit=1)[0]
    data_folder = os.path.join(input_path, sub_folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.isfile(data_file):
        print("Downloading Sonora Bobcat evolution (929 kB)...", end="", flush=True)
        urllib.request.urlretrieve(url, data_file)
        print(" [DONE]")

    print("Unpacking Sonora Bobcat evolution (929 kB)...", end="", flush=True)
    data_util.extract_tarfile(data_file, data_folder)
    print(" [DONE]")

    iso_files = [
        "evo_tables+0.0/nc+0.0_co1.0_age",
        "evo_tables+0.5/nc+0.5_co1.0_age",
        "evo_tables-0.5/nc-0.5_co1.0_age",
    ]

    labels = ["[M/H] = +0.0", "[M/H] = +0.5", "[M/H] = -0.5"]

    for i, item in enumerate(iso_files):
        iso_file = f"evolution_tables/{item}"
        iso_path = os.path.join(data_folder, iso_file)

        iso_data = []

        with open(iso_path, encoding="utf-8") as open_file:
            for j, line in enumerate(open_file):
                if j == 0 or " " not in line.strip():
                    continue

                # age(Gyr)  M/Msun  log(L/Lsun)  Teff(K)  log(g)  R/Rsun
                param = list(filter(None, line.strip().split(" ")))
                param = list(map(float, param))

                param[0] = 1e3 * param[0]  # (Gyr) -> (Myr)
                param[1] = (
                    param[1] * constants.M_SUN / constants.M_JUP
                )  # (Msun) -> (Mjup)
                param[5] = (
                    param[5] * constants.R_SUN / constants.R_JUP
                )  # (Rsun) -> (Rjup)

                iso_data.append(
                    [param[0], param[1], param[2], param[3], param[4], param[5]]
                )

            print(f"Adding isochrones: Sonora {labels[i]}...", end="", flush=True)

            iso_data = np.array(iso_data)

            metallicity = labels[i].split(" ")[2]

            dset = database.create_dataset(
                f"isochrones/sonora{metallicity}/age", data=iso_data[:, 0]
            )  # (Myr)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/mass", data=iso_data[:, 1]
            )  # (Mjup)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/log_lum", data=iso_data[:, 2]
            )  # log(L/Lsun)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/teff", data=iso_data[:, 3]
            )  # (K)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/log_g", data=iso_data[:, 4]
            )  # log(g)
            database.create_dataset(
                f"isochrones/sonora{metallicity}/radius", data=iso_data[:, 5]
            )  # (Rjup)

            dset.attrs["model"] = "sonora"

            print(" [DONE]")
            print(f"Database tag: sonora{metallicity}")


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
    data_util.extract_tarfile(data_file, data_folder)
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
